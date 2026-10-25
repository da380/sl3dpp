#include "sl3dpp/detail/BandSubstructure.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <stdexcept>

#include "sl3dpp/detail/BandSolvers.hpp"

namespace sl3dpp::detail {

template <typename Get>
void BandSubstructure::Factor::Build(int size, int bandwidth, Get&& get) {
  n = size;
  kd = std::min(bandwidth, std::max(size - 1, 0));
  if (n == 0) return;
  const auto ld = kd + 1;
  values.assign(static_cast<std::size_t>(ld) * n, 0.0);
  for (auto j = 0; j < n; ++j) {
    for (auto i = std::max(0, j - kd); i <= j; ++i) {
      values[static_cast<std::size_t>(j) * ld + (kd + i - j)] = get(i, j);
    }
  }
  if (BandCholeskyFactor(values, n, kd) == 0) return;
  // Banded LU on the same (symmetric) block, as DegreeOperator falls back.
  cholesky = false;
  const auto ldlu = 3 * kd + 1;
  values.assign(static_cast<std::size_t>(ldlu) * n, 0.0);
  for (auto j = 0; j < n; ++j) {
    for (auto i = std::max(0, j - kd); i <= std::min(n - 1, j + kd); ++i) {
      values[static_cast<std::size_t>(j) * ldlu + (2 * kd + i - j)] =
          i <= j ? get(i, j) : get(j, i);
    }
  }
  pivots.resize(n);
  if (const auto info = BandLuFactor(values, n, kd, kd, pivots); info != 0) {
    throw std::runtime_error(std::format(
        "a substructured block is singular (banded LU, info = {})", info));
  }
}

void BandSubstructure::Factor::Solve(std::span<double> x, int nrhs) const {
  if (n == 0) return;
  if (cholesky) {
    BandCholeskySolve(values, n, kd, x, nrhs);
  } else {
    BandLuSolve(values, n, kd, kd, pivots, x, nrhs);
  }
}

BandSubstructure::BandSubstructure(std::span<const double> ab, int n, int kd,
                                   std::vector<std::pair<int, int>> separators,
                                   std::span<const int> owned, Combine combine)
    : n_(n), separators_(std::move(separators)), combine_(std::move(combine)) {
  const auto ld = static_cast<std::size_t>(kd) + 1;
  if (n < 0 || kd < 0 || ab.size() != ld * static_cast<std::size_t>(n)) {
    throw std::invalid_argument(
        "the band must hold kd + 1 rows for each of the n columns");
  }
  // A(i, j) for i <= j <= i + kd.
  auto band = [&](int i, int j) {
    return ab[static_cast<std::size_t>(j) * ld +
              static_cast<std::size_t>(kd + i - j)];
  };
  auto previous = 0;
  for (const auto& [first, end] : separators_) {
    if (first < previous || end < first || end > n) {
      throw std::invalid_argument(
          "the separators must be sorted, disjoint dof ranges within the "
          "matrix");
    }
    previous = end;
  }
  owned_.assign(separators_.size() + 1, combine_ ? 0 : 1);
  for (const auto p : combine_ ? owned : std::span<const int>{}) {
    if (p < 0 || p > static_cast<int>(separators_.size())) {
      throw std::invalid_argument(std::format("there is no segment {}", p));
    }
    owned_[static_cast<std::size_t>(p)] = 1;
  }

  // The segments between the separators, and each dof's class: segment p
  // as 2 p, separator s as 2 s + 1. Segment p lies between separators
  // p - 1 and p.
  const auto count = static_cast<int>(separators_.size());
  auto classOf = std::vector<int>(static_cast<std::size_t>(n));
  auto offset = std::vector<int>(static_cast<std::size_t>(count) + 1, 0);
  auto start = 0;
  for (auto s = 0; s <= count; ++s) {
    const auto end = s < count ? separators_[s].first : n;
    segments_.push_back(Segment{.first = start, .count = end - start});
    for (auto i = start; i < end; ++i) classOf[i] = 2 * s;
    if (s == count) break;
    const auto [first, last] = separators_[s];
    for (auto i = first; i < last; ++i) classOf[i] = 2 * s + 1;
    offset[s + 1] = offset[s] + (last - first);
    start = last;
  }

  // Every coupling between segments must pass through a separator.
  for (auto j = 0; j < n; ++j) {
    for (auto i = std::max(0, j - kd); i < j; ++i) {
      const auto ci = classOf[i], cj = classOf[j];
      if (ci == cj || band(i, j) == 0.0) continue;
      const auto oddI = ci % 2 == 1, oddJ = cj % 2 == 1;
      if (oddI && oddJ) continue;  // separator to separator: in S directly
      const auto segment = oddI ? cj / 2 : ci / 2;
      const auto separator = oddI ? ci / 2 : cj / 2;
      if ((!oddI && !oddJ) ||
          (separator != segment - 1 && separator != segment)) {
        throw std::invalid_argument(
            std::format("dofs {} and {} couple across the separators", i, j));
      }
    }
  }

  // The separators' system: K_SS (from the owner of segment 0), less each
  // owned segment's reduction, summed over the processes.
  const auto ns = offset[count];
  const auto dense = static_cast<std::size_t>(ns);
  auto schur = std::vector<double>(dense * dense, 0.0);  // dense, column-major
  auto schurIndex = [&](int dof) {
    const auto s = classOf[dof] / 2;
    return offset[s] + (dof - separators_[s].first);
  };
  for (const auto& [first, end] : separators_) {
    if (!Owns(0)) break;
    for (auto j = first; j < end; ++j) {
      for (auto i = std::max(0, j - kd); i <= j; ++i) {
        if (classOf[i] % 2 == 0) continue;
        const auto a = static_cast<std::size_t>(schurIndex(i));
        const auto b = static_cast<std::size_t>(schurIndex(j));
        schur[a + b * dense] = band(i, j);
        schur[b + a * dense] = band(i, j);
      }
    }
  }
  for (auto p = 0; p <= count; ++p) {
    if (!Owns(p)) continue;
    auto& segment = segments_[p];
    const auto first = segment.first;
    const auto size = segment.count;
    segment.interior.Build(
        size, kd, [&](int i, int j) { return band(first + i, first + j); });
    if (size == 0) continue;
    auto dofs = std::vector<int>{};
    for (const auto s : {p - 1, p}) {
      if (s < 0 || s >= count) continue;
      for (auto d = separators_[s].first; d < separators_[s].second; ++d) {
        dofs.push_back(d);
        segment.columns.push_back(schurIndex(d));
      }
    }
    const auto columns = static_cast<int>(dofs.size());
    const auto rows = static_cast<std::size_t>(size);
    segment.coupling.assign(rows * dofs.size(), 0.0);
    for (auto c = 0; c < columns; ++c) {
      const auto d = dofs[c];
      const auto low = std::max(first, d - kd);
      const auto high = std::min(first + size - 1, d + kd);
      for (auto i = low; i <= high; ++i) {
        segment.coupling[static_cast<std::size_t>(c) * rows +
                         static_cast<std::size_t>(i - first)] =
            i <= d ? band(i, d) : band(d, i);
      }
    }
    segment.reduced = segment.coupling;
    segment.interior.Solve(segment.reduced, columns);
    for (auto a = 0; a < columns; ++a) {
      for (auto b = 0; b < columns; ++b) {
        auto dot = 0.0;
        const auto* u = &segment.coupling[static_cast<std::size_t>(a) * rows];
        const auto* v = &segment.reduced[static_cast<std::size_t>(b) * rows];
        for (std::size_t i = 0; i < rows; ++i) dot += u[i] * v[i];
        schur[static_cast<std::size_t>(segment.columns[a]) +
              static_cast<std::size_t>(segment.columns[b]) * dense] -= dot;
      }
    }
  }
  if (combine_) combine_(schur);
  // Its bandwidth (a few separators' worth: block tridiagonal) bounds the
  // factorisation's.
  auto bandwidth = 0;
  for (auto b = 0; b < ns; ++b) {
    for (auto a = 0; a < b; ++a) {
      if (schur[static_cast<std::size_t>(a) +
                static_cast<std::size_t>(b) * dense] != 0.0) {
        bandwidth = std::max(bandwidth, b - a);
      }
    }
  }
  schur_.Build(ns, bandwidth, [&](int i, int j) {
    return schur[static_cast<std::size_t>(i) +
                 static_cast<std::size_t>(j) * dense];
  });
}

bool BandSubstructure::UsedCholesky() const {
  return schur_.cholesky &&
         std::ranges::all_of(segments_, [](const Segment& s) {
           return s.interior.cholesky;  // true where not owned
         });
}

void BandSubstructure::Solve(std::span<double> x, int nrhs) const {
  const auto n = static_cast<std::size_t>(n_);
  if (nrhs < 0 || x.size() != n * static_cast<std::size_t>(nrhs)) {
    throw std::invalid_argument(
        std::format("expected {} x {} values, got {}", n_, nrhs, x.size()));
  }
  const auto ns = static_cast<std::size_t>(schur_.n);
  // The separators' right-hand sides, b_S (from the owner of segment 0).
  auto xs = std::vector<double>(ns * static_cast<std::size_t>(nrhs), 0.0);
  for (auto k = 0; k < nrhs && Owns(0); ++k) {
    auto a = static_cast<std::size_t>(k) * ns;
    for (const auto& [first, end] : separators_) {
      for (auto d = first; d < end; ++d) {
        xs[a++] = x[static_cast<std::size_t>(k) * n + d];
      }
    }
  }
  // Each owned segment's interior solve y = K_II^-1 b_I, and
  // g = b_S - K_SI y summed over the processes.
  auto interior = std::vector<std::vector<double>>(segments_.size());
  for (auto p = std::size_t{0}; p < segments_.size(); ++p) {
    const auto& segment = segments_[p];
    if (segment.count == 0 || owned_[p] == 0) continue;
    const auto rows = static_cast<std::size_t>(segment.count);
    auto& y = interior[p];
    y.resize(rows * static_cast<std::size_t>(nrhs));
    for (auto k = 0; k < nrhs; ++k) {
      const auto* from = &x[static_cast<std::size_t>(k) * n +
                            static_cast<std::size_t>(segment.first)];
      std::copy(from, from + rows, &y[static_cast<std::size_t>(k) * rows]);
    }
    segment.interior.Solve(y, nrhs);
    for (auto k = 0; k < nrhs; ++k) {
      const auto* yk = &y[static_cast<std::size_t>(k) * rows];
      for (auto c = std::size_t{0}; c < segment.columns.size(); ++c) {
        const auto* u = &segment.coupling[c * rows];
        auto dot = 0.0;
        for (std::size_t i = 0; i < rows; ++i) dot += u[i] * yk[i];
        xs[static_cast<std::size_t>(k) * ns +
           static_cast<std::size_t>(segment.columns[c])] -= dot;
      }
    }
  }
  if (combine_) combine_(xs);
  schur_.Solve(xs, nrhs);
  // Back-substitution, x_I = y - (K_II^-1 K_IS) x_S, and the separators.
  for (auto p = std::size_t{0}; p < segments_.size(); ++p) {
    const auto& segment = segments_[p];
    if (segment.count == 0 || owned_[p] == 0) continue;
    const auto rows = static_cast<std::size_t>(segment.count);
    const auto& y = interior[p];
    for (auto k = 0; k < nrhs; ++k) {
      auto* to = &x[static_cast<std::size_t>(k) * n +
                    static_cast<std::size_t>(segment.first)];
      std::copy(&y[static_cast<std::size_t>(k) * rows],
                &y[static_cast<std::size_t>(k) * rows] + rows, to);
      for (auto c = std::size_t{0}; c < segment.columns.size(); ++c) {
        const auto value = xs[static_cast<std::size_t>(k) * ns +
                              static_cast<std::size_t>(segment.columns[c])];
        const auto* w = &segment.reduced[c * rows];
        for (std::size_t i = 0; i < rows; ++i) to[i] -= w[i] * value;
      }
    }
  }
  for (auto k = 0; k < nrhs; ++k) {
    auto a = static_cast<std::size_t>(k) * ns;
    for (const auto& [first, end] : separators_) {
      for (auto d = first; d < end; ++d) {
        x[static_cast<std::size_t>(k) * n + d] = xs[a++];
      }
    }
  }
}

}  // namespace sl3dpp::detail
