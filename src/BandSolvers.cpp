#include "sl3dpp/detail/BandSolvers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace sl3dpp::detail {

namespace {

/// The band element A(i, j) in storage whose diagonal is row `diagonal` of
/// columns `ld` long.
template <typename T>
T& Band(std::span<T> ab, std::size_t ld, int diagonal, int i, int j) {
  return ab[static_cast<std::size_t>(j) * ld +
            static_cast<std::size_t>(diagonal + i - j)];
}

}  // namespace

int BandCholeskyFactor(std::span<double> ab, int n, int kd) {
  const auto ld = static_cast<std::size_t>(kd) + 1;
  auto row = std::vector<double>(ld);
  for (auto j = 0; j < n; ++j) {
    const auto d = Band(ab, ld, kd, j, j);
    if (!(d > 0.0)) return j + 1;
    const auto ujj = std::sqrt(d);
    Band(ab, ld, kd, j, j) = ujj;
    // Row j of U right of the diagonal, then the trailing block less its
    // outer product (dpbtf2's dscal and dsyr).
    const auto kn = std::min(kd, n - 1 - j);
    const auto inverse = 1.0 / ujj;
    for (auto c = 1; c <= kn; ++c) {
      row[c] = Band(ab, ld, kd, j, j + c) *= inverse;
    }
    for (auto c = 1; c <= kn; ++c) {
      auto* column = &Band(ab, ld, kd, j, j + c);  // A(j + r, j + c) at [r]
      const auto uc = row[c];
      for (auto r = 1; r <= c; ++r) column[r] -= row[r] * uc;
    }
  }
  return 0;
}

void BandCholeskySolve(std::span<const double> ab, int n, int kd,
                       std::span<double> b, int nrhs) {
  const auto ld = static_cast<std::size_t>(kd) + 1;
  for (auto k = 0; k < nrhs; ++k) {
    auto x =
        b.subspan(static_cast<std::size_t>(k) * n, static_cast<std::size_t>(n));
    // U^T y = b, forward: y_j from column j of U above the diagonal.
    for (auto j = 0; j < n; ++j) {
      auto sum = x[j];
      for (auto i = std::max(0, j - kd); i < j; ++i) {
        sum -= Band(ab, ld, kd, i, j) * x[i];
      }
      x[j] = sum / Band(ab, ld, kd, j, j);
    }
    // U x = y, backward: x_j, then its column taken from the rows above.
    for (auto j = n - 1; j >= 0; --j) {
      x[j] /= Band(ab, ld, kd, j, j);
      const auto xj = x[j];
      for (auto i = std::max(0, j - kd); i < j; ++i) {
        x[i] -= Band(ab, ld, kd, i, j) * xj;
      }
    }
  }
}

int BandLuFactor(std::span<double> ab, int n, int kl, int ku,
                 std::span<int> pivots) {
  const auto kv = kl + ku;  // the diagonal's row
  const auto ld = static_cast<std::size_t>(2 * kl + ku + 1);
  auto at = [&](int i, int j) -> double& { return Band(ab, ld, kv, i, j); };

  // The fill-in places of the first kv columns start zero (dgbtf2).
  for (auto j = ku + 1; j < std::min(kv, n); ++j) {
    for (auto i = 0; i < j - ku; ++i) at(i, j) = 0.0;
  }
  auto info = 0;
  auto ju = 0;  // the last column the pivots so far reach
  for (auto j = 0; j < n; ++j) {
    if (j + kv < n) {
      for (auto i = j; i < j + kl; ++i) at(i, j + kv) = 0.0;
    }
    // The pivot: the first largest magnitude on or below the diagonal.
    const auto km = std::min(kl, n - 1 - j);
    auto p = j;
    for (auto i = j + 1; i <= j + km; ++i) {
      if (std::abs(at(i, j)) > std::abs(at(p, j))) p = i;
    }
    pivots[j] = p;
    if (at(p, j) == 0.0) {
      if (info == 0) info = j + 1;
      continue;
    }
    ju = std::max(ju, std::min(p + ku, n - 1));
    if (p != j) {
      for (auto c = j; c <= ju; ++c) std::swap(at(p, c), at(j, c));
    }
    if (km > 0) {
      // The multipliers, then the trailing block within the band (dscal
      // and dger).
      const auto inverse = 1.0 / at(j, j);
      for (auto i = j + 1; i <= j + km; ++i) at(i, j) *= inverse;
      for (auto c = j + 1; c <= ju; ++c) {
        const auto ujc = at(j, c);
        for (auto i = j + 1; i <= j + km; ++i) at(i, c) -= at(i, j) * ujc;
      }
    }
  }
  return info;
}

void BandLuSolve(std::span<const double> ab, int n, int kl, int ku,
                 std::span<const int> pivots, std::span<double> b, int nrhs) {
  const auto kv = kl + ku;
  const auto ld = static_cast<std::size_t>(2 * kl + ku + 1);
  auto at = [&](int i, int j) { return Band(ab, ld, kv, i, j); };
  for (auto k = 0; k < nrhs; ++k) {
    auto x =
        b.subspan(static_cast<std::size_t>(k) * n, static_cast<std::size_t>(n));
    // L y = P^T b, as the interchanges and eliminations in turn.
    for (auto j = 0; j + 1 < n; ++j) {
      if (pivots[j] != j) std::swap(x[pivots[j]], x[j]);
      const auto xj = x[j];
      const auto lm = std::min(kl, n - 1 - j);
      for (auto i = j + 1; i <= j + lm; ++i) x[i] -= at(i, j) * xj;
    }
    // U x = y, backward; U has kl + ku superdiagonals.
    for (auto j = n - 1; j >= 0; --j) {
      x[j] /= at(j, j);
      const auto xj = x[j];
      for (auto i = std::max(0, j - kv); i < j; ++i) x[i] -= at(i, j) * xj;
    }
  }
}

}  // namespace sl3dpp::detail
