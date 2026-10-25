// A serial prototype of the radially substructured per-degree solve that an
// MPI version of the 3D step would use (docs/dev/radial-substructuring.md):
// the radial mesh cut into P contiguous segments at element boundaries, the
// interior dofs of each segment eliminated by its own banded Cholesky, and
// the separators (the one shared node between neighbouring segments, three
// dofs) solved as one small block-tridiagonal Schur system.
//
// The system is synthetic but has the degree systems' structure: a 1D
// spectral-element assembly of E elements of nGll nodes, three coupled
// components per node in node-major dof order (band kd = 3 nGll - 1, as
// DegreeOperator's), each element's stiffness a random symmetric positive
// definite block scaled by a factor that grows over `decades` decades
// across the radius (the degree systems' condition numbers reach 1e11).
// It is solved directly (the global banded Cholesky, the reference) and by
// substructuring, for `nrhs` right-hand sides; the report gives the
// relative difference of the solutions, both residuals, the Schur system's
// size, and the costs.
//
// Usage:
//   substructure_prototype [elements] [ngll] [parts] [nrhs] [decades]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

#include "sl3dpp/detail/BandSolvers.hpp"

namespace {

using sl3dpp::detail::BandCholeskyFactor;
using sl3dpp::detail::BandCholeskySolve;

constexpr auto kComponents = 3;

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

/// A symmetric band matrix in LAPACK's upper storage, with the element
/// assembly and a product.
struct Band {
  int n = 0, kd = 0;
  std::vector<double> ab;
  Band(int n_, int kd_)
      : n(n_), kd(kd_), ab(static_cast<std::size_t>(n_) * (kd_ + 1), 0.0) {}
  [[nodiscard]] double& At(int i, int j) {  // i <= j, j - i <= kd
    return ab[static_cast<std::size_t>(j) * (kd + 1) + kd + i - j];
  }
  [[nodiscard]] double Get(int i, int j) const {
    if (i > j) std::swap(i, j);
    if (j - i > kd) return 0.0;
    return ab[static_cast<std::size_t>(j) * (kd + 1) + kd + i - j];
  }
  /// y = A x for nrhs columns.
  [[nodiscard]] std::vector<double> Apply(std::span<const double> x,
                                          int nrhs) const {
    auto y = std::vector<double>(x.size(), 0.0);
    for (auto c = 0; c < nrhs; ++c) {
      const auto at = static_cast<std::size_t>(c) * n;
      for (auto i = 0; i < n; ++i) {
        auto sum = 0.0;
        for (auto j = std::max(0, i - kd); j <= std::min(n - 1, i + kd); ++j) {
          sum += Get(i, j) * x[at + j];
        }
        y[at + i] = sum;
      }
    }
    return y;
  }
};

/// The element stiffnesses: random symmetric positive definite blocks of
/// the element's 3 nGll dofs, scaled across the radius.
std::vector<std::vector<double>> ElementMatrices(int elements, int ngll,
                                                 double decades) {
  const auto m = kComponents * ngll;
  auto engine = std::mt19937(2026);
  auto normal = std::normal_distribution<double>(0.0, 1.0);
  auto out = std::vector<std::vector<double>>{};
  for (auto e = 0; e < elements; ++e) {
    auto g = std::vector<double>(static_cast<std::size_t>(m) * m);
    for (auto& v : g) v = normal(engine);
    const auto scale = std::pow(10.0, decades * e / std::max(1, elements - 1));
    auto k = std::vector<double>(static_cast<std::size_t>(m) * m, 0.0);
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < m; ++j) {
        auto sum = i == j ? 0.05 * m : 0.0;
        for (auto r = 0; r < m; ++r) {
          sum += g[static_cast<std::size_t>(r) * m + i] *
                 g[static_cast<std::size_t>(r) * m + j];
        }
        k[static_cast<std::size_t>(i) * m + j] = scale * sum / m;
      }
    }
    out.push_back(std::move(k));
  }
  return out;
}

/// Assembles elements [e0, e1) into a band over their nodes, node e0 (ngll
/// - 1) first.
Band Assemble(const std::vector<std::vector<double>>& elements, int e0, int e1,
              int ngll) {
  const auto nodes = (e1 - e0) * (ngll - 1) + 1;
  const auto m = kComponents * ngll;
  auto band = Band(kComponents * nodes, m - 1);
  for (auto e = e0; e < e1; ++e) {
    const auto base = kComponents * (e - e0) * (ngll - 1);
    const auto& k = elements[e];
    for (auto i = 0; i < m; ++i) {
      for (auto j = i; j < m; ++j) {
        band.At(base + i, base + j) += k[static_cast<std::size_t>(i) * m + j];
      }
    }
  }
  return band;
}

/// Dense Cholesky solve of the small Schur system, in place.
void DenseSolve(std::vector<double> a, int n, std::span<double> b, int nrhs) {
  for (auto j = 0; j < n; ++j) {
    auto d = a[static_cast<std::size_t>(j) * n + j];
    for (auto k = 0; k < j; ++k) {
      d -= a[static_cast<std::size_t>(k) * n + j] *
           a[static_cast<std::size_t>(k) * n + j];
    }
    d = std::sqrt(d);
    a[static_cast<std::size_t>(j) * n + j] = d;
    for (auto i = j + 1; i < n; ++i) {
      auto s = a[static_cast<std::size_t>(j) * n + i];
      for (auto k = 0; k < j; ++k) {
        s -= a[static_cast<std::size_t>(k) * n + j] *
             a[static_cast<std::size_t>(k) * n + i];
      }
      a[static_cast<std::size_t>(j) * n + i] = s / d;  // U(j, i)
    }
  }
  for (auto c = 0; c < nrhs; ++c) {
    auto* x = b.data() + static_cast<std::size_t>(c) * n;
    for (auto i = 0; i < n; ++i) {  // U^T y = b
      for (auto k = 0; k < i; ++k)
        x[i] -= a[static_cast<std::size_t>(k) * n + i] * x[k];
      x[i] /= a[static_cast<std::size_t>(i) * n + i];
    }
    for (auto i = n - 1; i >= 0; --i) {  // U x = y
      for (auto k = i + 1; k < n; ++k)
        x[i] -= a[static_cast<std::size_t>(i) * n + k] * x[k];
      x[i] /= a[static_cast<std::size_t>(i) * n + i];
    }
  }
}

double MaxAbs(std::span<const double> v) {
  auto out = 0.0;
  for (const auto x : v) out = std::max(out, std::abs(x));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const auto elements = argc > 1 ? std::atoi(argv[1]) : 240;
  const auto ngll = argc > 2 ? std::atoi(argv[2]) : 5;
  const auto parts = argc > 3 ? std::atoi(argv[3]) : 8;
  const auto nrhs = argc > 4 ? std::atoi(argv[4]) : 258;  // 2 (l + 1), l 128
  const auto decades = argc > 5 ? std::atof(argv[5]) : 10.0;
  if (elements < parts || parts < 1 || ngll < 2 || nrhs < 1) {
    std::fprintf(stderr,
                 "usage: %s [elements] [ngll] [parts <= elements] [nrhs] "
                 "[decades]\n",
                 argv[0]);
    return 1;
  }
  const auto nodes = elements * (ngll - 1) + 1;
  const auto n = kComponents * nodes;
  std::printf(
      "substructure_prototype: %d elements of %d GLL nodes, %d dofs (band "
      "%d), %d parts, %d right-hand sides, element scales over %g decades\n",
      elements, ngll, n, kComponents * ngll - 1, parts, nrhs, decades);

  const auto stiffness = ElementMatrices(elements, ngll, decades);
  auto engine = std::mt19937(7);
  auto normal = std::normal_distribution<double>(0.0, 1.0);
  auto b = std::vector<double>(static_cast<std::size_t>(n) * nrhs);
  for (auto& v : b) v = normal(engine);

  // The reference: one banded Cholesky over the whole radius.
  const auto global = Assemble(stiffness, 0, elements, ngll);
  auto start = std::chrono::steady_clock::now();
  auto factor = global;
  if (BandCholeskyFactor(factor.ab, n, factor.kd) != 0) {
    std::fprintf(stderr, "the global system is not positive definite\n");
    return 1;
  }
  const auto globalFactor = Seconds(start);
  start = std::chrono::steady_clock::now();
  auto direct = b;
  BandCholeskySolve(factor.ab, n, factor.kd, direct, nrhs);
  const auto globalSolve = Seconds(start);

  // The segments: contiguous element ranges, the separators the nodes
  // between them.
  struct Segment {
    int e0 = 0, e1 = 0;   // elements
    int n0 = 0, n1 = 0;   // nodes, both ends included
    bool lower = false;   // a separator at n0 (shared with the segment below)
    bool upper = false;   // a separator at n1
    int i0 = 0, ni = 0;   // the interior dofs: local first and count
    Band interior{0, 0};  // factorised
    std::vector<double> coupling;  // K_I,Gamma: ni x (3 per separator)
    std::vector<double> solved;    // K_II^-1 K_I,Gamma, likewise
    std::vector<double> bi;        // K_II^-1 b_I, ni x nrhs
  };
  auto segments = std::vector<Segment>(parts);
  for (auto p = 0; p < parts; ++p) {
    auto& s = segments[p];
    s.e0 = elements * p / parts;
    s.e1 = elements * (p + 1) / parts;
    s.n0 = s.e0 * (ngll - 1);
    s.n1 = s.e1 * (ngll - 1);
    s.lower = p > 0;
    s.upper = p + 1 < parts;
  }
  const auto separators = parts - 1;
  const auto ns = kComponents * separators;  // the Schur system's size

  auto localFactor = 0.0, localSolve = 0.0, schurTime = 0.0;
  auto schur = std::vector<double>(static_cast<std::size_t>(ns) * ns, 0.0);
  auto rhs = std::vector<double>(static_cast<std::size_t>(ns) * nrhs, 0.0);
  // the separator dofs of a segment, in its local numbering, and their place
  // in the Schur system
  auto separatorsOf = [&](const Segment& s, int p) {
    auto local = std::vector<int>{}, global = std::vector<int>{};
    const auto last = kComponents * (s.n1 - s.n0);
    for (auto c = 0; c < kComponents; ++c) {
      if (s.lower) {
        local.push_back(c);
        global.push_back(kComponents * (p - 1) + c);
      }
    }
    for (auto c = 0; c < kComponents; ++c) {
      if (s.upper) {
        local.push_back(last + c);
        global.push_back(kComponents * p + c);
      }
    }
    return std::pair(local, global);
  };

  for (auto p = 0; p < parts; ++p) {
    auto& s = segments[p];
    const auto band = Assemble(stiffness, s.e0, s.e1, ngll);
    const auto nl = band.n;
    s.i0 = s.lower ? kComponents : 0;
    s.ni = nl - s.i0 - (s.upper ? kComponents : 0);
    // the interior band, factorised
    start = std::chrono::steady_clock::now();
    s.interior = Band(s.ni, band.kd);
    for (auto j = 0; j < s.ni; ++j) {
      for (auto i = std::max(0, j - band.kd); i <= j; ++i) {
        s.interior.At(i, j) = band.Get(s.i0 + i, s.i0 + j);
      }
    }
    if (BandCholeskyFactor(s.interior.ab, s.ni, s.interior.kd) != 0) {
      std::fprintf(stderr, "segment %d is not positive definite\n", p);
      return 1;
    }
    // K_II^-1 K_I,Gamma: one solve per separator dof
    const auto [local, global] = separatorsOf(s, p);
    const auto m = static_cast<int>(local.size());
    s.coupling.assign(static_cast<std::size_t>(s.ni) * m, 0.0);
    for (auto k = 0; k < m; ++k) {
      for (auto i = 0; i < s.ni; ++i) {
        s.coupling[static_cast<std::size_t>(k) * s.ni + i] =
            band.Get(s.i0 + i, local[k]);
      }
    }
    s.solved = s.coupling;
    BandCholeskySolve(s.interior.ab, s.ni, s.interior.kd, s.solved, m);
    const auto& solved = s.solved;
    localFactor += Seconds(start);
    // S += K_Gamma,Gamma - K_Gamma,I K_II^-1 K_I,Gamma
    start = std::chrono::steady_clock::now();
    for (auto a = 0; a < m; ++a) {
      for (auto c = 0; c < m; ++c) {
        auto sum = band.Get(local[a], local[c]);
        for (auto i = 0; i < s.ni; ++i) {
          sum -= s.coupling[static_cast<std::size_t>(a) * s.ni + i] *
                 solved[static_cast<std::size_t>(c) * s.ni + i];
        }
        schur[static_cast<std::size_t>(global[c]) * ns + global[a]] += sum;
      }
    }
    schurTime += Seconds(start);
    // the reduced right-hand side: b_Gamma (each separator's b taken by the
    // segment below it) less K_Gamma,I K_II^-1 b_I
    start = std::chrono::steady_clock::now();
    auto& bi = s.bi;
    bi.resize(static_cast<std::size_t>(s.ni) * nrhs);
    for (auto c = 0; c < nrhs; ++c) {
      for (auto i = 0; i < s.ni; ++i) {
        bi[static_cast<std::size_t>(c) * s.ni + i] =
            b[static_cast<std::size_t>(c) * n + kComponents * s.n0 + s.i0 + i];
      }
    }
    BandCholeskySolve(s.interior.ab, s.ni, s.interior.kd, bi, nrhs);
    for (auto c = 0; c < nrhs; ++c) {
      for (auto a = 0; a < m; ++a) {
        auto sum = 0.0;
        for (auto i = 0; i < s.ni; ++i) {
          sum += s.coupling[static_cast<std::size_t>(a) * s.ni + i] *
                 bi[static_cast<std::size_t>(c) * s.ni + i];
        }
        rhs[static_cast<std::size_t>(c) * ns + global[a]] -= sum;
        const auto upperSeparator = local[a] >= kComponents && s.upper;
        if (upperSeparator) {
          rhs[static_cast<std::size_t>(c) * ns + global[a]] +=
              b[static_cast<std::size_t>(c) * n + kComponents * s.n0 +
                local[a]];
        }
      }
    }
    localSolve += Seconds(start);
  }

  // The separators, then each segment's interior from them: x_I =
  // K_II^-1 b_I - (K_II^-1 K_I,Gamma) x_Gamma, both kept from the
  // reduction, so no second interior solve.
  start = std::chrono::steady_clock::now();
  if (ns > 0) DenseSolve(schur, ns, rhs, nrhs);
  const auto interfaceSolve = Seconds(start);
  start = std::chrono::steady_clock::now();
  auto x = std::vector<double>(static_cast<std::size_t>(n) * nrhs, 0.0);
  for (auto p = 0; p < parts; ++p) {
    const auto& s = segments[p];
    const auto [local, global] = separatorsOf(s, p);
    const auto m = static_cast<int>(local.size());
    for (auto c = 0; c < nrhs; ++c) {
      for (auto i = 0; i < s.ni; ++i) {
        auto v = s.bi[static_cast<std::size_t>(c) * s.ni + i];
        for (auto a = 0; a < m; ++a) {
          v -= s.solved[static_cast<std::size_t>(a) * s.ni + i] *
               rhs[static_cast<std::size_t>(c) * ns + global[a]];
        }
        x[static_cast<std::size_t>(c) * n + kComponents * s.n0 + s.i0 + i] = v;
      }
      for (auto a = 0; a < m; ++a) {
        x[static_cast<std::size_t>(c) * n + kComponents * s.n0 + local[a]] =
            rhs[static_cast<std::size_t>(c) * ns + global[a]];
      }
    }
  }
  localSolve += Seconds(start);

  // Agreement and residuals.
  auto diff = std::vector<double>(x.size());
  for (auto i = std::size_t{0}; i < x.size(); ++i) diff[i] = x[i] - direct[i];
  const auto residual = [&](const std::vector<double>& y) {
    auto r = global.Apply(y, nrhs);
    for (auto i = std::size_t{0}; i < r.size(); ++i) r[i] -= b[i];
    return MaxAbs(r) / MaxAbs(b);
  };
  std::printf("  agreement  max|x - x_direct| / max|x_direct| = %.3e\n",
              MaxAbs(diff) / MaxAbs(direct));
  std::printf(
      "  residuals  direct %.3e, substructured %.3e (max|Ax - b| / "
      "max|b|)\n",
      residual(direct), residual(x));
  std::printf(
      "  Schur system: %d separators, %d dofs (dense here; block "
      "tridiagonal, %d x %d blocks)\n",
      separators, ns, kComponents, kComponents);
  std::printf(
      "  times [s]: direct factor %.4f solve %.4f | substructured "
      "local factor + K_II^-1 K_IG %.4f, Schur assembly %.4f, local "
      "solves %.4f, interface solve %.4f\n",
      globalFactor, globalSolve, localFactor, schurTime, localSolve,
      interfaceSolve);
  std::printf(
      "  per part, as one rank would do it: factor %.4f, solves %.4f "
      "(interior band %d dofs)\n",
      localFactor / parts, localSolve / parts, segments[0].ni);
  return 0;
}
