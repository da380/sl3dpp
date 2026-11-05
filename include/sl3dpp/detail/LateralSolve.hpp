#pragma once
/**
 * @file detail/LateralSolve.hpp
 * @brief Preconditioned conjugate gradients on the laterally heterogeneous
 * degree systems: (K_ref + dK) x = f with the radial reference solve as
 * the preconditioner (Maitra & Al-Attar 2019; Myhill, Maitra & Al-Attar
 * 2026), the operator matrix-free. Shared by the static solve
 * (ElasticPerturbation3D::SolveLateral), the 3D stepper's unrelaxed
 * start solve under lateral moduli (docs/adjoints.md §13j) and its
 * implicit step's solve (Viscoelastic3D::OpenStep::Solve, §13n).
 *
 * The unknown is a Viscoelastic3D::Solutions (per-degree complex m >= 0
 * columns, spheroidal and toroidal); the forces have the same layout. The
 * pairing is Hermitian over the degrees and orders (m > 0 weighted
 * twice): the real inner product of the real fields the columns stand
 * for, under which the operator is symmetric.
 */
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::detail {

using LateralFields = std::map<int, std::vector<std::complex<double>>>;

struct LateralSolveOptions {
  /// The relative residual (the preconditioned energy norm, relative to
  /// the forcing's) to stop at, and the iteration budget.
  double rtol = 1e-10;
  int max_iterations = 200;
  /// Set, the solve also stops once the residual is this fraction of
  /// the initial one (an inexact solve inside an outer iteration: each
  /// solve reduces what its start left by at least this factor).
  double reduction = 0.0;
};

struct LateralSolveInfo {
  int iterations = 0;
  double residual = 0.0;
  bool converged = false;
  /// The relative residual after each iteration (the first entry the
  /// reference response's).
  std::vector<double> residuals;
  /// Whether a curvature <p, A p> or <r, M r> came out non-positive: the
  /// iteration continued on magnitudes, but the operator pair is then not
  /// definite and conjugate gradients are not guaranteed.
  bool indefinite = false;
};

/// The full operator's action (K_ref + dK) x, and the reference solve
/// K_ref^-1 b, both over (spheroidal, toroidal) per-degree fields with
/// every degree present (zero columns where a block has none).
struct LateralSystem {
  std::function<std::pair<LateralFields, LateralFields>(
      const Viscoelastic3D::Solutions&)>
      apply;
  std::function<Viscoelastic3D::Solutions(const LateralFields&,
                                          const LateralFields&)>
      solve;
  /// Distributed over radial segments: the pairings are each process's
  /// part (its solution against the forces' shares) summed by this
  /// collective; empty undistributed.
  std::function<void(std::span<double>)> sum{};
};

inline double LateralPair(const Viscoelastic3D::Solutions& x,
                          const LateralFields& spheroidal,
                          const LateralFields& toroidal) {
  auto total = 0.0;
  auto pair = [&](const LateralFields& a, const LateralFields& b) {
    for (const auto& [l, al] : a) {
      const auto found = b.find(l);
      if (found == b.end()) continue;
      const auto& bl = found->second;
      const auto ndof = al.size() / static_cast<std::size_t>(l + 1);
      for (auto m = 0; m <= l; ++m) {
        const auto weight = m == 0 ? 1.0 : 2.0;
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          const auto i = static_cast<std::size_t>(m) * ndof + d;
          total += weight * (al[i] * std::conj(bl[i])).real();
        }
      }
    }
  };
  pair(x.spheroidal, spheroidal);
  pair(x.toroidal, toroidal);
  return total;
}

inline void LateralAxpy(LateralFields& y, double a, const LateralFields& x) {
  for (auto& [l, yl] : y) {
    const auto found = x.find(l);
    if (found == x.end()) continue;
    const auto& xl = found->second;
    for (auto i = std::size_t{0}; i < yl.size(); ++i) yl[i] += a * xl[i];
  }
}

/// The solve, started from the reference response x0 = K_ref^-1 f (so
/// the first preconditioned residual is the Born correction), or from a
/// given start (a warm start: the previous solution of a nearby system,
/// whose residual the iteration then reduces; the reference response is
/// still formed, for the relative residual's scale).
inline std::pair<Viscoelastic3D::Solutions, LateralSolveInfo>
LateralConjugateGradients(const LateralFields& fs, const LateralFields& ft,
                          const LateralSystem& system,
                          const LateralSolveOptions& options,
                          bool lateral = true,
                          const Viscoelastic3D::Solutions* start = nullptr) {
  auto x = system.solve(fs, ft);
  LateralSolveInfo info;
  auto pair = [&](const Viscoelastic3D::Solutions& a, const LateralFields& s,
                  const LateralFields& t) {
    auto value = LateralPair(a, s, t);
    if (system.sum) system.sum(std::span(&value, 1));
    return value;
  };
  const auto energy = pair(x, fs, ft);  // <f, K_ref^-1 f>
  if (!lateral) {
    info.converged = true;
    info.residuals.push_back(0.0);
    return {std::move(x), info};
  }
  if (start != nullptr) x = *start;
  // r = f - A x
  auto [rs, rt] = system.apply(x);
  LateralAxpy(rs, -1.0, fs);
  LateralAxpy(rt, -1.0, ft);
  for (auto& [l, v] : rs) for (auto& c : v) c = -c;
  for (auto& [l, v] : rt) for (auto& c : v) c = -c;
  auto z = system.solve(rs, rt);
  auto rz = pair(z, rs, rt);
  auto relative = [&](double value) {
    return energy > 0.0 ? std::sqrt(std::abs(value) / energy)
                        : std::sqrt(std::abs(value));
  };
  info.residual = relative(rz);
  info.residuals.push_back(info.residual);
  if (rz < 0.0) info.indefinite = true;
  const auto target = options.reduction > 0.0
                          ? std::max(options.rtol, options.reduction * info.residual)
                          : options.rtol;
  auto p = z;
  for (auto iteration = 0; iteration < options.max_iterations; ++iteration) {
    if (info.residual <= target) {
      info.converged = true;
      break;
    }
    auto [as, at] = system.apply(p);
    const auto curvature = pair(p, as, at);
    if (curvature <= 0.0) info.indefinite = true;
    const auto alpha = rz / curvature;
    LateralAxpy(x.spheroidal, alpha, p.spheroidal);
    LateralAxpy(x.toroidal, alpha, p.toroidal);
    LateralAxpy(rs, -alpha, as);
    LateralAxpy(rt, -alpha, at);
    z = system.solve(rs, rt);
    const auto next = pair(z, rs, rt);
    if (next < 0.0) info.indefinite = true;
    const auto beta = next / rz;
    rz = next;
    for (auto& [l, pl] : p.spheroidal) {
      const auto& zl = z.spheroidal.at(l);
      for (auto i = std::size_t{0}; i < pl.size(); ++i) pl[i] = zl[i] + beta * pl[i];
    }
    for (auto& [l, pl] : p.toroidal) {
      const auto& zl = z.toroidal.at(l);
      for (auto i = std::size_t{0}; i < pl.size(); ++i) pl[i] = zl[i] + beta * pl[i];
    }
    info.iterations = iteration + 1;
    info.residual = relative(rz);
    info.residuals.push_back(info.residual);
  }
  if (info.residual <= target) info.converged = true;
  return {std::move(x), info};
}

}  // namespace sl3dpp::detail
