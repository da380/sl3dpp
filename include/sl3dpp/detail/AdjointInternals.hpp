#pragma once

/**
 * @file detail/AdjointInternals.hpp
 * @brief The pieces the 1D adjoint sweeps share with their second-order
 * forms (AdjointSecondOrder.cpp): the dual packers between the
 * evolution's degree columns and coefficient space, the time grid's node
 * weights, the measurement check, and the trajectory cursor
 * (detail/TrajectoryCursor.hpp). Implementation detail; array layouts are
 * Viscoelastic.hpp's.
 */

#include <complex>
#include <format>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

#include "sl3dpp/detail/InteriorSources.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/StepStages.hpp"
#include "sl3dpp/detail/TrajectoryCursor.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp::detail {


using Complex = std::complex<double>;

/// The transpose of Evolution::Columns under the plain per-column
/// pairing on the degree side and the L2 coefficient pairing of real
/// fields on the other: per-degree column duals to a coefficient dual.
/// The pairing weights m > 0 twice, so those entries halve; the m = 0
/// imaginary column pairs with nothing (real fields) and is dropped.
inline std::vector<Complex> CoefficientDualFromColumnDuals(
    const SurfaceGrid& grid, const std::map<int, std::vector<double>>& duals,
    double factor) {
  auto out = grid.ZeroCoefficients();
  for (const auto& [l, cols] : duals) {
    out[SurfaceGrid::Index(l, 0)] = factor * cols[0];
    for (auto m = 1; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] =
          0.5 * factor * Complex(cols[m], cols[static_cast<std::size_t>(l) + 1 + m]);
    }
  }
  return out;
}

/// The moment kernels of dual solutions: the strain at the given points
/// (solver units) of a step's dual solutions, spheroidal and toroidal,
/// as real columns. The dual of a step's forces is its dual solution,
/// column by real column; a source's forces are the conjugate-paired
/// strain functional, so the kernel is the strain of the dual solution
/// with its m > 0 columns halved. Not `halved`: the natural columns as
/// the coefficients of a real field (a continuous adjoint displacement).
inline std::vector<SphericalTensor> MomentKernelsOfDuals(
    const ViscoelasticOperator& op, const ToroidalViscoelastic& toroidal,
    const Scales& sle, double toNdLength, std::span<const BodyPoint> points,
    const std::map<int, std::vector<double>>& y,
    const std::map<int, std::vector<double>>& yT, bool halved = true) {
  auto asComplex = [halved](const std::map<int, std::vector<double>>& from) {
    auto out = std::map<int, std::vector<Complex>>{};
    for (const auto& [l, x] : from) {
      const auto ndof = x.size() / (2 * (static_cast<std::size_t>(l) + 1));
      auto c = std::vector<Complex>(ndof * (static_cast<std::size_t>(l) + 1));
      for (auto m = 0; m <= l; ++m) {
        const auto weight = m == 0 || !halved ? 1.0 : 0.5;
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          c[static_cast<std::size_t>(m) * ndof + d] =
              weight * Complex(x[static_cast<std::size_t>(m) * ndof + d],
                               x[static_cast<std::size_t>(l + 1 + m) * ndof + d]);
        }
      }
      out.emplace(l, std::move(c));
    }
    return out;
  };
  const auto fields =
      DegreeFields{.spheroidal = asComplex(y), .toroidal = asComplex(yT)};
  const auto toMatMoment = MomentToMaterial(op, sle);
  auto out = std::vector<SphericalTensor>{};
  for (auto point : points) {
    point.radius /= toNdLength;
    auto e = PointStrain(
        op.BaseMaterial(), op.LMax(),
        [&](int l) -> const DegreeOperator& { return op.UnrelaxedSystem(l); },
        [&](int l) -> const ToroidalOperator& { return toroidal.Unrelaxed(l); },
        fields, point);
    for (auto* c : {&e.rr, &e.tt, &e.pp, &e.rt, &e.rp, &e.tp}) *c *= toMatMoment;
    out.push_back(e);
  }
  return out;
}

/// detail::FoldOnNodes for the moment kernels: an inner stage's, per
/// point, handed to the step's two nodes.
inline void FoldMomentsOnNodes(
    std::vector<std::vector<SphericalTensor>>& perNode, std::size_t node,
    double c, const std::vector<SphericalTensor>& stageKernel) {
  if (stageKernel.empty()) return;
  for (const auto& [to, a] :
       {std::pair{node - 1, 1.0 - c}, std::pair{node, c}}) {
    auto& kernel = perNode[to];
    if (kernel.empty()) kernel.assign(stageKernel.size(), SphericalTensor{});
    for (auto p = std::size_t{0}; p < kernel.size(); ++p) {
      auto& k = kernel[p];
      const auto& add = stageKernel[p];
      k.rr += a * add.rr;
      k.tt += a * add.tt;
      k.pp += a * add.pp;
      k.rt += a * add.rt;
      k.rp += a * add.rp;
      k.tp += a * add.tp;
    }
  }
}

/// The transpose of Evolution::Coefficients: a coefficient dual
/// to one degree's column duals (the m > 0 entries doubled by the
/// pairing).
inline std::vector<double> ColumnDualsOf(std::span<const Complex> dual, int l,
                                  double factor) {
  auto cols = std::vector<double>(2 * (static_cast<std::size_t>(l) + 1), 0.0);
  cols[0] = factor * dual[SurfaceGrid::Index(l, 0)].real();
  for (auto m = 1; m <= l; ++m) {
    const auto d = dual[SurfaceGrid::Index(l, m)];
    cols[m] = 2.0 * factor * d.real();
    cols[static_cast<std::size_t>(l) + 1 + m] = 2.0 * factor * d.imag();
  }
  return cols;
}

/// The trapezoid weights of the time grid.
inline std::vector<double> NodeWeights(std::span<const double> times) {
  auto w = std::vector<double>(times.size());
  // one node: the elastic problem, with no interval to weigh
  if (times.size() == 1) return std::vector<double>{0.0};
  w.front() = 0.5 * (times[1] - times[0]);
  w.back() = 0.5 * (times[times.size() - 1] - times[times.size() - 2]);
  for (auto n = std::size_t{1}; n + 1 < times.size(); ++n) {
    w[n] = 0.5 * (times[n + 1] - times[n - 1]);
  }
  return w;
}

inline void CheckGrids(const Measurement& m, std::size_t fieldSize,
                std::size_t node) {
  for (const auto* f :
       {&m.relativeSeaLevelDual, &m.displacementDual, &m.potentialDual,
        &m.tangentialPotentialDual, &m.toroidalPotentialDual}) {
    if (!f->empty() && f->size() != fieldSize) {
      throw std::invalid_argument(std::format(
          "the measurement at node {} does not live on the solver grid",
          node));
    }
  }
}

/// Whether any measurement reads the toroidal potential W (a dual of it
/// that is not identically zero).
inline bool ReadsTheToroidalPotential(
    const std::map<std::size_t, Measurement>& measurements) {
  for (const auto& [node, m] : measurements) {
    for (const auto v : m.toroidalPotentialDual) {
      if (v != 0.0) return true;
    }
  }
  return false;
}

/// The adjoint tangential traction of a measurement of the tangential
/// displacement's potential V, as the coefficients of its potential in
/// load units: the measurement int g_v V dS is that of the tangential
/// displacement against the vector weight grad_1 a, g_v = -div_1 grad_1 a,
/// and the adjoint traction is minus the weight, so the potential is
/// -a, at degree l the coefficient of -g_v over l (l + 1).
inline std::vector<Complex> TractionPotentialOfTangentialDual(
    const SurfaceGrid& grid, std::span<const double> dual) {
  auto out = grid.Expand(dual);
  for (auto l = 0; l <= grid.LMax(); ++l) {
    const auto factor = l > 0 ? -1.0 / (l * (l + 1.0)) : 0.0;
    for (auto m = 0; m <= l; ++m) out[SurfaceGrid::Index(l, m)] *= factor;
  }
  return out;
}

}  // namespace sl3dpp::detail

namespace sl3dpp {

/**
 * @brief The material pieces of the 1D second-order sweeps
 * (AdjointSecondOrder.cpp; docs/adjoints.md §15): a direction v in
 * the moduli (kappa, mu; may be empty), the branch fractions
 * f_k = mu_k / mu_U, v's variation of the unrelaxed and of each dt's
 * effective material (delta mu scaled by 1 - sum_k f_k (1 - w_k), the
 * material tangent's), v's forces on a full solution and a decayed
 * memory, and the recovery of v's full tangent solution at a node from
 * its recorded snapshot.
 */
struct AdjointSolver::MaterialTools {
  using Degree = ViscoelasticOperator::Degree;
  using Fields = std::map<int, std::vector<double>>;

  const Evolution* ev = nullptr;
  const ViscoelasticOperator* op = nullptr;
  /// v's direction: the moduli at the memory points.
  PointMaterialDirection direction;
  /// The branch fractions f_k = mu_k / mu_U at the memory points.
  std::vector<std::vector<double>> fractions;
  /// delta mu_v / mu_U at the points (zero where mu_U vanishes); empty
  /// without a mu direction.
  std::vector<double> muRatio;
  /// v's variation of the effective shear of a stage (its one-stage
  /// scheme and length) at the points.
  std::map<std::pair<Scheme, double>, std::vector<double>> effectiveMu;

  MaterialTools(const Evolution& evolution,
                const PointMaterialDirection& d)
      : ev(&evolution), op(&*evolution.op_), direction(d) {
    const auto& rheology = op->GetRheology();
    const auto muU = op->UnrelaxedShearAtPoints();
    if ((!d.kappa.empty() && d.kappa.size() != muU.size()) ||
        (!d.mu.empty() && d.mu.size() != muU.size())) {
      throw std::invalid_argument(
          "a material direction's moduli live at the memory points, "
          "(element, point)");
    }
    for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
      const auto f = op->BranchFractionAtPoints(k);
      fractions.emplace_back(f.begin(), f.end());
    }
    if (!direction.mu.empty()) {
      // against the unrelaxed shear the operator steps with, the
      // reference's under its own perturbation (Options::moduli)
      const auto& shear = op->options_.moduli.shear;
      muRatio.assign(muU.size(), 0.0);
      for (auto i = std::size_t{0}; i < muU.size(); ++i) {
        const auto current = muU[i] * (shear.empty() ? 1.0 : 1.0 + shear[i]);
        muRatio[i] = current > 0.0 ? direction.mu[i] / current : 0.0;
      }
    }
  }

  [[nodiscard]] bool HasDirection() const {
    return !direction.kappa.empty() || !direction.mu.empty();
  }
  /// The memory points over the whole mesh, where the kernels in kappa
  /// and mu live.
  [[nodiscard]] std::size_t PointCount() const {
    return op->UnrelaxedShearAtPoints().size();
  }

  /// v's variation of the effective shear of a (one-stage scheme, dt):
  /// delta mu scaled by 1 - sum_k f_k (1 - w_k) at the points.
  const std::vector<double>& EffectiveMu(Scheme scheme, double dt) {
    const auto key = std::pair(scheme, dt);
    auto found = effectiveMu.find(key);
    if (found != effectiveMu.end()) return found->second;
    const bool implicit = scheme == Scheme::BackwardEuler;
    const auto& weights = op->SchemeWeightsOf(scheme, dt);
    auto scaled = direction.mu;
    for (auto i = std::size_t{0}; i < scaled.size(); ++i) {
      auto factor = 1.0;
      for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
        const auto w = implicit ? weights[k].e[i] : 1.0 - weights[k].beta[i];
        factor -= fractions[k][i] * (1.0 - w);
      }
      scaled[i] *= factor;
    }
    return effectiveMu.emplace(key, std::move(scaled)).first->second;
  }

  /// v's variation of the effective system of a (one-stage scheme, dt) on
  /// x, and of the unrelaxed systems (every weight one).
  [[nodiscard]] std::vector<double> ApplyEffective(const DegreeOperator& system,
                                                   Scheme scheme, double dt,
                                                   std::span<const double> x,
                                                   int ncols) {
    const auto none = std::vector<double>{};
    return system.ApplyPointVariation(
        op->Points(), direction.kappa,
        direction.mu.empty() ? none : EffectiveMu(scheme, dt), x, ncols);
  }
  [[nodiscard]] std::vector<double> ApplyUnrelaxed(const DegreeOperator& system,
                                                   std::span<const double> x,
                                                   int ncols) const {
    return system.ApplyPointVariation(op->Points(), direction.kappa,
                                      direction.mu, x, ncols);
  }

  /// The memory force of a memory state (K, 3, ne, np, ncols) with the
  /// branch moduli's variation delta mu_k = f_k delta mu_v, per degree.
  [[nodiscard]] std::vector<double> MemoryVariationForce(
      const Degree& deg, int ncols, const std::vector<double>& mem) const {
    const auto block = deg.NodalSize(ncols);
    const auto wc = detail::ScalarWeights(deg.k2);
    auto weighted = std::vector<double>(3 * block, 0.0);
    for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto dmu = fractions[k][full] * direction.mu[full];
            if (dmu == 0.0) continue;
            const auto factor =
                wc[s] * dmu * deg.Wq[static_cast<std::size_t>(e) * deg.np + q];
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto src = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              weighted[s * block + node + c] += factor * mem[src + c];
            }
          }
        }
      }
    }
    return deg.DualFromScalars(weighted, ncols);
  }

  /// v's force on a full solution and a decayed memory under the dt
  /// flavour: -(dK_eff) x plus the memory force's variation.
  [[nodiscard]] Fields Force(double dt, const Fields& full, const Fields& mem) {
    const auto scheme = ev->options_.scheme;
    const auto flavour = op->FlavourOf(scheme, dt);
    auto out = Fields{};
    for (const auto& [l, x] : full) {
      const auto& deg = op->Deg(l);
      const auto ncols = static_cast<int>(x.size() / deg.ndof);
      auto force =
          ApplyEffective(op->EffectiveSystem(*flavour, l), scheme, dt, x, ncols);
      for (auto& v : force) v = -v;
      if (!direction.mu.empty()) {
        const auto mf = MemoryVariationForce(deg, ncols, mem.at(l));
        for (auto i = std::size_t{0}; i < force.size(); ++i) force[i] += mf[i];
      }
      out.emplace(l, std::move(force));
    }
    return out;
  }

  /// The same on the unrelaxed systems (every weight one): -(dK_U) x
  /// plus, with a memory state, its force with the branch moduli's
  /// variation — the form a step's END solution satisfies with its new
  /// memory.
  [[nodiscard]] Fields UnrelaxedForce(const Fields& full, const Fields* mem) const {
    auto out = Fields{};
    for (const auto& [l, x] : full) {
      const auto& deg = op->Deg(l);
      const auto ncols = static_cast<int>(x.size() / deg.ndof);
      auto force = ApplyUnrelaxed(deg.system, x, ncols);
      for (auto& v : force) v = -v;
      if (mem != nullptr && !direction.mu.empty()) {
        const auto mf = MemoryVariationForce(deg, ncols, mem->at(l));
        for (auto i = std::size_t{0}; i < force.size(); ++i) force[i] += mf[i];
      }
      out.emplace(l, std::move(force));
    }
    return out;
  }

  /// The unrelaxed systems' response to a force per degree.
  [[nodiscard]] Fields UnrelaxedResponse(const Fields& force) const {
    auto out = Fields{};
    for (const auto& [l, f] : force) {
      const auto& deg = op->Deg(l);
      const auto ncols = static_cast<int>(f.size() / deg.ndof);
      out.emplace(l, deg.system.Solve(f, ncols));
    }
    return out;
  }

  /// Adds the unrelaxed response to v's unrelaxed force on (x, mem) to
  /// `to` — what v's own material part adds to a tangent solution.
  void AddUnrelaxedResponse(Fields& to, const Fields& baseFull,
                            const Fields* baseMem) const {
    if (!HasDirection()) return;
    const auto extra = UnrelaxedResponse(UnrelaxedForce(baseFull, baseMem));
    for (auto& [l, xl] : to) {
      const auto& add = extra.at(l);
      for (auto i = std::size_t{0}; i < xl.size(); ++i) xl[i] += add[i];
    }
  }

  /// v's full tangent solution at a node from its recorded snapshot: the
  /// unrelaxed solve under the snapshot's loads, memory and (where v
  /// perturbs the sources) its sources' forces, plus v's own
  /// unrelaxed material force's response on the base solution and memory
  /// there (empty for the start).
  [[nodiscard]] Fields TangentSolution(const Evolution::Snapshot& snap,
                                       const Fields& baseFull,
                                       const Fields* baseMem) const {
    return TangentSolution(
        detail::TangentRecord{&*snap.internalState, &snap.loadCoefficients,
                              &snap.tideCoefficients,
                              snap.sourceForces ? &*snap.sourceForces : nullptr},
        baseFull, baseMem);
  }
  /// The same from its record at a stage's end (detail::TangentRecordOf).
  [[nodiscard]] Fields TangentSolution(const detail::TangentRecord& snap,
                                       const Fields& baseFull,
                                       const Fields* baseMem) const {
    auto x = op->SolveElastic(*snap.state,
                              ev->Columns(*snap.load, ev->toMatDensity_),
                              ev->Columns(*snap.tide, ev->toMatPotential_));
    if (snap.forces != nullptr) {
      // v perturbs the sources: its solution is under their forces too
      for (const auto& [l, f] : snap.forces->spheroidal) {
        const auto add = op->UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
        auto& to = x.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += add[i];
      }
    }
    AddUnrelaxedResponse(x, baseFull, baseMem);
    return x;
  }

  /// The material pairing of a dual solution y (with its strain scalars
  /// sy) and a base solution x at a degree's points: -(dK/d kappa,
  /// dK/d mu) of the degree form at the memory points
  /// (DegreeOperator::PointGradient), the shear scaled by `factor` there,
  /// and, with a memory state, the
  /// memory force's variation sum_k f_k <sy, memory_k> on the memory
  /// window (the first-order sweeps' materialTerms).
  [[nodiscard]] static std::pair<std::vector<double>, std::vector<double>>
  MaterialPairing(const MemoryPoints& points, const Degree& deg, int ncols,
                  std::span<const double> y,
                  std::span<const double> sy, std::span<const double> x,
                  std::span<const double> factor,
                  const std::vector<double>* memory,
                  const std::vector<std::vector<double>>& fractions) {
    auto [kappa, mu] = deg.system.PointGradient(points, y, x, ncols);
    for (auto i = std::size_t{0}; i < kappa.size(); ++i) {
      kappa[i] = -kappa[i];
      mu[i] = -factor[i] * mu[i];
    }
    if (memory != nullptr) {
      const auto block = deg.NodalSize(ncols);
      const auto wc = detail::ScalarWeights(deg.k2);
      for (auto e = 0; e < deg.ne; ++e) {
        for (auto q = 0; q < deg.np; ++q) {
          const auto p = static_cast<std::size_t>(e) * deg.np + q;
          const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
          const auto node = p * ncols;
          auto mem = 0.0;
          for (auto c = 0; c < ncols; ++c) {
            for (auto s = 0; s < 3; ++s) {
              const auto sn = s * block + node + c;
              for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
                mem += fractions[k][full] * wc[s] * sy[sn] *
                       (*memory)[(k * 3 + s) * block + node + c];
              }
            }
          }
          mu[full] += deg.Wq[p] * mem;
        }
      }
    }
    return std::pair(std::move(kappa), std::move(mu));
  }
};

/// Adds a pair of nodal arrays to a pair of kernels.
inline void AddPairs(std::vector<double>& kappaTo, std::vector<double>& muTo,
                     const std::pair<std::vector<double>, std::vector<double>>& p) {
  for (auto i = std::size_t{0}; i < kappaTo.size(); ++i) {
    kappaTo[i] += p.first[i];
    muTo[i] += p.second[i];
  }
}

}  // namespace sl3dpp
