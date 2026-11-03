// The second-order adjoint sweeps of AdjointSolver: the kernels of an
// objective and their derivative along a direction v of the model
// parameters, by the discrete method (the exact derivative of the exact
// gradient) and by the continuous one (the second-order adjoint equations
// discretised by the forward stepper), as pyslfp's
// `second_order_adjoint_ice_history` and `_second_order_continuous_sweep`
// under the sharp flotation rule; see docs/dev/adjoints.md.

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <stdexcept>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/detail/AdjointInternals.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

using detail::CoefficientDualFromColumnDuals;
using detail::ColumnDualsOf;
using detail::Complex;
using detail::NodeWeights;
using detail::TrajectoryCursor;

namespace {

/// The scheme weights of one branch at one memory point, with their first
/// and second h-derivatives, in the trapezoid's (E, A, B) form: backward
/// Euler keeps E = w, A = 0, B = 1 - w (pyslfp's `triples`).
struct Triple {
  double E, A, B;
};

struct Triples {
  Triple w, dw, d2w;
};

/// The thickness and initial-sea-level views of a set of node duals (the
/// kernels' assembly applied to the derivative duals).
void DerivativeViews(KernelDerivatives& out, std::span<const double> C,
                     double rhoI, double rhoW, std::span<const double> times,
                     std::vector<std::vector<double>> duals,
                     std::span<const std::vector<double>> oceanFunctions) {
  auto views = detail::AssembleIceViews(
      C, rhoI, times, std::move(duals),
      std::vector<std::vector<double>>(times.size(),
                                       std::vector<double>(C.size(), 0.0)),
      {}, oceanFunctions, rhoW);
  out.times.assign(times.begin(), times.end());
  out.thickness = std::move(views.thickness);
  out.initialSeaLevel = std::move(views.initialSeaLevel);
}

}  // namespace

std::pair<IceKernels, KernelDerivatives> AdjointSolver::SecondOrderKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const SeaLevelEvolution::Snapshot> tangent,
    const std::vector<std::vector<double>>& itau,
    const ForwardTrajectory& trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  if (times.size() < 2) {
    throw std::invalid_argument("times must hold at least two entries");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto fieldSize = ev.solver_->Grid().FieldSize();
  for (const auto& [node, m] : measurements) {
    if (node >= times.size()) {
      throw std::invalid_argument(
          std::format("measurement node {} is off the time grid", node));
    }
    detail::CheckGrids(m, fieldSize, node);
  }
  if (trajectory.states.size() != times.size() ||
      !trajectory.states.back().has_value() ||
      trajectory.loads.size() != times.size() ||
      trajectory.tides.size() != times.size()) {
    throw std::invalid_argument(
        "the trajectory must be recorded forward states, one entry per "
        "time node with the recorded load and centrifugal coefficients");
  }
  if (tangent.size() != times.size()) {
    throw std::invalid_argument(
        "the tangent-linear run must give one snapshot per time node");
  }
  for (const auto& snap : tangent) {
    if (!snap.internalState.has_value() || !snap.internalState->HasStrain()) {
      throw std::invalid_argument(
          "the tangent-linear run must be recorded at every node "
          "(TangentLinearSolver::Options::record)");
    }
  }
  if (!oceanFunctions.empty()) {
    if (oceanFunctions.size() != times.size()) {
      throw std::invalid_argument(
          "the ocean functions must give one grid per time node");
    }
    for (const auto& Cn : oceanFunctions) {
      if (Cn.size() != fieldSize) {
        throw std::invalid_argument(
            "the ocean functions must be fields on the solver's grid");
      }
    }
  }
  if (ev.options_.rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.BaseMaterial(), 2);
    if (ev.solver_->TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "SeaLevelEvolution::Run");
    }
  }
  // v's perturbation at the memory points; zero without one.
  auto ditauPoints = std::vector<std::vector<double>>{};
  if (itau.empty()) {
    ditauPoints.assign(op.GetRheology().BranchCount(),
                       std::vector<double>(op.pointItau_[0].size(), 0.0));
  } else {
    ditauPoints = op.ItauPerturbationAtPoints(itau);
  }
  return options_.method == AdjointMethod::Discrete
             ? DiscreteSecondOrder(times, measurements, tangent, ditauPoints,
                                   trajectory, oceanFunctions)
             : ContinuousSecondOrder(times, measurements, tangent,
                                     ditauPoints, trajectory, oceanFunctions);
}

// -- the discrete second-order sweep
// ------------------------------------------

std::pair<IceKernels, KernelDerivatives> AdjointSolver::DiscreteSecondOrder(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const SeaLevelEvolution::Snapshot> tangent,
    const std::vector<std::vector<double>>& ditauPoints,
    const ForwardTrajectory& trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto scheme = ev.options_.scheme;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto rotation = ev.options_.rotation;
  const auto& C = ev.state_.oceanFunction;
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return oceanFunctions.empty() ? C : oceanFunctions[node];
  };
  auto overrideAt = [&](std::size_t node) {
    return oceanFunctions.empty() ? std::vector<double>{}
                                  : oceanFunctions[node];
  };
  const auto rhoW = solver.WaterDensity();
  const auto rhoI = solver.IceDensity();
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto lmax = op.LMax();
  const auto K = op.GetRheology().BranchCount();
  const auto N = times.size() - 1;

  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<double, SeaLevelSolver>{};
  auto solverFor = [&](double dt) -> const SeaLevelSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(dt, SeaLevelSolver(
                                   grid,
                                   dt == 0.0 ? op.ElasticLoveNumbers()
                                             : op.EffectiveLoveNumbers(
                                                   ev.options_.scheme, dt),
                                   solver.Parameters()))
                  .first;
    }
    return found->second;
  };

  // The dual state and its derivative along v.
  std::map<int, std::vector<double>> mdual, ddual, dmdual, dddual;
  for (auto l = 0; l <= lmax; ++l) {
    const auto& deg = op.Deg(l);
    const auto ncols = 2 * (l + 1);
    mdual[l].assign(K * 3 * deg.NodalSize(ncols), 0.0);
    dmdual[l].assign(K * 3 * deg.NodalSize(ncols), 0.0);
    ddual[l].assign(3 * deg.NodalSize(ncols), 0.0);
    dddual[l].assign(3 * deg.NodalSize(ncols), 0.0);
  }
  auto dts = std::vector<double>{};
  for (auto m = std::size_t{0}; m + 1 < times.size(); ++m) {
    dts.push_back((times[m + 1] - times[m]) * ev.toMatTime_);
  }
  auto cursor = TrajectoryCursor(
      op, trajectory, dts, scheme,
      [&ev](const std::vector<Complex>& c) {
        return ev.Columns(c, ev.toMatDensity_);
      },
      [&ev](const std::vector<Complex>& c) {
        return ev.Columns(c, ev.toMatPotential_);
      });
  const auto meshSize = op.pointItau_[0].size();
  auto itauKernel = std::vector<std::vector<double>>(
      K, std::vector<double>(meshSize, 0.0));
  auto dItauKernel = itauKernel;

  // The engine-adjoint inputs at a node: the duals carried by the node's
  // total-load and centrifugal coefficients, plus the measurement's for
  // the first sweep.
  auto engineDuals = [&](const std::vector<Complex>& sigmaDual,
                         const std::vector<Complex>& psiDual,
                         std::size_t node, bool measured) {
    SeaLevelSolver::AdjointForcing duals;
    duals.rotation = rotation;
    duals.oceanFunctionOverride = overrideAt(node);
    const auto& Cn = oceanAt(node);
    auto synth = grid.Synthesise(sigmaDual);
    duals.seaLevelDual.assign(fieldSize, 0.0);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      duals.seaLevelDual[i] = rhoW * Cn[i] * synth[i] / r2;
    }
    auto gOmega = std::array<double, 3>{0.0, 0.0, 0.0};
    if (rotation != RotationalFeedback::None) {
      gOmega = solver.CentrifugalDual(psiDual);
    }
    if (auto found = measurements.find(node);
        measured && found != measurements.end()) {
      const auto& m = found->second;
      for (auto i = std::size_t{0}; i < m.seaLevelDual.size(); ++i) {
        duals.seaLevelDual[i] += m.seaLevelDual[i];
      }
      duals.displacementDual = m.displacementDual;
      duals.potentialDual = m.potentialDual;
      for (auto j = 0; j < 3; ++j) gOmega[j] += m.angularVelocityDual[j];
    }
    duals.angularVelocityDual = gOmega;
    return std::pair(std::move(duals), std::move(synth));
  };
  auto adjointSolve = [&](double dt, const std::vector<Complex>& sigmaDual,
                          const std::vector<Complex>& psiDual,
                          std::size_t node, bool measured,
                          std::vector<double>& loadDual) {
    auto [duals, synth] = engineDuals(sigmaDual, psiDual, node, measured);
    const auto adj = solverFor(dt).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(std::format(
          "the adjoint sea level solve at node {} did not converge "
          "(error {})",
          node, adj.error));
    }
    loadDual.assign(fieldSize, 0.0);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      loadDual[i] = adj.directLoadDual[i] + synth[i] / r2;
    }
    return adj;
  };
  // The contraction of a degree's dual vector with the unit responses:
  // the duals of the step's load and tide coefficients.
  auto contract = [&](const ViscoelasticOperator::Degree& deg,
                      const std::vector<double>& units,
                      const std::vector<double>& x, int ncols,
                      std::vector<double>& sc, std::vector<double>& pc) {
    sc.assign(ncols, 0.0);
    pc.assign(ncols, 0.0);
    for (auto c = 0; c < ncols; ++c) {
      auto s = 0.0, p = 0.0;
      for (auto d = 0; d < deg.ndof; ++d) {
        const auto v = x[static_cast<std::size_t>(c) * deg.ndof + d];
        s += units[d] * v;
        p += units[static_cast<std::size_t>(deg.ndof) + d] * v;
      }
      sc[c] = s;
      pc[c] = p;
    }
  };
  auto pack = [&](std::vector<std::vector<double>>& sig,
                  std::vector<std::vector<double>>& psi) {
    std::map<int, std::vector<double>> sigCols, psiCols;
    for (auto l = 0; l <= lmax; ++l) {
      sigCols.emplace(l, std::move(sig[l]));
      psiCols.emplace(l, std::move(psi[l]));
    }
    return std::pair(CoefficientDualFromColumnDuals(grid, sigCols, ev.toMatDensity_),
                     CoefficientDualFromColumnDuals(grid, psiCols,
                                                    ev.toMatPotential_));
  };
  auto addSurfaceDuals = [&](const ViscoelasticOperator::Degree& deg, int l,
                             int ncols, const SeaLevelSolver::AdjointResponse& adj,
                             std::vector<double>& rhs) {
    const auto suCols =
        ColumnDualsOf(adj.displacementResponseDual, l, ev.toNdLength_);
    const auto sphiCols =
        ColumnDualsOf(adj.potentialResponseDual, l, ev.toNdPotential_);
    const auto su = deg.system.SurfaceDof(Component::U);
    const auto sphi = deg.system.SurfaceDof(Component::Phi);
    for (auto c = 0; c < ncols; ++c) {
      if (su >= 0) rhs[static_cast<std::size_t>(c) * deg.ndof + su] += suCols[c];
      if (sphi >= 0) {
        rhs[static_cast<std::size_t>(c) * deg.ndof + sphi] += sphiCols[c];
      }
    }
    if (l == 0) {
      const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
      for (auto d = 0; d < deg.ndof; ++d) rhs[d] += deg.momentRow[d] * jd;
    }
  };

  auto loadDuals = std::vector<std::vector<double>>(times.size());
  auto dloadDuals = std::vector<std::vector<double>>(times.size());

  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = dts[n - 1];
    const auto& weights = op.SchemeWeightsOf(scheme, dt);
    const auto& dweights = op.SchemeWeightDerivativesOf(scheme, dt);
    const auto& d2weights = op.SchemeWeightSecondDerivativesOf(scheme, dt);
    auto& flavour = op.FlavourOf(scheme, dt);
    const ViscoelasticState fwdN = cursor.State(n);
    const ViscoelasticState& fwdPrev = cursor.State(n - 1);
    const ViscoelasticState& vN = *tangent[n].internalState;
    const ViscoelasticState& vPrev = *tangent[n - 1].internalState;

    auto degs = std::vector<const ViscoelasticOperator::Degree*>(lmax + 1);
    auto unitsOf = std::vector<const std::vector<double>*>(lmax + 1);
    auto systems = std::vector<const DegreeOperator*>(lmax + 1);
    for (auto l = 0; l <= lmax; ++l) {
      degs[l] = &op.Deg(l);
      unitsOf[l] = &op.UnitResponses(&flavour, l);
      systems[l] = &op.EffectiveSystem(flavour, l);
    }
    auto triplesAt = [&](std::size_t k, std::size_t full) -> Triples {
      const auto& w = weights[k];
      const auto& dw = dweights[k];
      const auto& d2w = d2weights[k];
      if (implicit) {
        return {{w.e[full], 0.0, 1.0 - w.e[full]},
                {dw.e[full], 0.0, -dw.e[full]},
                {d2w.e[full], 0.0, -d2w.e[full]}};
      }
      return {{w.e[full], w.alpha[full], w.beta[full]},
              {dw.e[full], dw.alpha[full], dw.beta[full]},
              {d2w.e[full], d2w.alpha[full], d2w.beta[full]}};
    };

    // 1. The duals of the end-of-step solve, through the strain
    // transpose, and their contraction with the unit responses.
    auto xd = std::vector<std::vector<double>>(lmax + 1);
    auto sigBy = std::vector<std::vector<double>>(lmax + 1);
    auto psiBy = std::vector<std::vector<double>>(lmax + 1);
    detail::ParallelFor(lmax + 1, op.options_.parallel, [&](int i) {
      const auto l = lmax - i;
      const auto& deg = *degs[l];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      auto summed = ddual.at(l);
      const auto& md = mdual.at(l);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const auto B = triplesAt(k, full).w.B;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const auto at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                summed[s * block + node + c] += B * md[at + c];
              }
            }
          }
        }
      }
      xd[l] = deg.DualFromScalars(summed, ncols);
      contract(deg, *unitsOf[l], xd[l], ncols, sigBy[l], psiBy[l]);
    });
    const auto [sigmaDual, psiDual] = pack(sigBy, psiBy);
    const auto adj = adjointSolve(dt, sigmaDual, psiDual, n, true, loadDuals[n]);

    // 2. The first sweep's radial solve, and from it the second sweep's
    // dual of the step's solution: the derivative of the effective system
    // acting on the first sweep's solution lands on the full solution, so
    // it reaches the load and the tide and passes through the sea level
    // adjoint with the rest.
    auto ys = std::vector<std::vector<double>>(lmax + 1);
    auto sys = std::vector<std::vector<double>>(lmax + 1);
    auto wsys = std::vector<std::vector<double>>(lmax + 1);
    auto dxd = std::vector<std::vector<double>>(lmax + 1);
    detail::ParallelFor(lmax + 1, op.options_.parallel, [&](int i) {
      const auto l = lmax - i;
      const auto& deg = *degs[l];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      auto rhs = xd[l];
      addSurfaceDuals(deg, l, ncols, adj, rhs);
      ys[l] = systems[l]->Solve(rhs, ncols);
      sys[l] = deg.Strain(ys[l], ncols);
      wsys[l] = deg.ApplyWeights(sys[l], ncols);
      auto dsummed = dddual.at(l);
      auto source = std::vector<double>(3 * block, 0.0);
      const auto& md = mdual.at(l);
      const auto& dmd = dmdual.at(l);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const auto T = triplesAt(k, full);
              const auto dh = dt * ditauPoints[k][full];
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const auto at = (k * 3 + s) * block + node;
              const auto sn = s * block + node;
              for (auto c = 0; c < ncols; ++c) {
                dsummed[sn + c] += T.w.B * dmd[at + c] + T.dw.B * dh * md[at + c];
                source[sn + c] += T.dw.B * dh * wsys[l][at + c];
              }
            }
          }
        }
      }
      dxd[l] = deg.DualFromScalars(dsummed, ncols);
      const auto fromSource = deg.DualFromScalars(source, ncols);
      for (auto j = std::size_t{0}; j < dxd[l].size(); ++j) {
        dxd[l][j] += fromSource[j];
      }
      contract(deg, *unitsOf[l], dxd[l], ncols, sigBy[l], psiBy[l]);
    });
    const auto [dsigmaDual, dpsiDual] = pack(sigBy, psiBy);
    const auto dadj =
        adjointSolve(dt, dsigmaDual, dpsiDual, n, false, dloadDuals[n]);

    // 3. The second sweep's radial solve, the itau pairings of both
    // sweeps, and the weight recursion closing both dual states.
    auto kernelTerms = std::vector<std::vector<double>>(lmax + 1);
    auto dKernelTerms = std::vector<std::vector<double>>(lmax + 1);
    detail::ParallelFor(lmax + 1, op.options_.parallel, [&](int i) {
      const auto l = lmax - i;
      const auto& deg = *degs[l];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      auto drhs = dxd[l];
      addSurfaceDuals(deg, l, ncols, dadj, drhs);
      const auto dy = systems[l]->Solve(drhs, ncols);
      const auto dsy = deg.Strain(dy, ncols);
      const auto wdsy = deg.ApplyWeights(dsy, ncols);
      const auto& sy = sys[l];
      const auto& wsy = wsys[l];
      const auto& mPrev = fwdPrev.m.at(l);
      const auto& d0 = fwdPrev.strain.at(l);
      const auto& d1 = fwdN.strain.at(l);
      const auto& vmPrev = vPrev.m.at(l);
      const auto& vd0 = vPrev.strain.at(l);
      const auto& vd1 = vN.strain.at(l);
      const auto Wd1 = deg.ApplyWeights(d1, ncols);
      const auto Wvd1 = deg.ApplyWeights(vd1, ncols);
      auto& md = mdual.at(l);
      auto& dmd = dmdual.at(l);
      auto& dd = ddual.at(l);
      auto& ddd = dddual.at(l);
      std::ranges::fill(dd, 0.0);
      std::ranges::fill(ddd, 0.0);
      auto& terms = kernelTerms[l];
      auto& dterms = dKernelTerms[l];
      terms.assign(K * static_cast<std::size_t>(deg.ne) * deg.np, 0.0);
      dterms.assign(K * static_cast<std::size_t>(deg.ne) * deg.np, 0.0);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const auto T = triplesAt(k, full);
              const auto dh = dt * ditauPoints[k][full];
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const auto at = (k * 3 + s) * block + node;
              const auto sn = s * block + node;
              const auto pt = (k * deg.ne + e) * deg.np + q;
              for (auto c = 0; c < ncols; ++c) {
                const auto mdOld = md[at + c];
                const auto dmdOld = dmd[at + c];
                const auto mtilde = mdOld + wsy[at + c];
                const auto dmtilde = dmdOld + wdsy[at + c];
                const auto g =
                    mtilde * (T.dw.E * mPrev[at + c] + T.dw.A * d0[sn + c]) +
                    (mdOld * d1[sn + c] + Wd1[at + c] * sy[sn + c]) * T.dw.B;
                const auto dg =
                    dmtilde * (T.dw.E * mPrev[at + c] + T.dw.A * d0[sn + c]) +
                    mtilde * (T.dw.E * vmPrev[at + c] + T.dw.A * vd0[sn + c]) +
                    mtilde * (T.d2w.E * mPrev[at + c] + T.d2w.A * d0[sn + c]) *
                        dh +
                    (dmdOld * d1[sn + c] + mdOld * vd1[sn + c] +
                     Wvd1[at + c] * sy[sn + c] + Wd1[at + c] * dsy[sn + c]) *
                        T.dw.B +
                    (mdOld * d1[sn + c] + Wd1[at + c] * sy[sn + c]) * T.d2w.B *
                        dh;
                terms[pt] += dt * g;
                dterms[pt] += dt * dg;
                md[at + c] = T.w.E * mtilde;
                dmd[at + c] = T.w.E * dmtilde + T.dw.E * dh * mtilde;
                dd[sn + c] += T.w.A * mtilde;
                ddd[sn + c] += T.w.A * dmtilde + T.dw.A * dh * mtilde;
              }
            }
          }
        }
      }
    });
    for (auto l = 0; l <= lmax; ++l) {
      const auto& deg = *degs[l];
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto pt = (k * deg.ne + e) * deg.np + q;
            itauKernel[k][full] += kernelTerms[l][pt];
            dItauKernel[k][full] += dKernelTerms[l][pt];
          }
        }
      }
    }
  }

  // The start: the transposed instantaneous elastic solve at times[0],
  // whose system does not depend on the relaxation times.
  {
    auto sigBy = std::vector<std::vector<double>>(lmax + 1);
    auto psiBy = std::vector<std::vector<double>>(lmax + 1);
    auto dsigBy = std::vector<std::vector<double>>(lmax + 1);
    auto dpsiBy = std::vector<std::vector<double>>(lmax + 1);
    for (auto l = 0; l <= lmax; ++l) {
      const auto& deg = op.Deg(l);
      const auto ncols = 2 * (l + 1);
      const auto& units = op.UnitResponses(nullptr, l);
      contract(deg, units, deg.DualFromScalars(ddual.at(l), ncols), ncols,
               sigBy[l], psiBy[l]);
      contract(deg, units, deg.DualFromScalars(dddual.at(l), ncols), ncols,
               dsigBy[l], dpsiBy[l]);
    }
    const auto [sigmaDual, psiDual] = pack(sigBy, psiBy);
    (void)adjointSolve(0.0, sigmaDual, psiDual, 0, true, loadDuals[0]);
    const auto [dsigmaDual, dpsiDual] = pack(dsigBy, dpsiBy);
    (void)adjointSolve(0.0, dsigmaDual, dpsiDual, 0, false, dloadDuals[0]);
  }

  // The delta amplitudes and the distributional view, as the first-order
  // sweep forms them.
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  for (const auto& [node, m] : measurements) {
    SeaLevelSolver::AdjointForcing duals;
    duals.rotation = rotation;
    duals.oceanFunctionOverride = overrideAt(node);
    duals.seaLevelDual = m.seaLevelDual;
    duals.displacementDual = m.displacementDual;
    duals.potentialDual = m.potentialDual;
    duals.angularVelocityDual = m.angularVelocityDual;
    const auto adj = solverFor(0.0).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          "the instantaneous adjoint fingerprint did not converge");
    }
    lumps.emplace(node, adj.directLoadDual);
  }
  const auto w = NodeWeights(times);
  auto densityLoad = std::vector<std::vector<double>>(times.size());
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    densityLoad[n].assign(fieldSize, 0.0);
    const auto lump = lumps.find(n);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double lumped = lump == lumps.end() ? 0.0 : lump->second[i];
      densityLoad[n][i] = (loadDuals[n][i] - lumped) / w[n];
    }
  }
  auto kernels = AssembleViews(times, std::move(loadDuals),
                               std::move(densityLoad), std::move(lumps),
                               oceanFunctions);
  kernels.itauKernel = op.ItauKernelsAtNodes(std::move(itauKernel));
  KernelDerivatives derivatives;
  DerivativeViews(derivatives, C, rhoI, rhoW, times, std::move(dloadDuals),
                  oceanFunctions);
  derivatives.itauKernel = op.ItauKernelsAtNodes(std::move(dItauKernel));
  return {std::move(kernels), std::move(derivatives)};
}

// -- the continuous second-order sweep
// ----------------------------------------

std::pair<IceKernels, KernelDerivatives> AdjointSolver::ContinuousSecondOrder(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const SeaLevelEvolution::Snapshot> tangent,
    const std::vector<std::vector<double>>& ditauPoints,
    const ForwardTrajectory& trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto scheme = ev.options_.scheme;
  const auto rotation = ev.options_.rotation;
  const auto& C = ev.state_.oceanFunction;
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return oceanFunctions.empty() ? C : oceanFunctions[node];
  };
  auto overrideAt = [&](std::size_t node) {
    return oceanFunctions.empty() ? std::vector<double>{}
                                  : oceanFunctions[node];
  };
  const auto rhoW = solver.WaterDensity();
  const auto rhoI = solver.IceDensity();
  const auto g = solver.SurfaceGravity();
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto K = op.GetRheology().BranchCount();
  const auto N = times.size() - 1;

  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<double, SeaLevelSolver>{};
  auto solverFor = [&](double dt) -> const SeaLevelSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(dt, SeaLevelSolver(
                                   grid,
                                   dt == 0.0 ? op.ElasticLoveNumbers()
                                             : op.EffectiveLoveNumbers(
                                                   ev.options_.scheme, dt),
                                   solver.Parameters()))
                  .first;
    }
    return found->second;
  };

  // The adjoint state and its derivative along v, each stepped by the
  // forward machinery in reversed time, with their SLE-closure load and
  // centrifugal coefficients.
  struct Carried {
    ViscoelasticState state;
    std::vector<Complex> sigma, psi;
  };
  auto vs = Carried{op.ZeroState([](int l) { return 2 * (l + 1); }),
                    grid.ZeroCoefficients(), grid.ZeroCoefficients()};
  auto ds = Carried{op.ZeroState([](int l) { return 2 * (l + 1); }),
                    grid.ZeroCoefficients(), grid.ZeroCoefficients()};

  auto psiStar = [&](const std::vector<double>& potentialDual) {
    auto d = solver.CentrifugalDual(grid.Expand(potentialDual));
    for (auto& v : d) v *= r2;
    return d;
  };

  // The instantaneous adjoint fingerprint of a measurement through the
  // forward channel mapping (Adjoint.cpp's jump), and the strain of its
  // solution, which jumps the adjoint internal variables over the
  // relaxation time (Yu et al. eq. 80) — and their derivative by v's
  // perturbation of its inverse.
  struct Jump {
    std::vector<double> xiA;
    std::vector<Complex> lam, psiA, zu, zphi;
  };
  auto jump = [&](const Measurement& m, std::size_t node) {
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(node);
    if (!m.seaLevelDual.empty()) f.directLoad = m.seaLevelDual;
    std::vector<double> zetaU, zetaPhi;
    if (!m.displacementDual.empty()) {
      zetaU.resize(fieldSize);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        zetaU[i] = -m.displacementDual[i];
      }
      f.displacementLoad = zetaU;
    }
    if (!m.potentialDual.empty()) {
      zetaPhi.resize(fieldSize);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        zetaPhi[i] = -g * m.potentialDual[i];
      }
      f.potentialLoad = zetaPhi;
    }
    if (rotation != RotationalFeedback::None) {
      auto avc = std::array<double, 3>{0.0, 0.0, 0.0};
      for (auto j = 0; j < 3; ++j) avc[j] = -g * m.angularVelocityDual[j];
      if (!m.potentialDual.empty()) {
        const auto ps = psiStar(m.potentialDual);
        for (auto j = 0; j < 3; ++j) avc[j] += g * ps[j];
      }
      f.angularMomentumChange = avc;
    }
    const auto resp = solverFor(0.0).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(
          "the instantaneous adjoint fingerprint did not converge");
    }
    Jump out;
    out.xiA = resp.seaLevelChange;
    const auto& Cn = oceanAt(node);
    auto lamGrid = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      lamGrid[i] = (m.seaLevelDual.empty() ? 0.0 : m.seaLevelDual[i]) +
                   rhoW * Cn[i] * out.xiA[i];
    }
    out.lam = grid.Expand(lamGrid);
    out.psiA = rotation != RotationalFeedback::None
                   ? solver.CentrifugalCoefficients(resp.angularVelocity)
                   : grid.ZeroCoefficients();
    if (!zetaU.empty()) out.zu = grid.Expand(zetaU);
    if (!zetaPhi.empty()) out.zphi = grid.Expand(zetaPhi);
    return out;
  };
  auto jumpStrains = [&](const Jump& j) {
    const auto lamCols = ev.Columns(j.lam, ev.toMatDensity_);
    const auto psiCols = ev.Columns(j.psiA, ev.toMatPotential_);
    const auto zuCols =
        j.zu.empty() ? LoadCallable{} : ev.Columns(j.zu, ev.toMatDensity_);
    const auto zphiCols =
        j.zphi.empty() ? LoadCallable{} : ev.Columns(j.zphi, ev.toMatDensity_);
    auto out = std::map<int, std::vector<double>>{};
    for (const auto& [l, m] : vs.state.m) {
      const auto& deg = op.Deg(l);
      const int ncols = 2 * (l + 1);
      auto x =
          std::vector<double>(static_cast<std::size_t>(deg.ndof) * ncols, 0.0);
      auto add = [&](const double* unit, const std::vector<double>& amps) {
        for (auto c = 0; c < ncols; ++c) {
          for (auto d = 0; d < deg.ndof; ++d) {
            x[static_cast<std::size_t>(c) * deg.ndof + d] += unit[d] * amps[c];
          }
        }
      };
      const auto& units = op.UnitResponses(nullptr, l);
      add(units.data(), *lamCols(l, 0.0));
      add(units.data() + deg.ndof, *psiCols(l, 0.0));
      if (zuCols || zphiCols) {
        const auto& split = op.SplitUnitResponses(nullptr, l);
        if (zuCols) add(split.data(), *zuCols(l, 0.0));
        if (zphiCols) add(split.data() + deg.ndof, *zphiCols(l, 0.0));
      }
      out.emplace(l, deg.Strain(x, ncols));
    }
    return out;
  };
  // m += factor(x) d-tilde, with the factor itau (the adjoint state) or
  // v's delta itau (its derivative), at the memory points.
  auto applyJump = [&](ViscoelasticState& state,
                       const std::map<int, std::vector<double>>& strains,
                       bool derivative) {
    for (auto& [l, m] : state.m) {
      const auto& deg = op.Deg(l);
      const int ncols = 2 * (l + 1);
      const std::size_t block = deg.NodalSize(ncols);
      const auto& dTilde = strains.at(l);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const auto full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const double factor =
                  derivative
                      ? ditauPoints[k][full]
                      : deg.itau[k][static_cast<std::size_t>(e) * deg.np + q];
              if (factor == 0.0) continue;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const std::size_t at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                m[at + c] += factor * dTilde[s * block + node + c];
              }
            }
          }
        }
      }
    }
    state.Invalidate();
  };
  // The memory-only adjoint equilibrium of a carried state: its sea level
  // (the smooth kernel density, per material time) and the coefficients
  // the next reversed step starts from.
  auto smoothEquilibrium = [&](Carried& carried, std::size_t node) {
    const auto solutions = op.SolveElastic(carried.state);
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(node);
    f.displacementResponse =
        ev.Coefficients(op.Surface(solutions, Component::U), ev.toNdLength_);
    f.potentialResponse =
        ev.Coefficients(op.Surface(solutions, Component::Phi),
                        ev.toNdPotential_);
    f.inertiaMomentChange =
        op.InertiaMoment(solutions.at(0), 2)[0] * ev.toNdMoment_;
    const auto resp = solverFor(0.0).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(
          "the adjoint equilibrium solve did not converge");
    }
    const auto& Cn = oceanAt(node);
    auto sig = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      sig[i] = rhoW * Cn[i] * resp.seaLevelChange[i];
    }
    carried.sigma = grid.Expand(sig);
    carried.psi = rotation != RotationalFeedback::None
                      ? solver.CentrifugalCoefficients(resp.angularVelocity)
                      : grid.ZeroCoefficients();
    return resp.seaLevelChange;
  };
  // One reversed step of a carried state under its closure, with an
  // optional memory drive; returns the step's sea level (the late
  // density at the node stepped to).
  auto reversedStep = [&](Carried& carried, double dt, std::size_t to,
                          const std::map<int, std::vector<double>>* drive) {
    auto step = op.StartStep(carried.state, dt, scheme,
                             ev.Columns(carried.sigma, ev.toMatDensity_),
                             ev.Columns(carried.psi, ev.toMatPotential_));
    if (drive != nullptr) step.AddMemory(*drive);
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(to);
    f.displacementResponse =
        ev.Coefficients(step.MemorySurface(Component::U), ev.toNdLength_);
    f.potentialResponse =
        ev.Coefficients(step.MemorySurface(Component::Phi), ev.toNdPotential_);
    f.inertiaMomentChange = step.InertiaMoment()[0] * ev.toNdMoment_;
    const auto resp = solverFor(dt).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(std::format(
          "the adjoint coupled solve at node {} did not converge", to));
    }
    const auto& Cn = oceanAt(to);
    auto sig = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      sig[i] = rhoW * Cn[i] * resp.seaLevelChange[i];
    }
    carried.sigma = grid.Expand(sig);
    carried.psi = rotation != RotationalFeedback::None
                      ? solver.CentrifugalCoefficients(resp.angularVelocity)
                      : grid.ZeroCoefficients();
    auto [next, solutions] =
        step.Finish(ev.Columns(carried.sigma, ev.toMatDensity_),
                    ev.Columns(carried.psi, ev.toMatPotential_));
    carried.state = std::move(next);
    return resp.seaLevelChange;
  };

  std::vector<std::vector<double>> densityLate(times.size()),
      densityEarly(times.size()), dDensityLate(times.size()),
      dDensityEarly(times.size());
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  const double perTime = ev.toMatTime_;
  std::vector<double> wEarly(times.size(), 0.0), wLate(times.size(), 0.0);
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    wEarly[n] = 0.5 * (times[n] - times[n - 1]);
  }
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    wLate[n] = 0.5 * (times[n + 1] - times[n]);
  }

  auto dts = std::vector<double>{};
  for (auto m = std::size_t{0}; m + 1 < times.size(); ++m) {
    dts.push_back((times[m + 1] - times[m]) * ev.toMatTime_);
  }
  auto cursor = TrajectoryCursor(
      op, trajectory, dts, scheme,
      [&ev](const std::vector<Complex>& c) {
        return ev.Columns(c, ev.toMatDensity_);
      },
      [&ev](const std::vector<Complex>& c) {
        return ev.Columns(c, ev.toMatPotential_);
      });
  const auto meshSize = op.pointItau_[0].size();
  auto itauKernel = std::vector<std::vector<double>>(
      K, std::vector<double>(meshSize, 0.0));
  auto dItauKernel = itauKernel;
  const auto& matMesh = op.BaseMaterial().Mesh();
  const double aMat =
      matMesh.Radius(matMesh.ElementCount() - 1, matMesh.NGll() - 1);
  const double pairing =
      r2 * ev.toNdLength_ /
      (aMat * aMat * ev.toMatDensity_ * op.BaseMaterial().SurfaceGravity());

  // Both pairings at a node: the branch-weighted adjoint state over itau
  // against the forward d - m, and its derivative, with the derivative
  // state, v's tangent-linear trajectory and the perturbation of itau
  // each in their place.
  auto accumulateItau = [&](std::size_t node, double weight) {
    if (weight == 0.0) return;
    const auto& fwd = cursor.State(node);
    const auto& tl = *tangent[node].internalState;
    for (const auto& [l, mdual] : vs.state.m) {
      const auto& deg = op.Deg(l);
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      const auto& dFwd = fwd.strain.at(l);
      const auto& mFwd = fwd.m.at(l);
      const auto& dTl = tl.strain.at(l);
      const auto& mTl = tl.m.at(l);
      const auto weighted = deg.ApplyWeights(mdual, ncols);
      const auto dweighted = deg.ApplyWeights(ds.state.m.at(l), ncols);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const double itau =
                deg.itau[k][static_cast<std::size_t>(e) * deg.np + q];
            if (itau <= 0.0) continue;
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const double ratio = ditauPoints[k][full] / itau;
            const std::size_t gridNode =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            auto sum = 0.0, dsum = 0.0;
            for (auto s = 0; s < 3; ++s) {
              const auto at = (k * 3 + s) * block + gridNode;
              const auto sn = s * block + gridNode;
              for (auto c = 0; c < ncols; ++c) {
                const auto m = c <= l ? c : c - (l + 1);
                const auto cw = m == 0 ? (c == 0 ? 1.0 : 0.0) : 2.0;
                const auto dm = dFwd[sn + c] - mFwd[at + c];
                const auto vdm = dTl[sn + c] - mTl[at + c];
                sum += cw * weighted[at + c] * dm;
                dsum += cw * (dweighted[at + c] * dm + weighted[at + c] * vdm);
              }
            }
            itauKernel[k][full] += pairing * weight * sum / itau;
            dItauKernel[k][full] +=
                pairing * weight * (dsum - ratio * sum) / itau;
          }
        }
      }
    }
  };

  auto observe = [&](std::size_t node) {
    const auto found = measurements.find(node);
    if (found == measurements.end()) return;
    const auto j = jump(found->second, node);
    lumps.emplace(node, j.xiA);
    const auto strains = jumpStrains(j);
    applyJump(vs.state, strains, false);
    applyJump(ds.state, strains, true);
    auto early = smoothEquilibrium(vs, node);
    auto dearly = smoothEquilibrium(ds, node);
    if (node > 0) {
      densityEarly[node] = std::move(early);
      dDensityEarly[node] = std::move(dearly);
    }
  };

  observe(N);
  accumulateItau(N, wEarly[N] * perTime);
  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = dts[n - 1];
    // The adjoint state before the step, with its strains, is the start
    // of the derivative state's drive.
    op.PrimeStrain(vs.state, ev.Columns(vs.sigma, ev.toMatDensity_),
                   ev.Columns(vs.psi, ev.toMatPotential_));
    const ViscoelasticState before = vs.state;
    densityLate[n - 1] = reversedStep(vs, dt, n - 1, nullptr);
    const auto drive = TangentLinearSolver::FirstOrderDrive(
        op, AdjointMethod::Continuous, scheme, dt, before, vs.state,
        ditauPoints);
    dDensityLate[n - 1] = reversedStep(ds, dt, n - 1, &drive);
    accumulateItau(n - 1, wLate[n - 1] * perTime);
    observe(n - 1);
    accumulateItau(n - 1, wEarly[n - 1] * perTime);
  }

  // Node duals from the densities and the lumps, as the first-order
  // sweep assembles them; the derivative duals carry no lumps.
  auto assemble = [&](const std::vector<std::vector<double>>& late,
                      const std::vector<std::vector<double>>& early,
                      const std::map<std::size_t, std::vector<double>>& lumpsOf,
                      std::vector<std::vector<double>>& duals,
                      std::vector<std::vector<double>>& density) {
    duals.assign(times.size(), {});
    density.assign(times.size(), {});
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      duals[n].assign(fieldSize, 0.0);
      density[n].assign(fieldSize, 0.0);
      const auto* lateP = late[n].empty() ? nullptr : &late[n];
      const auto* earlyP = early[n].empty() ? lateP : &early[n];
      const auto lump = lumpsOf.find(n);
      const auto totalW = wEarly[n] + wLate[n];
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        const double lateV = lateP == nullptr ? 0.0 : (*lateP)[i] * perTime;
        const double earlyV = earlyP == nullptr ? 0.0 : (*earlyP)[i] * perTime;
        const double smooth = wEarly[n] * earlyV + wLate[n] * lateV;
        duals[n][i] = smooth + (lump == lumpsOf.end() ? 0.0 : lump->second[i]);
        density[n][i] = smooth / totalW;
      }
    }
  };
  std::vector<std::vector<double>> loadDuals, densityLoad, dloadDuals,
      dDensityLoad;
  assemble(densityLate, densityEarly, lumps, loadDuals, densityLoad);
  assemble(dDensityLate, dDensityEarly, {}, dloadDuals, dDensityLoad);
  auto kernels = AssembleViews(times, std::move(loadDuals),
                               std::move(densityLoad), std::move(lumps),
                               oceanFunctions);
  kernels.itauKernel = op.ItauKernelsAtNodes(std::move(itauKernel));
  KernelDerivatives derivatives;
  DerivativeViews(derivatives, C, rhoI, rhoW, times, std::move(dloadDuals),
                  oceanFunctions);
  derivatives.itauKernel = op.ItauKernelsAtNodes(std::move(dItauKernel));
  return {std::move(kernels), std::move(derivatives)};
}

}  // namespace sl3dpp
