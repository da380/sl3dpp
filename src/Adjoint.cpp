#include "sl3dpp/Adjoint.hpp"

#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/detail/InteriorSources.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/AdjointInternals.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

using detail::Complex;
using detail::TrajectoryCursor;
using detail::CoefficientDualFromColumnDuals;
using detail::ColumnDualsOf;
using detail::NodeWeights;
using detail::CheckGrids;

ForwardTrajectory ForwardTrajectory::FromSnapshots(
    std::span<const Evolution::Snapshot> snapshots) {
  ForwardTrajectory out;
  for (const auto& snapshot : snapshots) {
    out.states.push_back(snapshot.internalState);
    out.loads.push_back(snapshot.loadCoefficients);
    out.tides.push_back(snapshot.tideCoefficients);
    if (snapshot.toroidalState) out.toroidal.push_back(*snapshot.toroidalState);
    if (snapshot.sourceForces) out.sourceForces.push_back(*snapshot.sourceForces);
  }
  if (!out.toroidal.empty() && (out.toroidal.size() != snapshots.size() ||
                                out.sourceForces.size() != snapshots.size())) {
    throw std::invalid_argument(
        "a run under interior sources is recorded at every node");
  }
  return out;
}

std::vector<std::vector<double>> Kernels::Viscosity(
    const Rheology& rheology) const {
  if (itauKernel.empty()) {
    throw std::invalid_argument(
        "no itau kernels: run the adjoint with a forward trajectory");
  }
  auto out = std::vector<std::vector<double>>{};
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    const auto& branch = rheology.BranchAt(k);
    auto kernel = std::vector<double>(itauKernel[k].size(), 0.0);
    for (auto i = std::size_t{0}; i < kernel.size(); ++i) {
      if (branch.mu[i] > 0.0) {
        kernel[i] =
            -(branch.itau[i] * branch.itau[i]) / branch.mu[i] * itauKernel[k][i];
      }
    }
    out.push_back(std::move(kernel));
  }
  return out;
}

std::vector<std::vector<double>> OceanFunctionsOf(
    std::span<const Evolution::Snapshot> snapshots) {
  auto out = std::vector<std::vector<double>>{};
  for (const auto& snapshot : snapshots) {
    if (snapshot.oceanFunction.empty()) {
      throw std::invalid_argument(
          "the snapshots carry no ocean functions: a migrating run records "
          "them");
    }
    out.push_back(snapshot.oceanFunction);
  }
  return out;
}

AdjointSolver::AdjointSolver(const Evolution& evolution,
                             Options options)
    : evolution_(&evolution), options_(options) {}

Kernels AdjointSolver::FirstOrderKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory* trajectory,
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
  const auto fieldSize = ev.solver_->Grid().FieldSize();
  for (const auto& [node, m] : measurements) {
    if (node >= times.size()) {
      throw std::invalid_argument(
          std::format("measurement node {} is off the time grid", node));
    }
    CheckGrids(m, fieldSize, node);
  }
  if (trajectory != nullptr) {
    if (trajectory->states.size() != times.size()) {
      throw std::invalid_argument(
          "the trajectory must give one entry per time node");
    }
    if (!trajectory->states.back().has_value()) {
      throw std::invalid_argument(
          "the trajectory must be recorded forward states (a run with "
          "record_stride)");
    }
    // The recorded loads regenerate the strains and fill the gaps.
    if (trajectory->loads.size() != times.size() ||
        trajectory->tides.size() != times.size()) {
      throw std::invalid_argument(
          "the trajectory needs the recorded load and centrifugal "
          "coefficients: build it from the recording run's snapshots");
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
    // The forward run's stability condition (see Evolution::Run).
    const auto kHyd = FluidTidalNumber(ev.op_->BaseMaterial(), 2);
    if (ev.solver_->TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "Evolution::Run");
    }
  }
  return options_.method == AdjointMethod::Discrete
             ? DiscreteSweep(times, measurements, trajectory, oceanFunctions)
             : ContinuousSweep(times, measurements, trajectory, oceanFunctions);
}

// -- the discrete transpose
// ---------------------------------------------------

Kernels AdjointSolver::DiscreteSweep(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto scheme = ev.options_.scheme;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto rotation = ev.options_.rotation;
  const auto& C = ev.state_.oceanFunction;
  // About a migrating-shoreline trajectory the adjoint stays linear with
  // the node's ocean function as its coefficient (Yu et al. 2025, §3.3);
  // frozen, every node has the background's.
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return oceanFunctions.empty() ? C : oceanFunctions[node];
  };
  auto overrideAt = [&](std::size_t node) {
    return oceanFunctions.empty() ? std::vector<double>{}
                                  : oceanFunctions[node];
  };
  const auto rhoW = solver.WaterDensity();
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto lmax = op.LMax();
  const auto K = op.GetRheology().BranchCount();
  const auto N = times.size() - 1;

  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;

  // The flavoured solvers, as the forward run builds them (keyed by the
  // material dt; 0 is the unrelaxed table).
  auto flavours = std::map<double, GRDSolver>{};
  auto solverFor = [&](double dt) -> const GRDSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(dt, GRDSolver(
                                   grid,
                                   dt == 0.0 ? op.ElasticLoveNumbers()
                                             : op.EffectiveLoveNumbers(
                                                   ev.options_.scheme, dt),
                                   solver.Parameters()))
                  .first;
    }
    return found->second;
  };

  // The dual state: m-dagger per branch and d-dagger, the layouts of
  // ViscoelasticState over every degree's 2 (l + 1) columns.
  std::map<int, std::vector<double>> mdual, ddual;
  for (auto l = 0; l <= lmax; ++l) {
    const auto& deg = op.Deg(l);
    const auto ncols = 2 * (l + 1);
    mdual[l].assign(K * 3 * deg.NodalSize(ncols), 0.0);
    ddual[l].assign(3 * deg.NodalSize(ncols), 0.0);
  }

  // The itau kernels pair the sweep's duals with the recorded forward
  // trajectory through the scheme-weight derivatives — the exact
  // discrete sensitivity.
  auto cursor = std::optional<TrajectoryCursor>{};
  auto itauKernel = std::vector<std::vector<double>>{};
  if (trajectory != nullptr) {
    auto dts = std::vector<double>{};
    for (auto m = std::size_t{0}; m + 1 < times.size(); ++m) {
      dts.push_back((times[m + 1] - times[m]) * ev.toMatTime_);
    }
    cursor.emplace(
        op, *trajectory, std::move(dts), scheme,
        [&ev](const std::vector<Complex>& c) {
          return ev.Columns(c, ev.toMatDensity_);
        },
        [&ev](const std::vector<Complex>& c) {
          return ev.Columns(c, ev.toMatPotential_);
        });
    // Accumulated at the memory points, mapped to the nodes at the end.
    const auto meshSize = op.pointItau_[0].size();
    itauKernel.assign(K, std::vector<double>(meshSize, 0.0));
  }

  // The material kernels: the dual solution of each step's effective
  // solve paired with the material derivative of the step's force — the
  // transpose of the material tangent (docs/adjoints.md). The
  // branch fractions f_k = mu_k / mu_U carry the branch moduli.
  const bool material = options_.material_kernels;
  auto fractions = std::vector<std::vector<double>>{};
  std::vector<double> kappaKernel, muKernel;
  auto x0 = std::map<int, std::vector<double>>{};  // the base start solution
  if (material) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the material kernels need the recorded forward trajectory");
    }
    // at the memory points, where the moduli's perturbations live
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto f = op.BranchFractionAtPoints(k);
      fractions.emplace_back(f.begin(), f.end());
    }
    const auto pointCount = op.UnrelaxedShearAtPoints().size();
    kappaKernel.assign(pointCount, 0.0);
    muKernel.assign(pointCount, 0.0);
  }
  // The pairing of a dual solution with a base one at a degree's points:
  // -(dK/d kappa, dK/d mu) of the step's system (the bulk product and
  // the weighted deviatoric products, the band assembly's integrands)
  // summed over the columns; the shear scaled by `factor` (the effective
  // modulus's variation) and, with the base decayed memory, the memory
  // force's variation added.
  auto materialTerms = [&](const ViscoelasticOperator::Degree& deg, int ncols,
                           std::span<const double> y,
                           std::span<const double> sy,
                           std::span<const double> x,
                           std::span<const double> factor,
                           const std::vector<double>* memory) {
    // -(dK) over every element of the system (DegreeOperator's gradient
    // at the memory points), the shear scaled by the effective factor ...
    auto [kappa, mu] = deg.system.PointGradient(op.Points(), y, x, ncols);
    for (auto i = std::size_t{0}; i < kappa.size(); ++i) {
      kappa[i] = -kappa[i];
      mu[i] = -factor[i] * mu[i];
    }
    // ... and the memory force's variation on the memory window.
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
              for (auto k = std::size_t{0}; k < K; ++k) {
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
  };

  // The engine-adjoint inputs at a node: the measurement's duals plus
  // those carried by the node's total-load and centrifugal coefficients.
  auto engineDuals = [&](const std::vector<Complex>& sigmaDual,
                         const std::vector<Complex>& psiDual,
                         std::size_t node) {
    GRDSolver::AdjointForcing duals;
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
    if (auto found = measurements.find(node); found != measurements.end()) {
      const auto& m = found->second;
      for (auto i = std::size_t{0}; i < m.seaLevelDual.size(); ++i) {
        duals.seaLevelDual[i] += m.seaLevelDual[i];
      }
      duals.displacementDual = m.displacementDual;
      duals.potentialDual = m.potentialDual;
      duals.tangentialPotentialDual = m.tangentialPotentialDual;
      duals.toroidalPotentialDual = m.toroidalPotentialDual;
      for (auto j = 0; j < 3; ++j) gOmega[j] += m.angularVelocityDual[j];
    }
    duals.angularVelocityDual = gOmega;
    return std::pair(std::move(duals), std::move(synth));
  };

  // The moment kernels: the strain at the given points of each step's
  // dual solution, spheroidal (the transposed effective solve's) and
  // toroidal (the toroidal systems' own transposed recurrence, forced by
  // the dual of the surface W).
  const bool moments = !options_.moment_points.empty();
  // About a source-driven trajectory the toroidal systems carry memory,
  // and the kernel in the relaxation times has their part.
  const bool sourcedBase = trajectory != nullptr && !trajectory->toroidal.empty();
  const bool twisting = moments || sourcedBase;
  auto toroidal = std::optional<ToroidalViscoelastic>{};
  auto tdual = ToroidalViscoelastic::Dual{};
  auto momentKernels =
      std::vector<std::vector<SphericalTensor>>(moments ? times.size() : 0);
  if (twisting) {
    toroidal.emplace(op);
    tdual = toroidal->ZeroDual();
  }
  auto toroidalSurfaceDual = [&](const GRDSolver::AdjointResponse& adj) {
    auto out = ToroidalViscoelastic::Fields{};
    if (adj.toroidalResponseDual.empty()) return out;
    for (auto l = 1; l <= lmax; ++l) {
      out.emplace(l, ColumnDualsOf(adj.toroidalResponseDual, l, ev.toNdLength_));
    }
    return out;
  };
  auto momentKernelsOf = [&](const std::map<int, std::vector<double>>& y,
                             const ToroidalViscoelastic::Fields& yT) {
    return detail::MomentKernelsOfDuals(op, *toroidal, solver.SolverScales(),
                                        ev.toNdLength_, options_.moment_points,
                                        y, yT);
  };

  auto directLoadKernels = std::vector<std::vector<double>>(times.size());
  // The tide kernels: a node's prescribed tide enters its sea level solve
  // and, beside the centrifugal potential, the radial steps' tide channel.
  auto tideKernels = std::vector<std::vector<Complex>>(times.size());
  auto tideKernelOf = [&](const GRDSolver::AdjointResponse& adj,
                          const std::vector<Complex>& psiDual) {
    auto kernel = adj.tideDual;
    for (auto i = std::size_t{0}; i < kernel.size(); ++i) kernel[i] += psiDual[i];
    kernel[SurfaceGrid::Index(0, 0)] = 0.0;
    return kernel;
  };
  // The dual of a step's memory surface values, on the surface rows of
  // the transposed solve.
  auto addSurfaceDuals = [&](const ViscoelasticOperator::Degree& deg, int l,
                             int ncols, const GRDSolver::AdjointResponse& adj,
                             std::vector<double>& rhs) {
    const auto suCols =
        ColumnDualsOf(adj.displacementResponseDual, l, ev.toNdLength_);
    const auto sphiCols =
        ColumnDualsOf(adj.potentialResponseDual, l, ev.toNdPotential_);
    const auto svCols =
        ColumnDualsOf(adj.tangentialResponseDual, l, ev.toNdLength_);
    const auto su = deg.system.SurfaceDof(Component::U);
    const auto sphi = deg.system.SurfaceDof(Component::Phi);
    const auto sv = deg.system.SurfaceDof(Component::V);
    for (auto c = 0; c < ncols; ++c) {
      const auto at = static_cast<std::size_t>(c) * deg.ndof;
      if (su >= 0) rhs[at + su] += suCols[c];
      if (sphi >= 0) rhs[at + sphi] += sphiCols[c];
      if (sv >= 0) rhs[at + sv] += svCols[c];
    }
  };

  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = (times[n] - times[n - 1]) * ev.toMatTime_;
    const auto& weights = op.SchemeWeightsOf(scheme, dt);
    auto& flavour = op.FlavourOf(scheme, dt);

    // The duals of the end-of-step solve, through the strain transpose,
    // and their contraction with the unit responses: the duals of the
    // step's load and tide coefficients. One task per degree, largest
    // first, the workspaces, systems and unit responses resolved serially
    // first (the caches fill lazily); each writes only its own degree's.
    auto degs = std::vector<const ViscoelasticOperator::Degree*>(lmax + 1);
    auto unitsOf = std::vector<const std::vector<double>*>(lmax + 1);
    auto systems = std::vector<const DegreeOperator*>(lmax + 1);
    auto mds = std::vector<std::vector<double>*>(lmax + 1);
    auto dds = std::vector<std::vector<double>*>(lmax + 1);
    for (auto l = 0; l <= lmax; ++l) {
      degs[l] = &op.Deg(l);
      unitsOf[l] = &op.UnitResponses(&flavour, l);
      systems[l] = &op.EffectiveSystem(flavour, l);
      mds[l] = &mdual.at(l);
      dds[l] = &ddual.at(l);
    }
    auto xd = std::vector<std::vector<double>>(lmax + 1);
    auto sigByDegree = std::vector<std::vector<double>>(lmax + 1);
    auto psiByDegree = std::vector<std::vector<double>>(lmax + 1);
    detail::ForEachTask(op.options_.parallel, lmax + 1, [&](int i) {
      const auto l = lmax - i;
      const auto& deg = *degs[l];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      auto summed = *dds[l];
      const auto& md = *mds[l];
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& w = weights[k];
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const double b = implicit ? 1.0 - w.e[full] : w.beta[full];
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const auto at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                summed[s * block + node + c] += b * md[at + c];
              }
            }
          }
        }
      }
      auto x = deg.DualFromScalars(summed, ncols);
      const auto& units = *unitsOf[l];
      std::vector<double> sc(ncols, 0.0), pc(ncols, 0.0);
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
      sigByDegree[l] = std::move(sc);
      psiByDegree[l] = std::move(pc);
      xd[l] = std::move(x);
    });
    std::map<int, std::vector<double>> sigCols, psiCols;
    for (auto l = 0; l <= lmax; ++l) {
      sigCols.emplace(l, std::move(sigByDegree[l]));
      psiCols.emplace(l, std::move(psiByDegree[l]));
    }
    const auto sigmaDual =
        CoefficientDualFromColumnDuals(grid, sigCols, ev.toMatDensity_);
    const auto psiDual =
        CoefficientDualFromColumnDuals(grid, psiCols, ev.toMatPotential_);

    auto [duals, synth] = engineDuals(sigmaDual, psiDual, n);
    const auto adj = solverFor(dt).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          std::format("the adjoint sea level solve at node {} did not "
                      "converge (error {})",
                      n, adj.error));
    }
    directLoadKernels[n].assign(fieldSize, 0.0);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      directLoadKernels[n][i] = adj.directLoadDual[i] + synth[i] / r2;
    }
    tideKernels[n] = tideKernelOf(adj, psiDual);

    // The forward states this step's itau sensitivity pairs with.
    const ViscoelasticState* fwdPrev = nullptr;
    auto fwdN = std::optional<ViscoelasticState>{};
    const std::vector<ViscoelasticOperator::BranchWeights>* dweights = nullptr;
    if (cursor.has_value()) {
      fwdN = cursor->State(n);  // copied before the cursor moves on
      fwdPrev = &cursor->State(n - 1);
      dweights = &op.SchemeWeightDerivativesOf(scheme, dt);
    }
    // The base full solutions the material pairing needs: the unrelaxed
    // solve under the node's memory and recorded loads (a step's
    // solution satisfies it), and the effective modulus's factor
    // 1 - sum_k f_k (1 - w_k) at the nodes.
    auto xn = std::map<int, std::vector<double>>{};
    auto factor = std::vector<double>{};
    if (material) {
      // (about a source-driven run, under the sources' forces too)
      auto withSources = [&](std::map<int, std::vector<double>> x,
                             std::size_t node) {
        if (!sourcedBase) return x;
        for (const auto& [l, f] :
             trajectory->sourceForces[node].spheroidal) {
          const auto add = op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
          auto& to = x.at(l);
          for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += add[i];
        }
        return x;
      };
      xn = withSources(
          op.SolveElastic(*fwdN,
                          ev.Columns(trajectory->loads[n], ev.toMatDensity_),
                          ev.Columns(trajectory->tides[n], ev.toMatPotential_)),
          n);
      if (n == 1) {
        x0 = withSources(
            op.SolveElastic(*fwdPrev,
                            ev.Columns(trajectory->loads[0], ev.toMatDensity_),
                            ev.Columns(trajectory->tides[0], ev.toMatPotential_)),
            0);
      }
      factor.assign(fractions[0].size(), 1.0);
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& w = weights[k];
        for (auto i = std::size_t{0}; i < factor.size(); ++i) {
          const double b = implicit ? 1.0 - w.e[i] : w.beta[i];
          factor[i] -= fractions[k][i] * b;
        }
      }
    }

    // The transposed radial update: one symmetric effective solve per
    // degree, then the weight recursion closing m-dagger and d-dagger;
    // one task per degree again. The itau kernel's terms meet at the same
    // points from every degree, so each degree sums its own, added to
    // the kernel in degree order after the loop.
    auto kernelTerms =
        std::vector<std::vector<double>>(dweights != nullptr ? lmax + 1 : 0);
    auto materialByDegree =
        std::vector<std::pair<std::vector<double>, std::vector<double>>>(
            material ? lmax + 1 : 0);
    auto dualSolutions = std::vector<std::vector<double>>(moments ? lmax + 1 : 0);
    detail::ForEachTask(op.options_.parallel, lmax + 1, [&](int i) {
      const auto l = lmax - i;
      const auto& deg = *degs[l];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      auto rhs = std::move(xd[l]);
      addSurfaceDuals(deg, l, ncols, adj, rhs);
      if (l == 0) {
        const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
        for (auto d = 0; d < deg.ndof; ++d) rhs[d] += deg.momentRow[d] * jd;
      }
      const auto y = systems[l]->Solve(rhs, ncols);
      if (moments) dualSolutions[l] = y;
      const auto sy = deg.Strain(y, ncols);
      const auto wsy = deg.ApplyWeights(sy, ncols);
      if (material) {
        // The base decayed memory m~ = e m + alpha d_0 (e m for backward
        // Euler) of the step, for the memory force's variation.
        const auto& mPrev = fwdPrev->m.at(l);
        auto decayed = std::vector<double>(mPrev.size());
        const auto& d0 = fwdPrev->strain.at(l);
        for (auto k = std::size_t{0}; k < K; ++k) {
          const auto& w = weights[k];
          for (auto s = 0; s < 3; ++s) {
            for (auto e = 0; e < deg.ne; ++e) {
              for (auto q = 0; q < deg.np; ++q) {
                const std::size_t full =
                    static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
                const std::size_t node =
                    (static_cast<std::size_t>(e) * deg.np + q) * ncols;
                const auto at = (k * 3 + s) * block + node;
                for (auto c = 0; c < ncols; ++c) {
                  decayed[at + c] =
                      w.e[full] * mPrev[at + c] +
                      (implicit ? 0.0 : w.alpha[full] * d0[s * block + node + c]);
                }
              }
            }
          }
        }
        materialByDegree[l] =
            materialTerms(deg, ncols, y, sy, xn.at(l), factor, &decayed);
      }
      // The forward pairings of this degree's itau sensitivity.
      const std::vector<double>* mPrev = nullptr;
      const std::vector<double>* d0 = nullptr;
      const std::vector<double>* d1 = nullptr;
      auto wd1 = std::vector<double>{};
      if (cursor.has_value()) {
        mPrev = &fwdPrev->m.at(l);
        d0 = &fwdPrev->strain.at(l);
        d1 = &fwdN->strain.at(l);
        wd1 = deg.ApplyWeights(*d1, ncols);
      }
      auto& md = *mds[l];
      auto& dd = *dds[l];
      std::ranges::fill(dd, 0.0);
      auto* terms = dweights != nullptr ? &kernelTerms[l] : nullptr;
      if (terms != nullptr) {
        terms->assign(K * static_cast<std::size_t>(deg.ne) * deg.np, 0.0);
      }
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& w = weights[k];
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const auto at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                const auto mdOld = md[at + c];
                const auto mtilde = mdOld + wsy[at + c];
                if (dweights != nullptr) {
                  // The derivative of the step in h = dt itau pairs the
                  // duals with the forward trajectory; dh = dt d(itau).
                  const auto& dw = (*dweights)[k];
                  const auto sn = s * block + node + c;
                  const double g =
                      implicit
                          ? (mtilde * (*mPrev)[at + c] - mdOld * (*d1)[sn] -
                             wd1[at + c] * sy[sn]) *
                                dw.e[full]
                          : mtilde * (dw.e[full] * (*mPrev)[at + c] +
                                      dw.alpha[full] * (*d0)[sn]) +
                                (mdOld * (*d1)[sn] + wd1[at + c] * sy[sn]) *
                                    dw.beta[full];
                  (*terms)[(k * deg.ne + e) * deg.np + q] += dt * g;
                }
                // backward Euler stores its weight 1/(1 + h) in e
                md[at + c] = w.e[full] * mtilde;
                if (!implicit) {
                  dd[s * block + node + c] += w.alpha[full] * mtilde;
                }
              }
            }
          }
        }
      }
    });
    for (auto l = 0; l < static_cast<int>(kernelTerms.size()); ++l) {
      const auto& deg = *degs[l];
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            itauKernel[k][static_cast<std::size_t>(deg.e0 + e) * deg.np + q] +=
                kernelTerms[l][(k * deg.ne + e) * deg.np + q];
          }
        }
      }
    }
    for (const auto& [kappa, mu] : materialByDegree) {
      for (auto i = std::size_t{0}; i < kappaKernel.size(); ++i) {
        kappaKernel[i] += kappa[i];
        muKernel[i] += mu[i];
      }
    }
    if (twisting) {
      auto pairing = ToroidalViscoelastic::ItauPairing{};
      if (sourcedBase) {
        pairing = {.prev = &trajectory->toroidal[n - 1],
                   .next = &trajectory->toroidal[n],
                   .kernel = &itauKernel};
      }
      auto materialPairing = ToroidalViscoelastic::MaterialPairing{};
      if (sourcedBase && material) {
        materialPairing = {.prev = &trajectory->toroidal[n - 1],
                           .force = &trajectory->sourceForces[n].toroidal,
                           .mu = &muKernel};
      }
      const auto yT = toroidal->StepTransposed(
          tdual, dt, scheme, toroidalSurfaceDual(adj),
          sourcedBase ? &pairing : nullptr,
          sourcedBase && material ? &materialPairing : nullptr);
      if (moments) {
        auto y = std::map<int, std::vector<double>>{};
        for (auto l = 0; l <= lmax; ++l) {
          y.emplace(l, std::move(dualSolutions[l]));
        }
        momentKernels[n] = momentKernelsOf(y, yT);
      }
    }
  }

  // The start: the transposed instantaneous elastic solve at times[0].
  {
    std::map<int, std::vector<double>> sigCols, psiCols;
    std::vector<Complex> sigmaDual, psiDual;
    for (auto l = 0; l <= lmax; ++l) {
      const auto& deg = op.Deg(l);
      const auto ncols = 2 * (l + 1);
      const auto x = deg.DualFromScalars(ddual[l], ncols);
      const auto& units = op.UnitResponses(nullptr, l);
      std::vector<double> sc(ncols, 0.0), pc(ncols, 0.0);
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
      sigCols.emplace(l, std::move(sc));
      psiCols.emplace(l, std::move(pc));
    }
    sigmaDual = CoefficientDualFromColumnDuals(grid, sigCols, ev.toMatDensity_);
    psiDual = CoefficientDualFromColumnDuals(grid, psiCols, ev.toMatPotential_);
    auto [duals, synth] = engineDuals(sigmaDual, psiDual, 0);
    const auto adj = solverFor(0.0).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          "the adjoint elastic sea level solve did not converge");
    }
    directLoadKernels[0].assign(fieldSize, 0.0);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      directLoadKernels[0][i] = adj.directLoadDual[i] + synth[i] / r2;
    }
    tideKernels[0] = tideKernelOf(adj, psiDual);
    if (moments) {
      // the start's dual solutions: the transposed unrelaxed solves
      auto y = std::map<int, std::vector<double>>{};
      for (auto l = 0; l <= lmax; ++l) {
        const auto& deg = op.Deg(l);
        const auto ncols = 2 * (l + 1);
        auto rhs = deg.DualFromScalars(ddual[l], ncols);
        addSurfaceDuals(deg, l, ncols, adj, rhs);
        if (l == 0) {
          const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
          for (auto d = 0; d < deg.ndof; ++d) rhs[d] += deg.momentRow[d] * jd;
        }
        y.emplace(l, deg.system.Solve(rhs, ncols));
      }
      momentKernels[0] = momentKernelsOf(
          y, toroidal->StartTransposed(tdual, toroidalSurfaceDual(adj)));
    }
    if (sourcedBase && material) {
      // the toroidal start's part of the shear kernel
      const auto base = toroidal->SolveElastic(
          toroidal->ZeroState(), trajectory->sourceForces[0].toroidal);
      (void)toroidal->StartTransposed(tdual, toroidalSurfaceDual(adj), &base,
                                      &muKernel);
    }
    if (material) {
      // The start's dual solution: the transposed unrelaxed solve under
      // the start duals and the start's response duals, paired with the
      // base start solution (every weight one, no memory).
      const auto ones = std::vector<double>(fractions[0].size(), 1.0);
      for (auto l = 0; l <= lmax; ++l) {
        const auto& deg = op.Deg(l);
        const auto ncols = 2 * (l + 1);
        auto rhs = deg.DualFromScalars(ddual[l], ncols);
        addSurfaceDuals(deg, l, ncols, adj, rhs);
        if (l == 0) {
          const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
          for (auto d = 0; d < deg.ndof; ++d) rhs[d] += deg.momentRow[d] * jd;
        }
        const auto y = deg.system.Solve(rhs, ncols);
        const auto sy = deg.Strain(y, ncols);
        const auto [kappa, mu] =
            materialTerms(deg, ncols, y, sy, x0.at(l), ones, nullptr);
        for (auto i = std::size_t{0}; i < kappaKernel.size(); ++i) {
          kappaKernel[i] += kappa[i];
          muKernel[i] += mu[i];
        }
      }
    }
  }

  // The delta amplitudes: the instantaneous (unrelaxed) fingerprint
  // adjoint of each measurement, independent of the time grid.
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  for (const auto& [node, m] : measurements) {
    GRDSolver::AdjointForcing duals;
    duals.rotation = rotation;
    duals.oceanFunctionOverride = overrideAt(node);
    duals.seaLevelDual = m.seaLevelDual;
    duals.displacementDual = m.displacementDual;
    duals.potentialDual = m.potentialDual;
    duals.tangentialPotentialDual = m.tangentialPotentialDual;
    duals.angularVelocityDual = m.angularVelocityDual;
    const auto adj = solverFor(0.0).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          "the instantaneous adjoint fingerprint did not converge");
    }
    lumps.emplace(node, adj.directLoadDual);
  }

  // The continuous part of the distributional view: the node duals less
  // the lumps, per unit time.
  const auto w = NodeWeights(times);
  auto densityLoad = std::vector<std::vector<double>>(times.size());
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    densityLoad[n].assign(fieldSize, 0.0);
    const auto lump = lumps.find(n);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double lumped =
          lump == lumps.end() ? 0.0 : lump->second[i];
      densityLoad[n][i] = (directLoadKernels[n][i] - lumped) / w[n];
    }
  }

  auto out = AssembleViews(times, std::move(directLoadKernels), std::move(densityLoad),
                           std::move(lumps), oceanFunctions);
  out.tideKernels = std::move(tideKernels);
  out.momentKernels = std::move(momentKernels);
  // the kernels as densities per unit volume: the weights divided out
  {
    const auto& mesh = op.BaseMaterial().Mesh();
    out.itauKernel = op.ItauKernelsAtNodes(std::move(itauKernel));
    for (auto& branch : out.itauKernel) {
      branch = detail::KernelSamples(mesh, std::move(branch));
    }
    if (material) {
      out.kappaKernel =
          detail::KernelSamples(op.Points(), mesh, std::move(kappaKernel));
      out.muKernel = detail::KernelSamples(op.Points(), mesh, std::move(muKernel));
    }
  }
  return out;
}

// -- the continuous adjoint, discretised
// --------------------------------------

Kernels AdjointSolver::ContinuousSweep(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto scheme = ev.options_.scheme;
  const auto rotation = ev.options_.rotation;
  const auto& C = ev.state_.oceanFunction;
  // About a migrating-shoreline trajectory the adjoint stays linear with
  // the node's ocean function as its coefficient (Yu et al. 2025, §3.3);
  // frozen, every node has the background's.
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return oceanFunctions.empty() ? C : oceanFunctions[node];
  };
  auto overrideAt = [&](std::size_t node) {
    return oceanFunctions.empty() ? std::vector<double>{}
                                  : oceanFunctions[node];
  };
  const auto rhoW = solver.WaterDensity();
  const auto g = solver.SurfaceGravity();
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto K = op.GetRheology().BranchCount();
  const auto N = times.size() - 1;

  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;

  auto flavours = std::map<double, GRDSolver>{};
  auto solverFor = [&](double dt) -> const GRDSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(dt, GRDSolver(
                                   grid,
                                   dt == 0.0 ? op.ElasticLoveNumbers()
                                             : op.EffectiveLoveNumbers(
                                                   ev.options_.scheme, dt),
                                   solver.Parameters()))
                  .first;
    }
    return found->second;
  };

  // The adjoint state, stepped by the forward machinery in reversed time
  // with no ice load; sigmaC/psiC are its SLE-closure load and
  // centrifugal coefficients.
  const bool material = options_.material_kernels;
  // The adjoint displacement itself is wanted for the material kernels,
  // the moment kernels (its strain at the sources) and the tide kernels
  // (its pairing with the tide's force).
  const bool moments = !options_.moment_points.empty();
  const bool tidal = options_.tide_kernels;
  const bool wantSolutions = material || moments || tidal;
  // About a source-driven trajectory the toroidal systems carry memory:
  // the toroidal adjoint is swept for the kernels' toroidal parts.
  const bool sourcedBase = trajectory != nullptr && !trajectory->toroidal.empty();
  const bool twisting = moments || sourcedBase;
  auto fractions = std::vector<std::vector<double>>{};
  std::vector<double> kappaKernel, muKernel;
  if (material) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the material kernels need the recorded forward trajectory");
    }
    // at the memory points, where the moduli's perturbations live
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto f = op.BranchFractionAtPoints(k);
      fractions.emplace_back(f.begin(), f.end());
    }
    const auto pointCount = op.UnrelaxedShearAtPoints().size();
    kappaKernel.assign(pointCount, 0.0);
    muKernel.assign(pointCount, 0.0);
  }

  auto vstate = op.ZeroState([](int l) { return 2 * (l + 1); });
  auto sigmaC = grid.ZeroCoefficients();
  auto psiC = grid.ZeroCoefficients();

  // Psi*: the adjoint of the centrifugal potential operator, a potential
  // dual (dS-paired density) to angular-velocity space.
  auto psiStar = [&](const std::vector<double>& potentialDual) {
    auto d = solver.CentrifugalDual(grid.Expand(potentialDual));
    for (auto& v : d) v *= r2;
    return d;
  };

  // The instantaneous adjoint fingerprint of a measurement, through the
  // forward channel mapping of the elastic adjoint (docs/adjoints.md):
  // its sea level (the delta amplitude), and the converged channel coefficients
  // that drive the radial jump.
  struct Jump {
    std::vector<double> xiA;
    std::vector<Complex> lam, psiA, zu, zphi, zv;  // z* empty when absent
  };
  auto jump = [&](const Measurement& m, std::size_t node) {
    GRDSolver::GeneralisedForcing f;
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
    // a measurement of V: its adjoint tangential traction
    std::vector<Complex> tractionV;
    if (!m.tangentialPotentialDual.empty()) {
      tractionV = detail::TractionPotentialOfTangentialDual(
          grid, m.tangentialPotentialDual);
      f.tangentialTraction = grid.Synthesise(tractionV);
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
    out.xiA = resp.relativeSeaLevelChange;
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
    out.zv = std::move(tractionV);
    return out;
  };

  // Yu et al. eq. (80): the strain of the instantaneous adjoint solution,
  // over the relaxation time, jumps the adjoint internal variables; the
  // split load channels carry displacement and potential measurements
  // into the radial field.
  std::map<int, std::vector<double>> lumpSolutions;  // the jump's, per degree
  auto applyJump = [&](const Jump& j) {
    const auto lamCols = ev.Columns(j.lam, ev.toMatDensity_);
    const auto psiCols = ev.Columns(j.psiA, ev.toMatPotential_);
    const auto zuCols =
        j.zu.empty() ? LoadCallable{} : ev.Columns(j.zu, ev.toMatDensity_);
    const auto zphiCols =
        j.zphi.empty() ? LoadCallable{} : ev.Columns(j.zphi, ev.toMatDensity_);
    const auto zvCols =
        j.zv.empty() ? LoadCallable{} : ev.Columns(j.zv, ev.toMatDensity_);
    // One task per degree, largest first; the workspaces, unit responses
    // and channel columns resolved serially first.
    struct Task {
      int l = 0;
      std::vector<double>* m = nullptr;
      const ViscoelasticOperator::Degree* deg = nullptr;
      const std::vector<double>* units = nullptr;
      const std::vector<double>* split = nullptr;
      std::optional<std::vector<double>> lam, psi, zu, zphi, zv;
    };
    auto tasks = std::vector<Task>{};
    for (auto& [l, m] : vstate.m) {
      auto task = Task{};
      task.l = l;
      task.m = &m;
      task.deg = &op.Deg(l);
      task.units = &op.UnitResponses(nullptr, l);
      task.lam = lamCols(l, 0.0);
      task.psi = psiCols(l, 0.0);
      if (zuCols || zphiCols || zvCols) {
        task.split = &op.SplitUnitResponses(nullptr, l);
        if (zuCols) task.zu = zuCols(l, 0.0);
        if (zphiCols) task.zphi = zphiCols(l, 0.0);
        if (zvCols) task.zv = zvCols(l, 0.0);
      }
      tasks.push_back(std::move(task));
    }
    const auto count = static_cast<int>(tasks.size());
    if (wantSolutions) {
      // the entries made before the parallel loop fills them
      for (const auto& task : tasks) lumpSolutions[task.l].clear();
    }
    detail::ForEachTask(op.options_.parallel, count, [&](int i) {
      const auto& task = tasks[count - 1 - i];
      const auto l = task.l;
      auto& m = *task.m;
      const auto& deg = *task.deg;
      const int ncols = 2 * (l + 1);
      const std::size_t block = deg.NodalSize(ncols);
      auto x =
          std::vector<double>(static_cast<std::size_t>(deg.ndof) * ncols, 0.0);
      auto add = [&](const double* unit, const std::vector<double>& amps) {
        for (auto c = 0; c < ncols; ++c) {
          for (auto d = 0; d < deg.ndof; ++d) {
            x[static_cast<std::size_t>(c) * deg.ndof + d] += unit[d] * amps[c];
          }
        }
      };
      add(task.units->data(), *task.lam);
      add(task.units->data() + deg.ndof, *task.psi);
      if (task.zu) add(task.split->data(), *task.zu);
      if (task.zphi) add(task.split->data() + deg.ndof, *task.zphi);
      if (task.zv) add(task.split->data() + 2 * deg.ndof, *task.zv);
      const auto dTilde = deg.Strain(x, ncols);
      if (wantSolutions) lumpSolutions.at(l) = x;
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const double itau =
                  deg.itau[k][static_cast<std::size_t>(e) * deg.np + q];
              if (itau == 0.0) continue;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const std::size_t at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                m[at + c] += itau * dTilde[s * block + node + c];
              }
            }
          }
        }
      }
    });
    vstate.Invalidate();
  };

  // The memory-only adjoint equilibrium of the current state: its sea
  // level (the smooth kernel density, per material time) and the load
  // and centrifugal coefficients the next reversed step starts from.
  auto smoothEquilibrium = [&](std::size_t node) {
    const auto solutions = op.SolveElastic(vstate);
    GRDSolver::GeneralisedForcing f;
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
      sig[i] = rhoW * Cn[i] * resp.relativeSeaLevelChange[i];
    }
    sigmaC = grid.Expand(sig);
    psiC = rotation != RotationalFeedback::None
               ? solver.CentrifugalCoefficients(resp.angularVelocity)
               : grid.ZeroCoefficients();
    return resp.relativeSeaLevelChange;
  };

  // The density is discontinuous at each observation (a finite jump), so
  // both one-sided values are kept: 'late' towards times.back(), 'early'
  // towards times[0]; empty means zero. The node weight splits across
  // the discontinuity the same way.
  std::vector<std::vector<double>> densityLate(times.size()),
      densityEarly(times.size());
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  const double perTime = ev.toMatTime_;
  std::vector<double> wEarly(times.size(), 0.0), wLate(times.size(), 0.0);
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    wEarly[n] = 0.5 * (times[n] - times[n - 1]);
  }
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    wLate[n] = 0.5 * (times[n + 1] - times[n]);
  }

  // The itau kernels: the multiplier integral int W_k m-dagger / itau
  // (d - m) dt over the reversed nodes with the split trapezoid (the
  // adjoint state jumps at observations; the forward integrand is
  // continuous). The self-adjoint channel mapping poses the adjoint in
  // the g-weighted energy form of the sea level equation, so the radial
  // state carried here is the surface gravity times the true multiplier
  // of the internal-variable dynamics — hence the division, with g in
  // the material's units.
  auto cursor = std::optional<TrajectoryCursor>{};
  auto itauKernel = std::vector<std::vector<double>>{};
  if (trajectory != nullptr) {
    auto dts = std::vector<double>{};
    for (auto m = std::size_t{0}; m + 1 < times.size(); ++m) {
      dts.push_back((times[m + 1] - times[m]) * ev.toMatTime_);
    }
    cursor.emplace(
        op, *trajectory, std::move(dts), scheme,
        [&ev](const std::vector<Complex>& c) {
          return ev.Columns(c, ev.toMatDensity_);
        },
        [&ev](const std::vector<Complex>& c) {
          return ev.Columns(c, ev.toMatPotential_);
        });
    // Accumulated at the memory points, mapped to the nodes at the end.
    const auto meshSize = op.pointItau_[0].size();
    itauKernel.assign(K, std::vector<double>(meshSize, 0.0));
  }
  // The self-adjoint channel mapping poses the adjoint in the sea level
  // equation's energy form on the MODEL surface (r^2, solver units),
  // while the jump drives the unit-response machinery whose load
  // pairing lives on the MATERIAL surface (g a^2, material units); the
  // carried state is the true multiplier times the ratio of the two
  // pairings. The ratio is dimensionless but not 1 even in matched
  // frames (the solver's sea-floor radius differs from the surface) —
  // pyslfp carries the same ratio.
  const auto& matMesh = op.BaseMaterial().Mesh();
  const double aMat =
      matMesh.Radius(matMesh.ElementCount() - 1, matMesh.NGll() - 1);
  const double pairing =
      r2 * ev.toNdLength_ /
      (aMat * aMat * ev.toMatDensity_ * op.BaseMaterial().SurfaceGravity());
  // -- the material pairing (continuous; docs/adjoints.md)
  // The continuous adjoint displacement lambda at a node — the unrelaxed
  // solve under the adjoint memory and the closed adjoint load, the
  // adjoint problem having the forward form — paired with the forward
  // solution and memory at that node through the material gradient of
  // the degree form and the memory force's variation, under the time
  // quadrature of the itau kernel; at an observation the lumped
  // instantaneous adjoint solution pairs without a time weight. The
  // Hermitian column weights and the units constant are the itau
  // kernel's.
  // The pairing of adjoint degree solutions (natural columns) with the
  // forward state at a node, scaled by `weight` (the time weight times
  // the units constant), added to the kernels.
  auto pairMaterial = [&](const std::map<int, std::vector<double>>& lambda,
                          std::size_t node, double weight) {
    if (weight == 0.0) return;
    const auto& fwd = cursor->State(node);
    auto x =
        op.SolveElastic(fwd, ev.Columns(trajectory->loads[node], ev.toMatDensity_),
                        ev.Columns(trajectory->tides[node], ev.toMatPotential_));
    if (sourcedBase) {
      // the forward solution of a source-driven run, under its forces too
      for (const auto& [l, f] : trajectory->sourceForces[node].spheroidal) {
        const auto add = op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
        auto& to = x.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += add[i];
      }
    }
    auto tasks = std::vector<int>{};
    for (const auto& [l, y] : lambda) tasks.push_back(l);
    auto terms =
        std::vector<std::pair<std::vector<double>, std::vector<double>>>(
            tasks.size());
    const auto count = static_cast<int>(tasks.size());
    detail::ForEachTask(op.options_.parallel, count, [&](int i) {
      const auto l = tasks[static_cast<std::size_t>(count - 1 - i)];
      const auto& deg = op.Deg(l);
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      // the columns weighted for the Hermitian pairing: m = 0 real once,
      // its imaginary part not at all, m > 0 twice
      auto y = lambda.at(l);
      for (auto c = 0; c < ncols; ++c) {
        const auto m = c <= l ? c : c - (l + 1);
        const auto cw = m == 0 ? (c == 0 ? 1.0 : 0.0) : 2.0;
        for (auto d = 0; d < deg.ndof; ++d) {
          y[static_cast<std::size_t>(c) * deg.ndof + d] *= cw;
        }
      }
      auto [kappa, mu] =
          deg.system.PointGradient(op.Points(), y, x.at(l), ncols);
      for (auto j = std::size_t{0}; j < kappa.size(); ++j) {
        kappa[j] = -weight * kappa[j];
        mu[j] = -weight * mu[j];
      }
      const auto sy = deg.Strain(y, ncols);
      const auto& mFwd = fwd.m.at(l);
      const auto wc = detail::ScalarWeights(deg.k2);
      for (auto e = 0; e < deg.ne; ++e) {
        for (auto q = 0; q < deg.np; ++q) {
          const auto p = static_cast<std::size_t>(e) * deg.np + q;
          const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
          const auto gridNode = p * ncols;
          auto mem = 0.0;
          for (auto k = std::size_t{0}; k < K; ++k) {
            const auto fk = fractions[k][full];
            if (fk == 0.0) continue;
            for (auto s = 0; s < 3; ++s) {
              const auto sn = s * block + gridNode;
              const auto at = (k * 3 + s) * block + gridNode;
              for (auto c = 0; c < ncols; ++c) {
                mem += fk * wc[s] * sy[sn + c] * mFwd[at + c];
              }
            }
          }
          mu[full] += weight * deg.Wq[p] * mem;
        }
      }
      terms[static_cast<std::size_t>(count - 1 - i)] = {std::move(kappa),
                                                         std::move(mu)};
    });
    for (const auto& [kappa, mu] : terms) {
      for (auto j = std::size_t{0}; j < kappaKernel.size(); ++j) {
        kappaKernel[j] += kappa[j];
        muKernel[j] += mu[j];
      }
    }
  };
  // The regular adjoint displacement at a node, under the current
  // adjoint memory and closed load.
  // The toroidal adjoint, where the moment kernels want it: the toroidal
  // systems in reversed time, with no force between observations and a
  // jump at each by the toroidal traction of a measurement of W.
  auto toroidal = std::optional<ToroidalViscoelastic>{};
  auto tadj = ToroidalViscoelastic::State{};
  auto lumpTwist = ToroidalViscoelastic::Fields{};
  if (twisting) {
    toroidal.emplace(op);
    tadj = toroidal->ZeroState();
    toroidal->SetStrain(tadj, toroidal->SolveElastic(tadj, {}));
  }
  auto momentKernels = std::vector<std::vector<SphericalTensor>>(
      moments ? times.size() : 0,
      std::vector<SphericalTensor>(options_.moment_points.size()));
  auto tideKernels = std::vector<std::vector<Complex>>(
      tidal ? times.size() : 0, grid.ZeroCoefficients());
  // The moment kernels' part of an adjoint displacement: its strain at
  // the points, scaled; the natural columns are the coefficients of a
  // real field.
  auto addMoments = [&](const std::map<int, std::vector<double>>& lambda,
                        const ToroidalViscoelastic::Fields& twist,
                        std::size_t node, double weight) {
    auto asComplex = [](const std::map<int, std::vector<double>>& from) {
      auto out = std::map<int, std::vector<Complex>>{};
      for (const auto& [l, x] : from) {
        const auto ndof = x.size() / (2 * (static_cast<std::size_t>(l) + 1));
        auto c = std::vector<Complex>(ndof * (static_cast<std::size_t>(l) + 1));
        for (auto m = 0; m <= l; ++m) {
          for (auto d = std::size_t{0}; d < ndof; ++d) {
            c[static_cast<std::size_t>(m) * ndof + d] =
                Complex(x[static_cast<std::size_t>(m) * ndof + d],
                        x[static_cast<std::size_t>(l + 1 + m) * ndof + d]);
          }
        }
        out.emplace(l, std::move(c));
      }
      return out;
    };
    const auto fields = DegreeFields{.spheroidal = asComplex(lambda),
                                     .toroidal = asComplex(twist)};
    const auto scale =
        weight * detail::MomentToMaterial(op, solver.SolverScales());
    for (auto p = std::size_t{0}; p < options_.moment_points.size(); ++p) {
      auto point = options_.moment_points[p];
      point.radius /= ev.toNdLength_;
      const auto e = PointStrain(
          op.BaseMaterial(), op.LMax(),
          [&](int l) -> const DegreeOperator& { return op.UnrelaxedSystem(l); },
          [&](int l) -> const ToroidalOperator& { return toroidal->Unrelaxed(l); },
          fields, point);
      auto& to = momentKernels[node][p];
      to.rr += scale * e.rr;
      to.tt += scale * e.tt;
      to.pp += scale * e.pp;
      to.rt += scale * e.rt;
      to.rp += scale * e.rp;
      to.tp += scale * e.tp;
    }
  };
  // The tide kernels' part: the adjoint displacement paired with the
  // tide's force on each degree, and the adjoint load, on which the
  // tide's own potential acts through the sea level.
  auto addTide = [&](const std::map<int, std::vector<double>>& lambda,
                     const std::vector<Complex>& load, std::size_t node,
                     double weight) {
    auto& to = tideKernels[node];
    for (const auto& [l, x] : lambda) {
      if (l == 0) continue;
      const auto& deg = op.Deg(l);
      for (auto m = 0; m <= l; ++m) {
        auto re = 0.0, im = 0.0;
        for (auto d = 0; d < deg.ndof; ++d) {
          re += deg.tideVec[d] * x[static_cast<std::size_t>(m) * deg.ndof + d];
          im += deg.tideVec[d] *
                x[static_cast<std::size_t>(l + 1 + m) * deg.ndof + d];
        }
        const auto i = SurfaceGrid::Index(l, m);
        to[i] += weight * (pairing * ev.toMatPotential_ * Complex(re, im) -
                           r2 / g * load[i]);
      }
    }
  };
  // The toroidal share of the shear kernel about a source-driven run: an
  // adjoint toroidal displacement with the forward toroidal solution and
  // memory at the node.
  auto pairToroidalMaterial = [&](const ToroidalViscoelastic::Fields& twist,
                                  std::size_t node, double weight) {
    if (twist.empty()) return;
    const auto& forward = trajectory->toroidal[node];
    toroidal->AccumulateMaterial(
        twist, forward,
        toroidal->SolveElastic(forward, trajectory->sourceForces[node].toroidal),
        weight, muKernel);
  };
  // The regular adjoint displacement at a node, under the current
  // adjoint memory and closed load.
  auto accumulateMaterial = [&](std::size_t node, double weight) {
    if (!wantSolutions || weight == 0.0) return;
    const auto lambda =
        op.SolveElastic(vstate, ev.Columns(sigmaC, ev.toMatDensity_),
                        ev.Columns(psiC, ev.toMatPotential_));
    if (material) pairMaterial(lambda, node, pairing * weight);
    auto twist = ToroidalViscoelastic::Fields{};
    if (twisting && (moments || material)) {
      twist = toroidal->SolveElastic(tadj, {});
    }
    if (moments) addMoments(lambda, twist, node, pairing * weight);
    if (material && sourcedBase) pairToroidalMaterial(twist, node, pairing * weight);
    if (tidal) addTide(lambda, sigmaC, node, weight);
  };

  auto accumulateItau = [&](std::size_t node, double weight) {
    if (!cursor.has_value() || weight == 0.0) return;
    if (sourcedBase) {
      toroidal->AccumulateItau(tadj, trajectory->toroidal[node],
                               pairing * weight, itauKernel);
    }
    const auto& fwd = cursor->State(node);
    // One task per degree, largest first, each forming its own term per
    // point; the terms are added to the kernel in degree order after, so
    // the sum does not depend on the threading.
    auto tasks = std::vector<std::pair<int, const std::vector<double>*>>{};
    auto degs = std::vector<const ViscoelasticOperator::Degree*>{};
    for (const auto& [l, mdual] : vstate.m) {
      tasks.emplace_back(l, &mdual);
      degs.push_back(&op.Deg(l));
    }
    auto terms = std::vector<std::vector<double>>(tasks.size());
    const auto count = static_cast<int>(tasks.size());
    detail::ForEachTask(op.options_.parallel, count, [&](int i) {
      const auto j = static_cast<std::size_t>(count - 1 - i);
      const auto l = tasks[j].first;
      const auto& mdual = *tasks[j].second;
      const auto& deg = *degs[j];
      const auto ncols = 2 * (l + 1);
      const auto block = deg.NodalSize(ncols);
      const auto& dFwd = fwd.strain.at(l);
      const auto& mFwd = fwd.m.at(l);
      const auto weighted = deg.ApplyWeights(mdual, ncols);
      auto& term = terms[j];
      term.assign(K * static_cast<std::size_t>(deg.ne) * deg.np, 0.0);
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const double itau =
                deg.itau[k][static_cast<std::size_t>(e) * deg.np + q];
            if (itau <= 0.0) continue;
            const std::size_t gridNode =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            auto sum = 0.0;
            for (auto s = 0; s < 3; ++s) {
              const auto at = (k * 3 + s) * block + gridNode;
              const auto sn = s * block + gridNode;
              for (auto c = 0; c < ncols; ++c) {
                // Both factors are forward-convention m >= 0 columns, so
                // the spherical sum over orders weights m > 0 twice (the
                // discrete sweep's duals absorb this in the transposes;
                // here it is explicit). The m = 0 imaginary column is
                // identically zero.
                const auto m = c <= l ? c : c - (l + 1);
                const auto cw = m == 0 ? (c == 0 ? 1.0 : 0.0) : 2.0;
                sum += cw * weighted[at + c] * (dFwd[sn + c] - mFwd[at + c]);
              }
            }
            term[(k * deg.ne + e) * deg.np + q] = pairing * weight * sum / itau;
          }
        }
      }
    });
    for (auto j = std::size_t{0}; j < tasks.size(); ++j) {
      const auto& deg = *degs[j];
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto at = static_cast<std::size_t>(e) * deg.np + q;
            if (deg.itau[k][at] <= 0.0) continue;
            itauKernel[k][static_cast<std::size_t>(deg.e0 + e) * deg.np + q] +=
                terms[j][k * deg.ne * deg.np + at];
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
    applyJump(j);
    // the lumped adjoint displacement of the observation, no time weight
    if (material) pairMaterial(lumpSolutions, node, pairing);
    if (twisting) {
      // the toroidal jump: the unrelaxed toroidal response to the
      // adjoint traction of a measurement of W
      lumpTwist.clear();
      const auto& dual = found->second.toroidalPotentialDual;
      if (!dual.empty()) {
        const auto zw = detail::TractionPotentialOfTangentialDual(grid, dual);
        const auto columns = ev.Columns(zw, ev.toMatDensity_);
        auto forces = ToroidalViscoelastic::Fields{};
        for (auto l = 1; l <= op.LMax(); ++l) {
          const auto& system = toroidal->Unrelaxed(l);
          const auto unit = system.Tangential();
          const auto amps = *columns(l, 0.0);
          const auto ndof = static_cast<std::size_t>(system.DofCount());
          auto b = std::vector<double>(ndof * amps.size());
          for (auto c = std::size_t{0}; c < amps.size(); ++c) {
            for (auto d = std::size_t{0}; d < ndof; ++d) {
              b[c * ndof + d] = unit[d] * amps[c];
            }
          }
          forces.emplace(l, std::move(b));
        }
        lumpTwist = toroidal->SolveElastic(toroidal->ZeroState(), forces);
        toroidal->Jump(tadj, lumpTwist);
        toroidal->SetStrain(tadj, toroidal->SolveElastic(tadj, {}));
      }
      if (moments) addMoments(lumpSolutions, lumpTwist, node, pairing);
      if (material && sourcedBase) {
        pairToroidalMaterial(lumpTwist, node, pairing);
      }
    }
    if (tidal) addTide(lumpSolutions, j.lam, node, 1.0);
    auto early = smoothEquilibrium(node);
    if (node > 0) densityEarly[node] = std::move(early);
  };

  // The reversed sweep, nodes N down to 0; the state starts undisturbed,
  // so an unmeasured terminal node has zero density.
  observe(N);
  accumulateItau(N, wEarly[N] * perTime);
  accumulateMaterial(N, wEarly[N] * perTime);
  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = (times[n] - times[n - 1]) * ev.toMatTime_;
    const auto step =
        op.StartStep(vstate, dt, scheme, ev.Columns(sigmaC, ev.toMatDensity_),
                     ev.Columns(psiC, ev.toMatPotential_));
    GRDSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(n - 1);
    f.displacementResponse = ev.Coefficients(
        step.MemorySurface(Component::U), ev.toNdLength_);
    f.potentialResponse = ev.Coefficients(
        step.MemorySurface(Component::Phi), ev.toNdPotential_);
    f.inertiaMomentChange = step.InertiaMoment()[0] * ev.toNdMoment_;
    const auto resp = solverFor(dt).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(std::format(
          "the adjoint coupled solve at node {} did not converge", n - 1));
    }
    densityLate[n - 1] = resp.relativeSeaLevelChange;
    const auto& Cn = oceanAt(n - 1);
    auto sig = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      sig[i] = rhoW * Cn[i] * resp.relativeSeaLevelChange[i];
    }
    sigmaC = grid.Expand(sig);
    psiC = rotation != RotationalFeedback::None
               ? solver.CentrifugalCoefficients(resp.angularVelocity)
               : grid.ZeroCoefficients();
    auto [next, solutions] =
        step.Finish(ev.Columns(sigmaC, ev.toMatDensity_),
                    ev.Columns(psiC, ev.toMatPotential_));
    vstate = std::move(next);
    if (twisting) tadj = toroidal->Step(tadj, dt, scheme, {}).first;
    accumulateItau(n - 1, wLate[n - 1] * perTime);
    accumulateMaterial(n - 1, wLate[n - 1] * perTime);
    observe(n - 1);
    accumulateItau(n - 1, wEarly[n - 1] * perTime);
    accumulateMaterial(n - 1, wEarly[n - 1] * perTime);
  }

  // Node duals from the density and the lumps. The jumps divide by
  // relaxation times in the MATERIAL's units, so the smooth fields are
  // densities per material time; toMatTime_ converts them to per SLE
  // time, and at an observation the node weight splits across the
  // discontinuity: the half-interval towards the present pairs with the
  // late side, the half towards times[0] with the early side.
  std::vector<std::vector<double>> directLoadKernels(times.size()),
      densityLoad(times.size());
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    directLoadKernels[n].assign(fieldSize, 0.0);
    densityLoad[n].assign(fieldSize, 0.0);
    const auto* late = densityLate[n].empty() ? nullptr : &densityLate[n];
    const auto* early =
        densityEarly[n].empty() ? late : &densityEarly[n];
    const auto lump = lumps.find(n);
    const auto totalW = wEarly[n] + wLate[n];
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double lateV = late == nullptr ? 0.0 : (*late)[i] * perTime;
      const double earlyV = early == nullptr ? 0.0 : (*early)[i] * perTime;
      const double smooth = wEarly[n] * earlyV + wLate[n] * lateV;
      directLoadKernels[n][i] =
          smooth + (lump == lumps.end() ? 0.0 : lump->second[i]);
      densityLoad[n][i] = smooth / totalW;
    }
  }

  auto out = AssembleViews(times, std::move(directLoadKernels), std::move(densityLoad),
                           std::move(lumps), oceanFunctions);
  out.momentKernels = std::move(momentKernels);
  for (auto& kernel : tideKernels) kernel[SurfaceGrid::Index(0, 0)] = 0.0;
  out.tideKernels = std::move(tideKernels);
  // the kernels as densities per unit volume: the weights divided out
  {
    const auto& mesh = op.BaseMaterial().Mesh();
    out.itauKernel = op.ItauKernelsAtNodes(std::move(itauKernel));
    for (auto& branch : out.itauKernel) {
      branch = detail::KernelSamples(mesh, std::move(branch));
    }
    if (material) {
      out.kappaKernel =
          detail::KernelSamples(op.Points(), mesh, std::move(kappaKernel));
      out.muKernel = detail::KernelSamples(op.Points(), mesh, std::move(muKernel));
    }
  }
  return out;
}

// -- the shared views
// ----------------------------------------------------------

Kernels AdjointSolver::AssembleViews(
    std::span<const double> times, std::vector<std::vector<double>> directLoadKernels,
    std::vector<std::vector<double>> density,
    std::map<std::size_t, std::vector<double>> lumps,
    std::span<const std::vector<double>> oceanFunctions) const {
  return detail::AssembleIceViews(
      evolution_->state_.oceanFunction, evolution_->solver_->IceDensity(),
      times, std::move(directLoadKernels), std::move(density), std::move(lumps),
      oceanFunctions, evolution_->solver_->WaterDensity());
}

Kernels detail::AssembleIceViews(
    std::span<const double> C, double rhoI, std::span<const double> times,
    std::vector<std::vector<double>> directLoadKernels,
    std::vector<std::vector<double>> density,
    std::map<std::size_t, std::vector<double>> lumps,
    std::span<const std::vector<double>> oceanFunctions, double rhoW) {
  const auto fieldSize = directLoadKernels.front().size();

  // The grounding factor rho_i (1 - C_n) at node n: the node's own ocean
  // function about a migrating trajectory, the background's when frozen.
  auto ground = [&](const std::vector<double>& f, std::size_t node) {
    const auto Cn = oceanFunctions.empty()
                        ? C
                        : std::span<const double>(oceanFunctions[node]);
    auto out = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      out[i] = rhoI * (1.0 - Cn[i]) * f[i];
    }
    return out;
  };

  Kernels out;
  out.times.assign(times.begin(), times.end());
  out.baseline.assign(fieldSize, 0.0);
  for (const auto& ld : directLoadKernels) {
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      out.baseline[i] -= rhoI * (1.0 - C[i]) * ld[i];
    }
  }
  for (auto n = std::size_t{0}; n < directLoadKernels.size(); ++n) {
    out.thickness.push_back(ground(directLoadKernels[n], n));
  }
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    out.thickness[0][i] += out.baseline[i];
  }
  // The initial sea level enters each node's load as rho_w (C_n - C_0)
  // delta S_0, the discrete analogue of Yu et al.'s eq. (65): identically
  // zero under frozen shorelines (every C_n equals C).
  out.initialSeaLevel.assign(fieldSize, 0.0);
  if (!oceanFunctions.empty()) {
    for (auto n = std::size_t{0}; n < directLoadKernels.size(); ++n) {
      const auto& Cn = oceanFunctions[n];
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        out.initialSeaLevel[i] += rhoW * (Cn[i] - C[i]) * directLoadKernels[n][i];
      }
    }
  }
  for (const auto& [node, lump] : lumps) {
    out.instantaneous.emplace(node, ground(lump, node));
  }
  for (auto n = std::size_t{0}; n < density.size(); ++n) {
    out.density.push_back(ground(density[n], n));
  }
  // The quelled view: the cumulative tail of the thickness duals.
  auto tail = std::vector<double>(fieldSize, 0.0);
  out.rate.resize(times.size() - 1);
  for (std::size_t n = times.size() - 1; n >= 1; --n) {
    const auto grounded = ground(directLoadKernels[n], n);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) tail[i] += grounded[i];
    out.rate[n - 1] = tail;
  }
  out.directLoadKernels = std::move(directLoadKernels);
  return out;
}

}  // namespace sl3dpp
