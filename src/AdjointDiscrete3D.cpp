// The discrete 3D adjoint: the exact transpose of the coupled
// lateral-viscosity time-stepped map (docs/dev/adjoints.md). Under the
// energy pairing the force assembly is the transpose of the strain
// evaluation (Entry = Exit^T), the pointwise weight and the reference
// systems are self-adjoint, and the sea level equation has its exact
// adjoint, so the transposed pass is the forward pass with the force and
// strain legs swapped and the sea level solve replaced by its adjoint; the
// transposed within-step fixed point is solved by the same iteration. The
// duals of grid tensors live in the energy pairing (sum over nodes of
// Wq r^2 int dOmega of the null-basis contraction), the duals of degree
// columns in the Hermitian m >= 0 pairing (m > 0 weighted twice).

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/AdjointInternals.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
using Forces = std::map<int, std::vector<Complex>>;
using Tensor = GridCoupling::TensorField;

}  // namespace

Kernels3D AdjointSolver3D::DiscreteSweep(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto& coupling = op.Coupling();
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
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto K = op.GetRheology().BranchCount();
  const auto N = times.size() - 1;
  const auto values = coupling.GridValueCount();
  const auto nodeCount = static_cast<std::size_t>(coupling.NodeCount());

  if (times.size() < 2) {
    throw std::invalid_argument("times must hold at least two entries");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
  if (op.GetOptions().collectives.Distributed()) {
    throw std::invalid_argument(
        "the discrete 3D adjoint is not distributed: use Continuous");
  }
  const bool material = options_.material_kernels;
  if (material) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the material kernels need the recorded forward trajectory");
    }
    if (op.GetOptions().memory_points != MemoryPointSet::Gll) {
      throw std::invalid_argument(
          "the material kernels need the memory nodes at the GLL points");
    }
    if (rotation != RotationalFeedback::None) {
      throw std::invalid_argument(
          "the material kernels take no rotational feedback");
    }
  }
  for (const auto& [node, m] : measurements) {
    if (node >= times.size()) {
      throw std::invalid_argument(
          std::format("measurement node {} is off the time grid", node));
    }
    detail::CheckGrids(m, fieldSize, node);
  }
  if (trajectory != nullptr) {
    if (trajectory->states.size() != times.size() || !trajectory->Kept(0) ||
        trajectory->loads.size() != times.size() ||
        trajectory->tides.size() != times.size()) {
      throw std::invalid_argument(
          "the trajectory must give one entry per time node, keep the "
          "first node's state, and carry the recorded load and centrifugal "
          "coefficients");
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
  if (rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.GetRheology().BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "SeaLevelEvolution3D::Run");
    }
  }

  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<const LoveNumbers*, SeaLevelSolver>{};
  auto solverFor = [&](const LoveNumbers& table) -> const SeaLevelSolver& {
    auto found = flavours.find(&table);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(&table, SeaLevelSolver(grid, table, solver.Parameters()))
              .first;
    }
    return found->second;
  };
  auto locked = [&](auto&& get) -> decltype(auto) {
    const auto lock = op.LockCaches();
    return get();
  };

  // The forward trajectory, served backwards with the gaps rebuilt by
  // forward coupled steps (the continuous sweep's cursor).
  auto loadsAt = [&](std::size_t node) {
    return ev.Columns(trajectory->loads[node], ev.toMatDensity_);
  };
  auto tidesAt = [&](std::size_t node) {
    return ev.Columns(trajectory->tides[node], ev.toMatPotential_);
  };
  auto rebuilt = std::map<std::size_t, Viscoelastic3D::State>{};
  auto forwardState = [&](std::size_t node) -> const Viscoelastic3D::State& {
    if (trajectory->states[node].has_value()) return *trajectory->states[node];
    rebuilt.erase(rebuilt.upper_bound(node), rebuilt.end());
    if (const auto found = rebuilt.find(node); found != rebuilt.end()) {
      return found->second;
    }
    rebuilt.clear();
    if (trajectory->Kept(node)) {
      return rebuilt[node] = trajectory->fetch(node);
    }
    auto start = node;
    while (!trajectory->Kept(start)) --start;
    const auto* previous = trajectory->states[start].has_value()
                               ? &*trajectory->states[start]
                               : &(rebuilt[start] = trajectory->fetch(start));
    for (auto j = start + 1; j <= node; ++j) {
      const auto dt = (times[j] - times[j - 1]) * ev.toMatTime_;
      auto step =
          op.StartStep(*previous, dt, scheme, loadsAt(j - 1), tidesAt(j - 1));
      const auto load = loadsAt(j);
      const auto tide = tidesAt(j);
      auto [strain, info] = step.Converge(
          [&](const GridCoupling::SpectralTensor& s) {
            auto solutions = step.Respond(s);
            step.AddChannels(solutions, load, tide);
            return step.StrainOf(solutions);
          },
          {.max_iterations = ev.options_.max_coupled_iterations,
           .rtol = ev.options_.coupled_rtol,
           .anderson_memory = ev.options_.anderson_memory,
           .anderson_rate = ev.options_.anderson_rate});
      if (!info.converged) {
        throw std::runtime_error(std::format(
            "rebuilding the forward state at node {} did not converge", j));
      }
      if (j > start + 1) rebuilt.at(j - 1).strain.reset();
      previous = &(rebuilt[j] = std::move(step).Finish(std::move(strain)));
    }
    return rebuilt.at(node);
  };
  auto strainNode = times.size();
  Tensor regenerated;
  auto forwardStrain = [&](std::size_t node) -> const Tensor& {
    const auto& fwd = forwardState(node);
    if (fwd.strain.has_value()) return *fwd.strain;
    if (strainNode != node) {
      regenerated = fwd.strainCoefficients.has_value()
                        ? coupling.Synthesise(*fwd.strainCoefficients)
                        : op.ElasticStrain(fwd, loadsAt(node), tidesAt(node));
      strainNode = node;
    }
    return regenerated;
  };

  // -- the pieces of a transposed pass
  // ---------------------------------------
  // The unit load and tide responses of a system (the channels'
  // responses), cached in the given map as AddUnitChannels keeps them.
  auto unitsOf = [&](const DegreeOperator& system, int l,
                     std::map<int, std::vector<double>>& cache)
      -> const std::vector<double>& {
    const auto lock = op.LockCaches();
    auto found = cache.find(l);
    if (found == cache.end()) {
      const auto ndof = system.DofCount();
      auto unit = std::vector<double>(static_cast<std::size_t>(ndof) * 2, 0.0);
      const auto loadVec = system.Load();
      const auto tideVec = system.Tide(l == 0 ? 2 : -1);
      std::ranges::copy(loadVec, unit.begin());
      std::ranges::copy(tideVec, unit.begin() + ndof);
      found = cache.emplace(l, system.Solve(unit, 2)).first;
    }
    return found->second;
  };
  // The duals of the step's load and centrifugal coefficients: a solution
  // dual contracted with the unit responses, column by column, times the
  // channel factors (the transpose of AddUnitChannels under the Hermitian
  // column pairing).
  auto channelDuals = [&](const Forces& b,
                          const std::function<const std::vector<double>&(int)>&
                              units) {
    auto sigmaDual = grid.ZeroCoefficients();
    auto psiDual = grid.ZeroCoefficients();
    for (const auto& [l, bl] : b) {
      const auto& u = units(l);
      const auto ndof = u.size() / 2;
      for (auto m = 0; m <= l; ++m) {
        Complex s = 0.0, p = 0.0;
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          const auto v = bl[static_cast<std::size_t>(m) * ndof + d];
          s += u[d] * v;
          p += u[ndof + d] * v;
        }
        sigmaDual[SurfaceGrid::Index(l, m)] = ev.toMatDensity_ * s;
        psiDual[SurfaceGrid::Index(l, m)] = ev.toMatPotential_ * p;
      }
    }
    return std::pair(std::move(sigmaDual), std::move(psiDual));
  };
  // The engine-adjoint inputs at a node: the duals carried by the node's
  // total-load and centrifugal coefficients, plus the measurement's.
  auto engineDuals = [&](const std::vector<Complex>& sigmaDual,
                         const std::vector<Complex>& psiDual,
                         std::size_t node) {
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
    if (auto found = measurements.find(node); found != measurements.end()) {
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
  // The engine adjoint's response duals placed on the surface dofs of the
  // spheroidal columns, and the inertia-moment dual on the degree-0
  // moment row (the transpose of the response channels).
  const auto& momentRow =
      locked([&]() -> const std::vector<double>& {
        return op.reference_.Deg(0).momentRow;
      });
  auto addSurfaceDuals =
      [&](Forces& rhs, const SeaLevelSolver::AdjointResponse& adj,
          const std::function<const DegreeOperator&(int)>& spheroidal) {
        for (auto& [l, bl] : rhs) {
          const auto& system = spheroidal(l);
          const auto ndof = system.DofCount();
          const auto su = system.SurfaceDof(Component::U);
          const auto sphi = system.SurfaceDof(Component::Phi);
          for (auto m = 0; m <= l; ++m) {
            const auto i = SurfaceGrid::Index(l, m);
            if (su >= 0) {
              bl[static_cast<std::size_t>(m) * ndof + su] +=
                  ev.toNdLength_ * adj.displacementResponseDual[i];
            }
            if (sphi >= 0) {
              bl[static_cast<std::size_t>(m) * ndof + sphi] +=
                  ev.toNdPotential_ * adj.potentialResponseDual[i];
            }
          }
          if (l == 0) {
            const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
            for (auto d = 0; d < ndof; ++d) bl[d] += momentRow[d] * jd;
          }
        }
      };
  // Exit: the grid strain of degree solutions; Entry: the forces of a
  // grid tensor's analysis — transposes of each other.
  auto exit = [&](const Viscoelastic3D::Solutions& x,
                  const std::function<const DegreeOperator&(int)>& sph,
                  const std::function<const ToroidalOperator&(int)>& tor) {
    return coupling.Synthesise(op.StrainCoefficients(x, sph, tor));
  };
  auto entry = [&](const GridCoupling::SpectralTensor& z,
                   const std::function<const DegreeOperator&(int)>& sph,
                   const std::function<const ToroidalOperator&(int)>& tor) {
    return std::pair(coupling.SpheroidalForce(z, sph),
                     coupling.ToroidalForce(z, tor));
  };
  auto loadDualOf = [&](const SeaLevelSolver::AdjointResponse& adj,
                        const std::vector<double>& synth) {
    auto out = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      out[i] = adj.directLoadDual[i] + synth[i] / r2;
    }
    return out;
  };

  // -- the dual state
  // ---------------------------------------------------------
  auto mdual = std::vector<Tensor>{};
  for (auto k = std::size_t{0}; k < K; ++k) mdual.push_back(coupling.ZeroTensor());
  auto ddual = coupling.ZeroTensor();
  auto itauKernel = std::vector<std::vector<double>>{};
  if (trajectory != nullptr) {
    itauKernel.assign(K, std::vector<double>(values, 0.0));
  }
  auto loadDuals = std::vector<std::vector<double>>(times.size());
  const auto zeroState = op.ZeroState();

  // -- the material pairing
  // ---------------------------------------------------
  // The branch fractions f_k = mu_k / mu_U per node, the kernels over the
  // grid values, and the pairings of a dual solution with a base one:
  // the energy contraction of their grid strains for the shear (what the
  // itau kernel pairs with), the product of their dilatations through
  // the scalar companion for the bulk, the radial weight folded in.
  std::vector<double> kappaKernel, muKernel;
  auto fractions = std::vector<std::vector<double>>{};
  if (material) {
    kappaKernel.assign(values, 0.0);
    muKernel.assign(values, 0.0);
    const auto& base = op.GetRheology().BaseMaterial();
    const auto ngll = base.Mesh().NGll();
    const auto [kappaU, muU] = base.Moduli().KappaMu();
    for (auto k = std::size_t{0}; k < K; ++k) {
      auto f = std::vector<double>(nodeCount, 0.0);
      for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
        const auto full =
            static_cast<std::size_t>(coupling.ElementOfNode(static_cast<int>(rn))) *
                ngll +
            coupling.NodeInElementOf(static_cast<int>(rn));
        f[rn] = muU[full] > 0.0 ? op.muBranch_[k][rn] / muU[full] : 0.0;
      }
      fractions.push_back(std::move(f));
    }
  }
  auto dilatationOf = [&](const Viscoelastic3D::Solutions& x,
                          const std::function<const DegreeOperator&(int)>& sph) {
    auto scalar = std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
    for (const auto& [l, xl] : x.spheroidal) {
      coupling.AddDilatationOfSolution(scalar, l, sph(l), xl);
    }
    return coupling.SynthesiseScalar(scalar);
  };
  // The bulk pairing -Wq D(y) D(x) and the shear pairing
  // -Wq r^2 factor C(e(y), e(x)) + Wq r^2 sum_k f_k C(e(y), m~_k), over
  // the grid values; `factor` and `memory` empty for the start.
  auto addMaterialPairing = [&](std::span<const double> Dy,
                                std::span<const double> Dx, const Tensor& ey,
                                const Tensor& ex, std::span<const double> factor,
                                const std::vector<Tensor>* memory) {
    detail::ParallelElements(nodeCount, op.options_.parallel, [&](std::size_t rn) {
      const auto base = coupling.FieldOffset(static_cast<int>(rn));
      const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
      const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
      const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto i = base + x;
        kappaKernel[i] -= wq * Dy[i] * Dx[i];
        auto contraction = [&](const Tensor& a, const Tensor& b) {
          return 12.0 * a.r[i] * b.r[i] +
                 8.0 * (a.c[1][i] * std::conj(b.c[1][i])).real() +
                 4.0 * (a.c[0][i] * std::conj(b.c[0][i])).real();
        };
        auto shear = -(factor.empty() ? 1.0 : factor[i]) * contraction(ey, ex);
        if (memory != nullptr) {
          for (auto k = std::size_t{0}; k < K; ++k) {
            shear += fractions[k][rn] * contraction(ey, (*memory)[k]);
          }
        }
        muKernel[i] += wq * rr * rr * shear;
      }
    });
  };

  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = (times[n] - times[n - 1]) * ev.toMatTime_;
    // The step the forward took, opened on its start state (its weight
    // field, effective systems and unit responses are what the transpose
    // needs; the start state itself only feeds the itau pairing).
    const auto& startState =
        trajectory != nullptr ? forwardState(n - 1) : zeroState;
    auto step = op.StartStep(
        startState, dt, scheme,
        trajectory != nullptr ? loadsAt(n - 1) : Viscoelastic3D::LoadCallable3D{},
        trajectory != nullptr ? tidesAt(n - 1) : Viscoelastic3D::LoadCallable3D{});
    const auto sph = [&](int l) -> const DegreeOperator& {
      return step.Spheroidal(l);
    };
    const auto tor = [&](int l) -> const ToroidalOperator& {
      return step.Toroidal(l);
    };
    const auto units = [&](int l) -> const std::vector<double>& {
      return step.Units(l);
    };
    const auto& effectiveSolver = solverFor(step.EffectiveLove());
    const auto& W = step.weight_;

    // The dual of the end strain: its own, plus the memory duals through
    // the finish m+ = m~ + beta(x) d.
    auto de = ddual;
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& beta = step.beta_[k];
      detail::ParallelElements(values, op.options_.parallel, [&](std::size_t i) {
        de.c[0][i] += beta[i] * mdual[k].c[0][i];
        de.c[1][i] += beta[i] * mdual[k].c[1][i];
        de.r[i] += beta[i] * mdual[k].r[i];
      });
    }

    // The transposed fixed point: z = A(de + W Exit(K^-1 [Entry(z) +
    // Surf^T SLE_adj(U^T Entry(z), g)])), the pass's byproducts kept.
    struct Last {
      SeaLevelSolver::AdjointResponse adj;
      std::vector<double> synth;
      Tensor e;
      Viscoelastic3D::Solutions xdag;
    } last;
    auto pass = [&](const GridCoupling::SpectralTensor& z) {
      auto [sphForces, torForces] = entry(z, sph, tor);
      const auto [sigmaDual, psiDual] = channelDuals(sphForces, units);
      auto [duals, synth] = engineDuals(sigmaDual, psiDual, n);
      auto adj = effectiveSolver.SolveAdjoint(ev.state_, duals, sleOptions);
      if (!adj.converged) {
        throw std::runtime_error(std::format(
            "the adjoint sea level solve at node {} did not converge", n));
      }
      addSurfaceDuals(sphForces, adj, sph);
      const auto xdag = op.SolveForces(sphForces, torForces, sph, tor);
      auto e = exit(xdag, sph, tor);
      auto y = e;
      coupling.Multiply(y, W);
      detail::ParallelElements(values, op.options_.parallel, [&](std::size_t i) {
        y.c[0][i] += de.c[0][i];
        y.c[1][i] += de.c[1][i];
        y.r[i] += de.r[i];
      });
      last = {std::move(adj), std::move(synth), std::move(e),
              material ? std::move(xdag) : Viscoelastic3D::Solutions{}};
      return coupling.Analyse(y);
    };
    const auto start = coupling.Analyse(de);
    auto [zstar, info] = step.Converge(
        pass,
        {.max_iterations = options_.max_coupled_iterations,
         .rtol = options_.coupled_rtol,
         .anderson_memory = options_.anderson_memory,
         .anderson_rate = options_.anderson_rate},
        &start);
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the transposed lateral fixed point at node {} did not converge",
          n));
    }
    // The byproducts at the converged dual itself.
    (void)pass(zstar);
    loadDuals[n] = loadDualOf(last.adj, last.synth);

    // The dual of the decayed memory: its own plus the fixed memory
    // force's, mu_k(r) times the adjoint strain; then the itau pairing
    // and the recursion closing m-dagger and d-dagger pointwise.
    // Both strains copied: the regenerated one lives in a buffer the next
    // regeneration overwrites.
    const auto* mPrev = trajectory != nullptr ? &startState.m : nullptr;
    Tensor d0Copy, d1Copy;
    if (trajectory != nullptr) {
      d1Copy = forwardStrain(n);
      d0Copy = forwardStrain(n - 1);
    }
    const Tensor* d0 = &d0Copy;
    const auto& e = last.e;
    if (material) {
      // The step's material pairing: the dual solution against the base
      // full solution at the node (one unrelaxed solve, whose dilatation
      // the bulk term needs; its strain is the recorded one) with the
      // step's effective factor 1 - sum_k f_k beta_k(x), and against the
      // base decayed memory the step holds.
      const auto xn = op.SolveElastic(forwardState(n), loadsAt(n), tidesAt(n));
      auto factor = std::vector<double>(values, 1.0);
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& beta = step.beta_[k];
        for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
          const auto fk = fractions[k][rn];
          if (fk == 0.0) continue;
          const auto base = coupling.FieldOffset(static_cast<int>(rn));
          const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
          for (auto x = std::size_t{0}; x < size; ++x) {
            factor[base + x] -= fk * beta[base + x];
          }
        }
      }
      const auto unrelaxed = [&](int l) -> const DegreeOperator& {
        return op.Spheroidal(l);
      };
      addMaterialPairing(dilatationOf(last.xdag, sph), dilatationOf(xn, unrelaxed),
                         e, d1Copy, factor, &step.memory_);
    }
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& mu = op.muBranch_[k];
      const auto& itau = op.itau_[k];
      auto& md = mdual[k];
      auto* kernel = trajectory != nullptr ? &itauKernel[k] : nullptr;
      detail::ParallelElements(nodeCount, op.options_.parallel, [&](std::size_t rn) {
        const auto base = coupling.FieldOffset(static_cast<int>(rn));
        const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
        const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
        const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
        for (auto x = std::size_t{0}; x < size; ++x) {
          const auto i = base + x;
          const Complex mt2 = md.c[0][i] + mu[rn] * e.c[0][i];
          const Complex mt1 = md.c[1][i] + mu[rn] * e.c[1][i];
          const double mt0 = md.r[i] + mu[rn] * e.r[i];
          const auto h = dt * itau[i];
          double E, A, dE, dA, dB;
          if (implicit) {
            E = BackwardEulerWeight(h);
            A = 0.0;
            dE = BackwardEulerWeightDerivative(h);
            dA = 0.0;
            dB = -dE;
          } else {
            const auto w = ExponentialTrapezoidWeights(h);
            const auto dw = ExponentialTrapezoidWeightDerivatives(h);
            E = w.e;
            A = w.alpha;
            dE = dw.e;
            dA = dw.alpha;
            dB = dw.beta;
          }
          if (kernel != nullptr && itau[i] > 0.0) {
            // d(m~)/d(itau) = dt (dE m + dA d0 + dB d1), paired with
            // m~-dagger in the energy pairing: an angular density with
            // the node's radial quadrature folded in.
            const auto& m = (*mPrev)[k];
            const Complex t2 = dE * m.c[0][i] + dA * d0->c[0][i] + dB * d1Copy.c[0][i];
            const Complex t1 = dE * m.c[1][i] + dA * d0->c[1][i] + dB * d1Copy.c[1][i];
            const double t0 = dE * m.r[i] + dA * d0->r[i] + dB * d1Copy.r[i];
            const double contraction = 12.0 * mt0 * t0 +
                                       8.0 * (mt1 * std::conj(t1)).real() +
                                       4.0 * (mt2 * std::conj(t2)).real();
            (*kernel)[i] += dt * wq * rr * rr * contraction;
          }
          md.c[0][i] = E * mt2;
          md.c[1][i] = E * mt1;
          md.r[i] = E * mt0;
          // the start strain's dual: sum over branches, so accumulated
          // after the branch loop below
          if (k == 0) {
            ddual.c[0][i] = A * mt2;
            ddual.c[1][i] = A * mt1;
            ddual.r[i] = A * mt0;
          } else {
            ddual.c[0][i] += A * mt2;
            ddual.c[1][i] += A * mt1;
            ddual.r[i] += A * mt0;
          }
        }
      });
    }
  }

  // The start: the transposed instantaneous elastic solve at times[0]
  // (the unrelaxed, laterally uniform systems; no memory).
  {
    const auto sph = [&](int l) -> const DegreeOperator& {
      return op.Spheroidal(l);
    };
    const auto tor = [&](int l) -> const ToroidalOperator& {
      return op.Toroidal(l);
    };
    auto unrelaxedUnits = std::map<int, std::vector<double>>{};
    const auto units = [&](int l) -> const std::vector<double>& {
      return unitsOf(sph(l), l, unrelaxedUnits);
    };
    auto [sphForces, torForces] = entry(coupling.Analyse(ddual), sph, tor);
    const auto [sigmaDual, psiDual] = channelDuals(sphForces, units);
    auto [duals, synth] = engineDuals(sigmaDual, psiDual, 0);
    const auto adj =
        solverFor(op.ElasticLove()).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          "the adjoint elastic sea level solve did not converge");
    }
    loadDuals[0] = loadDualOf(adj, synth);
    if (material) {
      // The start's dual solution under the start duals and the start's
      // response duals, against the base start solution: every weight
      // one, no memory.
      addSurfaceDuals(sphForces, adj, sph);
      const auto y0 = op.SolveForces(sphForces, torForces, sph, tor);
      const auto x0 = op.SolveElastic(forwardState(0), loadsAt(0), tidesAt(0));
      addMaterialPairing(dilatationOf(y0, sph), dilatationOf(x0, sph),
                         exit(y0, sph, tor), exit(x0, sph, tor), {}, nullptr);
    }
  }
  op.ReleaseScratch();

  // The delta amplitudes and the distributional view, as the 1D discrete
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
    const auto adj =
        solverFor(op.ElasticLove()).SolveAdjoint(ev.state_, duals, sleOptions);
    if (!adj.converged) {
      throw std::runtime_error(
          "the instantaneous adjoint fingerprint did not converge");
    }
    lumps.emplace(node, adj.directLoadDual);
  }
  const auto w = detail::NodeWeights(times);
  auto densityLoad = std::vector<std::vector<double>>(times.size());
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    densityLoad[n].assign(fieldSize, 0.0);
    const auto lump = lumps.find(n);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double lumped = lump == lumps.end() ? 0.0 : lump->second[i];
      densityLoad[n][i] = (loadDuals[n][i] - lumped) / w[n];
    }
  }
  Kernels3D out;
  out.ice = detail::AssembleIceViews(
      C, solver.IceDensity(), times, std::move(loadDuals),
      std::move(densityLoad), std::move(lumps), oceanFunctions, rhoW);
  if (trajectory != nullptr) {
    out.itauKernel3D = std::move(itauKernel);
    out.itauKernelSpherical.assign(K, std::vector<double>(nodeCount, 0.0));
    out.logViscosityKernel3D.assign(K, std::vector<double>(values, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        out.itauKernelSpherical[k][rn] = coupling.GridOf(rn).Integrate(
            std::span(out.itauKernel3D[k])
                .subspan(coupling.FieldOffset(rn), coupling.FieldSizeOf(rn)));
      }
      for (auto i = std::size_t{0}; i < values; ++i) {
        out.logViscosityKernel3D[k][i] =
            -op.itau_[k][i] * out.itauKernel3D[k][i];
      }
    }
  }
  if (material) {
    out.kappaKernel3D = std::move(kappaKernel);
    out.muKernel3D = std::move(muKernel);
    out.kappaKernelSpherical.assign(nodeCount, 0.0);
    out.muKernelSpherical.assign(nodeCount, 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto at = coupling.FieldOffset(rn);
      const auto size = coupling.FieldSizeOf(rn);
      out.kappaKernelSpherical[rn] = coupling.GridOf(rn).Integrate(
          std::span(out.kappaKernel3D).subspan(at, size));
      out.muKernelSpherical[rn] = coupling.GridOf(rn).Integrate(
          std::span(out.muKernel3D).subspan(at, size));
    }
  }
  return out;
}

}  // namespace sl3dpp
