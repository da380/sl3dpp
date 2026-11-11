// The discrete 3D adjoint: the exact transpose of the coupled
// lateral-viscosity time-stepped map (docs/adjoints.md). Under the
// energy pairing the force assembly is the transpose of the strain
// evaluation (Entry = Exit^T), the pointwise weight and the reference
// systems are self-adjoint, and the sea level equation has its exact
// adjoint, so the transposed pass is the forward pass with the force and
// strain legs swapped and the sea level solve replaced by its adjoint; the
// transposed within-step fixed point is solved by the same iteration, or
// (Viscoelastic3D::Options::solver) by conjugate gradients on the dual
// solutions with the sea level adjoint nested outside. The
// duals of grid tensors live in the energy pairing (sum over nodes of
// Wq r^2 int dOmega of the null-basis contraction), the duals of degree
// columns in the Hermitian m >= 0 pairing (m > 0 weighted twice).
//
// Given v's recorded tangent-linear run and its delta itau(x), the sweep
// also carries the derivative of the dual state along v — the discrete
// second-order adjoint, the 3D form of AdjointSolver::DiscreteSecondOrder:
// the derivative duals are stepped by the same transposed fixed point,
// forced by the derivative of the step's lateral weight acting on the first
// dual strain, and the itau pairing and the weight recursion are
// differentiated along v.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/detail/AdjointInternals.hpp"
#include "sl3dpp/detail/AndersonMixer.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
using Forces = std::map<int, std::vector<Complex>>;
using Tensor = GridCoupling::TensorField;

}  // namespace

std::pair<Kernels3D, KernelDerivatives3D> AdjointSolver3D::DiscreteSweep(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const Evolution3D::Snapshot> tangent,
    const std::vector<std::vector<double>>* ditau,
    const MaterialDirection3D* materialV) const {
  const bool second = ditau != nullptr;
  const bool hasMaterialV = materialV != nullptr && !materialV->Empty();
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
  // Distributed over radius (Viscoelastic3D::Options::collectives): each
  // rank transposes its own memory nodes' part of every pass, the duals
  // of the load and centrifugal coefficients are the ranks' partial
  // contractions summed, every rank solves the sea level adjoint alike
  // from them and rank 0's answers serve all (the ranks' transforms may
  // round differently), the surface duals and the moment row join the
  // forces on the rank holding their rows, and a solve that fails fails
  // on all — the forward step's conventions (Evolution3D::March)
  // transposed.
  const auto& collectives = op.GetOptions().collectives;
  auto agree = [&](auto& values) {
    if (!collectives.Distributed() || values.empty()) return;
    using Value = typename std::remove_reference_t<decltype(values)>::value_type;
    collectives.share(std::span(reinterpret_cast<double*>(values.data()),
                                values.size() * sizeof(Value) / sizeof(double)));
  };
  auto agreeAdjoint = [&](GRDSolver::AdjointResponse& adj) {
    if (!collectives.Distributed()) return;
    agree(adj.directLoadDual);
    agree(adj.displacementResponseDual);
    agree(adj.potentialResponseDual);
    auto scalars = std::array{adj.inertiaMomentDual, adj.converged ? 0.0 : 1.0};
    collectives.max(std::span(&scalars[1], 1));
    collectives.share(std::span(&scalars[0], 1));
    adj.inertiaMomentDual = scalars[0];
    adj.converged = scalars[1] == 0.0;
  };
  const bool material = options_.material_kernels;
  if (second) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the second-order sweep needs the recorded forward trajectory");
    }
    if (tangent.size() != times.size()) {
      throw std::invalid_argument(
          "the tangent-linear run must give one snapshot per time node");
    }
  }
  if (material) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the material kernels need the recorded forward trajectory");
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
          "Evolution3D::Run");
    }
  }

  // Whether the transposed steps are closed as the forward's are
  // (Viscoelastic3D::Options::solver): the transposed fixed point, or the
  // sea level adjoint nested over conjugate-gradient solves.
  const bool nested = op.GetOptions().solver ==
                      Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<const LoveNumbers*, GRDSolver>{};
  auto solverFor = [&](const LoveNumbers& table) -> const GRDSolver& {
    auto found = flavours.find(&table);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(&table, GRDSolver(grid, table, solver.Parameters()))
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
      auto [strain, info] = step.Close(
          loadsAt(j), tidesAt(j),
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
    if (collectives.Distributed()) {
      // the forces are held in shares (each rank its own rows), the unit
      // responses whole: the ranks' partial contractions sum
      auto all = sigmaDual;
      all.insert(all.end(), psiDual.begin(), psiDual.end());
      collectives.sum(std::span(reinterpret_cast<double*>(all.data()),
                                2 * all.size()));
      std::copy(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(sigmaDual.size()),
                sigmaDual.begin());
      std::copy(all.begin() + static_cast<std::ptrdiff_t>(sigmaDual.size()), all.end(),
                psiDual.begin());
    }
    return std::pair(std::move(sigmaDual), std::move(psiDual));
  };
  // The engine-adjoint inputs at a node: the duals carried by the node's
  // total-load and centrifugal coefficients, plus the measurement's.
  auto engineDuals = [&](const std::vector<Complex>& sigmaDual,
                         const std::vector<Complex>& psiDual,
                         std::size_t node, bool measured) {
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
  // The engine adjoint's response duals placed on the surface dofs of the
  // spheroidal columns, and the inertia-moment dual on the degree-0
  // moment row (the transpose of the response channels).
  const auto& momentRow =
      locked([&]() -> const std::vector<double>& {
        return op.reference_.Deg(0).momentRow;
      });
  auto addSurfaceDuals =
      [&](Forces& rhs, const GRDSolver::AdjointResponse& adj,
          const std::function<const DegreeOperator&(int)>& spheroidal) {
        for (auto& [l, bl] : rhs) {
          const auto& system = spheroidal(l);
          const auto ndof = system.DofCount();
          const auto su = system.SurfaceDof(Component::U);
          const auto sphi = system.SurfaceDof(Component::Phi);
          // the rows join the shares on the rank holding them
          const bool holdU = su >= 0 && system.Holds(su);
          const bool holdPhi = sphi >= 0 && system.Holds(sphi);
          for (auto m = 0; m <= l; ++m) {
            const auto i = SurfaceGrid::Index(l, m);
            if (holdU) {
              bl[static_cast<std::size_t>(m) * ndof + su] +=
                  ev.toNdLength_ * adj.displacementResponseDual[i];
            }
            if (holdPhi) {
              bl[static_cast<std::size_t>(m) * ndof + sphi] +=
                  ev.toNdPotential_ * adj.potentialResponseDual[i];
            }
          }
          if (l == 0) {
            const double jd = adj.inertiaMomentDual * ev.toNdMoment_;
            for (auto d = 0; d < ndof; ++d) {
              if (system.Holds(d)) bl[d] += momentRow[d] * jd;
            }
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
    // with the dilatation's force where z carries the scalar companion
    // (the lateral bulk modulus): the transpose of exit's dilatation
    return std::pair(op.SpheroidalForces(z, sph), coupling.ToroidalForce(z, tor));
  };
  // An iterate of the transposed fixed point shaped as the pass returns
  // it: with the dilatation's coefficients under a lateral bulk modulus.
  auto shaped = [&](GridCoupling::SpectralTensor s) {
    if (s.scalars.size() != op.CompanionCount()) {
      s.scalars.assign(op.CompanionCount(),
                       std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0));
    }
    return s;
  };
  auto loadDualOf = [&](const GRDSolver::AdjointResponse& adj,
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
  auto directLoadKernels = std::vector<std::vector<double>>(times.size());
  const auto zeroState = op.ZeroState();
  // The derivative dual state along v (the second-order sweep): the
  // derivatives of m-dagger and d-dagger, of the itau pairing, and of the
  // load duals.
  auto dmdual = std::vector<Tensor>{};
  auto dddual = Tensor{};
  auto dItauKernel = std::vector<std::vector<double>>{};
  auto dloadDuals = std::vector<std::vector<double>>{};
  if (second) {
    for (auto k = std::size_t{0}; k < K; ++k) {
      dmdual.push_back(coupling.ZeroTensor());
    }
    dddual = coupling.ZeroTensor();
    dItauKernel.assign(K, std::vector<double>(values, 0.0));
    dloadDuals.resize(times.size());
  }
  auto tangentState = [&](std::size_t node) -> const Viscoelastic3D::State& {
    return *tangent[node].internalState;
  };

  // -- the material pairing
  // ---------------------------------------------------
  // The branch fractions f_k = mu_k / mu_U per node, the kernels over the
  // grid values, and the pairings of a dual solution with a base one:
  // the energy contraction of their grid strains for the shear (what the
  // itau kernel pairs with), the product of their dilatations through
  // the scalar companion for the bulk, the radial weight folded in.
  std::vector<double> kappaKernel, muKernel, dKappaKernel, dMuKernel;
  auto fractions = std::vector<std::vector<double>>{};
  // v's delta mu over the grid values and delta mu / mu_U (for the
  // memory force's variation on the first dual strain); empty without
  // a mu direction.
  std::vector<double> dmuV, muRatioV;
  if (material || hasMaterialV) {
    // the reference's unrelaxed shear and branch fractions at each memory
    // node, the radial reference's at its memory points
    const auto np = static_cast<std::size_t>(coupling.Points().count);
    const auto muUAtPoints = op.Reference().UnrelaxedShearAtPoints();
    auto pointOf = [&](std::size_t rn) {
      return static_cast<std::size_t>(
                 coupling.ElementOfNode(static_cast<int>(rn))) *
                 np +
             coupling.NodeInElementOf(static_cast<int>(rn));
    };
    auto muUNode = std::vector<double>(nodeCount, 0.0);
    for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
      muUNode[rn] = muUAtPoints[pointOf(rn)];
    }
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto atPoints = op.Reference().BranchFractionAtPoints(k);
      auto f = std::vector<double>(nodeCount, 0.0);
      for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
        f[rn] = atPoints[pointOf(rn)];
      }
      fractions.push_back(std::move(f));
    }
    if (material) {
      kappaKernel.assign(values, 0.0);
      muKernel.assign(values, 0.0);
      if (second) {
        dKappaKernel.assign(values, 0.0);
        dMuKernel.assign(values, 0.0);
      }
    }
    if (hasMaterialV && !materialV->mu.empty()) {
      dmuV = materialV->mu;
      muRatioV.assign(values, 0.0);
      for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
        const auto base_ = coupling.FieldOffset(static_cast<int>(rn));
        const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
        for (auto x = std::size_t{0}; x < size; ++x) {
          muRatioV[base_ + x] =
              muUNode[rn] > 0.0 ? dmuV[base_ + x] / muUNode[rn] : 0.0;
        }
      }
    }
  }
  auto dilatationCoefficients =
      [&](const Viscoelastic3D::Solutions& x,
          const std::function<const DegreeOperator&(int)>& sph) {
        auto scalar =
            std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
        for (const auto& [l, xl] : x.spheroidal) {
          coupling.AddDilatationOfSolution(scalar, l, sph(l), xl);
        }
        return scalar;
      };
  auto dilatationOf =
      [&](const Viscoelastic3D::Solutions& x,
          const std::function<const DegreeOperator&(int)>& sph) {
        return coupling.SynthesiseScalar(dilatationCoefficients(x, sph));
      };
  // The bulk pairing -Wq D(y) D(x) and the shear pairing
  // -Wq r^2 factor C(e(y), e(x)) + Wq r^2 sum_k f_k C(e(y), m~_k), over
  // the grid values; `factor` and `memory` empty for the start.
  auto addMaterialPairing = [&](std::span<const double> Dy,
                                std::span<const double> Dx, const Tensor& ey,
                                const Tensor& ex,
                                std::span<const double> factor,
                                const std::vector<Tensor>* memory,
                                std::vector<double>& kappaTo,
                                std::vector<double>& muTo) {
    detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
      const auto base = coupling.FieldOffset(static_cast<int>(rn));
      const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
      const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
      const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto i = base + x;
        kappaTo[i] -= wq * Dy[i] * Dx[i];
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
        muTo[i] += wq * rr * rr * shear;
      }
    });
  };
  // v's forces on a solution with its grid strain (the material
  // tangent's: the shear weight on the deviatoric strain, pointwise, and
  // the bulk term through the scalar companion), through the given
  // spheroidal systems' layout: -(dK) x with `shear` as the shear
  // weight (delta mu_eff of a step, delta mu of the unrelaxed form), plus
  // an optional memory force's variation tensor.
  auto materialForcesOf =
      [&](const Tensor& strain, const Viscoelastic3D::Solutions* x,
          std::span<const double> shear, const Tensor* memoryForce,
          const std::function<const DegreeOperator&(int)>& sph,
          const std::function<const ToroidalOperator&(int)>& tor) {
        auto gridForce = coupling.ZeroTensor();
        detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
          const auto w = shear.empty() ? 0.0 : -shear[i];
          for (auto c = 0; c < 2; ++c) {
            gridForce.c[c][i] = w * strain.c[c][i] +
                                (memoryForce ? memoryForce->c[c][i] : Complex{});
          }
          gridForce.r[i] = w * strain.r[i] + (memoryForce ? memoryForce->r[i] : 0.0);
        });
        const auto analysed = coupling.Analyse(gridForce);
        auto sphForces = coupling.SpheroidalForce(analysed, sph);
        if (!materialV->kappa.empty() && x != nullptr) {
          // Through gridForce's real component, analysed (done with it),
          // and back over the dilatation's coefficients.
          auto scalar = dilatationCoefficients(*x, sph);
          auto& field = gridForce.r;
          coupling.SynthesiseScalar(scalar, field);
          for (auto i = std::size_t{0}; i < values; ++i) {
            field[i] *= -materialV->kappa[i];
          }
          coupling.AnalyseScalar(field, scalar);
          for (const auto& [l, f] : coupling.DilatationForce(scalar, sph)) {
            auto& b = sphForces.at(l);
            for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
          }
        }
        return std::pair(std::move(sphForces), coupling.ToroidalForce(analysed, tor));
      };
  // The memory force's variation of a memory state along v: sum_k f_k
  // delta mu_v(x) memory_k(x).
  auto memoryVariationV = [&](const std::vector<Tensor>& memory) {
    auto out = coupling.ZeroTensor();
    if (dmuV.empty()) return out;
    detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
      const auto base_ = coupling.FieldOffset(static_cast<int>(rn));
      const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto i = base_ + x;
        for (auto k = std::size_t{0}; k < K; ++k) {
          const auto fk = fractions[k][rn];
          if (fk == 0.0) continue;
          for (auto c = 0; c < 2; ++c) {
            out.c[c][i] += fk * dmuV[i] * memory[k].c[c][i];
          }
          out.r[i] += fk * dmuV[i] * memory[k].r[i];
        }
      }
    });
    return out;
  };
  auto addSolutions = [](Viscoelastic3D::Solutions& to,
                         const Viscoelastic3D::Solutions& add) {
    for (auto* pair : {&to.spheroidal, &to.toroidal}) {
      const auto& from = pair == &to.spheroidal ? add.spheroidal : add.toroidal;
      for (const auto& [l, f] : from) {
        auto found = pair->find(l);
        if (found == pair->end()) {
          pair->emplace(l, f);
        } else {
          for (auto i = std::size_t{0}; i < f.size(); ++i) found->second[i] += f[i];
        }
      }
    }
  };
  // v's full tangent solutions at a node from its recorded snapshot: the
  // unrelaxed solve under the snapshot's loads and memory, plus v's own
  // unrelaxed material force's response on the base solution (its strain
  // and memory) there.
  const auto unrelaxedSph = [&](int l) -> const DegreeOperator& {
    return op.Spheroidal(l);
  };
  const auto unrelaxedTor = [&](int l) -> const ToroidalOperator& {
    return op.Toroidal(l);
  };
  auto tangentSolution = [&](std::size_t node, const Tensor& baseStrain,
                             const Viscoelastic3D::Solutions& baseX,
                             const std::vector<Tensor>* baseMemory) {
    const auto& snap = tangent[node];
    auto x = op.SolveElastic(*snap.internalState,
                             ev.Columns(snap.loadCoefficients, ev.toMatDensity_),
                             ev.Columns(snap.tideCoefficients, ev.toMatPotential_));
    if (hasMaterialV) {
      auto memoryForce = std::optional<Tensor>{};
      if (baseMemory != nullptr && !dmuV.empty()) {
        memoryForce = memoryVariationV(*baseMemory);
      }
      auto [sphF, torF] =
          materialForcesOf(baseStrain, &baseX, dmuV,
                           memoryForce ? &*memoryForce : nullptr, unrelaxedSph,
                           unrelaxedTor);
      addSolutions(x, op.SolveUnrelaxed(sphF, torF));
    }
    return x;
  };

  const std::size_t lowest = op.HasLateralModuli() ? 0 : 1;
  for (std::size_t n = N + 1; n-- > lowest;) {
    if (options_.on_node) options_.on_node(n);
    // Under lateral moduli the start is the transposed zero-length step
    // at node 0 (the instantaneous coupled fixed point's transpose).
    const auto prev = n == 0 ? std::size_t{0} : n - 1;
    const double dt = n == 0 ? 0.0 : (times[n] - times[n - 1]) * ev.toMatTime_;
    // The step the forward took, opened on its start state (its weight
    // field, effective systems and unit responses are what the transpose
    // needs; the start state itself only feeds the itau pairing).
    const auto& startState =
        trajectory != nullptr ? forwardState(prev) : zeroState;
    auto step = op.StartStep(
        startState, dt, scheme,
        trajectory != nullptr ? loadsAt(prev) : Viscoelastic3D::LoadCallable3D{},
        trajectory != nullptr ? tidesAt(prev) : Viscoelastic3D::LoadCallable3D{});
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
    // the finish m+ = m~ + beta(x) d. Along v, the derivative duals'
    // likewise, plus the finish's derivative dbeta dh on the first memory
    // duals; and the derivative of the lateral weight,
    // sum_k mu_k(r) dbeta_k(x) dh_k(x), which forces the derivative
    // state's transposed fixed point through the first dual strain.
    auto de = ddual;
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& beta = step.beta_[k];
      detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
        de.c[0][i] += beta[i] * mdual[k].c[0][i];
        de.c[1][i] += beta[i] * mdual[k].c[1][i];
        de.r[i] += beta[i] * mdual[k].r[i];
      });
    }
    Tensor dde;
    std::vector<double> dW;
    if (second) {
      dde = dddual;
      dW.assign(values, 0.0);
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& beta = step.beta_[k];
        const auto& itau = op.itau_[k];
        const auto& mu = op.muBranch_[k];
        const auto& delta = (*ditau)[k];
        const auto& md = mdual[k];
        const auto& dmd = dmdual[k];
        detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
          const auto base = coupling.FieldOffset(static_cast<int>(rn));
          const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
          for (auto x = std::size_t{0}; x < size; ++x) {
            const auto i = base + x;
            const auto h = dt * itau[i];
            const auto dh = dt * delta[i];
            const auto dB = implicit
                                ? -BackwardEulerWeightDerivative(h)
                                : ExponentialTrapezoidWeightDerivatives(h).beta;
            dde.c[0][i] += beta[i] * dmd.c[0][i] + dB * dh * md.c[0][i];
            dde.c[1][i] += beta[i] * dmd.c[1][i] + dB * dh * md.c[1][i];
            dde.r[i] += beta[i] * dmd.r[i] + dB * dh * md.r[i];
            dW[i] += mu[rn] * op.ShearFactor(i) * dB * dh;
          }
        });
      }
    }

    // The transposed fixed point: z = A(src + W Exit(K^-1 [Entry(z) +
    // Surf^T SLE_adj(U^T Entry(z), g)])), the pass's byproducts kept: for
    // the first dual state src = de with the measurement duals, for the
    // derivative state src = dde + dW e with none.
    struct Last {
      GRDSolver::AdjointResponse adj;
      std::vector<double> synth;
      Tensor e;
      Viscoelastic3D::Solutions xdag;
    };
    auto passWith = [&](const Tensor& src, bool measured, Last& out,
                        const Forces* extraSph = nullptr,
                        const Forces* extraTor = nullptr) {
      return [&, measured, extraSph, extraTor](const GridCoupling::SpectralTensor& z) {
        auto [sphForces, torForces] = entry(z, sph, tor);
        if (extraSph != nullptr) {
          for (const auto& [l, f] : *extraSph) {
            auto& b = sphForces.at(l);
            for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
          }
          for (const auto& [l, f] : *extraTor) {
            auto& b = torForces.at(l);
            for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
          }
        }
        const auto [sigmaDual, psiDual] = channelDuals(sphForces, units);
        auto [duals, synth] = engineDuals(sigmaDual, psiDual, n, measured);
        auto adj = effectiveSolver.SolveAdjoint(ev.state_, duals, sleOptions);
        agreeAdjoint(adj);
        if (!adj.converged) {
          throw std::runtime_error(std::format(
              "the adjoint sea level solve at node {} did not converge", n));
        }
        addSurfaceDuals(sphForces, adj, sph);
        const auto xdag = op.SolveForces(sphForces, torForces, sph, tor);
        auto e = exit(xdag, sph, tor);
        auto y = e;
        coupling.Multiply(y, W);
        if (!y.scalars.empty()) op.WeightCompanions(y);
        detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
          y.c[0][i] += src.c[0][i];
          y.c[1][i] += src.c[1][i];
          y.r[i] += src.r[i];
        });
        out = {std::move(adj), std::move(synth), std::move(e),
               material || hasMaterialV ? std::move(xdag)
                                        : Viscoelastic3D::Solutions{}};
        return coupling.Analyse(y);
      };
    };
    const Viscoelastic3D::IterationControl control{
        .max_iterations = options_.max_coupled_iterations,
        .rtol = options_.coupled_rtol,
        .anderson_memory = options_.anderson_memory,
        .anderson_rate = options_.anderson_rate};
    // The same dual problem with the solve inside (Viscoelastic3D::
    // Options::solver = ConjugateGradients), the transpose of OpenStep::
    // Nest. In the dual solutions the step is
    //   (K - P^T W S) y = b0 + Surf^T SLE_adj(F^T r, g),
    // the forward step's own symmetric operator, b0 the source's forces
    // (with any added), F the unit channel forces and r = y less the
    // surface duals' reference response — the forces' reference response,
    // so F^T r is the contraction of the forces with the unit responses.
    // The outer iteration is on r: the engine's adjoint maps it to the
    // surface duals, the step is solved by conjugate gradients under
    // them, warm-started from r with their reference response (the
    // transposed pass's own image) and inexactly (solve_outer_factor),
    // and the outer residual is that warm start's. Picard, Anderson-mixed
    // on r by the control's rule.
    auto channelForces = std::map<int, std::array<std::vector<double>, 2>>{};
    auto channelDualsOfSolutions = [&](const Viscoelastic3D::Solutions& r) {
      auto sigmaDual = grid.ZeroCoefficients();
      auto psiDual = grid.ZeroCoefficients();
      for (const auto& [l, rl] : r.spheroidal) {
        auto found = channelForces.find(l);
        if (found == channelForces.end()) {
          const auto& system = sph(l);
          const auto lock = op.LockCaches();
          auto loadVec = system.Load();
          auto tideVec = system.Tide(l == 0 ? 2 : -1);
          if (collectives.Distributed()) {
            // each rank its own rows, the partial contractions summed
            system.MaskToShare(loadVec);
            system.MaskToShare(tideVec);
          }
          found = channelForces
                      .emplace(l, std::array{std::move(loadVec), std::move(tideVec)})
                      .first;
        }
        const auto& [loadVec, tideVec] = found->second;
        const auto ndof = loadVec.size();
        for (auto m = 0; m <= l; ++m) {
          Complex sum = 0.0, p = 0.0;
          for (auto d = std::size_t{0}; d < ndof; ++d) {
            const auto v = rl[static_cast<std::size_t>(m) * ndof + d];
            sum += loadVec[d] * v;
            p += tideVec[d] * v;
          }
          sigmaDual[SurfaceGrid::Index(l, m)] = ev.toMatDensity_ * sum;
          psiDual[SurfaceGrid::Index(l, m)] = ev.toMatPotential_ * p;
        }
      }
      if (collectives.Distributed()) {
        auto all = sigmaDual;
        all.insert(all.end(), psiDual.begin(), psiDual.end());
        collectives.sum(std::span(reinterpret_cast<double*>(all.data()),
                                  2 * all.size()));
        std::copy(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(sigmaDual.size()),
                  sigmaDual.begin());
        std::copy(all.begin() + static_cast<std::ptrdiff_t>(sigmaDual.size()), all.end(),
                  psiDual.begin());
      }
      return std::pair(std::move(sigmaDual), std::move(psiDual));
    };
    auto nestWith = [&](const GridCoupling::SpectralTensor& start, bool measured,
                        Last& out, const Forces* extraSph = nullptr,
                        const Forces* extraTor = nullptr) {
      auto [b0s, b0t] = entry(start, sph, tor);
      if (extraSph != nullptr) {
        for (const auto& [l, f] : *extraSph) {
          auto& b = b0s.at(l);
          for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
        }
        for (const auto& [l, f] : *extraTor) {
          auto& b = b0t.at(l);
          for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
        }
      }
      auto zeroSph = b0s;
      for (auto& [l, b] : zeroSph) std::ranges::fill(b, Complex{});
      auto zeroTor = b0t;
      for (auto& [l, b] : zeroTor) std::ranges::fill(b, Complex{});
      auto combine = [&](Viscoelastic3D::Solutions& to,
                         const Viscoelastic3D::Solutions& from, double factor) {
        for (auto* block : {&to.spheroidal, &to.toroidal}) {
          const auto& other =
              block == &to.spheroidal ? from.spheroidal : from.toroidal;
          for (auto& [l, v] : *block) {
            const auto& o = other.at(l);
            detail::ForEach(op.options_.parallel, v.size(),
                            [&](std::size_t i) { v[i] += factor * o[i]; });
          }
        }
      };
      auto segments = [](Viscoelastic3D::Solutions& x) {
        auto parts = std::vector<std::span<double>>{};
        for (auto* block : {&x.spheroidal, &x.toroidal}) {
          for (auto& [l, v] : *block) {
            parts.emplace_back(reinterpret_cast<double*>(v.data()), 2 * v.size());
          }
        }
        return parts;
      };
      auto constSegments = [](const Viscoelastic3D::Solutions& x) {
        auto parts = std::vector<std::span<const double>>{};
        for (const auto* block : {&x.spheroidal, &x.toroidal}) {
          for (const auto& [l, v] : *block) {
            parts.emplace_back(reinterpret_cast<const double*>(v.data()),
                               2 * v.size());
          }
        }
        return parts;
      };
      const auto adaptive = control.anderson_rate > 0.0;
      auto mixer = std::optional<detail::AndersonMixer>{};
      if (control.anderson_memory > 0 && !adaptive) {
        mixer.emplace(control.anderson_memory, op.options_.parallel, collectives.sum);
      }
      auto iterate = op.SolveForces(b0s, b0t, sph, tor);
      auto previous = 0.0;
      for (auto pass = 0; pass < control.max_iterations; ++pass) {
        const auto [sigmaDual, psiDual] = channelDualsOfSolutions(iterate);
        auto [duals, synth] = engineDuals(sigmaDual, psiDual, n, measured);
        auto adj = effectiveSolver.SolveAdjoint(ev.state_, duals, sleOptions);
        agreeAdjoint(adj);
        if (!adj.converged) {
          throw std::runtime_error(std::format(
              "the adjoint sea level solve at node {} did not converge", n));
        }
        auto surface = zeroSph;
        addSurfaceDuals(surface, adj, sph);
        const auto response = op.SolveForces(surface, zeroTor, sph, tor);
        for (auto& [l, b] : surface) {
          const auto& fixed = b0s.at(l);
          for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += fixed[i];
        }
        auto warm = iterate;
        combine(warm, response, 1.0);
        auto [solved, solveInfo] =
            step.Solve(surface, b0t, &warm, 0.0, op.options_.solve_outer_factor);
        const auto change = solveInfo.initial;
        if (!std::isfinite(change)) break;
        if (change <= control.rtol && solveInfo.converged) {
          out = {std::move(adj), std::move(synth), exit(solved, sph, tor),
                 material || hasMaterialV ? std::move(solved)
                                          : Viscoelastic3D::Solutions{}};
          return true;
        }
        auto image = std::move(solved);
        combine(image, response, -1.0);
        if (adaptive && !mixer && control.anderson_memory > 0 && pass > 0 &&
            change > control.anderson_rate * previous) {
          mixer.emplace(control.anderson_memory, op.options_.parallel,
                        collectives.sum);
        }
        previous = change;
        if (mixer) {
          mixer->UpdateInPlace(segments(iterate), constSegments(image));
        } else {
          iterate = std::move(image);
        }
      }
      return false;
    };
    Last last;
    {
      const auto start = shaped(coupling.Analyse(de));
      auto converged = false;
      if (nested) {
        converged = nestWith(start, true, last);
      } else {
        const auto pass = passWith(de, true, last);
        auto [zstar, info] = step.Converge(pass, control, &start);
        converged = info.converged;
        // The byproducts at the converged dual itself.
        if (converged) (void)pass(zstar);
      }
      if (!converged) {
        throw std::runtime_error(std::format(
            "the transposed lateral {} at node {} did not converge",
            nested ? "nested solve" : "fixed point", n));
      }
    }
    directLoadKernels[n] = loadDualOf(last.adj, last.synth);
    // The effective modulus's factor 1 - sum_k f_k beta_k(x) at the grid
    // values (the material pairings' and v's effective shear weight's).
    auto factor = std::vector<double>{};
    if (material || hasMaterialV) {
      factor.assign(values, 1.0);
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
    }
    Last dlast;
    if (second) {
      auto src = last.e;
      coupling.Multiply(src, dW);
      detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
        src.c[0][i] += dde.c[0][i];
        src.c[1][i] += dde.c[1][i];
        src.r[i] += dde.r[i];
      });
      // Along a material v: -(dK_eff along v) on the first dual solution,
      // the dual of the full tangent solution, joining the forces before
      // the contraction (so it reaches the sea level adjoint).
      auto extraSph = Forces{};
      auto extraTor = Forces{};
      if (hasMaterialV) {
        auto effShear = std::vector<double>{};
        if (!dmuV.empty()) {
          effShear.resize(values);
          for (auto i = std::size_t{0}; i < values; ++i) {
            effShear[i] = dmuV[i] * factor[i];
          }
        }
        std::tie(extraSph, extraTor) =
            materialForcesOf(last.e, &last.xdag, effShear, nullptr, sph, tor);
      }
      const auto pass = passWith(src, false, dlast, hasMaterialV ? &extraSph : nullptr,
                                 hasMaterialV ? &extraTor : nullptr);
      src.scalars.clear();  // the source is deviatoric; the companions start at zero
      const auto start = shaped(coupling.Analyse(src));
      auto converged = false;
      if (nested) {
        converged = nestWith(start, false, dlast, hasMaterialV ? &extraSph : nullptr,
                             hasMaterialV ? &extraTor : nullptr);
      } else {
        auto [zstar, info] = step.Converge(pass, control, &start);
        converged = info.converged;
        if (converged) (void)pass(zstar);
      }
      if (!converged) {
        throw std::runtime_error(std::format(
            "the transposed lateral {} of the derivative state at node {} did "
            "not converge",
            nested ? "nested solve" : "fixed point", n));
      }
      dloadDuals[n] = loadDualOf(dlast.adj, dlast.synth);
    }

    // The dual of the decayed memory: its own plus the fixed memory
    // force's, mu_k(r) times the adjoint strain; then the itau pairing
    // and the recursion closing m-dagger and d-dagger pointwise.
    // Both strains copied: the regenerated one lives in a buffer the next
    // regeneration overwrites.
    const auto* mPrev = trajectory != nullptr ? &startState.m : nullptr;
    Tensor d0Copy, d1Copy;
    if (trajectory != nullptr) {
      d1Copy = forwardStrain(n);
      d0Copy = forwardStrain(prev);
    }
    const Tensor* d0 = &d0Copy;
    const auto& e = last.e;
    const auto& ed = dlast.e;
    const auto* vmPrev = second ? &tangentState(prev).m : nullptr;
    const Tensor* vd0 = second ? &*tangentState(prev).strain : nullptr;
    const Tensor* vd1 = second ? &*tangentState(n).strain : nullptr;
    if (material) {
      // The step's material pairing: the dual solution against the base
      // full solution at the node (one unrelaxed solve, whose dilatation
      // the bulk term needs; its strain is the recorded one) with the
      // step's effective factor 1 - sum_k f_k beta_k(x), and against the
      // base decayed memory the step holds.
      auto fwdN = forwardState(n);  // a copy: the cache may drop it below
      const auto xn = op.SolveElastic(fwdN, loadsAt(n), tidesAt(n));
      const auto Dy = dilatationOf(last.xdag, sph);
      const auto Dx = dilatationOf(xn, unrelaxedSph);
      addMaterialPairing(Dy, Dx, e, d1Copy, factor, &step.memory_, kappaKernel,
                         muKernel);
      if (second) {
        // Their derivative along v: the derivative dual against the base,
        // and the first dual against v's tangent solution (its strain the
        // recorded one) and v's decayed tangent memory with v's drive.
        const auto xvn = tangentSolution(n, d1Copy, xn, &fwdN.m);
        const auto& d1v = *tangentState(n).strain;
        auto prevState = Viscoelastic3D::State{};
        prevState.m = startState.m;
        prevState.strain = d0Copy;
        fwdN.strain = d1Copy;
        auto vstep = op.StartStep(
            tangentState(prev), dt, scheme,
            ev.Columns(tangent[prev].loadCoefficients, ev.toMatDensity_),
            ev.Columns(tangent[prev].tideCoefficients, ev.toMatPotential_));
        vstep.AddMemory(TangentLinearSolver3D::FirstOrderDrive(
            op, AdjointMethod::Discrete, scheme, dt, prevState, fwdN, *ditau));
        addMaterialPairing(dilatationOf(dlast.xdag, sph), Dx, ed, d1Copy, factor,
                           &step.memory_, dKappaKernel, dMuKernel);
        addMaterialPairing(Dy, dilatationOf(xvn, unrelaxedSph), e, d1v, factor,
                           &vstep.memory_, dKappaKernel, dMuKernel);
      }
    }
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& mu = op.muBranch_[k];
      const auto& itau = op.itau_[k];
      auto& md = mdual[k];
      auto* kernel = trajectory != nullptr ? &itauKernel[k] : nullptr;
      auto* dmd = second ? &dmdual[k] : nullptr;
      auto* dkernel = second ? &dItauKernel[k] : nullptr;
      const auto* delta = second ? &(*ditau)[k] : nullptr;
      detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
        const auto base = coupling.FieldOffset(static_cast<int>(rn));
        const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
        const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
        const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
        auto contraction = [](Complex a2, Complex a1, double a0, Complex b2,
                              Complex b1, double b0) {
          return 12.0 * a0 * b0 + 8.0 * (a1 * std::conj(b1)).real() +
                 4.0 * (a2 * std::conj(b2)).real();
        };
        for (auto x = std::size_t{0}; x < size; ++x) {
          const auto i = base + x;
          const auto muAt = mu[rn] * op.ShearFactor(i);
          const Complex mt2 = md.c[0][i] + muAt * e.c[0][i];
          const Complex mt1 = md.c[1][i] + muAt * e.c[1][i];
          const double mt0 = md.r[i] + muAt * e.r[i];
          const auto h = dt * itau[i];
          double E, A, dE, dA, dB, d2E = 0.0, d2A = 0.0, d2B = 0.0;
          if (implicit) {
            E = BackwardEulerWeight(h);
            A = 0.0;
            dE = BackwardEulerWeightDerivative(h);
            dA = 0.0;
            dB = -dE;
            if (second) {
              d2E = BackwardEulerWeightSecondDerivative(h);
              d2B = -d2E;
            }
          } else {
            const auto w = ExponentialTrapezoidWeights(h);
            const auto dw = ExponentialTrapezoidWeightDerivatives(h);
            E = w.e;
            A = w.alpha;
            dE = dw.e;
            dA = dw.alpha;
            dB = dw.beta;
            if (second) {
              const auto d2w = ExponentialTrapezoidWeightSecondDerivatives(h);
              d2E = d2w.e;
              d2A = d2w.alpha;
              d2B = d2w.beta;
            }
          }
          // The derivative dual of the decayed memory, and dh along v.
          Complex dmt2 = 0.0, dmt1 = 0.0;
          double dmt0 = 0.0, dh = 0.0;
          if (second) {
            // the memory force's variation along v, f_k delta mu_v on the
            // first dual strain, joins the derivative dual
            const auto dmuK = dmuV.empty() ? 0.0 : fractions[k][rn] * dmuV[i];
            dmt2 = dmd->c[0][i] + muAt * ed.c[0][i] + dmuK * e.c[0][i];
            dmt1 = dmd->c[1][i] + muAt * ed.c[1][i] + dmuK * e.c[1][i];
            dmt0 = dmd->r[i] + muAt * ed.r[i] + dmuK * e.r[i];
            dh = dt * (*delta)[i];
          }
          if (kernel != nullptr && itau[i] > 0.0) {
            // d(m~)/d(itau) = dt (dE m + dA d0 + dB d1), paired with
            // m~-dagger in the energy pairing: an angular density with
            // the node's radial quadrature folded in.
            const auto& m = (*mPrev)[k];
            const Complex t2 = dE * m.c[0][i] + dA * d0->c[0][i] + dB * d1Copy.c[0][i];
            const Complex t1 = dE * m.c[1][i] + dA * d0->c[1][i] + dB * d1Copy.c[1][i];
            const double t0 = dE * m.r[i] + dA * d0->r[i] + dB * d1Copy.r[i];
            (*kernel)[i] +=
                dt * wq * rr * rr * contraction(mt2, mt1, mt0, t2, t1, t0);
            if (second) {
              // Its derivative along v: the derivative duals against the
              // same, m~-dagger against the tangent trajectory's
              // (v m, v d0, v d1) in place, and against the weights'
              // second derivatives times dh.
              const auto& vm = (*vmPrev)[k];
              const Complex vt2 =
                  dE * vm.c[0][i] + dA * vd0->c[0][i] + dB * vd1->c[0][i];
              const Complex vt1 =
                  dE * vm.c[1][i] + dA * vd0->c[1][i] + dB * vd1->c[1][i];
              const double vt0 = dE * vm.r[i] + dA * vd0->r[i] + dB * vd1->r[i];
              const Complex s2 =
                  d2E * m.c[0][i] + d2A * d0->c[0][i] + d2B * d1Copy.c[0][i];
              const Complex s1 =
                  d2E * m.c[1][i] + d2A * d0->c[1][i] + d2B * d1Copy.c[1][i];
              const double s0 = d2E * m.r[i] + d2A * d0->r[i] + d2B * d1Copy.r[i];
              (*dkernel)[i] +=
                  dt * wq * rr * rr *
                  (contraction(dmt2, dmt1, dmt0, t2, t1, t0) +
                   contraction(mt2, mt1, mt0, vt2, vt1, vt0) +
                   dh * contraction(mt2, mt1, mt0, s2, s1, s0));
            }
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
          if (second) {
            // The recursion differentiated: the weights' derivatives
            // times dh on the first duals.
            dmd->c[0][i] = E * dmt2 + dE * dh * mt2;
            dmd->c[1][i] = E * dmt1 + dE * dh * mt1;
            dmd->r[i] = E * dmt0 + dE * dh * mt0;
            if (k == 0) {
              dddual.c[0][i] = A * dmt2 + dA * dh * mt2;
              dddual.c[1][i] = A * dmt1 + dA * dh * mt1;
              dddual.r[i] = A * dmt0 + dA * dh * mt0;
            } else {
              dddual.c[0][i] += A * dmt2 + dA * dh * mt2;
              dddual.c[1][i] += A * dmt1 + dA * dh * mt1;
              dddual.r[i] += A * dmt0 + dA * dh * mt0;
            }
          }
        }
      });
    }
  }

  // The start: the transposed instantaneous elastic solve at times[0]
  // (the unrelaxed, laterally uniform systems; no memory). Under lateral
  // moduli the loop's zero-length step at node 0 has done this.
  if (!op.HasLateralModuli()) {
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
    auto [duals, synth] = engineDuals(sigmaDual, psiDual, 0, true);
    auto adj =
        solverFor(op.ElasticLove()).SolveAdjoint(ev.state_, duals, sleOptions);
    agreeAdjoint(adj);
    if (!adj.converged) {
      throw std::runtime_error(
          "the adjoint elastic sea level solve did not converge");
    }
    directLoadKernels[0] = loadDualOf(adj, synth);
    // The first start dual's solution, where the material needs it.
    auto y0 = Viscoelastic3D::Solutions{};
    auto e0 = Tensor{};
    if (material || hasMaterialV) {
      addSurfaceDuals(sphForces, adj, sph);
      y0 = op.SolveUnrelaxed(sphForces, torForces);
      e0 = exit(y0, sph, tor);
    }
    if (second) {
      // The derivative dual state's transposed instantaneous solve (the
      // systems do not depend on the relaxation times), no measurement;
      // along a material v the unrelaxed systems' derivative on the first
      // start dual joins its forces.
      auto [dsph, dtor] = entry(coupling.Analyse(dddual), sph, tor);
      if (hasMaterialV) {
        auto [vsph, vtor] = materialForcesOf(e0, &y0, dmuV, nullptr, sph, tor);
        for (const auto& [l, f] : vsph) {
          auto& b = dsph.at(l);
          for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
        }
        for (const auto& [l, f] : vtor) {
          auto& b = dtor.at(l);
          for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
        }
      }
      const auto [dsigmaDual, dpsiDual] = channelDuals(dsph, units);
      auto [dduals, dsynth] = engineDuals(dsigmaDual, dpsiDual, 0, false);
      auto dadj = solverFor(op.ElasticLove())
                      .SolveAdjoint(ev.state_, dduals, sleOptions);
      agreeAdjoint(dadj);
      if (!dadj.converged) {
        throw std::runtime_error(
            "the adjoint elastic sea level solve of the derivative state "
            "did not converge");
      }
      dloadDuals[0] = loadDualOf(dadj, dsynth);
      if (material) {
        addSurfaceDuals(dsph, dadj, sph);
        const auto dy0 = op.SolveUnrelaxed(dsph, dtor);
        const auto x0 = op.SolveElastic(forwardState(0), loadsAt(0), tidesAt(0));
        const auto ex0 = exit(x0, sph, tor);
        const auto xv0 = tangentSolution(0, ex0, x0, nullptr);
        addMaterialPairing(dilatationOf(dy0, sph), dilatationOf(x0, sph),
                           exit(dy0, sph, tor), ex0, {}, nullptr, dKappaKernel,
                           dMuKernel);
        addMaterialPairing(dilatationOf(y0, sph), dilatationOf(xv0, sph), e0,
                           exit(xv0, sph, tor), {}, nullptr, dKappaKernel,
                           dMuKernel);
      }
    }
    if (material) {
      // The start's dual solution under the start duals and the start's
      // response duals, against the base start solution: every weight
      // one, no memory.
      const auto x0 = op.SolveElastic(forwardState(0), loadsAt(0), tidesAt(0));
      addMaterialPairing(dilatationOf(y0, sph), dilatationOf(x0, sph), e0,
                         exit(x0, sph, tor), {}, nullptr, kappaKernel, muKernel);
    }
  }
  op.ReleaseScratch();

  // The delta amplitudes and the distributional view, as the 1D discrete
  // sweep forms them.
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  for (const auto& [node, m] : measurements) {
    GRDSolver::AdjointForcing duals;
    duals.rotation = rotation;
    duals.oceanFunctionOverride = overrideAt(node);
    duals.seaLevelDual = m.seaLevelDual;
    duals.displacementDual = m.displacementDual;
    duals.potentialDual = m.potentialDual;
    duals.angularVelocityDual = m.angularVelocityDual;
    auto adj =
        solverFor(op.ElasticLove()).SolveAdjoint(ev.state_, duals, sleOptions);
    agreeAdjoint(adj);
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
      densityLoad[n][i] = (directLoadKernels[n][i] - lumped) / w[n];
    }
  }
  Kernels3D out;
  out.ice = detail::AssembleIceViews(
      C, solver.IceDensity(), times, std::move(directLoadKernels),
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
  KernelDerivatives3D derivatives;
  if (second) {
    // The derivative views as the 1D discrete sweep forms them: the
    // derivative load duals alone, no density view and no lumps.
    auto views = detail::AssembleIceViews(
        C, solver.IceDensity(), times, std::move(dloadDuals),
        std::vector<std::vector<double>>(times.size(),
                                         std::vector<double>(fieldSize, 0.0)),
        {}, oceanFunctions, rhoW);
    derivatives.ice.times.assign(times.begin(), times.end());
    derivatives.ice.thickness = std::move(views.thickness);
    derivatives.ice.initialSeaLevel = std::move(views.initialSeaLevel);
    derivatives.itauKernel3D = std::move(dItauKernel);
    derivatives.itauKernelSpherical.assign(K, std::vector<double>(nodeCount, 0.0));
    derivatives.logViscosityKernel3D.assign(K, std::vector<double>(values, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        derivatives.itauKernelSpherical[k][rn] = coupling.GridOf(rn).Integrate(
            std::span(derivatives.itauKernel3D[k])
                .subspan(coupling.FieldOffset(rn), coupling.FieldSizeOf(rn)));
      }
      // d/dv of -itau K: -(delta itau K + itau dK).
      for (auto i = std::size_t{0}; i < values; ++i) {
        derivatives.logViscosityKernel3D[k][i] =
            -((*ditau)[k][i] * out.itauKernel3D[k][i] +
              op.itau_[k][i] * derivatives.itauKernel3D[k][i]);
      }
    }
    if (material) {
      derivatives.kappaKernel3D = std::move(dKappaKernel);
      derivatives.muKernel3D = std::move(dMuKernel);
      derivatives.kappaKernelSpherical.assign(nodeCount, 0.0);
      derivatives.muKernelSpherical.assign(nodeCount, 0.0);
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        const auto at = coupling.FieldOffset(rn);
        const auto size = coupling.FieldSizeOf(rn);
        derivatives.kappaKernelSpherical[rn] = coupling.GridOf(rn).Integrate(
            std::span(derivatives.kappaKernel3D).subspan(at, size));
        derivatives.muKernelSpherical[rn] = coupling.GridOf(rn).Integrate(
            std::span(derivatives.muKernel3D).subspan(at, size));
      }
    }
  }
  // as densities per unit volume (the reductions as means over the sphere)
  detail::KernelsToFields(coupling, out);
  detail::KernelsToFields(coupling, derivatives);
  return {std::move(out), std::move(derivatives)};
}

}  // namespace sl3dpp
