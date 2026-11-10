#include "sl3dpp/Adjoint3D.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;

}  // namespace

std::vector<std::vector<double>> OceanFunctionsOf(
    std::span<const SeaLevelEvolution3D::Snapshot> snapshots) {
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

ForwardTrajectory3D ForwardTrajectory3D::FromSnapshots(
    std::span<const SeaLevelEvolution3D::Snapshot> snapshots) {
  ForwardTrajectory3D out;
  for (const auto& snapshot : snapshots) {
    out.states.push_back(snapshot.internalState);
    out.loads.push_back(snapshot.loadCoefficients);
    out.tides.push_back(snapshot.tideCoefficients);
  }
  return out;
}

AdjointSolver3D::AdjointSolver3D(const SeaLevelEvolution3D& evolution,
                                 Options options)
    : evolution_(&evolution), options_(options) {}

Kernels3D AdjointSolver3D::IceHistoryKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  if (options_.method == AdjointMethod::Discrete) {
    return DiscreteSweep(times, measurements, trajectory, oceanFunctions, {},
                         nullptr, nullptr)
        .first;
  }
  return Sweep(times, measurements, trajectory, oceanFunctions, {}, nullptr,
               nullptr)
      .first;
}

std::pair<Kernels3D, KernelDerivatives3D> AdjointSolver3D::SecondOrderKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const SeaLevelEvolution3D::Snapshot> tangent,
    const std::vector<std::vector<double>>& itau,
    const ForwardTrajectory3D& trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    const MaterialDirection3D& material) const {
  const auto& op = *evolution_->op_;
  const auto K = op.GetRheology().BranchCount();
  const auto values = op.Coupling().GridValueCount();
  for (const auto* field : {&material.kappa, &material.mu}) {
    if (!field->empty() && field->size() != values) {
      throw std::invalid_argument(
          "a material direction lives on the coupling's grid values");
    }
  }
  const auto* materialV = material.Empty() ? nullptr : &material;
  if (tangent.size() != times.size()) {
    throw std::invalid_argument(
        "the tangent-linear run must give one snapshot per time node");
  }
  for (const auto& snap : tangent) {
    if (!snap.internalState.has_value() ||
        !snap.internalState->strain.has_value()) {
      throw std::invalid_argument(
          "the tangent-linear run must be recorded at every node with its "
          "strain (TangentLinearSolver3D::Options::record)");
    }
  }
  auto ditau = itau;
  if (ditau.empty()) {
    ditau.assign(K, std::vector<double>(values, 0.0));
  } else if (ditau.size() != K) {
    throw std::invalid_argument(std::format(
        "an itau perturbation gives one array per branch: {} branches", K));
  } else {
    for (const auto& a : ditau) {
      if (a.size() != values) {
        throw std::invalid_argument(
            "an itau perturbation's arrays live on the coupling's grid "
            "values");
      }
    }
  }
  if (options_.method == AdjointMethod::Discrete) {
    return DiscreteSweep(times, measurements, &trajectory, oceanFunctions,
                         tangent, &ditau, materialV);
  }
  return Sweep(times, measurements, &trajectory, oceanFunctions, tangent,
               &ditau, materialV);
}

std::pair<Kernels3D, KernelDerivatives3D> AdjointSolver3D::Sweep(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const SeaLevelEvolution3D::Snapshot> tangent,
    const std::vector<std::vector<double>>* ditau,
    const MaterialDirection3D* materialV) const {
  const bool second = ditau != nullptr;
  const bool material = options_.material_kernels;
  const bool hasMaterialV = materialV != nullptr && !materialV->Empty();
  // v's delta mu over the grid values and delta mu / mu_U (the branch
  // weight of the itau pairing and the memory force's variation move
  // with it); empty without a mu direction.
  std::vector<double> dmuV, muRatioV;
  if (second && trajectory == nullptr) {
    throw std::invalid_argument(
        "the second-order sweep needs the recorded forward trajectory");
  }
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto& coupling = op.Coupling();
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
  const auto lmax = op.LMax();
  const auto K = op.GetRheology().BranchCount();
  const bool lateral = op.HasLateralModuli();
  const bool nested = op.GetOptions().solver ==
                      Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  const auto N = times.size() - 1;

  for (const auto& [node, m] : measurements) {
    if (node >= times.size()) {
      throw std::invalid_argument(
          std::format("measurement node {} is off the time grid", node));
    }
    for (const auto* f :
         {&m.seaLevelDual, &m.displacementDual, &m.potentialDual}) {
      if (!f->empty() && f->size() != fieldSize) {
        throw std::invalid_argument(std::format(
            "the measurement at node {} does not live on the solver grid",
            node));
      }
    }
  }
  if (trajectory != nullptr) {
    if (trajectory->states.size() != times.size()) {
      throw std::invalid_argument(
          "the trajectory must give one entry per time node");
    }
    if (!trajectory->Kept(0)) {
      throw std::invalid_argument(
          "the trajectory must keep the first node's state (a run with "
          "record_stride set keeps it): the states between kept ones are "
          "rebuilt forward from the kept one below");
    }
    if (trajectory->loads.size() != times.size() ||
        trajectory->tides.size() != times.size()) {
      throw std::invalid_argument(
          "the trajectory needs the recorded load and centrifugal "
          "coefficients (to regenerate each node's strain): build it from "
          "the recording run's snapshots");
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

  // Distributed over radius (Viscoelastic3D::Options::collectives), the
  // sweep steps every rank's own memory nodes as the forward run does, and
  // every rank solves the sea level equations alike; their transforms may
  // round differently, so rank 0's results serve all (as in
  // SeaLevelEvolution3D::March), and a solve that fails fails on all.
  const auto& collectives = op.GetOptions().collectives;
  auto agree = [&](auto& values) {
    if (!collectives.Distributed() || values.empty()) return;
    using Value =
        typename std::remove_reference_t<decltype(values)>::value_type;
    collectives.share(
        std::span(reinterpret_cast<double*>(values.data()),
                  values.size() * sizeof(Value) / sizeof(double)));
  };
  auto everywhere = [&](bool converged) {
    if (!collectives.Distributed()) return converged;
    auto failed = converged ? 0.0 : 1.0;
    collectives.max(std::span(&failed, 1));
    return failed == 0.0;
  };

  // The adjoint state, stepped by the forward coupled machinery in
  // reversed time with no ice load, and — for the second-order sweep —
  // its derivative along v, stepped the same way with the continuous
  // drive of the adjoint trajectory.
  struct Carried {
    Viscoelastic3D::State state;
    std::vector<Complex> sigma, psi;
  };
  auto vs = Carried{op.ZeroState(), grid.ZeroCoefficients(),
                    grid.ZeroCoefficients()};
  auto ds = Carried{op.ZeroState(), grid.ZeroCoefficients(),
                    grid.ZeroCoefficients()};
  auto& vstate = vs.state;

  auto psiStar = [&](const std::vector<double>& potentialDual) {
    auto d = solver.CentrifugalDual(grid.Expand(potentialDual));
    for (auto& v : d) v *= r2;
    return d;
  };

  // The instantaneous adjoint fingerprint of a measurement (unrelaxed,
  // laterally uniform — identical to the 1D sweep's jump).
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
    const auto resp =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, f, sleOptions);
    if (!everywhere(resp.converged)) {
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
    agree(out.xiA);
    agree(out.lam);
    agree(out.psiA);
    agree(out.zu);
    agree(out.zphi);
    return out;
  };

  // Yu et al. eq. (80) on the grid: the strain of the instantaneous
  // adjoint solution — spheroidal channels through the unrelaxed
  // REFERENCE systems, mapped to tensor components by the pipeline —
  // jumps the adjoint state pointwise with the laterally varying itau.
  const auto& reference = op.reference_;
  // The reference operator's lazily filled caches, under the operator's
  // lock (Viscoelastic3D::LockCaches).
  auto locked = [&](auto&& get) -> decltype(auto) {
    const auto lock = op.LockCaches();
    return get();
  };
  Viscoelastic3D::Solutions lumpSolutions;  // the jump's, when material
  GridCoupling::TensorField lumpStrain;
  // m += factor(x) d-tilde: the adjoint state over itau, its derivative
  // over delta itau (and over itau with the jump's own derivative).
  auto bump = [&](Viscoelastic3D::State& state,
                  const std::vector<std::vector<double>>& factors,
                  const GridCoupling::TensorField& dTilde) {
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& factor = factors[k];
      auto& m = state.m[k];
      for (auto c = 0; c < 2; ++c) {
        for (auto i = std::size_t{0}; i < m.c[c].size(); ++i) {
          m.c[c][i] += factor[i] * dTilde.c[c][i];
        }
      }
      for (auto i = std::size_t{0}; i < m.r.size(); ++i) {
        m.r[i] += factor[i] * dTilde.r[i];
      }
    }
    state.Invalidate();
  };
  auto applyJump = [&](const Jump& j) {
    auto spectral = coupling.ZeroSpectral();
    const auto split = !j.zu.empty() || !j.zphi.empty();
    for (auto l = 0; l <= lmax; ++l) {
      const auto& units = locked([&]() -> const std::vector<double>& {
        return reference.UnitResponses(nullptr, l);
      });
      const auto ndof = units.size() / 2;
      auto x =
          std::vector<Complex>(ndof * (static_cast<std::size_t>(l) + 1), 0.0);
      const double* splitUnits =
          split ? locked([&]() -> const std::vector<double>& {
                    return reference.SplitUnitResponses(nullptr, l);
                  }).data()
                : nullptr;
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        const Complex lamAmp = ev.toMatDensity_ * j.lam[i];
        const Complex psiAmp = ev.toMatPotential_ * j.psiA[i];
        const Complex zuAmp =
            j.zu.empty() ? 0.0 : ev.toMatDensity_ * j.zu[i];
        const Complex zphiAmp =
            j.zphi.empty() ? 0.0 : ev.toMatDensity_ * j.zphi[i];
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          Complex v = units[d] * lamAmp + units[ndof + d] * psiAmp;
          if (splitUnits != nullptr) {
            v += splitUnits[d] * zuAmp + splitUnits[ndof + d] * zphiAmp;
          }
          x[static_cast<std::size_t>(m) * ndof + d] = v;
        }
      }
      coupling.AddStrainOfSolution(spectral, l, op.Spheroidal(l), x);
      if (material || hasMaterialV) lumpSolutions.spheroidal[l] = std::move(x);
    }
    const auto dTilde = coupling.Synthesise(spectral);
    if (material || hasMaterialV) lumpStrain = dTilde;
    bump(vs.state, op.itau_, dTilde);
    if (second) bump(ds.state, *ditau, dTilde);
  };

  // The memory-only adjoint equilibrium of the current state through
  // the unrelaxed reference systems (laterally uniform, no iteration).
  auto momentOf = [&](const Viscoelastic3D::Solutions& solutions) {
    const auto& deg = locked([&]() -> const auto& { return reference.Deg(0); });
    const auto& x = solutions.spheroidal.at(0);
    // Distributed, each rank's share of the solution, summed.
    auto column = std::vector<double>(x.size());
    for (auto d = std::size_t{0}; d < x.size(); ++d) column[d] = x[d].real();
    op.Spheroidal(0).MaskToShare(column);
    auto sum = 0.0;
    for (auto d = 0; d < deg.ndof; ++d) {
      sum += deg.momentRow[d] * column[d];
    }
    if (collectives.Distributed()) collectives.sum(std::span(&sum, 1));
    return sum;
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
  auto smoothEquilibrium = [&](Carried& carried, std::size_t node,
                               const Viscoelastic3D::Solutions* extra) {
    auto solutions = op.SolveElastic(carried.state);
    if (extra != nullptr) addSolutions(solutions, *extra);
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(node);
    f.displacementResponse =
        ev.Coefficients(op.Surface(solutions, Component::U), ev.toNdLength_);
    f.potentialResponse = ev.Coefficients(
        op.Surface(solutions, Component::Phi), ev.toNdPotential_);
    f.inertiaMomentChange = momentOf(solutions) * ev.toNdMoment_;
    auto resp =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, f, sleOptions);
    if (!everywhere(resp.converged)) {
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
    agree(carried.sigma);
    agree(carried.psi);
    agree(resp.seaLevelChange);
    return resp.seaLevelChange;
  };

  // The node-weight split across observation discontinuities.
  const double perTime = ev.toMatTime_;
  std::vector<double> wEarly(times.size(), 0.0), wLate(times.size(), 0.0);
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    wEarly[n] = 0.5 * (times[n] - times[n - 1]);
  }
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    wLate[n] = 0.5 * (times[n + 1] - times[n]);
  }

  // The viscosity-kernel accumulation: the null-basis contraction of
  // the adjoint state with the forward d - m, whose metric (12, 8, 4 on
  // the spin 0, -1, -2 slots, times r^2) reproduces the 1D force
  // weights exactly in the angle-uniform limit; branch modulus, radial
  // quadrature, the covariant pairing ratio (Adjoint.cpp) and the
  // division by the laterally varying itau complete it.
  const auto values = coupling.GridValueCount();
  const auto nodeCount = static_cast<std::size_t>(coupling.NodeCount());
  auto itauKernel = std::vector<std::vector<double>>{};
  const auto& matMesh = op.GetRheology().BaseMaterial().Mesh();
  const double aMat =
      matMesh.Radius(matMesh.ElementCount() - 1, matMesh.NGll() - 1);
  const double pairing =
      r2 * ev.toNdLength_ /
      (aMat * aMat * ev.toMatDensity_ *
       op.GetRheology().BaseMaterial().SurfaceGravity());
  auto dItauKernel = std::vector<std::vector<double>>{};
  if (trajectory != nullptr) {
    itauKernel.assign(K, std::vector<double>(values, 0.0));
    if (second) dItauKernel.assign(K, std::vector<double>(values, 0.0));
  }
  // The forward state at a node. A kept one is read as recorded. The
  // states between two kept ones are rebuilt when the sweep, coming down,
  // first needs one of them: forward steps from the kept state below under
  // the recorded loads, with the forward run's own iteration, so they
  // agree with the run to its tolerance. They are held, their strain as
  // coefficients only (the top one's grid strain kept), until the sweep
  // has passed below them: N / stride kept states plus one stride's worth.
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
    // a spilled kept node is fetched into the cache and dropped, like a
    // rebuilt one, once the sweep has passed below it
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
  // The forward strain at a node: the state's cache if it has one, else
  // from its coefficients, else one elastic solve under the node's
  // recorded loads (recordings keep only the internal variables); kept
  // for the node's two accumulations.
  auto strainNode = times.size();
  GridCoupling::TensorField regenerated;
  auto forwardStrain =
      [&](std::size_t node) -> const GridCoupling::TensorField& {
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
  // The second-order sweep's tangent-linear state at a node, with its
  // strain (recorded at every node).
  auto tangentState = [&](std::size_t node) -> const Viscoelastic3D::State& {
    return *tangent[node].internalState;
  };
  auto accumulateItau = [&](std::size_t node, double weight) {
    if (trajectory == nullptr || weight == 0.0) return;
    const auto& dFwd = forwardStrain(node);
    const auto& fwd = forwardState(node);
    const auto* tl = second ? &tangentState(node) : nullptr;
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& itau = op.itau_[k];
      const auto& mu = op.muBranch_[k];
      const auto& mA = vstate.m[k];
      const auto& mF = fwd.m[k];
      auto& kernel = itauKernel[k];
      // The derivative pairing's pieces: the derivative adjoint state, v's
      // tangent state and strain, and v's delta itau.
      const auto* dA = second ? &ds.state.m[k] : nullptr;
      const auto* mT = second ? &tl->m[k] : nullptr;
      const auto* dT = second ? &*tl->strain : nullptr;
      const auto* delta = second ? &(*ditau)[k] : nullptr;
      auto* dKernel = second ? &dItauKernel[k] : nullptr;
      // Node by node, each writing only its own grid values.
      detail::ForEach(
          op.options_.parallel, nodeCount, [&](std::size_t rn) {
            if (mu[rn] <= 0.0) return;
            const auto r = coupling.RadiusOfNode(static_cast<int>(rn));
            const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
            const double factorNode = pairing * weight * mu[rn] * wq * r * r;
            const auto base = coupling.FieldOffset(static_cast<int>(rn));
            const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
            for (auto x = std::size_t{0}; x < size; ++x) {
              const auto i = base + x;
              if (itau[i] <= 0.0) continue;
              // the branch modulus at the grid value, under lateral shear
              const double factor = factorNode * op.ShearFactor(i);
              // d - m of the forward state, against the adjoint state:
              // the reduced-storage contraction (derived spins by
              // reality).
              const auto dm2 = dFwd.c[0][i] - mF.c[0][i];
              const auto dm1 = dFwd.c[1][i] - mF.c[1][i];
              const double dm0 = dFwd.r[i] - mF.r[i];
              auto contract = [&](const GridCoupling::TensorField& a,
                                  const Complex& b2, const Complex& b1,
                                  double b0) {
                return 12.0 * a.r[i] * b0 +
                       8.0 * (a.c[1][i] * std::conj(b1)).real() +
                       4.0 * (a.c[0][i] * std::conj(b2)).real();
              };
              const double contraction = contract(mA, dm2, dm1, dm0);
              kernel[i] += factor * contraction / itau[i];
              if (!second) continue;
              // Its derivative along v: the derivative state against the
              // forward d - m, the adjoint state against the tangent
              // d - m, less (delta itau / itau) times the pairing itself.
              const auto vdm2 = dT->c[0][i] - mT->c[0][i];
              const auto vdm1 = dT->c[1][i] - mT->c[1][i];
              const double vdm0 = dT->r[i] - mT->r[i];
              const double dContraction =
                  contract(*dA, dm2, dm1, dm0) + contract(mA, vdm2, vdm1, vdm0) -
                  (*delta)[i] / itau[i] * contraction +
                  (muRatioV.empty() ? 0.0 : muRatioV[i] * contraction);
              (*dKernel)[i] += factor * dContraction / itau[i];
            }
          });
    }
  };

  // -- the material pairing (continuous; docs/adjoints.md)
  // The continuous adjoint displacement at a node (the unrelaxed solve
  // under the adjoint memory and closed load) paired with the forward
  // solution and memory there: the energy contraction of the grid
  // strains for the shear and of the dilatations for the bulk, under the
  // itau kernel's time quadrature and units constant; at an observation
  // the lumped instantaneous solution pairs without a time weight.
  std::vector<double> kappaKernel, muKernel, dKappaKernel, dMuKernel;
  auto fractions = std::vector<std::vector<double>>{};
  if (material) {
    if (trajectory == nullptr) {
      throw std::invalid_argument(
          "the material kernels need the recorded forward trajectory");
    }
    kappaKernel.assign(values, 0.0);
    muKernel.assign(values, 0.0);
    if (second) {
      dKappaKernel.assign(values, 0.0);
      dMuKernel.assign(values, 0.0);
    }
  }
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
  const auto unrelaxedSph = [&](int l) -> const DegreeOperator& {
    return op.Spheroidal(l);
  };
  const auto unrelaxedTor = [&](int l) -> const ToroidalOperator& {
    return op.Toroidal(l);
  };
  auto dilatationCoefficients = [&](const Viscoelastic3D::Solutions& x) {
    auto scalar = std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
    for (const auto& [l, xl] : x.spheroidal) {
      coupling.AddDilatationOfSolution(scalar, l, op.Spheroidal(l), xl);
    }
    return scalar;
  };
  auto dilatationOf = [&](const Viscoelastic3D::Solutions& x) {
    return coupling.SynthesiseScalar(dilatationCoefficients(x));
  };
  using Tensor = GridCoupling::TensorField;
  // v's forces on a solution with its grid strain (the material tangent's
  // Born force: the shear weight on the deviatoric strain, pointwise, and
  // the bulk term through the scalar companion), as an analysed tensor and
  // per-degree bulk forces (what a step's AddForce takes) — `shear` the
  // shear weight (delta mu_eff of a step, delta mu of the unrelaxed form)
  // and an optional memory force's variation tensor.
  auto materialStepForce = [&](const Tensor& strain,
                               const Viscoelastic3D::Solutions* x,
                               std::span<const double> shear,
                               const Tensor* memoryForce) {
    auto gridForce = coupling.ZeroTensor();
    detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
      const auto w = shear.empty() ? 0.0 : -shear[i];
      for (auto c = 0; c < 2; ++c) {
        gridForce.c[c][i] =
            w * strain.c[c][i] + (memoryForce ? memoryForce->c[c][i] : Complex{});
      }
      gridForce.r[i] = w * strain.r[i] + (memoryForce ? memoryForce->r[i] : 0.0);
    });
    auto analysed = coupling.Analyse(gridForce);
    auto bulk = std::map<int, std::vector<Complex>>{};
    if (hasMaterialV && !materialV->kappa.empty() && x != nullptr) {
      // Through gridForce's real component, analysed (done with it), and
      // back over the dilatation's coefficients.
      auto scalar = dilatationCoefficients(*x);
      auto& field = gridForce.r;
      coupling.SynthesiseScalar(scalar, field);
      for (auto i = std::size_t{0}; i < values; ++i) field[i] *= -materialV->kappa[i];
      coupling.AnalyseScalar(field, scalar);
      bulk = coupling.DilatationForce(scalar, unrelaxedSph);
    }
    return std::pair(std::move(analysed), std::move(bulk));
  };
  // The unrelaxed systems' response to v's forces (every weight one):
  // -(dK_U) x plus the memory force's variation.
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
  auto unrelaxedResponseV = [&](const Tensor& strain,
                                const Viscoelastic3D::Solutions& x,
                                const std::vector<Tensor>* memory) {
    auto memoryForce = std::optional<Tensor>{};
    if (memory != nullptr && !dmuV.empty()) memoryForce = memoryVariationV(*memory);
    auto [analysed, bulk] =
        materialStepForce(strain, &x, dmuV, memoryForce ? &*memoryForce : nullptr);
    auto forces = coupling.SpheroidalForce(analysed, unrelaxedSph);
    for (const auto& [l, f] : bulk) {
      auto& b = forces.at(l);
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
    }
    return op.SolveUnrelaxed(forces, coupling.ToroidalForce(analysed, unrelaxedTor));
  };
  auto strainOf = [&](const Viscoelastic3D::Solutions& x) {
    return coupling.Synthesise(
        op.StrainCoefficients(x, unrelaxedSph, unrelaxedTor));
  };
  // v's full tangent solutions at a node from its recorded snapshot: the
  // unrelaxed solve under the snapshot's loads and memory, plus v's own
  // unrelaxed material force's response on the base solution (its strain
  // and memory) there.
  auto tangentSolution = [&](std::size_t node, const Tensor& baseStrain,
                             const Viscoelastic3D::Solutions& baseX,
                             const std::vector<Tensor>* baseMemory) {
    const auto& snap = tangent[node];
    auto x = op.SolveElastic(*snap.internalState,
                             ev.Columns(snap.loadCoefficients, ev.toMatDensity_),
                             ev.Columns(snap.tideCoefficients, ev.toMatPotential_));
    if (hasMaterialV) {
      addSolutions(x, unrelaxedResponseV(baseStrain, baseX, baseMemory));
    }
    return x;
  };
  // The material pairing of adjoint degree solutions (with their grid
  // strain) against a solution x with its strain dX and a memory state,
  // scaled by `weight`, into a pair of kernels.
  auto pairMaterialWith = [&](const Viscoelastic3D::Solutions& lambda,
                              const Tensor& eLambda,
                              const Viscoelastic3D::Solutions& x,
                              const Tensor& dX, const std::vector<Tensor>& mem,
                              double weight, std::vector<double>& kappaTo,
                              std::vector<double>& muTo) {
    if (weight == 0.0) return;
    const auto Dy = dilatationOf(lambda);
    const auto Dx = dilatationOf(x);
    detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
      const auto base = coupling.FieldOffset(static_cast<int>(rn));
      const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
      const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
      const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
      for (auto xg = std::size_t{0}; xg < size; ++xg) {
        const auto i = base + xg;
        auto contraction = [&](const Tensor& a, const Tensor& b) {
          return 12.0 * a.r[i] * b.r[i] +
                 8.0 * (a.c[1][i] * std::conj(b.c[1][i])).real() +
                 4.0 * (a.c[0][i] * std::conj(b.c[0][i])).real();
        };
        kappaTo[i] -= weight * wq * Dy[i] * Dx[i];
        auto shear = -contraction(eLambda, dX);
        for (auto k = std::size_t{0}; k < K; ++k) {
          if (fractions[k][rn] == 0.0) continue;
          shear += fractions[k][rn] * contraction(eLambda, mem[k]);
        }
        muTo[i] += weight * wq * rr * rr * shear;
      }
    });
  };
  // Both pairings of an adjoint solution and its derivative along v at
  // a node: the kernels (lambda against the forward solution and memory),
  // and their derivative (the derivative against the forward, lambda
  // against v's tangent solution, strain and memory).
  auto pairBoth = [&](const Viscoelastic3D::Solutions& lambda,
                      const Tensor& eLambda,
                      const Viscoelastic3D::Solutions* dlambda,
                      const Tensor* eDlambda, std::size_t node,
                      double weight) {
    if (weight == 0.0) return;
    auto fwd = forwardState(node);  // a copy: the cache may drop it below
    const auto dFwd = forwardStrain(node);
    const auto x = op.SolveElastic(fwd, loadsAt(node), tidesAt(node));
    pairMaterialWith(lambda, eLambda, x, dFwd, fwd.m, weight, kappaKernel, muKernel);
    if (!second) return;
    pairMaterialWith(*dlambda, *eDlambda, x, dFwd, fwd.m, weight, dKappaKernel,
                     dMuKernel);
    const auto xv = tangentSolution(node, dFwd, x, node == 0 ? nullptr : &fwd.m);
    pairMaterialWith(lambda, eLambda, xv, *tangentState(node).strain,
                     tangentState(node).m, weight, dKappaKernel, dMuKernel);
  };
  auto pairMaterial = [&](const Viscoelastic3D::Solutions& lambda,
                          const GridCoupling::TensorField& eLambda,
                          std::size_t node, double weight) {
    if (weight == 0.0) return;
    const auto& fwd = forwardState(node);
    const auto& dFwd = forwardStrain(node);
    const auto x = op.SolveElastic(fwd, loadsAt(node), tidesAt(node));
    const auto Dy = dilatationOf(lambda);
    const auto Dx = dilatationOf(x);
    detail::ForEach(op.options_.parallel, nodeCount, [&](std::size_t rn) {
      const auto base = coupling.FieldOffset(static_cast<int>(rn));
      const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
      const auto rr = coupling.RadiusOfNode(static_cast<int>(rn));
      const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
      for (auto xg = std::size_t{0}; xg < size; ++xg) {
        const auto i = base + xg;
        auto contraction = [&](const GridCoupling::TensorField& a,
                               const GridCoupling::TensorField& b) {
          return 12.0 * a.r[i] * b.r[i] +
                 8.0 * (a.c[1][i] * std::conj(b.c[1][i])).real() +
                 4.0 * (a.c[0][i] * std::conj(b.c[0][i])).real();
        };
        kappaKernel[i] -= weight * wq * Dy[i] * Dx[i];
        auto shear = -contraction(eLambda, dFwd);
        for (auto k = std::size_t{0}; k < K; ++k) {
          if (fractions[k][rn] == 0.0) continue;
          shear += fractions[k][rn] * contraction(eLambda, fwd.m[k]);
        }
        muKernel[i] += weight * wq * rr * rr * shear;
      }
    });
  };
  // The full solution of a carried state under its closed load.
  auto fullOf = [&](const Carried& carried) {
    return op.SolveElastic(carried.state, ev.Columns(carried.sigma, ev.toMatDensity_),
                           ev.Columns(carried.psi, ev.toMatPotential_));
  };
  // The regular adjoint displacement at a node, under the current adjoint
  // memory and closed load, and its derivative: the derivative state's
  // plus v's unrelaxed force's response on the adjoint displacement and
  // memory.
  auto accumulateMaterial = [&](std::size_t node, double weight) {
    if (!material || weight == 0.0) return;
    const auto lambda = fullOf(vs);
    const auto e = strainOf(lambda);
    if (!second) {
      pairMaterial(lambda, e, node, pairing * weight);
      return;
    }
    auto dlambda = fullOf(ds);
    if (hasMaterialV) {
      addSolutions(dlambda, unrelaxedResponseV(e, lambda, &vs.state.m));
    }
    const auto de = strainOf(dlambda);
    pairBoth(lambda, e, &dlambda, &de, node, pairing * weight);
  };

  std::vector<std::vector<double>> densityLate(times.size()),
      densityEarly(times.size()), dDensityLate(times.size()),
      dDensityEarly(times.size());
  auto lumps = std::map<std::size_t, std::vector<double>>{};
  auto dlumps = std::map<std::size_t, std::vector<double>>{};

  // The derivative along a material v of the jump: its fingerprint moves
  // with the unrelaxed table, carried as the response channel of v's
  // unrelaxed variation on the jump solution (the forward start's
  // pattern); returns the derivative solution and the fingerprint's
  // derivative (the derivative lump).
  auto jumpDerivative = [&](std::size_t node) {
    auto response = unrelaxedResponseV(lumpStrain, lumpSolutions, nullptr);
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.oceanFunctionOverride = overrideAt(node);
    f.displacementResponse =
        ev.Coefficients(op.Surface(response, Component::U), ev.toNdLength_);
    f.potentialResponse =
        ev.Coefficients(op.Surface(response, Component::Phi), ev.toNdPotential_);
    f.inertiaMomentChange = momentOf(response) * ev.toNdMoment_;
    const auto resp =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(
          "the derivative of the instantaneous adjoint fingerprint did not "
          "converge");
    }
    const auto& Cn = oceanAt(node);
    auto dlam = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      dlam[i] = rhoW * Cn[i] * resp.seaLevelChange[i];
    }
    const auto dlamC = grid.Expand(dlam);
    for (auto l = 0; l <= lmax; ++l) {
      const auto& units = locked([&]() -> const std::vector<double>& {
        return reference.UnitResponses(nullptr, l);
      });
      const auto ndof = units.size() / 2;
      auto& x = response.spheroidal[l];
      for (auto m = 0; m <= l; ++m) {
        const Complex amp = ev.toMatDensity_ * dlamC[SurfaceGrid::Index(l, m)];
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          x[static_cast<std::size_t>(m) * ndof + d] += units[d] * amp;
        }
      }
    }
    return std::pair(std::move(response), resp.seaLevelChange);
  };

  // One reversed coupled step of a carried state under its closure, with
  // an optional memory drive; returns the step's sea level (the late
  // density at the node stepped to).
  // What v's material force on the derivative state's step is built
  // from: the first state's step END strain and full solution (the bulk
  // term's), and its decayed memory.
  struct MaterialSource {
    Tensor strain;
    Viscoelastic3D::Solutions full;
    std::vector<Tensor> memory;
  };
  auto reversedStep = [&](Carried& carried, double dt, std::size_t to,
                          const std::vector<GridCoupling::TensorField>* drive,
                          const MaterialSource* source, MaterialSource* record) {
    auto step = op.StartStep(carried.state, dt, scheme,
                             ev.Columns(carried.sigma, ev.toMatDensity_),
                             ev.Columns(carried.psi, ev.toMatPotential_));
    if (drive != nullptr) step.AddMemory(*drive);
    if (source != nullptr) {
      // -(dK_eff along v) on the first state's step solution, with the
      // step's own lateral weights, and the memory force's variation of
      // its decayed memory (the material tangent's forces).
      auto shear = std::vector<double>{};
      if (!dmuV.empty()) {
        shear.assign(values, 0.0);
        for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
          const auto base = coupling.FieldOffset(static_cast<int>(rn));
          const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
          for (auto x = std::size_t{0}; x < size; ++x) {
            auto factor = 1.0;
            for (auto k = std::size_t{0}; k < K; ++k) {
              factor -= fractions[k][rn] * step.beta_[k][base + x];
            }
            shear[base + x] = dmuV[base + x] * factor;
          }
        }
      }
      auto memoryForce = std::optional<Tensor>{};
      if (!dmuV.empty()) memoryForce = memoryVariationV(source->memory);
      auto [analysed, bulk] = materialStepForce(
          source->strain, &source->full, shear, memoryForce ? &*memoryForce : nullptr);
      step.AddForce(analysed, bulk);
    }
    if (record != nullptr) record->memory = step.memory_;
    const auto& effectiveSolver = solverFor(step.EffectiveLove());
    const auto& Cn = oceanAt(to);
    SeaLevelSolver::Response response;
    auto closure = [&](const Viscoelastic3D::Solutions& responses) {
      SeaLevelSolver::GeneralisedForcing coupled;
      coupled.displacementResponse = ev.Coefficients(
          op.Surface(responses, Component::U), ev.toNdLength_);
      coupled.potentialResponse = ev.Coefficients(
          op.Surface(responses, Component::Phi), ev.toNdPotential_);
      coupled.inertiaMomentChange = step.InertiaMoment(responses) * ev.toNdMoment_;
      coupled.rotation = rotation;
      coupled.oceanFunctionOverride = overrideAt(to);
      response = effectiveSolver.SolveGeneralised(ev.state_, coupled, sleOptions);
      auto sig = std::vector<double>(fieldSize);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        sig[i] = rhoW * Cn[i] * response.seaLevelChange[i];
      }
      carried.sigma = grid.Expand(sig);
      carried.psi = rotation != RotationalFeedback::None
                        ? solver.CentrifugalCoefficients(response.angularVelocity)
                        : grid.ZeroCoefficients();
      agree(carried.sigma);
      agree(carried.psi);
      agree(response.seaLevelChange);
      return std::pair{ev.Columns(carried.sigma, ev.toMatDensity_),
                       ev.Columns(carried.psi, ev.toMatPotential_)};
    };
    const Viscoelastic3D::IterationControl control{
        .max_iterations = options_.max_coupled_iterations,
        .rtol = options_.coupled_rtol,
        .anderson_memory = options_.anderson_memory,
        .anderson_rate = options_.anderson_rate};
    auto [strain, info] =
        nested ? step.Nest(closure, control)
               : step.Converge(
                     [&](const GridCoupling::SpectralTensor& s) {
                       auto responses = step.Respond(s);
                       const auto [load, tide] = closure(responses);
                       step.AddChannels(responses, load, tide);
                       return step.StrainOf(responses);
                     },
                     control);
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the reversed lateral fixed point at node {} did not converge "
          "(residual {} after {} passes)",
          to, info.residual, info.iterations));
    }
    op.Recycle(std::exchange(carried.state,
                             std::move(step).Finish(std::move(strain))));
    if (record != nullptr) {
      record->strain = *carried.state.strain;
      record->full = fullOf(carried);
    }
    return response.seaLevelChange;
  };


  // Under lateral moduli an instantaneous response is a coupled fixed
  // point too: a zero-length step from a zero state (Viscoelastic3D::
  // StartStep), forced by given forces (an analysed tensor with bulk
  // forces, or per-degree spheroidal forces), closed by the sea level
  // engine under the base forcing with the responses as its channels;
  // gives the total solutions, their grid strain and the engine's
  // response.
  struct Instantaneous {
    Viscoelastic3D::Solutions x;
    Tensor strain;
    SeaLevelSolver::Response response;
  };
  auto instantaneous = [&](std::size_t node,
                           const SeaLevelSolver::GeneralisedForcing& base,
                           const GridCoupling::SpectralTensor* analysed,
                           const std::map<int, std::vector<Complex>>* bulk,
                           const std::map<int, std::vector<Complex>>* split) {
    auto step = op.StartStep(op.ZeroState(), 0.0, scheme);
    if (analysed != nullptr) {
      step.AddForce(*analysed, bulk != nullptr
                                   ? *bulk
                                   : std::map<int, std::vector<Complex>>{});
    }
    if (split != nullptr) step.AddForce(coupling.ZeroSpectral(), *split);
    const auto& elastic = solverFor(step.EffectiveLove());
    const auto& Cn = oceanAt(node);
    Instantaneous out;
    auto closure = [&](const Viscoelastic3D::Solutions& responses) {
      auto f = base;
      f.displacementResponse = ev.Coefficients(
          op.Surface(responses, Component::U), ev.toNdLength_);
      f.potentialResponse = ev.Coefficients(
          op.Surface(responses, Component::Phi), ev.toNdPotential_);
      f.inertiaMomentChange = step.InertiaMoment(responses) * ev.toNdMoment_;
      out.response = elastic.SolveGeneralised(ev.state_, f, sleOptions);
      auto lamGrid = std::vector<double>(fieldSize);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        lamGrid[i] = (base.directLoad.empty() ? 0.0 : base.directLoad[i]) +
                     rhoW * Cn[i] * out.response.seaLevelChange[i];
      }
      const auto lam = grid.Expand(lamGrid);
      const auto psiA =
          rotation != RotationalFeedback::None
              ? solver.CentrifugalCoefficients(out.response.angularVelocity)
              : grid.ZeroCoefficients();
      return std::pair{ev.Columns(lam, ev.toMatDensity_),
                       ev.Columns(psiA, ev.toMatPotential_)};
    };
    const Viscoelastic3D::IterationControl control{
        .max_iterations = options_.max_coupled_iterations,
        .rtol = options_.coupled_rtol,
        .anderson_memory = options_.anderson_memory,
        .anderson_rate = options_.anderson_rate};
    auto [strain, info] =
        nested ? step.Nest(closure, control, &out.x)
               : step.Converge(
                     [&](const GridCoupling::SpectralTensor& s) {
                       auto responses = step.Respond(s);
                       const auto [load, tide] = closure(responses);
                       step.AddChannels(responses, load, tide);
                       out.x = responses;
                       return step.StrainOf(responses);
                     },
                     control);
    if (!info.converged || !out.response.converged) {
      throw std::runtime_error(std::format(
          "the instantaneous adjoint fixed point at node {} did not converge",
          node));
    }
    out.strain = coupling.Synthesise(strain);
    return out;
  };
  // The jump under lateral moduli: the measurement's duals as the base
  // forcing (its sea level and the rotational pieces through the engine,
  // its displacement and potential duals as the split loads' forces),
  // the instantaneous solution and its strain bumping the adjoint state.
  auto lateralJump = [&](const Measurement& m, std::size_t node) {
    SeaLevelSolver::GeneralisedForcing base;
    base.rotation = rotation;
    base.oceanFunctionOverride = overrideAt(node);
    if (!m.seaLevelDual.empty()) base.directLoad = m.seaLevelDual;
    if (rotation != RotationalFeedback::None) {
      auto avc = std::array<double, 3>{0.0, 0.0, 0.0};
      for (auto j = 0; j < 3; ++j) avc[j] = -g * m.angularVelocityDual[j];
      if (!m.potentialDual.empty()) {
        const auto ps = psiStar(m.potentialDual);
        for (auto j = 0; j < 3; ++j) avc[j] += g * ps[j];
      }
      base.angularMomentumChange = avc;
    }
    auto split = std::map<int, std::vector<Complex>>{};
    if (!m.displacementDual.empty() || !m.potentialDual.empty()) {
      auto zu = grid.ZeroCoefficients(), zphi = grid.ZeroCoefficients();
      if (!m.displacementDual.empty()) {
        auto zetaU = std::vector<double>(fieldSize);
        for (auto i = std::size_t{0}; i < fieldSize; ++i) {
          zetaU[i] = -m.displacementDual[i];
        }
        zu = grid.Expand(zetaU);
      }
      if (!m.potentialDual.empty()) {
        auto zetaPhi = std::vector<double>(fieldSize);
        for (auto i = std::size_t{0}; i < fieldSize; ++i) {
          zetaPhi[i] = -g * m.potentialDual[i];
        }
        zphi = grid.Expand(zetaPhi);
      }
      for (auto l = 0; l <= lmax; ++l) {
        const auto& system = op.Spheroidal(l);
        const auto ndof = static_cast<std::size_t>(system.DofCount());
        const auto force = system.Load(LoadPart::Force);
        const auto potential = system.Load(LoadPart::Potential);
        auto b = std::vector<Complex>(ndof * (static_cast<std::size_t>(l) + 1), 0.0);
        for (auto mm = 0; mm <= l; ++mm) {
          const auto i = SurfaceGrid::Index(l, mm);
          const Complex zuAmp = ev.toMatDensity_ * zu[i];
          const Complex zphiAmp = ev.toMatDensity_ * zphi[i];
          for (auto d = std::size_t{0}; d < ndof; ++d) {
            b[static_cast<std::size_t>(mm) * ndof + d] =
                force[d] * zuAmp + potential[d] * zphiAmp;
          }
        }
        split.emplace(l, std::move(b));
      }
    }
    auto inst = instantaneous(node, base, nullptr, nullptr,
                              split.empty() ? nullptr : &split);
    lumps.emplace(node, inst.response.seaLevelChange);
    if (material || hasMaterialV) {
      lumpSolutions = inst.x;
      lumpStrain = inst.strain;
    }
    bump(vs.state, op.itau_, inst.strain);
    if (second) bump(ds.state, *ditau, inst.strain);
  };
  // Its derivative along a material v: v's unrelaxed force on the jump
  // solution as the forces of the zero-length step.
  auto lateralJumpDerivative = [&](std::size_t node) {
    auto [analysed, bulk] =
        materialStepForce(lumpStrain, &lumpSolutions, dmuV, nullptr);
    SeaLevelSolver::GeneralisedForcing base;
    base.rotation = rotation;
    base.oceanFunctionOverride = overrideAt(node);
    auto inst = instantaneous(node, base, &analysed, &bulk, nullptr);
    return std::pair(std::move(inst.x), inst.response.seaLevelChange);
  };

  auto observe = [&](std::size_t node) {
    const auto found = measurements.find(node);
    if (found == measurements.end()) return;
    if (lateral) {
      lateralJump(found->second, node);
    } else {
      const auto j = jump(found->second, node);
      lumps.emplace(node, j.xiA);
      applyJump(j);
    }
    auto dxJ = Viscoelastic3D::Solutions{};
    auto deJ = Tensor{};
    if (hasMaterialV) {
      auto [dx, dxi] = lateral ? lateralJumpDerivative(node) : jumpDerivative(node);
      deJ = strainOf(dx);
      bump(ds.state, op.itau_, deJ);
      dlumps.emplace(node, std::move(dxi));
      dxJ = std::move(dx);
    } else if (material && second) {
      dxJ = lumpSolutions;
      for (auto& [l, x] : dxJ.spheroidal) std::ranges::fill(x, Complex{});
      deJ = coupling.ZeroTensor();
    }
    if (material) {
      pairBoth(lumpSolutions, lumpStrain, &dxJ, &deJ, node, pairing);
    }
    // The equilibria under lateral moduli are zero-length reversed steps
    // (their fixed points close the instantaneous lateral response; the
    // derivative state's carries v's material source as a step's does).
    auto early = lateral ? reversedStep(vs, 0.0, node, nullptr, nullptr, nullptr)
                         : smoothEquilibrium(vs, node, nullptr);
    if (node > 0) densityEarly[node] = std::move(early);
    if (second && lateral) {
      auto source = std::optional<MaterialSource>{};
      if (hasMaterialV) {
        const auto lambda = fullOf(vs);
        source.emplace();
        source->strain = strainOf(lambda);
        source->full = lambda;
        source->memory = vs.state.m;
      }
      auto dearly =
          reversedStep(ds, 0.0, node, nullptr, source ? &*source : nullptr, nullptr);
      if (node > 0) dDensityEarly[node] = std::move(dearly);
    } else if (second) {
      auto extra = Viscoelastic3D::Solutions{};
      if (hasMaterialV) {
        const auto lambda = fullOf(vs);
        extra = unrelaxedResponseV(strainOf(lambda), lambda, &vs.state.m);
      }
      auto dearly = smoothEquilibrium(ds, node, hasMaterialV ? &extra : nullptr);
      if (hasMaterialV) {
        // The derivative state's strain after the jump carries v's
        // material response too (what StartStep would regenerate lacks
        // it): the forward tangent's start-strain rule.
        op.PrimeStrain(ds.state, ev.Columns(ds.sigma, ev.toMatDensity_),
                       ev.Columns(ds.psi, ev.toMatPotential_));
        op.AddStrainCoefficients(*ds.state.strainCoefficients, extra,
                                 unrelaxedSph, unrelaxedTor);
        ds.state.strain = coupling.Synthesise(*ds.state.strainCoefficients);
      }
      if (node > 0) dDensityEarly[node] = std::move(dearly);
    }
  };

  // The reversed sweep: the forward coupled loop with no ice load; the
  // derivative state's step carries the continuous drive built from the
  // adjoint state before and after its own step.
  observe(N);
  accumulateItau(N, wEarly[N] * perTime);
  accumulateMaterial(N, wEarly[N] * perTime);
  for (std::size_t n = N; n >= 1; --n) {
    if (options_.on_node) options_.on_node(n);
    const double dt = (times[n] - times[n - 1]) * ev.toMatTime_;
    auto before = Viscoelastic3D::State{};
    if (second) {
      op.PrimeStrain(vs.state, ev.Columns(vs.sigma, ev.toMatDensity_),
                     ev.Columns(vs.psi, ev.toMatPotential_));
      before = vs.state;
    }
    MaterialSource record;
    densityLate[n - 1] =
        reversedStep(vs, dt, n - 1, nullptr, nullptr, hasMaterialV ? &record : nullptr);
    if (second) {
      const auto drive = TangentLinearSolver3D::FirstOrderDrive(
          op, AdjointMethod::Continuous, scheme, dt, before, vs.state, *ditau);
      dDensityLate[n - 1] = reversedStep(ds, dt, n - 1, &drive,
                                         hasMaterialV ? &record : nullptr, nullptr);
    }
    accumulateItau(n - 1, wLate[n - 1] * perTime);
    accumulateMaterial(n - 1, wLate[n - 1] * perTime);
    observe(n - 1);
    accumulateItau(n - 1, wEarly[n - 1] * perTime);
    accumulateMaterial(n - 1, wEarly[n - 1] * perTime);
  }
  op.ReleaseScratch();

  // Node duals from the densities and the lumps (the 1D assembly); the
  // derivative duals carry no lumps.
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
  std::vector<std::vector<double>> loadDuals, densityLoad;
  assemble(densityLate, densityEarly, lumps, loadDuals, densityLoad);

  Kernels3D out;
  out.ice = detail::AssembleIceViews(
      C, solver.IceDensity(), times, std::move(loadDuals),
      std::move(densityLoad), std::move(lumps), oceanFunctions, rhoW);
  auto spherical = [&](const std::vector<std::vector<double>>& kernel) {
    auto reduced = std::vector<std::vector<double>>(
        K, std::vector<double>(nodeCount, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        reduced[k][rn] = coupling.GridOf(rn).Integrate(
            std::span(kernel[k]).subspan(coupling.FieldOffset(rn),
                                         coupling.FieldSizeOf(rn)));
      }
    }
    return reduced;
  };
  if (trajectory != nullptr) {
    out.itauKernel3D = std::move(itauKernel);
    out.itauKernelSpherical = spherical(out.itauKernel3D);
    out.logViscosityKernel3D.assign(K, std::vector<double>(values, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto i = std::size_t{0}; i < values; ++i) {
        out.logViscosityKernel3D[k][i] =
            -op.itau_[k][i] * out.itauKernel3D[k][i];
      }
    }
  }
  KernelDerivatives3D derivatives;
  if (second) {
    std::vector<std::vector<double>> dloadDuals, dDensityLoad;
    assemble(dDensityLate, dDensityEarly, dlumps, dloadDuals, dDensityLoad);
    auto views = detail::AssembleIceViews(
        C, solver.IceDensity(), times, std::move(dloadDuals),
        std::move(dDensityLoad), {}, oceanFunctions, rhoW);
    derivatives.ice.times.assign(times.begin(), times.end());
    derivatives.ice.thickness = std::move(views.thickness);
    derivatives.ice.initialSeaLevel = std::move(views.initialSeaLevel);
    derivatives.itauKernel3D = std::move(dItauKernel);
    derivatives.itauKernelSpherical = spherical(derivatives.itauKernel3D);
    // d/dv of -itau K: -(delta itau K + itau dK).
    derivatives.logViscosityKernel3D.assign(K, std::vector<double>(values, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto i = std::size_t{0}; i < values; ++i) {
        derivatives.logViscosityKernel3D[k][i] =
            -((*ditau)[k][i] * out.itauKernel3D[k][i] +
              op.itau_[k][i] * derivatives.itauKernel3D[k][i]);
      }
    }
  }
  if (material && second) {
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
  // as densities per unit volume (the reductions as means over the sphere)
  detail::KernelsToFields(coupling, out);
  detail::KernelsToFields(coupling, derivatives);
  return {std::move(out), std::move(derivatives)};
}

namespace {

void CheckKernelLayout(const GridCoupling& coupling,
                       std::span<const double> kernel) {
  const auto expected = coupling.GridValueCount();
  if (kernel.size() != expected) {
    throw std::invalid_argument(std::format(
        "the kernel holds {} values; the coupling's nodes and grid {}",
        kernel.size(), expected));
  }
}

}  // namespace

namespace detail {

void ToVolumeDensity(const GridCoupling& coupling, std::vector<double>& kernel) {
  if (kernel.empty()) return;
  CheckKernelLayout(coupling, kernel);
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto r = coupling.RadiusOfNode(node);
    const auto scale = 1.0 / (coupling.WeightOfNode(node) * r * r);
    const auto at = coupling.FieldOffset(node);
    for (auto x = at; x < at + coupling.FieldSizeOf(node); ++x) {
      kernel[x] *= scale;
    }
  }
}

}  // namespace detail

double VolumePairing(const GridCoupling& coupling, std::span<const double> kernel,
                     std::span<const double> direction) {
  CheckKernelLayout(coupling, kernel);
  CheckKernelLayout(coupling, direction);
  auto total = 0.0;
  auto product = std::vector<double>{};
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto at = coupling.FieldOffset(node);
    const auto size = coupling.FieldSizeOf(node);
    product.resize(size);
    for (auto x = std::size_t{0}; x < size; ++x) {
      product[x] = kernel[at + x] * direction[at + x];
    }
    const auto r = coupling.RadiusOfNode(node);
    total += coupling.WeightOfNode(node) * r * r *
             coupling.GridOf(node).Integrate(product);
  }
  return total;
}

std::vector<double> SphericalMean(const GridCoupling& coupling,
                                  std::span<const double> kernel) {
  constexpr auto kFourPi = 4.0 * 3.14159265358979323846;
  CheckKernelLayout(coupling, kernel);
  auto out = std::vector<double>{};
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto slice =
        kernel.subspan(coupling.FieldOffset(node), coupling.FieldSizeOf(node));
    out.push_back(coupling.GridOf(node).Integrate(slice) / kFourPi);
  }
  return out;
}

double VolumePairingOfMeans(const GridCoupling& coupling,
                            std::span<const double> kernel,
                            std::span<const double> direction) {
  constexpr auto kFourPi = 4.0 * 3.14159265358979323846;
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  if (kernel.size() != nodes || direction.size() != nodes) {
    throw std::invalid_argument(std::format(
        "per-node arrays of {} and {} values; the coupling has {} nodes",
        kernel.size(), direction.size(), nodes));
  }
  auto total = 0.0;
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto r = coupling.RadiusOfNode(node);
    total += kFourPi * coupling.WeightOfNode(node) * r * r * kernel[node] *
             direction[node];
  }
  return total;
}

}  // namespace sl3dpp
