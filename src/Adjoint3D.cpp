#include "sl3dpp/Adjoint3D.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;

}  // namespace

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
    const ForwardTrajectory3D* trajectory) const {
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
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto& coupling = op.Coupling();
  const auto scheme = ev.options_.scheme;
  const auto rotation = ev.options_.rotation;
  const auto& C = ev.state_.oceanFunction;
  const auto rhoW = solver.WaterDensity();
  const auto g = solver.SurfaceGravity();
  const auto r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const auto lmax = op.LMax();
  const auto K = op.GetRheology().BranchCount();
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

  // The adjoint state, stepped by the forward coupled machinery in
  // reversed time with no ice load.
  auto vstate = op.ZeroState();
  auto sigmaC = grid.ZeroCoefficients();
  auto psiC = grid.ZeroCoefficients();

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
  auto jump = [&](const Measurement& m) {
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
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
    if (!resp.converged) {
      throw std::runtime_error(
          "the instantaneous adjoint fingerprint did not converge");
    }
    Jump out;
    out.xiA = resp.seaLevelChange;
    auto lamGrid = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      lamGrid[i] = (m.seaLevelDual.empty() ? 0.0 : m.seaLevelDual[i]) +
                   rhoW * C[i] * out.xiA[i];
    }
    out.lam = grid.Expand(lamGrid);
    out.psiA = rotation != RotationalFeedback::None
                   ? solver.CentrifugalCoefficients(resp.angularVelocity)
                   : grid.ZeroCoefficients();
    if (!zetaU.empty()) out.zu = grid.Expand(zetaU);
    if (!zetaPhi.empty()) out.zphi = grid.Expand(zetaPhi);
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
    }
    const auto dTilde = coupling.Synthesise(spectral);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& itau = op.itau_[k];
      auto& m = vstate.m[k];
      for (auto c = 0; c < 2; ++c) {
        for (auto i = std::size_t{0}; i < m.c[c].size(); ++i) {
          m.c[c][i] += itau[i] * dTilde.c[c][i];
        }
      }
      for (auto i = std::size_t{0}; i < m.r.size(); ++i) {
        m.r[i] += itau[i] * dTilde.r[i];
      }
    }
    vstate.Invalidate();
  };

  // The memory-only adjoint equilibrium of the current state through
  // the unrelaxed reference systems (laterally uniform, no iteration).
  auto momentOf = [&](const Viscoelastic3D::Solutions& solutions) {
    const auto& deg = locked([&]() -> const auto& { return reference.Deg(0); });
    const auto& x = solutions.spheroidal.at(0);
    auto sum = 0.0;
    for (auto d = 0; d < deg.ndof; ++d) {
      sum += deg.momentRow[d] * x[d].real();
    }
    return sum;
  };
  auto smoothEquilibrium = [&]() {
    const auto solutions = op.SolveElastic(vstate);
    SeaLevelSolver::GeneralisedForcing f;
    f.rotation = rotation;
    f.displacementResponse =
        ev.Coefficients(op.Surface(solutions, Component::U), ev.toNdLength_);
    f.potentialResponse = ev.Coefficients(
        op.Surface(solutions, Component::Phi), ev.toNdPotential_);
    f.inertiaMomentChange = momentOf(solutions) * ev.toNdMoment_;
    const auto resp =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, f, sleOptions);
    if (!resp.converged) {
      throw std::runtime_error(
          "the adjoint equilibrium solve did not converge");
    }
    auto sig = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      sig[i] = rhoW * C[i] * resp.seaLevelChange[i];
    }
    sigmaC = grid.Expand(sig);
    psiC = rotation != RotationalFeedback::None
               ? solver.CentrifugalCoefficients(resp.angularVelocity)
               : grid.ZeroCoefficients();
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
  const auto gridF = coupling.Grid().FieldSize();
  const auto nodeCount = static_cast<std::size_t>(coupling.NodeCount());
  auto itauKernel = std::vector<std::vector<double>>{};
  const auto& matMesh = op.GetRheology().BaseMaterial().Mesh();
  const double aMat =
      matMesh.Radius(matMesh.ElementCount() - 1, matMesh.NGll() - 1);
  const double pairing =
      r2 * ev.toNdLength_ /
      (aMat * aMat * ev.toMatDensity_ *
       op.GetRheology().BaseMaterial().SurfaceGravity());
  if (trajectory != nullptr) {
    itauKernel.assign(K, std::vector<double>(nodeCount * gridF, 0.0));
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
  auto accumulateItau = [&](std::size_t node, double weight) {
    if (trajectory == nullptr || weight == 0.0) return;
    const auto& dFwd = forwardStrain(node);
    const auto& fwd = forwardState(node);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& itau = op.itau_[k];
      const auto& mu = op.muBranch_[k];
      const auto& mA = vstate.m[k];
      const auto& mF = fwd.m[k];
      auto& kernel = itauKernel[k];
      // Node by node, each writing only its own grid values.
      detail::ParallelElements(
          nodeCount, op.options_.parallel, [&](std::size_t rn) {
            if (mu[rn] <= 0.0) return;
            const auto r = coupling.RadiusOfNode(static_cast<int>(rn));
            const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
            const double factor = pairing * weight * mu[rn] * wq * r * r;
            const auto base = rn * gridF;
            for (auto x = std::size_t{0}; x < gridF; ++x) {
              const auto i = base + x;
              if (itau[i] <= 0.0) continue;
              // d - m of the forward state, against the adjoint state:
              // the reduced-storage contraction (derived spins by
              // reality).
              const auto dm2 = dFwd.c[0][i] - mF.c[0][i];
              const auto dm1 = dFwd.c[1][i] - mF.c[1][i];
              const double dm0 = dFwd.r[i] - mF.r[i];
              const double contraction =
                  12.0 * mA.r[i] * dm0 +
                  8.0 * (mA.c[1][i] * std::conj(dm1)).real() +
                  4.0 * (mA.c[0][i] * std::conj(dm2)).real();
              kernel[i] += factor * contraction / itau[i];
            }
          });
    }
  };

  std::vector<std::vector<double>> densityLate(times.size()),
      densityEarly(times.size());
  auto lumps = std::map<std::size_t, std::vector<double>>{};

  auto observe = [&](std::size_t node) {
    const auto found = measurements.find(node);
    if (found == measurements.end()) return;
    const auto j = jump(found->second);
    lumps.emplace(node, j.xiA);
    applyJump(j);
    auto early = smoothEquilibrium();
    if (node > 0) densityEarly[node] = std::move(early);
  };

  // The reversed sweep: the forward coupled loop with no ice load.
  observe(N);
  accumulateItau(N, wEarly[N] * perTime);
  for (std::size_t n = N; n >= 1; --n) {
    const double dt = (times[n] - times[n - 1]) * ev.toMatTime_;
    auto step =
        op.StartStep(vstate, dt, scheme, ev.Columns(sigmaC, ev.toMatDensity_),
                     ev.Columns(psiC, ev.toMatPotential_));
    const auto& effectiveSolver = solverFor(step.EffectiveLove());

    SeaLevelSolver::Response response;
    auto [strain, info] = step.Converge(
        [&](const GridCoupling::SpectralTensor& s) {
          auto responses = step.Respond(s);
          SeaLevelSolver::GeneralisedForcing coupled;
          coupled.displacementResponse = ev.Coefficients(
              op.Surface(responses, Component::U), ev.toNdLength_);
          coupled.potentialResponse = ev.Coefficients(
              op.Surface(responses, Component::Phi), ev.toNdPotential_);
          coupled.inertiaMomentChange =
              step.InertiaMoment(responses) * ev.toNdMoment_;
          coupled.rotation = rotation;
          response =
              effectiveSolver.SolveGeneralised(ev.state_, coupled, sleOptions);
          auto sig = std::vector<double>(fieldSize);
          for (auto i = std::size_t{0}; i < fieldSize; ++i) {
            sig[i] = rhoW * C[i] * response.seaLevelChange[i];
          }
          sigmaC = grid.Expand(sig);
          psiC = rotation != RotationalFeedback::None
                     ? solver.CentrifugalCoefficients(response.angularVelocity)
                     : grid.ZeroCoefficients();
          step.AddChannels(responses, ev.Columns(sigmaC, ev.toMatDensity_),
                           ev.Columns(psiC, ev.toMatPotential_));
          return step.StrainOf(responses);
        },
        {.max_iterations = options_.max_coupled_iterations,
         .rtol = options_.coupled_rtol,
         .anderson_memory = options_.anderson_memory,
         .anderson_rate = options_.anderson_rate});
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the reversed lateral fixed point at node {} did not converge",
          n - 1));
    }
    densityLate[n - 1] = response.seaLevelChange;
    vstate = std::move(step).Finish(std::move(strain));
    accumulateItau(n - 1, wLate[n - 1] * perTime);
    observe(n - 1);
    accumulateItau(n - 1, wEarly[n - 1] * perTime);
  }

  // Node duals from the densities and the lumps (the 1D assembly).
  std::vector<std::vector<double>> loadDuals(times.size()),
      densityLoad(times.size());
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loadDuals[n].assign(fieldSize, 0.0);
    densityLoad[n].assign(fieldSize, 0.0);
    const auto* late = densityLate[n].empty() ? nullptr : &densityLate[n];
    const auto* early = densityEarly[n].empty() ? late : &densityEarly[n];
    const auto lump = lumps.find(n);
    const auto totalW = wEarly[n] + wLate[n];
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double lateV = late == nullptr ? 0.0 : (*late)[i] * perTime;
      const double earlyV = early == nullptr ? 0.0 : (*early)[i] * perTime;
      const double smooth = wEarly[n] * earlyV + wLate[n] * lateV;
      loadDuals[n][i] =
          smooth + (lump == lumps.end() ? 0.0 : lump->second[i]);
      densityLoad[n][i] = smooth / totalW;
    }
  }

  Kernels3D out;
  out.ice = detail::AssembleIceViews(C, solver.IceDensity(), times,
                                     std::move(loadDuals),
                                     std::move(densityLoad), std::move(lumps));
  if (trajectory != nullptr) {
    out.itauKernel3D = std::move(itauKernel);
    out.itauKernelSpherical.assign(K, std::vector<double>(nodeCount, 0.0));
    out.logViscosityKernel3D.assign(
        K, std::vector<double>(nodeCount * gridF, 0.0));
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto rn = std::size_t{0}; rn < nodeCount; ++rn) {
        out.itauKernelSpherical[k][rn] = coupling.Grid().Integrate(
            std::span(out.itauKernel3D[k]).subspan(rn * gridF, gridF));
      }
      for (auto i = std::size_t{0}; i < nodeCount * gridF; ++i) {
        out.logViscosityKernel3D[k][i] =
            -op.itau_[k][i] * out.itauKernel3D[k][i];
      }
    }
  }
  return out;
}

}  // namespace sl3dpp
