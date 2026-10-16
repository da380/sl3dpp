#include "sl3dpp/Adjoint3D.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/AndersonMixer.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
using TensorField = GridCoupling::TensorField;

std::vector<double> Flatten(const TensorField& tensor) {
  std::vector<double> flat;
  flat.reserve(2 * 2 * tensor.c[0].size() + tensor.r.size());
  for (auto c = 0; c < 2; ++c) {
    for (const auto& v : tensor.c[c]) {
      flat.push_back(v.real());
      flat.push_back(v.imag());
    }
  }
  for (auto v : tensor.r) flat.push_back(v);
  return flat;
}

TensorField Unflatten(const GridCoupling& coupling,
                      std::span<const double> flat) {
  auto tensor = coupling.ZeroTensor();
  std::size_t at = 0;
  for (auto c = 0; c < 2; ++c) {
    for (auto& v : tensor.c[c]) {
      v = {flat[at], flat[at + 1]};
      at += 2;
    }
  }
  for (auto& v : tensor.r) v = flat[at++];
  return tensor;
}

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
  for (std::size_t i = 1; i < times.size(); ++i) {
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
  const double rhoW = solver.WaterDensity();
  const double g = solver.SurfaceGravity();
  const double r2 = solver.Radius() * solver.Radius();
  const auto fieldSize = grid.FieldSize();
  const int lmax = op.LMax();
  const auto K = op.GetRheology().BranchCount();
  const std::size_t N = times.size() - 1;

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
    for (const auto& state : trajectory->states) {
      if (!state.has_value() || !state->strain.has_value()) {
        throw std::invalid_argument(
            "the 3D viscosity kernels need every node's recorded state "
            "(a run with record_stride = 1; no 3D backward "
            "reconstruction exists yet)");
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

  std::map<const LoveNumbers*, SeaLevelSolver> flavours;
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
      for (std::size_t i = 0; i < fieldSize; ++i) {
        zetaU[i] = -m.displacementDual[i];
      }
      f.displacementLoad = zetaU;
    }
    if (!m.potentialDual.empty()) {
      zetaPhi.resize(fieldSize);
      for (std::size_t i = 0; i < fieldSize; ++i) {
        zetaPhi[i] = -g * m.potentialDual[i];
      }
      f.potentialLoad = zetaPhi;
    }
    if (rotation != RotationalFeedback::None) {
      std::array<double, 3> avc{0.0, 0.0, 0.0};
      for (int j = 0; j < 3; ++j) avc[j] = -g * m.angularVelocityDual[j];
      if (!m.potentialDual.empty()) {
        const auto ps = psiStar(m.potentialDual);
        for (int j = 0; j < 3; ++j) avc[j] += g * ps[j];
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
    std::vector<double> lamGrid(fieldSize);
    for (std::size_t i = 0; i < fieldSize; ++i) {
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
  auto applyJump = [&](const Jump& j) {
    auto spectral = coupling.ZeroSpectral();
    const bool split = !j.zu.empty() || !j.zphi.empty();
    for (int l = 0; l <= lmax; ++l) {
      const auto& units = reference.UnitResponses(nullptr, l);
      const auto ndof = units.size() / 2;
      std::vector<Complex> x(ndof * (static_cast<std::size_t>(l) + 1), 0.0);
      const double* splitUnits =
          split ? reference.SplitUnitResponses(nullptr, l).data() : nullptr;
      for (int m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        const Complex lamAmp = ev.toMatDensity_ * j.lam[i];
        const Complex psiAmp = ev.toMatPotential_ * j.psiA[i];
        const Complex zuAmp =
            j.zu.empty() ? 0.0 : ev.toMatDensity_ * j.zu[i];
        const Complex zphiAmp =
            j.zphi.empty() ? 0.0 : ev.toMatDensity_ * j.zphi[i];
        for (std::size_t d = 0; d < ndof; ++d) {
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
    for (std::size_t k = 0; k < K; ++k) {
      const auto& itau = op.itau_[k];
      auto& m = vstate.m[k];
      for (auto c = 0; c < 2; ++c) {
        for (std::size_t i = 0; i < m.c[c].size(); ++i) {
          m.c[c][i] += itau[i] * dTilde.c[c][i];
        }
      }
      for (std::size_t i = 0; i < m.r.size(); ++i) {
        m.r[i] += itau[i] * dTilde.r[i];
      }
    }
    vstate.strain.reset();
  };

  // The memory-only adjoint equilibrium of the current state through
  // the unrelaxed reference systems (laterally uniform, no iteration).
  auto momentOf = [&](const Viscoelastic3D::Solutions& solutions) {
    const auto& deg = reference.Deg(0);
    const auto& x = solutions.spheroidal.at(0);
    double sum = 0.0;
    for (int d = 0; d < deg.ndof; ++d) {
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
    std::vector<double> sig(fieldSize);
    for (std::size_t i = 0; i < fieldSize; ++i) {
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
  for (std::size_t n = 1; n < times.size(); ++n) {
    wEarly[n] = 0.5 * (times[n] - times[n - 1]);
  }
  for (std::size_t n = 0; n + 1 < times.size(); ++n) {
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
  std::vector<std::vector<double>> itauKernel;
  const auto& mesh = op.GetRheology().BaseMaterial().Mesh();
  const auto meshW = mesh.Weights();
  const auto& matMesh = mesh;
  const double aMat =
      matMesh.Radius(matMesh.ElementCount() - 1, matMesh.NGll() - 1);
  const double pairing =
      r2 * ev.toNdLength_ /
      (aMat * aMat * ev.toMatDensity_ *
       op.GetRheology().BaseMaterial().SurfaceGravity());
  if (trajectory != nullptr) {
    itauKernel.assign(K, std::vector<double>(nodeCount * gridF, 0.0));
  }
  auto accumulateItau = [&](std::size_t node, double weight) {
    if (trajectory == nullptr || weight == 0.0) return;
    const auto& fwd = *trajectory->states[node];
    const auto& dFwd = *fwd.strain;
    for (std::size_t k = 0; k < K; ++k) {
      const auto& itau = op.itau_[k];
      const auto& mu = op.muBranch_[k];
      const auto& mA = vstate.m[k];
      const auto& mF = fwd.m[k];
      for (std::size_t rn = 0; rn < nodeCount; ++rn) {
        if (mu[rn] <= 0.0) continue;
        const int e = coupling.ElementOfNode(static_cast<int>(rn));
        const int q = coupling.NodeInElementOf(static_cast<int>(rn));
        const double r = coupling.RadiusOfNode(static_cast<int>(rn));
        const double wq = meshW[q] * mesh.Jacobian(e);
        const double factor = pairing * weight * mu[rn] * wq * r * r;
        const std::size_t base = rn * gridF;
        for (std::size_t x = 0; x < gridF; ++x) {
          const auto i = base + x;
          if (itau[i] <= 0.0) continue;
          // d - m of the forward state, against the adjoint state: the
          // reduced-storage contraction (derived spins by reality).
          const auto dm2 = dFwd.c[0][i] - mF.c[0][i];
          const auto dm1 = dFwd.c[1][i] - mF.c[1][i];
          const double dm0 = dFwd.r[i] - mF.r[i];
          const double contraction =
              12.0 * mA.r[i] * dm0 +
              8.0 * (mA.c[1][i] * std::conj(dm1)).real() +
              4.0 * (mA.c[0][i] * std::conj(dm2)).real();
          itauKernel[k][i] += factor * contraction / itau[i];
        }
      }
    }
  };

  std::vector<std::vector<double>> densityLate(times.size()),
      densityEarly(times.size());
  std::map<std::size_t, std::vector<double>> lumps;

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
    const auto step =
        op.StartStep(vstate, dt, scheme, ev.Columns(sigmaC, ev.toMatDensity_),
                     ev.Columns(psiC, ev.toMatPotential_));
    const auto& effectiveSolver = solverFor(step.EffectiveLove());

    auto d = step.StartStrain();
    detail::AndersonMixer mixer(options_.anderson_memory);
    SeaLevelSolver::Response response;
    bool converged = false;
    for (auto pass = 0; pass < options_.max_coupled_iterations; ++pass) {
      auto responses = step.Respond(d);
      SeaLevelSolver::GeneralisedForcing coupled;
      coupled.displacementResponse = ev.Coefficients(
          op.Surface(responses, Component::U), ev.toNdLength_);
      coupled.potentialResponse = ev.Coefficients(
          op.Surface(responses, Component::Phi), ev.toNdPotential_);
      coupled.inertiaMomentChange =
          step.InertiaMoment(responses) * ev.toNdMoment_;
      coupled.rotation = rotation;
      response = effectiveSolver.SolveGeneralised(ev.state_, coupled,
                                                  sleOptions);
      std::vector<double> sig(fieldSize);
      for (std::size_t i = 0; i < fieldSize; ++i) {
        sig[i] = rhoW * C[i] * response.seaLevelChange[i];
      }
      sigmaC = grid.Expand(sig);
      psiC = rotation != RotationalFeedback::None
                 ? solver.CentrifugalCoefficients(response.angularVelocity)
                 : grid.ZeroCoefficients();
      step.AddChannels(responses, ev.Columns(sigmaC, ev.toMatDensity_),
                       ev.Columns(psiC, ev.toMatPotential_));
      auto dNext = step.StrainOf(responses);

      double scale = 0.0, diff = 0.0;
      for (auto c = 0; c < 2; ++c) {
        for (std::size_t i = 0; i < dNext.c[c].size(); ++i) {
          scale = std::max(scale, std::abs(dNext.c[c][i]));
          diff = std::max(diff, std::abs(dNext.c[c][i] - d.c[c][i]));
        }
      }
      for (std::size_t i = 0; i < dNext.r.size(); ++i) {
        scale = std::max(scale, std::abs(dNext.r[i]));
        diff = std::max(diff, std::abs(dNext.r[i] - d.r[i]));
      }
      if (scale == 0.0 || diff <= options_.coupled_rtol * scale) {
        converged = true;
        d = std::move(dNext);
        break;
      }
      if (options_.anderson_memory > 0) {
        d = Unflatten(coupling, mixer.Update(Flatten(d), Flatten(dNext)));
      } else {
        d = std::move(dNext);
      }
    }
    if (!converged) {
      throw std::runtime_error(std::format(
          "the reversed lateral fixed point at node {} did not converge",
          n - 1));
    }
    densityLate[n - 1] = response.seaLevelChange;
    vstate = step.Finish(std::move(d));
    accumulateItau(n - 1, wLate[n - 1] * perTime);
    observe(n - 1);
    accumulateItau(n - 1, wEarly[n - 1] * perTime);
  }

  // Node duals from the densities and the lumps (the 1D assembly).
  std::vector<std::vector<double>> loadDuals(times.size()),
      densityLoad(times.size());
  for (std::size_t n = 0; n < times.size(); ++n) {
    loadDuals[n].assign(fieldSize, 0.0);
    densityLoad[n].assign(fieldSize, 0.0);
    const auto* late = densityLate[n].empty() ? nullptr : &densityLate[n];
    const auto* early = densityEarly[n].empty() ? late : &densityEarly[n];
    const auto lump = lumps.find(n);
    const double totalW = wEarly[n] + wLate[n];
    for (std::size_t i = 0; i < fieldSize; ++i) {
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
    for (std::size_t k = 0; k < K; ++k) {
      for (std::size_t rn = 0; rn < nodeCount; ++rn) {
        out.itauKernelSpherical[k][rn] = coupling.Grid().Integrate(
            std::span(out.itauKernel3D[k]).subspan(rn * gridF, gridF));
      }
      for (std::size_t i = 0; i < nodeCount * gridF; ++i) {
        out.logViscosityKernel3D[k][i] =
            -op.itau_[k][i] * out.itauKernel3D[k][i];
      }
    }
  }
  return out;
}

}  // namespace sl3dpp
