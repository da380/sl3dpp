// The adjoint sweeps: the discrete gradient identity at solver
// tolerance for a full measurement set, the thickness-view identity with
// the baseline, the continuous adjoint's agreement with the discrete and
// its convergence under step refinement, and the cross-validation of the
// delta amplitudes — the gate structure of pyslfp's
// test_continuous_adjoint_against_the_discrete.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 16;

/// The smooth cap of the solver tests (TestSeaLevel.cpp), the kernel
/// fixture's measurement formula.
double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;

  /// A viscous mantle with the given viscosity (Pa s).
  static Scene Make(double viscositySI) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();

    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i,
                          viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();

    ViscoelasticOperator op(rheology, kLmax);
    SeaLevelSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});

    Fingerprint fingerprint(kLmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model), std::move(material), std::move(rheology),
                 std::move(solver), std::move(state)};
  }

  /// A smooth random band-limited field of the given amplitude.
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng,
                                           double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng),
                                            m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }

  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }
};

/// The objective the measurements define, evaluated on a forward run.
double Objective(const Scene& setup,
                 const std::vector<SeaLevelEvolution::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.seaLevelDual.empty()) {
      total += setup.Surface(m.seaLevelDual, snap.seaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      total += setup.Surface(m.displacementDual, snap.displacement);
    }
    if (!m.potentialDual.empty()) {
      total += setup.Surface(m.potentialDual, snap.potential);
    }
    for (auto j = 0; j < 3; ++j) {
      total += m.angularVelocityDual[j] * snap.angularVelocity[j];
    }
  }
  return total;
}

/// A per-node load history as the forward driver's callable.
std::function<std::vector<double>(double)> History(
    const std::vector<double>& times,
    const std::vector<std::vector<double>>& loads) {
  return [&times, &loads](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        return loads[n];
      }
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
}

}  // namespace

TEST(Adjoint, DiscreteGradientIdentity) {
  // The exact transpose: delta J = sum_n <loadDuals[n], delta f_n> to the
  // solvers' tolerance, for a full measurement set (all four duals), both
  // schemes. The ice history enters linearly, so J(delta) is the exact
  // directional derivative.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(7);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);

  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }

  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = setup.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = setup.Smooth(rng, 1.0);
  middle.displacementDual = setup.Smooth(rng, -0.5);
  measurements[3] = std::move(middle);
  Measurement last;
  last.displacementDual = setup.Smooth(rng, 0.7);
  last.potentialDual = setup.Smooth(rng, 0.4);
  last.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[5] = std::move(last);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme, .rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto exact = Objective(setup, snapshots, measurements);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements);
    auto predicted = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      predicted += setup.Surface(kernels.loadDuals[n], loads[n]);
    }
    EXPECT_NEAR(predicted, exact, 1e-8 * std::abs(exact))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Adjoint, ThicknessViewCarriesTheBaseline) {
  // The thickness duals: with the forward load f_n = rho_i (1 - C)
  // (I_n - I_0), delta J = sum_n <thickness[n], delta I_n> — the
  // initial-thickness dependence folded into node 0. A uniform-in-time
  // thickness change loads nothing and pairs to zero by construction.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(11);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * kyr);

  const auto length = scales.Length();
  auto thicknesses = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    thicknesses.push_back(setup.Smooth(rng, 10.0 / length));
  }
  const auto& C = setup.state.oceanFunction;
  const auto rhoI = setup.solver.IceDensity();
  auto groundedLoads = [&](const std::vector<std::vector<double>>& I) {
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      auto load = std::vector<double>(I[n].size());
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] = rhoI * (1.0 - C[i]) * (I[n][i] - I[0][i]);
      }
      loads.push_back(std::move(load));
    }
    return loads;
  };
  const auto loads = groundedLoads(thicknesses);

  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[2].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[4].seaLevelDual = setup.Smooth(rng, -0.6);

  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution evolution(op, setup.solver, setup.state, {.rtol = 1e-12});
  auto snapshots = evolution.Run(times, History(times, loads));
  const auto exact = Objective(setup, snapshots, measurements);

  AdjointSolver adjoint(evolution,
                        {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  const auto kernels = adjoint.IceHistoryKernels(times, measurements);
  auto predicted = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    predicted += setup.Surface(kernels.thickness[n], thicknesses[n]);
  }
  EXPECT_NEAR(predicted, exact, 1e-8 * std::abs(exact));

  // A uniform shift of the whole history pairs to zero: the node
  // thickness duals and the baseline cancel identically.
  auto ones = std::vector<double>(setup.solver.Grid().FieldSize(), 1.0);
  auto shift = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    shift += setup.Surface(kernels.thickness[n], ones);
  }
  auto scale = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    scale = std::max(scale, std::abs(setup.Surface(kernels.thickness[n],
                                                   thicknesses[n])));
  }
  EXPECT_LT(std::abs(shift) * 10.0 / length, 1e-10 * scale);
}

TEST(Adjoint, ContinuousAgreesWithDiscreteAndConverges) {
  // The deliverable of the comparison (docs/dev/adjoints.md §1): the discrete
  // transpose is exact at any step; the continuous adjoint carries the
  // discretise/transpose commute error — at the per-cent level or better
  // once the steps resolve the relaxation, falling under refinement.
  // Their delta amplitudes agree although computed by different
  // realisations (the engine transpose against the forward channel
  // mapping), cross-validating both.
  std::mt19937 rng(4);

  auto gradientErrors = [&](Scene& setup, std::size_t npts, double T,
                            std::map<AdjointMethod, IceKernels>* kernelsOut =
                                nullptr) {
    const auto& scales = setup.solver.SolverScales();
    const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(T * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
    std::mt19937 local(17);  // the same loads at every resolution
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      loads.push_back(setup.Smooth(local, 100.0 / sigma));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].seaLevelDual = setup.Smooth(local, 1.0);
    measurements[npts - 1].seaLevelDual = setup.Smooth(local, 0.5);

    ViscoelasticOperator op(setup.rheology, kLmax);
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto exact = Objective(setup, snapshots, measurements);

    auto errors = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution, {.method = method, .rtol = 1e-12});
      auto kernels = adjoint.IceHistoryKernels(times, measurements);
      auto predicted = 0.0;
      for (auto n = std::size_t{0}; n < npts; ++n) {
        predicted += setup.Surface(kernels.loadDuals[n], loads[n]);
      }
      errors[method] = std::abs(predicted - exact) / std::abs(exact);
      if (kernelsOut) kernelsOut->emplace(method, std::move(kernels));
    }
    return errors;
  };

  // The stiff regime: 2.5 kyr steps over several 450-yr relaxation times.
  auto stiff = Scene::Make(1e21);
  auto kernels = std::map<AdjointMethod, IceKernels>{};
  auto coarse = gradientErrors(stiff, 5, 10.0, &kernels);
  auto fine = gradientErrors(stiff, 9, 10.0);
  EXPECT_LT(coarse[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(fine[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(coarse[AdjointMethod::Continuous], 0.5);
  EXPECT_GT(coarse[AdjointMethod::Continuous] /
                fine[AdjointMethod::Continuous],
            1.8);  // converging

  // The delta amplitudes cross-validate the two realisations.
  for (const auto& [node, a] : kernels[AdjointMethod::Discrete].instantaneous) {
    const auto& b = kernels[AdjointMethod::Continuous].instantaneous.at(node);
    auto top = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      top = std::max(top, std::abs(a[i]));
      diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    EXPECT_LT(diff, 1e-7 * top) << "node " << node;
  }

  // The resolved regime: relaxation ten times slower; the commute error
  // is small and falls under refinement.
  auto resolved = Scene::Make(1e22);
  auto coarseR = gradientErrors(resolved, 5, 10.0);
  auto fineR = gradientErrors(resolved, 9, 10.0);
  EXPECT_LT(coarseR[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(coarseR[AdjointMethod::Continuous], 5e-2);
  EXPECT_GT(coarseR[AdjointMethod::Continuous] /
                fineR[AdjointMethod::Continuous],
            2.0);
}

TEST(Adjoint, ViscosityKernelsAgainstFiniteDifferences) {
  // The discrete inverse-relaxation-time kernels: the reverse sweep given
  // the recorded trajectory pairs the scheme-weight derivatives with the
  // duals, and the FD residual against a smooth nodal itau perturbation
  // falls at second order in eps — the kernel itself is exact (Richardson
  // extrapolation lands on it to solver precision). Both implicit
  // schemes, and the viscosity mapping -itau^2 / mu.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(7);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);

  // A smooth nodal perturbation of the branch's inverse relaxation time,
  // in material units, supported where the branch lives.
  const auto& base = setup.rheology.BranchAt(0);
  const auto& mesh = setup.material.Mesh();
  auto itauTop = 0.0;
  for (auto v : base.itau) itauTop = std::max(itauTop, v);
  auto ditau = std::vector<double>(base.itau.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (base.itau[i] > 0.0) {
        const auto r = mesh.Radius(e, q);
        ditau[i] = (0.3 + 0.7 * r * r) * itauTop;
      }
    }
  }

  auto functional = [&](double eps, Scheme scheme) {
    auto branch = base;
    for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
      branch.itau[i] += eps * ditau[i];
    }
    auto rheology = Rheology::Create(setup.material, {std::move(branch)}).value();
    ViscoelasticOperator op(rheology, kLmax);
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme, .rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    return Objective(setup, snapshots, measurements);
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    SeaLevelEvolution evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &trajectory);
    ASSERT_FALSE(kernels.itauKernel.empty());
    auto predicted = 0.0;
    for (auto i = std::size_t{0}; i < ditau.size(); ++i) {
      predicted += kernels.itauKernel[0][i] * ditau[i];
    }

    auto fd = std::map<double, double>{};
    for (double eps : {0.02, 0.01}) {
      fd[eps] =
          (functional(eps, scheme) - functional(-eps, scheme)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.02] - predicted);
    const auto r2 = std::abs(fd[0.01] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
    // The identity to FD/solver noise: the directional derivative is
    // five orders below J, so the difference quotient keeps ~6 digits.
    EXPECT_NEAR(richardson, predicted, 1e-5 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);

    // The viscosity mapping, and the refusal without a trajectory.
    const auto eta = kernels.Viscosity(setup.rheology);
    for (auto i = std::size_t{0}; i < ditau.size(); ++i) {
      if (base.mu[i] > 0.0) {
        EXPECT_NEAR(eta[0][i],
                    -(base.itau[i] * base.itau[i]) / base.mu[i] *
                        kernels.itauKernel[0][i],
                    1e-14 * std::abs(kernels.itauKernel[0][i]) + 1e-300);
      }
    }
    const auto bare = adjoint.IceHistoryKernels(times, measurements);
    EXPECT_TRUE(bare.itauKernel.empty());
    EXPECT_THROW((void)bare.Viscosity(setup.rheology), std::invalid_argument);
  }
}

TEST(Adjoint, ContinuousViscosityKernelsConverge) {
  // The continuous itau kernels differ from the discrete by the commute
  // error, which falls under step refinement — the resolution diagnostic
  // on the viscosity kernels themselves.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(3);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();

  auto kernelDiff = [&](std::size_t npts) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
    std::mt19937 local(23);
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      loads.push_back(setup.Smooth(local, 100.0 / sigma));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts - 1].seaLevelDual = setup.Smooth(local, 1.0);

    ViscoelasticOperator op(setup.rheology, kLmax);
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12, .record_stride = 1});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

    auto kernels = std::map<AdjointMethod, std::vector<double>>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution, {.method = method, .rtol = 1e-12});
      kernels[method] =
          adjoint.IceHistoryKernels(times, measurements, &trajectory)
              .itauKernel[0];
    }
    auto diff = 0.0, norm = 0.0;
    const auto& a = kernels[AdjointMethod::Discrete];
    const auto& b = kernels[AdjointMethod::Continuous];
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      diff += (a[i] - b[i]) * (a[i] - b[i]);
      norm += a[i] * a[i];
    }
    return std::sqrt(diff / norm);
  };

  const auto coarse = kernelDiff(6);
  const auto fine = kernelDiff(11);
  EXPECT_LT(coarse, 0.3);
  EXPECT_LT(fine, coarse / 1.5);
}

TEST(Adjoint, SavedAndRegeneratedStrainsAgree) {
  // Recorded states keep only the internal variables unless
  // record_strain is set; the sweeps then regenerate each node's strain
  // by one elastic solve. Either way the viscosity kernels agree, the
  // regenerated strain differing from the saved one only at the forward
  // tolerance.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(13);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].seaLevelDual = setup.Smooth(rng, 1.0);

  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution saving(
      op, setup.solver, setup.state,
      {.rtol = 1e-12, .record_stride = 1, .record_strain = true});
  SeaLevelEvolution lean(op, setup.solver, setup.state,
                         {.rtol = 1e-12, .record_stride = 1});
  const auto saved = saving.Run(times, History(times, loads));
  const auto plain = lean.Run(times, History(times, loads));
  for (const auto& s : saved) EXPECT_TRUE(s.internalState->HasStrain());
  for (const auto& s : plain) EXPECT_FALSE(s.internalState->HasStrain());
  const auto withStrain = ForwardTrajectory::FromSnapshots(saved);
  const auto withoutStrain = ForwardTrajectory::FromSnapshots(plain);

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    AdjointSolver adjoint(saving, {.method = method, .rtol = 1e-12});
    const auto a = adjoint.IceHistoryKernels(times, measurements, &withStrain)
                       .itauKernel[0];
    const auto b =
        adjoint.IceHistoryKernels(times, measurements, &withoutStrain)
            .itauKernel[0];
    auto scale = 0.0, worst = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      scale = std::max(scale, std::abs(a[i]));
      worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-8 * scale) << "method " << static_cast<int>(method);
  }
}

TEST(Adjoint, CheckpointedKernelsReconstruct) {
  // A strided recording reproduces the full-record viscosity kernels
  // through exact backward reconstruction, the residual at the exp(sum h)
  // roundoff budget of the gaps; a gapped trajectory without the recorded
  // loads is refused.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(9);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 10; ++n) times.push_back(n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[10].seaLevelDual = setup.Smooth(rng, 1.0);

  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution full(op, setup.solver, setup.state,
                         {.rtol = 1e-12, .record_stride = 1});
  auto fullSnapshots = full.Run(times, History(times, loads));
  const auto fullTrajectory = ForwardTrajectory::FromSnapshots(fullSnapshots);

  SeaLevelEvolution sparse(op, setup.solver, setup.state,
                           {.rtol = 1e-12, .record_stride = 3});
  auto sparseSnapshots = sparse.Run(times, History(times, loads));
  const auto sparseTrajectory =
      ForwardTrajectory::FromSnapshots(sparseSnapshots);
  auto kept = 0;
  for (const auto& s : sparseSnapshots) kept += s.internalState.has_value();
  ASSERT_LT(kept, static_cast<int>(sparseSnapshots.size()));

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    AdjointSolver adjoint(full, {.method = method, .rtol = 1e-12});
    const auto reference =
        adjoint.IceHistoryKernels(times, measurements, &fullTrajectory)
            .itauKernel[0];
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &sparseTrajectory)
            .itauKernel[0];
    auto diff = 0.0, norm = 0.0;
    for (auto i = std::size_t{0}; i < reference.size(); ++i) {
      diff += (kernels[i] - reference[i]) * (kernels[i] - reference[i]);
      norm += reference[i] * reference[i];
    }
    EXPECT_LT(std::sqrt(diff / norm), 1e-9)
        << "method " << static_cast<int>(method);
  }

  // A gapped trajectory without the recorded loads is refused.
  auto gapped = sparseTrajectory;
  gapped.loads.clear();
  gapped.tides.clear();
  AdjointSolver adjoint(full, {.method = AdjointMethod::Discrete});
  EXPECT_THROW(
      (void)adjoint.IceHistoryKernels(times, measurements, &gapped),
      std::invalid_argument);
}

TEST(AdjointGate, KernelsMatchPyslfp) {
  // The reference is pyslfp's adjoint kernels on its GLQ grid (which
  // coincides with ours at this lmax, isolating implementation error):
  // load duals per node and the branch-0 itau kernel, each method gating
  // its own counterpart — the ported algebra is identical. The resolved
  // regime: the continuous kernels amplify exponentially when stiff
  // (see the generating script), so only here is parity meaningful.
  auto setup = Scene::Make(1e22);
  const auto& grid = setup.solver.Grid();
  ASSERT_EQ(grid.NumberOfLongitudes(), 2 * kLmax + 1)
      << "the GSHTrans grid no longer coincides with pyshtools GLQ here";
  const auto& scales = setup.solver.SolverScales();
  const auto& matScales = setup.material.ModelScales();

  // The fixture's setting: 6 nodes over 10 kyr, linear melt, duals at
  // nodes 3 and 5 (the Cap formulas of the generating script).
  std::ifstream file(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                     "pyslfp_kernels.txt");
  ASSERT_TRUE(file.is_open());

  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  // The potential and angular-velocity duals are SI-anchored (the
  // generating script's PHI_ANCHOR / OMEGA_ANCHOR): pyslfp's EarthModel
  // scales differ from planetmodel/sl3dpp geophysical (phi_nd ~ T^2, a
  // factor 4 pi / 3 — docs/design.md §9), so each code
  // multiplies its dual by ITS OWN factor against the shared anchor and
  // the objective is code-invariant. Lengths need no anchor.
  const double potConv =
      scales.Factor(PotentialDimensions) / 1.0e7;  // PHI_ANCHOR
  const double omConv =
      scales.Factor(Dimensions{.time = -1}) / 1.0e-4;  // OMEGA_ANCHOR
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3].seaLevelDual = grid.FromFunction([](double t, double p) {
    return Cap(t, p, 0.9, 1.2, 0.5);
  });
  Measurement last;
  last.displacementDual = grid.FromFunction([](double t, double p) {
    return 0.7 * Cap(t, p, 2.4, 0.3, 0.6);
  });
  last.potentialDual = grid.FromFunction([&](double t, double p) {
    return 0.4 * potConv * Cap(t, p, 1.4, 3.3, 0.7);
  });
  last.angularVelocityDual = {0.2 * omConv, -0.4 * omConv, 0.8 * omConv};
  measurements[5] = std::move(last);

  // The recorded linear-melt trajectory for the itau kernels.
  const auto& C = setup.state.oceanFunction;
  auto load = [&](double t) {
    auto out = std::vector<double>(setup.state.iceThickness.size());
    const auto fraction = -0.1 * (t / times.back());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = setup.solver.IceDensity() * (1.0 - C[i]) * fraction *
               setup.state.iceThickness[i];
    }
    return out;
  };
  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution evolution(op, setup.solver, setup.state,
                              {.rtol = 1e-12, .record_stride = 1});
  auto snapshots = evolution.Run(times, load);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

  const double loadSi =
      scales.Length() / scales.Factor(Dimensions{.mass = 1, .length = -2});
  const double itauSi =
      std::pow(scales.Length(), 3) * matScales.Time();
  const auto& mesh = setup.material.Mesh();

  constexpr auto tol = 1e-5;
  auto line = std::string{};
  auto method = std::optional<AdjointMethod>{};
  IceKernels kernels;
  auto ours = std::vector<std::complex<double>>{};
  auto coefScale = 0.0, coefWorst = 0.0;
  auto checkedMethods = 0;
  auto closeBlock = [&] {
    if (!method.has_value()) return;
    EXPECT_GT(coefScale, 0.0);
    EXPECT_LT(coefWorst, tol * coefScale)
        << "load duals, method " << static_cast<int>(*method);
    coefScale = coefWorst = 0.0;
  };
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (line.starts_with("method")) {
      closeBlock();
      std::string head, name;
      row >> head >> name;
      method = name == "discrete" ? AdjointMethod::Discrete
                                  : AdjointMethod::Continuous;
      AdjointSolver adjoint(evolution, {.method = *method, .rtol = 1e-12});
      kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
      ++checkedMethods;
      continue;
    }
    if (line.starts_with("node")) {
      auto head = std::string{};
      auto node = 0;
      auto tSi = 0.0;
      row >> head >> node >> tSi;
      EXPECT_NEAR(times[node] * scales.Time(), tSi,
                  1e-6 * std::max(tSi, 1.0));
      ours = grid.Expand(kernels.loadDuals[node]);
      for (auto& v : ours) v *= loadSi;
      continue;
    }
    if (line.starts_with("itau")) {
      auto head = std::string{};
      auto count = std::size_t{0};
      row >> head >> count;
      // The continuous block carries no itau reference (the pyslfp-side
      // continuous accumulation is under investigation; the generating
      // script records why). sl3dpp's own continuous kernel is gated
      // against its exact discrete in ContinuousViscosityKernelsConverge.
      if (count == 0) continue;
      ASSERT_EQ(count, kernels.itauKernel[0].size());
      auto kScale = 0.0, kWorst = 0.0;
      for (auto i = std::size_t{0}; i < count; ++i) {
        std::getline(file, line);
        std::istringstream krow(line);
        auto rSi = 0.0, value = 0.0;
        krow >> rSi >> value;
        const auto e = static_cast<int>(i) / mesh.NGll();
        const auto q = static_cast<int>(i) % mesh.NGll();
        ASSERT_NEAR(mesh.Radius(e, q) * matScales.Length(), rSi,
                    1e-6 * std::max(rSi, 1.0))
            << "radial meshes diverge at node " << i;
        kScale = std::max(kScale, std::abs(value));
        kWorst = std::max(
            kWorst, std::abs(kernels.itauKernel[0][i] * itauSi - value));
      }
      EXPECT_GT(kScale, 0.0);
      EXPECT_LT(kWorst, tol * kScale)
          << "itau kernel, method " << static_cast<int>(*method);
      continue;
    }
    auto l = 0, m = 0;
    auto a = 0.0, b = 0.0;
    row >> l >> m >> a >> b;
    const auto reference = ComplexFromRealPair({a, b}, m);
    coefScale = std::max(coefScale, std::abs(reference));
    coefWorst = std::max(
        coefWorst, std::abs(ours[SurfaceGrid::Index(l, m)] - reference));
  }
  closeBlock();
  EXPECT_EQ(checkedMethods, 2);
}
