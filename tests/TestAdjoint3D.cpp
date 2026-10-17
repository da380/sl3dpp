// The 3D continuous adjoint (stage 5d): the zero-lateral gate — with
// delta = 0 the 3D sweep's ice kernels and spherically reduced viscosity
// kernels must reproduce the 1D continuous sweep's — and the
// finite-difference check of the full 3D log-viscosity kernel with
// lateral structure on, at the continuous formulation's (commute-error)
// accuracy in the resolved regime.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr int kLmax = 8;

double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const double d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;
  std::vector<double> iceNd;

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
    std::map<std::size_t, RadialValue> viscosity;
    const auto b = model.DimensionlessView().Boundaries();
    for (std::size_t i = 0; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i,
                          viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator reference(rheology, kLmax);
    SeaLevelSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                          SeaLevelParameters{});
    Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model),  std::move(material), std::move(rheology),
                 std::move(solver), std::move(state),    std::move(ice)};
  }

  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    std::vector<double> load(iceNd.size());
    for (std::size_t i = 0; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }

  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const double a = solver.Radius();
    std::vector<double> fg(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }
};

}  // namespace

TEST(Adjoint3D, ZeroLateralKernelsMatchThe1DSweep) {
  // With no lateral perturbation the 3D continuous sweep is the 1D one
  // computed through the grid pipeline: ice kernels and the spherically
  // reduced viscosity kernels must agree to the pipeline's round-off.
  auto scene = Scene::Make(1e21);
  const auto& scales = scene.solver.SolverScales();
  const double kyr = 1000.0 * 3.15576e7 / scales.Time();
  std::vector<double> times;
  for (int n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  auto load = scene.MeltLoad(-0.1);
  auto iceLoad = [&](double t) {
    std::vector<double> out(load.size());
    const double ramp = t / times.back();
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  std::map<std::size_t, Measurement> measurements;
  measurements[5].seaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });

  // The 1D continuous sweep. At the GLL nodes, where the 3D memory nodes
  // are the 1D kernel's own; TestGaussMemory3D is the Gauss-point gate.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  auto snapshots1 = evolution1.Run(times, iceLoad);
  const auto trajectory1 = ForwardTrajectory::FromSnapshots(snapshots1);
  AdjointSolver adjoint1(evolution1,
                         {.method = AdjointMethod::Continuous, .rtol = 1e-12});
  const auto kernels1 =
      adjoint1.IceHistoryKernels(times, measurements, &trajectory1);

  // The 3D continuous sweep with delta = 0.
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  auto snapshots3 = evolution3.Run(times, iceLoad);
  const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(snapshots3);
  AdjointSolver3D adjoint3(evolution3,
                           {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto kernels3 =
      adjoint3.IceHistoryKernels(times, measurements, &trajectory3);

  // Ice kernels: node duals, all nodes against the 1D sweep.
  double scale = 0.0, worst = 0.0;
  for (std::size_t n = 0; n < times.size(); ++n) {
    for (std::size_t i = 0; i < kernels1.loadDuals[n].size(); ++i) {
      scale = std::max(scale, std::abs(kernels1.loadDuals[n][i]));
      worst = std::max(worst, std::abs(kernels3.ice.loadDuals[n][i] -
                                       kernels1.loadDuals[n][i]));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-6 * scale);

  // Viscosity kernels: the spherical reduction at each memory node
  // against the 1D nodal kernel at its (element, gll) node.
  const auto& coupling = op3.Coupling();
  const int ngll = scene.material.Mesh().NGll();
  for (std::size_t k = 0; k < scene.rheology.BranchCount(); ++k) {
    double kScale = 0.0, kWorst = 0.0;
    for (int rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) *
                            ngll +
                        coupling.NodeInElementOf(rn);
      kScale = std::max(kScale, std::abs(kernels1.itauKernel[k][full]));
      kWorst = std::max(kWorst,
                        std::abs(kernels3.itauKernelSpherical[k][rn] -
                                 kernels1.itauKernel[k][full]));
    }
    EXPECT_GT(kScale, 0.0);
    EXPECT_LT(kWorst, 1e-6 * kScale) << "branch " << k;
  }
}

TEST(Adjoint3D, SavedAndRegeneratedStrainsAgree) {
  // As the 1D test: a recording that saves the strain and one that keeps
  // only the internal variables (the sweep regenerating each node's
  // strain) give the same 3D viscosity kernels, to the forward tolerance.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const double kyr = 1000.0 * 3.15576e7 / scales.Time();
  std::vector<double> times;
  for (int n = 0; n <= 4; ++n) times.push_back(2.0 * n * kyr);
  auto load = scene.MeltLoad(-0.1);
  auto iceLoad = [&](double t) {
    std::vector<double> out(load.size());
    const double ramp = t / times.back();
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  std::map<std::size_t, Measurement> measurements;
  measurements[4].seaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

  Viscoelastic3D op(
      scene.rheology, kLmax, [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto kernels = [&](bool recordStrain) {
    SeaLevelEvolution3D evolution(op, scene.solver, scene.state,
                                  {.rtol = 1e-12,
                                   .coupled_rtol = 1e-12,
                                   .record_stride = 1,
                                   .record_strain = recordStrain});
    const auto snapshots = evolution.Run(times, iceLoad);
    for (const auto& s : snapshots) {
      EXPECT_EQ(s.internalState->strain.has_value(), recordStrain);
    }
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    AdjointSolver3D adjoint(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12});
    return adjoint.IceHistoryKernels(times, measurements, &trajectory)
        .itauKernel3D[0];
  };
  const auto a = kernels(true);
  const auto b = kernels(false);
  ASSERT_EQ(a.size(), b.size());
  double scale = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    scale = std::max(scale, std::abs(a[i]));
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-8 * scale);
}

TEST(Adjoint3D, LateralLogViscosityKernelAgainstFiniteDifferences) {
  // The full 3D kernel with lateral structure on: dJ/d delta paired
  // with an analytic direction against central differences of the
  // coupled forward. The continuous kernel is exact only up to the
  // discretise/transpose commute error, so the gate runs in the
  // resolved regime and asks for agreement at that level, improving
  // under step refinement.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const double kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto load = scene.MeltLoad(-0.1);

  auto run = [&](std::size_t npts, double eps,
                 Kernels3D* kernelsOut) -> double {
    std::vector<double> times;
    for (std::size_t n = 0; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    auto iceLoad = [&](double t) {
      std::vector<double> out(load.size());
      const double ramp = t / times.back();
      for (std::size_t i = 0; i < out.size(); ++i) out[i] = ramp * load[i];
      return out;
    };
    std::map<std::size_t, Measurement> measurements;
    measurements[npts - 1].seaLevelDual = scene.solver.Grid().FromFunction(
        [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) +
                               eps * direction(theta, phi);
                      });
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    auto snapshots = evolution.Run(times, iceLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution,
                              {.rtol = 1e-12, .coupled_rtol = 1e-12});
      *kernelsOut =
          adjoint.IceHistoryKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[npts - 1].seaLevelDual,
                         snapshots.back().seaLevelChange);
  };

  // The kernel from the unperturbed run, then central differences.
  double previousError = 0.0;
  for (const std::size_t npts : {5ul, 9ul}) {
    Kernels3D kernels;
    (void)run(npts, 0.0, &kernels);
    // The direction sampled on the coupling grid (radius-independent).
    Viscoelastic3D probe(scene.rheology, kLmax,
                         [](double, double, double) { return 0.0; });
    const auto& coupling = probe.Coupling();
    const auto& grid = coupling.Grid();
    const auto p = grid.FromFunction(direction);
    const auto gridF = grid.FieldSize();
    double predicted = 0.0;
    std::vector<double> product(gridF);
    for (int rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto row = std::span(kernels.logViscosityKernel3D[0])
                           .subspan(static_cast<std::size_t>(rn) * gridF,
                                    gridF);
      for (std::size_t x = 0; x < gridF; ++x) product[x] = row[x] * p[x];
      predicted += grid.Integrate(product);
    }
    const double eps = 0.1;
    const double fd = (run(npts, eps, nullptr) - run(npts, -eps, nullptr)) /
                      (2.0 * eps);
    const double error = std::abs(fd - predicted) / std::abs(fd);
    EXPECT_LT(error, 0.1) << "npts " << npts;
    if (npts > 5) {
      EXPECT_LT(error, previousError) << "no refinement improvement";
    }
    previousError = error;
  }
}

TEST(Adjoint3D, TaylorRemainderConvergence) {
  // The canonical Taylor test of the 3D log-viscosity kernel: along an
  // analytic direction p, the zeroth remainder |J(delta + h p) - J|
  // falls at O(h) while the gradient-corrected remainder
  // |J(delta + h p) - J - h <K, p>| falls at O(h^2) — until it meets
  // the continuous formulation's commute floor (h times the
  // discretise/transpose error), where the observed order relaxes
  // towards one. The machine-grade version of this test (second order
  // to solver precision) belongs to the discrete 3D transpose.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const double kyr = 1000.0 * 3.15576e7 / scales.Time();
  constexpr std::size_t npts = 9;
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto load = scene.MeltLoad(-0.1);
  std::vector<double> times;
  for (std::size_t n = 0; n < npts; ++n) {
    times.push_back(10.0 * static_cast<double>(n) /
                    static_cast<double>(npts - 1) * kyr);
  }
  auto iceLoad = [&](double t) {
    std::vector<double> out(load.size());
    const double ramp = t / times.back();
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  std::map<std::size_t, Measurement> measurements;
  measurements[npts - 1].seaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

  auto evaluate = [&](double h, Kernels3D* kernelsOut) -> double {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) +
                               h * direction(theta, phi);
                      });
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    auto snapshots = evolution.Run(times, iceLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution,
                              {.rtol = 1e-12, .coupled_rtol = 1e-12});
      *kernelsOut =
          adjoint.IceHistoryKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[npts - 1].seaLevelDual,
                         snapshots.back().seaLevelChange);
  };

  Kernels3D kernels;
  const double J0 = evaluate(0.0, &kernels);

  // <K, p>: the angular-density pairing per memory node.
  Viscoelastic3D probe(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; });
  const auto& coupling = probe.Coupling();
  const auto& grid = coupling.Grid();
  const auto p = grid.FromFunction(direction);
  const auto gridF = grid.FieldSize();
  double predicted = 0.0;
  std::vector<double> product(gridF);
  for (int rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto row =
        std::span(kernels.logViscosityKernel3D[0])
            .subspan(static_cast<std::size_t>(rn) * gridF, gridF);
    for (std::size_t x = 0; x < gridF; ++x) product[x] = row[x] * p[x];
    predicted += grid.Integrate(product);
  }
  ASSERT_NE(predicted, 0.0);

  const std::vector<double> hs{0.2, 0.1, 0.05, 0.025};
  std::vector<double> r0(hs.size()), r1(hs.size());
  for (std::size_t i = 0; i < hs.size(); ++i) {
    const double J = evaluate(hs[i], nullptr);
    r0[i] = std::abs(J - J0);
    r1[i] = std::abs(J - J0 - hs[i] * predicted);
  }

  // Zeroth order: the perturbation registers at O(h).
  for (std::size_t i = 1; i < hs.size(); ++i) {
    const double rate = std::log2(r0[i - 1] / r0[i]);
    EXPECT_GT(rate, 0.8) << "R0 rates: " << r0[0] << " " << r0[1] << " "
                         << r0[2] << " " << r0[3];
    EXPECT_LT(rate, 1.3) << "R0 rate " << rate << " at h " << hs[i];
  }
  // Gradient-corrected: second order at the curvature-dominated end,
  // and never worse than the commute floor anywhere.
  const double firstRate = std::log2(r1[0] / r1[1]);
  EXPECT_GT(firstRate, 1.5) << "R1: " << r1[0] << " " << r1[1] << " "
                            << r1[2] << " " << r1[3]
                            << " (predicted " << predicted << ")";
  for (std::size_t i = 0; i < hs.size(); ++i) {
    EXPECT_LT(r1[i], 0.1 * hs[i] * std::abs(predicted))
        << "commute floor exceeded at h " << hs[i];
    if (i > 0) {
      EXPECT_LT(r1[i], r1[i - 1]) << "R1 not decreasing at h " << hs[i];
    }
  }
}
