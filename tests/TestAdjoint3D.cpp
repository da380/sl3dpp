// The 3D continuous adjoint: the zero-lateral limit — with
// delta = 0 the 3D sweep's ice kernels and spherically reduced viscosity
// kernels must reproduce the 1D continuous sweep's, and so must its load
// kernels under lateral moduli without lateral structure — and the
// finite-difference check of the full 3D log-viscosity kernel with
// lateral structure on, at the continuous formulation's (commute-error)
// accuracy in the resolved regime; under Sdirk2 the same, the sweep stepped
// by its stages, and what is not built for it refused.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
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
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 8;

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
  GRDSolver solver;
  GRDSolver::State state;
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
    ViscoelasticOperator reference(rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
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
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }

  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
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
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  auto load = scene.MeltLoad(-0.1);
  auto directLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });

  // The 1D continuous sweep. At the GLL nodes, where the 3D memory nodes
  // are the 1D kernel's own; TestGaussMemory3D covers the Gauss points.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Evolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  auto snapshots1 = evolution1.Run(times, directLoad);
  const auto trajectory1 = ForwardTrajectory::FromSnapshots(snapshots1);
  AdjointSolver adjoint1(evolution1,
                         {.method = AdjointMethod::Continuous, .rtol = 1e-12});
  const auto kernels1 =
      adjoint1.FirstOrderKernels(times, measurements, &trajectory1);

  // The 3D continuous sweep with delta = 0.
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  Evolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  auto snapshots3 = evolution3.Run(times, directLoad);
  const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(snapshots3);
  AdjointSolver3D adjoint3(evolution3, {.method = AdjointMethod::Continuous,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
  const auto kernels3 =
      adjoint3.FirstOrderKernels(times, measurements, &trajectory3);

  // Ice kernels: node duals, all nodes against the 1D sweep.
  auto scale = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    for (auto i = std::size_t{0}; i < kernels1.directLoadKernels[n].size(); ++i) {
      scale = std::max(scale, std::abs(kernels1.directLoadKernels[n][i]));
      worst = std::max(worst, std::abs(kernels3.ice.directLoadKernels[n][i] -
                                       kernels1.directLoadKernels[n][i]));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-6 * scale);

  // Viscosity kernels: the spherical reduction at each memory node
  // against the 1D nodal kernel at its (element, gll) node.
  const auto& coupling = op3.Coupling();
  const auto ngll = scene.material.Mesh().NGll();
  for (auto k = std::size_t{0}; k < scene.rheology.BranchCount(); ++k) {
    auto kScale = 0.0, kWorst = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
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

TEST(Adjoint3D, ZeroLateralModuliMatchThe1DSweepUnderRotation) {
  // Under lateral moduli the continuous sweep closes its jumps and
  // equilibria by the instantaneous fixed point (a zero-length step with
  // the sea level solve) instead of the Love-number closure. With the
  // moduli's lateral structure one part in 1e100 (no solve sees it), its
  // load kernels are the 1D continuous sweep's, under the axial and the
  // full rotational feedback: a measurement of the potential at the last
  // node (its degree-0 part included, whose adjoint load's own mass the
  // feedback carries), every field at a middle one, and the toroidal
  // potential alone at another (its jump forcing the sea level solve with
  // nothing but the lateral coupling's leak).
  auto scene = Scene::Make(1e21);
  const auto& grid = scene.solver.Grid();
  const auto kyr = 1000.0 * 3.15576e7 / scene.solver.SolverScales().Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  auto field = [&](double scale, double colat, double lon, double width) {
    return grid.FromFunction([=](double t, double p) {
      return scale * Cap(t, p, colat, lon, width);
    });
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].potentialDual = field(0.7, 1.2, 0.8, 0.5);
  auto& every = measurements[3];
  every.relativeSeaLevelDual = field(1.0, 2.0, 3.0, 0.6);
  every.displacementDual = field(-0.5, 0.7, 4.1, 0.4);
  every.potentialDual = field(0.6, 2.4, 5.2, 0.7);
  every.tangentialPotentialDual = field(-0.6, 1.5, 1.9, 0.5);
  every.toroidalPotentialDual = field(0.8, 1.0, 2.6, 0.6);
  every.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[2].toroidalPotentialDual = field(0.8, 2.6, 0.4, 0.5);

  Viscoelastic3D op3(
      scene.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.moduli = {.shear = [](double, double, double) { return 1e-100; },
                  .bulk = {}}});
  ASSERT_TRUE(op3.HasLateralModuli());
  ViscoelasticOperator op1(scene.rheology, kLmax);
  for (const auto rotation :
       {RotationalFeedback::Axial, RotationalFeedback::Full}) {
    Evolution evolution1(op1, scene.solver, scene.state,
                         {.rotation = rotation, .rtol = 1e-12});
    const auto kernels1 =
        AdjointSolver(evolution1,
                      {.method = AdjointMethod::Continuous, .rtol = 1e-12})
            .FirstOrderKernels(times, measurements);
    Evolution3D evolution3(
        op3, scene.solver, scene.state,
        {.rotation = rotation, .rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto kernels3 =
        AdjointSolver3D(evolution3, {.method = AdjointMethod::Continuous,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12})
            .FirstOrderKernels(times, measurements);
    // every view of a surface load, all nodes
    for (const auto& [name, a, b] :
         {std::tuple{"direct load", &kernels1.directLoadKernels,
                     &kernels3.ice.directLoadKernels},
          std::tuple{"load outside the water budget",
                     &kernels1.directLoadOutsideWaterBudgetKernels,
                     &kernels3.ice.directLoadOutsideWaterBudgetKernels},
          std::tuple{"layer on the solid surface",
                     &kernels1.solidSurfaceChangeKernels,
                     &kernels3.ice.solidSurfaceChangeKernels}}) {
      ASSERT_EQ(a->size(), times.size());
      ASSERT_EQ(b->size(), times.size());
      auto scale = 0.0, worst = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        for (auto i = std::size_t{0}; i < (*a)[n].size(); ++i) {
          scale = std::max(scale, std::abs((*a)[n][i]));
          worst = std::max(worst, std::abs((*b)[n][i] - (*a)[n][i]));
        }
      }
      EXPECT_GT(scale, 0.0) << name;
      EXPECT_LT(worst, 1e-10 * scale)
          << name << ", "
          << (rotation == RotationalFeedback::Full ? "full" : "axial")
          << " feedback";
    }
  }
}

TEST(Adjoint3D, SavedAndRegeneratedStrainsAgree) {
  // As the 1D test: a recording that saves the strain and one that keeps
  // only the internal variables (the sweep regenerating each node's
  // strain) give the same 3D viscosity kernels, to the forward tolerance.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.0 * n * kyr);
  auto load = scene.MeltLoad(-0.1);
  auto directLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[4].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

  Viscoelastic3D op(
      scene.rheology, kLmax, [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto kernels = [&](bool recordStrain) {
    Evolution3D evolution(op, scene.solver, scene.state,
                                  {.rtol = 1e-12,
                                   .coupled_rtol = 1e-12,
                                   .record_stride = 1,
                                   .record_strain = recordStrain});
    const auto snapshots = evolution.Run(times, directLoad);
    for (const auto& s : snapshots) {
      EXPECT_EQ(s.internalState->strain.has_value(), recordStrain);
    }
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Continuous,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    return adjoint.FirstOrderKernels(times, measurements, &trajectory)
        .itauKernel3D[0];
  };
  const auto a = kernels(true);
  const auto b = kernels(false);
  ASSERT_EQ(a.size(), b.size());
  auto scale = 0.0, worst = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    scale = std::max(scale, std::abs(a[i]));
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-8 * scale);
}

TEST(Adjoint3D, CheckpointedRecordingsRebuildTheGaps) {
  // A recording that keeps every stride-th state (and the first and last):
  // the sweep rebuilds the states between by stepping forward from the
  // kept one below under the recorded loads, so its viscosity kernels
  // agree with a full recording's to the forward tolerance; the ice
  // kernels, which never read the forward states, are bitwise the same.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 6; ++n) times.push_back(1.5 * n * kyr);
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[6].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });

  const auto op = Viscoelastic3D(
      scene.rheology, kLmax, [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto kernels = [&](int stride) {
    auto evolution = Evolution3D(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = stride});
    const auto snapshots = evolution.Run(times, directLoad);
    for (auto n = std::size_t{0}; n < snapshots.size(); ++n) {
      const auto kept = n % stride == 0 || n + 1 == snapshots.size();
      EXPECT_EQ(snapshots[n].internalState.has_value(), kept) << "node " << n;
    }
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    auto adjoint = AdjointSolver3D(
        evolution, {.method = AdjointMethod::Continuous,
                    .rtol = 1e-12,
                    .coupled_rtol = 1e-12});
    return adjoint.FirstOrderKernels(times, measurements, &trajectory);
  };
  const auto full = kernels(1);
  for (const auto stride : {2, 4}) {
    const auto gapped = kernels(stride);
    // one operator in one process, the same transforms: bitwise
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      ASSERT_EQ(gapped.ice.directLoadKernels[n], full.ice.directLoadKernels[n])
          << "stride " << stride << ", node " << n;
    }
    const auto& a = full.itauKernel3D[0];
    const auto& b = gapped.itauKernel3D[0];
    ASSERT_EQ(a.size(), b.size());
    auto scale = 0.0, worst = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      scale = std::max(scale, std::abs(a[i]));
      worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-10 * scale) << "stride " << stride;
  }
}

TEST(Adjoint3D, LateralLogViscosityKernelAgainstFiniteDifferences) {
  // The full 3D kernel with lateral structure on: dJ/d delta paired
  // with an analytic direction against central differences of the
  // coupled forward. The continuous kernel is exact only up to the
  // discretise/transpose commute error, so the test runs in the
  // resolved regime and asks for agreement at that level, improving
  // under step refinement.
  auto scene = Scene::Make(1e22);
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto load = scene.MeltLoad(-0.1);

  auto run = [&](std::size_t npts, double eps,
                 Kernels3D* kernelsOut) -> double {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    auto directLoad = [&](double t) {
      auto out = std::vector<double>(load.size());
      const auto ramp = t / times.back();
      for (auto i = std::size_t{0}; i < out.size(); ++i)
        out[i] = ramp * load[i];
      return out;
    };
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts - 1].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
        [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) +
                               eps * direction(theta, phi);
                      });
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    auto snapshots = evolution.Run(times, directLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution,
                              {.method = AdjointMethod::Continuous,
                               .rtol = 1e-12,
                               .coupled_rtol = 1e-12});
      *kernelsOut =
          adjoint.FirstOrderKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[npts - 1].relativeSeaLevelDual,
                         snapshots.back().relativeSeaLevelChange);
  };

  // The kernel from the unperturbed run, then central differences.
  auto previousError = 0.0;
  for (const std::size_t npts : {5ul, 9ul}) {
    Kernels3D kernels;
    (void)run(npts, 0.0, &kernels);
    // The direction sampled on the coupling grid (radius-independent).
    Viscoelastic3D probe(scene.rheology, kLmax,
                         [](double, double, double) { return 0.0; });
    const auto& coupling = probe.Coupling();
    const auto& grid = coupling.Grid();
    const auto p = grid.FromFunction(direction);
    auto atNodes = std::vector<double>(coupling.GridValueCount());
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      std::ranges::copy(p, atNodes.begin() + static_cast<std::ptrdiff_t>(
                                                  coupling.FieldOffset(rn)));
    }
    const auto predicted =
        VolumePairing(coupling, kernels.logViscosityKernel3D[0], atNodes);
    const auto eps = 0.1;
    const double fd = (run(npts, eps, nullptr) - run(npts, -eps, nullptr)) /
                      (2.0 * eps);
    const auto error = std::abs(fd - predicted) / std::abs(fd);
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
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  constexpr auto npts = std::size_t{9};
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto load = scene.MeltLoad(-0.1);
  auto times = std::vector<double>{};
  for (auto n = std::size_t{0}; n < npts; ++n) {
    times.push_back(10.0 * static_cast<double>(n) /
                    static_cast<double>(npts - 1) * kyr);
  }
  auto directLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[npts - 1].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

  auto evaluate = [&](double h, Kernels3D* kernelsOut) -> double {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) +
                               h * direction(theta, phi);
                      });
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    auto snapshots = evolution.Run(times, directLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution,
                              {.method = AdjointMethod::Continuous,
                               .rtol = 1e-12,
                               .coupled_rtol = 1e-12});
      *kernelsOut =
          adjoint.FirstOrderKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[npts - 1].relativeSeaLevelDual,
                         snapshots.back().relativeSeaLevelChange);
  };

  Kernels3D kernels;
  const auto J0 = evaluate(0.0, &kernels);

  // <K, p>: the volume pairing over the memory nodes.
  Viscoelastic3D probe(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; });
  const auto& coupling = probe.Coupling();
  const auto& grid = coupling.Grid();
  const auto p = grid.FromFunction(direction);
  auto atNodes = std::vector<double>(coupling.GridValueCount());
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    std::ranges::copy(p, atNodes.begin() + static_cast<std::ptrdiff_t>(
                                                coupling.FieldOffset(rn)));
  }
  const auto predicted =
      VolumePairing(coupling, kernels.logViscosityKernel3D[0], atNodes);
  ASSERT_NE(predicted, 0.0);

  const auto hs = std::vector<double>{0.2, 0.1, 0.05, 0.025};
  std::vector<double> r0(hs.size()), r1(hs.size());
  for (auto i = std::size_t{0}; i < hs.size(); ++i) {
    const auto J = evaluate(hs[i], nullptr);
    r0[i] = std::abs(J - J0);
    r1[i] = std::abs(J - J0 - hs[i] * predicted);
  }

  // Zeroth order: the perturbation registers at O(h).
  for (auto i = std::size_t{1}; i < hs.size(); ++i) {
    const auto rate = std::log2(r0[i - 1] / r0[i]);
    EXPECT_GT(rate, 0.8) << "R0 rates: " << r0[0] << " " << r0[1] << " "
                         << r0[2] << " " << r0[3];
    EXPECT_LT(rate, 1.3) << "R0 rate " << rate << " at h " << hs[i];
  }
  // Gradient-corrected: second order at the curvature-dominated end,
  // and never worse than the commute floor anywhere.
  const auto firstRate = std::log2(r1[0] / r1[1]);
  EXPECT_GT(firstRate, 1.5) << "R1: " << r1[0] << " " << r1[1] << " "
                            << r1[2] << " " << r1[3]
                            << " (predicted " << predicted << ")";
  for (auto i = std::size_t{0}; i < hs.size(); ++i) {
    EXPECT_LT(r1[i], 0.1 * hs[i] * std::abs(predicted))
        << "commute floor exceeded at h " << hs[i];
    if (i > 0) {
      EXPECT_LT(r1[i], r1[i - 1]) << "R1 not decreasing at h " << hs[i];
    }
  }
}

TEST(Adjoint3D, Sdirk2ZeroLateralKernelsMatchThe1DSweep) {
  // Under Sdirk2 the 3D continuous sweep with delta = 0 steps the same
  // stages as the 1D one: ice kernels and the spherically reduced viscosity
  // kernels to the pipeline's round-off, as under the trapezoid
  // (ZeroLateralKernelsMatchThe1DSweep).
  auto scene = Scene::Make(1e21);
  const auto kyr = 1000.0 * 3.15576e7 / scene.solver.SolverScales().Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  auto load = scene.MeltLoad(-0.1);
  auto directLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Evolution evolution1(
      op1, scene.solver, scene.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
  const AdjointSolver adjoint1(
      evolution1, {.method = AdjointMethod::Continuous, .rtol = 1e-12});
  const auto kernels1 =
      adjoint1.FirstOrderKernels(times, measurements, &trajectory1);
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  Evolution3D evolution3(op3, scene.solver, scene.state,
                         {.scheme = Scheme::Sdirk2,
                          .rtol = 1e-12,
                          .coupled_rtol = 1e-12,
                          .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
  const AdjointSolver3D adjoint3(evolution3,
                                 {.method = AdjointMethod::Continuous,
                                  .rtol = 1e-12,
                                  .coupled_rtol = 1e-12});
  const auto kernels3 =
      adjoint3.FirstOrderKernels(times, measurements, &trajectory3);
  auto scale = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    for (auto i = std::size_t{0}; i < kernels1.directLoadKernels[n].size();
         ++i) {
      scale = std::max(scale, std::abs(kernels1.directLoadKernels[n][i]));
      worst = std::max(worst, std::abs(kernels3.ice.directLoadKernels[n][i] -
                                       kernels1.directLoadKernels[n][i]));
    }
  }
  const auto& coupling = op3.Coupling();
  const auto ngll = scene.material.Mesh().NGll();
  auto kScale = 0.0, kWorst = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full =
        static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
        coupling.NodeInElementOf(rn);
    kScale = std::max(kScale, std::abs(kernels1.itauKernel[0][full]));
    kWorst = std::max(kWorst, std::abs(kernels3.itauKernelSpherical[0][rn] -
                                       kernels1.itauKernel[0][full]));
  }
  std::cout << std::format("[sdirk2 3D kernels] ice {:.2e}, viscosity {:.2e}\n",
                           worst / scale, kWorst / kScale);
  // measured: 2e-15 and 1.7e-12
  EXPECT_GT(scale, 0.0);
  EXPECT_GT(kScale, 0.0);
  EXPECT_LT(worst, 1e-6 * scale);
  EXPECT_LT(kWorst, 1e-6 * kScale);
}

TEST(Adjoint3D, Sdirk2LateralLogViscosityKernelAgainstFiniteDifferences) {
  // LateralLogViscosityKernelAgainstFiniteDifferences under Sdirk2: the
  // full 3D log-viscosity kernel with lateral structure on, the continuous
  // sweep stepped by Sdirk2's stages, against central differences of
  // Sdirk2 forward runs at the same steps, closer as the steps halve.
  auto scene = Scene::Make(1e22);
  const auto kyr = 1000.0 * 3.15576e7 / scene.solver.SolverScales().Time();
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto load = scene.MeltLoad(-0.1);
  auto run = [&](std::size_t npts, double eps, Kernels3D* kernelsOut) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    auto directLoad = [&](double t) {
      auto out = std::vector<double>(load.size());
      const auto ramp = t / times.back();
      for (auto i = std::size_t{0}; i < out.size(); ++i)
        out[i] = ramp * load[i];
      return out;
    };
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts - 1].relativeSeaLevelDual =
        scene.solver.Grid().FromFunction(
            [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
    Viscoelastic3D op(
        scene.rheology, kLmax, [&](double r, double theta, double phi) {
          return baseDelta(r, theta, phi) + eps * direction(theta, phi);
        });
    Evolution3D evolution(op, scene.solver, scene.state,
                          {.scheme = Scheme::Sdirk2,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = kernelsOut != nullptr ? 1 : 0});
    auto snapshots = evolution.Run(times, directLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      const AdjointSolver3D adjoint(evolution,
                                    {.method = AdjointMethod::Continuous,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12});
      *kernelsOut = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[npts - 1].relativeSeaLevelDual,
                         snapshots.back().relativeSeaLevelChange);
  };
  Viscoelastic3D probe(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; });
  const auto& coupling = probe.Coupling();
  const auto p = coupling.Grid().FromFunction(direction);
  auto atNodes = std::vector<double>(coupling.GridValueCount());
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    std::ranges::copy(p, atNodes.begin() + static_cast<std::ptrdiff_t>(
                                               coupling.FieldOffset(rn)));
  }
  auto errors = std::vector<double>{};
  for (const std::size_t npts : {5ul, 9ul}) {
    Kernels3D kernels;
    static_cast<void>(run(npts, 0.0, &kernels));
    const auto predicted =
        VolumePairing(coupling, kernels.logViscosityKernel3D[0], atNodes);
    const auto eps = 0.1;
    const double fd =
        (run(npts, eps, nullptr) - run(npts, -eps, nullptr)) / (2.0 * eps);
    errors.push_back(std::abs(fd - predicted) / std::abs(fd));
  }
  std::cout << std::format(
      "[sdirk2 3D kernel] against differences {:.3e} {:.3e}\n", errors[0],
      errors[1]);
  // measured: 1.1 % at 2.5 kyr steps, 0.25 % at 1.25 kyr
  EXPECT_LT(errors[0], 0.03);
  EXPECT_LT(errors[1], 0.35 * errors[0]);
}

TEST(Adjoint3D, Sdirk2RefusesWhatIsNotBuilt) {
  // About a 3D run under Sdirk2 the derivatives of the stepped map are
  // built (TestSdirk2Derivatives). What is not, each refused by name: the
  // continuous second-order sweep, the continuous drive of a tangent-linear
  // run along the viscosity, and a trajectory that does not give the steps'
  // first-stage closures.
  auto scene = Scene::Make(1e21);
  const auto kyr = 1000.0 * 3.15576e7 / scene.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, 2.0 * kyr, 3.0 * kyr};
  const auto load = scene.MeltLoad(-0.1);
  auto directLoad = [&](double t) {
    auto out = load;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3].relativeSeaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  Viscoelastic3D op(
      scene.rheology, kLmax, [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto named = [](const std::function<void()>& call, std::string_view what) {
    try {
      call();
      ADD_FAILURE() << what << " ran under Sdirk2";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("Sdirk2"), std::string::npos)
          << what << ": " << error.what();
    }
  };
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .record_stride = 1});
  const auto snapshots = evolution.Run(times, directLoad);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
  // the discrete sweep and the discrete tangent-linear run are built
  EXPECT_FALSE(AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete})
                   .FirstOrderKernels(times, measurements, &trajectory)
                   .itauKernel3D.empty());
  TangentPerturbation3D alongViscosity;
  alongViscosity.logViscosity.assign(op.Coupling().GridValueCount(), 0.1);
  const TangentLinearSolver3D recorded(
      evolution, {.method = AdjointMethod::Discrete, .record = true});
  const auto tangent = recorded.Run(times, alongViscosity, &trajectory);
  named(
      [&] {
        const AdjointSolver3D adjoint(evolution,
                                      {.method = AdjointMethod::Continuous});
        static_cast<void>(adjoint.SecondOrderKernels(
            times, measurements, tangent, {}, trajectory));
      },
      "the continuous second-order sweep");
  named(
      [&] {
        const TangentLinearSolver3D continuous(
            evolution, {.method = AdjointMethod::Continuous});
        static_cast<void>(continuous.Run(times, alongViscosity, &trajectory));
      },
      "the continuous tangent-linear run along the viscosity");
  auto bare = trajectory;
  bare.stages.clear();
  named(
      [&] {
        const AdjointSolver3D adjoint(evolution,
                                      {.method = AdjointMethod::Discrete});
        static_cast<void>(adjoint.FirstOrderKernels(times, measurements, &bare));
      },
      "the discrete sweep about a trajectory without its stages");
  // a recording with gaps that does not give them: the continuous sweep's
  // rebuild repeats each stage under its closure
  Evolution3D strided(
      op, scene.solver, scene.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .record_stride = 2});
  auto gapped =
      ForwardTrajectory3D::FromSnapshots(strided.Run(times, directLoad));
  gapped.stages.clear();
  named(
      [&] {
        const AdjointSolver3D adjoint(strided,
                                      {.method = AdjointMethod::Continuous});
        static_cast<void>(
            adjoint.FirstOrderKernels(times, measurements, &gapped));
      },
      "the continuous sweep about a gapped trajectory without its stages");
}
