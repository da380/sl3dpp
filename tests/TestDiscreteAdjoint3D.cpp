// The discrete 3D adjoint: the pairing identity with the tangent-linear
// run at solver tolerance (the exact transpose, loads and a log-viscosity
// direction alike), the zero-lateral limit equal to the 1D discrete
// sweep, the machine-grade Taylor test of the 3D log-viscosity kernel
// (second order to solver precision at every step), and the continuous
// sweep's gap closing under refinement — docs/dev/adjoints.md.

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
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
#include "sl3dpp/TangentLinear.hpp"
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

  [[nodiscard]] double Kyr() const {
    return 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] double Sigma() const {
    return 1.0 / solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  }
  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }
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

std::vector<double> OnCoupling(const GridCoupling& coupling,
                               const std::function<double(double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto values = coupling.GridOf(rn).FromFunction(f);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

double GridPairing(const GridCoupling& coupling,
                   const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& direction) {
  auto total = 0.0;
  for (auto k = std::size_t{0}; k < kernel.size(); ++k) {
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto at = coupling.FieldOffset(rn);
      const auto size = coupling.FieldSizeOf(rn);
      auto product = std::vector<double>(size);
      for (auto x = std::size_t{0}; x < size; ++x) {
        product[x] = kernel[k][at + x] * direction[k][at + x];
      }
      total += coupling.GridOf(rn).Integrate(product);
    }
  }
  return total;
}

std::function<std::vector<double>(double)> Ramp(const std::vector<double>& load,
                                                double span) {
  return [load, span](double t) {
    auto out = load;
    for (auto& v : out) v *= t / span;
    return out;
  };
}

double Objective(const Scene& scene,
                 const std::vector<SeaLevelEvolution3D::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.seaLevelDual.empty()) {
      total += scene.Surface(m.seaLevelDual, snap.seaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      total += scene.Surface(m.displacementDual, snap.displacement);
    }
    if (!m.potentialDual.empty()) {
      total += scene.Surface(m.potentialDual, snap.potential);
    }
    for (auto j = 0; j < 3; ++j) {
      total += m.angularVelocityDual[j] * snap.angularVelocity[j];
    }
  }
  return total;
}

double MaxAbs(std::span<const double> f) {
  auto out = 0.0;
  for (auto v : f) out = std::max(out, std::abs(v));
  return out;
}

double MaxDiff(std::span<const double> a, std::span<const double> b) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    out = std::max(out, std::abs(a[i] - b[i]));
  }
  return out;
}

}  // namespace

TEST(DiscreteAdjoint3D, PairsWithTheTangentLinearRun) {
  // The exact transpose: the objective of the discrete tangent-linear
  // run along (delta f, delta itau) equals the discrete sweep's load duals
  // paired with delta f (node 0 included) plus its itau kernel paired with
  // delta itau, to the solvers' tolerance, with lateral structure on and
  // a full measurement set; each argument alone pairs with its own kernel.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(13);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = scene.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = scene.Smooth(rng, 1.0);
  middle.displacementDual = scene.Smooth(rng, -0.5);
  measurements[2] = std::move(middle);
  Measurement last;
  last.displacementDual = scene.Smooth(rng, 0.7);
  last.potentialDual = scene.Smooth(rng, 0.4);
  last.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[4] = std::move(last);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
      return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
    });
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    TangentPerturbation3D d;
    d.directLoad = deltas;
    d.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto ditau = TangentLinearSolver3D::ItauOfLogViscosity(op, d.logViscosity);

    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += scene.Surface(kernels.ice.loadDuals[n], deltas[n]);
    }
    const auto itauRhs = GridPairing(op.Coupling(), kernels.itauKernel3D, ditau);
    const auto lhs = Objective(scene, tangent.Run(times, d, &trajectory), measurements);
    EXPECT_NEAR(lhs, loadsRhs + itauRhs, 1e-8 * std::abs(lhs))
        << "scheme " << static_cast<int>(scheme);

    TangentPerturbation3D loadsOnly;
    loadsOnly.directLoad = deltas;
    EXPECT_NEAR(Objective(scene, tangent.Run(times, loadsOnly), measurements),
                loadsRhs, 1e-8 * std::abs(loadsRhs))
        << "scheme " << static_cast<int>(scheme);
    TangentPerturbation3D itauOnly;
    itauOnly.logViscosity = d.logViscosity;
    EXPECT_NEAR(
        Objective(scene, tangent.Run(times, itauOnly, &trajectory), measurements),
        itauRhs, 1e-8 * std::abs(itauRhs))
        << "scheme " << static_cast<int>(scheme);

    // A stride-2 trajectory serves the same sweep.
    SeaLevelEvolution3D sparse(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 2});
    const auto sparseTrajectory =
        ForwardTrajectory3D::FromSnapshots(sparse.Run(times, iceLoad));
    const auto again =
        adjoint.IceHistoryKernels(times, measurements, &sparseTrajectory);
    EXPECT_NEAR(GridPairing(op.Coupling(), again.itauKernel3D, ditau), itauRhs,
                1e-9 * std::abs(itauRhs));
  }
}

TEST(DiscreteAdjoint3D, ZeroLateralMatchesThe1DDiscreteSweep) {
  // With delta = 0 at the GLL points the discrete 3D sweep is the 1D
  // discrete sweep through the grid pipeline: the load duals and the
  // spherically reduced itau kernel at the pipeline's round-off.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto iceLoad = Ramp(scene.MeltLoad(-0.1), times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[3].displacementDual = scene.Smooth(rng, -0.6);
  measurements[5].angularVelocityDual = {0.1, 0.3, -0.2};

  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, iceLoad));
  const auto kernels1 =
      AdjointSolver(evolution1, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .IceHistoryKernels(times, measurements, &trajectory1);

  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, iceLoad));
  const auto kernels3 =
      AdjointSolver3D(evolution3, {.method = AdjointMethod::Discrete,
                                   .rtol = 1e-12,
                                   .coupled_rtol = 1e-12})
          .IceHistoryKernels(times, measurements, &trajectory3);

  auto scale = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    scale = std::max(scale, MaxAbs(kernels1.loadDuals[n]));
    worst = std::max(worst, MaxDiff(kernels3.ice.loadDuals[n], kernels1.loadDuals[n]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-8 * scale);
  const auto& coupling = op3.Coupling();
  const auto ngll = scene.material.Mesh().NGll();
  auto kScale = 0.0, kWorst = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    kScale = std::max(kScale, std::abs(kernels1.itauKernel[0][full]));
    kWorst = std::max(kWorst, std::abs(kernels3.itauKernelSpherical[0][rn] -
                                       kernels1.itauKernel[0][full]));
  }
  EXPECT_GT(kScale, 0.0);
  EXPECT_LT(kWorst, 1e-8 * kScale);
}

TEST(DiscreteAdjoint3D, MachineGradeTaylorTest) {
  // Along a log-viscosity direction p with lateral structure on, the
  // gradient-corrected remainder |J(delta + h p) - J - h <K, p>| falls at
  // second order at EVERY step, to solver precision — the test the
  // continuous kernel can only pass down to its commute floor.
  auto scene = Scene::Make(1e22);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto iceLoad = Ramp(scene.MeltLoad(-0.1), times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[4].seaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  auto evaluate = [&](double h, Kernels3D* kernelsOut) {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) + h * direction(theta, phi);
                      });
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    const auto snapshots = evolution.Run(times, iceLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      *kernelsOut = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                               .rtol = 1e-12,
                                               .coupled_rtol = 1e-12})
                        .IceHistoryKernels(times, measurements, &trajectory);
    }
    return Objective(scene, snapshots, measurements);
  };
  Kernels3D kernels;
  const auto J0 = evaluate(0.0, &kernels);
  Viscoelastic3D probe(scene.rheology, kLmax, baseDelta);
  const auto p = OnCoupling(probe.Coupling(), direction);
  const auto predicted = GridPairing(probe.Coupling(), kernels.logViscosityKernel3D, {p});
  ASSERT_NE(predicted, 0.0);
  const auto hs = std::vector<double>{0.2, 0.1, 0.05, 0.025};
  auto r1 = std::vector<double>{};
  for (const auto h : hs) {
    r1.push_back(std::abs(evaluate(h, nullptr) - J0 - h * predicted));
  }
  for (auto i = std::size_t{1}; i < hs.size(); ++i) {
    const auto rate = std::log2(r1[i - 1] / r1[i]);
    EXPECT_GT(rate, 1.9) << "R1: " << r1[0] << " " << r1[1] << " " << r1[2]
                         << " " << r1[3] << " (predicted " << predicted << ")";
    EXPECT_LT(rate, 2.1) << "rate " << rate << " at h " << hs[i];
  }
  // Richardson on central differences lands on the kernel's pairing.
  const auto fd = [&](double h) {
    return (evaluate(h, nullptr) - evaluate(-h, nullptr)) / (2.0 * h);
  };
  const auto richardson = (4.0 * fd(0.01) - fd(0.02)) / 3.0;
  EXPECT_NEAR(richardson, predicted, 1e-7 * std::abs(predicted));
}

TEST(DiscreteAdjoint3D, ContinuousGapClosesUnderRefinement) {
  // The resolution diagnostic: the continuous sweep's kernels differ from
  // the discrete ones by the commute error, which falls as the steps are
  // halved, in the loads and in the log-viscosity kernel alike.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(19);
  const auto load = scene.MeltLoad(-0.1);
  const auto g = scene.Smooth(rng, 1.0);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto p = OnCoupling(op.Coupling(), direction);
  auto loadGaps = std::vector<double>{}, viscosityGaps = std::vector<double>{};
  for (const auto nodes : {9, 17}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Ramp(load, span)));
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].seaLevelDual = g;
    auto results = std::map<AdjointMethod, std::pair<double, double>>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver3D(evolution, {.method = method, .rtol = 1e-12,
                                      .coupled_rtol = 1e-12})
              .IceHistoryKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.loadDuals[n], Ramp(load, span)(times[n]));
      }
      results[method] = {loads, GridPairing(op.Coupling(),
                                            kernels.logViscosityKernel3D, {p})};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    loadGaps.push_back(std::abs(d.first - c.first) / std::abs(d.first));
    viscosityGaps.push_back(std::abs(d.second - c.second) / std::abs(d.second));
  }
  for (const auto* gaps : {&loadGaps, &viscosityGaps}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_GT((*gaps)[0] / (*gaps)[1], 2.5) << (*gaps)[0] << " " << (*gaps)[1];
  }
}

// -- the material ------------------------------------------------------------

namespace {

double FieldPairing(const GridCoupling& coupling, std::span<const double> kernel,
                    std::span<const double> direction) {
  auto total = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto at = coupling.FieldOffset(rn);
    const auto size = coupling.FieldSizeOf(rn);
    auto product = std::vector<double>(size);
    for (auto x = std::size_t{0}; x < size; ++x) {
      product[x] = kernel[at + x] * direction[at + x];
    }
    total += coupling.GridOf(rn).Integrate(product);
  }
  return total;
}

/// A radial profile times an angular function over the grid values, in
/// units of a nodal modulus.
std::vector<double> LateralModulus(const GridCoupling& coupling,
                                   const Material& material,
                                   std::span<const double> modulus,
                                   const std::function<double(double)>& radial,
                                   const std::function<double(double, double)>& angular) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    const auto r = coupling.RadiusOfNode(rn);
    const auto values = coupling.GridOf(rn).FromFunction(
        [&](double theta, double phi) {
          return modulus[full] * radial(r) * angular(theta, phi);
        });
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

}  // namespace

TEST(DiscreteAdjoint3D, MaterialKernelsPairWithTheTangent) {
  // The exact transpose in the material: with lateral viscosity on, the
  // discrete material kernels paired with LATERAL directions in kappa and
  // mu equal the 3D material tangent's objective change, both schemes,
  // each modulus alone and together. Refused by the continuous sweep and
  // without a trajectory.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = scene.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = scene.Smooth(rng, 1.0);
  middle.displacementDual = scene.Smooth(rng, -0.5);
  measurements[2] = std::move(middle);
  Measurement last;
  last.displacementDual = scene.Smooth(rng, 0.7);
  last.potentialDual = scene.Smooth(rng, 0.4);
  measurements[4] = std::move(last);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [](double, double theta, double phi) {
                        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
                      },
                      {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    const auto& coupling = op.Coupling();
    MaterialDirection3D d;
    d.kappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    d.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });

    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12,
                                        .material_kernels = true});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    ASSERT_EQ(kernels.kappaKernel3D.size(), coupling.GridValueCount());
    ASSERT_EQ(kernels.muKernel3D.size(), coupling.GridValueCount());
    const auto kappaRhs = FieldPairing(coupling, kernels.kappaKernel3D, d.kappa);
    const auto muRhs = FieldPairing(coupling, kernels.muKernel3D, d.mu);

    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    for (const auto& [name, kappa, mu, rhs] :
         {std::tuple{"kappa", true, false, kappaRhs},
          std::tuple{"mu", false, true, muRhs},
          std::tuple{"both", true, true, kappaRhs + muRhs}}) {
      TangentPerturbation3D direction;
      if (kappa) direction.material.kappa = d.kappa;
      if (mu) direction.material.mu = d.mu;
      const auto lhs =
          Objective(scene, tangent.Run(times, direction, &trajectory), measurements);
      EXPECT_NEAR(lhs, rhs, 1e-8 * std::abs(lhs))
          << name << " scheme " << static_cast<int>(scheme);
      EXPECT_NE(lhs, 0.0) << name;
    }

    // The spherical reductions pair with angle-uniform directions.
    auto uniform = std::vector<double>(coupling.GridValueCount(), 0.0);
    auto nodal = std::vector<double>(coupling.NodeCount(), 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      nodal[rn] = m0[static_cast<std::size_t>(coupling.ElementOfNode(rn)) *
                         scene.material.Mesh().NGll() +
                     coupling.NodeInElementOf(rn)] *
                  0.1 * coupling.RadiusOfNode(rn);
      const auto at = coupling.FieldOffset(rn);
      for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
        uniform[at + x] = nodal[rn];
      }
    }
    auto spherical = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      spherical += kernels.muKernelSpherical[rn] * nodal[rn];
    }
    const auto full = FieldPairing(coupling, kernels.muKernel3D, uniform);
    EXPECT_NEAR(spherical, full, 1e-12 * std::abs(full));

    // The refusal without a trajectory.
    EXPECT_THROW((void)adjoint.IceHistoryKernels(times, measurements),
                 std::invalid_argument);
  }
}

TEST(DiscreteAdjoint3D, ContinuousMaterialKernelsAgreeAndConverge) {
  // The 3D continuous material kernels against the exact discrete ones,
  // paired with lateral directions in kappa and mu, lateral viscosity on:
  // a few per cent at 2.5 kyr steps, falling under refinement, both
  // schemes — one problem at every resolution (a ramped cap load, the
  // measurements at the same times).
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
  const auto load = scene.MeltLoad(-0.1);
  const auto gMid = scene.Smooth(rng, 1.0);
  const auto gEnd = scene.Smooth(rng, 0.6);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto gap = [&](std::size_t npts, Scheme scheme) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * scene.Kyr());
    }
    const auto iceLoad = Ramp(load, times.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].seaLevelDual = gMid;
    measurements[npts - 1].displacementDual = gEnd;
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    const auto& coupling = op.Coupling();
    const auto dKappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    const auto dMu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    auto paired = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels = AdjointSolver3D(evolution, {.method = method,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12,
                                                       .material_kernels = true})
                               .IceHistoryKernels(times, measurements, &trajectory);
      paired[method] = FieldPairing(coupling, kernels.kappaKernel3D, dKappa) +
                       FieldPairing(coupling, kernels.muKernel3D, dMu);
    }
    return std::abs(paired[AdjointMethod::Continuous] - paired[AdjointMethod::Discrete]) /
           std::abs(paired[AdjointMethod::Discrete]);
  };
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto coarse = gap(5, scheme);
    const auto fine = gap(9, scheme);
    EXPECT_LT(coarse, 0.1) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(fine, coarse / 1.5) << "scheme " << static_cast<int>(scheme)
                                  << " " << coarse << " " << fine;
  }
}

TEST(DiscreteAdjoint3D, ZeroLateralMaterialKernelsMatchThe1D) {
  // With delta = 0 the spherical reductions of the 3D material kernels
  // equal the 1D discrete material kernels at the memory nodes, both
  // schemes: the zero-lateral bridge in the material.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(19);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[2].displacementDual = scene.Smooth(rng, 0.5);
  const auto ngll = scene.material.Mesh().NGll();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op1(scene.rheology, kLmax,
                             {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                                 {.scheme = scheme,
                                  .rotation = RotationalFeedback::None,
                                  .rtol = 1e-12,
                                  .record_stride = 1});
    const auto trajectory1 =
        ForwardTrajectory::FromSnapshots(evolution1.Run(times, iceLoad));
    const auto kernels1 =
        AdjointSolver(evolution1, {.method = AdjointMethod::Discrete,
                                   .rtol = 1e-12,
                                   .material_kernels = true})
            .IceHistoryKernels(times, measurements, &trajectory1);

    Viscoelastic3D op3(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; },
                       {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution3D evolution3(
        op3, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory3 =
        ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, iceLoad));
    const auto kernels3 =
        AdjointSolver3D(evolution3, {.method = AdjointMethod::Discrete,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .material_kernels = true})
            .IceHistoryKernels(times, measurements, &trajectory3);
    const auto& coupling = op3.Coupling();
    auto scaleK = 0.0, scaleM = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                        coupling.NodeInElementOf(rn);
      scaleK = std::max(scaleK, std::abs(kernels1.kappaKernel[full]));
      scaleM = std::max(scaleM, std::abs(kernels1.muKernel[full]));
    }
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                        coupling.NodeInElementOf(rn);
      EXPECT_NEAR(kernels3.kappaKernelSpherical[rn], kernels1.kappaKernel[full],
                  1e-7 * scaleK)
          << "node " << rn << " scheme " << static_cast<int>(scheme);
      EXPECT_NEAR(kernels3.muKernelSpherical[rn], kernels1.muKernel[full],
                  1e-7 * scaleM)
          << "node " << rn << " scheme " << static_cast<int>(scheme);
    }
  }
}


// -- lateral moduli in the background ----------------------------------------

namespace {

/// Angle-varying 10% shear and 5% bulk perturbations of the elastic moduli
/// (Viscoelastic3D::Options::moduli).
Viscoelastic3D::Options::LateralModuli TestModuli() {
  return {.shear =
              [](double r, double theta, double phi) {
                return 0.1 * (0.5 + r * r) *
                       (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                        0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
              },
          .bulk = [](double r, double theta, double) {
            return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
          }};
}

}  // namespace

TEST(DiscreteAdjoint3D, PairsWithTheTangentUnderLateralModuli) {
  // The exact transpose with lateral ELASTIC moduli in the background as
  // well as lateral viscosity: the discrete sweep's load duals and itau
  // kernel pair with the tangent-linear run's objective (the bulk term's
  // transposes, the lateral start solve and its channel duals, the branch
  // moduli's shear factor all exercised), both schemes.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(13);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = scene.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = scene.Smooth(rng, 1.0);
  middle.displacementDual = scene.Smooth(rng, -0.5);
  measurements[2] = std::move(middle);
  Measurement last;
  last.displacementDual = scene.Smooth(rng, 0.7);
  last.potentialDual = scene.Smooth(rng, 0.4);
  last.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[4] = std::move(last);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    Viscoelastic3D op(
        scene.rheology, kLmax,
        [](double, double theta, double phi) {
          return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        {.moduli = TestModuli()});
    ASSERT_TRUE(op.HasLateralModuli());
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    TangentPerturbation3D d;
    d.directLoad = deltas;
    d.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto ditau = TangentLinearSolver3D::ItauOfLogViscosity(op, d.logViscosity);
    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += scene.Surface(kernels.ice.loadDuals[n], deltas[n]);
    }
    const auto itauRhs = GridPairing(op.Coupling(), kernels.itauKernel3D, ditau);
    const auto lhs = Objective(scene, tangent.Run(times, d, &trajectory), measurements);
    EXPECT_NEAR(lhs, loadsRhs + itauRhs, 1e-8 * std::abs(lhs))
        << "scheme " << static_cast<int>(scheme);
    TangentPerturbation3D loadsOnly;
    loadsOnly.directLoad = deltas;
    EXPECT_NEAR(Objective(scene, tangent.Run(times, loadsOnly), measurements),
                loadsRhs, 1e-8 * std::abs(loadsRhs))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(DiscreteAdjoint3D, MaterialKernelsPairWithTheTangentUnderLateralModuli) {
  // The material kernels at a laterally heterogeneous elastic background:
  // the discrete kappa(x) and mu(x) kernels paired with lateral directions
  // equal the 3D material tangent's objective change, both schemes.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = scene.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = scene.Smooth(rng, 1.0);
  middle.displacementDual = scene.Smooth(rng, -0.5);
  measurements[2] = std::move(middle);
  Measurement last;
  last.displacementDual = scene.Smooth(rng, 0.7);
  last.potentialDual = scene.Smooth(rng, 0.4);
  measurements[4] = std::move(last);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    Viscoelastic3D op(
        scene.rheology, kLmax,
        [](double, double theta, double phi) {
          return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        {.memory_points = MemoryPointSet::Gll, .moduli = TestModuli()});
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    const auto& coupling = op.Coupling();
    MaterialDirection3D d;
    d.kappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    d.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12,
                                        .material_kernels = true});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    const auto kappaRhs = FieldPairing(coupling, kernels.kappaKernel3D, d.kappa);
    const auto muRhs = FieldPairing(coupling, kernels.muKernel3D, d.mu);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    for (const auto& [name, kappa, mu, rhs] :
         {std::tuple{"kappa", true, false, kappaRhs},
          std::tuple{"mu", false, true, muRhs},
          std::tuple{"both", true, true, kappaRhs + muRhs}}) {
      TangentPerturbation3D direction;
      if (kappa) direction.material.kappa = d.kappa;
      if (mu) direction.material.mu = d.mu;
      const auto lhs =
          Objective(scene, tangent.Run(times, direction, &trajectory), measurements);
      EXPECT_NEAR(lhs, rhs, 1e-8 * std::abs(lhs))
          << name << " scheme " << static_cast<int>(scheme);
      EXPECT_NE(lhs, 0.0) << name;
    }
  }
}

TEST(DiscreteAdjoint3D, ContinuousGapClosesUnderLateralModuli) {
  // The continuous sweep under lateral moduli: its gap to the discrete
  // kernels falls as the steps are halved, loads and log-viscosity alike.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(19);
  const auto load = scene.MeltLoad(-0.1);
  const auto g = scene.Smooth(rng, 1.0);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(
      scene.rheology, kLmax,
      [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      },
      {.moduli = TestModuli()});
  const auto p = OnCoupling(op.Coupling(), direction);
  auto loadGaps = std::vector<double>{}, viscosityGaps = std::vector<double>{};
  for (const auto nodes : {9, 17}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Ramp(load, span)));
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].seaLevelDual = g;
    auto results = std::map<AdjointMethod, std::pair<double, double>>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver3D(evolution, {.method = method, .rtol = 1e-12,
                                      .coupled_rtol = 1e-12})
              .IceHistoryKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.loadDuals[n], Ramp(load, span)(times[n]));
      }
      results[method] = {loads, GridPairing(op.Coupling(),
                                            kernels.logViscosityKernel3D, {p})};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    loadGaps.push_back(std::abs(d.first - c.first) / std::abs(d.first));
    viscosityGaps.push_back(std::abs(d.second - c.second) / std::abs(d.second));
  }
  for (const auto* gaps : {&loadGaps, &viscosityGaps}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_GT((*gaps)[0] / (*gaps)[1], 2.5) << (*gaps)[0] << " " << (*gaps)[1];
  }
  std::cout << "lateral-moduli gaps: loads " << loadGaps[0] << " -> " << loadGaps[1]
            << ", viscosity " << viscosityGaps[0] << " -> " << viscosityGaps[1] << '\n';
}

/// The continuous material kernels at a laterally heterogeneous elastic
/// background (the equilibria and jumps now zero-length steps) against
/// the exact discrete ones: a few per cent at 2.5 kyr steps, falling
/// under refinement, both schemes; the forward steps, the continuous
/// sweep's reversed steps and the discrete sweep's regenerations closed
/// by the given solver (the discrete transpose is a fixed point either
/// way).
void ContinuousMaterialKernelsAgreeUnderLateralModuli(
    Viscoelastic3D::Options::StepSolver solver) {
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
  const auto load = scene.MeltLoad(-0.1);
  const auto gMid = scene.Smooth(rng, 1.0);
  const auto gEnd = scene.Smooth(rng, 0.6);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto gap = [&](std::size_t npts, Scheme scheme) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * scene.Kyr());
    }
    const auto iceLoad = Ramp(load, times.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].seaLevelDual = gMid;
    measurements[npts - 1].displacementDual = gEnd;
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.solver = solver,
                       .solve_rtol = 1e-12,
                       .memory_points = MemoryPointSet::Gll,
                       .moduli = TestModuli()});
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    const auto& coupling = op.Coupling();
    const auto dKappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    const auto dMu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    auto paired = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels = AdjointSolver3D(evolution, {.method = method,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12,
                                                       .material_kernels = true})
                               .IceHistoryKernels(times, measurements, &trajectory);
      paired[method] = FieldPairing(coupling, kernels.kappaKernel3D, dKappa) +
                       FieldPairing(coupling, kernels.muKernel3D, dMu);
    }
    return std::abs(paired[AdjointMethod::Continuous] - paired[AdjointMethod::Discrete]) /
           std::abs(paired[AdjointMethod::Discrete]);
  };
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto coarse = gap(5, scheme);
    const auto fine = gap(9, scheme);
    EXPECT_LT(coarse, 0.1) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(fine, coarse / 1.5) << "scheme " << static_cast<int>(scheme)
                                  << " " << coarse << " " << fine;
    std::cout << "continuous material gap under lateral moduli, scheme "
              << static_cast<int>(scheme) << ": " << coarse << " -> " << fine << '\n';
  }
}

TEST(DiscreteAdjoint3D, ContinuousMaterialKernelsAgreeUnderLateralModuli) {
  ContinuousMaterialKernelsAgreeUnderLateralModuli(
      Viscoelastic3D::Options::StepSolver::FixedPoint);
}

TEST(DiscreteAdjoint3D, ContinuousMaterialKernelsAgreeUnderConjugateGradientSteps) {
  ContinuousMaterialKernelsAgreeUnderLateralModuli(
      Viscoelastic3D::Options::StepSolver::ConjugateGradients);
}

TEST(DiscreteAdjoint3D, DISABLED_ContinuousGapDiagnostic) {
  // Not a gate: the continuous sweep's gap to the discrete at three
  // resolutions under a chosen lateral background (SL3DPP_BACKGROUND:
  // "none" or "moduli"), printed.
  const auto* env = std::getenv("SL3DPP_BACKGROUND");
  const std::string background = env ? env : "moduli";
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(19);
  const auto load = scene.MeltLoad(-0.1);
  const auto g = scene.Smooth(rng, 1.0);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto span = 10.0 * scene.Kyr();
  auto moduli = Viscoelastic3D::Options::LateralModuli{};
  const auto full = TestModuli();
  if (background == "moduli") {
    moduli.shear = full.shear;
    moduli.bulk = full.bulk;
  }
  Viscoelastic3D op(
      scene.rheology, kLmax,
      [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      },
      {.moduli = moduli});
  const auto p = OnCoupling(op.Coupling(), direction);
  const auto rotation = std::getenv("SL3DPP_ROTATION") ? RotationalFeedback::None
                                                       : RotationalFeedback::Full;
  const bool recordStrain = std::getenv("SL3DPP_RECORD_STRAIN") != nullptr;
  for (const auto nodes : {9, 17, 33}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.rotation = rotation, .rtol = 1e-12, .coupled_rtol = 1e-12,
         .record_stride = 1, .record_strain = recordStrain});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Ramp(load, span)));
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].seaLevelDual = g;
    auto results = std::map<AdjointMethod, std::pair<double, double>>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver3D(evolution, {.method = method, .rtol = 1e-12,
                                      .coupled_rtol = 1e-12})
              .IceHistoryKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.loadDuals[n], Ramp(load, span)(times[n]));
      }
      results[method] = {loads, GridPairing(op.Coupling(),
                                            kernels.logViscosityKernel3D, {p})};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    std::cout << background << " nodes " << nodes << ": loads " << d.first << " vs "
              << c.first << " (gap " << std::abs(d.first - c.first) / std::abs(d.first)
              << "), viscosity " << d.second << " vs " << c.second << " (gap "
              << std::abs(d.second - c.second) / std::abs(d.second) << ")\n";
  }
}

