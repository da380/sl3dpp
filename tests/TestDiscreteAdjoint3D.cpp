// The discrete 3D adjoint: the pairing identity with the tangent-linear
// run at solver tolerance (the exact transpose, loads and a log-viscosity
// direction alike), the zero-lateral limit equal to the 1D discrete
// sweep, the machine-grade Taylor test of the 3D log-viscosity kernel
// (second order to solver precision at every step), and the continuous
// sweep's gap closing under refinement — docs/adjoints.md.

#include <numbers>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include <cmath>
#include <complex>
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

/// The volume pairing of per-branch kernels with per-branch directions.
double GridPairing(const GridCoupling& coupling,
                   const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& direction) {
  auto total = 0.0;
  for (auto k = std::size_t{0}; k < kernel.size(); ++k) {
    total += VolumePairing(coupling, kernel[k], direction[k]);
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
                 const std::vector<Evolution3D::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.seaLevelDual.empty()) {
      total += scene.Surface(m.seaLevelDual, snap.relativeSeaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      total += scene.Surface(m.displacementDual, snap.displacement);
    }
    if (!m.potentialDual.empty()) {
      total += scene.Surface(m.potentialDual, snap.potential);
    }
    if (!m.tangentialPotentialDual.empty()) {
      total +=
          scene.Surface(m.tangentialPotentialDual, snap.tangentialPotential);
    }
    if (!m.toroidalPotentialDual.empty() && !snap.toroidalPotential.empty()) {
      total += scene.Surface(m.toroidalPotentialDual, snap.toroidalPotential);
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
  const auto directLoad = Ramp(load, times.back());
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
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    TangentPerturbation3D d;
    d.directLoad = deltas;
    d.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto ditau = TangentLinearSolver3D::ItauOfLogViscosity(op, d.logViscosity);

    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += scene.Surface(kernels.ice.directLoadKernels[n], deltas[n]);
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
    Evolution3D sparse(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 2});
    const auto sparseTrajectory =
        ForwardTrajectory3D::FromSnapshots(sparse.Run(times, directLoad));
    const auto again =
        adjoint.FirstOrderKernels(times, measurements, &sparseTrajectory);
    EXPECT_NEAR(GridPairing(op.Coupling(), again.itauKernel3D, ditau), itauRhs,
                1e-9 * std::abs(itauRhs));
  }
}

TEST(DiscreteAdjoint3D, ATideAndTheTangentialDisplacementInThePair) {
  // Under a load and a prescribed tide, with the tangential
  // displacement's potential measured, under lateral viscosity: the
  // discrete sweep is the transpose of the discrete tangent-linear run
  // in the load, the tide and the viscosity, with an ocean and without
  // one.
  auto scene = Scene::Make(1e21);
  const auto& grid = scene.solver.Grid();
  std::mt19937 rng(29);
  using Complex = std::complex<double>;
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto potential =
      1.0 / scene.solver.SolverScales().Factor(PotentialDimensions);
  auto smoothTide = [&](double amplitude) {
    auto out = grid.Expand(scene.Smooth(rng, amplitude * potential));
    out[SurfaceGrid::Index(0, 0)] = 0.0;
    return out;
  };
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto tides = std::vector<std::vector<Complex>>{};
  auto tideDeltas = std::vector<std::vector<Complex>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(scene.Smooth(rng, 100.0 * scene.Sigma()));
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
    tides.push_back(smoothTide(1.0));
    tideDeltas.push_back(smoothTide(0.1));
  }
  auto nodeOf = [&](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * times.back()) return n;
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
  auto pair = [&](const std::vector<Complex>& a, const std::vector<Complex>& b) {
    const auto fa = grid.Synthesise(a);
    const auto fb = grid.Synthesise(b);
    auto product = std::vector<double>(fa.size());
    for (auto i = std::size_t{0}; i < fa.size(); ++i) product[i] = fa[i] * fb[i];
    return grid.Integrate(product);
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].tangentialPotentialDual = scene.Smooth(rng, 0.3);
  measurements[2].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[2].tangentialPotentialDual = scene.Smooth(rng, -0.5);
  measurements[3].displacementDual = scene.Smooth(rng, 0.7);
  measurements[3].potentialDual = scene.Smooth(rng, 0.4);
  measurements[3].tangentialPotentialDual = scene.Smooth(rng, 0.6);
  measurements[3].angularVelocityDual = {0.2, -0.4, 0.8};
  // and the toroidal potential, which the lateral structure makes
  measurements[2].toroidalPotentialDual = scene.Smooth(rng, 0.8);
  measurements[3].toroidalPotentialDual = scene.Smooth(rng, -0.5);
  auto toroidalAlone = std::map<std::size_t, Measurement>{};
  toroidalAlone[1].toroidalPotentialDual = scene.Smooth(rng, 1.0);
  toroidalAlone[3].toroidalPotentialDual = scene.Smooth(rng, 1.0);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };

  const auto dry = scene.solver.NoOceanState();
  const auto& wet = scene.state;
  for (const auto* background : {&wet, &dry}) {
    const auto label = background == &dry ? "no ocean" : "ocean";
    Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
      return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
    });
    Evolution3D evolution(
        op, scene.solver, *background,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(
        times, Evolution3D::Forcing{
                   .directLoad = [&](double t) { return loads[nodeOf(t)]; },
                   .tide = [&](double t) { return tides[nodeOf(t)]; }});
    ASSERT_FALSE(snapshots.back().toroidalPotential.empty());
    EXPECT_GT(MaxAbs(snapshots.back().toroidalPotential), 0.0);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    ASSERT_EQ(kernels.ice.tideKernels.size(), times.size());
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    auto loadRhs = 0.0, tideRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadRhs += scene.Surface(kernels.ice.directLoadKernels[n], deltas[n]);
      tideRhs += pair(kernels.ice.tideKernels[n], tideDeltas[n]);
    }
    TangentPerturbation3D loadOnly, tideOnly, viscosityOnly;
    loadOnly.directLoad = deltas;
    tideOnly.tide = tideDeltas;
    viscosityOnly.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto alongLoad = Objective(scene, tangent.Run(times, loadOnly), measurements);
    const auto alongTide = Objective(scene, tangent.Run(times, tideOnly), measurements);
    EXPECT_NE(alongTide, 0.0);
    EXPECT_NEAR(alongLoad, loadRhs, 1e-8 * std::abs(alongLoad)) << label;
    EXPECT_NEAR(alongTide, tideRhs, 1e-8 * std::abs(alongTide)) << label;
    const auto ditau =
        TangentLinearSolver3D::ItauOfLogViscosity(op, viscosityOnly.logViscosity);
    const auto alongViscosity = Objective(
        scene, tangent.Run(times, viscosityOnly, &trajectory), measurements);
    const auto viscosityRhs =
        GridPairing(op.Coupling(), kernels.itauKernel3D, ditau);
    EXPECT_NE(alongViscosity, 0.0);
    EXPECT_NEAR(alongViscosity, viscosityRhs, 1e-8 * std::abs(alongViscosity))
        << label;

    // the toroidal potential measured alone
    const auto alone =
        adjoint.FirstOrderKernels(times, toroidalAlone, &trajectory);
    auto aloneLoad = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      aloneLoad += scene.Surface(alone.ice.directLoadKernels[n], deltas[n]);
    }
    const auto toroidalAlongLoad =
        Objective(scene, tangent.Run(times, loadOnly), toroidalAlone);
    EXPECT_NE(toroidalAlongLoad, 0.0);
    EXPECT_NEAR(toroidalAlongLoad, aloneLoad, 1e-8 * std::abs(toroidalAlongLoad))
        << label;
    const auto toroidalAlongViscosity = Objective(
        scene, tangent.Run(times, viscosityOnly, &trajectory), toroidalAlone);
    EXPECT_NEAR(toroidalAlongViscosity,
                GridPairing(op.Coupling(), alone.itauKernel3D, ditau),
                1e-8 * std::abs(toroidalAlongViscosity))
        << label;
  }
}

TEST(DiscreteAdjoint3D, InteriorSourcesInThePair) {
  // Point sources as parameters under lateral viscosity: the
  // tangent-linear run along a perturbation of their moment tensors is
  // the run under the perturbation, and the discrete sweep's moment
  // kernels, the strain of its dual displacement at the sources, pair
  // with the perturbation to the same number; every field measured, the
  // toroidal potential among them, with an ocean and without one.
  auto scene = Scene::Make(1e21);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(37);
  std::normal_distribution<double> normal;
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(0.5 * n * scene.Kyr());
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto points = std::vector<BodyPoint>{
      {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0},
      {.radius = surface - 800e3 / sle.Length(), .colatitude = 2.2, .longitude = 1.0}};
  auto perturbation = std::vector<std::vector<PointSource>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    auto at = std::vector<PointSource>{};
    for (const auto& point : points) {
      at.push_back({.at = point,
                    .moment = {.rr = unit * normal(rng),
                               .tt = unit * normal(rng),
                               .pp = unit * normal(rng),
                               .rt = unit * normal(rng),
                               .rp = unit * normal(rng),
                               .tp = unit * normal(rng)}});
    }
    perturbation.push_back(std::move(at));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].displacementDual = scene.Smooth(rng, 0.3);
  measurements[0].toroidalPotentialDual = scene.Smooth(rng, 0.4);
  measurements[2].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[2].tangentialPotentialDual = scene.Smooth(rng, -0.5);
  measurements[3].displacementDual = scene.Smooth(rng, 0.7);
  measurements[3].potentialDual = scene.Smooth(rng, 0.4);
  measurements[3].toroidalPotentialDual = scene.Smooth(rng, 0.6);
  measurements[3].angularVelocityDual = {0.2, -0.4, 0.8};
  auto nodeOf = [&](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * times.back()) return n;
    }
    throw std::runtime_error("the test history is read off its nodes");
  };

  const auto dry = scene.solver.NoOceanState();
  const auto& wet = scene.state;
  for (const auto* background : {&wet, &dry}) {
    const auto label = background == &dry ? "no ocean" : "ocean";
    Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
      return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
    });
    Evolution3D evolution(op, scene.solver, *background,
                          {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto forward = Objective(
        scene,
        evolution.Run(times, Evolution3D::Forcing{.sources = [&](double t) {
          return perturbation[nodeOf(t)];
        }}),
        measurements);
    ASSERT_NE(forward, 0.0);
    TangentPerturbation3D direction;
    direction.sources = perturbation;
    const auto along = Objective(
        scene,
        TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12})
            .Run(times, direction),
        measurements);
    EXPECT_NEAR(along, forward, 1e-8 * std::abs(forward)) << label;

    const auto kernels =
        AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                    .rtol = 1e-12,
                                    .coupled_rtol = 1e-12,
                                    .moment_points = points})
            .FirstOrderKernels(times, measurements);
    ASSERT_EQ(kernels.ice.momentKernels.size(), times.size());
    auto paired = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      for (auto p = std::size_t{0}; p < points.size(); ++p) {
        paired +=
            kernels.ice.momentKernels[n][p].Contract(perturbation[n][p].moment);
      }
    }
    EXPECT_NEAR(paired, along, 1e-8 * std::abs(along)) << label << " " << paired
                                                        << " " << along;
  }
}

TEST(DiscreteAdjoint3D, ViscosityAboutASourceDrivenRun) {
  // The viscosity as a parameter of a run under point sources with
  // lateral viscosity: the discrete tangent-linear run along a lateral
  // log-viscosity direction is the derivative of the forward run
  // (central differences at second order, Richardson to 1e-6), and the
  // discrete sweep's kernel pairs with the direction to the same number;
  // the toroidal potential measured among the fields, and alone.
  auto scene = Scene::Make(1e21);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(43);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(0.5 * n * scene.Kyr());
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto sources = std::vector<PointSource>{
      {.at = {.radius = surface - 400e3 / sle.Length(),
              .colatitude = 1.1,
              .longitude = 4.0},
       .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                  .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}},
      {.at = {.radius = surface - 800e3 / sle.Length(),
              .colatitude = 2.2,
              .longitude = 1.0},
       .moment = {.rr = -0.4 * unit, .tt = 0.8 * unit, .pp = -0.4 * unit,
                  .rt = 0.5 * unit, .rp = 0.1 * unit, .tp = -0.3 * unit}}};
  const auto forcing =
      Evolution3D::Forcing{.sources = [&](double) { return sources; }};
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[2].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[2].tangentialPotentialDual = scene.Smooth(rng, -0.5);
  measurements[3].displacementDual = scene.Smooth(rng, 0.7);
  measurements[3].toroidalPotentialDual = scene.Smooth(rng, 0.6);
  auto toroidalAlone = std::map<std::size_t, Measurement>{};
  toroidalAlone[1].toroidalPotentialDual = scene.Smooth(rng, 1.0);
  toroidalAlone[3].toroidalPotentialDual = scene.Smooth(rng, 1.0);
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto& all = measurements;
  const auto& alone = toroidalAlone;
  for (const auto* measured : {&all, &alone}) {
    const auto label = measured == &alone ? "the toroidal dual alone" : "all";
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta);
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, forcing));
    ASSERT_EQ(trajectory.sourceForces.size(), times.size());
    TangentPerturbation3D d;
    d.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto along = Objective(
        scene,
        TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12})
            .Run(times, d, &trajectory),
        *measured);
    ASSERT_NE(along, 0.0) << label;

    auto objectiveAt = [&](double h) {
      Viscoelastic3D moved(scene.rheology, kLmax,
                           [&](double r, double theta, double phi) {
                             return baseDelta(r, theta, phi) +
                                    h * direction(theta, phi);
                           });
      Evolution3D movedEvolution(moved, scene.solver, scene.state,
                                 {.rtol = 1e-12, .coupled_rtol = 1e-12});
      return Objective(scene, movedEvolution.Run(times, forcing), *measured);
    };
    auto fd = std::map<double, double>{};
    for (double h : {0.04, 0.02}) {
      fd[h] = (objectiveAt(h) - objectiveAt(-h)) / (2.0 * h);
    }
    const auto r1 = std::abs(fd[0.04] - along);
    const auto r2 = std::abs(fd[0.02] - along);
    EXPECT_GT(r1 / r2, 3.0) << label;
    EXPECT_LT(r1 / r2, 5.0) << label;
    EXPECT_NEAR((4.0 * fd[0.02] - fd[0.04]) / 3.0, along, 1e-6 * std::abs(along))
        << label;

    const auto kernels =
        AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                    .rtol = 1e-12,
                                    .coupled_rtol = 1e-12})
            .FirstOrderKernels(times, *measured, &trajectory);
    const auto ditau = TangentLinearSolver3D::ItauOfLogViscosity(op, d.logViscosity);
    const auto paired = GridPairing(op.Coupling(), kernels.itauKernel3D, ditau);
    EXPECT_NEAR(paired, along, 1e-8 * std::abs(along)) << label;

    // what is not formed about a source-driven trajectory says so
    EXPECT_THROW((void)AdjointSolver3D(evolution,
                                       {.method = AdjointMethod::Discrete,
                                        .material_kernels = true})
                     .FirstOrderKernels(times, *measured, &trajectory),
                 std::invalid_argument);
  }
}

TEST(DiscreteAdjoint3D, TheContinuousSweepAboutASourceDrivenRun) {
  // The continuous 3D sweep about a run under a point source with
  // lateral viscosity: its viscosity kernel tends to the discrete
  // sweep's as the steps are halved, the displacement and the toroidal
  // potential measured.
  auto scene = Scene::Make(1e22);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(47);
  const auto span = 10.0 * scene.Kyr();
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto source = PointSource{
      .at = {.radius = surface - 400e3 / sle.Length(),
             .colatitude = 1.1,
             .longitude = 4.0},
      .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                 .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}};
  const auto gU = scene.Smooth(rng, 0.5);
  const auto gW = scene.Smooth(rng, 0.8);
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto p = OnCoupling(op.Coupling(), direction);
  auto gaps = std::vector<double>{};
  for (const auto nodes : {5, 9, 17}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    const auto forcing = Evolution3D::Forcing{.sources = [&](double t) {
      auto at = source;
      const auto shape = 0.3 + t / span;
      for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp,
                      &at.moment.rt, &at.moment.rp, &at.moment.tp}) {
        *c *= shape;
      }
      return std::vector<PointSource>{at};
    }};
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, forcing));
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].displacementDual = gU;
    measurements[times.size() - 1].toroidalPotentialDual = gW;
    auto paired = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver3D(evolution, {.method = method,
                                      .rtol = 1e-12,
                                      .coupled_rtol = 1e-12})
              .FirstOrderKernels(times, measurements, &trajectory);
      paired[method] = GridPairing(op.Coupling(), kernels.logViscosityKernel3D, {p});
    }
    ASSERT_NE(paired[AdjointMethod::Discrete], 0.0);
    gaps.push_back(std::abs(paired[AdjointMethod::Continuous] -
                            paired[AdjointMethod::Discrete]) /
                   std::abs(paired[AdjointMethod::Discrete]));
  }
  EXPECT_LT(gaps[2], 0.1) << gaps[0] << " " << gaps[1] << " " << gaps[2];
  EXPECT_GT(gaps[0] / gaps[1], 2.0) << gaps[0] << " " << gaps[1];
  EXPECT_GT(gaps[1] / gaps[2], 2.0) << gaps[1] << " " << gaps[2];
}

TEST(DiscreteAdjoint3D, ZeroLateralMatchesThe1DDiscreteSweep) {
  // With delta = 0 at the GLL points the discrete 3D sweep is the 1D
  // discrete sweep through the grid pipeline: the load duals and the
  // spherically reduced itau kernel at the pipeline's round-off.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto directLoad = Ramp(scene.MeltLoad(-0.1), times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[3].displacementDual = scene.Smooth(rng, -0.6);
  measurements[5].angularVelocityDual = {0.1, 0.3, -0.2};

  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Evolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
  const auto kernels1 =
      AdjointSolver(evolution1, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(times, measurements, &trajectory1);

  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  Evolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
  const auto kernels3 =
      AdjointSolver3D(evolution3, {.method = AdjointMethod::Discrete,
                                   .rtol = 1e-12,
                                   .coupled_rtol = 1e-12})
          .FirstOrderKernels(times, measurements, &trajectory3);

  auto scale = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    scale = std::max(scale, MaxAbs(kernels1.directLoadKernels[n]));
    worst = std::max(worst, MaxDiff(kernels3.ice.directLoadKernels[n], kernels1.directLoadKernels[n]));
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
  const auto directLoad = Ramp(scene.MeltLoad(-0.1), times.back());
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
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12,
         .record_stride = kernelsOut != nullptr ? 1 : 0});
    const auto snapshots = evolution.Run(times, directLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      *kernelsOut = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                               .rtol = 1e-12,
                                               .coupled_rtol = 1e-12})
                        .FirstOrderKernels(times, measurements, &trajectory);
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
    Evolution3D evolution(
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
              .FirstOrderKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.directLoadKernels[n], Ramp(load, span)(times[n]));
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
  // each modulus alone and together, with the memory nodes at the GLL
  // points and at the Gauss points, and at the Gauss points with the
  // rotational feedback on, axial and full. Refused without a trajectory.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
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

  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss})
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    // rotational feedback at the default points, one scheme each
    if (rotation != RotationalFeedback::None &&
        (set != MemoryPointSet::Gauss ||
         (rotation == RotationalFeedback::Axial) !=
             (scheme == Scheme::BackwardEuler))) {
      continue;
    }
    Viscoelastic3D op(scene.rheology, kLmax,
                      [](double, double theta, double phi) {
                        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
                      },
                      {.memory_points = set});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = rotation,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
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
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    ASSERT_EQ(kernels.kappaKernel3D.size(), coupling.GridValueCount());
    ASSERT_EQ(kernels.muKernel3D.size(), coupling.GridValueCount());
    const auto kappaRhs = VolumePairing(coupling, kernels.kappaKernel3D, d.kappa);
    const auto muRhs = VolumePairing(coupling, kernels.muKernel3D, d.mu);

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
          << name << " scheme " << static_cast<int>(scheme) << ", points "
          << static_cast<int>(set) << ", rotation " << static_cast<int>(rotation);
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
    const auto spherical =
        VolumePairingOfMeans(coupling, kernels.muKernelSpherical, nodal);
    const auto full = VolumePairing(coupling, kernels.muKernel3D, uniform);
    EXPECT_NEAR(spherical, full, 1e-12 * std::abs(full));

    // The refusal without a trajectory.
    EXPECT_THROW((void)adjoint.FirstOrderKernels(times, measurements),
                 std::invalid_argument);
  }
}

TEST(DiscreteAdjoint3D, ContinuousMaterialKernelsAgreeAndConverge) {
  // The 3D continuous material kernels against the exact discrete ones,
  // paired with lateral directions in kappa and mu, lateral viscosity on:
  // a few per cent at 2.5 kyr steps, falling under refinement, both
  // schemes — one problem at every resolution (a ramped cap load, the
  // measurements at the same times). At the default memory nodes, the
  // Gauss points.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
  const auto load = scene.MeltLoad(-0.1);
  const auto gMid = scene.Smooth(rng, 1.0);
  const auto gEnd = scene.Smooth(rng, 0.6);
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto gap = [&](std::size_t npts, Scheme scheme,
                 RotationalFeedback rotation = RotationalFeedback::None) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * scene.Kyr());
    }
    const auto directLoad = Ramp(load, times.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].seaLevelDual = gMid;
    measurements[npts - 1].displacementDual = gEnd;
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.memory_points = MemoryPointSet::Gauss});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = rotation,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
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
                               .FirstOrderKernels(times, measurements, &trajectory);
      paired[method] = VolumePairing(coupling, kernels.kappaKernel3D, dKappa) +
                       VolumePairing(coupling, kernels.muKernel3D, dMu);
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
  // the same with the full rotational feedback on
  const auto coarse = gap(5, Scheme::ExponentialTrapezoid, RotationalFeedback::Full);
  const auto fine = gap(9, Scheme::ExponentialTrapezoid, RotationalFeedback::Full);
  EXPECT_LT(coarse, 0.1);
  EXPECT_LT(fine, coarse / 1.5) << coarse << " " << fine;
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
  const auto directLoad = Ramp(load, times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[2].displacementDual = scene.Smooth(rng, 0.5);
  const auto ngll = scene.material.Mesh().NGll();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op1(scene.rheology, kLmax,
                             {.memory_points = MemoryPointSet::Gll});
    Evolution evolution1(op1, scene.solver, scene.state,
                                 {.scheme = scheme,
                                  .rotation = RotationalFeedback::None,
                                  .rtol = 1e-12,
                                  .record_stride = 1});
    const auto trajectory1 =
        ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
    const auto kernels1 =
        AdjointSolver(evolution1, {.method = AdjointMethod::Discrete,
                                   .rtol = 1e-12,
                                   .material_kernels = true})
            .FirstOrderKernels(times, measurements, &trajectory1);

    Viscoelastic3D op3(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; },
                       {.memory_points = MemoryPointSet::Gll});
    Evolution3D evolution3(
        op3, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory3 =
        ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
    const auto kernels3 =
        AdjointSolver3D(evolution3, {.method = AdjointMethod::Discrete,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .material_kernels = true})
            .FirstOrderKernels(times, measurements, &trajectory3);
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

void PairsWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver solver) {
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
  const auto directLoad = Ramp(load, times.back());
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
        {.solver = solver, .solve_rtol = 1e-12, .moduli = TestModuli()});
    ASSERT_TRUE(op.HasLateralModuli());
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12});
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    TangentPerturbation3D d;
    d.directLoad = deltas;
    d.logViscosity = OnCoupling(op.Coupling(), direction);
    const auto ditau = TangentLinearSolver3D::ItauOfLogViscosity(op, d.logViscosity);
    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += scene.Surface(kernels.ice.directLoadKernels[n], deltas[n]);
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

/// The lateral-moduli material gates' rotational feedback: full under
/// the backward Euler scheme, none under the trapezoid, so that the
/// material derivative about a laterally heterogeneous elastic
/// background is gated both ways.
RotationalFeedback RotationOf(Scheme scheme) {
  return scheme == Scheme::BackwardEuler ? RotationalFeedback::Full
                                         : RotationalFeedback::None;
}

TEST(DiscreteAdjoint3D, PairsWithTheTangentUnderLateralModuli) {
  PairsWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver::FixedPoint);
}

// The same with the steps closed by conjugate gradients: the forward's
// and the tangent's nested, the discrete transpose's the sea level
// adjoint nested over the same solve.
TEST(DiscreteAdjoint3D, PairsWithTheTangentUnderConjugateGradientSteps) {
  PairsWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver::ConjugateGradients);
}

void MaterialKernelsPairWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver solver) {
  // The material kernels at a laterally heterogeneous elastic background:
  // the discrete kappa(x) and mu(x) kernels paired with lateral directions
  // equal the 3D material tangent's objective change, both schemes.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
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
        {.solver = solver,
         .solve_rtol = 1e-12,
         .memory_points = MemoryPointSet::Gll,
         .moduli = TestModuli()});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationOf(scheme),
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
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
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    const auto kappaRhs = VolumePairing(coupling, kernels.kappaKernel3D, d.kappa);
    const auto muRhs = VolumePairing(coupling, kernels.muKernel3D, d.mu);
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

TEST(DiscreteAdjoint3D, MaterialKernelsPairWithTheTangentUnderLateralModuli) {
  MaterialKernelsPairWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver::FixedPoint);
}

// The same with the steps closed by conjugate gradients: the forward's
// and the tangent's nested, the discrete transpose's the sea level
// adjoint nested over the same solve.
TEST(DiscreteAdjoint3D, MaterialKernelsPairWithTheTangentUnderConjugateGradientSteps) {
  MaterialKernelsPairWithTheTangentUnderLateralModuli(Viscoelastic3D::Options::StepSolver::ConjugateGradients);
}

TEST(DiscreteAdjoint3D, TheContinuousSweepTakesATangentialMeasurement) {
  // A measurement of the tangential displacement's potential in the 3D
  // continuous sweep, an adjoint tangential traction:
  // with no lateral structure the kernels are the 1D continuous sweep's,
  // and under lateral viscosity and lateral moduli (where the jump is
  // the instantaneous fixed point under the traction's forces) the gap
  // to the discrete sweep's load kernels falls as the steps are halved.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(31);
  const auto span = 10.0 * scene.Kyr();
  const auto load = scene.MeltLoad(-0.1);
  const auto gMiddle = scene.Smooth(rng, 1.0);
  const auto gLast = scene.Smooth(rng, 0.5);
  auto gridOf = [&](int nodes) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    return times;
  };
  auto measured = [&](std::size_t nodes) {
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[nodes / 2].tangentialPotentialDual = gMiddle;
    measurements[nodes - 1].tangentialPotentialDual = gLast;
    return measurements;
  };

  {
    const auto times = gridOf(5);
    const auto measurements = measured(times.size());
    ViscoelasticOperator op1(scene.rheology, kLmax);
    Evolution evolution1(op1, scene.solver, scene.state, {.rtol = 1e-12});
    const auto kernels1 =
        AdjointSolver(evolution1, {.method = AdjointMethod::Continuous, .rtol = 1e-12})
            .FirstOrderKernels(times, measurements);
    Viscoelastic3D op3(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; });
    Evolution3D evolution3(op3, scene.solver, scene.state,
                           {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto kernels3 =
        AdjointSolver3D(evolution3, {.method = AdjointMethod::Continuous,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12})
            .FirstOrderKernels(times, measurements);
    auto scale = 0.0, worst = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      scale = std::max(scale, MaxAbs(kernels1.directLoadKernels[n]));
      worst = std::max(worst, MaxDiff(kernels3.ice.directLoadKernels[n],
                                      kernels1.directLoadKernels[n]));
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-8 * scale);
  }

  // The gap to the discrete sweep under lateral structure: the potential
  // V under lateral viscosity and moduli, and the toroidal potential W,
  // whose adjoint traction is toroidal, under lateral viscosity alone
  // (the jump by the unrelaxed toroidal response) and with lateral
  // moduli too (the jump by the instantaneous fixed point).
  struct Case {
    const char* name;
    bool toroidal;
    bool moduli;
  };
  for (const auto& which : {Case{"V, lateral moduli", false, true},
                            Case{"W, lateral viscosity", true, false},
                            Case{"W, lateral moduli", true, true}}) {
    auto options = Viscoelastic3D::Options{};
    if (which.moduli) options.moduli = TestModuli();
    Viscoelastic3D op(
        scene.rheology, kLmax,
        [](double, double theta, double phi) {
          return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        options);
    auto gaps = std::vector<double>{};
    for (const auto nodes : {5, 9, 17}) {
      const auto times = gridOf(nodes);
      auto measurements = measured(times.size());
      if (which.toroidal) {
        for (auto& [node, m] : measurements) {
          m.toroidalPotentialDual = std::move(m.tangentialPotentialDual);
          m.tangentialPotentialDual.clear();
        }
      }
      Evolution3D evolution(op, scene.solver, scene.state,
                            {.rtol = 1e-12, .coupled_rtol = 1e-12});
      auto paired = std::map<AdjointMethod, double>{};
      for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
        const auto kernels =
            AdjointSolver3D(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12})
                .FirstOrderKernels(times, measurements);
        auto total = 0.0;
        for (auto n = std::size_t{0}; n < times.size(); ++n) {
          auto at = load;
          for (auto& v : at) v *= times[n] / span;
          total += scene.Surface(kernels.ice.directLoadKernels[n], at);
        }
        paired[method] = total;
      }
      ASSERT_NE(paired[AdjointMethod::Discrete], 0.0) << which.name;
      gaps.push_back(std::abs(paired[AdjointMethod::Continuous] -
                              paired[AdjointMethod::Discrete]) /
                     std::abs(paired[AdjointMethod::Discrete]));
    }
    EXPECT_LT(gaps[2], 0.1)
        << which.name << ": " << gaps[0] << " " << gaps[1] << " " << gaps[2];
    EXPECT_GT(gaps[0] / gaps[1], 2.0)
        << which.name << ": " << gaps[0] << " " << gaps[1];
    EXPECT_GT(gaps[1] / gaps[2], 3.0)
        << which.name << ": " << gaps[1] << " " << gaps[2];
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
    Evolution3D evolution(
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
              .FirstOrderKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.directLoadKernels[n], Ramp(load, span)(times[n]));
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
/// way). At the default memory nodes, the Gauss points.
void ContinuousMaterialKernelsAgreeUnderLateralModuli(
    Viscoelastic3D::Options::StepSolver solver,
    std::initializer_list<Scheme> schemes) {
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
  const auto load = scene.MeltLoad(-0.1);
  const auto gMid = scene.Smooth(rng, 1.0);
  const auto gEnd = scene.Smooth(rng, 0.6);
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto gap = [&](std::size_t npts, Scheme scheme) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * scene.Kyr());
    }
    const auto directLoad = Ramp(load, times.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].seaLevelDual = gMid;
    measurements[npts - 1].displacementDual = gEnd;
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.solver = solver,
                       .solve_rtol = 1e-12,
                       .memory_points = MemoryPointSet::Gauss,
                       .moduli = TestModuli()});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
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
                               .FirstOrderKernels(times, measurements, &trajectory);
      paired[method] = VolumePairing(coupling, kernels.kappaKernel3D, dKappa) +
                       VolumePairing(coupling, kernels.muKernel3D, dMu);
    }
    return std::abs(paired[AdjointMethod::Continuous] - paired[AdjointMethod::Discrete]) /
           std::abs(paired[AdjointMethod::Discrete]);
  };
  for (auto scheme : schemes) {
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
      Viscoelastic3D::Options::StepSolver::FixedPoint,
      {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler});
}

TEST(DiscreteAdjoint3D, ContinuousMaterialKernelsAgreeUnderConjugateGradientSteps) {
  // one scheme: the solver changes how a step is closed, not the scheme
  ContinuousMaterialKernelsAgreeUnderLateralModuli(
      Viscoelastic3D::Options::StepSolver::ConjugateGradients,
      {Scheme::ExponentialTrapezoid});
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
    Evolution3D evolution(
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
              .FirstOrderKernels(times, measurements, &trajectory);
      auto loads = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        loads += scene.Surface(kernels.ice.directLoadKernels[n], Ramp(load, span)(times[n]));
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


TEST(DiscreteAdjoint3D, RotatingStepsSolveTheLateralSystem) {
  // The rotational feedback about a laterally heterogeneous model. The
  // run closes the rotation in closed form with the REFERENCE model's
  // Love numbers inside each pass of the step's fixed point, the lateral
  // part of the response to the centrifugal potential arriving through
  // the iteration. Converged, that must be the centrifugal potential
  // applied to the 3D model: a step re-solved by conjugate gradients
  // under the recorded load and centrifugal coefficients (no sea level
  // engine, no Love numbers) reproduces the run's surface displacement
  // and potential at every degree, and its own degree-2, order-1
  // potential gives the recorded polar motion (MacCullagh), its (2, 0)
  // potential and the inertia moment of its degree-0 part the recorded
  // axial rate. The centrifugal potential alone drives the other
  // degrees too.
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto directLoad = Ramp(scene.MeltLoad(-0.1), times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  using Solver = Viscoelastic3D::Options::StepSolver;
  Viscoelastic3D op(scene.rheology, kLmax, baseDelta, {.moduli = TestModuli()});
  Viscoelastic3D cg(scene.rheology, kLmax, baseDelta,
                    {.solver = Solver::ConjugateGradients,
                     .solve_rtol = 1e-13,
                     .moduli = TestModuli()});
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.rotation = RotationalFeedback::Full, .rtol = 1e-13, .coupled_rtol = 1e-13,
       .record_stride = 1, .record_strain = true});
  const auto run = evolution.Run(times, directLoad);
  const auto& material = scene.rheology.BaseMaterial().ModelScales();
  const auto& sle = scene.solver.SolverScales();
  auto ratio = [&](Dimensions dims) { return material.Factor(dims) / sle.Factor(dims); };
  const auto toNdLength = ratio(Dimensions{.length = 1});
  const auto toNdPotential = ratio(PotentialDimensions);
  const auto toMatDensity = 1.0 / ratio(Dimensions{.mass = 1, .length = -2});
  const auto toMatPotential = 1.0 / toNdPotential;
  using Complex = std::complex<double>;
  auto columns = [](const std::vector<Complex>& c, double factor) {
    return [c, factor](int l, double) -> std::optional<std::vector<Complex>> {
      auto out = std::vector<Complex>(l + 1);
      for (auto m = 0; m <= l; ++m) out[m] = factor * c[SurfaceGrid::Index(l, m)];
      return out;
    };
  };
  const auto& grid = scene.solver.Grid();
  for (auto n : {std::size_t{1}, std::size_t{4}}) {
    const auto& prev = *run[n - 1].internalState;
    const auto& here = run[n];
    const auto dt = here.internalState->t - prev.t;
    const auto solved =
        cg.Step(prev, dt, Scheme::ExponentialTrapezoid,
                columns(here.loadCoefficients, toMatDensity),
                columns(here.tideCoefficients, toMatPotential));
    const auto u = cg.Surface(solved.solutions, Component::U);
    const auto phi = cg.Surface(solved.solutions, Component::Phi);
    const auto uRun = grid.Expand(here.displacement);
    const auto phiRun = grid.Expand(here.potential);
    auto worstU = 0.0, worstPhi = 0.0, sizeU = 0.0, sizePhi = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        worstU = std::max(worstU, std::abs(toNdLength * u.at(l)[m] - uRun[i]));
        worstPhi = std::max(worstPhi, std::abs(toNdPotential * phi.at(l)[m] - phiRun[i]));
        sizeU = std::max(sizeU, std::abs(uRun[i]));
        sizePhi = std::max(sizePhi, std::abs(phiRun[i]));
      }
    }
    const auto c = 1.0 - scene.solver.TransverseMarginFor(1.0);
    const auto i21 = SurfaceGrid::Index(2, 1);
    EXPECT_LT(worstU, 1e-9 * sizeU) << "node " << n;
    EXPECT_LT(worstPhi, 1e-9 * sizePhi) << "node " << n;
    // the polar motion from the 3D solution's own potential: c is the
    // product of the engine's inertia and rotation factors
    const auto psi21 = here.tideCoefficients[i21];
    EXPECT_GT(std::abs(psi21), 0.0);
    EXPECT_LT(std::abs(psi21 - c * toNdPotential * phi.at(2)[1]),
              1e-9 * std::abs(psi21))
        << "node " << n;
    // the axial rate (the length of day) from the 3D solution's own
    // (2, 0) potential, the inertia moment of its degree-0 part and the
    // load's shell, with the factors formed here from the parameters
    {
      const auto& p = scene.solver.Parameters();
      const auto kPi = std::numbers::pi;
      const auto a = scene.solver.Radius();
      const auto G = p.gravitational_constant / sle.Factor(GravitationalConstantDimensions);
      const auto polar =
          p.polar_moment_of_inertia / sle.Factor(Dimensions{.mass = 1, .length = 2});
      const auto rate = p.rotation_frequency / sle.Factor(Dimensions{.time = -1});
      const auto axialInertia = std::sqrt(5.0 / (9.0 * kPi)) * rate * a * a * a / (G * polar);
      const auto trace = (4.0 / 3.0) * rate / polar;
      const auto shell = (2.0 / 3.0) * rate * std::pow(a, 4) * std::sqrt(4.0 * kPi) / polar;
      const auto moment = cg.InertiaMoment(solved.solutions) *
                          ratio(Dimensions{.mass = 1, .length = 2});
      const auto load0 = here.loadCoefficients[SurfaceGrid::Index(0, 0)].real();
      const auto phi20 = toNdPotential * phi.at(2)[0].real();
      const auto parts = std::array{-axialInertia * phi20, -trace * moment, -shell * load0};
      const auto axial = parts[0] + parts[1] + parts[2];
      // (the moment's part is about 1% of the rate here; the shell's
      // vanishes with the load's mean)
      EXPECT_GT(std::abs(parts[1]), 1e-3 * std::abs(axial));
      EXPECT_NEAR(axial, here.angularVelocity[2], 1e-9 * std::abs(axial)) << "node " << n;
    }
    // the centrifugal potential alone on the 3D model, from rest
    const auto rest = cg.ZeroState();
    const auto tideOnly = cg.Step(rest, dt, Scheme::ExponentialTrapezoid, {},
                                  columns(here.tideCoefficients, toMatPotential));
    const auto ut = cg.Surface(tideOnly.solutions, Component::U);
    auto size = std::vector<double>(kLmax + 1, 0.0);
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) size[l] += std::norm(ut.at(l)[m]);
      size[l] = std::sqrt(size[l]);
    }
    // (degree 4 at about 1% of degree 2 under these 10% moduli)
    EXPECT_GT(size[4], 1e-3 * size[2]);
    EXPECT_GT(size[0], 1e-4 * size[2]);
  }
}
