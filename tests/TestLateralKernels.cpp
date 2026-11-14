// The lateral kernels about a radial background (docs/adjoints.md §21):
// the radial sweeps with the grid expansion against the 3D sweeps on the
// same model with zero lateral structure, configuration by configuration
// and both methods; against central differences of the 3D forward run along
// lateral perturbations; their means over the sphere against the radial
// kernels; and the static problem's kernels against the transpose of
// ElasticPerturbation3D's lateral force on the static adjoint and forward
// solutions.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LateralKernels.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 8;

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> iceNd;

  static Scene Make(double viscositySI, int lmax = kLmax) {
    auto deck =
        Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
                   DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator reference(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
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
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) * fraction * iceNd[i];
    }
    return load;
  }
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng, double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= grid.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }
};

std::function<std::vector<double>(double)> Ramp(const std::vector<double>& load,
                                                double span) {
  return [load, span](double t) {
    auto out = load;
    for (auto& v : out) v *= t / span;
    return out;
  };
}

/// The largest departure of a field from a reference, relative to the
/// reference's largest magnitude.
double Departure(std::span<const double> got, std::span<const double> want) {
  EXPECT_EQ(got.size(), want.size());
  auto top = 0.0, worst = 0.0;
  for (auto i = std::size_t{0}; i < std::min(got.size(), want.size()); ++i) {
    top = std::max(top, std::abs(want[i]));
    worst = std::max(worst, std::abs(got[i] - want[i]));
  }
  return top > 0.0 ? worst / top : worst;
}

/// A lateral modulus perturbation that changes nothing a solve can see
/// (one part in 1e100) but has the 3D operator carry memory nodes over the
/// whole solid, where the radial route's moduli kernels live. The forces
/// it makes stay normal numbers: at one part in 1e300 those of a toroidal
/// traction alone are subnormal, and the sea level solve they force
/// cannot meet a relative tolerance on them.
Viscoelastic3D::Options::LateralModuli EverySolidElement() {
  return {.shear = [](double, double, double) { return 1e-100; }, .bulk = {}};
}

struct Gaps {
  double itau = 0.0, logViscosity = 0.0, kappa = 0.0, mu = 0.0, ice = 0.0;
};

/// The 1D route against the 3D sweep's kernels, field by field, over the
/// 3D coupling's nodes (the radial coupling may hold more: matched by
/// element and point, each node's grid the same).
Gaps Compare(const Kernels3D& radial, const Kernels3D& lateral,
             const GridCoupling& a, const GridCoupling& b) {
  auto nodeOf = std::map<std::pair<int, int>, int>{};
  for (auto rn = 0; rn < a.NodeCount(); ++rn) {
    nodeOf[{a.ElementOfNode(rn), a.NodeInElementOf(rn)}] = rn;
  }
  // the radial route's values at the 3D nodes, in the 3D layout
  auto restrict = [&](const std::vector<double>& field) {
    auto out = std::vector<double>(b.GridValueCount(), 0.0);
    for (auto rn = 0; rn < b.NodeCount(); ++rn) {
      const auto found = nodeOf.find({b.ElementOfNode(rn), b.NodeInElementOf(rn)});
      EXPECT_NE(found, nodeOf.end());
      if (found == nodeOf.end()) continue;
      EXPECT_EQ(a.FieldSizeOf(found->second), b.FieldSizeOf(rn));
      for (auto x = std::size_t{0}; x < b.FieldSizeOf(rn); ++x) {
        out[b.FieldOffset(rn) + x] = field[a.FieldOffset(found->second) + x];
      }
    }
    return out;
  };
  Gaps gaps;
  for (auto k = std::size_t{0}; k < lateral.itauKernel3D.size(); ++k) {
    gaps.itau = std::max(gaps.itau, Departure(restrict(radial.itauKernel3D.at(k)),
                                              lateral.itauKernel3D[k]));
    gaps.logViscosity = std::max(
        gaps.logViscosity, Departure(restrict(radial.logViscosityKernel3D.at(k)),
                                     lateral.logViscosityKernel3D[k]));
  }
  if (!lateral.kappaKernel3D.empty()) {
    gaps.kappa = Departure(restrict(radial.kappaKernel3D), lateral.kappaKernel3D);
    gaps.mu = Departure(restrict(radial.muKernel3D), lateral.muKernel3D);
  }
  for (auto n = std::size_t{0}; n < lateral.ice.directLoadKernels.size(); ++n) {
    gaps.ice = std::max(gaps.ice, Departure(radial.ice.directLoadKernels[n],
                                            lateral.ice.directLoadKernels[n]));
  }
  return gaps;
}

std::string NameOf(AdjointMethod method) {
  return method == AdjointMethod::Discrete ? "discrete" : "continuous";
}

/// One configuration of the zero-lateral gate: the forcing (any absent),
/// the measurements, the background and the run's options.
struct Case {
  std::string label;
  std::vector<double> times;
  std::map<std::size_t, Measurement> measurements;
  GRDSolver::State state;
  RotationalFeedback rotation = RotationalFeedback::None;
  Scheme scheme = Scheme::ExponentialTrapezoid;
  MemoryPointSet points = MemoryPointSet::Gauss;
  std::function<std::vector<double>(double)> directLoad, outside, layer;
  std::function<std::vector<std::complex<double>>(double)> tide;
  std::function<std::vector<PointSource>(double)> sources;
  /// A migrating run's ice thickness history (empty: frozen shorelines).
  std::function<std::vector<double>(double)> iceThickness;
  int stride = 1;
  /// The kernels asked for; a branchless rheology has no viscosity's.
  bool viscosity = true;
  const Rheology* rheology = nullptr;
};

template <class Forcing>
Forcing ForcingOf(const Case& c) {
  Forcing f;
  f.directLoad = c.directLoad;
  f.directLoadOutsideWaterBudget = c.outside;
  f.solidSurfaceChange = c.layer;
  f.tide = c.tide;
  f.sources = c.sources;
  return f;
}

/// The radial route on the radial run and the 3D sweep on the 3D run of
/// the same model with zero lateral structure (and memory nodes over the
/// whole solid), by one method; the gaps field by field.
Gaps RunGate(const Scene& scene, const Case& c, AdjointMethod method,
             bool wholeSolid = true) {
  const auto& rheology = c.rheology != nullptr ? *c.rheology : scene.rheology;
  ViscoelasticOperator op1(rheology, kLmax, {.memory_points = c.points});
  Evolution evolution1(op1, scene.solver, c.state,
                       {.scheme = c.scheme,
                        .rotation = c.rotation,
                        .rtol = 1e-12,
                        .record_stride = c.stride});
  const auto migrating = static_cast<bool>(c.iceThickness);
  const auto forcing1 = ForcingOf<Evolution::Forcing>(c);
  const auto run1 = migrating ? evolution1.RunMigrating(c.times, c.iceThickness, forcing1)
                              : evolution1.Run(c.times, forcing1);
  const auto trajectory1 = ForwardTrajectory::FromSnapshots(run1);
  const auto oceans1 =
      migrating ? OceanFunctionsOf(run1) : std::vector<std::vector<double>>{};

  const auto zero = [](double, double, double) { return 0.0; };
  auto options3 = Viscoelastic3D::Options{.memory_points = c.points};
  if (wholeSolid) options3.moduli = EverySolidElement();
  Viscoelastic3D op3(rheology, kLmax, zero, options3);
  Evolution3D evolution3(op3, scene.solver, c.state,
                         {.scheme = c.scheme,
                          .rotation = c.rotation,
                          .rtol = 1e-12,
                          .coupled_rtol = 1e-12,
                          .record_stride = c.stride});
  const auto forcing3 = ForcingOf<Evolution3D::Forcing>(c);
  const auto run3 = migrating ? evolution3.RunMigrating(c.times, c.iceThickness, forcing3)
                              : evolution3.Run(c.times, forcing3);
  const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(run3);
  const auto oceans3 =
      migrating ? OceanFunctionsOf(run3) : std::vector<std::vector<double>>{};

  const LateralKernelSolver radial(evolution1, {.method = method, .rtol = 1e-12},
                                   {.viscosity = c.viscosity, .moduli = true});
  EXPECT_EQ(radial.GetRoute(), LateralKernelSolver::Route::Radial);
  const auto viaRadial =
      radial.FirstOrderKernels(c.times, c.measurements, trajectory1, oceans1);
  const auto via3D = AdjointSolver3D(evolution3, {.method = method,
                                                  .rtol = 1e-12,
                                                  .coupled_rtol = 1e-12,
                                                  .material_kernels = true})
                         .FirstOrderKernels(c.times, c.measurements, &trajectory3, oceans3);
  const auto gaps = Compare(viaRadial, c.viscosity ? via3D : [&] {
    auto out = via3D;
    out.itauKernel3D.clear();
    out.logViscosityKernel3D.clear();
    return out;
  }(), radial.Coupling(), op3.Coupling());
  std::cout << std::format(
      "[{}, {}{}] ln eta {:.2e} (itau {:.2e}), kappa {:.2e}, mu {:.2e}; the "
      "load kernels {:.2e}\n",
      c.label, NameOf(method), wholeSolid ? "" : ", viscous nodes", gaps.logViscosity, gaps.itau, gaps.kappa, gaps.mu,
      gaps.ice);
  return gaps;
}

void ExpectGate(const Gaps& gaps, double tolerance) {
  EXPECT_LT(gaps.itau, tolerance);
  EXPECT_LT(gaps.logViscosity, tolerance);
  EXPECT_LT(gaps.kappa, tolerance);
  EXPECT_LT(gaps.mu, tolerance);
  EXPECT_LT(gaps.ice, tolerance);
}

/// The gate of one configuration and method: against the 3D sweep on the
/// 3D operator of zero lateral viscosity (its memory nodes the viscous
/// elements'), and against the 3D sweep on one whose memory nodes cover
/// the whole solid (a lateral modulus one part in 1e100), where the
/// radial route's moduli kernels live — the latter closing its jumps and
/// equilibria by the instantaneous fixed point of lateral moduli.
void Gate(const Scene& scene, const Case& c, AdjointMethod method) {
  if (c.rheology == nullptr || c.rheology->BranchCount() > 0) {
    ExpectGate(RunGate(scene, c, method, false), 1e-9);
  }
  ExpectGate(RunGate(scene, c, method, true), 1e-9);
}

/// A point source 400 km down, its moment growing through the run.
std::function<std::vector<PointSource>(double)> GrowingSource(const Scene& scene,
                                                               double span) {
  const auto& sle = scene.solver.SolverScales();
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto unit = 1e21 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto at = BodyPoint{.radius = surface - 400e3 / sle.Length(),
                            .colatitude = 1.1,
                            .longitude = 4.0};
  return [=](double t) {
    const auto s = 0.5 + t / span;
    return std::vector<PointSource>{
        {.at = at,
         .moment = {.rr = 0.9 * unit * s, .tt = -0.2 * unit * s,
                    .pp = -0.7 * unit * s, .rt = 0.3 * unit * s,
                    .rp = -0.7 * unit * s, .tp = 0.6 * unit * s}}};
  };
}

/// Every kind of measurement at a node.
Measurement EveryField(const Scene& scene, std::mt19937& rng, bool seaLevel) {
  Measurement m;
  if (seaLevel) m.relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  m.displacementDual = scene.Smooth(rng, 0.5);
  m.potentialDual = scene.Smooth(rng, 0.7);
  m.tangentialPotentialDual = scene.Smooth(rng, -0.6);
  m.toroidalPotentialDual = scene.Smooth(rng, 0.8);
  m.angularVelocityDual = {0.2, -0.4, 0.8};
  return m;
}

}  // namespace

TEST(LateralKernels, ZeroLateralGateLoadsAndAnOcean) {
  // A melting ice load on the analytic ocean, measured in the sea level
  // and the displacement: the radial route's kernels in the log viscosity
  // and the moduli are the 3D sweep's at zero lateral structure, each
  // method, the memory points at the GLL nodes and at the Gauss points,
  // both schemes.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(41);
  Case c;
  for (auto n = 0; n <= 4; ++n) c.times.push_back(2.0 * n * scene.Kyr());
  c.state = scene.state;
  c.directLoad = Ramp(scene.MeltLoad(-0.1), c.times.back());
  c.measurements[4].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  c.measurements[2].displacementDual = scene.Smooth(rng, 0.5);
  for (const auto points : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    for (const auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
      c.points = points;
      c.scheme = scheme;
      c.label = std::format("loads, {} points, {}",
                            points == MemoryPointSet::Gll ? "GLL" : "Gauss",
                            scheme == Scheme::BackwardEuler ? "backward Euler"
                                                            : "trapezoid");
      for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
        Gate(scene, c, method);
      }
    }
  }
}

TEST(LateralKernels, ZeroLateralGateRotationAndEveryMeasurement) {
  // The full rotational feedback, every field measured at three nodes
  // (the toroidal potential among them, whose adjoint the radial route
  // carries for the lateral kernels alone), the toroidal potential alone
  // at a fourth (its jump forces the sea level solve with nothing but the
  // lateral coupling's leak), and uneven steps.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(43);
  Case c;
  c.times = {0.0, 1.0 * scene.Kyr(), 4.0 * scene.Kyr(), 4.5 * scene.Kyr(),
             6.5 * scene.Kyr(), 8.0 * scene.Kyr()};
  c.state = scene.state;
  c.rotation = RotationalFeedback::Full;
  c.directLoad = Ramp(scene.MeltLoad(-0.1), c.times.back());
  c.measurements[5] = EveryField(scene, rng, true);
  c.measurements[3] = EveryField(scene, rng, true);
  c.measurements[0] = EveryField(scene, rng, true);
  c.measurements[4].toroidalPotentialDual = scene.Smooth(rng, 0.8);
  c.label = "rotation, every field, uneven steps";
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    Gate(scene, c, method);
  }
}

TEST(LateralKernels, ZeroLateralGateWithoutAnOcean) {
  // No ocean: a surface load, a load outside the water budget and a
  // tide, measured in the displacement, the potentials and the rotation,
  // the axial feedback on.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(47);
  Case c;
  for (auto n = 0; n <= 3; ++n) c.times.push_back(3.0 * n * scene.Kyr());
  c.state = scene.solver.NoOceanState();
  c.rotation = RotationalFeedback::Axial;
  c.directLoad = Ramp(scene.Smooth(rng, 1e4 * scene.Sigma()), c.times.back());
  auto tide = scene.solver.Grid().Expand(
      scene.Smooth(rng, 1.0 / scene.solver.SolverScales().Factor(PotentialDimensions)));
  ZeroBelowTideDegrees(tide);
  c.tide = [tide](double t) {
    auto out = tide;
    for (auto& v : out) v *= std::cos(t);
    return out;
  };
  c.measurements[3] = EveryField(scene, rng, false);
  c.measurements[1] = EveryField(scene, rng, false);
  c.label = "no ocean, a load and a tide";
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    Gate(scene, c, method);
  }
}

TEST(LateralKernels, ZeroLateralGateUnderASeismicSource) {
  // A point source growing through the run beside the ice load: the
  // forward run carries toroidal motion, which the lateral kernels pair
  // pointwise with the spheroidal adjoint fields and the toroidal ones
  // with the spheroidal forward fields.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(53);
  Case c;
  for (auto n = 0; n <= 3; ++n) c.times.push_back(1.0 * n * scene.Kyr());
  c.state = scene.state;
  c.rotation = RotationalFeedback::Full;
  c.directLoad = Ramp(scene.MeltLoad(-0.05), c.times.back());
  c.sources = GrowingSource(scene, c.times.back());
  c.measurements[3] = EveryField(scene, rng, true);
  c.measurements[1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  c.label = "a seismic source, every field";
  for (const auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    c.scheme = scheme;
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      Gate(scene, c, method);
    }
  }
  // without the toroidal measurement: the toroidal forward fields against
  // the spheroidal adjoint alone
  c.measurements[3].toroidalPotentialDual.clear();
  c.label = "a seismic source, no toroidal measurement";
  c.scheme = Scheme::ExponentialTrapezoid;
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    Gate(scene, c, method);
  }
}

TEST(LateralKernels, ZeroLateralGateMigratingShorelines) {
  // A marine ice cap melting on a uniform ocean, its shorelines migrating,
  // the start solved under the first node's forcing (a load beside the
  // ice and a layer on the solid surface): the radial route about the
  // run's own ocean functions against the 3D sweep about the 3D run's.
  auto scene = Scene::Make(1e21);
  const auto& grid = scene.solver.Grid();
  const auto metre = 1.0 / scene.solver.SolverScales().Length();
  std::mt19937 rng(59);
  Case c;
  for (auto n = 0; n <= 3; ++n) c.times.push_back(10.0 * n / 3.0 * scene.Kyr());
  const auto ice0 = grid.Synthesise(grid.Expand(grid.FromFunction([&](double theta, double phi) {
    const auto theta0 = 30.0 * std::numbers::pi / 180.0;
    const auto phi0 = 275.0 * std::numbers::pi / 180.0;
    const auto radius = 20.0 * std::numbers::pi / 180.0;
    const auto cosd = std::cos(theta) * std::cos(theta0) +
                      std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
    const auto d = std::acos(std::clamp(cosd, -1.0, 1.0));
    if (d >= radius) return 0.0;
    const auto taper = std::cos(0.5 * std::numbers::pi * d / radius);
    return 5000.0 * metre * taper * taper;
  })));
  c.state = scene.solver.MakeState(ice0, std::vector<double>(grid.FieldSize(), 3000.0 * metre));
  const auto span = c.times.back();
  c.iceThickness = [ice0, span](double t) {
    auto out = ice0;
    for (auto& v : out) v *= std::clamp(1.0 - t / span, 0.0, 1.0);
    return out;
  };
  c.directLoad = [load = scene.Smooth(rng, 2e4 * scene.Sigma())](double) { return load; };
  c.layer = [layer = scene.Smooth(rng, 400.0 * metre)](double) { return layer; };
  c.rotation = RotationalFeedback::Full;
  c.measurements[3].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  c.measurements[2].displacementDual = scene.Smooth(rng, 0.5);
  c.label = "migrating shorelines, a solved start";
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    Gate(scene, c, method);
  }
}

TEST(LateralKernels, ZeroLateralGateTheStaticProblem) {
  // One time node, the elastic problem, on a rheology with no branch at
  // all (the static problem's) and on the Maxwell one: with an ocean and
  // without one, every field measured, a seismic source beside the load.
  auto scene = Scene::Make(1e21);
  const auto elastic = Rheology::Create(scene.material, {}).value();
  std::mt19937 rng(61);
  Case c;
  c.times = {0.0};
  c.rotation = RotationalFeedback::Full;
  c.directLoad = [load = scene.MeltLoad(-0.1)](double) { return load; };
  c.sources = GrowingSource(scene, 1.0);
  for (const Rheology* rheology : {&elastic, static_cast<const Rheology*>(&scene.rheology)}) {
    for (const auto ocean : {true, false}) {
      c.rheology = rheology;
      c.viscosity = false;
      c.state = ocean ? scene.state : scene.solver.NoOceanState();
      c.measurements.clear();
      c.measurements[0] = EveryField(scene, rng, ocean);
      c.label = std::format("one node, {}, {}",
                            rheology == &elastic ? "no branch" : "Maxwell",
                            ocean ? "an ocean" : "no ocean");
      for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
        Gate(scene, c, method);
      }
    }
  }
}


TEST(LateralKernels, ACheckpointedRecordingAsTheRadialKernels) {
  // A recording kept at every second node: the radial sweeps rebuild the
  // gaps by exact backward steps, and the lateral kernels come from the
  // rebuilt states as the radial ones do, equal to the full recording's to
  // the reconstruction's roundoff (exp(sum h) across a gap; here the
  // steps are a quarter of the relaxation time).
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(67);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 6; ++n) times.push_back(1.0 * n * scene.Kyr());
  const auto load = Ramp(scene.MeltLoad(-0.1), times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[6] = EveryField(scene, rng, true);
  measurements[3].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  ViscoelasticOperator op(scene.rheology, kLmax);
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    auto kernels = std::vector<Kernels3D>{};
    for (const auto stride : {1, 2}) {
      Evolution evolution(op, scene.solver, scene.state,
                          {.rotation = RotationalFeedback::Full,
                           .rtol = 1e-12,
                           .record_stride = stride});
      const auto trajectory = ForwardTrajectory::FromSnapshots(evolution.Run(times, load));
      kernels.push_back(LateralKernelSolver(evolution, {.method = method, .rtol = 1e-12},
                                            {.viscosity = true, .moduli = true})
                            .FirstOrderKernels(times, measurements, trajectory));
    }
    const auto lnEta = Departure(kernels[1].logViscosityKernel3D[0],
                                 kernels[0].logViscosityKernel3D[0]);
    const auto mu = Departure(kernels[1].muKernel3D, kernels[0].muKernel3D);
    const auto radial = Departure(kernels[1].ice.itauKernel[0], kernels[0].ice.itauKernel[0]);
    std::cout << std::format(
        "[stride 2 against 1, {}] ln eta {:.2e}, mu {:.2e}; the radial itau kernel "
        "{:.2e}\n",
        NameOf(method), lnEta, mu, radial);
    EXPECT_LT(lnEta, 1e-9);
    EXPECT_LT(mu, 1e-9);
  }
}

// -- finite differences of the 3D forward run ---------------------------------

namespace {

/// The objective the measurements define on a 3D run's snapshots.
double Objective(const Scene& scene, const std::vector<Evolution3D::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  const auto a = scene.solver.Radius();
  auto surface = [&](std::span<const double> f, std::span<const double> g) {
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * scene.solver.Grid().Integrate(fg);
  };
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.relativeSeaLevelDual.empty()) {
      total += surface(m.relativeSeaLevelDual, snap.relativeSeaLevelChange);
    }
    if (!m.displacementDual.empty()) total += surface(m.displacementDual, snap.displacement);
    if (!m.potentialDual.empty()) total += surface(m.potentialDual, snap.potential);
    if (!m.tangentialPotentialDual.empty()) {
      total += surface(m.tangentialPotentialDual, snap.tangentialPotential);
    }
    if (!m.toroidalPotentialDual.empty() && !snap.toroidalPotential.empty()) {
      total += surface(m.toroidalPotentialDual, snap.toroidalPotential);
    }
    for (auto j = 0; j < 3; ++j) total += m.angularVelocityDual[j] * snap.angularVelocity[j];
  }
  return total;
}

using Pattern = std::function<double(double, double, double)>;

/// A pattern sampled over a coupling's grid values, times a per-node
/// factor (a modulus at the node, say).
std::vector<double> Sampled(const GridCoupling& coupling, const Pattern& f,
                            std::span<const double> factor = {}) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto r = coupling.RadiusOfNode(rn);
    const auto& grid = coupling.GridOf(rn);
    auto x = coupling.FieldOffset(rn);
    const auto scale = factor.empty() ? 1.0 : factor[static_cast<std::size_t>(rn)];
    for (auto theta : grid.CoLatitudes()) {
      for (auto phi : grid.Longitudes()) out[x++] = scale * f(r, theta, phi);
    }
  }
  return out;
}

/// A modulus at the memory points over the mesh, read at a coupling's
/// nodes.
std::vector<double> AtNodes(const GridCoupling& coupling, std::span<const double> atPoints) {
  const auto np = static_cast<std::size_t>(coupling.Points().count);
  auto out = std::vector<double>(static_cast<std::size_t>(coupling.NodeCount()));
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    out[static_cast<std::size_t>(rn)] =
        atPoints[static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
                 coupling.NodeInElementOf(rn)];
  }
  return out;
}

}  // namespace

TEST(LateralKernels, AgainstDifferencesOfThe3DRun) {
  // The lateral kernels about the radial Earth paired with a smooth
  // lateral perturbation — of the log viscosity, of the unrelaxed shear and
  // of the bulk modulus, each a pattern of low degree (2) and one of high
  // (6, at lmax 8) — against central differences of the 3D forward run
  // with that perturbation, Richardson-extrapolated over steps of 2% and
  // 1%: the discrete kernels to the differences' own error; the continuous
  // ones at their commute error, which falls at the trapezoid's second
  // order as the time steps halve.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(71);
  // one problem at every resolution: the same measurements at the same
  // times (the end and the middle)
  const auto seaLevel = scene.Smooth(rng, 1.0);
  const auto lateDisplacement = scene.Smooth(rng, 0.4);
  const auto midDisplacement = scene.Smooth(rng, 0.5);
  const auto zero = [](double, double, double) { return 0.0; };
  const auto low = Pattern([](double r, double theta, double phi) {
    return (0.5 + r * r) * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto high = Pattern([](double r, double theta, double phi) {
    return (1.2 - r) * std::pow(std::sin(theta), 6) * std::cos(6.0 * phi + 0.3);
  });
  ViscoelasticOperator op1(scene.rheology, kLmax);
  const auto muU = std::vector<double>(op1.UnrelaxedShearAtPoints().begin(),
                                       op1.UnrelaxedShearAtPoints().end());
  const auto kappa = std::vector<double>(op1.BulkModulusAtPoints().begin(),
                                         op1.BulkModulusAtPoints().end());
  // the gap of the continuous kernels at each of two time grids, per case
  auto commute = std::map<std::string, std::vector<double>>{};
  for (const auto steps : {8, 16}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n <= steps; ++n) times.push_back(16.0 * n / steps * scene.Kyr());
    const auto load = Ramp(scene.MeltLoad(-0.1), times.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[static_cast<std::size_t>(steps)].relativeSeaLevelDual = seaLevel;
    measurements[static_cast<std::size_t>(steps)].displacementDual = lateDisplacement;
    measurements[static_cast<std::size_t>(steps / 2)].displacementDual = midDisplacement;
    Evolution evolution1(op1, scene.solver, scene.state,
                         {.rotation = RotationalFeedback::None,
                          .rtol = 1e-12,
                          .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(evolution1.Run(times, load));
    // J of the 3D run with a lateral perturbation
    auto objective = [&](const Pattern& delta, const Viscoelastic3D::Options::LateralModuli& moduli) {
      Viscoelastic3D op3(scene.rheology, kLmax, delta, {.moduli = moduli});
      Evolution3D evolution3(op3, scene.solver, scene.state,
                             {.rotation = RotationalFeedback::None,
                              .rtol = 1e-13,
                              .coupled_rtol = 1e-13});
      return Objective(scene, evolution3.Run(times, load), measurements);
    };
    struct Direction {
      std::string name;
      Pattern pattern;
      int kind;  // 0 log viscosity, 1 shear, 2 bulk
    };
    const auto directions = std::vector<Direction>{
        {"ln eta, degree 2", low, 0}, {"ln eta, degree 6", high, 0},
        {"shear, degree 2", low, 1},  {"shear, degree 6", high, 1},
        {"bulk, degree 2", low, 2},   {"bulk, degree 6", high, 2}};
    auto derivatives = std::vector<double>{};
    for (const auto& d : directions) {
      auto at = [&](double eps) {
        const auto scaled = Pattern([&, eps](double r, double t, double p) {
          return eps * d.pattern(r, t, p);
        });
        if (d.kind == 0) return objective(scaled, {});
        if (d.kind == 1) return objective(zero, {.shear = scaled, .bulk = {}});
        return objective(zero, {.shear = {}, .bulk = scaled});
      };
      const auto wide = (at(0.02) - at(-0.02)) / 0.04;
      const auto narrow = (at(0.01) - at(-0.01)) / 0.02;
      derivatives.push_back((4.0 * narrow - wide) / 3.0);
    }
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const LateralKernelSolver solver(evolution1, {.method = method, .rtol = 1e-12},
                                       {.viscosity = true, .moduli = true});
      const auto kernels = solver.FirstOrderKernels(times, measurements, trajectory);
      const auto& coupling = solver.Coupling();
      for (auto i = std::size_t{0}; i < directions.size(); ++i) {
        const auto& d = directions[i];
        auto paired = 0.0;
        if (d.kind == 0) {
          paired = VolumePairing(coupling, kernels.logViscosityKernel3D[0],
                                 Sampled(coupling, d.pattern));
        } else if (d.kind == 1) {
          paired = VolumePairing(coupling, kernels.muKernel3D,
                                 Sampled(coupling, d.pattern, AtNodes(coupling, muU)));
        } else {
          paired = VolumePairing(coupling, kernels.kappaKernel3D,
                                 Sampled(coupling, d.pattern, AtNodes(coupling, kappa)));
        }
        const auto gap = std::abs(paired - derivatives[i]) / std::abs(derivatives[i]);
        std::cout << std::format("[{} steps, {}, {}] kernel {:.10e}, differences {:.10e}, "
                                 "gap {:.2e}\n",
                                 steps, NameOf(method), d.name, paired, derivatives[i], gap);
        ASSERT_NE(derivatives[i], 0.0);
        if (method == AdjointMethod::Discrete) {
          EXPECT_LT(gap, 1e-6) << d.name;
        } else {
          commute[d.name].push_back(gap);
        }
      }
    }
  }
  for (const auto& [name, gaps] : commute) {
    std::cout << std::format("[continuous, {}] commute gap {:.2e} -> {:.2e}, ratio {:.2f}\n",
                             name, gaps[0], gaps[1], gaps[0] / gaps[1]);
    EXPECT_LT(gaps[0], 0.05) << name;
    EXPECT_GT(gaps[0] / gaps[1], 3.0) << name;
  }
}

TEST(LateralKernels, TheirMeansOverTheSphereAreTheRadialKernels) {
  // The lateral kernels' mean over the sphere at each memory node is the
  // radial kernel there (the kernel of an angle-uniform perturbation), to
  // rounding (1e-11; the dilatation's products cancel to parts in 1e12):
  // the grids integrate the products of band-limited fields exactly. In the viscosity at the GLL points (where the radial kernel's
  // nodes are the points) and in the moduli at both point sets, each
  // method, about a source-driven run measured in every field.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(73);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(1.5 * n * scene.Kyr());
  auto forcing = Evolution::Forcing{};
  forcing.directLoad = Ramp(scene.MeltLoad(-0.1), times.back());
  forcing.sources = GrowingSource(scene, times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3] = EveryField(scene, rng, true);
  measurements[1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  const auto ngll = static_cast<std::size_t>(scene.material.Mesh().NGll());
  for (const auto points : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    ViscoelasticOperator op(scene.rheology, kLmax, {.memory_points = points});
    Evolution evolution(op, scene.solver, scene.state,
                        {.rotation = RotationalFeedback::Full, .rtol = 1e-12, .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(evolution.Run(times, forcing));
    const auto np = static_cast<std::size_t>(op.Points().count);
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const LateralKernelSolver solver(evolution, {.method = method, .rtol = 1e-12},
                                       {.viscosity = true, .moduli = true});
      const auto kernels = solver.FirstOrderKernels(times, measurements, trajectory);
      const auto& coupling = solver.Coupling();
      auto itau = std::vector<double>{}, itauMean = std::vector<double>{};
      auto kappa = std::vector<double>{}, mu = std::vector<double>{};
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        const auto e = static_cast<std::size_t>(coupling.ElementOfNode(rn));
        const auto p = static_cast<std::size_t>(coupling.NodeInElementOf(rn));
        kappa.push_back(kernels.ice.kappaKernel.at(e * np + p));
        mu.push_back(kernels.ice.muKernel.at(e * np + p));
        if (points == MemoryPointSet::Gll && kernels.itauKernelSpherical[0][rn] != 0.0) {
          itau.push_back(kernels.ice.itauKernel[0].at(e * ngll + p));
          itauMean.push_back(kernels.itauKernelSpherical[0][rn]);
        }
      }
      const auto gapKappa = Departure(kernels.kappaKernelSpherical, kappa);
      const auto gapMu = Departure(kernels.muKernelSpherical, mu);
      const auto gapItau = itau.empty() ? 0.0 : Departure(itauMean, itau);
      std::cout << std::format("[{} points, {}] the means against the radial kernels: "
                               "itau {:.2e}, kappa {:.2e}, mu {:.2e}\n",
                               points == MemoryPointSet::Gll ? "GLL" : "Gauss",
                               NameOf(method), gapItau, gapKappa, gapMu);
      EXPECT_LT(gapKappa, 1e-11);
      EXPECT_LT(gapMu, 1e-11);
      EXPECT_LT(gapItau, 1e-11);
      // and the uniform perturbation of ln eta wherever the kernels live
      // pairs with the lateral kernel as with the radial one
      const auto ones = std::vector<double>(coupling.GridValueCount(), 1.0);
      const auto lateral = VolumePairing(coupling, kernels.logViscosityKernel3D[0], ones);
      auto radial = 0.0;
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        const auto r = coupling.RadiusOfNode(rn);
        radial += 4.0 * std::numbers::pi * coupling.WeightOfNode(rn) * r * r *
                  kernels.itauKernelSpherical[0][rn] *
                  -op.PointItau(0)[static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
                                   static_cast<std::size_t>(coupling.NodeInElementOf(rn))];
      }
      EXPECT_NEAR(lateral, radial, 1e-12 * std::abs(radial));
    }
  }
}

// -- the static problem against ElasticPerturbation3D ------------------------

namespace {

using Complex = std::complex<double>;

/// The unit ratios between the material's scales and the sea level
/// solver's (the evolution's toNd / toMat factors).
struct Ratios {
  double toNdLength = 1.0, toNdPotential = 1.0, toMatDensity = 1.0, toMatPotential = 1.0;

  static Ratios Of(const Scales& material, const Scales& sle) {
    auto ratio = [&](Dimensions d) { return material.Factor(d) / sle.Factor(d); };
    return {.toNdLength = ratio(Dimensions{.length = 1}),
            .toNdPotential = ratio(PotentialDimensions),
            .toMatDensity = 1.0 / ratio(Dimensions{.mass = 1, .length = -2}),
            .toMatPotential = 1.0 / ratio(PotentialDimensions)};
  }
};

std::vector<Complex> Scaled(std::vector<Complex> c, double factor) {
  for (auto& v : c) v *= factor;
  return c;
}

/// A surface measurement a^2 int (gU U + gV V + gPhi phi) dOmega as the
/// static problem's surface duals (TestStaticDerivatives' construction):
/// each field's coefficients scaled by the solver's radius squared, the
/// coefficient pairing's normalisation against the grid's integral, and
/// the unit ratio of its channel.
ElasticPerturbation3D::Forces StaticDualForces(const ElasticPerturbation3D& born,
                                               const GRDSolver& solver,
                                               const Ratios& ratios,
                                               const Measurement& m) {
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();
  auto coefficients = grid.ZeroCoefficients();
  for (auto l = 0; l <= grid.LMax(); ++l) {
    for (auto mm = 0; mm <= l; ++mm) {
      coefficients[SurfaceGrid::Index(l, mm)] = {1.0 / (1.0 + l + mm),
                                                 mm == 0 ? 0.0 : 0.5 / (1.0 + l)};
    }
  }
  const auto probe = grid.Synthesise(coefficients);
  auto square = probe;
  for (auto& x : square) x *= x;
  const auto c = grid.Expand(probe);
  auto pairing = 0.0;
  for (auto l = 0; l <= grid.LMax(); ++l) {
    for (auto mm = 0; mm <= l; ++mm) {
      pairing += (mm == 0 ? 1.0 : 2.0) * std::norm(c[SurfaceGrid::Index(l, mm)]);
    }
  }
  const auto scale = a * a * grid.Integrate(square) / pairing;
  auto out = born.SurfaceDualForces(
      Scaled(grid.Expand(m.displacementDual), scale * ratios.toNdLength), Component::U);
  for (const auto& [dual, component] :
       {std::pair{Scaled(grid.Expand(m.tangentialPotentialDual), scale * ratios.toNdLength),
                  Component::V},
        std::pair{Scaled(grid.Expand(m.potentialDual), scale * ratios.toNdPotential),
                  Component::Phi}}) {
    for (const auto& [l, f] : born.SurfaceDualForces(dual, component)) {
      auto& to = out.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += f[i];
    }
  }
  return out;
}

double Seconds(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
}

/// The static problem on the radial body at one degree: the gaps of the
/// radial route's moduli kernels from minus the transpose of
/// ElasticPerturbation3D's Force on the static adjoint and forward
/// solutions (ForceTranspose; both methods, a load and a tide), and the
/// times of the two, kernels only and with their setup.
struct StaticComparison {
  double kappa = 0.0, mu = 0.0;
  double radialSetup = 0.0, radialKernels = 0.0, bornSetup = 0.0, bornKernels = 0.0;
};

StaticComparison CompareStaticRoutes(int lmax) {
  auto scene = Scene::Make(1e21, lmax);
  const auto elastic = Rheology::Create(scene.material, {}).value();
  ViscoelasticOperator op(elastic, lmax, {.memory_points = MemoryPointSet::Gll});
  Evolution evolution(op, scene.solver, scene.solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None, .rtol = 1e-12, .record_stride = 1});
  auto clock = std::chrono::steady_clock::now();
  const ElasticPerturbation3D born(scene.material, lmax);
  StaticComparison out;
  out.bornSetup = Seconds(clock);
  const auto& grid = scene.solver.Grid();
  const auto ratios = Ratios::Of(scene.material.ModelScales(), scene.solver.SolverScales());
  std::mt19937 rng(79);
  const auto load = scene.Smooth(rng, 100.0 * scene.Sigma());
  auto tide = grid.Expand(scene.Smooth(rng, 1.0 / scene.solver.SolverScales().Factor(PotentialDimensions)));
  ZeroBelowTideDegrees(tide);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].displacementDual = scene.Smooth(rng, 0.5);
  measurements[0].tangentialPotentialDual = scene.Smooth(rng, 0.6);
  measurements[0].potentialDual = scene.Smooth(rng, 0.7);
  const auto times = std::vector<double>{0.0};
  for (const bool tidal : {false, true}) {
    auto forcing = Evolution::Forcing{};
    if (tidal) {
      forcing.tide = [&](double) { return tide; };
    } else {
      forcing.directLoad = [&](double) { return load; };
    }
    const auto trajectory = ForwardTrajectory::FromSnapshots(evolution.Run(times, forcing));
    clock = std::chrono::steady_clock::now();
    const auto y = born.Solve(StaticDualForces(born, scene.solver, ratios, measurements[0]), {});
    const auto x =
        born.Solve(tidal ? born.TideForces(Scaled(tide, ratios.toMatPotential))
                         : born.LoadForces(Scaled(grid.Expand(load), ratios.toMatDensity)),
                   {});
    auto transpose = born.ForceTranspose(y, x);
    out.bornKernels = Seconds(clock);
    for (auto& v : transpose.kappa) v = -v;
    for (auto& v : transpose.mu) v = -v;
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      clock = std::chrono::steady_clock::now();
      const LateralKernelSolver solver(evolution, {.method = method, .rtol = 1e-12},
                                       {.viscosity = false, .moduli = true});
      out.radialSetup = Seconds(clock);
      clock = std::chrono::steady_clock::now();
      const auto kernels = solver.FirstOrderKernels(times, measurements, trajectory);
      out.radialKernels = Seconds(clock);
      const auto& a = solver.Coupling();
      const auto& b = born.Coupling();
      EXPECT_EQ(a.NodeCount(), b.NodeCount());
      EXPECT_EQ(a.GridValueCount(), b.GridValueCount());
      for (auto rn = 0; rn < std::min(a.NodeCount(), b.NodeCount()); ++rn) {
        EXPECT_EQ(a.ElementOfNode(rn), b.ElementOfNode(rn));
        EXPECT_EQ(a.NodeInElementOf(rn), b.NodeInElementOf(rn));
      }
      out.kappa = std::max(out.kappa, Departure(kernels.kappaKernel3D, transpose.kappa));
      out.mu = std::max(out.mu, Departure(kernels.muKernel3D, transpose.mu));
    }
  }
  return out;
}

}  // namespace

TEST(LateralKernels, TheStaticRouteAgreesWithTheTransposeOfTheForce) {
  // The entry point's static case against an independent form of the
  // same kernels: on a radial body without an ocean or rotational
  // feedback, measured in U, V and phi, the memory points at the GLL
  // nodes, the radial route at one node is minus the transpose of
  // ElasticPerturbation3D's Force on the static adjoint and forward
  // solutions (ForceTranspose), at every grid value. The two times are
  // printed (the transpose's are its solves and expansions alone).
  const auto c = CompareStaticRoutes(kLmax);
  std::cout << std::format(
      "[static, lmax {}] kappa {:.2e}, mu {:.2e}; radial route {:.3f} s (+{:.3f} s "
      "setup), ElasticPerturbation3D {:.3f} s (+{:.3f} s setup)\n",
      kLmax, c.kappa, c.mu, c.radialKernels, c.radialSetup, c.bornKernels, c.bornSetup);
  EXPECT_LT(c.kappa, 1e-10);
  EXPECT_LT(c.mu, 1e-10);
}

TEST(LateralKernels, DISABLED_StaticRoutesTiming) {
  // The same at higher degree, for the cost of the route and of the
  // transpose (SL3DPP_LMAX, default 32).
  const auto* env = std::getenv("SL3DPP_LMAX");
  const auto lmax = env != nullptr ? std::atoi(env) : 32;
  const auto c = CompareStaticRoutes(lmax);
  std::cout << std::format(
      "[static, lmax {}] kappa {:.2e}, mu {:.2e}; radial route {:.3f} s (+{:.3f} s "
      "setup), ElasticPerturbation3D {:.3f} s (+{:.3f} s setup)\n",
      lmax, c.kappa, c.mu, c.radialKernels, c.radialSetup, c.bornKernels, c.bornSetup);
}

// -- cost -----------------------------------------------------------------------

namespace {

std::size_t BytesOf(const ViscoelasticState& state) {
  auto bytes = std::size_t{0};
  for (const auto& [l, m] : state.m) bytes += m.size() * sizeof(double);
  for (const auto& [l, d] : state.strain) bytes += d.size() * sizeof(double);
  return bytes;
}

std::size_t BytesOf(const GridCoupling::TensorField& t) {
  return (t.c[0].size() + t.c[1].size()) * sizeof(Complex) + t.r.size() * sizeof(double);
}

std::size_t BytesOf(const Viscoelastic3D::State& state) {
  auto bytes = std::size_t{0};
  for (const auto& m : state.m) bytes += BytesOf(m);
  if (state.strain) bytes += BytesOf(*state.strain);
  if (state.strainCoefficients) {
    const auto& s = *state.strainCoefficients;
    bytes += (s.c[0].size() + s.c[1].size() + s.r.size()) * sizeof(Complex);
  }
  return bytes;
}

}  // namespace

TEST(LateralKernels, DISABLED_CostAgainstTheAlternatives) {
  // The lateral kernels about a radial Earth by the radial route against
  // the radial kernels alone and against the 3D sweep at zero lateral
  // structure (its forward run included), each method, 10 steps of a
  // 1e21 Pa s mantle, a sea-level and a displacement measurement
  // (SL3DPP_LMAX, default 32; the kernels in the viscosity and in the
  // moduli, the 3D operator's memory nodes over the whole solid like the
  // radial route's).
  const auto* env = std::getenv("SL3DPP_LMAX");
  const auto lmax = env != nullptr ? std::atoi(env) : 32;
  auto scene = Scene::Make(1e21, lmax);
  std::mt19937 rng(83);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 10; ++n) times.push_back(1.0 * n * scene.Kyr());
  const auto load = Ramp(scene.MeltLoad(-0.1), times.back());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[10].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  measurements[5].displacementDual = scene.Smooth(rng, 0.5);
  for (const auto moduli : {false, true}) {
    ViscoelasticOperator op1(scene.rheology, lmax);
    Evolution evolution1(op1, scene.solver, scene.state,
                         {.rotation = RotationalFeedback::None, .rtol = 1e-10, .record_stride = 1});
    auto clock = std::chrono::steady_clock::now();
    const auto run1 = evolution1.Run(times, load);
    const auto forward1 = Seconds(clock);
    const auto trajectory1 = ForwardTrajectory::FromSnapshots(run1);
    auto bytes1 = std::size_t{0};
    for (const auto& s : trajectory1.states) bytes1 += BytesOf(*s);

    const auto zero = [](double, double, double) { return 0.0; };
    auto options3 = Viscoelastic3D::Options{};
    if (moduli) options3.moduli = EverySolidElement();
    Viscoelastic3D op3(scene.rheology, lmax, zero, options3);
    Evolution3D evolution3(op3, scene.solver, scene.state,
                           {.rotation = RotationalFeedback::None, .rtol = 1e-10, .record_stride = 1});
    clock = std::chrono::steady_clock::now();
    const auto run3 = evolution3.Run(times, load);
    const auto forward3 = Seconds(clock);
    const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(run3);
    auto bytes3 = std::size_t{0};
    for (const auto& s : trajectory3.states) bytes3 += BytesOf(*s);
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      clock = std::chrono::steady_clock::now();
      const auto radialOnly =
          AdjointSolver(evolution1, {.method = method, .rtol = 1e-10, .material_kernels = moduli})
              .FirstOrderKernels(times, measurements, &trajectory1);
      const auto radialSweep = Seconds(clock);
      clock = std::chrono::steady_clock::now();
      const LateralKernelSolver solver(evolution1, {.method = method, .rtol = 1e-10},
                                       {.viscosity = true, .moduli = moduli});
      const auto lateral = solver.FirstOrderKernels(times, measurements, trajectory1);
      const auto lateralSweep = Seconds(clock);
      clock = std::chrono::steady_clock::now();
      const auto via3D = AdjointSolver3D(evolution3, {.method = method,
                                                      .rtol = 1e-10,
                                                      .material_kernels = moduli})
                             .FirstOrderKernels(times, measurements, &trajectory3);
      const auto sweep3 = Seconds(clock);
      const auto& coupling = solver.Coupling();
      std::cout << std::format(
          "[lmax {}, {}, {}] radial kernels {:.2f} s; lateral by the radial route "
          "{:.2f} s; 3D sweep {:.2f} s (+ its forward run {:.2f} s, the radial run's "
          "{:.2f} s); recordings {:.1f} MB radial, {:.1f} MB 3D; the kernels {:.1f} "
          "MB over {} nodes x {} values\n",
          lmax, NameOf(method), moduli ? "viscosity and moduli" : "viscosity", radialSweep,
          lateralSweep, sweep3, forward3, forward1, bytes1 / 1e6, bytes3 / 1e6,
          (lateral.itauKernel3D.size() + lateral.logViscosityKernel3D.size() +
           (moduli ? 2 : 0)) * coupling.GridValueCount() * sizeof(double) / 1e6,
          coupling.NodeCount(), coupling.GridValueCount());
      static_cast<void>(radialOnly);
      static_cast<void>(via3D);
    }
  }
}
