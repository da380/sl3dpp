// The derivative classes about migrating-shoreline runs whose start is
// solved: a run whose ice at the first node is not the background state's
// (the start's nonlinear solve moves the shorelines there), and a run
// forced beside the ice by loads, a layer on the solid surface and a point
// source. The tangent-linear runs against central differences of
// migrating runs that flip no cell, the discrete sweeps as their
// transposes, the continuous sweeps converging to the discrete ones, the
// second-order sweeps symmetric and equal to the second-order runs, and
// the kernels in the moduli — 1D and, under lateral viscosity, 3D.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

/// The node of a grid at time t.
std::size_t NodeOf(std::span<const double> times, double t) {
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) return n;
  }
  throw std::runtime_error("the test history is read off its nodes");
}

/// A per-node field history as the forward driver's callable, moved by
/// eps times a per-node direction (empty: none).
std::function<std::vector<double>(double)> Field(
    std::vector<double> times, std::vector<std::vector<double>> base,
    std::vector<std::vector<double>> direction = {}, double eps = 0.0) {
  return [times = std::move(times), base = std::move(base),
          direction = std::move(direction), eps](double t) {
    const auto n = NodeOf(times, t);
    auto out = base[n];
    if (!direction.empty()) {
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += eps * direction[n][i];
    }
    return out;
  };
}

std::vector<double> Moved(std::vector<double> x, const std::vector<double>& d,
                          double eps) {
  for (auto i = std::size_t{0}; i < x.size(); ++i) x[i] += eps * d[i];
  return x;
}

/// Central differences at eps and eps / 2 against a prediction: the
/// ratio of their residuals (four at second order) and the Richardson
/// value.
struct Differences {
  double ratio = 0.0;
  double richardson = 0.0;
};
Differences Difference(const std::function<double(double)>& f, double eps,
                       double predicted) {
  const auto wide = (f(eps) - f(-eps)) / (2.0 * eps);
  const auto narrow = (f(0.5 * eps) - f(-0.5 * eps)) / eps;
  return {.ratio = std::abs(wide - predicted) / std::abs(narrow - predicted),
          .richardson = (4.0 * narrow - wide) / 3.0};
}

double Gap(double got, double want) { return std::abs(got - want) / std::abs(want); }

double NodePairing(const Scene& setup, const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& d) {
  auto out = 0.0;
  for (auto n = std::size_t{0}; n < d.size(); ++n) out += setup.Surface(kernel[n], d[n]);
  return out;
}

/// The operator with its inverse relaxation times moved by eps along ditau.
ViscoelasticOperator MovedItau(const Scene& setup, double eps,
                               const std::vector<std::vector<double>>& ditau) {
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < setup.rheology.BranchCount(); ++k) {
    auto branch = setup.rheology.BranchAt(k);
    for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
      branch.itau[i] += eps * ditau[k][i];
    }
    branches.push_back(std::move(branch));
  }
  return ViscoelasticOperator(Rheology::Create(setup.material, branches).value(), kLmax);
}

/// The cap's thickness at the nodes of a grid: a fraction `start` of the
/// background's cap at the first node, melting linearly to `end` of it.
std::vector<std::vector<double>> Thinning(const MarineCap& cap,
                                          std::span<const double> times, double start,
                                          double end) {
  auto out = std::vector<std::vector<double>>{};
  for (const auto t : times) {
    const auto s = t / times.back();
    auto ice = cap.ice0;
    for (auto& v : ice) v *= start + (end - start) * s;
    out.push_back(std::move(ice));
  }
  return out;
}

}  // namespace

// -- an ice change at the first node, 1D ------------------------------------

TEST(MigratingDerivatives, AStartWithAnIceChange) {
  // The marine cap's background state, and an ice history at 85% of its
  // cap at the first node, thinning to 35% by the end, with no other
  // forcing: the run's start is the nonlinear solve under the ice change
  // there, which moves the shorelines, and they move on through the run.
  //
  // The classes take a thickness direction relative to the first node's:
  // its load at node n is rho_i (1 - C_n) dI_n - rho_i (1 - C) dI_0, C the
  // background's ocean function — the derivative with the background
  // state's ice moved with the first node's. Gated against central
  // differences of migrating runs that flip no cell, both methods:
  //  (a) a viscosity direction (the discrete run at second order with
  //      Richardson to 1e-6; the continuous within the commute level);
  //  (b) a thickness direction non-zero at the first node, against runs
  //      with the background's ice moved with the first node's; with the
  //      background held instead (the run's own ice change moved, as an
  //      ice change from a given state is), the derivative is the
  //      tangent's plus the run along the load rho_i (1 - C) dI_0 at
  //      every node, which the kernels give as the thickness view with the
  //      baseline taken back out of its first node;
  //  (c) a thickness direction zero at the first node, where the two
  //      readings agree;
  //  (d) the initial sea level.
  // The discrete sweep pairs with each to the tangent's number.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto cap = MarineCap::Make(setup, 5);
  const auto& times = cap.times;
  const auto nodes = times.size();
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  const auto ice = Thinning(cap, times, 0.85, 0.35);
  const auto rhoI = setup.solver.IceDensity();
  std::mt19937 rng(89);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 0.4);
  measurements[1].displacementDual = setup.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[nodes - 1].potentialDual = setup.Smooth(rng, 0.5);
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    deltaIce.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
  }
  auto rising = deltaIce;
  rising[0].assign(rising[0].size(), 0.0);
  const auto deltaSea = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto ditau = setup.DeltaItau(0.1);
  const Evolution::Options options{
      .rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1};

  Evolution evolution(op, setup.solver, state, options);
  const auto run = evolution.RunMigrating(times, Field(times, ice), Evolution::Forcing{});
  const auto oceans = OceanFunctionsOf(run);
  ASSERT_NE(oceans.front(), state.oceanFunction) << "the start moves no shoreline";
  ASSERT_NE(oceans.back(), oceans.front());
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);

  // The objective of a migrating run from a state, under the ice history
  // moved by eps along a direction, every node's shorelines checked.
  auto objective = [&](const ViscoelasticOperator& at, const GRDSolver::State& start,
                       const std::vector<std::vector<double>>& direction, double eps) {
    EXPECT_EQ(start.oceanFunction, state.oceanFunction) << "the background flipped";
    Evolution trial(at, setup.solver, start, options);
    const auto snaps =
        trial.RunMigrating(times, Field(times, ice, direction, eps), Evolution::Forcing{});
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_TRUE(snaps[n].converged) << "node " << n;
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped at node " << n;
    }
    return Objective(setup, snaps, measurements);
  };
  auto held = [&](const std::vector<std::vector<double>>& direction) {
    return 0.5 * (objective(op, state, direction, 1.0) - objective(op, state, direction, -1.0));
  };
  auto withBackground = [&](const std::vector<std::vector<double>>& direction) {
    auto run = [&](double sign) {
      const auto start = setup.solver.MakeState(Moved(state.iceThickness, direction[0], sign),
                                                state.seaLevel);
      return objective(op, start, direction, sign);
    };
    return 0.5 * (run(1.0) - run(-1.0));
  };
  const auto none = std::vector<std::vector<double>>{};
  const auto alongSea = [&] {
    auto run = [&](double sign) {
      return objective(op, setup.solver.MakeState(state.iceThickness,
                                                  Moved(state.seaLevel, deltaSea, sign)),
                       none, 0.0);
    };
    return 0.5 * (run(1.0) - run(-1.0));
  }();

  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(times, measurements, &trajectory, oceans);
  // the background-held reading of a thickness direction: the thickness
  // view with the baseline taken back out of the first node
  auto heldPairing = [&](const std::vector<std::vector<double>>& d) {
    return NodePairing(setup, kernels.thickness, d) - setup.Surface(kernels.baseline, d[0]);
  };
  const auto fdIce = withBackground(deltaIce);
  const auto fdHeld = held(deltaIce);
  const auto fdRising = held(rising);
  ASSERT_NE(fdIce, 0.0);
  std::cout << std::format(
      "[solved start, 1D] the two readings of a thickness direction differ by {:.2e} of "
      "it\n",
      Gap(fdHeld, fdIce));
  EXPECT_GT(Gap(fdHeld, fdIce), 1e-3) << "the first node's thickness matters";

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto label = std::string(method == AdjointMethod::Discrete ? "discrete" : "continuous");
    TangentLinearSolver tangent(evolution, {.method = method, .rtol = 1e-12});
    auto along = [&](const TangentPerturbation& p) {
      return Objective(setup, tangent.Run(times, p, &trajectory, oceans), measurements);
    };
    TangentPerturbation pIce, pRising, pSea, pBaseline;
    pIce.iceThickness = deltaIce;
    pRising.iceThickness = rising;
    pSea.initialSeaLevel = deltaSea;
    // the load the background-held reading adds at every node
    pBaseline.directLoad.assign(nodes, std::vector<double>(deltaIce[0].size()));
    for (auto& f : pBaseline.directLoad) {
      for (auto i = std::size_t{0}; i < f.size(); ++i) {
        f[i] = rhoI * (1.0 - state.oceanFunction[i]) * deltaIce[0][i];
      }
    }
    const auto tIce = along(pIce);
    const auto tRising = along(pRising);
    const auto tSea = along(pSea);
    const auto tHeld = tIce + along(pBaseline);
    std::cout << std::format(
        "[solved start, 1D, {}] ice with the background: {:.2e}; background held: {:.2e}; "
        "zero at the first node: {:.2e}; initial sea level: {:.2e}\n",
        label, Gap(tIce, fdIce), Gap(tHeld, fdHeld), Gap(tRising, fdRising),
        Gap(tSea, alongSea));
    EXPECT_NEAR(tIce, fdIce, 1e-7 * std::abs(fdIce)) << label;
    EXPECT_NEAR(tHeld, fdHeld, 1e-7 * std::abs(fdHeld)) << label;
    EXPECT_NEAR(tRising, fdRising, 1e-7 * std::abs(fdRising)) << label;
    EXPECT_NEAR(tSea, alongSea, 1e-7 * std::abs(alongSea)) << label;
    if (method == AdjointMethod::Discrete) {
      EXPECT_NEAR(NodePairing(setup, kernels.thickness, deltaIce), tIce, 1e-8 * std::abs(tIce));
      EXPECT_NEAR(heldPairing(deltaIce), tHeld, 1e-8 * std::abs(tHeld));
      EXPECT_NEAR(NodePairing(setup, kernels.thickness, rising), tRising,
                  1e-8 * std::abs(tRising));
      EXPECT_NEAR(setup.Surface(kernels.initialSeaLevel, deltaSea), tSea,
                  1e-8 * std::abs(tSea));
    }
  }

  // (a) the relaxation times
  TangentPerturbation pItau;
  pItau.itau = ditau;
  auto alongItau = [&](AdjointMethod method) {
    return Objective(setup,
                     TangentLinearSolver(evolution, {.method = method, .rtol = 1e-12})
                         .Run(times, pItau, &trajectory, oceans),
                     measurements);
  };
  const auto discrete = alongItau(AdjointMethod::Discrete);
  const auto continuous = alongItau(AdjointMethod::Continuous);
  const auto fd = Difference(
      [&](double eps) { return objective(MovedItau(setup, eps, ditau), state, none, 0.0); },
      0.02, discrete);
  const auto paired = VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], ditau[0]);
  std::cout << std::format(
      "[solved start, 1D] itau: ratio {:.3f}, Richardson gap {:.2e}; continuous run "
      "{:.2e} from the discrete; discrete kernel {:.2e}\n",
      fd.ratio, Gap(fd.richardson, discrete), Gap(continuous, discrete),
      Gap(paired, discrete));
  EXPECT_GT(fd.ratio, 3.0);
  EXPECT_LT(fd.ratio, 5.0);
  EXPECT_NEAR(fd.richardson, discrete, 1e-6 * std::abs(discrete));
  EXPECT_NEAR(paired, discrete, 1e-8 * std::abs(discrete));
  EXPECT_LT(Gap(continuous, discrete), 0.2);
}

TEST(MigratingDerivatives, TheContinuousSweepAboutAStartWithAnIceChange) {
  // The same solved start on a slowly relaxing mantle (1e22 Pa s), one
  // problem at 5 and 9 nodes over the 10 kyr (the history, the directions
  // and the measurements functions of time): the continuous sweep's
  // thickness view along a direction non-zero at the first node (both
  // readings), its initial sea level kernel and its relaxation-time
  // kernel (in its volume norm) tend to the discrete sweep's, which the
  // gate above holds exact, as the steps are halved.
  auto setup = Scene::Make(1e22);
  ViscoelasticOperator op(setup.rheology, kLmax);
  std::mt19937 rng(97);
  const auto fieldA = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto fieldB = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto deltaSea = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto g0 = setup.Smooth(rng, 0.4);
  const auto gMid = setup.Smooth(rng, 0.7);
  const auto gEnd = setup.Smooth(rng, 1.0);
  const auto& mesh = setup.material.Mesh();
  auto gapsAt = [&](int nodes) {
    const auto cap = MarineCap::Make(setup, nodes);
    const auto& times = cap.times;
    const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
    auto deltaIce = std::vector<std::vector<double>>{};
    for (const auto t : times) {
      const auto s = t / times.back();
      auto d = fieldA;
      for (auto i = std::size_t{0}; i < d.size(); ++i) {
        d[i] = (1.0 - s) * fieldA[i] + s * s * fieldB[i];
      }
      deltaIce.push_back(std::move(d));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].relativeSeaLevelDual = g0;
    measurements[NodeOf(times, 0.5 * times.back())].displacementDual = gMid;
    measurements[times.size() - 1].relativeSeaLevelDual = gEnd;
    Evolution evolution(op, setup.solver, state,
                        {.rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1});
    const auto run = evolution.RunMigrating(
        times, Field(times, Thinning(cap, times, 0.85, 0.35)), Evolution::Forcing{});
    const auto oceans = OceanFunctionsOf(run);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    auto kernels = std::map<AdjointMethod, Kernels>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      kernels[method] = AdjointSolver(evolution, {.method = method, .rtol = 1e-12})
                            .FirstOrderKernels(times, measurements, &trajectory, oceans);
    }
    const auto& d = kernels[AdjointMethod::Discrete];
    const auto& c = kernels[AdjointMethod::Continuous];
    auto withBackground = [&](const Kernels& k) {
      return NodePairing(setup, k.thickness, deltaIce);
    };
    auto held = [&](const Kernels& k) {
      return withBackground(k) - setup.Surface(k.baseline, deltaIce[0]);
    };
    auto e = c.itauKernel[0];
    for (auto i = std::size_t{0}; i < e.size(); ++i) e[i] -= d.itauKernel[0][i];
    return std::array<double, 4>{
        Gap(withBackground(c), withBackground(d)), Gap(held(c), held(d)),
        Gap(setup.Surface(c.initialSeaLevel, deltaSea), setup.Surface(d.initialSeaLevel, deltaSea)),
        std::sqrt(VolumePairing(mesh, e, e) /
                  VolumePairing(mesh, d.itauKernel[0], d.itauKernel[0]))};
  };
  const auto coarse = gapsAt(5);
  const auto fine = gapsAt(9);
  const auto names = std::array<const char*, 4>{
      "thickness with the background", "thickness, background held", "initial sea level",
      "itau kernel"};
  for (auto q = std::size_t{0}; q < names.size(); ++q) {
    std::cout << std::format("[solved start, 1D continuous] {}: {:.2e} {:.2e}\n", names[q],
                             coarse[q], fine[q]);
    EXPECT_LT(coarse[q], 0.1) << names[q];
    EXPECT_GT(coarse[q] / fine[q], 2.0) << names[q];
  }
}

// -- 3D, under lateral viscosity ------------------------------------------

namespace {

/// The prem8 body with the Maxwell mantle at degree 8, its solver and the
/// analytic test state: the 3D gates' scene.
struct Scene3D {
  static constexpr int kLmax3 = 8;
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;

  static Scene3D Make(double viscositySI) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax3}).value();
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
    ViscoelasticOperator reference(rheology, kLmax3);
    GRDSolver solver(SurfaceGrid(kLmax3), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    return Scene3D{std::move(model), std::move(material), std::move(rheology),
                   std::move(solver)};
  }

  [[nodiscard]] double Metre() const { return 1.0 / solver.SolverScales().Length(); }
  [[nodiscard]] double Kyr() const { return 1e3 * kYear / solver.SolverScales().Time(); }
  [[nodiscard]] double Sigma() const {
    return 1.0 / solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  }
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng, double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax3; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }
  [[nodiscard]] double Surface(std::span<const double> f, std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }
  /// A marine-grounded cap after pyslfp's (KernelScene's MarineCap) on
  /// this grid, of the given peak thickness in metres and angular radius
  /// in degrees: its ice and the uniform 3000 m ocean. On this degree-8
  /// grid the 5 km, 20-degree cap's band-limited thickness grounds no
  /// cell, so the gates here take a wider one.
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> MarineCapState(
      double peak, double radiusDegrees) const {
    const auto& grid = solver.Grid();
    const auto metre = Metre();
    auto ice = grid.Synthesise(grid.Expand(grid.FromFunction([&](double theta, double phi) {
      const auto theta0 = 30.0 * std::numbers::pi / 180.0;
      const auto phi0 = 275.0 * std::numbers::pi / 180.0;
      const auto radius = radiusDegrees * std::numbers::pi / 180.0;
      const auto c = std::cos(theta) * std::cos(theta0) +
                     std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
      const auto d = std::acos(std::clamp(c, -1.0, 1.0));
      if (d >= radius) return 0.0;
      const auto taper = std::cos(0.5 * std::numbers::pi * d / radius);
      return peak * metre * taper * taper;
    })));
    return {std::move(ice), std::vector<double>(grid.FieldSize(), 3000.0 * metre)};
  }
};

template <class Snapshot>
double Objective3D(const Scene3D& scene, const std::vector<Snapshot>& snapshots,
                   const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    for (const auto& [dual, field] :
         {std::pair{&m.relativeSeaLevelDual, &snap.relativeSeaLevelChange},
          std::pair{&m.displacementDual, &snap.displacement},
          std::pair{&m.potentialDual, &snap.potential},
          std::pair{&m.tangentialPotentialDual, &snap.tangentialPotential},
          std::pair{&m.toroidalPotentialDual, &snap.toroidalPotential}}) {
      if (!dual->empty() && !field->empty()) total += scene.Surface(*dual, *field);
    }
    for (auto j = 0; j < 3; ++j) total += m.angularVelocityDual[j] * snap.angularVelocity[j];
  }
  return total;
}

double NodePairing3D(const Scene3D& scene, const std::vector<std::vector<double>>& kernel,
                     const std::vector<std::vector<double>>& d) {
  auto out = 0.0;
  for (auto n = std::size_t{0}; n < d.size(); ++n) out += scene.Surface(kernel[n], d[n]);
  return out;
}

double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

double LateralDelta(double, double theta, double phi) {
  return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}

const auto kPattern = [](double theta, double phi) {
  return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
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

/// A cap state's thickness at the nodes: a fraction `start` of the cap at
/// the first node, melting linearly to `end` of it.
std::vector<std::vector<double>> Thinning3D(const std::vector<double>& ice0,
                                            std::span<const double> times, double start,
                                            double end) {
  auto out = std::vector<std::vector<double>>{};
  for (const auto t : times) {
    const auto s = t / times.back();
    auto ice = ice0;
    for (auto& v : ice) v *= start + (end - start) * s;
    out.push_back(std::move(ice));
  }
  return out;
}

}  // namespace

TEST(MigratingDerivatives, AStartWithAnIceChange3D) {
  // The 1D solved-start gate under lateral viscosity: a wider marine cap
  // at 85% at the first node, thinning to 55%, no other forcing, four
  // nodes over 10 kyr. The 3D tangent along a thickness direction non-zero at
  // the first node is the central difference of migrating runs with the
  // background's ice moved with the first node's, and with the background
  // held it is that plus the run along rho_i (1 - C) dI_0 at every node;
  // along a direction zero at the first node and along the initial sea
  // level the same, both methods; along a lateral log-viscosity direction
  // the discrete run is the derivative (second order, Richardson to 1e-6)
  // and the continuous within the commute level. The discrete 3D sweep
  // pairs with each to the tangent's number.
  auto scene = Scene3D::Make(1e21);
  const auto [ice0, sea0] = scene.MarineCapState(6000.0, 50.0);
  const auto state = scene.solver.MakeState(ice0, sea0);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(10.0 * n / 3.0 * scene.Kyr());
  const auto nodes = times.size();
  const auto ice = Thinning3D(ice0, times, 0.85, 0.55);
  const auto rhoI = scene.solver.IceDensity();
  std::mt19937 rng(101);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = scene.Smooth(rng, 0.4);
  measurements[1].displacementDual = scene.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    deltaIce.push_back(scene.Smooth(rng, 0.1 * scene.Metre()));
  }
  auto rising = deltaIce;
  rising[0].assign(rising[0].size(), 0.0);
  const auto deltaSea = scene.Smooth(rng, 0.1 * scene.Metre());
  const Evolution3D::Options options{.rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .max_nonlinear_iterations = 200,
                                     .record_stride = 1};
  Viscoelastic3D op(scene.rheology, Scene3D::kLmax3, LateralDelta);
  const auto& coupling = op.Coupling();
  Evolution3D evolution(op, scene.solver, state, options);
  const auto run = evolution.RunMigrating(times, Field(times, ice), Evolution3D::Forcing{});
  const auto oceans = OceanFunctionsOf(run);
  ASSERT_NE(oceans.front(), state.oceanFunction) << "the start moves no shoreline";
  ASSERT_NE(oceans.back(), oceans.front());
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);

  auto objective = [&](const Viscoelastic3D& at, const GRDSolver::State& start,
                       const std::vector<std::vector<double>>& direction, double eps) {
    EXPECT_EQ(start.oceanFunction, state.oceanFunction) << "the background flipped";
    Evolution3D trial(at, scene.solver, start, options);
    const auto snaps = trial.RunMigrating(times, Field(times, ice, direction, eps),
                                          Evolution3D::Forcing{});
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_TRUE(snaps[n].converged) << "node " << n;
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped at node " << n;
    }
    return Objective3D(scene, snaps, measurements);
  };
  const auto none = std::vector<std::vector<double>>{};
  auto held = [&](const std::vector<std::vector<double>>& direction) {
    return 0.5 * (objective(op, state, direction, 1.0) - objective(op, state, direction, -1.0));
  };
  auto withBackground = [&](const std::vector<std::vector<double>>& direction) {
    auto runAt = [&](double sign) {
      return objective(op,
                       scene.solver.MakeState(Moved(state.iceThickness, direction[0], sign),
                                              state.seaLevel),
                       direction, sign);
    };
    return 0.5 * (runAt(1.0) - runAt(-1.0));
  };
  const auto fdIce = withBackground(deltaIce);
  const auto fdHeld = held(deltaIce);
  const auto fdRising = held(rising);
  const auto fdSea = 0.5 * (objective(op,
                                      scene.solver.MakeState(state.iceThickness,
                                                             Moved(state.seaLevel, deltaSea, 1.0)),
                                      none, 0.0) -
                            objective(op,
                                      scene.solver.MakeState(state.iceThickness,
                                                             Moved(state.seaLevel, deltaSea, -1.0)),
                                      none, 0.0));
  EXPECT_GT(Gap(fdHeld, fdIce), 1e-3) << "the first node's thickness matters";

  const auto kernels = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                    .rtol = 1e-12,
                                                    .coupled_rtol = 1e-12})
                           .FirstOrderKernels(times, measurements, &trajectory, oceans);
  const auto pattern = OnCoupling(coupling, kPattern);
  auto pairings = [&](const Kernels3D& k) {
    return std::array<double, 5>{
        NodePairing3D(scene, k.ice.thickness, deltaIce),
        NodePairing3D(scene, k.ice.thickness, deltaIce) -
            scene.Surface(k.ice.baseline, deltaIce[0]),
        NodePairing3D(scene, k.ice.thickness, rising),
        scene.Surface(k.ice.initialSeaLevel, deltaSea),
        VolumePairing(coupling, k.logViscosityKernel3D[0], pattern)};
  };
  const auto discreteKernels = pairings(kernels);

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto label = std::string(method == AdjointMethod::Discrete ? "discrete" : "continuous");
    TangentLinearSolver3D tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .coupled_rtol = 1e-12});
    auto along = [&](const TangentPerturbation3D& p) {
      return Objective3D(scene, tangent.Run(times, p, &trajectory, oceans), measurements);
    };
    TangentPerturbation3D pIce, pRising, pSea, pBaseline;
    pIce.iceThickness = deltaIce;
    pRising.iceThickness = rising;
    pSea.initialSeaLevel = deltaSea;
    pBaseline.directLoad.assign(nodes, std::vector<double>(deltaIce[0].size()));
    for (auto& f : pBaseline.directLoad) {
      for (auto i = std::size_t{0}; i < f.size(); ++i) {
        f[i] = rhoI * (1.0 - state.oceanFunction[i]) * deltaIce[0][i];
      }
    }
    const auto tIce = along(pIce);
    const auto tHeld = tIce + along(pBaseline);
    const auto tRising = along(pRising);
    const auto tSea = along(pSea);
    std::cout << std::format(
        "[solved start, 3D, {}] ice with the background: {:.2e}; background held: {:.2e}; "
        "zero at the first node: {:.2e}; initial sea level: {:.2e}\n",
        label, Gap(tIce, fdIce), Gap(tHeld, fdHeld), Gap(tRising, fdRising),
        Gap(tSea, fdSea));
    EXPECT_NEAR(tIce, fdIce, 1e-7 * std::abs(fdIce)) << label;
    EXPECT_NEAR(tHeld, fdHeld, 1e-7 * std::abs(fdHeld)) << label;
    EXPECT_NEAR(tRising, fdRising, 1e-7 * std::abs(fdRising)) << label;
    EXPECT_NEAR(tSea, fdSea, 1e-7 * std::abs(fdSea)) << label;
    if (method == AdjointMethod::Discrete) {
      for (const auto& [q, want] : {std::pair{0, tIce}, std::pair{1, tHeld},
                                    std::pair{2, tRising}, std::pair{3, tSea}}) {
        EXPECT_NEAR(discreteKernels[q], want, 1e-8 * std::abs(want)) << "kernel " << q;
      }
    }
  }

  // (a) the log viscosity
  TangentPerturbation3D pViscosity;
  pViscosity.logViscosity = pattern;
  auto alongViscosity = [&](AdjointMethod method) {
    return Objective3D(scene,
                       TangentLinearSolver3D(evolution, {.method = method,
                                                         .rtol = 1e-12,
                                                         .coupled_rtol = 1e-12})
                           .Run(times, pViscosity, &trajectory, oceans),
                       measurements);
  };
  const auto discrete = alongViscosity(AdjointMethod::Discrete);
  const auto continuous = alongViscosity(AdjointMethod::Continuous);
  const auto fd = Difference(
      [&](double eps) {
        Viscoelastic3D moved(scene.rheology, Scene3D::kLmax3,
                             [&](double r, double theta, double phi) {
                               return LateralDelta(r, theta, phi) + eps * kPattern(theta, phi);
                             });
        return objective(moved, state, none, 0.0);
      },
      0.02, discrete);
  std::cout << std::format(
      "[solved start, 3D] log viscosity: ratio {:.3f}, Richardson gap {:.2e}; continuous "
      "run {:.2e} from the discrete; discrete kernel {:.2e}\n",
      fd.ratio, Gap(fd.richardson, discrete), Gap(continuous, discrete),
      Gap(discreteKernels[4], discrete));
  EXPECT_GT(fd.ratio, 3.0);
  EXPECT_LT(fd.ratio, 5.0);
  EXPECT_NEAR(fd.richardson, discrete, 1e-6 * std::abs(discrete));
  EXPECT_NEAR(discreteKernels[4], discrete, 1e-8 * std::abs(discrete));
  EXPECT_LT(Gap(continuous, discrete), 0.2);
}

TEST(MigratingDerivatives, TheContinuousSweepAboutAStartWithAnIceChange3D) {
  // The 3D solved start on a slowly relaxing mantle (1e22 Pa s), one
  // problem at 4 and 7 nodes over the 10 kyr: the continuous 3D sweep's
  // thickness view along a direction non-zero at the first node (both
  // readings), its initial sea level kernel and its log-viscosity kernel
  // (in its volume norm) tend to the discrete sweep's, which the gate
  // above holds exact, as the steps are halved.
  auto scene = Scene3D::Make(1e22);
  const auto [ice0, sea0] = scene.MarineCapState(6000.0, 50.0);
  const auto state = scene.solver.MakeState(ice0, sea0);
  std::mt19937 rng(103);
  const auto fieldA = scene.Smooth(rng, 0.1 * scene.Metre());
  const auto fieldB = scene.Smooth(rng, 0.1 * scene.Metre());
  const auto deltaSea = scene.Smooth(rng, 0.1 * scene.Metre());
  const auto g0 = scene.Smooth(rng, 0.4);
  const auto gEnd = scene.Smooth(rng, 1.0);
  Viscoelastic3D op(scene.rheology, Scene3D::kLmax3, LateralDelta);
  const auto& coupling = op.Coupling();
  auto gapsAt = [&](int nodes) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(10.0 * n / (nodes - 1) * scene.Kyr());
    auto deltaIce = std::vector<std::vector<double>>{};
    for (const auto t : times) {
      const auto s = t / times.back();
      auto d = fieldA;
      for (auto i = std::size_t{0}; i < d.size(); ++i) {
        d[i] = (1.0 - s) * fieldA[i] + s * s * fieldB[i];
      }
      deltaIce.push_back(std::move(d));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].relativeSeaLevelDual = g0;
    measurements[times.size() - 1].relativeSeaLevelDual = gEnd;
    Evolution3D evolution(op, scene.solver, state,
                          {.rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .max_nonlinear_iterations = 200,
                           .record_stride = 1});
    const auto run = evolution.RunMigrating(
        times, Field(times, Thinning3D(ice0, times, 0.85, 0.55)), Evolution3D::Forcing{});
    const auto oceans = OceanFunctionsOf(run);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
    auto kernels = std::map<AdjointMethod, Kernels3D>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      kernels[method] = AdjointSolver3D(evolution, {.method = method,
                                                    .rtol = 1e-12,
                                                    .coupled_rtol = 1e-12})
                            .FirstOrderKernels(times, measurements, &trajectory, oceans);
    }
    const auto& d = kernels[AdjointMethod::Discrete];
    const auto& c = kernels[AdjointMethod::Continuous];
    auto withBackground = [&](const Kernels3D& k) {
      return NodePairing3D(scene, k.ice.thickness, deltaIce);
    };
    auto held = [&](const Kernels3D& k) {
      return withBackground(k) - scene.Surface(k.ice.baseline, deltaIce[0]);
    };
    auto e = c.logViscosityKernel3D[0];
    for (auto i = std::size_t{0}; i < e.size(); ++i) e[i] -= d.logViscosityKernel3D[0][i];
    return std::array<double, 4>{
        Gap(withBackground(c), withBackground(d)), Gap(held(c), held(d)),
        Gap(scene.Surface(c.ice.initialSeaLevel, deltaSea),
            scene.Surface(d.ice.initialSeaLevel, deltaSea)),
        std::sqrt(VolumePairing(coupling, e, e) /
                  VolumePairing(coupling, d.logViscosityKernel3D[0],
                                d.logViscosityKernel3D[0]))};
  };
  const auto coarse = gapsAt(4);
  const auto fine = gapsAt(7);
  const auto names = std::array<const char*, 4>{
      "thickness with the background", "thickness, background held", "initial sea level",
      "log-viscosity kernel"};
  for (auto q = std::size_t{0}; q < names.size(); ++q) {
    std::cout << std::format("[solved start, 3D continuous] {}: {:.2e} {:.2e}\n", names[q],
                             coarse[q], fine[q]);
    EXPECT_LT(coarse[q], 0.2) << names[q];
    EXPECT_GT(coarse[q] / fine[q], 2.0) << names[q];
  }
}

// -- a forced migrating trajectory ----------------------------------------

namespace {

/// The forcings beside the ice of a forced migrating run, per node: a load
/// inside the water budget, one outside it, the height of a layer on the
/// solid surface and a point source (the base run's), and a direction in
/// each.
struct Beside {
  std::vector<std::vector<double>> inside, outside, layer;
  std::vector<std::vector<PointSource>> sources;
};

/// The forcing of a run (Evolution or Evolution3D) under base + eps d.
template <class Forcing>
Forcing ForcingOf(const std::vector<double>& times, const Beside& base, const Beside* d,
                  double eps) {
  auto field = [&](const std::vector<std::vector<double>>& b,
                   const std::vector<std::vector<double>>* x) {
    return Field(times, b, x != nullptr ? *x : std::vector<std::vector<double>>{}, eps);
  };
  return Forcing{
      .directLoad = field(base.inside, d != nullptr ? &d->inside : nullptr),
      .directLoadOutsideWaterBudget = field(base.outside, d != nullptr ? &d->outside : nullptr),
      .solidSurfaceChange = field(base.layer, d != nullptr ? &d->layer : nullptr),
      .sources = [times, base, d = d != nullptr ? *d : Beside{}, eps](double t) {
        const auto n = NodeOf(times, t);
        auto out = base.sources[n];
        if (!d.sources.empty()) {
          auto extra = d.sources[n][0];
          for (auto* c : {&extra.moment.rr, &extra.moment.tt, &extra.moment.pp,
                          &extra.moment.rt, &extra.moment.rp, &extra.moment.tp}) {
            *c *= eps;
          }
          out.push_back(extra);
        }
        return out;
      }};
}

/// The base forcing beside the ice on a grid, as functions of time: the
/// loads ramped up, the layer held (it moves the start's shorelines and no
/// more), the source ramped.
Beside BaseForcing(std::span<const double> times, const std::vector<double>& inside,
                   const std::vector<double>& outside, const std::vector<double>& layer,
                   const PointSource& source) {
  Beside out;
  for (const auto t : times) {
    const auto shape = 0.5 + t / times.back();
    auto scaled = [&](std::vector<double> f) {
      for (auto& v : f) v *= shape;
      return f;
    };
    out.inside.push_back(scaled(inside));
    out.outside.push_back(scaled(outside));
    out.layer.push_back(layer);
    auto at = source;
    for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp, &at.moment.rt,
                    &at.moment.rp, &at.moment.tp}) {
      *c *= shape;
    }
    out.sources.push_back({at});
  }
  return out;
}

/// A point source 400 km deep, of a moment of the given scale (N m).
PointSource SourceAt(const GRDSolver& solver, double surface, double scale) {
  const auto& sle = solver.SolverScales();
  const auto unit = scale / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  return PointSource{
      .at = {.radius = surface - 400e3 / sle.Length(), .colatitude = 0.6, .longitude = 4.6},
      .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                 .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}};
}

/// A direction in kappa and mu at the operator's memory points on the
/// solid, at a tenth of the moduli (kappa at half the shear's share).
PointMaterialDirection PointDirection(const ViscoelasticOperator& op, const Material& material,
                                      const std::function<double(double)>& profile) {
  const auto& mesh = material.Mesh();
  const auto& points = op.Points();
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  PointMaterialDirection d;
  d.kappa.assign(muU.size(), 0.0);
  d.mu.assign(muU.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto value = 0.1 * profile(points.RadiusAt(mesh, e, p));
      d.kappa[i] = 0.5 * value * kappa0[i];
      d.mu[i] = value * muU[i];
    }
  }
  return d;
}

/// The operator with its moduli at the points moved by eps along d.
ViscoelasticOperator MovedModuli(const Scene& setup, const ViscoelasticOperator& op,
                                 double eps, const PointMaterialDirection& d) {
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  ViscoelasticOperator::Options::PointModuli moduli;
  moduli.shear.assign(muU.size(), 0.0);
  moduli.bulk.assign(muU.size(), 0.0);
  for (auto i = std::size_t{0}; i < muU.size(); ++i) {
    if (muU[i] > 0.0) moduli.shear[i] = eps * d.mu[i] / muU[i];
    if (kappa0[i] > 0.0) moduli.bulk[i] = eps * d.kappa[i] / kappa0[i];
  }
  return ViscoelasticOperator(setup.rheology, kLmax, {.moduli = moduli});
}

/// A 1D kernel object (kernels or their derivatives) paired with a
/// direction in every argument a forced migrating run takes.
template <class K>
double PairingOf(const Scene& setup, const ViscoelasticOperator& op, const K& k,
                 const TangentPerturbation& d) {
  const auto& mesh = setup.material.Mesh();
  auto total = NodePairing(setup, k.thickness, d.iceThickness) +
               NodePairing(setup, k.directLoadKernels, d.directLoad) +
               NodePairing(setup, k.directLoadOutsideWaterBudgetKernels,
                           d.directLoadOutsideWaterBudget) +
               NodePairing(setup, k.solidSurfaceChangeKernels, d.solidSurfaceChange);
  if (!d.initialSeaLevel.empty()) total += setup.Surface(k.initialSeaLevel, d.initialSeaLevel);
  if (!d.itau.empty()) total += VolumePairing(mesh, k.itauKernel[0], d.itau[0]);
  if (!d.material.kappa.empty()) {
    total += VolumePairing(op.Points(), mesh, k.kappaKernel, d.material.kappa);
  }
  if (!d.material.mu.empty()) {
    total += VolumePairing(op.Points(), mesh, k.muKernel, d.material.mu);
  }
  return total;
}

}  // namespace

TEST(MigratingDerivatives, TheViscosityAndTheModuliAboutAForcedRun) {
  // About a migrating run forced beside the ice (a load inside the water
  // budget, one outside it, a layer on the solid surface that moves the
  // start's shorelines, a point source) through a melting marine cap: the
  // discrete tangent along the relaxation times and along a direction in
  // the moduli is the derivative of the migrating run (central
  // differences flipping no cell — steps of up to 0.04% of the relaxation
  // times and of the moduli, below which the cells nearest flotation stay
  // put, and so near the solves' noise that the residuals' ratio is not
  // read — Richardson to 1e-6 and 1e-5), the discrete sweep's kernels in
  // both pair with it to 1e-8 and the continuous run sits within the
  // commute level; and the discrete
  // second-order sweep, along directions u and w in every argument of the
  // run (the loads, the layer, the thickness, the initial sea level, the
  // relaxation times, the moduli), carries the first-order kernels, gives
  // <H u, w> = <H w, u> and equals the second-order run's <g, F''[u, w]>.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto cap = MarineCap::Make(setup, 5);
  const auto& times = cap.times;
  const auto nodes = times.size();
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  std::mt19937 rng(107);
  const auto surface = setup.material.Radius() * setup.model.ModelScales().Length() /
                       setup.solver.SolverScales().Length();
  const auto base =
      BaseForcing(times, setup.Smooth(rng, 2e4 * setup.Sigma()),
                  setup.Smooth(rng, 1e4 * setup.Sigma()),
                  setup.Smooth(rng, 400.0 * setup.Metre()),
                  SourceAt(setup.solver, surface, 1e21));
  auto ice = std::vector<std::vector<double>>{};
  for (const auto t : times) ice.push_back(cap.Thickness(t));
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 0.4);
  measurements[1].displacementDual = setup.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[nodes - 1].tangentialPotentialDual = setup.Smooth(rng, 0.6);
  const Evolution::Options options{
      .rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1};
  Evolution evolution(op, setup.solver, state, options);
  const auto run = evolution.RunMigrating(times, Field(times, ice),
                                          ForcingOf<Evolution::Forcing>(times, base, nullptr, 0.0));
  const auto oceans = OceanFunctionsOf(run);
  ASSERT_NE(oceans.front(), state.oceanFunction);
  ASSERT_NE(oceans.back(), oceans.front());
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);
  auto objective = [&](const ViscoelasticOperator& at) {
    Evolution trial(at, setup.solver, state, options);
    const auto snaps = trial.RunMigrating(
        times, Field(times, ice), ForcingOf<Evolution::Forcing>(times, base, nullptr, 0.0));
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_TRUE(snaps[n].converged) << "node " << n;
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped at node " << n;
    }
    return Objective(setup, snaps, measurements);
  };
  const auto& mesh = setup.material.Mesh();
  const auto ditau = setup.DeltaItau(0.1);
  const auto material = PointDirection(
      op, setup.material, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  const auto kernels = AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                                 .rtol = 1e-12,
                                                 .material_kernels = true})
                           .FirstOrderKernels(times, measurements, &trajectory, oceans);
  // (the moduli's quotient keeps fewer digits: their direction moves the
  // elastic response everywhere, so the step that flips no cell leaves
  // the solves' noise at parts in 1e6 of it)
  for (const auto& [name, eps, tolerance, direction, moved, paired] :
       {std::tuple{"itau", 0.004, 1e-6,
                   TangentPerturbation{.itau = ditau},
                   std::function<ViscoelasticOperator(double)>(
                       [&](double e) { return MovedItau(setup, e, ditau); }),
                   VolumePairing(mesh, kernels.itauKernel[0], ditau[0])},
        std::tuple{"moduli", 0.004, 1e-5, TangentPerturbation{.material = material},
                   std::function<ViscoelasticOperator(double)>(
                       [&](double e) { return MovedModuli(setup, op, e, material); }),
                   VolumePairing(op.Points(), mesh, kernels.kappaKernel, material.kappa) +
                       VolumePairing(op.Points(), mesh, kernels.muKernel, material.mu)}}) {
    auto along = [&](AdjointMethod method) {
      return Objective(setup,
                       TangentLinearSolver(evolution, {.method = method, .rtol = 1e-12})
                           .Run(times, direction, &trajectory, oceans),
                       measurements);
    };
    const auto discrete = along(AdjointMethod::Discrete);
    const auto continuous = along(AdjointMethod::Continuous);
    const auto fd =
        Difference([&](double e) { return objective(moved(e)); }, eps, discrete);
    std::cout << std::format(
        "[forced, 1D] {}: ratio {:.3f}, Richardson gap {:.2e}; discrete kernel {:.2e}; "
        "continuous run {:.2e}\n",
        name, fd.ratio, Gap(fd.richardson, discrete), Gap(paired, discrete),
        Gap(continuous, discrete));
    EXPECT_NEAR(fd.richardson, discrete, tolerance * std::abs(discrete)) << name;
    EXPECT_NEAR(paired, discrete, 1e-8 * std::abs(discrete)) << name;
    // the commute level of the stiff regime (2.5 kyr steps against the
    // 450-year relaxation time; cf. Adjoint.ContinuousAgreesWithDiscrete)
    EXPECT_LT(Gap(continuous, discrete), 0.5) << name;
  }

  // second order
  std::mt19937 local(109);
  auto perNode = [&](double amplitude, bool zeroFirst) {
    auto out = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      out.push_back(setup.Smooth(local, amplitude));
      if (zeroFirst && n == 0) out.back().assign(out.back().size(), 0.0);
    }
    return out;
  };
  TangentPerturbation u, w;
  u.directLoad = perNode(10.0 * setup.Sigma(), false);
  u.iceThickness = perNode(0.1 * setup.Metre(), true);
  u.itau = ditau;
  u.material = material;
  w.directLoadOutsideWaterBudget = perNode(10.0 * setup.Sigma(), false);
  w.solidSurfaceChange = perNode(0.02 * setup.Metre(), false);
  w.iceThickness = perNode(0.1 * setup.Metre(), true);
  w.initialSeaLevel = setup.Smooth(local, 0.1 * setup.Metre());
  w.itau = setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
  w.material = PointDirection(op, setup.material, [](double r) { return 1.0 - 0.5 * r; });
  TangentLinearSolver tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
  const auto runU = tangent.Run(times, u, &trajectory, oceans);
  const auto runW = tangent.Run(times, w, &trajectory, oceans);
  AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                    .rtol = 1e-12,
                                    .material_kernels = true});
  const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.itau, trajectory, oceans, u.material);
  const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
      times, measurements, runW, w.itau, trajectory, oceans, w.material);
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    EXPECT_LE(MaxDiff(kernelsU.thickness[n], kernels.thickness[n]),
              1e-12 * MaxAbs(kernels.thickness[n]))
        << "node " << n;
  }
  EXPECT_LE(MaxDiff(kernelsU.muKernel, kernels.muKernel), 1e-12 * MaxAbs(kernels.muKernel));
  const auto uw = PairingOf(setup, op, alongU, w);
  const auto wu = PairingOf(setup, op, alongW, u);
  SecondOrderDirection direction;
  direction.itauU = u.itau;
  direction.itauV = w.itau;
  direction.materialU = u.material;
  direction.materialV = w.material;
  const auto forward = Objective(
      setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory, oceans),
      measurements);
  std::cout << std::format(
      "[forced, 1D] Hessian: symmetry gap {:.2e}, against the second-order run {:.2e}\n",
      Gap(wu, uw), Gap(uw, forward));
  ASSERT_NE(forward, 0.0);
  EXPECT_NEAR(uw, wu, 1e-9 * std::abs(uw));
  EXPECT_NEAR(uw, forward, 1e-9 * std::abs(forward));
}

namespace {

/// Surface fields per node: the L2 distance of x from y over the nodes,
/// relative to y's.
double NodeDistance(const std::function<double(std::span<const double>,
                                               std::span<const double>)>& surface,
                    const std::vector<std::vector<double>>& x,
                    const std::vector<std::vector<double>>& y) {
  auto num = 0.0, den = 0.0;
  for (auto n = std::size_t{0}; n < y.size(); ++n) {
    auto e = x[n];
    for (auto i = std::size_t{0}; i < e.size(); ++i) e[i] -= y[n][i];
    num += surface(e, e);
    den += surface(y[n], y[n]);
  }
  return std::sqrt(num / den);
}

}  // namespace

TEST(MigratingDerivatives, TheContinuousSweepsAboutAForcedRun) {
  // About the forced migrating run on a slowly relaxing mantle (1e22 Pa
  // s), one problem at 5 and 9 nodes over the 10 kyr: the continuous
  // sweep's kernels of every forcing (the loads inside and outside the
  // water budget, the layer, the thickness, the initial sea level, the
  // source's moment) and its kernels in the relaxation times and the
  // moduli, and the continuous second-order sweep's derivatives of those
  // kernels along a direction in the thickness, the relaxation times and
  // the moduli, tend to the discrete sweeps' (exact, the gate above) as
  // the steps are halved — the surface kernels in their L2 norm over the
  // nodes, the others in their volume norms, the moment kernels through
  // the source's own moment.
  auto setup = Scene::Make(1e22);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto& mesh = setup.material.Mesh();
  std::mt19937 rng(113);
  const auto surface = setup.material.Radius() * setup.model.ModelScales().Length() /
                       setup.solver.SolverScales().Length();
  const auto inside = setup.Smooth(rng, 2e4 * setup.Sigma());
  const auto outside = setup.Smooth(rng, 1e4 * setup.Sigma());
  const auto layer = setup.Smooth(rng, 400.0 * setup.Metre());
  const auto source = SourceAt(setup.solver, surface, 1e21);
  const auto iceField = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto g0 = setup.Smooth(rng, 0.4);
  const auto gMid = setup.Smooth(rng, 0.7);
  const auto gEnd = setup.Smooth(rng, 1.0);
  const auto gTangential = setup.Smooth(rng, 0.6);
  const auto ditau = setup.DeltaItau(0.1);
  const auto material = PointDirection(
      op, setup.material, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  auto surfaceL2 = [&](std::span<const double> f, std::span<const double> g) {
    return setup.Surface(f, g);
  };
  enum Quantity {
    kInside, kOutside, kLayer, kThickness, kSea, kMoment, kItau, kModuli, kHessian, kCount
  };
  const auto names = std::array<const char*, kCount>{
      "inside", "outside", "layer", "thickness", "initial sea level", "moment", "itau",
      "moduli", "Hessian"};
  auto gapsAt = [&](int nodes) {
    const auto cap = MarineCap::Make(setup, nodes);
    const auto& times = cap.times;
    const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
    const auto base = BaseForcing(times, inside, outside, layer, source);
    auto ice = std::vector<std::vector<double>>{};
    for (const auto t : times) ice.push_back(cap.Thickness(t));
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].relativeSeaLevelDual = g0;
    measurements[NodeOf(times, 0.5 * times.back())].displacementDual = gMid;
    measurements[times.size() - 1].relativeSeaLevelDual = gEnd;
    measurements[times.size() - 1].tangentialPotentialDual = gTangential;
    Evolution evolution(op, setup.solver, state,
                        {.rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1});
    const auto run = evolution.RunMigrating(
        times, Field(times, ice), ForcingOf<Evolution::Forcing>(times, base, nullptr, 0.0));
    const auto oceans = OceanFunctionsOf(run);
    EXPECT_NE(oceans.front(), state.oceanFunction);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    TangentPerturbation u;
    for (const auto t : times) {
      auto d = iceField;
      for (auto& v : d) v *= t / times.back();
      u.iceThickness.push_back(std::move(d));
    }
    u.itau = ditau;
    u.material = material;
    auto kernels = std::map<AdjointMethod, Kernels>{};
    auto derivatives = std::map<AdjointMethod, KernelDerivatives>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .material_kernels = true,
                                        .moment_points = {source.at}});
      kernels[method] = adjoint.FirstOrderKernels(times, measurements, &trajectory, oceans);
      const auto runU =
          TangentLinearSolver(evolution, {.method = method, .rtol = 1e-12, .record = true})
              .Run(times, u, &trajectory, oceans);
      derivatives[method] = adjoint
                                .SecondOrderKernels(times, measurements, runU, u.itau,
                                                    trajectory, oceans, u.material)
                                .second;
    }
    const auto& d = kernels[AdjointMethod::Discrete];
    const auto& c = kernels[AdjointMethod::Continuous];
    EXPECT_EQ(c.directLoadOutsideWaterBudgetKernels.size(), times.size());
    EXPECT_EQ(c.solidSurfaceChangeKernels.size(), times.size());
    EXPECT_EQ(c.momentKernels.size(), times.size());
    auto volume = [&](const std::vector<double>& x, const std::vector<double>& y) {
      auto e = x;
      for (auto i = std::size_t{0}; i < e.size(); ++i) e[i] -= y[i];
      return std::sqrt(VolumePairing(mesh, e, e) / VolumePairing(mesh, y, y));
    };
    auto atPoints = [&](const auto& x, const auto& y) {
      auto ek = x.kappaKernel, em = x.muKernel;
      for (auto i = std::size_t{0}; i < ek.size(); ++i) {
        ek[i] -= y.kappaKernel[i];
        em[i] -= y.muKernel[i];
      }
      const auto& p = op.Points();
      return std::sqrt(
          (VolumePairing(p, mesh, ek, ek) + VolumePairing(p, mesh, em, em)) /
          (VolumePairing(p, mesh, y.kappaKernel, y.kappaKernel) +
           VolumePairing(p, mesh, y.muKernel, y.muKernel)));
    };
    auto moment = [&](const Kernels& k) {
      auto out = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        out += k.momentKernels[n][0].Contract(base.sources[n][0].moment);
      }
      return out;
    };
    const auto& dd = derivatives[AdjointMethod::Discrete];
    const auto& dc = derivatives[AdjointMethod::Continuous];
    auto gaps = std::array<double, kCount>{};
    gaps[kInside] = NodeDistance(surfaceL2, c.directLoadKernels, d.directLoadKernels);
    gaps[kOutside] = NodeDistance(surfaceL2, c.directLoadOutsideWaterBudgetKernels,
                                  d.directLoadOutsideWaterBudgetKernels);
    gaps[kLayer] =
        NodeDistance(surfaceL2, c.solidSurfaceChangeKernels, d.solidSurfaceChangeKernels);
    gaps[kThickness] = NodeDistance(surfaceL2, c.thickness, d.thickness);
    gaps[kSea] = NodeDistance(surfaceL2, {c.initialSeaLevel}, {d.initialSeaLevel});
    gaps[kMoment] = Gap(moment(c), moment(d));
    gaps[kItau] = volume(c.itauKernel[0], d.itauKernel[0]);
    gaps[kModuli] = atPoints(c, d);
    gaps[kHessian] = std::hypot(volume(dc.itauKernel[0], dd.itauKernel[0]), atPoints(dc, dd));
    return gaps;
  };
  const auto coarse = gapsAt(5);
  const auto fine = gapsAt(9);
  for (auto q = 0; q < kCount; ++q) {
    std::cout << std::format("[forced, 1D continuous] {}: {:.2e} {:.2e}\n", names[q],
                             coarse[q], fine[q]);
    EXPECT_LT(coarse[q], 0.2) << names[q];
    EXPECT_GT(coarse[q] / fine[q], 2.0) << names[q];
  }
}

namespace {

/// A lateral direction in kappa and mu over the coupling's grid values: a
/// radial profile times an angular pattern times the point moduli (kappa
/// at half the shear's share), at a tenth of the moduli.
MaterialDirection3D LateralDirection(const Viscoelastic3D& op,
                                     const std::function<double(double)>& radial,
                                     const std::function<double(double, double)>& angular) {
  const auto& coupling = op.Coupling();
  const auto np = static_cast<std::size_t>(coupling.Points().count);
  const auto muU = op.Reference().UnrelaxedShearAtPoints();
  const auto kappa0 = op.Reference().BulkModulusAtPoints();
  MaterialDirection3D d;
  d.kappa.assign(coupling.GridValueCount(), 0.0);
  d.mu.assign(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto point = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
                       coupling.NodeInElementOf(rn);
    const auto value = 0.1 * radial(coupling.RadiusOfNode(rn));
    const auto pattern = coupling.GridOf(rn).FromFunction(angular);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < pattern.size(); ++x) {
      d.kappa[at + x] = 0.5 * value * pattern[x] * kappa0[point];
      d.mu[at + x] = value * pattern[x] * muU[point];
    }
  }
  return d;
}

/// The 3D operator with its moduli moved by eps times LateralDirection's
/// relative pattern, confined to the coupling's elements (the moved
/// operator carries the same nodes).
std::unique_ptr<Viscoelastic3D> MovedModuli3D(const Scene3D& scene, const Viscoelastic3D& base,
                                              double eps,
                                              const std::function<double(double)>& radial,
                                              const std::function<double(double, double)>& angular) {
  const auto& coupling = base.Coupling();
  const auto& mesh = scene.material.Mesh();
  auto inner = mesh.Right(mesh.ElementCount() - 1), outer = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto e = coupling.ElementOfNode(rn);
    inner = std::min(inner, mesh.Left(e));
    outer = std::max(outer, mesh.Right(e));
  }
  auto relative = [=](double r, double theta, double phi) {
    return r > inner && r < outer ? 0.1 * eps * radial(r) * angular(theta, phi) : 0.0;
  };
  auto moved = std::make_unique<Viscoelastic3D>(
      scene.rheology, Scene3D::kLmax3, LateralDelta,
      Viscoelastic3D::Options{.moduli = {.shear = relative,
                                         .bulk = [=](double r, double theta, double phi) {
                                           return 0.5 * relative(r, theta, phi);
                                         }}});
  EXPECT_EQ(moved->Coupling().NodeCount(), coupling.NodeCount());
  return moved;
}

/// A 3D kernel object (kernels or their derivatives) paired with a
/// direction in every argument a forced migrating run takes.
template <class K>
double PairingOf3D(const Scene3D& scene, const GridCoupling& coupling, const K& k,
                   const TangentPerturbation3D& d) {
  auto total = NodePairing3D(scene, k.ice.thickness, d.iceThickness) +
               NodePairing3D(scene, k.ice.directLoadKernels, d.directLoad) +
               NodePairing3D(scene, k.ice.directLoadOutsideWaterBudgetKernels,
                             d.directLoadOutsideWaterBudget) +
               NodePairing3D(scene, k.ice.solidSurfaceChangeKernels, d.solidSurfaceChange);
  if (!d.initialSeaLevel.empty()) {
    total += scene.Surface(k.ice.initialSeaLevel, d.initialSeaLevel);
  }
  for (auto b = std::size_t{0}; b < d.itau.size(); ++b) {
    total += VolumePairing(coupling, k.itauKernel3D[b], d.itau[b]);
  }
  if (!d.material.kappa.empty()) {
    total += VolumePairing(coupling, k.kappaKernel3D, d.material.kappa);
  }
  if (!d.material.mu.empty()) total += VolumePairing(coupling, k.muKernel3D, d.material.mu);
  return total;
}

const auto kAngular = [](double theta, double phi) {
  return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
};
const auto kRadial = [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); };

}  // namespace

TEST(MigratingDerivatives, TheViscosityAndTheModuliAboutAForcedRun3D) {
  // The 1D forced gate under lateral viscosity, four nodes over 10 kyr, a
  // wide marine cap melting from the background's to 55%: the discrete 3D
  // tangent along a lateral log-viscosity direction and a lateral
  // direction in the moduli is the derivative of the migrating 3D run
  // (central differences flipping no cell at second order, Richardson to
  // 1e-6),
  // the discrete 3D sweep's kernels in both pair with it to 1e-8, the
  // continuous run sits within the commute level; and the discrete 3D
  // second-order sweep along directions u and w in every argument of the
  // run carries the first-order kernels, gives <H u, w> = <H w, u> and
  // equals the second-order run's <g, F''[u, w]> (its log
  // parameterisation's itau_uv term apart).
  auto scene = Scene3D::Make(1e21);
  const auto [ice0, sea0] = scene.MarineCapState(6000.0, 50.0);
  const auto state = scene.solver.MakeState(ice0, sea0);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(10.0 * n / 3.0 * scene.Kyr());
  const auto nodes = times.size();
  std::mt19937 rng(127);
  const auto base = BaseForcing(times, scene.Smooth(rng, 2e4 * scene.Sigma()),
                                scene.Smooth(rng, 1e4 * scene.Sigma()),
                                scene.Smooth(rng, 400.0 * scene.Metre()),
                                SourceAt(scene.solver, scene.solver.Radius(), 1e21));
  const auto ice = Thinning3D(ice0, times, 1.0, 0.55);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = scene.Smooth(rng, 0.4);
  measurements[1].displacementDual = scene.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  measurements[nodes - 1].toroidalPotentialDual = scene.Smooth(rng, 0.5);
  const Evolution3D::Options options{.rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .max_nonlinear_iterations = 200,
                                     .record_stride = 1};
  Viscoelastic3D op(scene.rheology, Scene3D::kLmax3, LateralDelta);
  const auto& coupling = op.Coupling();
  Evolution3D evolution(op, scene.solver, state, options);
  const auto forcing = ForcingOf<Evolution3D::Forcing>(times, base, nullptr, 0.0);
  const auto run = evolution.RunMigrating(times, Field(times, ice), forcing);
  const auto oceans = OceanFunctionsOf(run);
  ASSERT_NE(oceans.front(), state.oceanFunction);
  ASSERT_NE(oceans.back(), oceans.front());
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
  auto objective = [&](const Viscoelastic3D& at) {
    Evolution3D trial(at, scene.solver, state, options);
    const auto snaps = trial.RunMigrating(times, Field(times, ice), forcing);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_TRUE(snaps[n].converged) << "node " << n;
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped at node " << n;
    }
    return Objective3D(scene, snaps, measurements);
  };
  const auto pattern = OnCoupling(coupling, kPattern);
  const auto material = LateralDirection(op, kRadial, kAngular);
  const auto kernels = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                   .rtol = 1e-12,
                                                   .coupled_rtol = 1e-12,
                                                   .material_kernels = true})
                           .FirstOrderKernels(times, measurements, &trajectory, oceans);
  using Moved = std::function<std::unique_ptr<Viscoelastic3D>(double)>;
  for (const auto& [name, eps, tolerance, direction, moved, paired] :
       {std::tuple{"log viscosity", 0.004, 1e-6,
                   TangentPerturbation3D{.logViscosity = pattern},
                   Moved([&](double e) {
                     return std::make_unique<Viscoelastic3D>(
                         scene.rheology, Scene3D::kLmax3,
                         [=](double r, double theta, double phi) {
                           return LateralDelta(r, theta, phi) + e * kPattern(theta, phi);
                         });
                   }),
                   VolumePairing(coupling, kernels.logViscosityKernel3D[0], pattern)},
        std::tuple{"moduli", 0.004, 1e-6, TangentPerturbation3D{.material = material},
                   Moved([&](double e) {
                     return MovedModuli3D(scene, op, e, kRadial, kAngular);
                   }),
                   VolumePairing(coupling, kernels.kappaKernel3D, material.kappa) +
                       VolumePairing(coupling, kernels.muKernel3D, material.mu)}}) {
    auto along = [&](AdjointMethod method) {
      return Objective3D(scene,
                         TangentLinearSolver3D(evolution, {.method = method,
                                                           .rtol = 1e-12,
                                                           .coupled_rtol = 1e-12})
                             .Run(times, direction, &trajectory, oceans),
                         measurements);
    };
    const auto discrete = along(AdjointMethod::Discrete);
    const auto continuous = along(AdjointMethod::Continuous);
    const auto fd =
        Difference([&](double e) { return objective(*moved(e)); }, eps, discrete);
    std::cout << std::format(
        "[forced, 3D] {}: ratio {:.3f}, Richardson gap {:.2e}; discrete kernel {:.2e}; "
        "continuous run {:.2e}\n",
        name, fd.ratio, Gap(fd.richardson, discrete), Gap(paired, discrete),
        Gap(continuous, discrete));
    EXPECT_GT(fd.ratio, 3.0) << name;
    EXPECT_LT(fd.ratio, 5.0) << name;
    EXPECT_NEAR(fd.richardson, discrete, tolerance * std::abs(discrete)) << name;
    EXPECT_NEAR(paired, discrete, 1e-8 * std::abs(discrete)) << name;
    EXPECT_LT(Gap(continuous, discrete), 0.5) << name;
  }

  // second order
  std::mt19937 local(131);
  auto perNode = [&](double amplitude, bool zeroFirst) {
    auto out = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      out.push_back(scene.Smooth(local, amplitude));
      if (zeroFirst && n == 0) out.back().assign(out.back().size(), 0.0);
    }
    return out;
  };
  const auto patternW = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  TangentPerturbation3D u, w;
  u.directLoad = perNode(10.0 * scene.Sigma(), false);
  u.iceThickness = perNode(0.1 * scene.Metre(), true);
  u.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, pattern);
  u.material = material;
  w.directLoadOutsideWaterBudget = perNode(10.0 * scene.Sigma(), false);
  w.solidSurfaceChange = perNode(0.02 * scene.Metre(), false);
  w.iceThickness = perNode(0.1 * scene.Metre(), true);
  w.initialSeaLevel = scene.Smooth(local, 0.1 * scene.Metre());
  w.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, patternW));
  w.material = LateralDirection(op, [](double r) { return 1.0 - 0.5 * r; },
                                [](double theta, double phi) {
                                  return Cap(theta, phi, 1.9, 4.4, 0.7) +
                                         0.3 * std::sin(theta) * std::cos(theta) *
                                             std::cos(phi);
                                });
  TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                            .rtol = 1e-12,
                                            .coupled_rtol = 1e-12,
                                            .record = true});
  const auto runU = tangent.Run(times, u, &trajectory, oceans);
  const auto runW = tangent.Run(times, w, &trajectory, oceans);
  AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .coupled_rtol = 1e-12,
                                      .material_kernels = true});
  const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.itau, trajectory, oceans, u.material);
  const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
      times, measurements, runW, w.itau, trajectory, oceans, w.material);
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    EXPECT_LE(MaxDiff(kernelsU.ice.thickness[n], kernels.ice.thickness[n]),
              1e-10 * MaxAbs(kernels.ice.thickness[n]))
        << "node " << n;
  }
  EXPECT_LE(MaxDiff(kernelsU.muKernel3D, kernels.muKernel3D),
            1e-10 * MaxAbs(kernels.muKernel3D));
  const auto uw = PairingOf3D(scene, coupling, alongU, w);
  const auto wu = PairingOf3D(scene, coupling, alongW, u);
  SecondOrderDirection3D direction;
  direction.itauU = u.itau;
  direction.itauV = w.itau;
  direction.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
      op, pattern, OnCoupling(coupling, patternW));
  direction.materialU = u.material;
  direction.materialV = w.material;
  const auto forward = Objective3D(
      scene, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory, oceans),
      measurements);
  auto uvTerm = 0.0;
  for (auto b = std::size_t{0}; b < direction.itauUV.size(); ++b) {
    uvTerm += VolumePairing(coupling, kernels.itauKernel3D[b], direction.itauUV[b]);
  }
  std::cout << std::format(
      "[forced, 3D] Hessian: symmetry gap {:.2e}, against the second-order run {:.2e}\n",
      Gap(wu, uw), Gap(uw + uvTerm, forward));
  ASSERT_NE(forward, 0.0);
  EXPECT_NEAR(uw, wu, 1e-8 * std::abs(uw));
  EXPECT_NEAR(uw + uvTerm, forward, 1e-8 * std::abs(forward));
}

TEST(MigratingDerivatives, TheContinuousSweepsAboutAForcedRun3D) {
  // The 1D continuous forced gate under lateral viscosity (1e22 Pa s
  // mantle), one problem at 4 and 7 nodes over the 10 kyr: the continuous
  // 3D sweep's kernels of every forcing, its log-viscosity and moduli
  // kernels, and the continuous 3D second-order sweep's derivatives of
  // those along a direction in the thickness, the log viscosity and the
  // moduli tend to the discrete 3D sweeps' as the steps are halved.
  auto scene = Scene3D::Make(1e22);
  const auto [ice0, sea0] = scene.MarineCapState(6000.0, 50.0);
  const auto state = scene.solver.MakeState(ice0, sea0);
  std::mt19937 rng(137);
  const auto inside = scene.Smooth(rng, 2e4 * scene.Sigma());
  const auto outside = scene.Smooth(rng, 1e4 * scene.Sigma());
  const auto layer = scene.Smooth(rng, 400.0 * scene.Metre());
  const auto source = SourceAt(scene.solver, scene.solver.Radius(), 1e21);
  const auto iceField = scene.Smooth(rng, 0.1 * scene.Metre());
  const auto g0 = scene.Smooth(rng, 0.4);
  const auto gEnd = scene.Smooth(rng, 1.0);
  const auto gDisp = scene.Smooth(rng, 0.7);
  Viscoelastic3D op(scene.rheology, Scene3D::kLmax3, LateralDelta);
  const auto& coupling = op.Coupling();
  const auto pattern = OnCoupling(coupling, kPattern);
  const auto material = LateralDirection(op, kRadial, kAngular);
  auto surfaceL2 = [&](std::span<const double> f, std::span<const double> g) {
    return scene.Surface(f, g);
  };
  enum Quantity {
    kInside, kOutside, kLayer, kThickness, kSea, kMoment, kViscosity, kModuli, kHessian, kCount
  };
  const auto names = std::array<const char*, kCount>{
      "inside", "outside", "layer", "thickness", "initial sea level", "moment",
      "log viscosity", "moduli", "Hessian"};
  auto gapsAt = [&](int nodes) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(10.0 * n / (nodes - 1) * scene.Kyr());
    const auto base = BaseForcing(times, inside, outside, layer, source);
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].relativeSeaLevelDual = g0;
    measurements[times.size() - 1].relativeSeaLevelDual = gEnd;
    measurements[times.size() - 1].displacementDual = gDisp;
    Evolution3D evolution(op, scene.solver, state,
                          {.rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .max_nonlinear_iterations = 200,
                           .record_stride = 1});
    const auto run = evolution.RunMigrating(
        times, Field(times, Thinning3D(ice0, times, 1.0, 0.55)),
        ForcingOf<Evolution3D::Forcing>(times, base, nullptr, 0.0));
    const auto oceans = OceanFunctionsOf(run);
    EXPECT_NE(oceans.front(), state.oceanFunction);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
    TangentPerturbation3D u;
    for (const auto t : times) {
      auto d = iceField;
      for (auto& v : d) v *= t / times.back();
      u.iceThickness.push_back(std::move(d));
    }
    u.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, pattern);
    u.material = material;
    auto kernels = std::map<AdjointMethod, Kernels3D>{};
    auto derivatives = std::map<AdjointMethod, KernelDerivatives3D>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12,
                                          .material_kernels = true,
                                          .moment_points = {source.at}});
      kernels[method] = adjoint.FirstOrderKernels(times, measurements, &trajectory, oceans);
      const auto runU = TangentLinearSolver3D(evolution, {.method = method,
                                                          .rtol = 1e-12,
                                                          .coupled_rtol = 1e-12,
                                                          .record = true})
                            .Run(times, u, &trajectory, oceans);
      derivatives[method] = adjoint
                                .SecondOrderKernels(times, measurements, runU, u.itau,
                                                    trajectory, oceans, u.material)
                                .second;
    }
    const auto& d = kernels[AdjointMethod::Discrete];
    const auto& c = kernels[AdjointMethod::Continuous];
    EXPECT_EQ(c.ice.directLoadOutsideWaterBudgetKernels.size(), times.size());
    EXPECT_EQ(c.ice.solidSurfaceChangeKernels.size(), times.size());
    EXPECT_EQ(c.ice.momentKernels.size(), times.size());
    auto volume = [&](const std::vector<double>& x, const std::vector<double>& y) {
      auto e = x;
      for (auto i = std::size_t{0}; i < e.size(); ++i) e[i] -= y[i];
      return std::sqrt(VolumePairing(coupling, e, e) / VolumePairing(coupling, y, y));
    };
    auto moduli = [&](const auto& x, const auto& y) {
      auto ek = x.kappaKernel3D, em = x.muKernel3D;
      for (auto i = std::size_t{0}; i < ek.size(); ++i) {
        ek[i] -= y.kappaKernel3D[i];
        em[i] -= y.muKernel3D[i];
      }
      return std::sqrt(
          (VolumePairing(coupling, ek, ek) + VolumePairing(coupling, em, em)) /
          (VolumePairing(coupling, y.kappaKernel3D, y.kappaKernel3D) +
           VolumePairing(coupling, y.muKernel3D, y.muKernel3D)));
    };
    auto moment = [&](const Kernels3D& k) {
      auto out = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        out += k.ice.momentKernels[n][0].Contract(base.sources[n][0].moment);
      }
      return out;
    };
    const auto& dd = derivatives[AdjointMethod::Discrete];
    const auto& dc = derivatives[AdjointMethod::Continuous];
    auto gaps = std::array<double, kCount>{};
    gaps[kInside] = NodeDistance(surfaceL2, c.ice.directLoadKernels, d.ice.directLoadKernels);
    gaps[kOutside] = NodeDistance(surfaceL2, c.ice.directLoadOutsideWaterBudgetKernels,
                                  d.ice.directLoadOutsideWaterBudgetKernels);
    gaps[kLayer] = NodeDistance(surfaceL2, c.ice.solidSurfaceChangeKernels,
                                d.ice.solidSurfaceChangeKernels);
    gaps[kThickness] = NodeDistance(surfaceL2, c.ice.thickness, d.ice.thickness);
    gaps[kSea] = NodeDistance(surfaceL2, {c.ice.initialSeaLevel}, {d.ice.initialSeaLevel});
    gaps[kMoment] = Gap(moment(c), moment(d));
    gaps[kViscosity] = volume(c.logViscosityKernel3D[0], d.logViscosityKernel3D[0]);
    gaps[kModuli] = moduli(c, d);
    gaps[kHessian] = std::hypot(volume(dc.logViscosityKernel3D[0], dd.logViscosityKernel3D[0]),
                                moduli(dc, dd));
    return gaps;
  };
  const auto coarse = gapsAt(4);
  const auto fine = gapsAt(7);
  for (auto q = 0; q < kCount; ++q) {
    std::cout << std::format("[forced, 3D continuous] {}: {:.2e} {:.2e}\n", names[q],
                             coarse[q], fine[q]);
    // (the second-order derivatives carry the larger commute error at the
    // coarse 3.3 kyr steps, as in SecondOrderAdjoint3D's lateral gates)
    EXPECT_LT(coarse[q], q == kHessian ? 0.5 : 0.2) << names[q];
    EXPECT_GT(coarse[q] / fine[q], 2.0) << names[q];
  }
}
