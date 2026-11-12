// The 3D tangent-linear runs: with no lateral perturbation the 3D run is
// the 1D one through the grid pipeline; with lateral structure on, the
// run is linear in the loads, the discrete run along a log-viscosity
// direction is the exact derivative of the coupled forward (central
// differences at second order, Richardson to 1e-6), the continuous run
// pairs with the continuous 3D kernel at the commute level, and the
// second-order run is the derivative of the first-order one and symmetric.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
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

/// With the memory points at the GLL nodes, as in these tests, the points
/// are the nodes: a nodal direction as one for the time-stepped problem,
/// and back.
[[maybe_unused]] PointMaterialDirection AtGll(const MaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}
[[maybe_unused]] MaterialDirection Nodal(const PointMaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}

constexpr auto kLmax = 8;

double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// The 3D adjoint tests' scene: the prem8 body with a Maxwell mantle at
/// lmax 8, its solver and the analytic test state.
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

/// A function of (theta, phi) sampled over the coupling's grid values,
/// radius-independent.
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

/// The ramp of a load over the time grid.
std::function<std::vector<double>(double)> Ramp(const std::vector<double>& load,
                                                double span) {
  return [load, span](double t) {
    auto out = load;
    for (auto& v : out) v *= t / span;
    return out;
  };
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

double Objective(const Scene& scene, const std::vector<double>& g,
                 std::span<const double> field) {
  return scene.Surface(g, field);
}

}  // namespace

TEST(TangentLinear3D, ZeroLateralMatchesThe1DRun) {
  // With delta = 0 and the memory nodes at the GLL points the 3D
  // tangent-linear run is the 1D one through the grid pipeline, along
  // loads and an angle-uniform itau perturbation, both methods.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(5);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
  }

  // The 1D side: a smooth nodal itau perturbation of the branch.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Evolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
  const auto& base = scene.rheology.BranchAt(0);
  const auto& mesh = scene.material.Mesh();
  auto top = 0.0;
  for (auto v : base.itau) top = std::max(top, v);
  auto ditau1 = std::vector<double>(base.itau.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (base.itau[i] > 0.0) {
        const auto r = mesh.Radius(e, q);
        ditau1[i] = 0.1 * (0.3 + 0.7 * r * r) * top;
      }
    }
  }

  // The 3D side with delta = 0: the same perturbation, angle-uniform.
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  Evolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
  const auto& coupling = op3.Coupling();
  auto ditau3 = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full =
        static_cast<std::size_t>(coupling.ElementOfNode(rn)) * mesh.NGll() +
        coupling.NodeInElementOf(rn);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
      ditau3[at + x] = ditau1[full];
    }
  }

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    TangentPerturbation d1;
    d1.directLoad = deltas;
    d1.itau = {ditau1};
    const auto run1 =
        TangentLinearSolver(evolution1, {.method = method, .rtol = 1e-12})
            .Run(times, d1, &trajectory1);
    TangentPerturbation3D d3;
    d3.directLoad = deltas;
    d3.itau = {ditau3};
    const auto run3 = TangentLinearSolver3D(evolution3, {.method = method,
                                                         .rtol = 1e-12,
                                                         .coupled_rtol = 1e-12})
                          .Run(times, d3, &trajectory3);
    ASSERT_EQ(run3.size(), times.size());
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(run3[n].relativeSeaLevelChange, run1[n].relativeSeaLevelChange),
                1e-7 * MaxAbs(run1[n].relativeSeaLevelChange))
          << "node " << n << " method " << static_cast<int>(method);
      EXPECT_LT(MaxDiff(run3[n].displacement, run1[n].displacement),
                1e-7 * MaxAbs(run1[n].displacement))
          << "node " << n;
      EXPECT_LT(MaxDiff(run3[n].potential, run1[n].potential),
                1e-7 * MaxAbs(run1[n].potential))
          << "node " << n;
    }
  }
}

TEST(TangentLinear3D, LinearInTheLoadsWithLateralStructure) {
  // With lateral structure on, the coupled run is still linear in its
  // loads: the run along a load perturbation is the difference of two
  // forward runs to the fixed points' tolerance.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(7);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
  }
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  Evolution3D evolution(op, scene.solver, scene.state,
                                {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto base = evolution.Run(times, Ramp(load, times.back()));
  const auto plus = evolution.Run(times, [&](double t) {
    auto out = Ramp(load, times.back())(t);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) > 1e-12 * times.back()) continue;
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += deltas[n][i];
    }
    return out;
  });
  TangentPerturbation3D d;
  d.directLoad = deltas;
  const auto run =
      TangentLinearSolver3D(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12})
          .Run(times, d);
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    auto difference = plus[n].relativeSeaLevelChange;
    for (auto i = std::size_t{0}; i < difference.size(); ++i) {
      difference[i] -= base[n].relativeSeaLevelChange[i];
    }
    EXPECT_LT(MaxDiff(run[n].relativeSeaLevelChange, difference),
              1e-8 * MaxAbs(difference))
        << "node " << n;
    EXPECT_TRUE(run[n].converged);
  }
}

TEST(TangentLinear3D, ATideAndTheTangentialDisplacementAreCarried) {
  // The coupled run is linear in its load and its prescribed tide, so
  // the tangent-linear run along a perturbation of the two is the run
  // under that perturbation as its forcing, in every field (the
  // tangential displacement's potential among them), with an ocean and
  // without one, under lateral viscosity.
  auto scene = Scene::Make(1e21);
  const auto& grid = scene.solver.Grid();
  std::mt19937 rng(19);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto potential =
      1.0 / scene.solver.SolverScales().Factor(PotentialDimensions);
  auto deltas = std::vector<std::vector<double>>{};
  auto tides = std::vector<std::vector<std::complex<double>>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
    tides.push_back(grid.Expand(scene.Smooth(rng, potential)));
    tides.back()[SurfaceGrid::Index(0, 0)] = 0.0;
  }
  auto nodeOf = [&](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * times.back()) return n;
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto dry = scene.solver.NoOceanState();
  const auto& wet = scene.state;
  for (const auto* background : {&wet, &dry}) {
    const auto label = background == &dry ? "no ocean" : "ocean";
    Evolution3D evolution(op, scene.solver, *background,
                          {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto forced = evolution.Run(
        times, Evolution3D::Forcing{
                   .directLoad = [&](double t) { return deltas[nodeOf(t)]; },
                   .tide = [&](double t) { return tides[nodeOf(t)]; }});
    TangentPerturbation3D d;
    d.directLoad = deltas;
    d.tide = tides;
    const auto run =
        TangentLinearSolver3D(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12})
            .Run(times, d);
    ASSERT_EQ(run.size(), forced.size());
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_TRUE(run[n].converged);
      EXPECT_LT(MaxDiff(run[n].relativeSeaLevelChange,
                        forced[n].relativeSeaLevelChange),
                1e-8 * MaxAbs(forced[n].relativeSeaLevelChange))
          << "node " << n << ", " << label;
      EXPECT_LT(MaxDiff(run[n].displacement, forced[n].displacement),
                1e-8 * MaxAbs(forced[n].displacement))
          << "node " << n << ", " << label;
      EXPECT_LT(MaxDiff(run[n].potential, forced[n].potential),
                1e-8 * MaxAbs(forced[n].potential))
          << "node " << n << ", " << label;
      EXPECT_GT(MaxAbs(forced[n].tangentialPotential), 0.0);
      EXPECT_LT(
          MaxDiff(run[n].tangentialPotential, forced[n].tangentialPotential),
          1e-8 * MaxAbs(forced[n].tangentialPotential))
          << "node " << n << ", " << label;
    }
  }
}

TEST(TangentLinear3D, LateralDirectionIsTheDerivativeOfTheCoupledRun) {
  // Along a log-viscosity direction p with lateral structure on, the
  // discrete run is the exact derivative of the coupled forward: central
  // differences of a sea level functional, with the operator's delta
  // moved by eps p, converge at second order and Richardson lands on it;
  // the continuous run pairs with the continuous 3D kernel at the
  // commute level.
  auto scene = Scene::Make(1e22);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = g;

  auto functional = [&](double eps) {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) + eps * direction(theta, phi);
                      });
    Evolution3D evolution(op, scene.solver, scene.state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-12});
    return Objective(scene, g, evolution.Run(times, directLoad).back().relativeSeaLevelChange);
  };

  Viscoelastic3D op(scene.rheology, kLmax, baseDelta);
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto snapshots = evolution.Run(times, directLoad);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
  TangentPerturbation3D d;
  d.logViscosity = OnCoupling(op.Coupling(), direction);
  auto predicted = std::map<AdjointMethod, double>{};
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto run = TangentLinearSolver3D(evolution, {.method = method,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, d, &trajectory);
    predicted[method] = Objective(scene, g, run.back().relativeSeaLevelChange);
  }
  auto fd = std::map<double, double>{};
  for (double eps : {0.02, 0.01}) {
    fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
  }
  const auto exact = predicted[AdjointMethod::Discrete];
  const auto r1 = std::abs(fd[0.02] - exact);
  const auto r2 = std::abs(fd[0.01] - exact);
  EXPECT_GT(r1 / r2, 3.0);
  EXPECT_LT(r1 / r2, 5.0);
  const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
  EXPECT_NEAR(richardson, exact, 1e-6 * std::abs(exact));

  // The continuous pair: the continuous run against the continuous
  // kernel paired with p, at the commute level.
  AdjointSolver3D adjoint(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
  const auto paired =
      VolumePairing(op.Coupling(), kernels.logViscosityKernel3D[0], d.logViscosity);
  const auto continuous = predicted[AdjointMethod::Continuous];
  EXPECT_LT(std::abs(paired - continuous), 0.1 * std::abs(continuous))
      << paired << " " << continuous << " discrete " << exact;
  EXPECT_LT(std::abs(exact - continuous), 0.1 * std::abs(exact));
}

TEST(TangentLinear3D, SecondOrderRunIsTheDerivativeOfTheFirst) {
  // The second-order run along (u, v) — loads and log-viscosity
  // directions, with the second variation itau u v of the log
  // parameterisation — is the derivative along u of the first-order run
  // along v evaluated at the moved operator: central differences converge
  // at second order and Richardson lands on it; it is symmetric in u and
  // v, and it vanishes without a viscosity perturbation.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(9);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto pV = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  auto loadsU = std::vector<std::vector<double>>{};
  auto loadsV = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loadsU.push_back(scene.Smooth(rng, 20.0 * scene.Sigma()));
    loadsV.push_back(scene.Smooth(rng, 20.0 * scene.Sigma()));
  }
  const auto g = scene.Smooth(rng, 1.0);

  /// The forward and first-order run along v at the operator moved by
  /// eps p_u and the loads by eps loads_u, with v's itau at the moved
  /// operator (-itau(eps) p_v).
  auto firstOrderAt = [&](double eps, std::vector<Evolution3D::Snapshot>* runU,
                          std::vector<Evolution3D::Snapshot>* runV,
                          ForwardTrajectory3D* trajectoryOut,
                          Viscoelastic3D** opOut, Evolution3D** evOut) {
    auto* op = new Viscoelastic3D(scene.rheology, kLmax,
                                  [&](double r, double theta, double phi) {
                                    return baseDelta(r, theta, phi) + eps * pU(theta, phi);
                                  });
    auto* evolution = new Evolution3D(
        *op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution->Run(times, [&](double t) {
      auto out = directLoad(t);
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        if (std::abs(t - times[n]) > 1e-12 * times.back()) continue;
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] += eps * loadsU[n][i];
        }
      }
      return out;
    });
    auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    TangentLinearSolver3D tangent(
        *evolution,
        {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12,
         .record = true});
    TangentPerturbation3D v;
    v.directLoad = loadsV;
    v.logViscosity = OnCoupling(op->Coupling(), pV);
    auto rv = tangent.Run(times, v, &trajectory);
    const auto value = Objective(scene, g, rv.back().relativeSeaLevelChange);
    if (runV != nullptr) *runV = std::move(rv);
    if (runU != nullptr) {
      TangentPerturbation3D u;
      u.directLoad = loadsU;
      u.logViscosity = OnCoupling(op->Coupling(), pU);
      *runU = tangent.Run(times, u, &trajectory);
    }
    if (trajectoryOut != nullptr) *trajectoryOut = std::move(trajectory);
    if (opOut != nullptr && evOut != nullptr) {
      *opOut = op;
      *evOut = evolution;
    } else {
      delete evolution;
      delete op;
    }
    return value;
  };

  std::vector<Evolution3D::Snapshot> runU, runV;
  ForwardTrajectory3D trajectory;
  Viscoelastic3D* op = nullptr;
  Evolution3D* evolution = nullptr;
  (void)firstOrderAt(0.0, &runU, &runV, &trajectory, &op, &evolution);
  const auto fieldU = OnCoupling(op->Coupling(), pU);
  const auto fieldV = OnCoupling(op->Coupling(), pV);
  SecondOrderDirection3D uv;
  uv.itauU = TangentLinearSolver3D::ItauOfLogViscosity(*op, fieldU);
  uv.itauV = TangentLinearSolver3D::ItauOfLogViscosity(*op, fieldV);
  uv.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(*op, fieldU, fieldV);
  TangentLinearSolver3D tangent(
      *evolution,
      {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto second = tangent.RunSecondOrder(times, runU, runV, uv, &trajectory);
  const auto predicted = Objective(scene, g, second.back().relativeSeaLevelChange);

  auto fd = std::map<double, double>{};
  for (double eps : {0.02, 0.01}) {
    fd[eps] = (firstOrderAt(eps, nullptr, nullptr, nullptr, nullptr, nullptr) -
               firstOrderAt(-eps, nullptr, nullptr, nullptr, nullptr, nullptr)) /
              (2.0 * eps);
  }
  const auto r1 = std::abs(fd[0.02] - predicted);
  const auto r2 = std::abs(fd[0.01] - predicted);
  EXPECT_GT(r1 / r2, 3.0);
  EXPECT_LT(r1 / r2, 5.0);
  const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
  EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted));

  // Symmetry in u and v.
  SecondOrderDirection3D vu;
  vu.itauU = uv.itauV;
  vu.itauV = uv.itauU;
  vu.itauUV = uv.itauUV;
  const auto swapped = tangent.RunSecondOrder(times, runV, runU, vu, &trajectory);
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_LT(MaxDiff(swapped[n].relativeSeaLevelChange, second[n].relativeSeaLevelChange),
              1e-9 * MaxAbs(second[n].relativeSeaLevelChange) + 1e-300)
        << "node " << n;
  }

  // No viscosity perturbation: an affine map, no second derivative.
  const auto none = tangent.RunSecondOrder(times, runU, runV, {}, &trajectory);
  for (const auto& snap : none) EXPECT_EQ(MaxAbs(snap.relativeSeaLevelChange), 0.0);

  delete evolution;
  delete op;
}

// -- the material ------------------------------------------------------------

namespace {

/// A nodal field over the material's mesh on the coupling's grid values,
/// angle-uniform.
std::vector<double> UniformOnCoupling(const GridCoupling& coupling,
                                      const Material& material,
                                      std::span<const double> nodal) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
      out[at + x] = nodal[full];
    }
  }
  return out;
}

/// A field over the memory points, (element, point), spread uniformly in
/// angle over each of the coupling's memory nodes.
std::vector<double> UniformFromPoints(const GridCoupling& coupling,
                                      std::span<const double> atPoints) {
  const auto np = static_cast<std::size_t>(coupling.Points().count);
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto value =
        atPoints[static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
                 coupling.NodeInElementOf(rn)];
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
      out[at + x] = value;
    }
  }
  return out;
}

/// A smooth direction in kappa and mu on the relaxing elements (the
/// coupling's nodes), as nodal fields over the mesh.
MaterialDirection SmoothMaterialDirection(const Scene& scene) {
  const auto& material = scene.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (!scene.rheology.IsActive(i)) continue;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      d.kappa[i] = 0.5 * profile * k0[i];
      d.mu[i] = profile * m0[i];
    }
  }
  return d;
}

/// The rheology of the material shifted along a direction: the branch
/// moduli as the same fractions of the (shifted) unrelaxed modulus, the
/// inverse times held.
std::pair<Material, Rheology> ShiftedRheology(const Scene& scene,
                                              const MaterialDirection& d,
                                              double eps) {
  const auto shifted = scene.material.Shifted(d, eps);
  const auto muU = scene.rheology.MuUnrelaxed();
  const auto [kappa, mu] = shifted.Moduli().KappaMu();
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < scene.rheology.BranchCount(); ++k) {
    auto branch = scene.rheology.BranchAt(k);
    for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
      branch.mu[i] = muU[i] > 0.0 ? branch.mu[i] / muU[i] * mu[i] : 0.0;
    }
    branches.push_back(std::move(branch));
  }
  auto rheology = Rheology::Create(shifted, branches).value();
  return {shifted, std::move(rheology)};
}

}  // namespace

TEST(TangentLinear3D, MaterialDirectionMatchesDifferences) {
  // Along an angle-uniform direction in kappa and mu on the memory nodes,
  // with lateral viscosity ON (so the effective modulus's variation
  // carries the step's lateral weights), the 3D tangent run matches
  // central differences of the coupled 3D run on the shifted material at
  // second order with Richardson landing at 1e-6, both schemes.
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  const auto d = SmoothMaterialDirection(scene);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto functional = [&](double eps) {
      const auto [shifted, rheology] = ShiftedRheology(scene, d, eps);
      Viscoelastic3D op(rheology, kLmax, baseDelta,
                        {.memory_points = MemoryPointSet::Gll});
      Evolution3D evolution(
          op, scene.solver, scene.state,
          {.scheme = scheme, .rotation = RotationalFeedback::None,
           .rtol = 1e-12, .coupled_rtol = 1e-12});
      return Objective(scene, g, evolution.Run(times, directLoad).back().relativeSeaLevelChange);
    };

    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.memory_points = MemoryPointSet::Gll});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    TangentPerturbation3D direction;
    direction.material.kappa = UniformOnCoupling(op.Coupling(), scene.material, d.kappa);
    direction.material.mu = UniformOnCoupling(op.Coupling(), scene.material, d.mu);
    const auto run = TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, direction, &trajectory);
    const auto predicted = Objective(scene, g, run.back().relativeSeaLevelChange);

    // wide steps (1%, 0.5% of the moduli): the truncation error above
    // the solves' noise, as in the 1D test
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(TangentLinear3D, MaterialDirectionMatchesDifferencesAtGaussPoints) {
  // The same at the default memory nodes, the Gauss points: along an
  // angle-uniform direction in kappa and mu on the memory nodes, with
  // lateral viscosity ON, the 3D tangent run matches central differences
  // of the coupled 3D run with its moduli perturbed at the nodes
  // (Viscoelastic3D::Options::moduli, the relative perturbation eps times
  // the profile) at second order with Richardson landing at 1e-6, both
  // schemes; and with the rotational feedback on, axial and full.
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  auto profile = [](double r) {
    return 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
  };

  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    // rotational feedback under one scheme each
    if (rotation != RotationalFeedback::None &&
        (rotation == RotationalFeedback::Axial) !=
            (scheme == Scheme::BackwardEuler)) {
      continue;
    }
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta);
    ASSERT_EQ(op.Coupling().Points().set, MemoryPointSet::Gauss);
    const auto& coupling = op.Coupling();
    const auto& mesh = scene.material.Mesh();
    // the relaxing elements, which the coupling carries: the perturbation
    // is confined to them, so that the perturbed runs carry the same nodes
    auto inner = mesh.Right(mesh.ElementCount() - 1), outer = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto e = coupling.ElementOfNode(rn);
      inner = std::min(inner, mesh.Left(e));
      outer = std::max(outer, mesh.Right(e));
    }
    auto inside = [=](double r) { return r > inner && r < outer; };

    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = rotation,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    const auto np = static_cast<std::size_t>(coupling.Points().count);
    const auto muU = op.Reference().UnrelaxedShearAtPoints();
    const auto kappa = op.Reference().BulkModulusAtPoints();
    TangentPerturbation3D direction;
    direction.material.kappa.assign(coupling.GridValueCount(), 0.0);
    direction.material.mu.assign(coupling.GridValueCount(), 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto point =
          static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
          coupling.NodeInElementOf(rn);
      const auto value = profile(coupling.RadiusOfNode(rn));
      const auto at = coupling.FieldOffset(rn);
      for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
        direction.material.kappa[at + x] = 0.5 * value * kappa[point];
        direction.material.mu[at + x] = value * muU[point];
      }
    }
    const auto run = TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, direction, &trajectory);
    const auto predicted = Objective(scene, g, run.back().relativeSeaLevelChange);

    auto functional = [&](double eps) {
      Viscoelastic3D moved(
          scene.rheology, kLmax, baseDelta,
          {.moduli = {.shear =
                          [=](double r, double, double) {
                            return inside(r) ? eps * profile(r) : 0.0;
                          },
                      .bulk = [=](double r, double, double) {
                        return inside(r) ? 0.5 * eps * profile(r) : 0.0;
                      }}});
      EXPECT_EQ(moved.Coupling().NodeCount(), coupling.NodeCount());
      Evolution3D movedEvolution(
          moved, scene.solver, scene.state,
          {.scheme = scheme, .rotation = rotation,
           .rtol = 1e-12, .coupled_rtol = 1e-12});
      return Objective(scene, g,
                       movedEvolution.Run(times, directLoad).back().relativeSeaLevelChange);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme) << " rotation "
        << static_cast<int>(rotation);
  }
}

TEST(TangentLinear3D, ZeroLateralMaterialMatchesThe1DRun) {
  // With delta = 0 the 3D material tangent along an angle-uniform
  // direction is the 1D one through the grid pipeline (the effective
  // system's variation by the pointwise product against the 1D assembled
  // variation operator), both schemes, with the memory points at the GLL
  // nodes and at the Gauss points (the two take a perturbation of the
  // moduli at the points by the one rule); and a lateral direction with
  // the same angular mean is not the same run.
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  const auto d = SmoothMaterialDirection(scene);
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op1(scene.rheology, kLmax, {.memory_points = set});
    Evolution evolution1(op1, scene.solver, scene.state,
                                 {.scheme = scheme,
                                  .rotation = RotationalFeedback::None,
                                  .rtol = 1e-12,
                                  .record_stride = 1});
    const auto trajectory1 =
        ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
    TangentPerturbation d1;
    d1.material = AtMemoryPoints(op1.Points(), d);
    const auto run1 = TangentLinearSolver(evolution1, {.rtol = 1e-12})
                          .Run(times, d1, &trajectory1);

    Viscoelastic3D op3(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; },
                       {.memory_points = set});
    Evolution3D evolution3(
        op3, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory3 =
        ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
    const auto& coupling = op3.Coupling();
    TangentPerturbation3D d3;
    d3.material.kappa = UniformFromPoints(coupling, d1.material.kappa);
    d3.material.mu = UniformFromPoints(coupling, d1.material.mu);
    const auto run3 = TangentLinearSolver3D(evolution3, {.rtol = 1e-12,
                                                         .coupled_rtol = 1e-12})
                          .Run(times, d3, &trajectory3);
    ASSERT_EQ(run3.size(), times.size());
    // the ramp starts from nothing: no start response at node 0
    EXPECT_EQ(MaxAbs(run1[0].relativeSeaLevelChange), 0.0);
    EXPECT_EQ(MaxAbs(run3[0].relativeSeaLevelChange), 0.0);
    for (auto n = std::size_t{1}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(run3[n].relativeSeaLevelChange, run1[n].relativeSeaLevelChange),
                1e-7 * MaxAbs(run1[n].relativeSeaLevelChange))
          << "node " << n << " scheme " << static_cast<int>(scheme)
          << ", points " << static_cast<int>(set);
      EXPECT_LT(MaxDiff(run3[n].displacement, run1[n].displacement),
                1e-7 * MaxAbs(run1[n].displacement))
          << "node " << n << " scheme " << static_cast<int>(scheme);
      EXPECT_LT(MaxDiff(run3[n].potential, run1[n].potential),
                1e-7 * MaxAbs(run1[n].potential))
          << "node " << n << " scheme " << static_cast<int>(scheme);
    }

    // A lateral direction of the same angular mean: a different run,
    // with toroidal coupling and degree mixing the 1D run cannot have.
    const auto lateral = OnCoupling(coupling, [](double theta, double phi) {
      return 1.0 + 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
    });
    TangentPerturbation3D dl;
    dl.material.mu = d3.material.mu;
    for (auto i = std::size_t{0}; i < lateral.size(); ++i) {
      dl.material.mu[i] *= lateral[i];
    }
    const auto runL = TangentLinearSolver3D(evolution3, {.rtol = 1e-12,
                                                         .coupled_rtol = 1e-12})
                          .Run(times, dl, &trajectory3);
    EXPECT_GT(MaxDiff(runL.back().relativeSeaLevelChange, run3.back().relativeSeaLevelChange),
              1e-3 * MaxAbs(run3.back().relativeSeaLevelChange));
  }
}


TEST(TangentLinear3D, SecondOrderRunInTheMaterial) {
  // With directions u and v in the loads, the log viscosity (lateral) AND
  // the moduli (angle-uniform, so that the radial material can be
  // shifted), with lateral viscosity on, the second-order run is the
  // derivative at the base point along u of the first-order run along v:
  // central differences of the latter — the loads moved by eps loads_u,
  // delta by eps p_u and the material by eps delta m_u — converge to it
  // at second order and Richardson lands on it; and it is symmetric in u
  // and v.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(13);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto pV = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  auto loadsU = std::vector<std::vector<double>>{};
  auto loadsV = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loadsU.push_back(scene.Smooth(rng, 20.0 * scene.Sigma()));
    loadsV.push_back(scene.Smooth(rng, 20.0 * scene.Sigma()));
  }
  const auto g = scene.Smooth(rng, 1.0);
  const auto dU = SmoothMaterialDirection(scene);
  auto dV = SmoothMaterialDirection(scene);
  {
    const auto& mesh = scene.material.Mesh();
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        const auto r = mesh.Radius(e, q);
        dV.kappa[i] *= (1.0 - 0.5 * r) / 0.7;
        dV.mu[i] *= (1.0 - 0.5 * r) / 0.7;
      }
    }
  }
  struct Moved {
    std::unique_ptr<Viscoelastic3D> op;
    std::unique_ptr<Evolution3D> evolution;
    ForwardTrajectory3D trajectory;
  };
  auto moved = [&](double eps) {
    const auto [shifted, rheology] = ShiftedRheology(scene, dU, eps);
    Moved m;
    m.op = std::make_unique<Viscoelastic3D>(
        rheology, kLmax,
        [&, eps](double r, double theta, double phi) {
          return baseDelta(r, theta, phi) + eps * pU(theta, phi);
        },
        Viscoelastic3D::Options{.memory_points = MemoryPointSet::Gll});
    m.evolution = std::make_unique<Evolution3D>(
        *m.op, scene.solver, scene.state,
        Evolution3D::Options{.rotation = RotationalFeedback::None,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .record_stride = 1});
    m.trajectory = ForwardTrajectory3D::FromSnapshots(
        m.evolution->Run(times, [&](double t) {
          auto out = directLoad(t);
          for (auto n = std::size_t{0}; n < times.size(); ++n) {
            if (std::abs(t - times[n]) > 1e-12 * times.back()) continue;
            for (auto i = std::size_t{0}; i < out.size(); ++i) {
              out[i] += eps * loadsU[n][i];
            }
          }
          return out;
        }));
    return m;
  };
  auto directionV = [&](const Viscoelastic3D& op) {
    TangentPerturbation3D v;
    v.directLoad = loadsV;
    v.logViscosity = OnCoupling(op.Coupling(), pV);
    v.material.kappa = UniformOnCoupling(op.Coupling(), scene.material, dV.kappa);
    v.material.mu = UniformOnCoupling(op.Coupling(), scene.material, dV.mu);
    return v;
  };
  const auto base = moved(0.0);
  const auto& coupling = base.op->Coupling();
  TangentLinearSolver3D tangent(
      *base.evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12,
                        .coupled_rtol = 1e-12, .record = true});
  TangentPerturbation3D u;
  u.directLoad = loadsU;
  u.logViscosity = OnCoupling(coupling, pU);
  u.material.kappa = UniformOnCoupling(coupling, scene.material, dU.kappa);
  u.material.mu = UniformOnCoupling(coupling, scene.material, dU.mu);
  const auto v = directionV(*base.op);
  const auto runU = tangent.Run(times, u, &base.trajectory);
  const auto runV = tangent.Run(times, v, &base.trajectory);
  const auto fieldU = OnCoupling(coupling, pU);
  const auto fieldV = OnCoupling(coupling, pV);
  SecondOrderDirection3D uv;
  uv.itauU = TangentLinearSolver3D::ItauOfLogViscosity(*base.op, fieldU);
  uv.itauV = TangentLinearSolver3D::ItauOfLogViscosity(*base.op, fieldV);
  uv.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(*base.op, fieldU, fieldV);
  uv.materialU = u.material;
  uv.materialV = v.material;
  const auto second = tangent.RunSecondOrder(times, runU, runV, uv, &base.trajectory);
  const auto predicted = Objective(scene, g, second.back().relativeSeaLevelChange);

  auto firstOrderAt = [&](double eps) {
    const auto m = moved(eps);
    TangentLinearSolver3D along(*m.evolution, {.method = AdjointMethod::Discrete,
                                               .rtol = 1e-12,
                                               .coupled_rtol = 1e-12});
    return Objective(scene, g,
                     along.Run(times, directionV(*m.op), &m.trajectory).back().relativeSeaLevelChange);
  };
  // wide steps (1%, 0.5% of the moduli), as the material tangent's gate
  auto fd = std::map<double, double>{};
  for (double eps : {0.1, 0.05}) {
    fd[eps] = (firstOrderAt(eps) - firstOrderAt(-eps)) / (2.0 * eps);
  }
  const auto r1 = std::abs(fd[0.1] - predicted);
  const auto r2 = std::abs(fd[0.05] - predicted);
  EXPECT_GT(r1 / r2, 3.0) << fd[0.1] << " " << fd[0.05] << " " << predicted;
  EXPECT_LT(r1 / r2, 5.0);
  const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
  EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted));

  SecondOrderDirection3D vu;
  vu.itauU = uv.itauV;
  vu.itauV = uv.itauU;
  vu.itauUV = uv.itauUV;
  vu.materialU = v.material;
  vu.materialV = u.material;
  const auto swapped = tangent.RunSecondOrder(times, runV, runU, vu, &base.trajectory);
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_LT(MaxDiff(swapped[n].relativeSeaLevelChange, second[n].relativeSeaLevelChange),
              1e-9 * MaxAbs(second[n].relativeSeaLevelChange) + 1e-300)
        << "node " << n;
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

/// At a laterally heterogeneous elastic background (10% shear, 5% bulk,
/// angle-varying) with lateral viscosity on: the 3D tangent run along a
/// LATERAL direction in kappa and mu matches central differences of the
/// coupled run with the background moduli shifted along it, second
/// order with Richardson at 1e-6, both schemes; the steps closed by the
/// given solver (the coupled fixed point, or the nested conjugate-
/// gradient solves) in the runs and the tangent alike.
void MaterialDirectionMatchesDifferencesUnderLateralModuli(
    Viscoelastic3D::Options::StepSolver solver) {
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto directLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  // the background and the (relative) direction
  const auto shear0 = [](double r, double theta, double phi) {
    return 0.1 * (0.5 + r * r) *
           (std::sin(theta) * std::cos(theta) * std::cos(phi) +
            0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
  };
  const auto bulk0 = [](double r, double theta, double) {
    return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
  };
  const auto vShear = [](double r, double theta, double phi) {
    return 0.1 * (0.5 + r * r) *
           (Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6));
  };
  const auto vBulk = [](double r, double theta, double phi) {
    return 0.05 * (1.0 - 0.5 * r) * (1.0 + 0.5 * std::sin(theta) * std::sin(phi));
  };
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto functional = [&](double eps) {
      Viscoelastic3D op(
          scene.rheology, kLmax, baseDelta,
          {.solver = solver,
           .solve_rtol = 1e-12,
           .memory_points = MemoryPointSet::Gll,
           .moduli = {.shear = [&](double r, double t, double p) {
                        return shear0(r, t, p) + eps * vShear(r, t, p);
                      },
                      .bulk = [&](double r, double t, double p) {
                        return bulk0(r, t, p) + eps * vBulk(r, t, p);
                      }}});
      Evolution3D evolution(
          op, scene.solver, scene.state,
          {.scheme = scheme, .rotation = RotationOf(scheme),
           .rtol = 1e-12, .coupled_rtol = 1e-12});
      return Objective(scene, g, evolution.Run(times, directLoad).back().relativeSeaLevelChange);
    };
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.solver = solver,
                       .solve_rtol = 1e-12,
                       .memory_points = MemoryPointSet::Gll,
                       .moduli = {.shear = shear0, .bulk = bulk0}});
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationOf(scheme),
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    // the absolute direction over the grid values: the relative one
    // times the reference moduli at the nodes
    const auto& coupling = op.Coupling();
    TangentPerturbation3D direction;
    direction.material.mu.assign(coupling.GridValueCount(), 0.0);
    direction.material.kappa.assign(coupling.GridValueCount(), 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto r = coupling.RadiusOfNode(rn);
      const auto at = coupling.FieldOffset(rn);
      const auto mu = coupling.GridOf(rn).FromFunction(
          [&](double t, double p) { return vShear(r, t, p) * op.UnrelaxedShearAtNodes()[rn]; });
      const auto kappa = coupling.GridOf(rn).FromFunction(
          [&](double t, double p) { return vBulk(r, t, p) * op.BulkModulusAtNodes()[rn]; });
      for (auto x = std::size_t{0}; x < mu.size(); ++x) {
        direction.material.mu[at + x] = mu[x];
        direction.material.kappa[at + x] = kappa[x];
      }
    }
    const auto run = TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, direction, &trajectory);
    const auto predicted = Objective(scene, g, run.back().relativeSeaLevelChange);
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(TangentLinear3D, MaterialDirectionMatchesDifferencesUnderLateralModuli) {
  MaterialDirectionMatchesDifferencesUnderLateralModuli(
      Viscoelastic3D::Options::StepSolver::FixedPoint);
}

TEST(TangentLinear3D, MaterialDirectionMatchesDifferencesUnderConjugateGradientSteps) {
  MaterialDirectionMatchesDifferencesUnderLateralModuli(
      Viscoelastic3D::Options::StepSolver::ConjugateGradients);
}

TEST(TangentLinear3D, ModuliAboutASourceDrivenRun) {
  // The moduli as parameters of a 3D run under point sources with
  // lateral viscosity: along a direction in kappa and mu the tangent run
  // is the derivative of the forward run on the shifted material
  // (central differences at second order, Richardson to 1e-6), and the
  // discrete sweep's material kernels pair with the direction to the
  // same number; the displacement and the toroidal potential measured.
  // The base solutions are redone under the sources' recorded forces.
  auto scene = Scene::Make(1e21);
  const auto& sle = scene.solver.SolverScales();
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
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto gU = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  const auto gW = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 2.0, 3.5, 0.6); });
  auto objective = [&](const std::vector<Evolution3D::Snapshot>& run) {
    return Objective(scene, gU, run.back().displacement) +
           Objective(scene, gW, run.back().toroidalPotential);
  };
  const auto d = SmoothMaterialDirection(scene);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = "scheme " + std::to_string(static_cast<int>(scheme));
    auto functional = [&](double eps) {
      const auto [shifted, rheology] = ShiftedRheology(scene, d, eps);
      Viscoelastic3D op(rheology, kLmax, baseDelta,
                        {.memory_points = MemoryPointSet::Gll});
      Evolution3D evolution(op, scene.solver, scene.state,
                            {.scheme = scheme,
                             .rotation = RotationalFeedback::None,
                             .rtol = 1e-12,
                             .coupled_rtol = 1e-12});
      return objective(evolution.Run(times, forcing));
    };
    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.memory_points = MemoryPointSet::Gll});
    Evolution3D evolution(op, scene.solver, scene.state,
                          {.scheme = scheme,
                           .rotation = RotationalFeedback::None,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, forcing));
    TangentPerturbation3D direction;
    direction.material.kappa =
        UniformOnCoupling(op.Coupling(), scene.material, d.kappa);
    direction.material.mu = UniformOnCoupling(op.Coupling(), scene.material, d.mu);
    const auto predicted =
        objective(TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                    .rtol = 1e-12,
                                                    .coupled_rtol = 1e-12})
                      .Run(times, direction, &trajectory));
    ASSERT_NE(predicted, 0.0) << label;
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << label << " " << fd[0.1] << " " << fd[0.05] << " "
                            << predicted;
    EXPECT_LT(r1 / r2, 5.0) << label;
    EXPECT_NEAR((4.0 * fd[0.05] - fd[0.1]) / 3.0, predicted,
                1e-6 * std::abs(predicted))
        << label;

    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].displacementDual = gU;
    measurements[times.size() - 1].toroidalPotentialDual = gW;
    const auto kernels =
        AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                    .rtol = 1e-12,
                                    .coupled_rtol = 1e-12,
                                    .material_kernels = true})
            .FirstOrderKernels(times, measurements, &trajectory);
    const auto paired =
        VolumePairing(op.Coupling(), kernels.kappaKernel3D,
                      direction.material.kappa) +
        VolumePairing(op.Coupling(), kernels.muKernel3D, direction.material.mu);
    EXPECT_NEAR(paired, predicted, 1e-8 * std::abs(predicted)) << label;

    // The sources and the moduli perturbed in one run: the sum of the
    // run along each, in every field at every node.
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    TangentPerturbation3D sourcesOnly;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      auto at = sources;
      for (auto& source : at) {
        auto& M = source.moment;
        for (auto* c : {&M.rr, &M.tt, &M.pp, &M.rt, &M.rp, &M.tp}) {
          *c *= 0.2 + 0.1 * static_cast<double>(n);
        }
      }
      sourcesOnly.sources.push_back(std::move(at));
    }
    auto together = direction;
    together.sources = sourcesOnly.sources;
    const auto alongModuli = tangent.Run(times, direction, &trajectory);
    const auto alongSources = tangent.Run(times, sourcesOnly, &trajectory);
    const auto alongBoth = tangent.Run(times, together, &trajectory);
    for (auto field : {&Evolution3D::Snapshot::displacement,
                       &Evolution3D::Snapshot::tangentialPotential,
                       &Evolution3D::Snapshot::toroidalPotential,
                       &Evolution3D::Snapshot::relativeSeaLevelChange}) {
      auto worst = 0.0, scale = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        const auto& both = alongBoth[n].*field;
        const auto& m = alongModuli[n].*field;
        const auto& q = alongSources[n].*field;
        ASSERT_EQ(both.size(), m.size());
        ASSERT_EQ(both.size(), q.size());
        for (auto i = std::size_t{0}; i < both.size(); ++i) {
          worst = std::max(worst, std::abs(both[i] - m[i] - q[i]));
          scale = std::max({scale, std::abs(m[i]), std::abs(q[i])});
        }
      }
      EXPECT_GT(scale, 0.0) << label;
      EXPECT_LT(worst, 1e-9 * scale) << label;
    }
  }
}
