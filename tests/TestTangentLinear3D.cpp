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
  const auto iceLoad = Ramp(load, times.back());
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltas.push_back(scene.Smooth(rng, 10.0 * scene.Sigma()));
  }

  // The 1D side: a smooth nodal itau perturbation of the branch.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, iceLoad));
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
  SeaLevelEvolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, iceLoad));
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
      EXPECT_LT(MaxDiff(run3[n].seaLevelChange, run1[n].seaLevelChange),
                1e-7 * MaxAbs(run1[n].seaLevelChange))
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
  SeaLevelEvolution3D evolution(op, scene.solver, scene.state,
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
    auto difference = plus[n].seaLevelChange;
    for (auto i = std::size_t{0}; i < difference.size(); ++i) {
      difference[i] -= base[n].seaLevelChange[i];
    }
    EXPECT_LT(MaxDiff(run[n].seaLevelChange, difference),
              1e-8 * MaxAbs(difference))
        << "node " << n;
    EXPECT_TRUE(run[n].converged);
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
  const auto iceLoad = Ramp(load, times.back());
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = g;

  auto functional = [&](double eps) {
    Viscoelastic3D op(scene.rheology, kLmax,
                      [&](double r, double theta, double phi) {
                        return baseDelta(r, theta, phi) + eps * direction(theta, phi);
                      });
    SeaLevelEvolution3D evolution(op, scene.solver, scene.state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-12});
    return Objective(scene, g, evolution.Run(times, iceLoad).back().seaLevelChange);
  };

  Viscoelastic3D op(scene.rheology, kLmax, baseDelta);
  SeaLevelEvolution3D evolution(
      op, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto snapshots = evolution.Run(times, iceLoad);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
  TangentPerturbation3D d;
  d.logViscosity = OnCoupling(op.Coupling(), direction);
  auto predicted = std::map<AdjointMethod, double>{};
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto run = TangentLinearSolver3D(evolution, {.method = method,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, d, &trajectory);
    predicted[method] = Objective(scene, g, run.back().seaLevelChange);
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
  const auto kernels = adjoint.IceHistoryKernels(times, measurements, &trajectory);
  const auto& coupling = op.Coupling();
  auto paired = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto at = coupling.FieldOffset(rn);
    const auto size = coupling.FieldSizeOf(rn);
    auto product = std::vector<double>(size);
    for (auto x = std::size_t{0}; x < size; ++x) {
      product[x] = kernels.logViscosityKernel3D[0][at + x] * d.logViscosity[at + x];
    }
    paired += coupling.GridOf(rn).Integrate(product);
  }
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
  const auto iceLoad = Ramp(load, times.back());
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
  auto firstOrderAt = [&](double eps, std::vector<SeaLevelEvolution3D::Snapshot>* runU,
                          std::vector<SeaLevelEvolution3D::Snapshot>* runV,
                          ForwardTrajectory3D* trajectoryOut,
                          Viscoelastic3D** opOut, SeaLevelEvolution3D** evOut) {
    auto* op = new Viscoelastic3D(scene.rheology, kLmax,
                                  [&](double r, double theta, double phi) {
                                    return baseDelta(r, theta, phi) + eps * pU(theta, phi);
                                  });
    auto* evolution = new SeaLevelEvolution3D(
        *op, scene.solver, scene.state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution->Run(times, [&](double t) {
      auto out = iceLoad(t);
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
    const auto value = Objective(scene, g, rv.back().seaLevelChange);
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

  std::vector<SeaLevelEvolution3D::Snapshot> runU, runV;
  ForwardTrajectory3D trajectory;
  Viscoelastic3D* op = nullptr;
  SeaLevelEvolution3D* evolution = nullptr;
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
  const auto predicted = Objective(scene, g, second.back().seaLevelChange);

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
    EXPECT_LT(MaxDiff(swapped[n].seaLevelChange, second[n].seaLevelChange),
              1e-9 * MaxAbs(second[n].seaLevelChange) + 1e-300)
        << "node " << n;
  }

  // No viscosity perturbation: an affine map, no second derivative.
  const auto none = tangent.RunSecondOrder(times, runU, runV, {}, &trajectory);
  for (const auto& snap : none) EXPECT_EQ(MaxAbs(snap.seaLevelChange), 0.0);

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
  const auto iceLoad = Ramp(load, times.back());
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
      SeaLevelEvolution3D evolution(
          op, scene.solver, scene.state,
          {.scheme = scheme, .rotation = RotationalFeedback::None,
           .rtol = 1e-12, .coupled_rtol = 1e-12});
      return Objective(scene, g, evolution.Run(times, iceLoad).back().seaLevelChange);
    };

    Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                      {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution3D evolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
    TangentPerturbation3D direction;
    direction.material.kappa = UniformOnCoupling(op.Coupling(), scene.material, d.kappa);
    direction.material.mu = UniformOnCoupling(op.Coupling(), scene.material, d.mu);
    const auto run = TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                       .rtol = 1e-12,
                                                       .coupled_rtol = 1e-12})
                         .Run(times, direction, &trajectory);
    const auto predicted = Objective(scene, g, run.back().seaLevelChange);

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

TEST(TangentLinear3D, ZeroLateralMaterialMatchesThe1DRun) {
  // With delta = 0 the 3D material tangent along an angle-uniform
  // direction is the 1D one through the grid pipeline (the effective
  // system's variation by the pointwise product against the 1D assembled
  // variation operator), both schemes; and a lateral direction with the
  // same angular mean is not the same run.
  auto scene = Scene::Make(1e21);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.0 * n * scene.Kyr());
  const auto load = scene.MeltLoad(-0.1);
  const auto iceLoad = Ramp(load, times.back());
  const auto d = SmoothMaterialDirection(scene);
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
    TangentPerturbation d1;
    d1.material = d;
    const auto run1 = TangentLinearSolver(evolution1, {.rtol = 1e-12})
                          .Run(times, d1, &trajectory1);

    Viscoelastic3D op3(scene.rheology, kLmax,
                       [](double, double, double) { return 0.0; },
                       {.memory_points = MemoryPointSet::Gll});
    SeaLevelEvolution3D evolution3(
        op3, scene.solver, scene.state,
        {.scheme = scheme, .rotation = RotationalFeedback::None,
         .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto trajectory3 =
        ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, iceLoad));
    const auto& coupling = op3.Coupling();
    TangentPerturbation3D d3;
    d3.material.kappa = UniformOnCoupling(coupling, scene.material, d.kappa);
    d3.material.mu = UniformOnCoupling(coupling, scene.material, d.mu);
    const auto run3 = TangentLinearSolver3D(evolution3, {.rtol = 1e-12,
                                                         .coupled_rtol = 1e-12})
                          .Run(times, d3, &trajectory3);
    ASSERT_EQ(run3.size(), times.size());
    // the ramp starts from nothing: no start response at node 0
    EXPECT_EQ(MaxAbs(run1[0].seaLevelChange), 0.0);
    EXPECT_EQ(MaxAbs(run3[0].seaLevelChange), 0.0);
    for (auto n = std::size_t{1}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(run3[n].seaLevelChange, run1[n].seaLevelChange),
                1e-7 * MaxAbs(run1[n].seaLevelChange))
          << "node " << n << " scheme " << static_cast<int>(scheme);
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
    EXPECT_GT(MaxDiff(runL.back().seaLevelChange, run3.back().seaLevelChange),
              1e-3 * MaxAbs(run3.back().seaLevelChange));
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
  const auto iceLoad = Ramp(load, times.back());
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
    std::unique_ptr<SeaLevelEvolution3D> evolution;
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
    m.evolution = std::make_unique<SeaLevelEvolution3D>(
        *m.op, scene.solver, scene.state,
        SeaLevelEvolution3D::Options{.rotation = RotationalFeedback::None,
                                     .rtol = 1e-12,
                                     .coupled_rtol = 1e-12,
                                     .record_stride = 1});
    m.trajectory = ForwardTrajectory3D::FromSnapshots(
        m.evolution->Run(times, [&](double t) {
          auto out = iceLoad(t);
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
  const auto predicted = Objective(scene, g, second.back().seaLevelChange);

  auto firstOrderAt = [&](double eps) {
    const auto m = moved(eps);
    TangentLinearSolver3D along(*m.evolution, {.method = AdjointMethod::Discrete,
                                               .rtol = 1e-12,
                                               .coupled_rtol = 1e-12});
    return Objective(scene, g,
                     along.Run(times, directionV(*m.op), &m.trajectory).back().seaLevelChange);
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
    EXPECT_LT(MaxDiff(swapped[n].seaLevelChange, second[n].seaLevelChange),
              1e-9 * MaxAbs(second[n].seaLevelChange) + 1e-300)
        << "node " << n;
  }
}
