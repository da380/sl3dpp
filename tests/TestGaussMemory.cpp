// Internal variables at the Gauss points (plan.md §5.5, performance chunk
// 7b): 20% fewer memory points than at the GLL nodes; the implicit step
// stays an exact elimination of the new internal variables (the effective
// operator carries the relaxation term at the points, MemoryPoints.hpp);
// the forward agrees with the GLL discretisation to its own accuracy,
// converging with it under mesh refinement; and the adjoint sweeps, which
// accumulate the itau kernels at the points and carry them to the nodal
// itau through the interpolation's transpose, keep the discrete gradient
// identity and agree with the GLL kernels.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

constexpr int kLmax = 8;

struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Body Make(RadialMesh::Options meshOptions) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, meshOptions).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    std::map<std::size_t, RadialValue> viscosity;
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology)};
  }
};

LoadCallable UnitLoad() {
  return [](int, double) { return std::vector<double>{1.0}; };
}

ViscoelasticOperator Operator(const Rheology& rheology, MemoryPointSet set) {
  return ViscoelasticOperator(rheology, kLmax, {.memory_points = set});
}

// Surface U per degree after `steps` trapezoid steps of the unit load.
std::vector<double> SurfaceAfter(const ViscoelasticOperator& op, int steps) {
  const auto dt = op.MinRelaxationTime();
  auto state = op.ZeroState();
  std::map<int, std::vector<double>> solutions;
  for (auto i = 0; i < steps; ++i) {
    auto [next, sols] =
        op.Step(state, dt, Scheme::ExponentialTrapezoid, UnitLoad());
    state = std::move(next);
    solutions = std::move(sols);
  }
  auto out = std::vector<double>{};
  for (const auto& [l, u] : op.Surface(solutions, Component::U)) {
    out.push_back(u[0]);
  }
  return out;
}

double RelativeGap(const std::vector<double>& a, const std::vector<double>& b) {
  auto worst = 0.0;
  for (auto i = std::size_t{1}; i < a.size(); ++i) {  // l >= 1
    worst = std::max(worst, std::abs(a[i] - b[i]) / std::abs(b[i]));
  }
  return worst;
}

// A coupled scene for the adjoint tests: the analytic present-day state
// on the solver grid, Gauss or GLL memory chosen per operator.
struct Scene {
  Body body;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;

  static Scene Make() {
    auto body = Body::Make({.lmax = kLmax});
    const auto elastic = ViscoelasticOperator(body.rheology, kLmax);
    auto solver = SeaLevelSolver(
        SurfaceGrid(kLmax), elastic.ElasticLoveNumbers(), SeaLevelParameters{});
    auto fingerprint = Fingerprint(kLmax, elastic.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene{std::move(body), std::move(solver), std::move(state)};
  }

  /// A smooth random band-limited field of the given amplitude.
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng,
                                           double amplitude) const {
    const auto& grid = solver.Grid();
    auto normal = std::normal_distribution<double>{};
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

  /// The objective the measurements define on a forward run.
  [[nodiscard]] double Objective(
      const std::vector<SeaLevelEvolution::Snapshot>& snapshots,
      const std::map<std::size_t, Measurement>& measurements) const {
    auto total = 0.0;
    for (const auto& [node, m] : measurements) {
      const auto& snap = snapshots[node];
      if (!m.seaLevelDual.empty()) {
        total += Surface(m.seaLevelDual, snap.seaLevelChange);
      }
      if (!m.displacementDual.empty()) {
        total += Surface(m.displacementDual, snap.displacement);
      }
      if (!m.potentialDual.empty()) {
        total += Surface(m.potentialDual, snap.potential);
      }
      for (auto j = 0; j < 3; ++j) {
        total += m.angularVelocityDual[j] * snap.angularVelocity[j];
      }
    }
    return total;
  }
};

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

TEST(GaussMemory, FewerPointsAndAnExactElimination) {
  auto body = Body::Make({.lmax = kLmax});
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    const auto op = Operator(body.rheology, set);
    const auto ngll = body.material.Mesh().NGll();

    // The memory state holds ngll - 1 points per element at Gauss, ngll
    // at GLL: 20% fewer at the default ngll = 5.
    const auto zero = op.ZeroState();
    const auto gll = Operator(body.rheology, MemoryPointSet::Gll).ZeroState();
    for (const auto& [l, m] : zero.m) {
      const auto ratio = static_cast<double>(m.size()) /
                         static_cast<double>(gll.m.at(l).size());
      EXPECT_DOUBLE_EQ(ratio,
                       set == MemoryPointSet::Gll ? 1.0 : (ngll - 1.0) / ngll);
    }

    // The step eliminates the new internal variables exactly, so the end
    // strain is what one elastic solve under the new memory gives (what
    // the adjoints regenerate it by).
    const auto dt = op.MinRelaxationTime();
    auto state = op.ZeroState();
    for (auto i = 0; i < 3; ++i) {
      state =
          op.Step(state, dt, Scheme::ExponentialTrapezoid, UnitLoad()).first;
    }
    auto bare = state;
    bare.Invalidate();
    const auto regenerated = op.Strain(op.SolveElastic(bare, UnitLoad()));
    auto scale = 0.0, worst = 0.0;
    for (const auto& [l, d] : state.strain) {
      const auto& r = regenerated.at(l);
      for (auto i = std::size_t{0}; i < d.size(); ++i) {
        scale = std::max(scale, std::abs(d[i]));
        worst = std::max(worst, std::abs(d[i] - r[i]));
      }
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-11 * scale)
        << (set == MemoryPointSet::Gll ? "GLL" : "Gauss");
  }
}

TEST(GaussMemory, AgreesWithGllAndConverges) {
  // Two discretisations of the same internal variables: their surface
  // responses agree to discretisation accuracy, and the gap falls under
  // mesh refinement.
  auto gap = [](double drmax) {
    auto body = Body::Make({.drmax = drmax});
    const auto gll =
        SurfaceAfter(Operator(body.rheology, MemoryPointSet::Gll), 4);
    const auto gauss =
        SurfaceAfter(Operator(body.rheology, MemoryPointSet::Gauss), 4);
    return RelativeGap(gauss, gll);
  };
  const auto coarse = gap(600e3);
  const auto fine = gap(300e3);
  EXPECT_GT(coarse, 0.0);
  EXPECT_LT(coarse, 1e-6);
  EXPECT_LT(fine, 0.25 * coarse);
}

TEST(GaussMemory, CoupledSeaLevelRunAgrees) {
  // The coupled sea level loop with Gauss-point memory: its effective
  // Love numbers come through the relaxation term. Against the GLL run,
  // agreement at discretisation level.
  auto body = Body::Make({.lmax = kLmax});
  const ViscoelasticOperator elastic(body.rheology, kLmax);
  SeaLevelSolver solver(SurfaceGrid(kLmax), elastic.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  Fingerprint fingerprint(kLmax, elastic.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  const auto length = solver.SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  auto load = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i) {
    load[i] = -0.1 * solver.IceDensity() * ice[i];
  }
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  auto iceLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = (t / times.back()) * load[i];
    }
    return out;
  };

  std::vector<std::vector<double>> finals;
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    const auto op = Operator(body.rheology, set);
    SeaLevelEvolution evolution(op, solver, state, {.rtol = 1e-12});
    const auto snapshots = evolution.Run(times, iceLoad);
    finals.push_back(snapshots.back().seaLevelChange);
  }
  auto scale = 0.0, worst = 0.0;
  for (auto i = std::size_t{0}; i < finals[0].size(); ++i) {
    scale = std::max(scale, std::abs(finals[0][i]));
    worst = std::max(worst, std::abs(finals[1][i] - finals[0][i]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-9 * scale);
}

TEST(GaussMemory, DiscreteGradientIdentity) {
  // The exact transpose with the internal variables at the Gauss points:
  // delta J = sum_n <loadDuals[n], delta f_n> to the solvers' tolerance,
  // for a full measurement set, both schemes.
  const auto scene = Scene::Make();
  auto rng = std::mt19937(7);
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(scene.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = scene.Smooth(rng, 0.3);
  measurements[3].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[3].displacementDual = scene.Smooth(rng, -0.5);
  measurements[5].displacementDual = scene.Smooth(rng, 0.7);
  measurements[5].potentialDual = scene.Smooth(rng, 0.4);
  measurements[5].angularVelocityDual = {0.2, -0.4, 0.8};

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto op = Operator(scene.body.rheology, MemoryPointSet::Gauss);
    auto evolution = SeaLevelEvolution(op, scene.solver, scene.state,
                                       {.scheme = scheme, .rtol = 1e-12});
    const auto snapshots = evolution.Run(times, History(times, loads));
    const auto exact = scene.Objective(snapshots, measurements);
    const auto adjoint = AdjointSolver(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels = adjoint.IceHistoryKernels(times, measurements);
    auto predicted = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      predicted += scene.Surface(kernels.loadDuals[n], loads[n]);
    }
    EXPECT_NEAR(predicted, exact, 1e-8 * std::abs(exact))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(GaussMemory, ItauKernelAgainstFiniteDifferences) {
  // The discrete itau kernel accumulated at the Gauss points and carried
  // to the nodal itau by the interpolation's transpose: the FD residual
  // against a smooth nodal perturbation falls at second order and
  // Richardson extrapolation removes it. (Under backward Euler the
  // directional derivative cancels to ~4e-4 of J, so the check is
  // against the residual removed, not the derivative's size.)
  const auto scene = Scene::Make();
  auto rng = std::mt19937(7);
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(scene.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = scene.Smooth(rng, 1.0);

  const auto& material = scene.body.material;
  const auto& base = scene.body.rheology.BranchAt(0);
  const auto& mesh = material.Mesh();
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
    const auto rheology =
        Rheology::Create(material, {std::move(branch)}).value();
    const auto op = Operator(rheology, MemoryPointSet::Gauss);
    auto evolution = SeaLevelEvolution(op, scene.solver, scene.state,
                                       {.scheme = scheme, .rtol = 1e-12});
    return scene.Objective(evolution.Run(times, History(times, loads)),
                           measurements);
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto op = Operator(scene.body.rheology, MemoryPointSet::Gauss);
    auto evolution = SeaLevelEvolution(
        op, scene.solver, scene.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
    const auto adjoint = AdjointSolver(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &trajectory);
    ASSERT_EQ(kernels.itauKernel[0].size(), ditau.size());
    auto predicted = 0.0;
    for (auto i = std::size_t{0}; i < ditau.size(); ++i) {
      predicted += kernels.itauKernel[0][i] * ditau[i];
    }
    auto fd = std::map<double, double>{};
    for (auto eps : {0.02, 0.01}) {
      fd[eps] =
          (functional(eps, scheme) - functional(-eps, scheme)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.02] - predicted);
    const auto r2 = std::abs(fd[0.01] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
    EXPECT_LT(std::abs(richardson - predicted), 0.01 * r2)
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(GaussMemory, KernelsAgreeWithGllAndContinuousConverges) {
  // The nodal itau kernels of the two point sets agree at discretisation
  // level, discrete and continuous, and the continuous kernel at the Gauss
  // points converges to the discrete one under step refinement, as at the
  // GLL nodes (each run draws its own load history, so the comparison
  // waits for the asymptotic regime; at GLL and Gauss alike the gap is
  // 0.19, 1.0, 0.22, 0.033 at 6, 11, 21, 41 nodes).
  const auto scene = Scene::Make();
  const auto& scales = scene.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto kernels = [&](MemoryPointSet set, AdjointMethod method,
                     std::size_t npts) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
    auto local = std::mt19937(23);
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      loads.push_back(scene.Smooth(local, 100.0 / sigma));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts - 1].seaLevelDual = scene.Smooth(local, 1.0);
    const auto op = Operator(scene.body.rheology, set);
    auto evolution = SeaLevelEvolution(op, scene.solver, scene.state,
                                       {.rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
    const auto adjoint =
        AdjointSolver(evolution, {.method = method, .rtol = 1e-12});
    return adjoint.IceHistoryKernels(times, measurements, &trajectory)
        .itauKernel[0];
  };
  auto relative = [](const std::vector<double>& a,
                     const std::vector<double>& b) {
    auto diff = 0.0, norm = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      diff += (a[i] - b[i]) * (a[i] - b[i]);
      norm += b[i] * b[i];
    }
    return std::sqrt(diff / norm);
  };

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto gap = relative(kernels(MemoryPointSet::Gauss, method, 21),
                              kernels(MemoryPointSet::Gll, method, 21));
    EXPECT_LT(gap, 1e-5) << "method " << static_cast<int>(method);
  }

  auto commute = [&](std::size_t npts) {
    return relative(
        kernels(MemoryPointSet::Gauss, AdjointMethod::Continuous, npts),
        kernels(MemoryPointSet::Gauss, AdjointMethod::Discrete, npts));
  };
  const auto coarse = commute(21);
  const auto fine = commute(41);
  EXPECT_LT(coarse, 0.3);
  EXPECT_LT(fine, coarse / 3.0);
}
