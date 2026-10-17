// Internal variables at the Gauss points (plan.md §5.5, performance chunk
// 7b): 20% fewer memory points than at the GLL nodes; the implicit step
// stays an exact elimination of the new internal variables (the effective
// operator carries the relaxation term at the points, MemoryPoints.hpp);
// and the forward agrees with the GLL discretisation to its own accuracy,
// converging with it under mesh refinement.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
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

TEST(GaussMemory, AdjointRefusesForNow) {
  // The Gauss-point kernels are chunk 7b-ii; until then the 1D adjoint
  // says so rather than computing with the wrong point set.
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
  const auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                      toNd(fingerprint.SeaLevel()));
  const auto op = Operator(body.rheology, MemoryPointSet::Gauss);
  SeaLevelEvolution evolution(op, solver, state);
  EXPECT_THROW(AdjointSolver(evolution, {}), std::invalid_argument);
}
