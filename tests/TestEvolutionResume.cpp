// Checkpointed evolution: a run hands out its recurrence at the asked
// stride, and a fresh evolution resumed from a mid-run checkpoint
// reproduces the uninterrupted history node for node, bitwise on one
// thread, which checks the checkpoint bookkeeping.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace {

using namespace sl3dpp;

constexpr auto kLmax = 8;

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;

  static Scene Make() {
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
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator op(rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});
    Fingerprint fingerprint(kLmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model), std::move(material), std::move(rheology),
                 std::move(solver), std::move(state)};
  }
};

TEST(EvolutionResume, AResumedRunIsTheUninterruptedOne) {
  const auto scene = Scene::Make();
  const auto& grid = scene.solver.Grid();
  ViscoelasticOperator op(scene.rheology, kLmax);

  // a smooth cap melting linearly over the window
  const auto mask = grid.FromFunction([](double theta, double phi) {
    const auto c = std::cos(theta) * 0.5 +
                   std::sin(theta) * std::sqrt(0.75) * std::cos(phi);
    return std::exp(-8.0 * (1.0 - c));
  });
  auto times = std::vector<double>{};
  for (auto n = 0; n < 9; ++n) times.push_back(0.02 * n);
  auto directLoad = [&](double t) {
    auto load = std::vector<double>(mask.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = -1e-4 * (t / times.back()) * mask[i];
    }
    return load;
  };

  auto checkpoints = std::map<std::size_t, Evolution::Checkpoint>{};
  auto options = Evolution::Options{};
  options.checkpoint_stride = 3;
  options.on_checkpoint = [&](std::size_t node,
                              const Evolution::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  Evolution whole(op, scene.solver, scene.state, options);
  const auto full = whole.Run(times, directLoad);
  ASSERT_EQ(full.size(), times.size());
  // the stride hands out nodes 0, 3, 6 and the last
  ASSERT_TRUE(checkpoints.contains(0) && checkpoints.contains(3) &&
              checkpoints.contains(6) && checkpoints.contains(8));
  EXPECT_FALSE(checkpoints.contains(4));
  EXPECT_EQ(checkpoints.at(3).t, times[3]);

  // a fresh evolution, resumed from node 3, marches the same history
  Evolution resumed(op, scene.solver, scene.state, {});
  const auto tail = resumed.Resume(
      checkpoints.at(3),
      std::vector<double>(times.begin() + 3, times.end()), directLoad);
  ASSERT_EQ(tail.size(), times.size() - 4);
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    const auto& a = full[4 + k];
    const auto& b = tail[k];
    EXPECT_EQ(a.t, b.t);
    EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << "node " << 4 + k;
    EXPECT_EQ(a.displacement, b.displacement) << "node " << 4 + k;
    EXPECT_EQ(a.potential, b.potential) << "node " << 4 + k;
    EXPECT_EQ(a.angularVelocity, b.angularVelocity) << "node " << 4 + k;
  }

  // a resumed grid must start at the checkpoint's own node
  EXPECT_THROW(static_cast<void>(resumed.Resume(
                   checkpoints.at(3),
                   std::vector<double>(times.begin() + 4, times.end()),
                   directLoad)),
               std::invalid_argument);
}

TEST(EvolutionResume, AResumedRunUnderInteriorSourcesIsTheUninterruptedOne) {
  // Under a point source the checkpoint carries the toroidal systems'
  // state beside the spheroidal recurrence and its strain, and the
  // resumed run is the uninterrupted one field for field, the toroidal
  // potential among them, with an ocean and a load beside the source.
  const auto scene = Scene::Make();
  const auto& grid = scene.solver.Grid();
  const auto& sle = scene.solver.SolverScales();
  ViscoelasticOperator op(scene.rheology, kLmax);
  const auto mask = grid.FromFunction([](double theta, double phi) {
    const auto c = std::cos(theta) * 0.5 +
                   std::sin(theta) * std::sqrt(0.75) * std::cos(phi);
    return std::exp(-8.0 * (1.0 - c));
  });
  auto times = std::vector<double>{};
  for (auto n = 0; n < 9; ++n) times.push_back(0.02 * n);
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  Evolution::Forcing forcing;
  forcing.directLoad = [&](double t) {
    auto load = std::vector<double>(mask.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = -1e-4 * (t / times.back()) * mask[i];
    }
    return load;
  };
  forcing.sources = [&](double t) {
    const auto s = unit * (0.3 + t / times.back());
    return std::vector<PointSource>{
        {.at = {.radius = surface - 400e3 / sle.Length(),
                .colatitude = 1.1,
                .longitude = 4.0},
         .moment = {.rr = 0.9 * s, .tt = -0.2 * s, .pp = -0.7 * s,
                    .rt = 0.3 * s, .rp = -0.7 * s, .tp = 0.6 * s}}};
  };
  auto checkpoints = std::map<std::size_t, Evolution::Checkpoint>{};
  auto options = Evolution::Options{};
  options.checkpoint_stride = 3;
  options.on_checkpoint = [&](std::size_t node, const Evolution::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  Evolution whole(op, scene.solver, scene.state, options);
  const auto full = whole.Run(times, forcing);
  ASSERT_TRUE(checkpoints.contains(3));
  ASSERT_TRUE(checkpoints.at(3).toroidal.has_value());

  Evolution resumed(op, scene.solver, scene.state, {});
  const auto tail = resumed.Resume(
      checkpoints.at(3), std::vector<double>(times.begin() + 3, times.end()),
      forcing);
  ASSERT_EQ(tail.size(), times.size() - 4);
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    const auto& a = full[4 + k];
    const auto& b = tail[k];
    EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << "node " << 4 + k;
    EXPECT_EQ(a.displacement, b.displacement) << "node " << 4 + k;
    EXPECT_EQ(a.tangentialPotential, b.tangentialPotential) << "node " << 4 + k;
    EXPECT_EQ(a.toroidalPotential, b.toroidalPotential) << "node " << 4 + k;
    EXPECT_EQ(a.angularVelocity, b.angularVelocity) << "node " << 4 + k;
    EXPECT_FALSE(a.toroidalPotential.empty());
  }
  // a checkpoint of a run without sources does not start one with them
  auto bare = checkpoints.at(3);
  bare.toroidal.reset();
  EXPECT_THROW(static_cast<void>(resumed.Resume(
                   bare, std::vector<double>(times.begin() + 3, times.end()),
                   forcing)),
               std::invalid_argument);
}

}  // namespace
