// The coupled 3D evolution: the combined (load + rotation + lateral)
// within-step fixed point. The zero-lateral gate: with delta = 0, the
// coupled history must equal the 1D coupled evolution's.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 8;

struct Setup {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;
  std::vector<double> iceNd;

  static Setup Make() {
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
    return Setup{std::move(model),  std::move(material), std::move(rheology),
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
};

}  // namespace

TEST(Stage4Gate, ZeroLateralCoupledEvolutionMatchesThe1DLoop) {
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 4 * kyr};
  auto load = setup.MeltLoad(-0.1);
  auto iceLoad = [&](double) { return load; };

  // The 1D coupled loop.
  ViscoelasticOperator op1(setup.rheology, kLmax);
  SeaLevelEvolution evolution1(op1, setup.solver, setup.state, {.rtol = 1e-11});
  const auto reference = evolution1.Run(times, iceLoad);

  // The 3D coupled loop with no lateral perturbation.
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  SeaLevelEvolution3D evolution3(op3, setup.solver, setup.state,
                                 {.rtol = 1e-11, .coupled_rtol = 1e-10});
  const auto coupled = evolution3.Run(times, iceLoad);

  ASSERT_EQ(coupled.size(), reference.size());
  for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
    EXPECT_TRUE(coupled[node].converged);
    if (node > 0) EXPECT_LE(coupled[node].coupledIterations, 4);
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < reference[node].seaLevelChange.size();
         ++i) {
      scale = std::max(scale, std::abs(reference[node].seaLevelChange[i]));
      diff = std::max(diff, std::abs(coupled[node].seaLevelChange[i] -
                                     reference[node].seaLevelChange[i]));
    }
    EXPECT_LT(diff, 1e-8 * scale) << "node " << node;
    for (auto i = 0; i < 3; ++i) {
      EXPECT_NEAR(coupled[node].angularVelocity[i],
                  reference[node].angularVelocity[i],
                  1e-8 * std::abs(reference[node].angularVelocity[1]) + 1e-30)
          << "node " << node << " omega " << i;
    }
  }
}

TEST(Stage4, CoupledLateralEvolutionRunsAndConservesMass) {
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto& grid = setup.solver.Grid();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3 * kyr};
  auto load = setup.MeltLoad(-0.1);
  auto iceLoad = [&](double) { return load; };

  // A strong lateral pattern: one decade peak to peak.
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  SeaLevelEvolution3D evolution(op, setup.solver, setup.state, {});
  const auto run = evolution.Run(times, iceLoad);

  // The zero-lateral twin, for comparison of the physics.
  Viscoelastic3D flat(setup.rheology, kLmax,
                      [](double, double, double) { return 0.0; });
  SeaLevelEvolution3D twin(flat, setup.solver, setup.state, {});
  const auto uniform = twin.Run(times, iceLoad);

  auto maxRelDiff = 0.0;
  for (auto node = std::size_t{1}; node < run.size(); ++node) {
    EXPECT_TRUE(run[node].converged) << "node " << node;

    // Exact ocean mass conservation per node, lateral viscosity or not.
    auto gain = std::vector<double>(grid.FieldSize());
    for (auto i = std::size_t{0}; i < gain.size(); ++i) {
      gain[i] = setup.solver.WaterDensity() * setup.state.oceanFunction[i] *
                run[node].seaLevelChange[i];
    }
    const auto a = setup.solver.Radius();
    const auto gained = a * a * grid.Integrate(gain);
    const auto removed = -a * a * grid.Integrate(load);
    EXPECT_NEAR(gained, removed, 1e-7 * std::abs(removed)) << "node " << node;

    // The lateral viscosity changes the physics measurably.
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < run[node].seaLevelChange.size(); ++i) {
      scale = std::max(scale, std::abs(uniform[node].seaLevelChange[i]));
      diff = std::max(diff, std::abs(run[node].seaLevelChange[i] -
                                     uniform[node].seaLevelChange[i]));
    }
    maxRelDiff = std::max(maxRelDiff, diff / scale);
  }
  EXPECT_GT(maxRelDiff, 1e-3);
  EXPECT_LT(maxRelDiff, 0.5);
}
