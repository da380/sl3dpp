// The coupled GIA evolution: the elastic-rheology limit against the
// fingerprint solver (machine-tight, fully self-contained), qualitative
// relaxation, and the pyslfp coupled gate.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 16;

struct Setup {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;
  std::vector<double> iceNd;  // the state's ice thickness, solver units

  static Setup Make(bool viscous) {
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

    Rheology rheology = [&] {
      if (!viscous) {
        // A branch that never relaxes: the elastic limit of the coupling.
        Rheology::Branch zero;
        zero.mu.assign(material.Rho().size(), 0.0);
        zero.itau.assign(material.Rho().size(), 0.0);
        return Rheology::Create(material, {std::move(zero)}).value();
      }
      auto viscosity = std::map<std::size_t, RadialValue>{};
      const auto b = model.DimensionlessView().Boundaries();
      for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
        if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
            b[i + 1] <= 6.1511e6 / scales.Length()) {
          viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
        }
      }
      return Rheology::Maxwell(material, viscosity).value();
    }();

    ViscoelasticOperator op(rheology, kLmax);
    SeaLevelSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});

    // The analytic test state, built through the fingerprint helper and
    // rebuilt in the solver's units.
    Fingerprint fingerprint(kLmax, op.ElasticLoveNumbers());
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

  /// The grounded load of a melt fraction, solver units.
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

TEST(Evolution, ElasticRheologyReducesToTheFingerprint) {
  auto setup = Setup::Make(false);
  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution evolution(op, setup.solver, setup.state, {.rtol = 1e-12});

  const auto tauScale = 1e10 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, tauScale, 3.0 * tauScale};
  auto load = setup.MeltLoad(-0.1);
  auto snapshots = evolution.Run(times, [&](double) { return load; });
  ASSERT_EQ(snapshots.size(), 3u);

  // With nothing relaxing, every node is the same elastic solve.
  auto reference = setup.solver.Solve(
      setup.state, load, RotationalFeedback::Full, {.rtol = 1e-12});
  auto scale = 0.0;
  for (auto v : reference.seaLevelChange) scale = std::max(scale, std::abs(v));
  for (const auto& snapshot : snapshots) {
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < reference.seaLevelChange.size(); ++i) {
      diff = std::max(diff, std::abs(snapshot.seaLevelChange[i] -
                                     reference.seaLevelChange[i]));
    }
    EXPECT_LT(diff, 1e-9 * scale) << "t = " << snapshot.t;
    for (auto i = 0; i < 3; ++i) {
      EXPECT_NEAR(snapshot.angularVelocity[i], reference.angularVelocity[i],
                  1e-9 * std::abs(reference.angularVelocity[1]) + 1e-30);
    }
  }
}

TEST(Evolution, ViscousMantleKeepsRelaxing) {
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution evolution(op, setup.solver, setup.state, {});
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto i = 0; i <= 10; ++i) times.push_back(i * kyr);

  // A load switched on at t = 0 and held.
  auto load = setup.MeltLoad(-0.1);
  auto snapshots = evolution.Run(times, [&](double) { return load; });

  // Viscous flow keeps amplifying the response after the elastic step:
  // the peak sea-level change grows monotonically over the history.
  auto peaks = std::vector<double>{};
  for (const auto& snapshot : snapshots) {
    auto peak = 0.0;
    for (auto v : snapshot.seaLevelChange) peak = std::max(peak, std::abs(v));
    peaks.push_back(peak);
  }
  for (auto i = std::size_t{1}; i < peaks.size(); ++i) {
    EXPECT_GT(peaks[i], peaks[i - 1] * (1.0 - 1e-9)) << "node " << i;
  }
  EXPECT_GT(peaks.back(), 1.2 * peaks.front());
}

TEST(EvolutionGate, CoupledEvolutionMatchesPyslfp) {
  // The reference is pyslfp on its GLQ grid, which coincides with ours at
  // this lmax, so the comparison isolates the implementation; the sharp
  // coastline makes the solution itself grid-determined at the ~19%
  // level (the fixture header records pyslfp's own DH-vs-GLQ spread).
  auto setup = Setup::Make(true);
  ASSERT_EQ(setup.solver.Grid().NumberOfLongitudes(), 2 * kLmax + 1)
      << "the GSHTrans grid no longer coincides with pyshtools GLQ here";
  ViscoelasticOperator op(setup.rheology, kLmax);
  SeaLevelEvolution evolution(op, setup.solver, setup.state, {.rtol = 1e-11});
  const auto& grid = setup.solver.Grid();
  const auto& scales = setup.solver.SolverScales();

  const auto tFinal = 5000.0 * 3.15576e7;  // s
  std::vector<double> times, timesSi;
  for (auto i = 0; i <= 10; ++i) {
    timesSi.push_back(i * tFinal / 10.0);
    times.push_back(timesSi.back() / scales.Time());
  }
  auto iceLoad = [&](double t) {
    return setup.MeltLoad(-0.1 * (t / times.back()));
  };
  auto snapshots = evolution.Run(times, iceLoad);

  // Parse the fixture and compare, SI, in coefficient space.
  std::ifstream file(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                     "pyslfp_evolution.txt");
  ASSERT_TRUE(file.is_open());
  const auto length = scales.Length();
  const auto frequency = scales.Factor(Dimensions{.time = -1});
  auto node = -1;
  auto ours = std::vector<std::complex<double>>{};
  auto scale = 0.0, worst = 0.0;
  constexpr auto tol = 1e-5;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (line.starts_with("node")) {
      auto head = std::string{};
      auto t = 0.0;
      auto omega = std::array<double, 3>{};
      row >> head >> node >> t >> omega[0] >> omega[1] >> omega[2];
      ASSERT_LT(static_cast<std::size_t>(node), snapshots.size());
      EXPECT_NEAR(timesSi[node], t, 1e-6 * std::max(t, 1.0));
      ours = grid.Expand(snapshots[node].seaLevelChange);
      for (auto& v : ours) v *= length;
      if (node > 0) {
        const double omegaScale = std::max(
            {std::abs(omega[0]), std::abs(omega[1]), std::abs(omega[2])});
        for (auto i = 0; i < 3; ++i) {
          EXPECT_NEAR(snapshots[node].angularVelocity[i] * frequency, omega[i],
                      tol * omegaScale)
              << "node " << node << " omega " << i;
        }
      }
      continue;
    }
    auto l = 0, m = 0;
    auto a = 0.0, b = 0.0;
    row >> l >> m >> a >> b;
    const auto reference = ComplexFromRealPair({a, b}, m);
    scale = std::max(scale, std::abs(reference));
    if (node > 0) {
      worst =
          std::max(worst, std::abs(ours[SurfaceGrid::Index(l, m)] - reference));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, tol * scale);
}
