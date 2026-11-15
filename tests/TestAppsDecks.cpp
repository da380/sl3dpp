// planetmodel's viscoelastic decks in sl3d (`model.format`: "prony",
// "burgers") and the Burgers regions of a Maxwell deck (`model.burgers`):
// a one-branch prony deck is its Maxwell deck's body; one Burgers body
// three ways (a burgers deck, a two-branch prony deck of planetmodel's
// branches, a Maxwell deck with `model.burgers`) gives one run; the
// regions are the post-seismic example's Burgers twin; the refusals; and
// the viscosity kernels of a Burgers body, per branch, against finite
// differences. The decks are written by planetmodel's own writer
// (planetmodel::WriteDeck) on a lidded homogeneous ball. A build whose
// planetmodel has no prony or burgers deck refuses the formats.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <planetmodel/deck.hpp>
#include <planetmodel/error.hpp>
#include <planetmodel/json.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"

namespace {

namespace fs = std::filesystem;
namespace pm = planetmodel;
using namespace sl3dpp;

constexpr auto kNan = std::numeric_limits<double>::quiet_NaN();

class DecksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ =
        fs::temp_directory_path() /
        ("sl3dpp_test_decks_" +
         std::string{
             ::testing::UnitTest::GetInstance()->current_test_info()->name()});
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path Write(const std::string& name, const std::string& text) const {
    const auto path = dir_ / name;
    auto out = std::ofstream(path);
    out << text;
    return path;
  }

  /// A config of the quasi-static problem on the run's own clock in
  /// years, from the event to 1000 yr, at lmax 8 without an ocean or
  /// rotational feedback, under a thrust 50 km deep, on the `model` block
  /// given, with `extra` members.
  fs::path Solve(const std::string& name, const std::string& model,
                 const std::string& extra = "") const {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static", "model": {},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [0, 2, 10, 50, 200, 1000]}},
      "forcing": {{"seismic": [{{"point": {{"latitude": 10, "longitude": 20,
          "depth_km": 50, "strike_deg": 30, "dip_deg": 40, "rake_deg": 80,
          "moment_nm": 1e21}}}}]}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}{}}})",
                                             model, name, extra));
  }

  fs::path dir_;
};

/// The largest departure of two files' fields, each over its largest
/// value (zero where both vanish), over every variable of the first.
double Departure(const pm::SurfaceFile& a, const pm::SurfaceFile& b) {
  auto worst = 0.0;
  for (const auto& [name, variable] : a.variables) {
    const auto& x = variable.values;
    const auto& y = b.variables.at(name).values;
    EXPECT_EQ(x.size(), y.size()) << name;
    auto diff = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      diff = std::max(diff, std::abs(x[i] - y[i]));
      scale = std::max(scale, std::abs(y[i]));
    }
    worst = std::max(worst, scale > 0.0 ? diff / scale : diff);
  }
  return worst;
}

/// Runs `run` on `config` and expects a planetmodel::Error whose message
/// names `member` and says `what`.
template <class Runner>
void ExpectRefused(Runner&& run, const fs::path& config,
                   const std::string& member, const std::string& what) {
  try {
    static_cast<void>(run(config));
    ADD_FAILURE() << config << " was not refused";
  } catch (const pm::Error& error) {
    const auto message = std::string{error.what()};
    EXPECT_NE(message.find(member), std::string::npos) << message;
    EXPECT_NE(message.find(what), std::string::npos) << message;
  }
}

/// The post-seismic deck of the examples.
fs::path PostseismicDeck() {
  return fs::path(SL3DPP_TEST_DATA_DIR) / ".." / ".." / "examples" / "data" /
         "prem8.postseismic.deck";
}


/// The lidded ball: a viscous interior under an elastic lid 100 km thick,
/// its elastic values homogeneous (rho 5500 kg m^-3, vp 10000 m/s, vs 5500
/// m/s: mu = 5500^3 Pa) and isotropic, two knots a layer; `interior` the
/// further columns, each its value in the interior and NaN in the lid.
pm::Deck LiddedBall(const std::vector<std::pair<std::string, double>>& interior,
                    std::vector<std::string> header) {
  const auto radius = std::vector<double>{0.0, 6271e3, 6271e3, 6371e3};
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{
      {"rho", std::vector<double>(4, 5500.0)},
      {"vpv", std::vector<double>(4, 10000.0)},
      {"vsv", std::vector<double>(4, 5500.0)},
      {"vph", std::vector<double>(4, 10000.0)},
      {"vsh", std::vector<double>(4, 5500.0)},
      {"eta", std::vector<double>(4, 1.0)}};
  for (const auto& [name, value] : interior) {
    columns.emplace_back(name, std::vector<double>{value, value, kNan, kNan});
  }
  return pm::Deck(radius, std::move(columns), std::move(header));
}

/// The ball's Burgers body: the Maxwell element's viscosity, the Kelvin
/// element's viscosity and its modulus fraction.
constexpr auto kViscosity = 1e21;
constexpr auto kKelvinViscosity = 1e20;
constexpr auto kKelvinFraction = 0.5;

/// That body's two Prony branches, the faster first, as planetmodel's
/// Python gives them (conversion.prony.burgers_terms of mu = 5500^3 Pa and
/// the three above, planetmodel at c9c288b): strengths and times (s).
constexpr auto kStrengths = std::array{0.6952833664712358, 0.3047166335287642};
constexpr auto kTimes = std::array{383295870.33074903, 18850363032.74964};

/// Writes the lidded ball as a Maxwell deck of viscosity `viscosity`.
void WriteMaxwell(const fs::path& path, double viscosity = kViscosity) {
  pm::WriteDeck(
      path,
      LiddedBall({{"viscosity", viscosity}}, pm::MaxwellHeader("lidded ball")),
      pm::MAXWELL());
}

/// Writes the lidded ball as a burgers deck, its viscosities scaled by
/// `scale`.
void WriteBurgers(const fs::path& path, double scale = 1.0) {
  pm::WriteDeck(path,
                LiddedBall({{"viscosity", scale * kViscosity},
                            {"kelvin_viscosity", scale * kKelvinViscosity},
                            {"kelvin_modulus_fraction", kKelvinFraction}},
                           pm::BurgersHeader("lidded ball")),
                pm::BURGERS());
}

/// Writes the lidded ball as a prony deck of the branches given.
void WriteProny(const fs::path& path,
                const std::vector<std::pair<double, double>>& branches) {
  auto interior = std::vector<std::pair<std::string, double>>{};
  for (auto k = std::size_t{0}; k < branches.size(); ++k) {
    interior.emplace_back(std::format("relaxation_strength_{}", k + 1),
                          branches[k].first);
    interior.emplace_back(std::format("relaxation_time_{}", k + 1),
                          branches[k].second);
  }
  pm::WriteDeck(
      path,
      LiddedBall(interior, pm::PronyHeader("lidded ball", branches.size())),
      pm::PRONY());
}

TEST_F(DecksTest, AOneBranchPronyDeckIsItsMaxwellDeck) {
  // The Maxwell deck and the prony deck of one branch of strength one and
  // time eta / mu (planetmodel's prony_deck of it) are the same body: their
  // runs agree to the eight digits the deck writes the time with. On the
  // post-seismic deck, whose moduli vary between knots, they are the same
  // body at the knots and differ between them at second order in the
  // knots' spacing, the prony deck's time geometric between the knots'
  // eta / mu where the Maxwell deck divides its geometric viscosity by the
  // cubic of the modulus.
  WriteMaxwell(dir_ / "ball.maxwell.deck");
  const auto mu = 5500.0 * 5500.0 * 5500.0;
  WriteProny(dir_ / "ball.prony.deck", {{1.0, kViscosity / mu}});
  static_cast<void>(app::RunSolve(Solve(
      "maxwell", R"({"deck": "ball.maxwell.deck", "format": "maxwell"})")));
  const auto summary = app::RunSolve(
      Solve("prony", R"({"deck": "ball.prony.deck", "format": "prony"})"));
  EXPECT_TRUE(summary.starts_with("1-D, viscoelastic")) << summary;
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto ball = Departure(read("prony"), read("maxwell"));
  RecordProperty("ball", std::format("{:.2e}", ball));
  EXPECT_LT(ball, 1e-8);
  // the post-seismic deck: its knots' eta / mu as the one branch's time
  const auto deck = pm::ReadDeck(PostseismicDeck(), pm::MAXWELL());
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
  for (const auto& name : pm::PRONY_COLUMNS)
    columns.emplace_back(name, deck.Column(name));
  auto strength = std::vector<double>{}, time = std::vector<double>{};
  for (auto i = std::size_t{0}; i < deck.NKnots(); ++i) {
    const auto eta = deck.Column("viscosity")[i];
    const auto shear =
        deck.Column("rho")[i] * std::pow(deck.Column("vsv")[i], 2);
    strength.push_back(std::isnan(eta) ? kNan : 1.0);
    time.push_back(std::isnan(eta) ? kNan : eta / shear);
  }
  columns.emplace_back("relaxation_strength_1", strength);
  columns.emplace_back("relaxation_time_1", time);
  pm::WriteDeck(
      dir_ / "postseismic.prony.deck",
      pm::Deck(deck.Radius(), std::move(columns),
               pm::PronyHeader("the post-seismic deck, one branch", 1)),
      pm::PRONY());
  static_cast<void>(app::RunSolve(
      Solve("postseismic_maxwell",
            std::format(R"({{"deck": "{}", "format": "maxwell"}})",
                        PostseismicDeck().string()))));
  static_cast<void>(app::RunSolve(
      Solve("postseismic_prony",
            R"({"deck": "postseismic.prony.deck", "format": "prony"})")));
  const auto knots =
      Departure(read("postseismic_prony"), read("postseismic_maxwell"));
  RecordProperty("postseismic", std::format("{:.2e}", knots));
  EXPECT_LT(knots, 1e-4);
}

TEST_F(DecksTest, OneBurgersBodyThreeWaysIsOneRun) {
  // The ball's interior a Burgers body (steady viscosity 1e21 Pa s, a
  // Kelvin element of 1e20 Pa s and half the shear modulus): as a burgers
  // deck, as a Maxwell deck with `model.burgers` over the interior's whole
  // depth, and as a prony deck of the body's two branches as planetmodel
  // computes them. The first two are one run to rounding (the deck's
  // numbers are exact in its text); the prony deck's to the eight digits
  // it writes its branches with. The Burgers body relaxes more than the
  // Maxwell body early, its transient the Kelvin element's.
  WriteMaxwell(dir_ / "ball.maxwell.deck");
  WriteBurgers(dir_ / "ball.burgers.deck");
  WriteProny(dir_ / "ball.prony.deck",
             {{kStrengths[0], kTimes[0]}, {kStrengths[1], kTimes[1]}});
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  static_cast<void>(app::RunSolve(
      Solve("regions", std::format(R"({{"deck": "ball.maxwell.deck",
        "burgers": [{{"depth_km": [100, 6371], "kelvin_viscosity": {:.17g},
                      "kelvin_modulus_fraction": {:.17g}}}]}})",
                                   kKelvinViscosity, kKelvinFraction))));
  static_cast<void>(app::RunSolve(Solve(
      "burgers", R"({"deck": "ball.burgers.deck", "format": "burgers"})")));
  static_cast<void>(app::RunSolve(
      Solve("prony", R"({"deck": "ball.prony.deck", "format": "prony"})")));
  static_cast<void>(
      app::RunSolve(Solve("maxwell", R"({"deck": "ball.maxwell.deck"})")));
  const auto regions = read("regions");
  const auto byDeck = Departure(read("burgers"), regions);
  const auto byBranches = Departure(read("prony"), regions);
  const auto fromMaxwell = Departure(read("maxwell"), regions);
  RecordProperty("burgers_deck", std::format("{:.2e}", byDeck));
  RecordProperty("prony_deck", std::format("{:.2e}", byBranches));
  RecordProperty("maxwell", std::format("{:.2e}", fromMaxwell));
  EXPECT_LT(byDeck, 1e-13);
  EXPECT_LT(byBranches, 1e-7);
  EXPECT_GT(fromMaxwell, 1e-2);
  EXPECT_EQ(
      std::get<std::string>(regions.meta.at("burgers")),
      "100 to 6371 km: kelvin_viscosity 1e+20 Pa s, kelvin_modulus_fraction "
      "0.5");
}

TEST_F(DecksTest, RegionsAreThePostseismicExamplesBurgersTwin) {
  // `model.burgers` over the post-seismic deck's asthenosphere (its viscous
  // layers from the lid's base at 50 km to 217 km), a Kelvin element of
  // the unrelaxed modulus and 1e18 Pa s, a tenth of the steady viscosity
  // there: the rheology the program builds is ExampleSeismic's Burgers twin
  // (Rheology::Burgers over the deck's viscous layers, the Kelvin element
  // on those above PREM's 220 km discontinuity), branch for branch at
  // every node, to rounding.
  const auto config =
      Solve("twin",
            std::format(R"({{"deck": "{}", "burgers": [{{"depth_km": [50, 217],
        "kelvin_viscosity": 1e18, "kelvin_modulus_fraction": 1}}]}})",
                        PostseismicDeck().string()));
  const auto top = pm::ReadJsonFile(config);
  const auto run = app::ReadSolveInput(config).run;
  const auto earth = app::BuildEarth(run, top);
  const auto rheology = app::RheologyOf(earth, top);
  // the example's own construction (ExampleSeismic.cpp, PostseismicEarth)
  const auto& model = earth.model;
  const auto nd = model.DimensionlessView();
  const auto b = model.Boundaries();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  auto kelvin = std::map<std::size_t, Rheology::Kelvin>{};
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.Has("viscosity", i)) continue;
    auto eta = [nd, i](double r) {
      return nd.EvaluateInLayer("viscosity", i, r);
    };
    viscosity[i] = eta;
    if (b[i] >= 6.151e6 - 1.0) {
      kelvin[i] = Rheology::Kelvin{
          .modulus_fraction = 1.0,
          .viscosity = [eta](double r) { return 0.1 * eta(r); }};
    }
  }
  EXPECT_EQ(kelvin.size(), 2u);
  EXPECT_EQ(earth.kelvin.size(), 2u);
  const auto twin =
      Rheology::Burgers(earth.material, viscosity, kelvin).value();
  ASSERT_EQ(rheology.BranchCount(), 2u);
  ASSERT_EQ(twin.BranchCount(), 2u);
  auto worst = 0.0;
  for (auto k = std::size_t{0}; k < 2; ++k) {
    const auto& got = rheology.BranchAt(k);
    const auto& want = twin.BranchAt(k);
    ASSERT_EQ(got.mu.size(), want.mu.size());
    for (auto i = std::size_t{0}; i < got.mu.size(); ++i) {
      worst = std::max({worst,
                        std::abs(got.mu[i] - want.mu[i]) /
                            std::max(std::abs(want.mu[i]), 1e-300),
                        std::abs(got.itau[i] - want.itau[i]) /
                            std::max(std::abs(want.itau[i]), 1e-300)});
    }
  }
  RecordProperty("worst", std::format("{:.2e}", worst));
  EXPECT_LT(worst, 1e-15);
}

TEST_F(DecksTest, AreRefusedByName) {
  // `model.burgers` on a deck that states its rheology or has no viscosity,
  // beside a viscosity file, in the static problem; regions cutting a
  // layer, covering an elastic or a fluid one, beyond the deck,
  // overlapping, malformed or missing a member; a viscosity file beside a
  // prony or burgers deck.
  WriteMaxwell(dir_ / "ball.maxwell.deck");
  WriteBurgers(dir_ / "ball.burgers.deck");
  WriteProny(dir_ / "ball.prony.deck", {{1.0, 6e9}});
  const auto region = [](const std::string& depth,
                         const std::string& more = "") {
    return std::format(R"({{"depth_km": {}, "kelvin_viscosity": 1e20,
                           "kelvin_modulus_fraction": 0.5{}}})",
                       depth, more);
  };
  const auto ball = [&](const std::string& name, const std::string& regions,
                        const std::string& deck = "ball.maxwell.deck",
                        const std::string& more = "") {
    return Solve(name, std::format(R"({{"deck": "{}"{}, "burgers": [{}]}})",
                                   deck, more, regions));
  };
  const auto refused = [&](const fs::path& config, const std::string& member,
                           const std::string& what) {
    ExpectRefused(app::RunSolve, config, member, what);
  };
  refused(ball("prony", region("[100, 6371]"), "ball.prony.deck",
               R"(, "format": "prony")"),
          "model.burgers", "a deck of format \"prony\" states its rheology");
  refused(ball("burgers", region("[100, 6371]"), "ball.burgers.deck",
               R"(, "format": "burgers")"),
          "model.burgers", "a deck of format \"burgers\" states its rheology");
  refused(ball("elastic", region("[100, 6371]"), "ball.maxwell.deck",
               R"(, "format": "elastic")"),
          "model.burgers", "a deck of format \"elastic\" has none");
  const auto viscosityFile =
      std::format(R"(, "viscosity": {{"file": "{}/viscosity_log.nc"}})",
                  SL3DPP_TEST_DATA_DIR);
  refused(
      ball("file", region("[100, 6371]"), "ball.maxwell.deck", viscosityFile),
      "model.burgers", "a Burgers body over a viscosity file is not built");
  refused(
      Solve("prony_file",
            std::format(R"({{"deck": "ball.prony.deck", "format": "prony"{}}})",
                        viscosityFile)),
      "model.viscosity", "beside a deck of format \"prony\"");
  refused(Write("static.json", R"({"problem": "static",
      "model": {"deck": "ball.maxwell.deck", "burgers": [{"depth_km": [100, 6371],
        "kelvin_viscosity": 1e20, "kelvin_modulus_fraction": 0.5}]},
      "discretisation": {"lmax": 8}, "rotation": "none",
      "forcing": {"seismic": [{"point": {"latitude": 10, "longitude": 20,
          "depth_km": 50, "strike_deg": 30, "dip_deg": 40, "rake_deg": 80,
          "moment_nm": 1e21}}]},
      "output": {"surface": "out/static.nc"}})"),
          "model.burgers", "is a rheology, Burgers layers, which");
  refused(ball("cut", region("[100, 3000]")), "model.burgers[0].depth_km",
          "puts an end at 3000 km, inside layer 0 (100 to 6371 km deep)");
  refused(ball("lid", region("[0, 6371]")), "model.burgers[0].depth_km",
          "covers layer 1 (0 to 100 km deep), which is elastic");
  refused(ball("beyond", region("[100, 7000]")), "model.burgers[0].depth_km",
          "reaches 7000 km deep, beyond the deck");
  refused(ball("overlap", region("[100, 6371]") + ", " + region("[200, 300]")),
          "model.burgers[1].depth_km", "overlaps region 0 (100 to 6371 km)");
  refused(ball("short", region("[100]")), "model.burgers[0].depth_km",
          "is not [top, bottom]");
  refused(ball("upside", region("[6371, 100]")), "model.burgers[0].depth_km",
          "0 <= top < bottom");
  refused(
      ball("missing", R"({"depth_km": [100, 6371], "kelvin_viscosity": 1e20})"),
      "model.burgers[0]", "has no `kelvin_modulus_fraction`");
  refused(ball("unknown", region("[100, 6371]", R"(, "extra": 1)")),
          "model.burgers[0].extra", "");
  refused(ball("empty", ""), "model.burgers", "names no region");
  refused(Solve("fluid", std::format(R"({{"deck": "{}", "burgers": [{}]}})",
                                     PostseismicDeck().string(),
                                     region("[2888, 5146.5]"))),
          "model.burgers[0].depth_km",
          "covers layer 1 (2888 to 5146.5 km deep), which is fluid");
}

/// The value of a measurement in a kernel run's `measurements.txt`.
double MeasuredValue(const fs::path& directory, const std::string& name) {
  auto in = std::ifstream(directory / "measurements.txt");
  for (std::string line; std::getline(in, line);) {
    if (!line.starts_with(name + ' ')) continue;
    auto words = std::istringstream(line);
    auto label = std::string{}, type = std::string{};
    auto value = 0.0;
    words >> label >> type >> value;
    return value;
  }
  ADD_FAILURE() << name << " is not in " << directory / "measurements.txt";
  return 0.0;
}

/// The rows of numbers of a text table, its `#` lines left out.
std::vector<std::vector<double>> Rows(const fs::path& path) {
  auto in = std::ifstream(path);
  auto out = std::vector<std::vector<double>>{};
  for (auto line = std::string{}; std::getline(in, line);) {
    if (line.empty() || line.starts_with('#')) continue;
    auto words = std::istringstream(line);
    auto row = std::vector<double>{};
    for (double v; words >> v;) row.push_back(v);
    out.push_back(std::move(row));
  }
  return out;
}

TEST_F(DecksTest, TheViscosityKernelsOfABurgersBodyAreItsBranches) {
  // `kernel` with the viscosity parameter on the Burgers ball: the radial
  // kernels a column a branch (`K_branch_0` the slower term, `K_branch_1`
  // the faster), and with `kernels.lateral` the log-viscosity kernels of
  // each branch as fields. The burgers deck's and the regions' are one to
  // rounding. Their pairing with a uniform change of every branch's ln eta,
  // every relaxation time raised alike (the file's
  // `uniform_log_viscosity_pairing`), by the discrete adjoint, is the
  // derivative of the measurement as the deck's two viscosities are scaled
  // together (the strengths held), by central differences of burgers
  // decks scaled by 1 +- 1e-3; and
  // the prony deck's, its branches the other way round, the same to the
  // eight digits it writes them with.
  WriteMaxwell(dir_ / "ball.maxwell.deck");
  WriteBurgers(dir_ / "ball.burgers.deck");
  WriteBurgers(dir_ / "plus.burgers.deck", 1.0 + 1e-3);
  WriteBurgers(dir_ / "minus.burgers.deck", 1.0 - 1e-3);
  WriteProny(dir_ / "ball.prony.deck",
             {{kStrengths[0], kTimes[0]}, {kStrengths[1], kTimes[1]}});
  const auto kernels = [](const std::string& name) {
    return std::format(R"(, "kernels": {{"lateral": true, "method": "discrete",
      "parameters": ["viscosity"],
      "measurements": [{{"type": "displacement_cap", "name": "uplift",
        "latitude": 12, "longitude": 22, "width_deg": 15, "time": 50}}],
      "output": "out/{}_kernels"}})",
                       name);
  };
  const auto run = [&](const std::string& name, const std::string& model) {
    static_cast<void>(app::RunKernel(Solve(name, model, kernels(name))));
    return dir_ / "out" / (name + "_kernels");
  };
  const auto burgers =
      run("burgers", R"({"deck": "ball.burgers.deck", "format": "burgers"})");
  const auto regions =
      run("regions", std::format(R"({{"deck": "ball.maxwell.deck",
        "burgers": [{{"depth_km": [100, 6371], "kelvin_viscosity": {:.17g},
                      "kelvin_modulus_fraction": {:.17g}}}]}})",
                                 kKelvinViscosity, kKelvinFraction));
  const auto prony =
      run("prony", R"({"deck": "ball.prony.deck", "format": "prony"})");
  const auto plus =
      run("plus", R"({"deck": "plus.burgers.deck", "format": "burgers"})");
  const auto minus =
      run("minus", R"({"deck": "minus.burgers.deck", "format": "burgers"})");
  const auto table = Rows(burgers / "uplift_viscosity_kernel.txt");
  const auto byRegions = Rows(regions / "uplift_viscosity_kernel.txt");
  ASSERT_FALSE(table.empty());
  ASSERT_EQ(table.size(), byRegions.size());
  auto scale = 0.0, gapRows = 0.0;
  for (auto r = std::size_t{0}; r < table.size(); ++r) {
    ASSERT_EQ(table[r].size(), byRegions[r].size());
    for (auto c = std::size_t{1}; c + 1 < table[r].size(); ++c) {
      scale = std::max(scale, std::abs(table[r][c]));
      gapRows = std::max(gapRows, std::abs(table[r][c] - byRegions[r][c]));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LE(gapRows, 1e-12 * scale);
  auto header = std::string{};
  {
    auto in = std::ifstream(burgers / "uplift_viscosity_kernel.txt");
    for (std::string line; std::getline(in, line);) {
      if (line.starts_with("# r_m")) header = line;
    }
  }
  EXPECT_EQ(header, "# r_m K_branch_0 K_branch_1 W_m3");
  const auto uniform = [](const fs::path& directory) {
    const auto file =
        pm::ReadModelFile(directory / "uplift_viscosity_kernels.nc");
    auto held = std::vector<std::string>{};
    for (const auto& part : file.Parts()) {
      for (const auto& [name, values] : part.variables) {
        if (name.starts_with("K_ln_eta") && !name.ends_with("_poles") &&
            std::ranges::find(held, name) == held.end()) {
          held.push_back(name);
        }
      }
    }
    std::ranges::sort(held);
    EXPECT_EQ(held, (std::vector<std::string>{"K_ln_eta_0", "K_ln_eta_1"}));
    return std::get<std::vector<double>>(
               file.Meta().at("uniform_log_viscosity_pairing"))
        .front();
  };
  const auto predicted = uniform(burgers);
  EXPECT_NEAR(uniform(regions), predicted, 1e-12 * std::abs(predicted));
  const auto measured =
      (MeasuredValue(plus, "uplift") - MeasuredValue(minus, "uplift")) /
      (std::log(1.0 + 1e-3) - std::log(1.0 - 1e-3));
  const auto gap = std::abs(measured - predicted) / std::abs(predicted);
  const auto branches =
      std::abs(uniform(prony) - predicted) / std::abs(predicted);
  RecordProperty("fd_gap", std::format("{:.2e}", gap));
  RecordProperty("prony_gap", std::format("{:.2e}", branches));
  EXPECT_LT(gap, 1e-5);
  EXPECT_LT(branches, 1e-7);
}


}  // namespace
