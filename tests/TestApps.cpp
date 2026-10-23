// The sl3d configs and commands: a misspelt or missing member is refused
// naming the config and the place, relative paths resolve against the
// config's directory, love-numbers writes its provenance-headed table,
// and fingerprint runs a toy state read from a planetmodel surface file
// through the sea level equation and writes the fingerprint back as one.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/surfacefile.hpp>
#include <string>
#include <vector>

#include "Commands.hpp"

namespace {

namespace fs = std::filesystem;
using namespace sl3dpp;

class AppsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("sl3dpp_test_apps_" + std::string{::testing::UnitTest::GetInstance()
                                                  ->current_test_info()
                                                  ->name()});
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  /// A file under the test's directory, written whole.
  fs::path Write(const std::string& name, const std::string& text) const {
    const auto path = dir_ / name;
    auto out = std::ofstream(path);
    out << text;
    return path;
  }

  /// A homogeneous GIA-format deck (two header lines; columns r, rho,
  /// vpv, vsv, vph, vsh, eta, viscosity).
  fs::path Deck() const {
    return Write("ball.deck",
                 "homogeneous test ball\n"
                 "gia deck for the sl3d tests\n"
                 "0 5500 10000 5500 10000 5500 1 1e21\n"
                 "6371000 5500 10000 5500 10000 5500 1 1e21\n");
  }

  fs::path dir_;
};

TEST_F(AppsTest, RefusesWhatTheConfigCannotSay) {
  const auto deck = Deck();
  auto refuse = [&](const std::string& text, const std::string& what) {
    const auto config = Write("bad.json", text);
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::ReadLoveNumbersInput(config));
          } catch (const planetmodel::Error& error) {
            EXPECT_NE(std::string{error.what()}.find(what), std::string::npos)
                << error.what();
            throw;
          }
        },
        planetmodel::Error)
        << text;
  };
  refuse(R"({"model": "ball.deck", "lmax": 4, "output": "t.txt",
             "lmxa": 3})",
         "not a recognised member");
  refuse(R"({"lmax": 4, "output": "t.txt"})", "has no member");
  refuse(R"({"model": "ball.deck", "lmax": 4.5, "output": "t.txt"})",
         "not an integer");
  refuse(R"({"model": "ball.deck", "lmax": -2, "output": "t.txt"})",
         "not a degree");
  refuse(R"({"model": "ball.deck", "model_format": "fortran", "lmax": 4,
             "output": "t.txt"})",
         "none of");
}

TEST_F(AppsTest, ResolvesPathsAgainstTheConfig) {
  static_cast<void>(Deck());
  const auto config = Write("run.json", R"({"model": "ball.deck", "lmax": 4,
                                            "output": "out/love.txt"})");
  const auto input = app::ReadLoveNumbersInput(config);
  EXPECT_EQ(input.model, dir_ / "ball.deck");
  EXPECT_EQ(input.output, dir_ / "out" / "love.txt");
  EXPECT_EQ(input.lmax, 4);
  EXPECT_EQ(input.format, app::ModelFormat::Gia);
}

TEST_F(AppsTest, LoveNumbersWritesItsTable) {
  static_cast<void>(Deck());
  const auto config = Write("run.json", R"({"model": "ball.deck", "lmax": 4,
                                            "output": "out/love.txt"})");
  const auto summary = app::RunLoveNumbers(config);
  EXPECT_NE(summary.find("wrote"), std::string::npos);
  auto in = std::ifstream(dir_ / "out" / "love.txt");
  ASSERT_TRUE(in);
  auto line = std::string{};
  auto header = 0, rows = 0;
  auto saw_version = false;
  while (std::getline(in, line)) {
    if (line.starts_with("#")) {
      ++header;
      saw_version = saw_version || line.find("sl3d love-numbers") !=
                                       std::string::npos;
    } else if (!line.empty()) {
      ++rows;
    }
  }
  EXPECT_TRUE(saw_version);
  EXPECT_GE(header, 8);
  EXPECT_EQ(rows, 5);  // degrees 0..4
}

TEST_F(AppsTest, FingerprintRunsAToyStateEndToEnd) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  // the toy present day: land to the north, ocean to the south, a cap
  // of ice on the land
  const auto grid = pm::Equiangular(19, 36);
  auto topography = std::vector<double>{};
  auto ice = std::vector<double>{};
  for (const auto theta : grid.Colatitudes()) {
    for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
      topography.push_back(3000.0 * std::cos(theta));
      ice.push_back(theta < 0.45 ? 1500.0 * std::exp(-std::pow(theta / 0.3, 2))
                                 : 0.0);
    }
  }
  auto state = pm::SurfaceFile{};
  state.grid = grid;
  state.source = "sl3dpp tests";
  state.variables.emplace(
      "ice_thickness",
      pm::SurfaceField("ice_thickness", ice,
                       {.dimensions = pm::units::LENGTH}));
  state.variables.emplace(
      "topography",
      pm::SurfaceField("topography", topography,
                       {.dimensions = pm::units::LENGTH}));
  pm::WriteSurfaceFile(dir_ / "state.nc", state);

  const auto config = Write("run.json", R"({
    "model": "ball.deck",
    "lmax": 8,
    "state": {"file": "state.nc"},
    "change": {"type": "cap_fraction", "latitude": 90, "longitude": 0,
               "width_deg": 15, "fraction": -0.1},
    "rotation": "full",
    "output": "out/fingerprint.nc"})");
  const auto summary = app::RunFingerprint(config);
  EXPECT_NE(summary.find("wrote"), std::string::npos);

  const auto result = pm::ReadSurfaceFile(dir_ / "out" / "fingerprint.nc");
  for (const auto name : {"ice_change", "sea_level_change", "displacement",
                          "potential", "ocean_function"}) {
    EXPECT_TRUE(result.variables.contains(name)) << name;
  }
  EXPECT_TRUE(result.meta.contains("omega_z_rad_s"));
  EXPECT_EQ(result.grid->Kind(), "gauss_legendre");
  const auto& ocean = result.variables.at("ocean_function").values;
  for (const auto value : ocean) {
    EXPECT_GE(value, -0.05);
    EXPECT_LE(value, 1.05);
  }
  // melting ice must raise the far-field sea: somewhere positive
  const auto& change = result.variables.at("sea_level_change").values;
  auto largest = 0.0;
  for (const auto value : change) largest = std::max(largest, value);
  EXPECT_GT(largest, 0.0);
}

/// A toy world for the evolution: land north, ocean south, a polar ice
/// cap shrinking linearly over the window (2 kyr BP to the present).
void WriteToyHistory(const fs::path& path) {
  namespace pm = planetmodel;
  constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::vector<double>{2.0, 1.0, 0.0};  // kyr BP
  auto time = std::vector<double>{};
  for (const auto d : dates) time.push_back(-d * kyr);
  auto ice = std::vector<double>{};
  auto topo = std::vector<double>{};
  for (auto k = std::size_t{0}; k < dates.size(); ++k) {
    const auto fraction = 1.0 - 0.3 * static_cast<double>(k) /
                                    static_cast<double>(dates.size() - 1);
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        topo.push_back(3000.0 * std::cos(theta));
        ice.push_back(theta < 0.45 ? fraction * 1500.0 *
                                         std::exp(-std::pow(theta / 0.3, 2))
                                   : 0.0);
      }
    }
  }
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  file.time = time;
  auto options = pm::SurfaceFieldOptions{};
  options.support = "node";
  options.time_interpolation = "linear";
  options.dimensions = pm::units::LENGTH;
  file.variables.emplace("ice_thickness",
                         pm::SurfaceField("ice_thickness", ice, options));
  file.variables.emplace("topography",
                         pm::SurfaceField("topography", topo, options));
  pm::WriteSurfaceFile(path, file);
}

TEST_F(AppsTest, EvolveRunsAndARestartedRunMatchesAnUninterruptedOne) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{R"(
    "model": "ball.deck",
    "lmax": 8,
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
    "rotation": "full", "moments": "hydrostatic")"};
  const auto whole = Write(
      "whole.json", "{" + body + R"(, "output": "out/whole.nc"})");
  auto summary = app::RunEvolve(whole);
  EXPECT_NE(summary.find("wrote"), std::string::npos);
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  ASSERT_TRUE(reference.time);
  EXPECT_EQ(reference.time->size(), 5u);
  EXPECT_EQ(
      pm::detail::TextOf(reference.variables.at("sea_level_change").attributes,
                         "time_support")
          .value_or(""),
      "node");

  // the same run through wall-time stops: each attempt checkpoints and
  // quits, the reruns resume, and the final output is the reference's
  const auto pieces = Write(
      "pieces.json",
      "{" + body +
          R"(, "checkpoint": {"directory": "ckpt", "every": 2,
                              "max_walltime_min": 1e-9},
              "output": "out/pieces.nc"})");
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    summary = app::RunEvolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
    EXPECT_NE(summary.find("checkpointed"), std::string::npos) << summary;
  }
  EXPECT_GE(attempts, 1);
  EXPECT_NE(summary.find("resumed"), std::string::npos);
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  ASSERT_TRUE(rebuilt.time);
  EXPECT_EQ(*rebuilt.time, *reference.time);
  for (const auto name : {"sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
  EXPECT_EQ(rebuilt.meta.at("omega_z_rad_s"), reference.meta.at("omega_z_rad_s"));

  // a different config must not adopt this run's checkpoints
  const auto other = Write(
      "other.json",
      "{" + body +
          R"(, "checkpoint": {"directory": "ckpt", "every": 2},
              "output": "out/other.nc"})");
  EXPECT_THROW(static_cast<void>(app::RunEvolve(other)), std::runtime_error);
}

TEST_F(AppsTest, Evolve3DRunsAndARestartedRunMatchesAnUninterruptedOne) {
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto body = std::string{R"(
    "model": ")"} + data + R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" + data + R"(/stage11_viscosity_log.nc"},
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
    "rotation": "full")";
  const auto whole = Write(
      "whole3d.json", "{" + body + R"(, "output": "out/whole.nc"})");
  auto summary = app::RunEvolve3D(whole);
  EXPECT_NE(summary.find("wrote"), std::string::npos);
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  ASSERT_TRUE(reference.time);
  EXPECT_EQ(reference.time->size(), 4u);
  EXPECT_EQ(reference.meta.at("viscosity"),
            pm::netcdf::Attribute{data + "/stage11_viscosity_log.nc"});

  const auto pieces = Write(
      "pieces3d.json",
      "{" + body +
          R"(, "checkpoint": {"directory": "ckpt3d", "every": 1,
                              "max_walltime_min": 1e-9},
              "output": "out/pieces.nc"})");
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    summary = app::RunEvolve3D(pieces);
    if (summary.find("wrote") != std::string::npos) break;
    EXPECT_NE(summary.find("checkpointed"), std::string::npos) << summary;
  }
  EXPECT_GE(attempts, 1);
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  for (const auto name : {"sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
  EXPECT_EQ(rebuilt.meta.at("omega_z_rad_s"),
            reference.meta.at("omega_z_rad_s"));

  // the 1D command refuses a 3D config
  EXPECT_THROW(static_cast<void>(app::RunEvolve(whole)), planetmodel::Error);
}

TEST_F(AppsTest, FingerprintRefusesAStateWithoutTheFields) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  auto state = pm::SurfaceFile{};
  state.grid = pm::Equiangular(9, 18);
  state.source = "sl3dpp tests";
  state.variables.emplace(
      "ice_thickness",
      pm::SurfaceField("ice_thickness",
                       std::vector<double>(9 * 18, 0.0),
                       {.dimensions = pm::units::LENGTH}));
  pm::WriteSurfaceFile(dir_ / "state.nc", state);
  const auto config = Write("run.json", R"({
    "model": "ball.deck",
    "lmax": 4,
    "state": {"file": "state.nc"},
    "change": {"type": "cap_thickness", "latitude": 0, "longitude": 0,
               "width_deg": 10, "thickness_m": -1.0},
    "output": "out/fingerprint.nc"})");
  EXPECT_THROW(static_cast<void>(app::RunFingerprint(config)),
               planetmodel::Error);
}

}  // namespace
