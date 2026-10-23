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
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

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
/// `bump_last` adds a smooth mid-latitude patch to the final date's ice
/// alone, for finite differences of the kernels.
void WriteToyHistory(const fs::path& path, double bump_last = 0.0) {
  namespace pm = planetmodel;
  constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::vector<double>{2.0, 1.0, 0.0};  // kyr BP
  auto time = std::vector<double>{};
  for (const auto d : dates) time.push_back(-d * kyr);
  auto ice = std::vector<double>{};
  auto topo = std::vector<double>{};
  // melt from 2 to 1 kyr BP, then hold: the hold is the decay window
  const auto fractions = std::vector<double>{1.0, 0.7, 0.7};
  for (auto k = std::size_t{0}; k < dates.size(); ++k) {
    const auto fraction = fractions[k];
    const auto bump = k + 1 == dates.size() ? bump_last : 0.0;
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        topo.push_back(3000.0 * std::cos(theta));
        auto thickness = theta < 0.45 ? fraction * 1500.0 *
                                            std::exp(-std::pow(theta / 0.3, 2))
                                      : 0.0;
        // the bump sits on grounded northern land, where ice may change
        thickness += bump * std::exp(-std::pow((theta - 0.6) / 0.2, 2));
        ice.push_back(thickness);
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

  // the checkpoints belong to the run, not to the config's text: the same
  // run with another output, no wall-time limit and other spacing adopts
  // them...
  const auto same = Write("same.json", "{\n" + body +
                                           R"(,
              "checkpoint": {"directory": "ckpt", "every": 2},
              "output": "out/same.nc"
          })");
  summary = app::RunEvolve(same);
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  EXPECT_EQ(pm::ReadSurfaceFile(dir_ / "out" / "same.nc")
                .variables.at("sea_level_change")
                .values,
            reference.variables.at("sea_level_change").values);

  // ...while another computation, or the same config over a changed input
  // file, is refused
  const auto refused = [](const fs::path& config) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::RunEvolve(config));
          } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string{error.what()}.find("different run"),
                      std::string::npos)
                << error.what();
            throw;
          }
        },
        std::runtime_error)
        << config;
  };
  refused(Write("tighter.json", "{" + body +
                                    R"(, "rtol": 1e-11,
                        "checkpoint": {"directory": "ckpt", "every": 2},
                        "output": "out/tighter.nc"})"));
  WriteToyHistory(dir_ / "history.nc", 1.0);
  refused(same);

  // starting over replaces the earlier run's checkpoints: a shorter run's
  // own would otherwise be pruned against the earlier run's later nodes,
  // which a resumption would then pick. Other files are left alone.
  ASSERT_TRUE(fs::exists(dir_ / "ckpt" / "node_000004.ckpt"));
  static_cast<void>(Write("ckpt/notes.ckpt", "not a checkpoint"));
  auto shorter = body;
  shorter.replace(shorter.find(R"("nodes": 5)"), 10, R"("nodes": 3)");
  const auto over = Write("over.json", "{" + shorter +
                                           R"(, "restart": false,
              "checkpoint": {"directory": "ckpt", "every": 1},
              "output": "out/over.nc"})");
  summary = app::RunEvolve(over);
  EXPECT_NE(summary.find("replacing 2 earlier checkpoint(s)"),
            std::string::npos)
      << summary;
  EXPECT_FALSE(fs::exists(dir_ / "ckpt" / "node_000004.ckpt"));
  EXPECT_TRUE(fs::exists(dir_ / "ckpt" / "node_000002.ckpt"));
  EXPECT_TRUE(fs::exists(dir_ / "ckpt" / "notes.ckpt"));
  // and the same run, restarting, adopts the new checkpoints
  const auto again = Write(
      "again.json", "{" + shorter +
                        R"(, "checkpoint": {"directory": "ckpt", "every": 1},
              "output": "out/again.nc"})");
  summary = app::RunEvolve(again);
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
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

TEST_F(AppsTest, KernelsMatchFiniteDifferencesOfTheForwardValue) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "state.nc");
  WriteToyHistory(dir_ / "plus.nc", 20.0);
  WriteToyHistory(dir_ / "minus.nc", -20.0);
  auto config_with = [&](const std::string& history, const std::string& name,
                         const std::string& extra) {
    return Write(name, std::string{R"({
      "model": "ball.deck",
      "lmax": 8,
      "state": {"file": "state.nc"},
      "ice_history": {"file": ")"} + history + R"("},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 9},
      "rotation": "full", "moments": "hydrostatic",
      "adjoint": "discrete",
      "measurements": [
        {"type": "sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0},
        {"type": "stokes_coefficient", "name": "j2", "degree": 2,
         "order": 0, "date_kyr_bp": 0.0},
        {"type": "rotation", "name": "lod", "component": "z",
         "date_kyr_bp": 0.0},
        {"type": "decay_time", "name": "rebound", "latitude": 80.0,
         "longitude": 0.0, "width_deg": 15.0, "window_start_kyr_bp": 1.0,
         "window_end_kyr_bp": 0.0, "time_uncertainty_kyr": 0.0,
         "value_uncertainty_m": 0.1}],
      "output": ")" + extra + R"("})");
  };

  auto value_of = [&](const fs::path& output, const std::string& name) {
    auto in = std::ifstream(output / "measurements.txt");
    auto line = std::string{};
    while (std::getline(in, line)) {
      if (line.starts_with(name + ' ')) {
        auto words = std::istringstream(line);
        auto label = std::string{}, type = std::string{};
        auto value = 0.0;
        words >> label >> type >> value;
        return value;
      }
    }
    throw std::runtime_error(name + " is not in measurements.txt");
  };

  auto summary = app::RunKernel1D(
      config_with("state.nc", "kernels.json", "out_kernels"));
  EXPECT_NE(summary.find("4 measurement(s)"), std::string::npos);
  const auto out = dir_ / "out_kernels";
  for (const auto name :
       {"site_ice_kernels.nc", "j2_ice_kernels.nc", "lod_ice_kernels.nc",
        "rebound_ice_kernels.nc", "site_viscosity_kernel.txt",
        "measurements.txt"}) {
    EXPECT_TRUE(fs::exists(out / name)) << name;
  }
  const auto rebound = value_of(out, "rebound");
  EXPECT_GT(rebound, 0.0);  // a decay time, seconds

  // the finite-difference gate: the site value's change under an ice
  // bump at the final date equals the thickness duals' prediction
  static_cast<void>(
      app::RunKernel1D(config_with("plus.nc", "plus.json", "out_plus")));
  static_cast<void>(
      app::RunKernel1D(config_with("minus.nc", "minus.json", "out_minus")));
  const auto difference = (value_of(dir_ / "out_plus", "site") -
                           value_of(dir_ / "out_minus", "site")) /
                          2.0;

  const auto kernels = pm::ReadSurfaceFile(out / "site_ice_kernels.nc");
  const auto& grid = *kernels.grid;
  const auto& thickness = kernels.variables.at("thickness_dual").values;
  const auto nphi = grid.NPhi();
  const auto ntheta = grid.NTheta();
  const auto a = 6371000.0;  // the pairing's radius: the deck's surface
  const auto weights = *grid.Weights();
  // the perturbation exactly as the runs saw it: the bumped histories
  // through the same resampling, at each run node
  const auto solver_grid = SurfaceGrid(8);
  const auto plus = *io::FieldHistory::Read(dir_ / "plus.nc", solver_grid);
  const auto minus = *io::FieldHistory::Read(dir_ / "minus.nc", solver_grid);
  constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
  auto predicted = 0.0;
  for (auto n = std::size_t{0}; n < 9; ++n) {
    const auto date = 2.0 - 0.25 * static_cast<double>(n);
    const auto up = plus.Values("ice_thickness", -date * kyr);
    const auto down = minus.Values("ice_thickness", -date * kyr);
    for (auto it = std::size_t{0}; it < ntheta; ++it) {
      auto ring = 0.0;
      for (auto j = std::size_t{0}; j < nphi; ++j) {
        ring += thickness[(n * ntheta + it) * nphi + j] *
                0.5 * (up[it * nphi + j] - down[it * nphi + j]);
      }
      predicted += weights[it] * ring;
    }
  }
  predicted *= 2.0 * std::numbers::pi / static_cast<double>(nphi) * a * a;
  EXPECT_NEAR(predicted, difference,
              1e-5 * std::max(std::abs(difference), 1e-12))
      << "predicted " << predicted << " measured " << difference;
}

TEST_F(AppsTest, Kernel3DWritesIceAndViscosityKernels) {
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto config = Write("kernel3d.json", std::string{R"({
    "model": ")"} + data + R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" + data + R"(/stage11_viscosity_log.nc"},
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
    "rotation": "full",
    "measurements": [
      {"type": "sea_level_cap", "name": "site", "latitude": -30.0,
       "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}],
    "output": "out_kernel3d"})");
  const auto summary = app::RunKernel3D(config);
  EXPECT_NE(summary.find("1 measurement(s)"), std::string::npos);
  const auto out = dir_ / "out_kernel3d";
  const auto ice = pm::ReadSurfaceFile(out / "site_ice_kernels.nc");
  EXPECT_TRUE(ice.variables.contains("thickness_dual"));
  EXPECT_TRUE(ice.variables.contains("rate_dual"));
  const auto viscosity =
      pm::ReadModelFile(out / "site_viscosity_kernels.nc");
  auto held = false;
  for (const auto& part : viscosity.Parts()) {
    held = held || part.variables.contains("K_ln_eta_0");
  }
  EXPECT_TRUE(held);

  // the disk-backed trajectory: the recorded states spilled during the
  // run and fetched back by the sweep give the same kernels exactly
  const auto spilled_config = Write("kernel3d_spill.json", std::string{R"({
    "model": ")"} + data + R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" + data + R"(/stage11_viscosity_log.nc"},
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
    "rotation": "full",
    "record_stride": 2,
    "trajectory": {"directory": "spill"},
    "measurements": [
      {"type": "sea_level_cap", "name": "site", "latitude": -30.0,
       "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}],
    "output": "out_kernel3d_spill"})");
  const auto in_memory_config = Write("kernel3d_mem.json", std::string{R"({
    "model": ")"} + data + R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" + data + R"(/stage11_viscosity_log.nc"},
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
    "rotation": "full",
    "record_stride": 2,
    "measurements": [
      {"type": "sea_level_cap", "name": "site", "latitude": -30.0,
       "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}],
    "output": "out_kernel3d_mem"})");
  static_cast<void>(app::RunKernel3D(in_memory_config));
  static_cast<void>(app::RunKernel3D(spilled_config));
  EXPECT_TRUE(fs::exists(dir_ / "spill" / "node_000000.traj"));
  EXPECT_TRUE(fs::exists(dir_ / "spill" / "node_000002.traj"));
  const auto memory =
      pm::ReadSurfaceFile(dir_ / "out_kernel3d_mem" / "site_ice_kernels.nc");
  const auto disk =
      pm::ReadSurfaceFile(dir_ / "out_kernel3d_spill" / "site_ice_kernels.nc");
  for (const auto name : {"thickness_dual", "rate_dual"}) {
    EXPECT_EQ(memory.variables.at(name).values, disk.variables.at(name).values)
        << name;
  }
  const auto k_memory =
      pm::ReadModelFile(dir_ / "out_kernel3d_mem" / "site_viscosity_kernels.nc");
  const auto k_disk = pm::ReadModelFile(dir_ / "out_kernel3d_spill" /
                                        "site_viscosity_kernels.nc");
  ASSERT_EQ(k_memory.Parts().size(), k_disk.Parts().size());
  for (auto p = std::size_t{0}; p < k_memory.Parts().size(); ++p) {
    const auto& held_memory = k_memory.Parts()[p].variables;
    const auto& held_disk = k_disk.Parts()[p].variables;
    for (const auto& [name, var] : held_memory) {
      if (!name.starts_with("K_ln_eta")) continue;
      EXPECT_EQ(var.values, held_disk.at(name).values) << name;
    }
  }
}

// A measurement is held to the run when the config is read: a date off the
// times grid, a coefficient beyond the band or a decay window too short for
// the fit costs a read, not the forward run before the sweeps.
TEST_F(AppsTest, KernelMeasurementsAreCheckedWhenTheConfigIsRead) {
  const auto config = [&](const std::string& measurement) {
    return Write("kernel_check.json", R"({
      "model": "ball.deck", "lmax": 8,
      "state": {"file": "history.nc"},
      "ice_history": {"file": "history.nc"},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
      "measurements": [)" + measurement + R"(],
      "output": "out"})");
  };
  const auto refuse = [&](const std::string& measurement,
                          const std::string& what) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::ReadKernelInput(config(measurement)));
          } catch (const planetmodel::Error& error) {
            EXPECT_NE(std::string{error.what()}.find(what), std::string::npos)
                << error.what();
            throw;
          }
        },
        planetmodel::Error)
        << measurement;
  };
  // the nodes are 2, 1.5, 1, 0.5 and 0 kyr BP
  refuse(R"({"type": "sea_level_point", "latitude": 10, "longitude": 20,
             "date_kyr_bp": 0.3})",
         "date_kyr_bp: 0.3 kyr BP is not a node");
  refuse(R"({"type": "rotation", "component": "z", "date_kyr_bp": 2.5})",
         "is not a node");
  refuse(R"({"type": "stokes_coefficient", "degree": 9, "order": 0})",
         "(9, 0) is not a degree and order at lmax 8");
  refuse(R"({"type": "stokes_coefficient", "degree": 2, "order": -3})",
         "(2, -3) is not a degree and order");
  refuse(R"({"type": "decay_time", "latitude": 80, "longitude": 0,
             "width_deg": 15, "window_start_kyr_bp": 1.0,
             "time_uncertainty_kyr": 0.1, "value_uncertainty_m": 1.0})",
         "holds 3 nodes");
  const auto input = app::ReadKernelInput(config(
      R"({"type": "stokes_coefficient", "degree": 2, "order": -2,
          "date_kyr_bp": 0.5},
         {"type": "decay_time", "latitude": 80, "longitude": 0,
          "width_deg": 15, "window_start_kyr_bp": 1.5,
          "time_uncertainty_kyr": 0.1, "value_uncertainty_m": 1.0})"));
  EXPECT_EQ(input.measurements.size(), 2u);
}

TEST_F(AppsTest, Kernel1DRefusesASpilledTrajectory) {
  const auto config = Write("kernel1d_spill.json", R"({
    "model": "ball.deck", "lmax": 4,
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 3},
    "trajectory": {"directory": "spill"},
    "measurements": [
      {"type": "rotation", "name": "lod", "component": "z",
       "date_kyr_bp": 0.0}],
    "output": "out"})");
  EXPECT_THROW(
      {
        try {
          static_cast<void>(app::RunKernel1D(config));
        } catch (const planetmodel::Error& error) {
          EXPECT_NE(
              std::string{error.what()}.find("fits in memory"),
              std::string::npos)
              << error.what();
          throw;
        }
      },
      planetmodel::Error);
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
