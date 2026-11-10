// The sl3d configs and commands: a misspelt or missing member is refused
// naming the config and the place, relative paths resolve against the
// config's directory, love-numbers writes its provenance-headed table,
// fingerprint runs a toy state read from a planetmodel surface file
// through the sea level equation and writes the fingerprint back as one,
// and response takes its load and tide as sums of terms of every kind,
// elastic or, with a times block, viscoelastic through time.

#include <gtest/gtest.h>

#include <GaussQuad/All>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <map>
#include <numbers>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/json.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "AppFixtures.hpp"
#include "Commands.hpp"
#include "Common.hpp"
#include "Ranks.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/ModelModuli.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace {

namespace fs = std::filesystem;
using namespace sl3dpp;
using Complex = std::complex<double>;

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

  /// Runs `sl3d response` on the radial ball at `lmax` under the forcing
  /// block given, and returns its surface file.
  planetmodel::SurfaceFile Respond(const std::string& name,
                                   const std::string& forcing,
                                   int lmax = 8) const {
    static_cast<void>(Deck());
    const auto config = Write(name + ".json", std::format(R"({{
      "model": {{"deck": "ball.deck"}}, "lmax": {}, "forcing": {},
      "output": {{"surface": "out/{}.nc"}}}})",
                                                          lmax, forcing, name));
    const auto summary = app::RunResponse(config);
    EXPECT_TRUE(summary.starts_with("1-D, elastic, ")) << summary;
    return planetmodel::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  }

  /// Runs `sl3d response` on a config of the members given, its output
  /// block written after them (the surface file out/<name>.nc, a table per
  /// node out/<name>_{node}.txt), and returns the summary.
  std::string Run(const std::string& name, const std::string& members) const {
    const auto config = Write(
        name + ".json", std::format(R"({{{}, "output": {{"surface": "out/{}.nc",
                                          "exchange": "out/{}_{{node}}.txt"}}}})",
                                    members, name, name));
    return app::RunResponse(config);
  }

  /// A run's surface file.
  planetmodel::SurfaceFile Output(const std::string& name) const {
    return planetmodel::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  }

  fs::path dir_;
};

/// The largest difference of two fields over the largest magnitude of the
/// second.
double Departure(const std::vector<double>& got,
                 const std::vector<double>& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

/// The response fields of a surface file that every run writes.
constexpr auto kResponseFields = {"displacement", "tangential_potential",
                                  "potential"};

/// The largest difference of two coefficient sets over the largest
/// magnitude of the second.
double Departure(const std::vector<Complex>& got,
                 const std::vector<Complex>& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

/// One time slice of a surface file's node field.
std::vector<double> Slice(const planetmodel::SurfaceFile& file,
                          const std::string& name, std::size_t node) {
  const auto& values = file.variables.at(name).values;
  const auto size = values.size() / file.time->size();
  return {values.begin() + static_cast<std::ptrdiff_t>(node * size),
          values.begin() + static_cast<std::ptrdiff_t>((node + 1) * size)};
}

/// A response exchange table: its named scalars (the text after each
/// name) and its coefficients of U and Phi in the grid's layout.
struct ExchangeTable {
  std::map<std::string, std::string> scalars;
  std::vector<Complex> U, Phi;
};

ExchangeTable ReadExchange(const fs::path& path) {
  auto in = std::ifstream(path);
  EXPECT_TRUE(in) << path;
  auto out = ExchangeTable{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.starts_with("#")) continue;
    if (!std::isdigit(static_cast<unsigned char>(line.front()))) {
      const auto space = line.find(' ');
      out.scalars[line.substr(0, space)] = line.substr(space + 1);
      continue;
    }
    auto row = std::istringstream(line);
    auto l = 0, m = 0;
    auto ure = 0.0, uim = 0.0, pre = 0.0, pim = 0.0;
    row >> l >> m >> ure >> uim >> pre >> pim;
    const auto k = SurfaceGrid::Index(l, m);
    if (out.U.size() <= k) {
      out.U.resize(k + 1);
      out.Phi.resize(k + 1);
    }
    out.U[k] = {ure, uim};
    out.Phi[k] = {pre, pim};
  }
  return out;
}

/// A planetmodel model file of viscosity on a model's body, its fluid
/// layers left out: eta0 exp(amplitude cos(theta)) Pa s, stored laterally
/// (on a Gauss-Legendre grid, as a field that varies with direction even
/// when the amplitude is zero), or, with `radial`, eta0 as a field of the
/// radius alone.
void WriteViscosity(const fs::path& path, const EarthModel& model, double eta0,
                    double amplitude, bool radial = false) {
  namespace pm = planetmodel;
  const auto& boundaries = model.Boundaries();
  auto file = pm::ModelFile(
      pm::Skeleton(std::vector<double>(boundaries.begin(), boundaries.end())));
  file.SetName("sl3dpp tests viscosity");
  const auto& skeleton = file.GetSkeleton();
  const auto layers = static_cast<int>(skeleton.NLayers());
  file.AddPart(0, layers - 1, pm::ElementEdges(skeleton, 0, layers - 1, 500e3),
               5, pm::GaussLegendre(8));
  auto eta = std::map<int, pm::LayerSource>{};
  for (auto layer = 0; layer < layers; ++layer) {
    if (model.IsFluid(static_cast<std::size_t>(layer))) continue;
    eta.emplace(layer,
                radial
                    ? pm::LayerSource::Radial([eta0](double) { return eta0; })
                    : pm::LayerSource::Lateral(
                          [eta0, amplitude](double, double theta, double) {
                            return eta0 * std::exp(amplitude * std::cos(theta));
                          }));
  }
  file.AddField("viscosity", eta);
  pm::WriteModelFile(path, file);
}

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

  // with conventional_output, the dimensionless numbers too: seven
  // columns, the tidal ones nan below degree 2
  const auto both = Write("both.json", R"({"model": "ball.deck", "lmax": 4,
                                           "output": "out/love.txt",
                                           "conventional_output":
                                             "out/conventional.txt"})");
  EXPECT_NE(app::RunLoveNumbers(both).find("conventional.txt"),
            std::string::npos);
  auto table = std::ifstream(dir_ / "out" / "conventional.txt");
  ASSERT_TRUE(table);
  auto degrees = 0;
  while (std::getline(table, line)) {
    if (line.empty() || line.starts_with("#")) continue;
    auto row = std::istringstream(line);
    auto words = std::vector<std::string>{};
    for (auto word = std::string{}; row >> word;) words.push_back(word);
    ASSERT_EQ(words.size(), 7u) << line;
    EXPECT_EQ(words[6] == "nan", degrees < 2) << line;
    ++degrees;
  }
  EXPECT_EQ(degrees, 5);
  // the two tables may not share a file
  EXPECT_THROW(static_cast<void>(app::ReadLoveNumbersInput(
                   Write("same.json", R"({"model": "ball.deck", "lmax": 4,
                          "output": "out/love.txt",
                          "conventional_output": "out/./love.txt"})"))),
               planetmodel::Error);
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

using sl3dpp::test::WriteToyHistory;

TEST_F(AppsTest, ResponseRespondsOnAToyModuliFile) {
  // The response command on the homogeneous ball with a 10% lateral shear
  // modulus from a model file written by io::WriteModuli: a polar cap
  // load and a degree-2 tide, the surface file, the exchange table and
  // the volume file all written; the 3D displacement departs from the
  // radial model's by a few per cent, as a 10% modulus should move it.
  const auto deck = Deck();
  const auto read = sl3dpp::Deck::Read(deck, app::FormatOf(app::ModelFormat::Gia));
  ASSERT_TRUE(read) << read.error().message;
  const auto model = EarthModel::FromDeck(*read);
  ASSERT_TRUE(model) << model.error().message;
  const SurfaceGrid grid(8);
  const auto written = io::WriteModuli(
      dir_ / "moduli.nc", *model, grid,
      [](double, double theta, double) { return 0.1 * std::cos(theta); }, {},
      {.name = "toy moduli", .elementMetres = 500e3});
  ASSERT_TRUE(written) << written.error().message;

  const auto config = Write("response.json", R"({
    "model": {"deck": "ball.deck", "moduli": {"file": "moduli.nc"}},
    "lmax": 8,
    "forcing": {
      "load": [{"cap": {"latitude": 90, "longitude": 0, "width_deg": 20,
                        "amplitude_kg_m2": 1.0e6}}],
      "tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0}]},
    "output": {"surface": "out/response.nc",
               "exchange": "out/response.txt",
               "volume": "out/response_volume.nc"}})");
  const auto summary = app::RunResponse(config);
  EXPECT_NE(summary.find("wrote"), std::string::npos) << summary;
  EXPECT_NE(summary.find("conjugate-gradient"), std::string::npos) << summary;
  EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), elastic, load (1 "
                                  "term) + tide (1 coefficient); wrote"))
      << summary;

  namespace pm = planetmodel;
  const auto result = pm::ReadSurfaceFile(dir_ / "out" / "response.nc");
  for (const auto name : {"load", "displacement", "tangential_potential",
                          "potential", "displacement_radial_model",
                          "potential_radial_model"}) {
    EXPECT_TRUE(result.variables.contains(name)) << name;
  }
  const auto& u = result.variables.at("displacement").values;
  const auto& u1 = result.variables.at("displacement_radial_model").values;
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < u.size(); ++i) {
    scale = std::max(scale, std::abs(u1[i]));
    diff = std::max(diff, std::abs(u[i] - u1[i]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_GT(diff, 1e-3 * scale);
  EXPECT_LT(diff, 0.2 * scale);

  auto exchange = std::ifstream(dir_ / "out" / "response.txt");
  auto lines = std::vector<std::string>{};
  for (std::string line; std::getline(exchange, line);) lines.push_back(line);
  ASSERT_GT(lines.size(), 10u);
  EXPECT_EQ(lines[0].substr(0, 1), "#");
  const auto has = [&lines](std::string_view start) {
    return std::ranges::find_if(lines, [start](const std::string& l) {
             return l.starts_with(start);
           }) != lines.end();
  };
  EXPECT_TRUE(has("columns l m U_re U_im Phi_re Phi_im"));
  // one cap and one tide coefficient: their scalars, with the counts
  for (const auto* name :
       {"load_terms 1", "tide_coefficients 1", "cap_latitude_deg 90",
        "cap_width_deg 20", "cap_amplitude_kg_m2 1000000", "tide_degree 2",
        "tide_order 0", "tide_amplitude_m2_s2 1 0"}) {
    EXPECT_TRUE(has(name)) << name;
  }
  const auto volume = pm::ReadModelFile(dir_ / "out" / "response_volume.nc");
  for (const auto name : {"radial_displacement", "deviatoric_strain",
                          "mu_relative"}) {
    EXPECT_TRUE(volume.HasField(0, name)) << name;
  }
}

TEST_F(AppsTest, ResponseIsBitwiseIndependentOfTheCache) {
  // cache_mb reaches every transform grid of a response run (the forcing's,
  // the radial reference's coupling and the 3D operator's), and the run is
  // bit for bit the same under any setting. At lmax 24 the smallest
  // setting, 1 MiB, cuts the couplings' batched spin -2 transforms into
  // chunks of 53 fields against the default's 63, and the ball's mesh
  // carries more nodes than a chunk, so the chunks fall differently
  // through every batch: the surface file, the exchange table and the
  // volume file are nevertheless the default's exactly.
  constexpr auto kLmax = 24;
  const auto deck = Deck();
  const auto read =
      sl3dpp::Deck::Read(deck, app::FormatOf(app::ModelFormat::Gia));
  ASSERT_TRUE(read) << read.error().message;
  const auto model = EarthModel::FromDeck(*read);
  ASSERT_TRUE(model) << model.error().message;
  // the chunk of a spin -2 batch (one copy of its block: a stored table)
  const auto block = static_cast<GSHTrans::Chunking::Int>(
      sizeof(std::complex<double>) * ((kLmax + 1) * (kLmax + 1) - 4));
  const auto chunk = GSHTrans::Chunking::ForCache(1 << 20).Count(block, 1);
  EXPECT_LT(chunk, GSHTrans::Chunking::Automatic().Count(block, 1));
  // every element but the centre's carries nodes
  const auto mesh = RadialMesh::Create(*model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  EXPECT_GT(mesh->ElementCount() - 1, chunk);
  const auto written = io::WriteModuli(
      dir_ / "moduli.nc", *model, SurfaceGrid(8),
      [](double, double theta, double) { return 0.1 * std::cos(theta); }, {},
      {.name = "toy moduli", .elementMetres = 500e3});
  ASSERT_TRUE(written) << written.error().message;

  const auto run = [&](const std::string& name, std::string_view cache) {
    const auto text = std::format(R"({{
      "model": {{"deck": "ball.deck", "moduli": {{"file": "moduli.nc"}}}},
      "lmax": {},
      "forcing": {{
        "load": [{{"cap": {{"latitude": 60, "longitude": 30, "width_deg": 10,
                           "amplitude_kg_m2": 1.0e6}}}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                  "imaginary_m2_s2": 0.5}}]}},{}
      "output": {{"surface": "{}/response.nc",
                 "exchange": "{}/response.txt",
                 "volume": "{}/response_volume.nc"}}}})",
                                  kLmax, cache, name, name, name);
    const auto summary = app::RunResponse(Write(name + ".json", text));
    EXPECT_NE(summary.find("conjugate-gradient"), std::string::npos) << summary;
  };
  run("standard", "");
  run("small", R"( "solver": {"cache_mb": 1},)");

  namespace pm = planetmodel;
  const auto standard = pm::ReadSurfaceFile(dir_ / "standard" / "response.nc");
  const auto small = pm::ReadSurfaceFile(dir_ / "small" / "response.nc");
  for (const auto& [name, field] : standard.variables) {
    EXPECT_EQ(field.values, small.variables.at(name).values) << name;
  }
  EXPECT_EQ(standard.meta.at("solve_iterations"),
            small.meta.at("solve_iterations"));
  EXPECT_EQ(standard.meta.at("solve_residual"),
            small.meta.at("solve_residual"));

  // the exchange tables' rows, the coefficients at 17 digits (the comment
  // lines name the config)
  const auto rows = [&](const std::string& name) {
    auto in = std::ifstream(dir_ / name / "response.txt");
    auto out = std::vector<std::string>{};
    for (std::string line; std::getline(in, line);) {
      if (!line.starts_with("#")) out.push_back(line);
    }
    return out;
  };
  const auto standardRows = rows("standard");
  EXPECT_GT(standardRows.size(), static_cast<std::size_t>(kLmax));
  EXPECT_EQ(standardRows, rows("small"));

  const auto standardVolume =
      pm::ReadModelFile(dir_ / "standard" / "response_volume.nc");
  const auto smallVolume =
      pm::ReadModelFile(dir_ / "small" / "response_volume.nc");
  ASSERT_EQ(standardVolume.Parts().size(), smallVolume.Parts().size());
  for (auto p = std::size_t{0}; p < standardVolume.Parts().size(); ++p) {
    const auto& held = smallVolume.Parts()[p].variables;
    for (const auto& [name, var] : standardVolume.Parts()[p].variables) {
      EXPECT_EQ(var.values, held.at(name).values) << name << " in part " << p;
    }
  }
}

TEST_F(AppsTest, ResponseFieldTermMatchesTheCap) {
  // A field term holding a cap's field gives the cap's response. Written on
  // the run's own Gauss-Legendre grid, the file's round trip is exact but
  // for rounding: the lateral rule gives a field back at its own nodes.
  // Written on a 1-degree equiangular grid and read by the cubic rule, it
  // carries the spline's error onto the run's grid, fourth order in the
  // spacing over the cap's width: about 3e-8 of the 20-degree cap's
  // amplitude in the load, and 1.5e-8 in the response, held below 1e-6.
  namespace pm = planetmodel;
  constexpr auto kLmax = 8;
  const auto site = GeographicPoint{60.0, -85.0};
  const auto width = 20.0 * std::numbers::pi / 180.0;
  const auto amplitude = 1.0e6;
  const auto density = pm::Dimensions{.mass = 1, .length = -2};
  const SurfaceGrid grid(kLmax);
  auto own = pm::SurfaceFile{};
  own.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", kLmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  own.source = "sl3dpp tests";
  own.variables.emplace(
      "load", pm::SurfaceField("load", app::CapField(grid, site, width, amplitude),
                               {.dimensions = density}));
  pm::WriteSurfaceFile(dir_ / "own.nc", own);
  // the cap at the nodes of a 1-degree grid with both poles, by its formula
  const auto fine = pm::Equiangular(181, 360, true);
  auto values = std::vector<double>{};
  const auto colat0 = site.CoLatitude(), lon0 = site.Longitude();
  for (const auto theta : fine.Colatitudes()) {
    for (const auto phi : fine.Longitudes()) {
      const auto c = std::cos(theta) * std::cos(colat0) +
                     std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
      const auto d = std::acos(std::clamp(c, -1.0, 1.0));
      values.push_back(amplitude * std::exp(-0.5 * (d / width) * (d / width)));
    }
  }
  auto other = pm::SurfaceFile{};
  other.grid = fine;
  other.source = "sl3dpp tests";
  other.variables.emplace(
      "sigma", pm::SurfaceField("sigma", values,
                                {.lateral_interpolation = "cubic",
                                 .dimensions = density}));
  pm::WriteSurfaceFile(dir_ / "fine.nc", other);

  const auto cap = Respond("cap", R"({"load": [{"cap": {"latitude": 60,
      "longitude": -85, "width_deg": 20, "amplitude_kg_m2": 1.0e6}}]})");
  const auto same = Respond("same", R"({"load": [{"field": {"file": "own.nc"}}]})");
  const auto resampled = Respond("resampled", R"({"load": [{"field":
      {"file": "fine.nc", "name": "sigma"}}]})");
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto& want = cap.variables.at(name).values;
    EXPECT_LT(Departure(same.variables.at(name).values, want), 1e-12) << name;
    const auto departure = Departure(resampled.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-6) << name;
  }
  RecordProperty("resampled_departure", std::format("{:.3e}", worst));
  EXPECT_LT(Departure(same.variables.at("load").values,
                      cap.variables.at("load").values),
            1e-14);
  const auto loadDeparture = Departure(resampled.variables.at("load").values,
                                       cap.variables.at("load").values);
  RecordProperty("resampled_load_departure", std::format("{:.3e}", loadDeparture));
  EXPECT_LT(loadDeparture, 1e-6);
}

TEST_F(AppsTest, ResponseCoefficientsTermMatchesTheCap) {
  // A coefficient table holding a cap's coefficients, at 17 digits, gives
  // the cap's response to rounding; absent degrees are zero, so a table of
  // degrees 0 to 4 alone is the cap truncated there.
  constexpr auto kLmax = 8;
  const auto site = GeographicPoint{60.0, -85.0};
  const SurfaceGrid grid(kLmax);
  const auto coefficients = grid.Expand(
      app::CapField(grid, site, 20.0 * std::numbers::pi / 180.0, 1.0e6));
  const auto table = [&](int top) {
    auto text = std::string{"# a Gaussian cap's coefficients\n#\n"};
    for (auto l = 0; l <= top; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto c = coefficients[SurfaceGrid::Index(l, m)];
        text += std::format("{} {} {:.17g} {:.17g}\n", l, m, c.real(), c.imag());
      }
    }
    return text;
  };
  static_cast<void>(Write("cap_lm.txt", table(kLmax)));
  static_cast<void>(Write("cap_lm4.txt", table(4)));
  const auto cap = Respond("cap", R"({"load": [{"cap": {"latitude": 60,
      "longitude": -85, "width_deg": 20, "amplitude_kg_m2": 1.0e6}}]})");
  const auto read = Respond(
      "read", R"({"load": [{"coefficients": {"file": "cap_lm.txt"}}]})");
  for (const auto* name : kResponseFields) {
    EXPECT_LT(Departure(read.variables.at(name).values,
                        cap.variables.at(name).values),
              1e-12)
        << name;
  }
  // the load written is the table's synthesis, whose expansion is the cap's
  EXPECT_LT(Departure(grid.Synthesise(coefficients),
                      read.variables.at("load").values),
            1e-13);

  auto truncated = coefficients;
  for (auto l = 5; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) truncated[SurfaceGrid::Index(l, m)] = 0.0;
  }
  const auto four = Respond(
      "four", R"({"load": [{"coefficients": {"file": "cap_lm4.txt"}}]})");
  EXPECT_LT(Departure(four.variables.at("load").values,
                      grid.Synthesise(truncated)),
            1e-13);
}

TEST_F(AppsTest, ResponseLoadTermsAndTideCoefficientsAdd) {
  // The forcing is linear in its terms: two load terms in one run (a cap and
  // a disk) give the sum of the two single-term runs, and so do two tide
  // coefficients, to rounding (the radial model's solves are direct).
  const auto capTerm = std::string{R"({"cap": {"latitude": 60,
      "longitude": -85, "width_deg": 20, "amplitude_kg_m2": 1.0e6}})"};
  const auto diskTerm = std::string{R"({"disk": {"latitude": -75,
      "longitude": 10, "radius_deg": 15, "amplitude_kg_m2": -9.0e5}})"};
  const auto tide21 = std::string{R"({"degree": 2, "order": 1,
      "amplitude_m2_s2": 1.0, "imaginary_m2_s2": -0.5})"};
  const auto tide30 = std::string{R"({"degree": 3, "order": 0,
      "amplitude_m2_s2": 2.0})"};
  const auto check = [](const planetmodel::SurfaceFile& both,
                        const planetmodel::SurfaceFile& a,
                        const planetmodel::SurfaceFile& b, auto&& names) {
    for (const auto* name : names) {
      const auto& x = a.variables.at(name).values;
      const auto& y = b.variables.at(name).values;
      auto sum = std::vector<double>(x.size());
      for (auto i = std::size_t{0}; i < x.size(); ++i) sum[i] = x[i] + y[i];
      EXPECT_LT(Departure(both.variables.at(name).values, sum), 1e-12) << name;
    }
  };
  const auto cap = Respond("cap", "{\"load\": [" + capTerm + "]}");
  const auto disk = Respond("disk", "{\"load\": [" + diskTerm + "]}");
  const auto loads =
      Respond("loads", "{\"load\": [" + capTerm + ", " + diskTerm + "]}");
  check(loads, cap, disk, kResponseFields);
  check(loads, cap, disk, std::initializer_list<const char*>{"load"});

  const auto one = Respond("one", "{\"tide\": [" + tide21 + "]}");
  const auto other = Respond("other", "{\"tide\": [" + tide30 + "]}");
  const auto tides =
      Respond("tides", "{\"tide\": [" + tide21 + ", " + tide30 + "]}");
  check(tides, one, other, kResponseFields);
  EXPECT_FALSE(tides.variables.contains("load"));

  // the summary says what the config was taken to mean, and the metadata
  // count the terms
  static_cast<void>(Deck());
  const auto config = Write("all.json", "{\"model\": {\"deck\": \"ball.deck\"}, "
                                        "\"lmax\": 8, \"forcing\": {\"load\": [" +
                                            capTerm + ", " + diskTerm +
                                            "], \"tide\": [" + tide21 + ", " +
                                            tide30 + "]}, \"output\": "
                                            "{\"surface\": \"out/all.nc\", "
                                            "\"exchange\": \"out/all.txt\"}}");
  const auto summary = app::RunResponse(config);
  EXPECT_TRUE(summary.starts_with("1-D, elastic, load (2 terms) + tide (2 "
                                  "coefficients); wrote "))
      << summary;
  const auto all = planetmodel::ReadSurfaceFile(dir_ / "out" / "all.nc");
  EXPECT_EQ(std::get<std::vector<long long>>(all.meta.at("load_terms")),
            std::vector<long long>{2});
  EXPECT_EQ(std::get<std::vector<long long>>(all.meta.at("tide_coefficients")),
            std::vector<long long>{2});
  EXPECT_FALSE(all.meta.contains("load_latitude_deg"));
  EXPECT_FALSE(all.meta.contains("tide_degree"));
  auto exchange = std::ifstream(dir_ / "out" / "all.txt");
  auto scalars = std::vector<std::string>{};
  for (std::string line; std::getline(exchange, line);) {
    if (line.empty() || line.starts_with("#") ||
        std::isdigit(static_cast<unsigned char>(line.front()))) {
      continue;
    }
    scalars.push_back(line.substr(0, line.find(' ')));
  }
  EXPECT_EQ(scalars, (std::vector<std::string>{
                         "radius", "surface_gravity", "lmax", "lateral",
                         "load_terms", "tide_coefficients", "columns"}));
}

TEST_F(AppsTest, DiskIsExpandedExactly) {
  // The disk's coefficients are the truncated exact expansion in the grid's
  // own harmonics: the harmonics at a point are the grid's synthesis there;
  // the mean is the amplitude times the disk's share of the sphere,
  // (1 - cos alpha) / 2; a disk about a pole has m = 0 terms alone; the
  // coefficients are the disk's integrals of conj(Y_lm) by a quadrature in
  // the disk's own frame, exact for the band, to rounding; and they agree
  // with the expansion of the disk sampled on a much finer grid, to that
  // sampling's error across the disk's edge.
  constexpr auto kLmax = 32;
  const SurfaceGrid grid(kLmax);
  const auto size = grid.CoefficientSize();

  // the harmonics at a point against the grid's synthesis of a unit
  // coefficient (2 Re Y_lm for m > 0, a real field; -2 Im Y_lm for i)
  auto worstHarmonic = 0.0;
  const auto nphi = grid.Longitudes().size();
  for (auto l = 0; l <= kLmax; l += 3) {
    for (auto m = 0; m <= l; m += 2) {
      for (const auto unit : {Complex{1.0, 0.0}, Complex{0.0, 1.0}}) {
        if (m == 0 && unit.imag() != 0.0) continue;
        auto coefficients = grid.ZeroCoefficients();
        coefficients[SurfaceGrid::Index(l, m)] = unit;
        const auto field = grid.Synthesise(coefficients);
        for (const auto i : {std::size_t{0}, std::size_t{7}, std::size_t{20}}) {
          for (const auto j : {std::size_t{0}, std::size_t{5}, nphi - 3}) {
            const auto point = GeographicPoint::FromSpherical(
                grid.CoLatitudes()[i], grid.Longitudes()[j]);
            const auto y = app::HarmonicsAt(kLmax, point)[SurfaceGrid::Index(l, m)];
            const auto want = m == 0 ? y.real() : 2.0 * (unit * y).real();
            worstHarmonic =
                std::max(worstHarmonic, std::abs(field[i * nphi + j] - want));
          }
        }
      }
    }
  }
  RecordProperty("harmonic_departure", std::format("{:.3e}", worstHarmonic));
  EXPECT_LT(worstHarmonic, 1e-13);

  const auto centre = GeographicPoint{55.0, -100.0};
  const auto alpha = 8.0 * std::numbers::pi / 180.0;
  const auto amplitude = 2.5e3;
  const auto disk = app::DiskCoefficients(grid, centre, alpha, amplitude);
  ASSERT_EQ(disk.size(), size);
  // the grid's degree-0 normalisation: a unit field's coefficient
  const auto unit = grid.Expand(std::vector<double>(grid.FieldSize(), 1.0))[0];
  const auto mean = amplitude * (1.0 - std::cos(alpha)) / 2.0;
  EXPECT_NEAR(disk[0].real(), mean * unit.real(), 1e-14 * mean * unit.real());
  EXPECT_EQ(disk[0].imag(), 0.0);

  // about either pole, m = 0 alone (at the south pole to the rounding of
  // its colatitude, whose sine is 1.2e-16 rather than zero)
  for (const auto latitude : {90.0, -90.0}) {
    const auto polar = app::DiskCoefficients(grid, {latitude, 30.0}, alpha, 1.0);
    for (auto l = 0; l <= kLmax; ++l) {
      EXPECT_NE(polar[SurfaceGrid::Index(l, 0)], Complex{}) << l;
      for (auto m = 1; m <= l; ++m) {
        EXPECT_LT(std::abs(polar[SurfaceGrid::Index(l, m)]),
                  1e-14 * std::abs(polar[0]))
            << latitude << " " << l << " " << m;
      }
    }
  }

  // the disk's integrals of conj(Y_lm), in the disk's own frame (polar
  // angle t about the centre, azimuth s): Gauss-Legendre in cos t over
  // [cos alpha, 1] and the trapezoid in s are exact for the band
  const auto theta0 = centre.CoLatitude(), phi0 = centre.Longitude();
  const auto e3 = std::array{std::sin(theta0) * std::cos(phi0),
                             std::sin(theta0) * std::sin(phi0), std::cos(theta0)};
  const auto e1 = std::array{std::cos(theta0) * std::cos(phi0),
                             std::cos(theta0) * std::sin(phi0), -std::sin(theta0)};
  const auto e2 = std::array{-std::sin(phi0), std::cos(phi0), 0.0};
  const auto rule =
      GaussQuad::GaussLegendreQuadrature1D<double>(kLmax + 2, std::cos(alpha), 1.0);
  const auto azimuths = 2 * kLmax + 2;
  auto quadrature = std::vector<Complex>(size);
  for (auto q = std::size_t{0}; q < rule.Points().size(); ++q) {
    const auto u = rule.Points()[q];
    const auto v = std::sqrt(1.0 - u * u);
    for (auto k = 0; k < azimuths; ++k) {
      const auto s = 2.0 * std::numbers::pi * k / azimuths;
      auto n = std::array<double, 3>{};
      for (auto c = 0; c < 3; ++c) {
        n[c] = v * std::cos(s) * e1[c] + v * std::sin(s) * e2[c] + u * e3[c];
      }
      const auto point = GeographicPoint::FromSpherical(
          std::acos(std::clamp(n[2], -1.0, 1.0)), std::atan2(n[1], n[0]));
      const auto y = app::HarmonicsAt(kLmax, point);
      const auto weight =
          amplitude * rule.Weights()[q] * 2.0 * std::numbers::pi / azimuths;
      for (auto i = std::size_t{0}; i < size; ++i) {
        quadrature[i] += weight * std::conj(y[i]);
      }
    }
  }
  auto worstQuadrature = 0.0;
  for (auto i = std::size_t{0}; i < size; ++i) {
    worstQuadrature = std::max(worstQuadrature, std::abs(disk[i] - quadrature[i]));
  }
  worstQuadrature /= std::abs(disk[0]);
  RecordProperty("quadrature_departure", std::format("{:.3e}", worstQuadrature));
  EXPECT_LT(worstQuadrature, 1e-13);

  // the disk sampled on much finer grids, expanded and cut to the band: the
  // sampling's error across the edge falls with the fine grid's spacing
  // (about 2.6e-2 of the mean coefficient at degree 128 and 7.2e-3 at 256),
  // where a wrong phase or normalisation would leave a departure of order 1
  const auto cosAlpha = std::cos(alpha);
  const auto sampledDeparture = [&](int degree) {
    const SurfaceGrid fine(degree);
    const auto sampled =
        fine.Expand(fine.FromFunction([&](double theta, double phi) {
          const auto c = std::cos(theta) * std::cos(theta0) +
                         std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
          return c >= cosAlpha ? amplitude : 0.0;
        }));
    auto worst = 0.0;
    for (auto i = std::size_t{0}; i < size; ++i) {
      worst = std::max(worst, std::abs(disk[i] - sampled[i]));
    }
    const auto departure = worst / std::abs(disk[0]);
    RecordProperty(std::format("sampled_departure_{}", degree),
                   std::format("{:.3e}", departure));
    return departure;
  };
  const auto coarser = sampledDeparture(128);
  const auto finer = sampledDeparture(256);
  EXPECT_LT(finer, 1e-2);
  EXPECT_LT(finer, 0.5 * coarser);
}

TEST_F(AppsTest, ResponseRefusesWhatItCannotDo) {
  // Each fault is refused naming the member: an unknown member anywhere, a
  // load term of two kinds or none, an amplitude history or a viscosity
  // without a times block, a times block without its end, an empty
  // forcing, the old layout's members; and, once the run reads its files,
  // a field file with a time axis and a coefficient table above lmax.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  static_cast<void>(Write("high.txt", "# degree 5 at lmax 4\n0 0 1 0\n5 2 1 1\n"));
  const auto cap = std::string{
      R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                  "amplitude_kg_m2": 1.0}})"};
  const auto config = [&](const std::string& model, const std::string& forcing,
                          const std::string& extra = "") {
    return std::format(R"({{"model": {}, "lmax": 4, "forcing": {},{}
                             "output": {{"surface": "out.nc"}}}})",
                       model, forcing, extra);
  };
  const auto ball = std::string{R"({"deck": "ball.deck"})"};
  const auto loads = [](const std::string& terms) {
    return "{\"load\": [" + terms + "]}";
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    const auto path = Write("bad.json", text);
    EXPECT_THROW(
        {
          try {
            static_cast<void>(run(path));
          } catch (const planetmodel::Error& error) {
            const auto message = std::string{error.what()};
            EXPECT_NE(message.find(member), std::string::npos) << message;
            EXPECT_NE(message.find(what), std::string::npos) << message;
            throw;
          }
        },
        planetmodel::Error)
        << text;
  };
  const auto read = [](const fs::path& path) {
    return app::ReadResponseInput(path);
  };
  const auto run = [](const fs::path& path) { return app::RunResponse(path); };

  refused(read, config(ball, loads(cap), R"( "lmxa": 4,)"), "lmxa",
          "not a recognised member");
  refused(read,
          config(ball, loads(R"({"cap": {"latitude": 60, "longitude": 0,
                                 "width": 10, "amplitude_kg_m2": 1.0}})")),
          "forcing.load[0].cap.width", "not a recognised member");
  refused(read, config(R"({"deck": "ball.deck", "moduli": {"fil": "m.nc"}})",
                       loads(cap)),
          "model.moduli.fil", "not a recognised member");
  refused(read,
          std::format(R"({{"model": {}, "lmax": 4, "forcing": {},
                          "output": {{"surface": "out.nc", "volum": "v.nc"}}}})",
                      ball, loads(cap)),
          "output.volum", "not a recognised member");
  refused(read, config(ball, loads(R"({"cap": {"latitude": 60, "longitude": 0,
      "width_deg": 10, "amplitude_kg_m2": 1.0}, "disk": {"latitude": 0,
      "longitude": 0, "radius_deg": 5, "amplitude_kg_m2": 1.0}})")),
          "forcing.load[0]", "names several terms");
  refused(read, config(ball, loads(cap + ", {}")), "forcing.load[1]",
          "names no term");
  refused(read,
          config(ball, loads(R"({"cap": {"latitude": 60, "longitude": 0,
                                 "width_deg": 10, "amplitude_kg_m2": 1.0},
                                 "history_kyr": [[0, 1]]})")),
          "forcing.load[0].history_kyr", "needs a `times` block");
  refused(read, config(ball, loads(cap), R"( "times": {"nodes": 3},)"), "times",
          "gives `nodes` without `end_kyr`");
  refused(read,
          config(R"({"deck": "ball.deck", "viscosity": {"file": "v.nc"}})",
                 loads(cap)),
          "model.viscosity", "needs a `times` block");
  refused(read, config(ball, "{}"), "forcing", "needs at least one");
  refused(read, config(ball, R"({"load": [], "tide": []})"), "forcing",
          "needs at least one");
  refused(read,
          config(ball, R"({"tide": [{"degree": 2, "order": 0,
                                     "amplitude_m2_s2": 1.0,
                                     "imaginary_m2_s2": 1.0}]})"),
          "forcing.tide[0].imaginary_m2_s2", "not zero at order 0");
  refused(read,
          config(ball, loads(R"({"disk": {"latitude": 0, "longitude": 0,
                                 "radius_deg": 0, "amplitude_kg_m2": 1.0}})")),
          "forcing.load[0].disk.radius_deg", "is not an angle");
  // the old layout's members
  refused(read,
          R"({"model": "ball.deck", "lmax": 4, "tide": {"degree": 2,
              "amplitude_m2_s2": 1.0}, "output": "out.nc"})",
          "tide", "not a recognised member");

  refused(run,
          config(ball, loads(R"({"field": {"file": "history.nc",
                                           "name": "ice_thickness"}})")),
          "forcing.load[0].field.file", "has a time axis");
  refused(run, config(ball, loads(R"({"field": {"file": "history.nc"}})")),
          "forcing.load[0].field.file", "needs a `times` block");
  refused(run,
          config(ball, loads(R"({"coefficients": {"file": "high.txt"}})")),
          "forcing.load[0].coefficients.file", "line 3: degree 5 exceeds lmax = 4");
  static_cast<void>(Write("imaginary.txt", "0 0 1 0.5\n"));
  refused(run,
          config(ball, loads(R"({"coefficients": {"file": "imaginary.txt"}})")),
          "forcing.load[0].coefficients.file",
          "line 1: the imaginary part at order 0 is not zero");
  static_cast<void>(Write("short.txt", "2 1 1\n"));
  refused(run, config(ball, loads(R"({"coefficients": {"file": "short.txt"}})")),
          "forcing.load[0].coefficients.file", "line 1: is not a row");
}

/// The ball's model block and degree, for a viscoelastic response config.
constexpr auto kBall = R"("model": {"deck": "ball.deck"}, "lmax": 8)";

/// A cap and a degree-2 tide coefficient, as a forcing's terms.
constexpr auto kCap = R"({"cap": {"latitude": 60, "longitude": -85,
    "width_deg": 20, "amplitude_kg_m2": 1.0e6}})";
constexpr auto kTide = R"({"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
    "imaginary_m2_s2": 0.5})";

/// The prem8 deck with the test viscosity file, at lmax 8: the 3D runs'
/// model block (the moduli member appended where given).
std::string Prem8(const std::string& moduli = "") {
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  return std::format(R"("model": {{"deck": "{}/prem8.elastic.deck",
      "format": "elastic", "viscosity": {{"file": "{}/viscosity_log.nc"}}{}}},
      "lmax": 8)",
                     data, data, moduli);
}

/// The test's models: the ball's, and the prem8 deck's.
EarthModel ModelOf(const fs::path& deck, app::ModelFormat format) {
  const auto read = sl3dpp::Deck::Read(deck, app::FormatOf(format));
  EXPECT_TRUE(read) << read.error().message;
  auto model = EarthModel::FromDeck(*read);
  EXPECT_TRUE(model) << model.error().message;
  return std::move(*model);
}

TEST_F(AppsTest, ViscoelasticResponseStartsFromTheElasticResponse) {
  // The first node of a viscoelastic run is the unrelaxed solve under the
  // forcing there with no memory before it: the elastic response of the
  // same forcing. On the radial ball the radial stepper's unrelaxed
  // systems (the branch moduli at the memory points) and the elastic
  // command's agree to rounding, about 6e-12; under lateral moduli both
  // close the 3D problem by conjugate gradients on the radial systems, to
  // the same tolerance (1e-10), and agree well within it (about 7e-12). A load
  // ramped up from zero has no response at the first node. The surface file has
  // a slice per node, its time axis the run's clock in seconds.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  const auto forcing =
      std::format(R"({{"load": [{}], "tide": [{}]}})", kCap, kTide);
  const auto elastic = Respond("elastic", forcing);
  const auto summary =
      Run("visco", std::format(R"({}, "times": {{"end_kyr": 2, "nodes": 5}},
                              "forcing": {})",
                               kBall, forcing));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 0 to 2 kyr, load (1 term) + tide (1 "
      "coefficient); wrote "))
      << summary;
  EXPECT_TRUE(summary.ends_with("and 5 exchange tables")) << summary;
  const auto visco = Output("visco");
  ASSERT_TRUE(visco.time);
  ASSERT_EQ(visco.time->size(), 5u);
  for (auto n = std::size_t{0}; n < 5; ++n) {
    EXPECT_DOUBLE_EQ((*visco.time)[n], 0.5 * static_cast<double>(n) * app::KYR);
  }
  EXPECT_EQ(std::get<std::vector<double>>(visco.meta.at("dates_kyr")),
            (std::vector<double>{0.0, 0.5, 1.0, 1.5, 2.0}));
  EXPECT_EQ(pm::detail::TextOf(visco.variables.at("displacement").attributes,
                               "time_support")
                .value_or(""),
            "node");
  EXPECT_FALSE(visco.variables.contains("displacement_radial_model"));
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto departure =
        Departure(Slice(visco, name, 0), elastic.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-10) << name;
  }
  RecordProperty("radial_departure", std::format("{:.3e}", worst));
  EXPECT_EQ(Slice(visco, "load", 0), elastic.variables.at("load").values);
  // and the response relaxes from there
  EXPECT_GT(Departure(Slice(visco, "displacement", 4),
                      Slice(visco, "displacement", 0)),
            1e-2);

  // under lateral moduli, the 3D problem
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", model, SurfaceGrid(8),
      [](double, double theta, double) { return 0.1 * std::cos(theta); }, {},
      {.name = "toy moduli", .elementMetres = 500e3}));
  const auto lateral = std::string{
      R"("model": {"deck": "ball.deck", "moduli": {"file": "moduli.nc"}},
         "lmax": 8)"};
  static_cast<void>(app::RunResponse(
      Write("elastic3d.json",
            std::format(R"({{{}, "forcing": {}, "output": {{"surface":
                       "out/elastic3d.nc"}}}})",
                        lateral, forcing))));
  const auto lateralSummary =
      Run("visco3d", std::format(R"({}, "times": {{"end_kyr": 1, "nodes": 2}},
                                    "forcing": {})",
                                 lateral, forcing));
  EXPECT_TRUE(lateralSummary.starts_with(
      "3-D (lateral moduli), viscoelastic, 2 nodes 0 to 1 kyr, "))
      << lateralSummary;
  const auto elastic3d = Output("elastic3d");
  const auto visco3d = Output("visco3d");
  auto worst3d = 0.0;
  for (const auto* name :
       {"displacement", "tangential_potential", "potential",
        "displacement_radial_model", "potential_radial_model"}) {
    const auto departure =
        Departure(Slice(visco3d, name, 0), elastic3d.variables.at(name).values);
    worst3d = std::max(worst3d, departure);
    EXPECT_LT(departure, 1e-10) << name;
  }
  RecordProperty("lateral_departure", std::format("{:.3e}", worst3d));
  const auto iterations =
      std::get<std::vector<long long>>(visco3d.meta.at("step_iterations"));
  EXPECT_GT(iterations.front(), 0);  // the first node's conjugate gradients

  // a load ramped up from zero: nothing at the first node
  static_cast<void>(
      Run("ramp", std::format(R"({}, "times": {{"end_kyr": 2, "nodes": 3}},
          "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
              "width_deg": 20, "amplitude_kg_m2": 1.0e6}},
              "history_kyr": [[0, 0], [2, 1]]}}]}})",
                              kBall)));
  const auto ramp = Output("ramp");
  for (const auto* name : {"load", "displacement", "potential"}) {
    const auto first = Slice(ramp, name, 0);
    EXPECT_TRUE(std::ranges::all_of(first, [](double v) { return v == 0.0; }))
        << name;
  }
  const auto last = Slice(ramp, "displacement", 2);
  EXPECT_TRUE(std::ranges::any_of(last, [](double v) { return v != 0.0; }));
}

TEST_F(AppsTest, ViscoelasticResponseMatchesTheLibraryStepper) {
  // The command is the library's stepper under the forcing at each node:
  // a loop written here, the radial ball's ViscoelasticOperator stepped
  // per degree (the orders as columns) and the 3D Viscoelastic3D::Step on
  // the prem8 deck's lateral viscosity, each under the cap's coefficients
  // and the tide's scaled by their histories at the nodes and made
  // non-dimensional by the model's scales, gives the command's exchange
  // tables to 1e-10 relative (the same algorithm: the departures are
  // rounding, about 6e-16). Uneven nodes, so that each step's length
  // counts.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  const auto nodes = std::vector<double>{0.0, 0.4, 1.0, 1.5};
  const auto capFactor = [](double t) {
    return t <= 0.0 ? 0.5 : (t >= 1.0 ? 1.0 : 0.5 + 0.5 * t);
  };
  const auto tideFactor = [](double t) {
    return t >= 1.5 ? -1.0 : 1.0 - 2.0 * t / 1.5;
  };
  const auto forcing = std::string{R"("forcing": {
      "load": [{"cap": {"latitude": 60, "longitude": -85, "width_deg": 20,
                        "amplitude_kg_m2": 1.0e6},
                "history_kyr": [[0, 0.5], [1, 1]]}],
      "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                "imaginary_m2_s2": 0.5, "history_kyr": [[0, 1], [1.5, -1]]}]})"};
  const auto times = std::string{R"("times": {"at_kyr": [0, 0.4, 1.0, 1.5]})"};
  const SurfaceGrid grid(kLmax);
  const auto cap = grid.Expand(app::CapField(
      grid, {60.0, -85.0}, 20.0 * std::numbers::pi / 180.0, 1.0e6));
  const auto tideIndex = SurfaceGrid::Index(2, 1);

  // the command's tables, node by node
  const auto compare = [&](const std::string& name, std::size_t n,
                           const std::vector<Complex>& U,
                           const std::vector<Complex>& Phi) {
    const auto table =
        ReadExchange(dir_ / "out" / std::format("{}_{}.txt", name, n));
    const auto departure =
        std::max(Departure(table.U, U), Departure(table.Phi, Phi));
    EXPECT_LT(departure, 1e-10) << name << " node " << n;
    return departure;
  };

  // 1D: the ball's radial stepper
  {
    EXPECT_TRUE(Run("radial", std::format("{}, {}, {}", kBall, times, forcing))
                    .starts_with("1-D, viscoelastic, 4 nodes 0 to 1.5 kyr"));
    const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
    const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
    ASSERT_TRUE(mesh);
    const auto material = Material::Create(model, *mesh);
    ASSERT_TRUE(material);
    const auto rheology = Rheology::FromModel(*material, model);
    ASSERT_TRUE(rheology);
    const ViscoelasticOperator op(*rheology, kLmax);
    const auto& scales = model.ModelScales();
    const auto density = scales.Factor(Dimensions{.mass = 1, .length = -2});
    const auto potential = scales.Factor(PotentialDimensions);
    const auto toTime = app::KYR / scales.Time();
    const auto columns = [](std::vector<Complex> c, double factor,
                            int only) -> LoadCallable {
      return [c = std::move(c), factor, only](
                 int l, double) -> std::optional<std::vector<double>> {
        if (only >= 0 && l != only) return std::nullopt;
        auto out = std::vector<double>(2 * (l + 1));
        for (auto m = 0; m <= l; ++m) {
          out[m] = factor * c[SurfaceGrid::Index(l, m)].real();
          out[l + 1 + m] = factor * c[SurfaceGrid::Index(l, m)].imag();
        }
        return out;
      };
    };
    auto tide = grid.ZeroCoefficients();
    tide[tideIndex] = {1.0, 0.5};
    const auto channels = [&](double t) {
      return std::pair{columns(cap, capFactor(t) / density, -1),
                       columns(tide, tideFactor(t) / potential, 2)};
    };
    const auto surface = [&](const std::map<int, std::vector<double>>& x,
                             Component component, double factor) {
      auto out = grid.ZeroCoefficients();
      for (const auto& [l, c] : op.Surface(x, component)) {
        for (auto m = 0; m <= l; ++m) {
          out[SurfaceGrid::Index(l, m)] = factor * Complex{c[m], c[l + 1 + m]};
        }
      }
      return out;
    };
    auto state = op.ZeroState([](int l) { return 2 * (l + 1); });
    auto now = channels(nodes[0]);
    auto x = op.SolveElastic(state, now.first, now.second);
    state.strain = op.Strain(x);
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < nodes.size(); ++n) {
      if (n > 0) {
        auto next = channels(nodes[n]);
        const auto step =
            op.StartStep(state, (nodes[n] - nodes[n - 1]) * toTime,
                         Scheme::ExponentialTrapezoid, now.first, now.second);
        auto [after, solutions] = step.Finish(next.first, next.second);
        state = std::move(after);
        x = std::move(solutions);
        now = std::move(next);
      }
      worst = std::max(
          worst, compare("radial", n, surface(x, Component::U, scales.Length()),
                         surface(x, Component::Phi, potential)));
    }
    RecordProperty("radial_departure", std::format("{:.3e}", worst));
  }

  // 3D: the prem8 deck's lateral viscosity, Viscoelastic3D::Step
  {
    EXPECT_TRUE(
        Run("lateral", std::format("{}, {}, {}", Prem8(), times, forcing))
            .starts_with("3-D (lateral viscosity), viscoelastic, 4 "
                         "nodes 0 to 1.5 kyr"));
    const auto data = std::string{SL3DPP_TEST_DATA_DIR};
    const auto model =
        ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
    const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
    ASSERT_TRUE(mesh);
    const auto material = Material::Create(model, *mesh);
    ASSERT_TRUE(material);
    const auto viscosity =
        io::ModelViscosity::Read(data + "/viscosity_log.nc", model);
    ASSERT_TRUE(viscosity);
    const auto rheology = Rheology::Maxwell(*material, viscosity->Viscosity());
    ASSERT_TRUE(rheology);
    auto options = Viscoelastic3D::Options{};
    options.rtol = 1e-10;
    options.max_iterations = 200;
    const Viscoelastic3D op(*rheology, kLmax, viscosity->Delta(), options);
    const auto& scales = model.ModelScales();
    const auto density = scales.Factor(Dimensions{.mass = 1, .length = -2});
    const auto potential = scales.Factor(PotentialDimensions);
    const auto toTime = app::KYR / scales.Time();
    const auto columns = [](std::vector<Complex> c, double factor,
                            int only) -> Viscoelastic3D::LoadCallable3D {
      return [c = std::move(c), factor, only](
                 int l, double) -> std::optional<std::vector<Complex>> {
        if (only >= 0 && l != only) return std::nullopt;
        auto out = std::vector<Complex>(l + 1);
        for (auto m = 0; m <= l; ++m) {
          out[m] = factor * c[SurfaceGrid::Index(l, m)];
        }
        return out;
      };
    };
    auto tide = grid.ZeroCoefficients();
    tide[tideIndex] = {1.0, 0.5};
    const auto channels = [&](double t) {
      return std::pair{columns(cap, capFactor(t) / density, -1),
                       columns(tide, tideFactor(t) / potential, 2)};
    };
    const auto surface = [&](const Viscoelastic3D::Solutions& x,
                             Component component, double factor) {
      auto out = grid.ZeroCoefficients();
      for (const auto& [l, c] : op.Surface(x, component)) {
        for (auto m = 0; m <= l; ++m)
          out[SurfaceGrid::Index(l, m)] = factor * c[m];
      }
      return out;
    };
    auto state = op.ZeroState();
    const auto start = channels(nodes[0]);
    auto x = op.SolveElastic(state, start.first, start.second);
    op.PrimeStrain(state, start.first, start.second);
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < nodes.size(); ++n) {
      if (n > 0) {
        const auto next = channels(nodes[n]);
        auto result =
            op.Step(state, (nodes[n] - nodes[n - 1]) * toTime,
                    Scheme::ExponentialTrapezoid, next.first, next.second);
        ASSERT_TRUE(result.info.converged);
        state = std::move(result.state);
        x = std::move(result.solutions);
      }
      worst = std::max(worst, compare("lateral", n,
                                      surface(x, Component::U, scales.Length()),
                                      surface(x, Component::Phi, potential)));
    }
    RecordProperty("lateral_departure", std::format("{:.3e}", worst));
  }
}

TEST_F(AppsTest, ViscoelasticResponseIn3DReducesToTheRadialRun) {
  // A viscosity file whose viscosity is the deck's column everywhere: held
  // as a field of the radius alone it is radial and the run is the radial
  // model's (to rounding, about 3e-12: the same 1e21 Pa s by another
  // route, through the relaxation); held as a field on a grid, laterally
  // uniform, the run is 3D, the lateral perturbation zero to rounding, and
  // it equals the radial run to the fixed point's tolerance (1e-10; about
  // 7e-12 here). Its radial reference model, written beside it, is the
  // radial run to rounding.
  static_cast<void>(Deck());
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  WriteViscosity(dir_ / "radial.nc", model, 1e21, 0.0, true);
  WriteViscosity(dir_ / "uniform.nc", model, 1e21, 0.0);
  const auto rest = std::format(
      R"("times": {{"end_kyr": 2, "nodes": 5}},
         "forcing": {{"load": [{}], "tide": [{}]}})",
      kCap, kTide);
  const auto withFile = [&](const std::string& file) {
    return std::format(R"("model": {{"deck": "ball.deck", "viscosity":
        {{"file": "{}"}}}}, "lmax": 8, {})",
                       file, rest);
  };
  EXPECT_TRUE(Run("deck", std::string{kBall} + ", " + rest)
                  .starts_with("1-D, viscoelastic"));
  const auto radialSummary = Run("radial", withFile("radial.nc"));
  EXPECT_TRUE(radialSummary.starts_with(
      "1-D (the viscosity file is radial), viscoelastic"))
      << radialSummary;
  const auto uniformSummary = Run("uniform", withFile("uniform.nc"));
  EXPECT_TRUE(
      uniformSummary.starts_with("3-D (lateral viscosity), viscoelastic"))
      << uniformSummary;
  const auto deck = Output("deck");
  const auto radial = Output("radial");
  const auto uniform = Output("uniform");
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto& want = deck.variables.at(name).values;
    EXPECT_LT(Departure(radial.variables.at(name).values, want), 1e-11) << name;
    const auto departure = Departure(uniform.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-10) << name;
  }
  RecordProperty("lateral_departure", std::format("{:.3e}", worst));
  EXPECT_LT(Departure(uniform.variables.at("displacement_radial_model").values,
                      deck.variables.at("displacement").values),
            1e-11);
  EXPECT_LT(Departure(uniform.variables.at("potential_radial_model").values,
                      deck.variables.at("potential").values),
            1e-11);
}

TEST_F(AppsTest, ViscoelasticResponseHistoriesScaleTheForcing) {
  // An amplitude history scales its term by the piecewise-linear
  // interpolant of its factors, held at the end factors outside its
  // times: the load written at each node is the factor there times the
  // cap. The response is linear in the forcing's histories: the run under
  // h1 + h2 (the sum of two histories, on the union of their times) is the
  // sum of the runs under h1 and under h2, to rounding, for a load term
  // and a tide coefficient alike.
  static_cast<void>(Deck());
  const auto nodes =
      std::vector<double>{0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
  const auto times = std::string{
      R"("times": {"at_kyr": [0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]})"};
  const auto run = [&](const std::string& name, const std::string& load,
                       const std::string& tide) {
    static_cast<void>(Run(name, std::format(R"({}, {}, "forcing": {{
            "load": [{{"cap": {{"latitude": 60, "longitude": -85,
                                "width_deg": 20, "amplitude_kg_m2": 1.0e6}},
                       "history_kyr": {}}}],
            "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                       "imaginary_m2_s2": 0.5, "history_kyr": {}}}]}})",
                                            kBall, times, load, tide)));
    return Output(name);
  };
  const auto one = run("one", "[[0.5, 0], [2, 1]]", "[[0, 0], [3, 1]]");
  const auto two = run("two", "[[0, 1], [1, 0.2], [3, 0.7]]", "[[1, 2]]");
  const auto both =
      run("both",
          std::format("[[0, 1], [0.5, 0.6], [1, {:.17g}], [2, 1.45], [3, 1.7]]",
                      0.2 + 1.0 / 3.0),
          "[[0, 2], [3, 3]]");

  // the load at each node: the factor of [[0.5, 0], [2, 1]] times the cap
  const SurfaceGrid grid(8);
  const auto cap = app::CapField(grid, {60.0, -85.0},
                                 20.0 * std::numbers::pi / 180.0, 1.0e6);
  for (auto n = std::size_t{0}; n < nodes.size(); ++n) {
    const auto t = nodes[n];
    const auto factor =
        t <= 0.5 ? 0.0 : (t >= 2.0 ? 1.0 : 0.0 + (t - 0.5) / 1.5 * 1.0);
    auto want = cap;
    for (auto& v : want) v *= factor;
    const auto got = Slice(one, "load", n);
    if (factor == 0.0) {
      EXPECT_TRUE(std::ranges::all_of(got, [](double v) { return v == 0.0; }))
          << t;
    } else {
      EXPECT_LT(Departure(got, want), 1e-15) << t;
    }
  }

  auto worst = 0.0;
  for (const auto* name :
       {"load", "displacement", "tangential_potential", "potential"}) {
    const auto& a = one.variables.at(name).values;
    const auto& b = two.variables.at(name).values;
    auto sum = std::vector<double>(a.size());
    for (auto i = std::size_t{0}; i < a.size(); ++i) sum[i] = a[i] + b[i];
    const auto departure = Departure(both.variables.at(name).values, sum);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("linearity_departure", std::format("{:.3e}", worst));
}

TEST_F(AppsTest, ViscoelasticResponseReadsAFieldOnATimeAxis) {
  // A field term whose file has a time axis is read at each node by the
  // file's own rule: a cap ramped linearly over its two dates is the cap
  // term under the matching history. Written on the run's own grid the two
  // agree to rounding; the same file with its axis counted back from the
  // present, read from time_origin_kyr = 2, likewise; and on a 1-degree
  // grid read by the cubic rule, to the resampling's error (about 3e-8, as
  // the static field term's). Past the file's last date its last field is
  // held. A history on a field term multiplies it.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  const auto site = GeographicPoint{60.0, -85.0};
  const auto width = 20.0 * std::numbers::pi / 180.0;
  const auto density = pm::Dimensions{.mass = 1, .length = -2};
  const SurfaceGrid grid(kLmax);
  const auto field = app::CapField(grid, site, width, 1.0e6);
  const auto ramp = [&](const std::string& name, std::vector<double> axis) {
    auto file = pm::SurfaceFile{};
    file.grid = pm::AngularGrid(
        std::vector<double>(grid.CoLatitudes().begin(),
                            grid.CoLatitudes().end()),
        std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
        "gauss_legendre", kLmax,
        std::vector<double>(grid.CoLatitudeWeights().begin(),
                            grid.CoLatitudeWeights().end()));
    file.source = "sl3dpp tests";
    file.time = std::move(axis);
    auto values = std::vector<double>(field.size(), 0.0);
    values.insert(values.end(), field.begin(), field.end());
    file.variables.emplace("load",
                           pm::SurfaceField("load", values,
                                            {.support = "node",
                                             .time_interpolation = "linear",
                                             .dimensions = density}));
    pm::WriteSurfaceFile(dir_ / name, file);
  };
  ramp("ramp.nc", {0.0, 2.0 * app::KYR});
  ramp("ramp_bp.nc", {-2.0 * app::KYR, 0.0});
  // the cap at the nodes of a 1-degree grid with both poles, by its formula
  {
    const auto fine = pm::Equiangular(181, 360, true);
    auto values = std::vector<double>{};
    const auto colat0 = site.CoLatitude(), lon0 = site.Longitude();
    for (const auto theta : fine.Colatitudes()) {
      for (const auto phi : fine.Longitudes()) {
        const auto c =
            std::cos(theta) * std::cos(colat0) +
            std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
        const auto d = std::acos(std::clamp(c, -1.0, 1.0));
        values.push_back(1.0e6 * std::exp(-0.5 * (d / width) * (d / width)));
      }
    }
    auto both = std::vector<double>(values.size(), 0.0);
    both.insert(both.end(), values.begin(), values.end());
    auto file = pm::SurfaceFile{};
    file.grid = fine;
    file.source = "sl3dpp tests";
    file.time = std::vector<double>{0.0, 2.0 * app::KYR};
    file.variables.emplace("sigma",
                           pm::SurfaceField("sigma", both,
                                            {.support = "node",
                                             .time_interpolation = "linear",
                                             .lateral_interpolation = "cubic",
                                             .dimensions = density}));
    pm::WriteSurfaceFile(dir_ / "fine.nc", file);
  }
  const auto run = [&](const std::string& name, const std::string& term) {
    static_cast<void>(Run(name, std::format(R"({}, "times": {{"end_kyr": 3,
        "nodes": 7}}, "forcing": {{"load": [{}]}})",
                                            kBall, term)));
    return Output(name);
  };
  const auto cap = run("cap", R"({"cap": {"latitude": 60, "longitude": -85,
      "width_deg": 20, "amplitude_kg_m2": 1.0e6},
      "history_kyr": [[0, 0], [2, 1]]})");
  const auto own = run("own", R"({"field": {"file": "ramp.nc"}})");
  const auto shifted = run(
      "shifted", R"({"field": {"file": "ramp_bp.nc", "time_origin_kyr": 2}})");
  const auto resampled =
      run("resampled", R"({"field": {"file": "fine.nc", "name": "sigma"}})");
  const auto doubled = run(
      "doubled", R"({"field": {"file": "ramp.nc"}, "history_kyr": [[0, 2]]})");
  auto worst = 0.0;
  for (const auto* name :
       {"load", "displacement", "tangential_potential", "potential"}) {
    const auto& want = cap.variables.at(name).values;
    EXPECT_LT(Departure(own.variables.at(name).values, want), 1e-12) << name;
    EXPECT_LT(Departure(shifted.variables.at(name).values, want), 1e-12)
        << name;
    const auto departure = Departure(resampled.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-6) << name;
    auto twice = want;
    for (auto& v : twice) v *= 2.0;
    EXPECT_LT(Departure(doubled.variables.at(name).values, twice), 1e-12)
        << name;
  }
  RecordProperty("resampled_departure", std::format("{:.3e}", worst));
}

TEST_F(AppsTest, ViscoelasticResponseConvergesUnderStepHalving) {
  // A load switched on at the start and held: the final displacement at
  // the cap's peak converges as the step halves, at second order under the
  // exponential trapezoid and first under backward Euler. On the ball (a
  // Maxwell time of about 0.2 kyr) over 1 kyr in 8, 16, 32 and 64 steps,
  // the successive differences fall by 3.96 and 3.99 (trapezoid) and by
  // 1.94 and 1.97 (backward Euler).
  static_cast<void>(Deck());
  const auto final = [&](const std::string& scheme, int nodes) {
    const auto name = std::format("{}_{}", scheme, nodes);
    static_cast<void>(
        Run(name, std::format(R"({}, "times": {{"end_kyr": 1, "nodes": {}}},
            "forcing": {{"load": [{}]}},
            "solver": {{"scheme": "{}"}})",
                              kBall, nodes, kCap, scheme)));
    const auto file = Output(name);
    const auto last = Slice(file, "displacement", file.time->size() - 1);
    // the peak, where the cap's centre is
    return *std::ranges::max_element(
        last, [](double a, double b) { return std::abs(a) < std::abs(b); });
  };
  for (const auto& [scheme, low, high] :
       {std::tuple{std::string{"exponential_trapezoid"}, 3.0, 5.5},
        std::tuple{std::string{"backward_euler"}, 1.7, 2.3}}) {
    auto values = std::vector<double>{};
    for (const auto nodes : {9, 17, 33, 65})
      values.push_back(final(scheme, nodes));
    auto ratios = std::string{};
    for (auto k = std::size_t{2}; k < values.size(); ++k) {
      const auto ratio =
          (values[k - 2] - values[k - 1]) / (values[k - 1] - values[k]);
      ratios += std::format("{}{:.3f}", ratios.empty() ? "" : " ", ratio);
      EXPECT_GT(ratio, low) << scheme << " " << k;
      EXPECT_LT(ratio, high) << scheme << " " << k;
    }
    RecordProperty(scheme + "_ratios", ratios);
  }
}

TEST_F(AppsTest, ViscoelasticResponseStepSolversAgree) {
  // A 3D run under a lateral viscosity and lateral moduli closes each step
  // by the fixed point or by conjugate gradients, to the same tolerance
  // (1e-10 a step): the two agree to it over the run (about 1.4e-11).
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  ASSERT_TRUE(io::WriteModuli(dir_ / "moduli.nc", model, SurfaceGrid(8),
                              [](double, double theta, double phi) {
                                return 0.05 * std::sin(theta) * std::cos(phi);
                              },
                              {},
                              {.name = "toy moduli", .elementMetres = 500e3}));
  const auto body = std::format(
      R"({}, "times": {{"end_kyr": 1, "nodes": 3}},
         "forcing": {{"load": [{}], "tide": [{}]}})",
      Prem8(R"(, "moduli": {"file": "moduli.nc"})"), kCap, kTide);
  const auto fixed = Run("fixed", body);
  const auto conjugate =
      Run("conjugate", body + R"(, "solver": {"step": "conjugate_gradients"})");
  EXPECT_TRUE(fixed.starts_with(
      "3-D (lateral viscosity, lateral moduli), viscoelastic, 3 nodes 0 to 1 "
      "kyr, load (1 term) + tide (1 coefficient); wrote "))
      << fixed;
  EXPECT_TRUE(fixed.ends_with("fixed-point iterations over 2 steps)")) << fixed;
  EXPECT_TRUE(
      conjugate.ends_with("conjugate-gradient iterations over 2 steps)"))
      << conjugate;
  const auto a = Output("fixed");
  const auto b = Output("conjugate");
  EXPECT_EQ(std::get<std::string>(a.meta.at("step_solver")), "fixed_point");
  EXPECT_EQ(std::get<std::string>(b.meta.at("step_solver")),
            "conjugate_gradients");
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto departure =
        Departure(b.variables.at(name).values, a.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-9) << name;
  }
  RecordProperty("solver_departure", std::format("{:.3e}", worst));
}

TEST_F(AppsTest, ViscoelasticResponseTakesItsNodesEitherWay) {
  // Nodes listed with at_kyr are the evenly spaced ones when they
  // coincide, to the bit; uneven ones run and are recorded as given. The
  // exchange tables are named by the node's index padded to the width of
  // the last, and exchange_nodes chooses them.
  static_cast<void>(Deck());
  const auto forcing =
      std::format(R"("forcing": {{"load": [{}], "tide": [{}]}})", kCap, kTide);
  static_cast<void>(Run(
      "even", std::format(R"({}, "times": {{"end_kyr": 2, "nodes": 5}}, {})",
                          kBall, forcing)));
  static_cast<void>(
      Run("listed",
          std::format(R"({}, "times": {{"at_kyr": [0, 0.5, 1, 1.5, 2]}}, {})",
                      kBall, forcing)));
  const auto even = Output("even");
  const auto listed = Output("listed");
  EXPECT_EQ(*even.time, *listed.time);
  for (const auto& [name, field] : even.variables) {
    EXPECT_EQ(field.values, listed.variables.at(name).values) << name;
  }

  // uneven nodes, starting late, with the tables of three of them
  const auto config = Write(
      "uneven.json",
      std::format(R"({{{}, "times": {{"at_kyr": [1, 1.1, 1.4, 2, 3.5, 4, 5,
                     6, 7, 8, 9.5]}}, {},
          "output": {{"surface": "out/uneven.nc",
                      "exchange": "out/uneven_{{node}}_{{node}}.txt",
                      "exchange_nodes": [10, 0, 3, 3]}}}})",
                  kBall, forcing));
  const auto summary = app::RunResponse(config);
  EXPECT_TRUE(summary.starts_with("1-D, viscoelastic, 11 nodes 1 to 9.5 kyr"))
      << summary;
  EXPECT_TRUE(summary.ends_with("and 3 exchange tables")) << summary;
  const auto uneven = Output("uneven");
  ASSERT_EQ(uneven.time->size(), 11u);
  EXPECT_DOUBLE_EQ(uneven.time->front(), app::KYR);
  EXPECT_DOUBLE_EQ(uneven.time->back(), 9.5 * app::KYR);
  for (const auto* index : {"00", "03", "10"}) {
    EXPECT_TRUE(fs::exists(dir_ / "out" /
                           std::format("uneven_{}_{}.txt", index, index)))
        << index;
  }
  EXPECT_FALSE(fs::exists(dir_ / "out" / "uneven_01_01.txt"));
  const auto table = ReadExchange(dir_ / "out" / "uneven_03_03.txt");
  EXPECT_EQ(table.scalars.at("node"), "3");
  EXPECT_EQ(table.scalars.at("time_kyr"), "2");
  // the first node is the elastic response at whatever time it is
  EXPECT_LT(Departure(Slice(uneven, "displacement", 0),
                      Slice(even, "displacement", 0)),
            1e-14);
}

TEST_F(AppsTest, ViscoelasticResponseRefusesWhatItCannotDo) {
  // Each fault of a viscoelastic config is refused naming the member: the
  // times block's forms, given both ways or neither, out of order or too
  // short; a history out of order or malformed; an elastic deck with no
  // viscosity file; the output's {node}, missing with times and present
  // without, a volume file, exchange nodes out of range or without their
  // tables; the time step's members and a time origin without times; and
  // the members of other commands. Once the run reads its deck, a Maxwell
  // deck with no viscosity anywhere.
  static_cast<void>(Deck());
  static_cast<void>(Write("inviscid.deck",
                          "a ball with no viscosity\n"
                          "gia deck for the sl3d tests\n"
                          "0 5500 10000 5500 10000 5500 1 nan\n"
                          "6371000 5500 10000 5500 10000 5500 1 nan\n"));
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    const auto path = Write("bad.json", text);
    EXPECT_THROW(
        {
          try {
            static_cast<void>(run(path));
          } catch (const planetmodel::Error& error) {
            const auto message = std::string{error.what()};
            EXPECT_NE(message.find(member), std::string::npos) << message;
            EXPECT_NE(message.find(what), std::string::npos) << message;
            throw;
          }
        },
        planetmodel::Error)
        << text;
  };
  const auto read = [](const fs::path& path) {
    return app::ReadResponseInput(path);
  };
  const auto run = [](const fs::path& path) { return app::RunResponse(path); };
  const auto config =
      [&](const std::string& times, const std::string& extra = "",
          const std::string& output =
              R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
          const std::string& model = R"({"deck": "ball.deck"})",
          const std::string& load = kCap) {
        return std::format(R"({{"model": {}, "lmax": 4, {} "forcing":
                           {{"load": [{}]}},{} "output": {}}})",
                           model, times, load, extra, output);
      };
  const auto even = std::string{R"("times": {"end_kyr": 2, "nodes": 3},)"};

  refused(read,
          config(R"("times": {"end_kyr": 2, "nodes": 3, "at_kyr": [0, 1]},)"),
          "times", "gives the nodes twice");
  refused(read, config(R"("times": {"start_kyr": 1},)"), "times",
          "gives no nodes");
  refused(read, config(R"("times": {"end_kyr": 2},)"), "times",
          "gives `end_kyr` without `nodes`");
  refused(read, config(R"("times": {"end_kyr": 2, "nodes": 1},)"),
          "times.nodes", "fewer than two nodes");
  refused(read,
          config(R"("times": {"start_kyr": 2, "end_kyr": 2, "nodes": 3},)"),
          "times.end_kyr", "is not later than start_kyr");
  refused(read, config(R"("times": {"at_kyr": [0, 1, 1]},)"), "times.at_kyr[2]",
          "is not later than the node before it");
  refused(read, config(R"("times": {"at_kyr": [0]},)"), "times.at_kyr",
          "fewer than two nodes");
  refused(read, config(R"("times": {"start_kyr": 0, "at_kyr": [0, 1]},)"),
          "times.start_kyr", "is given by `at_kyr`");
  refused(read, config(R"("times": {"end_kyr": 2, "nodes": 3, "step": 1},)"),
          "times.step", "not a recognised member");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history_kyr": [[0, 1], [0, 2]]})"),
          "forcing.load[0].history_kyr[1][0]", "is not later than the time");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history_kyr": [[0, 1, 2]]})"),
          "forcing.load[0].history_kyr[0]", "is not a pair");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history_kyr": []})"),
          "forcing.load[0].history_kyr", "holds no [time_kyr, factor] pair");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck", "format": "elastic"})"),
          "model.format", "no viscosity column");
  refused(read, config(even, "", R"({"surface": "o.nc", "exchange": "o.txt"})"),
          "output.exchange", "lacks `{node}`");
  refused(read,
          config("", "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})"),
          "output.exchange", "holds `{node}`");
  refused(read, config(even, "", R"({"surface": "o.nc", "volume": "v.nc"})"),
          "output.volume", "not written for a viscoelastic run yet");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt",
                               "exchange_nodes": [0, 3]})"),
          "output.exchange_nodes[1]", "is not a node index in [0, 2]");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange_nodes": [0]})"),
          "output.exchange_nodes", "need `output.exchange`");
  refused(read, config("", "", R"({"surface": "o.nc", "exchange_nodes": [0]})"),
          "output.exchange_nodes", "need a `times` block");
  refused(read,
          config("", R"( "solver": {"scheme": "backward_euler"},)",
                 R"({"surface": "o.nc"})"),
          "solver.scheme", "needs a `times` block");
  refused(read,
          config("", R"( "solver": {"step": "fixed_point"},)",
                 R"({"surface": "o.nc"})"),
          "solver.step", "needs a `times` block");
  refused(read, config(even, R"( "solver": {"step": "picard"},)"),
          "solver.step", "none of");
  refused(read,
          config("", "", R"({"surface": "o.nc"})", R"({"deck": "ball.deck"})",
                 R"({"field": {"file": "f.nc", "time_origin_kyr": 2}})"),
          "forcing.load[0].field.time_origin_kyr", "needs a `times` block");
  for (const auto* member : {"rotation", "checkpoint", "kernels"}) {
    refused(read, config(even, std::format(R"( "{}": {{}},)", member)), member,
            "not a recognised member");
  }
  refused(run,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "inviscid.deck"})"),
          "model.deck", "gives no viscosity");
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

TEST_F(AppsTest, EvolveCalibratesTheInitialSeaLevel) {
  // With calibrate_initial_sea_level the run reaches the state file's sea
  // level at the window's end: exactly under frozen shorelines (one linear
  // step), to the calibration's tolerance under migrating ones; a resumed
  // calibrating run reads its calibrated state back instead of calibrating
  // again, and ends where the uninterrupted run does.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{R"(
    "model": "ball.deck",
    "lmax": 8,
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
    "rotation": "full", "moments": "hydrostatic",
    "calibrate_initial_sea_level": {"rtol": 1e-8})"};

  // the target: the state file's present-day sea level on the solver grid
  const auto grid = SurfaceGrid(8);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto target =
      SeaLevelFromTopography(history->Values("topography", 0.0),
                             history->Values("ice_thickness", 0.0));
  auto worstOf = [&](const pm::SurfaceFile& file) {
    const auto& s0 = file.variables.at("initial_sea_level").values;
    const auto& slc = file.variables.at("sea_level_change").values;
    const auto last = slc.size() - target.size();
    auto worst = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < target.size(); ++i) {
      worst = std::max(worst, std::abs(s0[i] + slc[last + i] - target[i]));
      scale = std::max(scale, std::abs(target[i]));
    }
    return worst / scale;
  };

  const auto frozen =
      Write("frozen.json", "{" + body + R"(, "output": "out/frozen.nc"})");
  auto summary = app::RunEvolve(frozen);
  EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "frozen.nc");
  EXPECT_LT(worstOf(reference), 1e-9);
  EXPECT_FALSE(reference.variables.contains("ocean_function"));

  const auto migrating =
      Write("migrating.json", "{" + body + R"(, "shorelines": "migrating",
                        "output": "out/migrating.nc"})");
  static_cast<void>(app::RunEvolve(migrating));
  const auto moving = pm::ReadSurfaceFile(dir_ / "out" / "migrating.nc");
  EXPECT_LT(worstOf(moving), 1e-6);
  EXPECT_TRUE(moving.variables.contains("ocean_function"));

  // through wall-time stops: the calibration runs once, at the start
  const auto pieces = Write("pieces.json", "{" + body +
                                               R"(, "shorelines": "migrating",
              "checkpoint": {"directory": "ckpt", "every": 2,
                             "max_walltime_min": 1e-9},
              "output": "out/pieces.nc"})");
  for (auto attempts = 0; attempts < 8; ++attempts) {
    summary = app::RunEvolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
  }
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  EXPECT_TRUE(fs::exists(dir_ / "ckpt" / "initial_state.bin"));
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  EXPECT_NE(std::get<std::string>(rebuilt.meta.at("initial_sea_level"))
                .find("read back"),
            std::string::npos);
  for (const auto name :
       {"sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              moving.variables.at(name).values)
        << name;
  }

  // starting over calibrates afresh, keeping its own state
  const auto over =
      Write("over.json",
            "{" + body + R"(, "shorelines": "migrating", "restart": false,
              "checkpoint": {"directory": "ckpt2", "every": 2},
              "output": "out/over.nc"})");
  static_cast<void>(app::RunEvolve(over));
  ASSERT_TRUE(fs::exists(dir_ / "ckpt2" / "initial_state.bin"));
  static_cast<void>(app::RunEvolve(over));
  EXPECT_TRUE(fs::exists(dir_ / "ckpt2" / "initial_state.bin"));

  // the kernels about the calibrated migrating run
  const auto kernels =
      Write("kernels.json",
            "{" + body + R"(, "shorelines": "migrating", "adjoint": "discrete",
              "measurements": [{"type": "sea_level_point", "name": "site",
                                "latitude": -30.0, "longitude": 20.0,
                                "date_kyr_bp": 0.0}],
              "output": "out/kernels"})");
  summary = app::RunKernel1D(kernels);
  EXPECT_TRUE(fs::exists(dir_ / "out" / "kernels" / "site_ice_kernels.nc"))
      << summary;

  // a state file that ends before the window's end gives no target
  auto beyond = body;
  beyond.replace(beyond.find(R"("end_kyr_bp": 0.0)"), 17,
                 R"("end_kyr_bp": -0.5)");
  const auto future =
      Write("future.json", "{" + beyond + R"(, "output": "out/future.nc"})");
  try {
    static_cast<void>(app::RunEvolve(future));
    ADD_FAILURE() << "a target outside the state file was accepted";
  } catch (const planetmodel::Error& error) {
    EXPECT_NE(std::string{error.what()}.find("leaves out the window's end"),
              std::string::npos)
        << error.what();
  }

  // a migrating solve that cannot meet its tolerance is refused, not
  // finished quietly with its last iterate
  const auto strict =
      Write("strict.json", "{" + body + R"(, "shorelines": "migrating",
                                    "rtol": 1e-17,
                                    "output": "out/strict.nc"})");
  try {
    static_cast<void>(app::RunEvolve(strict));
    ADD_FAILURE() << "an unconverged migrating run was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("did not converge"),
              std::string::npos)
        << error.what();
  }
}

TEST_F(AppsTest, The3DCommandsCalibrateUnderMigratingShorelines) {
  // The 3D commands take migrating shorelines as the 1D ones do: the
  // calibrated run reaches the state file's present-day sea level and
  // records its ocean functions, through wall-time stops as uninterrupted,
  // and the kernels come about its own shorelines. The history's polar
  // ice sits in a basin below sea level and floats off as it thins.
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto body = std::string{R"(
    "model": ")"} + data +
                    R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" +
                    data + R"(/viscosity_log.nc"},
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
    "rotation": "full",
    "shorelines": "migrating",
    "calibrate_initial_sea_level": {"rtol": 1e-8})";
  const auto grid = SurfaceGrid(8);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto target =
      SeaLevelFromTopography(history->Values("topography", 0.0),
                             history->Values("ice_thickness", 0.0));

  const auto whole =
      Write("whole.json", "{" + body + R"(, "output": "out/whole.nc"})");
  auto summary = app::RunEvolve3D(whole);
  EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  EXPECT_EQ(std::get<std::string>(reference.meta.at("shorelines")),
            "migrating");
  ASSERT_TRUE(reference.variables.contains("ocean_function"));
  const auto& s0 = reference.variables.at("initial_sea_level").values;
  const auto& slc = reference.variables.at("sea_level_change").values;
  const auto last = slc.size() - target.size();
  auto worst = 0.0, scale = 0.0;
  for (auto i = std::size_t{0}; i < target.size(); ++i) {
    worst = std::max(worst, std::abs(s0[i] + slc[last + i] - target[i]));
    scale = std::max(scale, std::abs(target[i]));
  }
  EXPECT_LT(worst, 1e-6 * scale);
  const auto& ocean = reference.variables.at("ocean_function").values;
  EXPECT_FALSE(
      std::equal(ocean.begin(), ocean.begin() + target.size(),
                 ocean.end() - static_cast<std::ptrdiff_t>(target.size())))
      << "no cell flipped";

  const auto pieces = Write("pieces.json", "{" + body + R"(,
              "checkpoint": {"directory": "ckpt", "every": 1,
                             "max_walltime_min": 1e-9},
              "output": "out/pieces.nc"})");
  for (auto attempts = 0; attempts < 8; ++attempts) {
    summary = app::RunEvolve3D(pieces);
    if (summary.find("wrote") != std::string::npos) break;
  }
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  for (const auto name :
       {"sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }

  const auto kernels = Write("kernels.json", "{" + body + R"(,
              "measurements": [{"type": "sea_level_point", "name": "site",
                                "latitude": -30.0, "longitude": 20.0,
                                "date_kyr_bp": 0.0}],
              "output": "out/kernels"})");
  summary = app::RunKernel3D(kernels);
  const auto ice =
      pm::ReadSurfaceFile(dir_ / "out" / "kernels" / "site_ice_kernels.nc");
  // about moving shorelines the initial sea level kernel is not zero
  const auto& dual = ice.variables.at("initial_sea_level_dual").values;
  EXPECT_TRUE(std::ranges::any_of(dual, [](double v) { return v != 0.0; }))
      << summary;
}

TEST_F(AppsTest, Evolve3DRunsAndARestartedRunMatchesAnUninterruptedOne) {
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto body = std::string{R"(
    "model": ")"} + data +
                    R"(/prem8.elastic.deck",
    "model_format": "elastic",
    "lmax": 8,
    "viscosity": {"file": ")" +
                    data + R"(/viscosity_log.nc"},
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
            pm::netcdf::Attribute{data + "/viscosity_log.nc"});

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
  // the decay row's fit: tau in kyr (the value's), then the amplitude and
  // offset in metres -- of the order of the toy history's sea level, not
  // a length scale too large
  {
    auto in = std::ifstream(out / "measurements.txt");
    auto line = std::string{};
    while (std::getline(in, line) && !line.starts_with("rebound ")) {
    }
    auto words = std::istringstream(line);
    auto label = std::string{}, type = std::string{};
    auto value = 0.0, tau_kyr = 0.0, amplitude = 0.0, offset = 0.0;
    words >> label >> type >> value >> tau_kyr >> amplitude >> offset;
    EXPECT_NEAR(tau_kyr * 1.0e3 * 365.25 * 86400.0, value, 1e-9 * value);
    EXPECT_GT(std::abs(amplitude), 0.0);
    EXPECT_LT(std::abs(amplitude), 1.0e3);
    EXPECT_LT(std::abs(offset), 1.0e3);
  }

  // the finite-difference check: the site value's change under an ice
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
    "viscosity": {"file": ")" + data + R"(/viscosity_log.nc"},
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
    "viscosity": {"file": ")" + data + R"(/viscosity_log.nc"},
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
    "viscosity": {"file": ")" + data + R"(/viscosity_log.nc"},
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
  // the spilled states go once the sweeps are done; other files stay
  fs::create_directories(dir_ / "spill");
  static_cast<void>(Write("spill/notes.txt", "not a state"));
  const auto spilled_summary = app::RunKernel3D(spilled_config);
  EXPECT_NE(spilled_summary.find("spilled state(s) removed"), std::string::npos)
      << spilled_summary;
  EXPECT_FALSE(fs::exists(dir_ / "spill" / "node_000000.traj"));
  EXPECT_FALSE(fs::exists(dir_ / "spill" / "node_000002.traj"));
  EXPECT_TRUE(fs::exists(dir_ / "spill" / "notes.txt"));
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
  // a name is its files' stem in the output directory
  refuse(R"({"type": "rotation", "component": "z", "name": "../escape"})",
         "is not a plain file name");
  refuse(R"({"type": "rotation", "component": "z", "name": ".hidden"})",
         "is not a plain file name");
  refuse(R"({"type": "rotation", "component": "z", "name": "lod"},
            {"type": "rotation", "component": "x", "name": "lod"})",
         "names another measurement too");
  const auto input = app::ReadKernelInput(config(
      R"({"type": "stokes_coefficient", "degree": 2, "order": -2,
          "date_kyr_bp": 0.5},
         {"type": "decay_time", "latitude": 80, "longitude": 0,
          "width_deg": 15, "window_start_kyr_bp": 1.5,
          "time_uncertainty_kyr": 0.1, "value_uncertainty_m": 1.0})"));
  EXPECT_EQ(input.measurements.size(), 2u);
}

TEST_F(AppsTest, The1DCommandsRefuseTheCoupledFixedPoint) {
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{R"(
    "model": "ball.deck", "lmax": 4,
    "state": {"file": "history.nc"},
    "ice_history": {"file": "history.nc"},
    "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 3},
    "coupled": {"rtol": 1e-8})"};
  const auto refused = [](auto&& run, const fs::path& config) {
    try {
      static_cast<void>(run(config));
      ADD_FAILURE() << config << " was accepted";
    } catch (const planetmodel::Error& error) {
      EXPECT_NE(std::string{error.what()}.find("coupled"), std::string::npos)
          << error.what();
    }
  };
  refused(app::RunEvolve,
          Write("evolve1d.json", "{" + body + R"(, "output": "out.nc"})"));
  refused(app::RunKernel1D,
          Write("kernel1d.json",
                "{" + body + R"(, "measurements": [{"type": "rotation",
                                  "component": "z"}], "output": "out"})"));
}

TEST_F(AppsTest, TransformDegreesAreReadAndChecked) {
  const auto config = [&](std::string_view degrees) {
    return Write("transform.json", std::string{R"({
      "model": "ball.deck", "lmax": 4,
      "state": {"file": "history.nc"},
      "ice_history": {"file": "history.nc"},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 3},
      "coupled": {"transform_degrees": )"} +
                                       std::string{degrees} +
                                       R"(}, "output": "out.nc"})");
  };
  EXPECT_EQ(app::ReadEvolveInput(config("[2, 3]")).transform_degrees,
            (std::vector<int>{2, 3}));
  for (const auto* bad : {"[5]", "[1]", "[2.5]"}) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::ReadEvolveInput(config(bad)));
          } catch (const planetmodel::Error& error) {
            EXPECT_NE(std::string{error.what()}.find("[2, lmax = 4]"),
                      std::string::npos)
                << error.what();
            throw;
          }
        },
        planetmodel::Error)
        << bad;
  }
}

TEST_F(AppsTest, CacheMbIsReadAndChecked) {
  // evolve's and kernel's coupled.cache_mb, and response's solver.cache_mb:
  // a count of MiB that reaches the operator's options as bytes, 0 or
  // absent leaving the library's default
  static_cast<void>(Deck());
  const auto evolve = [&](std::string_view coupled) {
    return Write("cache.json", std::string{R"({
      "model": "ball.deck", "lmax": 4,
      "state": {"file": "history.nc"},
      "ice_history": {"file": "history.nc"},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 3},
      "coupled": {)"} + std::string{coupled} +
                                   R"(}, "output": "out.nc"})");
  };
  const auto responseConfig = [&](std::string_view member) {
    return Write("response.json", std::string{R"({
      "model": {"deck": "ball.deck"}, "lmax": 4,
      "forcing": {"tide": [{"degree": 2, "amplitude_m2_s2": 1.0}]},
      "solver": {)"} + std::string{member} +
                                      R"(}, "output": {"surface": "out.nc"}})");
  };
  const auto absent = app::ReadEvolveInput(evolve(""));
  const auto zero = app::ReadEvolveInput(evolve(R"("cache_mb": 0)"));
  const auto set = app::ReadEvolveInput(evolve(R"("cache_mb": 32)"));
  EXPECT_EQ(absent.cache_mb, 0u);
  EXPECT_EQ(zero.cache_mb, 0u);
  EXPECT_EQ(set.cache_mb, 32u);
  const auto earth =
      app::BuildEarth(absent, planetmodel::ReadJsonFile(evolve("")));
  const auto rheology = Rheology::Create(earth.material, {});
  ASSERT_TRUE(rheology);
  const auto library = Viscoelastic3D::Options{}.cache_bytes;
  EXPECT_EQ(app::OperatorOptions(absent, *rheology).cache_bytes, library);
  EXPECT_EQ(app::OperatorOptions(zero, *rheology).cache_bytes, library);
  EXPECT_EQ(app::OperatorOptions(set, *rheology).cache_bytes,
            std::size_t{32} << 20);
  EXPECT_EQ(app::ReadResponseInput(responseConfig("")).solver.cache_mb, 0u);
  EXPECT_EQ(
      app::ReadResponseInput(responseConfig(R"("cache_mb": 16)")).solver.cache_mb,
      16u);

  // refused naming the member: negative, fractional, or not a number
  const auto refused = [](auto&& read, const fs::path& config,
                          std::string_view what) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(read(config));
          } catch (const planetmodel::Error& error) {
            const auto message = std::string{error.what()};
            EXPECT_NE(message.find("cache_mb"), std::string::npos) << message;
            EXPECT_NE(message.find(what), std::string::npos) << message;
            throw;
          }
        },
        planetmodel::Error)
        << config;
  };
  for (const auto& [bad, what] :
       {std::pair{"-1", "is not a cache size in MiB"},
        std::pair{"2.5", "is not an integer"},
        std::pair{R"("32")", "expected a number"}}) {
    const auto member = std::string{R"("cache_mb": )"} + bad;
    refused(app::ReadEvolveInput, evolve(member), what);
    refused(app::ReadResponseInput, responseConfig(member), what);
  }
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
