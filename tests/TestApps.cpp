// The sl3d configs and commands: a misspelt or missing member is refused
// naming the config and the place, relative paths resolve against the
// config's directory, love-numbers writes its provenance-headed table,
// solve takes its loads and tide as sums of terms of every kind and runs
// a body without an ocean, elastic in the static problem or viscoelastic
// through time in the quasi-static one, and a toy ice and sea level
// through the sea level equation, as the elastic fingerprint or through
// time, 1-D or 3-D as its model block makes it, kernel reads the same
// config for its forward run and its kernels pair as finite differences of
// solve's runs, and every example config is read.

#include <limits>
#include <gtest/gtest.h>

#include <GaussQuad/All>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/json.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "AppFixtures.hpp"
#include "Commands.hpp"
#include "Common.hpp"
#include "Outputs.hpp"
#include "Ranks.hpp"
#include "RunHash.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"
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
    // a run that died leaves its directory: its stores and checkpoints
    // are not this run's
    fs::remove_all(dir_);
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

  /// Runs `sl3d solve` in the static problem on the radial ball at `lmax`
  /// under the forcing block given, on a body without an ocean and without
  /// rotational feedback: the solid body's elastic response. Returns its
  /// surface file.
  planetmodel::SurfaceFile Respond(const std::string& name,
                                   const std::string& forcing,
                                   int lmax = 8) const {
    static_cast<void>(Deck());
    const auto config = Write(name + ".json", std::format(R"({{
      "problem": "static", "model": {{"deck": "ball.deck"}},
      "discretisation": {{"lmax": {}}},
      "forcing": {}, "rotation": "none",
      "output": {{"surface": "out/{}.nc"}}}})",
                                                          lmax, forcing, name));
    const auto summary = app::RunSolve(config);
    EXPECT_TRUE(
        summary.starts_with("1-D, elastic, static, no ocean, rotation none; "))
        << summary;
    return planetmodel::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  }

  /// Runs `sl3d solve` in the quasi-static problem on a body without an
  /// ocean and without rotational feedback, on a config of the members
  /// given (its times on the run's own clock), its output block written
  /// after them (the surface file out/<name>.nc, a table per node
  /// out/<name>_{node}.txt), and returns the summary.
  std::string Run(const std::string& name, const std::string& members) const {
    const auto config =
        Write(name + ".json", std::format(R"({{"problem": "quasi_static", {},
                                          "rotation": "none",
                                          "output": {{"surface": "out/{}.nc",
                                          "exchange": "out/{}_{{node}}.txt"}}}})",
                                          members, name, name));
    return app::RunSolve(config);
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

/// A field of a surface file at a node (a static field: `node` 0).
std::vector<double> FieldAt(const planetmodel::SurfaceFile& file,
                            const std::string& name, std::size_t node) {
  if (!file.time) return file.variables.at(name).values;
  return Slice(file, name, node);
}

/// The surface pairing int K delta dS of two fields on a surface file's
/// grid over the sphere of the file's `radius_m` (a kernel file's).
double SurfacePairing(const planetmodel::SurfaceFile& kernels,
                      std::span<const double> kernel,
                      std::span<const double> delta) {
  const auto& grid = *kernels.grid;
  const auto weights = *grid.Weights();
  const auto radius =
      std::get<std::vector<double>>(kernels.meta.at("radius_m")).front();
  const auto nphi = grid.NPhi();
  auto out = 0.0;
  for (auto it = std::size_t{0}; it < grid.NTheta(); ++it) {
    auto ring = 0.0;
    for (auto j = std::size_t{0}; j < nphi; ++j) {
      ring += kernel[it * nphi + j] * delta[it * nphi + j];
    }
    out += weights[it] * ring;
  }
  return out * 2.0 * std::numbers::pi / static_cast<double>(nphi) * radius *
         radius;
}

/// The surface pairing of a kernel file's node field with fields given at
/// every node (`deltas`, one per node), summed over the nodes.
double NodePairing(const planetmodel::SurfaceFile& kernels,
                   const std::string& name,
                   const std::vector<std::vector<double>>& deltas) {
  auto out = 0.0;
  for (auto n = std::size_t{0}; n < deltas.size(); ++n) {
    out += SurfacePairing(kernels, FieldAt(kernels, name, n), deltas[n]);
  }
  return out;
}

/// The value of a measurement in a kernel run's measurements.txt.
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
  throw std::runtime_error(name + " is not in measurements.txt");
}

/// The rows of numbers of a kernel table, its `#` lines left out.
std::vector<std::vector<double>> TableRows(const fs::path& path) {
  auto in = std::ifstream(path);
  EXPECT_TRUE(in) << path;
  auto out = std::vector<std::vector<double>>{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.starts_with("#")) continue;
    auto words = std::istringstream(line);
    auto row = std::vector<double>{};
    for (double v; words >> v;) row.push_back(v);
    out.push_back(std::move(row));
  }
  return out;
}

/// The cap mean of a field of a solve surface file at a node, as the
/// kernels' measurements read it (CapMeanDual on the solver grid of
/// `lmax`).
double CapMeanOf(const planetmodel::SurfaceFile& file, const std::string& name,
                 std::size_t node, GeographicPoint site, double width_deg,
                 int lmax = 8) {
  const SurfaceGrid grid(lmax);
  return PairedValue(
      grid, 1.0,
      CapMeanDual(grid, 1.0, site, width_deg * std::numbers::pi / 180.0),
      FieldAt(file, name, node));
}

/// The band-limited value of a field of a solve surface file at a point.
double PointValueOf(const planetmodel::SurfaceFile& file,
                    const std::string& name, std::size_t node,
                    GeographicPoint site, int lmax = 8) {
  const SurfaceGrid grid(lmax);
  return PairedValue(grid, 1.0, PointDual(grid, 1.0, site),
                     FieldAt(file, name, node));
}

/// The displacement of a solve surface file at a point along a direction
/// (the radial, tangential and toroidal potentials paired with
/// PointDisplacementMeasurement's duals).
double VectorPointOf(const planetmodel::SurfaceFile& file, std::size_t node,
                     GeographicPoint site, SurfaceDirection direction,
                     int lmax = 8) {
  const SurfaceGrid grid(lmax);
  const auto m = PointDisplacementMeasurement(grid, 1.0, site, direction);
  auto out = 0.0;
  for (const auto& [dual, name] :
       {std::pair{&m.displacementDual, "displacement"},
        std::pair{&m.tangentialPotentialDual, "tangential_potential"},
        std::pair{&m.toroidalPotentialDual, "toroidal_potential"}}) {
    if (dual->empty() || !file.variables.contains(name)) continue;
    out += PairedValue(grid, 1.0, *dual, FieldAt(file, name, node));
  }
  return out;
}

/// The displacement of a solve surface file smoothed over a Gaussian cap
/// as a vector, read at its centre along a direction there (the radial,
/// tangential and toroidal potentials paired with
/// CapDisplacementMeasurement's duals).
double VectorCapOf(const planetmodel::SurfaceFile& file, std::size_t node,
                   GeographicPoint site, double width_deg,
                   SurfaceDirection direction, int lmax = 8) {
  const SurfaceGrid grid(lmax);
  const auto m = CapDisplacementMeasurement(
      grid, 1.0, site, width_deg * std::numbers::pi / 180.0, direction);
  auto out = 0.0;
  for (const auto& [dual, name] :
       {std::pair{&m.displacementDual, "displacement"},
        std::pair{&m.tangentialPotentialDual, "tangential_potential"},
        std::pair{&m.toroidalPotentialDual, "toroidal_potential"}}) {
    if (dual->empty() || !file.variables.contains(name)) continue;
    out += PairedValue(grid, 1.0, *dual, FieldAt(file, name, node));
  }
  return out;
}

/// The homogeneous test ball's GIA deck with its moduli scaled by
/// 1 + `moduli` (the wave speeds by its square root) and its viscosity
/// `viscosity` Pa s.
std::string BallDeck(double moduli, double viscosity = 1e21) {
  const auto root = std::sqrt(1.0 + moduli);
  const auto vp = 10000.0 * root;
  const auto vs = 5500.0 * root;
  auto out = std::string{"homogeneous test ball\ngia deck for the sl3d tests\n"};
  for (const auto r : {0.0, 6371000.0}) {
    out += std::format("{} 5500 {:.17g} {:.17g} {:.17g} {:.17g} 1 {:.17g}\n",
                       r, vp, vs, vp, vs, viscosity);
  }
  return out;
}

/// The relative gap of a prediction from a measured value.
double Gap(double got, double want) {
  return std::abs(got - want) / std::abs(want);
}

/// An exchange table: its named scalars (the text after each name) and
/// its coefficients of U and Phi in the grid's layout.
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

/// An exchange table's provenance header, its `#` lines in order.
std::vector<std::string> HeaderOf(const fs::path& path) {
  auto in = std::ifstream(path);
  auto out = std::vector<std::string>{};
  for (std::string line; std::getline(in, line);) {
    if (line.starts_with("#")) out.push_back(line);
  }
  return out;
}

/// Expects two exchange tables of the same problem from two configs (its
/// times in two forms, say): the same named scalars but those listed in
/// `besides` (`mine` alone holds them, or holds them otherwise), the same
/// header lines but the config's (the third) and the forcing's terms
/// (described in each config's form); returns the largest relative
/// departure of the coefficients.
double CompareTables(const fs::path& mine, const fs::path& theirs,
                     std::initializer_list<std::string_view> besides = {}) {
  const auto a = ReadExchange(mine);
  const auto b = ReadExchange(theirs);
  for (const auto& [name, value] : b.scalars) {
    if (std::ranges::find(besides, name) != besides.end()) continue;
    EXPECT_TRUE(a.scalars.contains(name)) << name << " in " << mine;
    if (a.scalars.contains(name)) {
      EXPECT_EQ(a.scalars.at(name), value) << name << " in " << mine;
    }
  }
  for (const auto& [name, value] : a.scalars) {
    EXPECT_TRUE(b.scalars.contains(name) ||
                std::ranges::find(besides, name) != besides.end())
        << name << " in " << mine;
  }
  const auto ours = HeaderOf(mine);
  const auto other = HeaderOf(theirs);
  EXPECT_EQ(ours.size(), other.size()) << mine;
  for (auto k = std::size_t{0}; k < std::min(ours.size(), other.size()); ++k) {
    const auto may = k == 2 || ours[k].starts_with("# forcing ");
    EXPECT_TRUE(may || ours[k] == other[k]) << ours[k];
  }
  return std::max(Departure(a.U, b.U), Departure(a.Phi, b.Phi));
}

/// The largest relative departure of two volume files' fields, each field
/// over every part.
double VolumeDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto a = planetmodel::ReadModelFile(mine);
  const auto b = planetmodel::ReadModelFile(theirs);
  EXPECT_EQ(a.Parts().size(), b.Parts().size());
  auto got = std::map<std::string, std::vector<double>>{};
  auto want = std::map<std::string, std::vector<double>>{};
  for (auto p = std::size_t{0}; p < b.Parts().size(); ++p) {
    for (const auto& [name, var] : b.Parts()[p].variables) {
      if (!std::string_view{name}.ends_with("_relative") &&
          name != "radial_displacement" && name != "deviatoric_strain") {
        continue;
      }
      const auto& held = a.Parts()[p].variables;
      EXPECT_TRUE(held.contains(name)) << name << " in part " << p;
      if (!held.contains(name)) continue;
      want[name].insert(want[name].end(), var.values.begin(), var.values.end());
      got[name].insert(got[name].end(), held.at(name).values.begin(),
                       held.at(name).values.end());
    }
  }
  EXPECT_FALSE(want.empty());
  auto worst = 0.0;
  for (const auto& [name, values] : want) {
    worst = std::max(worst, Departure(got.at(name), values));
  }
  return worst;
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

/// Runs `run` on `config` and expects a planetmodel::Error whose message
/// names `member` and says `what`.
template <class Runner>
void ExpectRefused(Runner&& run, const fs::path& config,
                   const std::string& member, const std::string& what) {
  EXPECT_THROW(
      {
        try {
          static_cast<void>(run(config));
        } catch (const planetmodel::Error& error) {
          const auto message = std::string{error.what()};
          EXPECT_NE(message.find(member), std::string::npos) << message;
          EXPECT_NE(message.find(what), std::string::npos) << message;
          throw;
        }
      },
      planetmodel::Error)
      << config;
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
  refuse(R"({"model": "ball.deck", "discretisation": {"lmax": 4},
             "output": "t.txt", "lmxa": 3})",
         "not a recognised member");
  refuse(R"({"discretisation": {"lmax": 4}, "output": "t.txt"})",
         "has no member");
  refuse(R"({"model": "ball.deck", "discretisation": {"lmax": 4.5},
             "output": "t.txt"})",
         "discretisation.lmax: is not an integer");
  refuse(R"({"model": "ball.deck", "discretisation": {"lmax": -2},
             "output": "t.txt"})",
         "discretisation.lmax: is not a degree");
  refuse(R"({"model": "ball.deck", "model_format": "fortran",
             "discretisation": {"lmax": 4}, "output": "t.txt"})",
         "none of");
  // the degree is the discretisation block's, which holds it alone
  refuse(R"({"model": "ball.deck", "lmax": 4, "output": "t.txt"})",
         "lmax: is not a top-level member: it belongs in "
         "`discretisation.lmax`");
  refuse(R"({"model": "ball.deck", "output": "t.txt"})",
         "has no `discretisation` block");
  refuse(R"({"model": "ball.deck", "discretisation": 4, "output": "t.txt"})",
         "discretisation: is not a block");
  refuse(R"({"model": "ball.deck", "discretisation": {"lmax": 4, "eps": 1e-8},
             "output": "t.txt"})",
         "discretisation.eps: is not a recognised member");
  refuse(R"({"model": "ball.deck", "discretisation": {}, "output": "t.txt"})",
         "has no member");
}

TEST_F(AppsTest, ResolvesPathsAgainstTheConfig) {
  static_cast<void>(Deck());
  const auto config = Write("run.json", R"({"model": "ball.deck",
                                            "discretisation": {"lmax": 4},
                                            "output": "out/love.txt"})");
  const auto input = app::ReadLoveNumbersInput(config);
  EXPECT_EQ(input.model, dir_ / "ball.deck");
  EXPECT_EQ(input.output, dir_ / "out" / "love.txt");
  EXPECT_EQ(input.lmax, 4);
  EXPECT_EQ(input.format, app::ModelFormat::Gia);
}

TEST_F(AppsTest, TheExampleConfigsAreRead) {
  // Every config in examples/configs is read by its command's reader
  // without a fault: a `model` that is a path is love-numbers', a config
  // with a `kernels` block kernel's (and solve's, which checks the block),
  // and the rest solve's, with an ocean or without one.
  const auto configs =
      fs::path{SL3DPP_TEST_DATA_DIR}.parent_path().parent_path() / "examples" /
      "configs";
  auto read = 0;
  for (const auto& entry : fs::directory_iterator(configs)) {
    const auto& path = entry.path();
    if (path.extension() != ".json") continue;
    const auto top = planetmodel::ReadJsonFile(path);
    try {
      if (top.At("model").IsString()) {
        static_cast<void>(app::ReadLoveNumbersInput(path));
      } else if (app::Has(top, "kernels")) {
        static_cast<void>(app::ReadKernelInput(path));
        if (app::Has(top, "output")) {
          static_cast<void>(app::ReadSolveInput(path));
        }
      } else {
        static_cast<void>(app::ReadSolveInput(path));
      }
    } catch (const planetmodel::Error& error) {
      ADD_FAILURE() << error.what();
    }
    ++read;
  }
  EXPECT_EQ(read, 71);
}

TEST_F(AppsTest, LoveNumbersWritesItsTable) {
  static_cast<void>(Deck());
  const auto config = Write("run.json", R"({"model": "ball.deck",
                                            "discretisation": {"lmax": 4},
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
  const auto both = Write("both.json", R"({"model": "ball.deck",
                                           "discretisation": {"lmax": 4},
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
                   Write("same.json", R"({"model": "ball.deck",
                          "discretisation": {"lmax": 4},
                          "output": "out/love.txt",
                          "conventional_output": "out/./love.txt"})"))),
               planetmodel::Error);
}

using sl3dpp::test::WriteToyHistory;

TEST_F(AppsTest, SolveRespondsWithoutAnOceanOnAToyModuliFile) {
  // The static problem without an ocean on the homogeneous ball with a 10%
  // lateral shear modulus from a model file written by io::WriteModuli: a
  // polar cap load and a degree-2 tide, the surface file, the exchange
  // table and the volume file all written; the 3D displacement departs
  // from the radial model's by a few per cent, as a 10% modulus should
  // move it.
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
    "problem": "static",
    "model": {"deck": "ball.deck", "moduli": {"file": "moduli.nc"}},
    "discretisation": {"lmax": 8},
    "forcing": {
      "load": [{"cap": {"latitude": 90, "longitude": 0, "width_deg": 20,
                        "amplitude_kg_m2": 1.0e6}}],
      "tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0}]},
    "rotation": "none",
    "output": {"surface": "out/response.nc",
               "exchange": "out/response.txt",
               "volume": "out/response_volume.nc"}})");
  const auto summary = app::RunSolve(config);
  EXPECT_NE(summary.find("wrote"), std::string::npos) << summary;
  EXPECT_NE(summary.find("fixed-point iterations"), std::string::npos)
      << summary;
  EXPECT_TRUE(
      summary.starts_with("3-D (lateral moduli), elastic, static, no ocean, "
                          "rotation none; load (1 term) + tide (1 "
                          "coefficient); wrote"))
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

TEST_F(AppsTest, SolveIsIndependentOfTheCache) {
  // cache_mb reaches every transform grid of the static problem on lateral
  // moduli (the sea level grid, the 3D operator's coupling and the volume
  // file's radial reference coupling), and the run is the same to
  // rounding under any setting. At lmax 24 the smallest
  // setting, 1 MiB, cuts the couplings' batched spin -2 transforms into
  // chunks of 53 fields against the default's 63, and the ball's mesh
  // carries more nodes than a chunk, so the chunks fall differently
  // through every batch: the surface file, the exchange table and the
  // volume file are nevertheless the default's, to rounding.
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
      "problem": "static",
      "model": {{"deck": "ball.deck", "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": {}}},
      "forcing": {{
        "load": [{{"cap": {{"latitude": 60, "longitude": 30, "width_deg": 10,
                           "amplitude_kg_m2": 1.0e6}}}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                  "imaginary_m2_s2": 0.5}}]}},
      "rotation": "none",{}
      "output": {{"surface": "{}/response.nc",
                 "exchange": "{}/response.txt",
                 "volume": "{}/response_volume.nc"}}}})",
                                  kLmax, cache, name, name, name);
    const auto summary = app::RunSolve(Write(name + ".json", text));
    EXPECT_NE(summary.find("fixed-point iterations"), std::string::npos)
        << summary;
  };
  run("standard", "");
  run("small", R"( "solver": {"cache_mb": 1},)");

  namespace pm = planetmodel;
  const auto standard = pm::ReadSurfaceFile(dir_ / "standard" / "response.nc");
  const auto small = pm::ReadSurfaceFile(dir_ / "small" / "response.nc");
  // to rounding throughout: the chunks of another size are transformed
  // under plans of their own, which the planner's timings may choose
  // differently from the default's
  const auto close = [](std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) return false;
    auto scale = 0.0, worst = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      scale = std::max(scale, std::abs(a[i]));
      worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst <= 1e-12 * scale;
  };
  for (const auto& [name, field] : standard.variables) {
    EXPECT_TRUE(close(field.values, small.variables.at(name).values)) << name;
  }
  for (const auto* name : {"step_iterations", "solve_iterations"}) {
    EXPECT_EQ(standard.meta.at(name), small.meta.at(name)) << name;
  }

  // the exchange tables' rows, the coefficients at 17 digits (the comment
  // lines name the config)
  const auto rows = [&](const std::string& name) {
    auto in = std::ifstream(dir_ / name / "response.txt");
    auto out = std::vector<double>{};
    auto count = std::size_t{0};
    for (std::string line; std::getline(in, line);) {
      if (line.starts_with("#")) continue;
      ++count;
      auto fields = std::istringstream(line);
      for (double value; fields >> value;) out.push_back(value);
    }
    return std::pair{count, out};
  };
  const auto [standardCount, standardRows] = rows("standard");
  const auto [smallCount, smallRows] = rows("small");
  EXPECT_GT(standardCount, static_cast<std::size_t>(kLmax));
  EXPECT_EQ(standardCount, smallCount);
  EXPECT_TRUE(close(standardRows, smallRows));

  const auto standardVolume =
      pm::ReadModelFile(dir_ / "standard" / "response_volume.nc");
  const auto smallVolume =
      pm::ReadModelFile(dir_ / "small" / "response_volume.nc");
  ASSERT_EQ(standardVolume.Parts().size(), smallVolume.Parts().size());
  for (auto p = std::size_t{0}; p < standardVolume.Parts().size(); ++p) {
    const auto& held = smallVolume.Parts()[p].variables;
    for (const auto& [name, var] : standardVolume.Parts()[p].variables) {
      EXPECT_TRUE(close(var.values, held.at(name).values))
          << name << " in part " << p;
    }
  }
}

TEST_F(AppsTest, FieldLoadTermMatchesTheCap) {
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
      {"file": "fine.nc", "variable": "sigma"}}]})");
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

TEST_F(AppsTest, CoefficientsLoadTermMatchesTheCap) {
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

TEST_F(AppsTest, LoadTermsAndTideCoefficientsAdd) {
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
  const auto config = Write("all.json",
                            "{\"problem\": \"static\", "
                            "\"model\": {\"deck\": \"ball.deck\"}, "
                            "\"discretisation\": {\"lmax\": 8}, "
                            "\"forcing\": {\"load\": [" +
                                capTerm + ", " + diskTerm + "], \"tide\": [" +
                                tide21 + ", " + tide30 +
                                "]}, \"rotation\": \"none\", \"output\": "
                                "{\"surface\": \"out/all.nc\", "
                                "\"exchange\": \"out/all.txt\"}}");
  const auto summary = app::RunSolve(config);
  EXPECT_TRUE(
      summary.starts_with("1-D, elastic, static, no ocean, rotation none; "
                          "load (2 terms) + tide (2 coefficients); "
                          "wrote "))
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
  EXPECT_LT(worstHarmonic, 1e-12);

  const auto centre = GeographicPoint{55.0, -100.0};
  const auto alpha = 8.0 * std::numbers::pi / 180.0;
  const auto amplitude = 2.5e3;
  const auto disk = app::DiskCoefficients(grid, centre, alpha, amplitude);
  ASSERT_EQ(disk.size(), size);
  // the grid's degree-0 normalisation: a unit field's coefficient
  const auto unit = grid.Expand(std::vector<double>(grid.FieldSize(), 1.0))[0];
  const auto mean = amplitude * (1.0 - std::cos(alpha)) / 2.0;
  EXPECT_NEAR(disk[0].real(), mean * unit.real(), 1e-12 * mean * unit.real());
  // m = 0: the phase is exactly 1, the imaginary part exactly zero
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
  EXPECT_LT(worstQuadrature, 1e-12);

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

TEST_F(AppsTest, SolveRefusesWhatItsTermsCannotSay) {
  // Each fault of a config without an ocean is refused naming the member:
  // an unknown member anywhere, a load term of two kinds or none, an
  // amplitude history or a viscosity in the static problem, the problem
  // missing, unknown or at odds with the times block, a times block
  // without its end, an empty forcing, a tide of degree 0, 1 or above
  // lmax, the old layout's members, a
  // field's old `name`, an unknown format, a field's date in the
  // quasi-static problem; and, once the run reads its files, a field file
  // with a time axis and no date, a date on a file without one, a
  // variable the file lacks (the file's own named), and a coefficient
  // table above lmax.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  static_cast<void>(Write("high.txt", "# degree 5 at lmax 4\n0 0 1 0\n5 2 1 1\n"));
  {
    namespace pm = planetmodel;
    auto file = pm::SurfaceFile{};
    file.grid = pm::Equiangular(9, 18);
    file.source = "sl3dpp tests";
    file.variables.emplace(
        "load", pm::SurfaceField(
                    "load", std::vector<double>(9 * 18, 1.0),
                    {.dimensions = pm::Dimensions{.mass = 1, .length = -2}}));
    pm::WriteSurfaceFile(dir_ / "static.nc", file);
  }
  const auto cap = std::string{
      R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                  "amplitude_kg_m2": 1.0}})"};
  const auto config = [&](const std::string& model, const std::string& forcing,
                          const std::string& extra = "",
                          const std::string& problem = "static") {
    return std::format(R"({{"problem": "{}", "model": {},
                             "discretisation": {{"lmax": 4}},
                             "forcing": {},{} "output": {{"surface":
                             "out.nc"}}}})",
                       problem, model, forcing, extra);
  };
  const auto ball = std::string{R"({"deck": "ball.deck"})"};
  const auto loads = [](const std::string& terms) {
    return "{\"load\": [" + terms + "]}";
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto run = [](const fs::path& path) { return app::RunSolve(path); };

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
          std::format(R"({{"problem": "static", "model": {},
                           "discretisation": {{"lmax": 4}},
                          "forcing": {}, "output": {{"surface": "out.nc",
                          "volum": "v.nc"}}}})",
                      ball, loads(cap)),
          "output.volum", "not a recognised member");
  refused(read, config(ball, loads(R"({"cap": {"latitude": 60, "longitude": 0,
      "width_deg": 10, "amplitude_kg_m2": 1.0}, "disk": {"latitude": 0,
      "longitude": 0, "radius_deg": 5, "amplitude_kg_m2": 1.0}})")),
          "forcing.load[0]", "names several terms");
  refused(read, config(ball, loads(cap + ", {}")), "forcing.load[1]",
          "names no term");
  refused(read, config(ball, loads(R"({"cap": {"latitude": 60, "longitude": 0,
                                 "width_deg": 10, "amplitude_kg_m2": 1.0},
                                 "history": [[0, 1]]})")),
          "forcing.load[0].history", "needs the quasi-static problem");
  refused(read,
          config(ball, loads(cap), R"( "times": {"unit": "kyr", "nodes": 3},)",
                 "quasi_static"),
          "times", "gives `nodes` without `end`");
  refused(read,
          config(R"({"deck": "ball.deck", "viscosity": {"file": "v.nc"}})",
                 loads(cap)),
          "model.viscosity", "needs the quasi-static problem");
  // the problem: required, one of two words, and held to the times block
  refused(read,
          R"({"model": {"deck": "ball.deck"}, "discretisation": {"lmax": 4},
              "forcing": {"load": [{"cap": {"latitude": 60, "longitude": 0,
                  "width_deg": 10, "amplitude_kg_m2": 1.0}}]},
              "output": {"surface": "out.nc"}})",
          "bad.json", "has no `problem`");
  refused(read, config(ball, loads(cap), "", "dynamic"), "problem",
          "none of \"static\", \"quasi_static\"");
  refused(read,
          config(ball, loads(cap),
                 R"( "times": {"unit": "kyr", "end": 1, "nodes": 3},)"),
          "times", "the static problem does not take");
  refused(read, config(ball, loads(cap), "", "quasi_static"), "problem",
          "needs a `times` block");
  // a field term's members
  refused(read, config(ball, loads(R"({"field": {"file": "static.nc",
                                           "name": "load"}})")),
          "forcing.load[0].field.name", "its `variable`");
  refused(read, config(ball, loads(R"({"field": {"file": "static.nc",
                                           "format": "netcdf"}})")),
          "forcing.load[0].field.format", "none of \"planetmodel_surface\"");
  refused(read,
          config(ball, loads(R"({"field": {"file": "history.nc",
                                           "date_kyr": 1}})"),
                 R"( "times": {"unit": "kyr", "end": 1, "nodes": 3},)",
                 "quasi_static"),
          "forcing.load[0].field.date_kyr", "dates the file");
  refused(read, config(ball, "{}"), "forcing", "holds no forcing");
  refused(read, config(ball, R"({"load": [], "tide": []})"), "forcing",
          "holds no forcing");
  refused(read,
          config(ball, R"({"tide": [{"degree": 2, "order": 0,
                                     "amplitude_m2_s2": 1.0,
                                     "imaginary_m2_s2": 1.0}]})"),
          "forcing.tide[0].imaginary_m2_s2", "not zero at order 0");
  // a tide is taken relative to the centre of mass: degrees 2 to lmax
  for (const auto degree : {0, 1, 5}) {
    refused(read,
            config(ball, std::format(R"({{"tide": [{{"degree": {},
                                         "amplitude_m2_s2": 1.0}}]}})",
                                     degree)),
            "forcing.tide[0].degree", "is not a tidal degree within [2, lmax]");
  }
  refused(read,
          config(ball, loads(R"({"disk": {"latitude": 0, "longitude": 0,
                                 "radius_deg": 0, "amplitude_kg_m2": 1.0}})")),
          "forcing.load[0].disk.radius_deg", "is not an angle");
  // the old layout's members
  refused(read,
          R"({"model": "ball.deck",
              "discretisation": {"lmax": 4}, "tide": {"degree": 2,
              "amplitude_m2_s2": 1.0}, "output": "out.nc"})",
          "tide", "not a recognised member");

  refused(run, config(ball, loads(R"({"field": {"file": "history.nc",
                                           "variable": "ice_thickness"}})")),
          "forcing.load[0].field", "names a file with a time axis");
  refused(run, config(ball, loads(R"({"field": {"file": "static.nc",
                                           "date_kyr_bp": 0}})")),
          "forcing.load[0].field.date_kyr_bp",
          "dates a file without a time axis");
  refused(run, config(ball, loads(R"({"field": {"file": "history.nc"}})")),
          "forcing.load[0].field.file",
          "names a variable \"load\" that " + (dir_ / "history.nc").string() +
              " does not hold; it holds \"ice_thickness\", \"topography\"");
  refused(run, config(ball, loads(R"({"field": {"file": "history.nc",
                                           "variable": "thk"}})")),
          "forcing.load[0].field.variable",
          "it holds \"ice_thickness\", \"topography\"");
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

/// The ball's model block and degree.
constexpr auto kBall = R"("model": {"deck": "ball.deck"},
                          "discretisation": {"lmax": 8})";

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
      "discretisation": {{"lmax": 8}})",
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

/// The surface coefficients of a response, SI, in the grid's layout: the
/// radial displacement U and the tangential displacement's potential V
/// (m), and the potential perturbation Phi (m^2 s^-2).
struct SurfaceCoefficients {
  std::vector<Complex> U, V, Phi;
};

/// Per-degree m >= 0 columns as coefficients to `lmax`, each times
/// `factor`.
std::vector<Complex> CoefficientsOf(
    const std::map<int, std::vector<Complex>>& columns, int lmax,
    double factor) {
  auto out = std::vector<Complex>(SurfaceGrid::Index(lmax, lmax) + 1);
  for (const auto& [l, column] : columns) {
    for (auto m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] = factor * column[m];
    }
  }
  return out;
}

/// Writes the response inside the body of degree solutions `x` (the
/// reference's dof layout) as the command writes its volume file: the
/// radial displacement and the deviatoric strain's magnitude at the
/// memory nodes of the reference's coupling, and, under lateral moduli,
/// the relative perturbations applied.
void WriteVolume(const fs::path& path, const ElasticPerturbation3D& reference,
                 const Viscoelastic3D::Solutions& x, const RadialMesh& mesh,
                 const EarthModel& model,
                 const io::ModelModuli* moduli = nullptr) {
  const auto& coupling = reference.Coupling();
  const auto fields = io::ResponseFieldsAtNodes(
      coupling,
      [&](int l) -> const DegreeOperator& { return reference.Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return reference.Toroidal(l); },
      x);
  auto shear = std::vector<double>{}, bulk = std::vector<double>{};
  auto written = std::vector<io::NodeField>{
      {.name = "radial_displacement",
       .meaning = "radial displacement",
       .dimensions = Dimensions{.length = 1},
       .values = fields.radialDisplacement},
      {.name = "deviatoric_strain",
       .meaning = "the deviatoric strain's magnitude",
       .dimensions = {},
       .values = fields.deviatoricStrain}};
  if (moduli != nullptr && moduli->Shear()) {
    shear = io::SampleNodeField(coupling, moduli->Shear());
    written.push_back({.name = "mu_relative",
                       .meaning = "the shear modulus's perturbation",
                       .dimensions = {},
                       .values = shear});
  }
  if (moduli != nullptr && moduli->Bulk()) {
    bulk = io::SampleNodeField(coupling, moduli->Bulk());
    written.push_back({.name = "kappa_relative",
                       .meaning = "the bulk modulus's perturbation",
                       .dimensions = {},
                       .values = bulk});
  }
  const auto done = io::WriteNodeFields(path, coupling, mesh, model, written);
  EXPECT_TRUE(done) << done.error().message;
}

/// The elastic response of a deck's radial model at `lmax`, without an
/// ocean or rotational feedback, to a surface load and an external
/// potential given by their coefficients (SI, the grid's layout), by the
/// library's static solves on the radial reference systems
/// (ElasticPerturbation3D::Solve under its load and tide forces): an
/// oracle that does not pass through the command's one-node evolution.
/// With `volume`, the response inside the body is written there
/// (WriteVolume).
SurfaceCoefficients RadialElastic(const fs::path& deck, app::ModelFormat format,
                                  int lmax, std::vector<Complex> load,
                                  std::vector<Complex> tide,
                                  const fs::path& volume = {}) {
  const auto model = ModelOf(deck, format);
  const auto mesh = RadialMesh::Create(model, {.lmax = lmax});
  EXPECT_TRUE(mesh);
  const auto material = Material::Create(model, mesh.value());
  EXPECT_TRUE(material);
  const auto& scales = model.ModelScales();
  const auto density = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto potential = scales.Factor(PotentialDimensions);
  for (auto& c : load) c /= density;
  for (auto& c : tide) c /= potential;
  const ElasticPerturbation3D reference(material.value(), lmax);
  auto forces = reference.LoadForces(load);
  for (auto& [l, f] : reference.TideForces(tide)) {
    auto& sum = forces[l];
    if (sum.empty()) sum.assign(f.size(), Complex{});
    for (auto i = std::size_t{0}; i < f.size(); ++i) sum[i] += f[i];
  }
  const auto x = reference.Solve(forces, {});
  if (!volume.empty()) WriteVolume(volume, reference, x, mesh.value(), model);
  return {CoefficientsOf(reference.Surface(x, Component::U), lmax,
                         scales.Length()),
          CoefficientsOf(reference.Surface(x, Component::V), lmax,
                         scales.Length()),
          CoefficientsOf(reference.Surface(x, Component::Phi), lmax,
                         potential)};
}

/// The elastic response of a deck's model under the lateral moduli of a
/// model file (about their lateral mean), without an ocean or rotational
/// feedback, to a surface load and an external potential given by their
/// coefficients (SI, the grid's layout), by the library's
/// conjugate-gradient elastic solve on a branchless rheology
/// (Viscoelastic3D::SolveElastic, to `rtol`): an oracle that does not
/// pass through the command's zero-length step. With `volume`, the
/// response inside the body is written there (WriteVolume, on the radial
/// reference's coupling).
SurfaceCoefficients LateralElastic(const fs::path& deck, app::ModelFormat format,
                                   int lmax, const fs::path& moduliFile,
                                   std::vector<Complex> load,
                                   std::vector<Complex> tide, double rtol,
                                   const fs::path& volume = {}) {
  const auto model = ModelOf(deck, format);
  const auto mesh = RadialMesh::Create(model, {.lmax = lmax});
  EXPECT_TRUE(mesh);
  const auto material = Material::Create(model, mesh.value());
  EXPECT_TRUE(material);
  const auto moduli = io::ModelModuli::Read(
      moduliFile, model, {.reference = io::ModulusReference::LateralMean});
  EXPECT_TRUE(moduli) << moduli.error().message;
  const auto elastic = Rheology::Create(material.value(), {});
  EXPECT_TRUE(elastic);
  auto options = Viscoelastic3D::Options{};
  options.solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  options.elastic_rtol = rtol;
  options.elastic_max_iterations = 200;
  options.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
  const Viscoelastic3D op(elastic.value(), lmax,
                          [](double, double, double) { return 0.0; }, options);
  const auto& scales = model.ModelScales();
  const auto density = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto potential = scales.Factor(PotentialDimensions);
  // the degrees a field has, its m >= 0 columns non-dimensional
  const auto columns = [lmax](std::vector<Complex> c, double scale)
      -> Viscoelastic3D::LoadCallable3D {
    for (auto& v : c) v /= scale;
    return [c = std::move(c), lmax](int l, double)
               -> std::optional<std::vector<Complex>> {
      if (l > lmax) return std::nullopt;
      auto out = std::vector<Complex>(l + 1);
      for (auto m = 0; m <= l; ++m) out[m] = c[SurfaceGrid::Index(l, m)];
      if (std::ranges::all_of(out, [](Complex v) { return v == Complex{}; })) {
        return std::nullopt;
      }
      return out;
    };
  };
  const auto x = op.SolveElastic(op.ZeroState(), columns(std::move(load), density),
                                 columns(std::move(tide), potential));
  EXPECT_TRUE(op.LastElasticSolve().converged);
  if (!volume.empty()) {
    const ElasticPerturbation3D reference(material.value(), lmax);
    WriteVolume(volume, reference, x, mesh.value(), model, &moduli.value());
  }
  return {CoefficientsOf(op.Surface(x, Component::U), lmax, scales.Length()),
          CoefficientsOf(op.Surface(x, Component::V), lmax, scales.Length()),
          CoefficientsOf(op.Surface(x, Component::Phi), lmax, potential)};
}

TEST_F(AppsTest, ViscoelasticResponseStartsFromTheElasticResponse) {
  // The first node of a viscoelastic run without an ocean is the
  // unrelaxed solve under the forcing there with no memory before it: the
  // static problem's elastic response to the same forcing. On the radial
  // ball the radial stepper's unrelaxed systems (the branch moduli at the
  // memory points) and the static problem's branchless ones agree to
  // rounding, about 6e-12; under lateral moduli both close the 3D problem
  // by the combined fixed point of a zero-length step, to the same
  // tolerance (1e-10), and agree well within it. A load ramped up from zero
  // has no response at the first node. The surface file has a slice per
  // node, its time axis the run's clock in seconds, its nodes listed in
  // the run's unit.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  const auto forcing =
      std::format(R"({{"load": [{}], "tide": [{}]}})", kCap, kTide);
  const auto elastic = Respond("elastic", forcing);
  const auto summary =
      Run("visco", std::format(R"({}, "times": {{"unit": "kyr", "end": 2,
                                                 "nodes": 5}},
                              "forcing": {})",
                               kBall, forcing));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 0 to 2 kyr on the run's clock, no ocean, "
      "rotation none; load (1 term) + tide (1 coefficient); wrote "))
      << summary;
  EXPECT_TRUE(summary.ends_with("and 5 exchange tables")) << summary;
  const auto visco = Output("visco");
  ASSERT_TRUE(visco.time);
  ASSERT_EQ(visco.time->size(), 5u);
  for (auto n = std::size_t{0}; n < 5; ++n) {
    EXPECT_DOUBLE_EQ((*visco.time)[n], 0.5 * static_cast<double>(n) * app::KYR);
  }
  EXPECT_EQ(std::get<std::vector<double>>(visco.meta.at("times")),
            (std::vector<double>{0.0, 0.5, 1.0, 1.5, 2.0}));
  EXPECT_EQ(std::get<std::string>(visco.meta.at("time_unit")), "kyr");
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
  // the cap as sampled in both, no transform between: bitwise
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
         "discretisation": {"lmax": 8})"};
  constexpr auto kSolver = R"("solver": {"step_rtol": 1e-10,
                                         "step_max_iterations": 200})";
  static_cast<void>(app::RunSolve(Write(
      "elastic3d.json", std::format(R"({{"problem": "static", {}, "forcing": {},
                       {}, "rotation": "none",
                       "output": {{"surface": "out/elastic3d.nc"}}}})",
                                    lateral, forcing, kSolver))));
  const auto lateralSummary = Run(
      "visco3d", std::format(R"({}, "times": {{"unit": "kyr", "end": 1,
                                             "nodes": 2}},
                                "forcing": {}, {})",
                             lateral, forcing, kSolver));
  EXPECT_TRUE(lateralSummary.starts_with(
      "3-D (lateral moduli), viscoelastic, 2 nodes 0 to 1 kyr on the run's "
      "clock, no ocean, rotation none; "))
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
  EXPECT_GT(iterations.front(), 0);  // the first node's zero-length step

  // a load ramped up from zero: nothing at the first node
  static_cast<void>(
      Run("ramp", std::format(R"({}, "times": {{"unit": "kyr", "end": 2,
                                                "nodes": 3}},
          "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
              "width_deg": 20, "amplitude_kg_m2": 1.0e6}},
              "history": [[0, 0], [2, 1]]}}]}})",
                              kBall)));
  const auto ramp = Output("ramp");
  // exactly zero: the forcing there is zero, and so is all formed from it
  for (const auto* name : {"load", "displacement", "potential"}) {
    const auto first = Slice(ramp, name, 0);
    EXPECT_TRUE(std::ranges::all_of(first, [](double v) { return v == 0.0; }))
        << name;
  }
  const auto last = Slice(ramp, "displacement", 2);
  EXPECT_TRUE(std::ranges::any_of(last, [](double v) { return v != 0.0; }));
}

TEST_F(AppsTest, AStepThatMissesItsToleranceSaysWhereItStopped) {
  // A 3-D step's combined fixed point that misses `solver.step_rtol` is an
  // error stating the residual it stopped at, its passes and the tolerance
  // asked for, so that a tolerance below the iteration's floor shows as
  // such, with no check of its own: under lateral moduli the static
  // problem's one zero-length step at 1e-15, below the floor of its
  // residual; and a step of a run through time under a lateral viscosity,
  // stopped by its budget.
  static_cast<void>(Deck());
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", model, SurfaceGrid(8),
      [](double, double theta, double) { return 0.1 * std::cos(theta); }, {},
      {.name = "toy moduli", .elementMetres = 500e3}));
  WriteViscosity(dir_ / "lateral.nc", model, 1e21, 0.5);
  const auto message = [&](const std::string& name, const std::string& text) {
    try {
      static_cast<void>(app::RunSolve(Write(name + ".json", text)));
      ADD_FAILURE() << name << ": a step that missed its tolerance ran";
    } catch (const std::runtime_error& error) {
      return std::string{error.what()};
    }
    return std::string{};
  };
  // the residual a message states
  const auto residualOf = [](const std::string& text) {
    const auto at = text.find("residual ");
    return at == std::string::npos ? 0.0
                                   : std::stod(text.substr(at + 9));
  };

  const auto floor = message(
      "floor", std::format(R"({{"problem": "static",
        "model": {{"deck": "ball.deck", "moduli": {{"file": "moduli.nc"}}}},
        "discretisation": {{"lmax": 8}}, "forcing": {{"load": [{}]}},
        "rotation": "none",
        "solver": {{"step_rtol": 1e-15, "step_max_iterations": 40}},
        "output": {{"surface": "out/floor.nc"}}}})",
                           kCap));
  EXPECT_NE(floor.find("the instantaneous coupled fixed point at the start "
                       "did not converge: residual "),
            std::string::npos)
      << floor;
  EXPECT_NE(floor.find(" after 40 passes, against the tolerance 1e-15; "
                       "loosen solver.step_rtol or raise "
                       "solver.step_max_iterations"),
            std::string::npos)
      << floor;
  // the floor, a rounding level well above the tolerance and well below
  // the default's 1e-9 (measured 6.3e-14; it wanders at that level, 1.4e-13
  // after 200 passes)
  const auto stalled = residualOf(floor);
  RecordProperty("moduli_floor", std::format("{:.3g}", stalled));
  EXPECT_GT(stalled, 1e-14) << floor;
  EXPECT_LT(stalled, 1e-11) << floor;

  const auto budget = message(
      "budget", std::format(R"({{"problem": "quasi_static",
        "model": {{"deck": "ball.deck", "viscosity": {{"file": "lateral.nc"}}}},
        "discretisation": {{"lmax": 8}},
        "times": {{"unit": "kyr", "end": 2, "nodes": 3}},
        "forcing": {{"load": [{}]}}, "rotation": "none",
        "solver": {{"step_rtol": 1e-15, "step_max_iterations": 2}},
        "output": {{"surface": "out/budget.nc"}}}})",
                            kCap));
  EXPECT_NE(budget.find("the coupled fixed point or its sea level solve did "
                        "not converge at 1 kyr"),
            std::string::npos)
      << budget;
  EXPECT_NE(budget.find(": the fixed point's residual "), std::string::npos)
      << budget;
  EXPECT_NE(budget.find(" after 2 passes, against the tolerance 1e-15 "
                        "(solver.step_rtol;"),
            std::string::npos)
      << budget;
  EXPECT_GT(residualOf(budget), 1e-15) << budget;
}

TEST_F(AppsTest, ViscoelasticResponseMatchesTheLibraryStepper) {
  // A run without an ocean and without rotational feedback is the
  // library's stepper under the forcing at each node: a loop written here,
  // the radial ball's ViscoelasticOperator stepped per degree (the orders
  // as columns) and the 3D Viscoelastic3D::Step on the prem8 deck's
  // lateral viscosity (its fixed point to 1e-10, the run's solver.step_rtol
  // too), each under the cap's coefficients and the tide's scaled by their
  // histories at the nodes and made non-dimensional by the model's scales,
  // gives the command's exchange tables to 1e-10 relative (the same
  // steps: the departures are rounding, about 3e-12 here). Uneven nodes,
  // so that each step's length counts.
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
                "history": [[0, 0.5], [1, 1]]}],
      "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                "imaginary_m2_s2": 0.5, "history": [[0, 1], [1.5, -1]]}]})"};
  const auto times =
      std::string{R"("times": {"unit": "kyr", "at": [0, 0.4, 1.0, 1.5]})"};
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
        Run("lateral",
            std::format(R"({}, {}, {}, "solver": {{"step_rtol": 1e-10,
                                                  "step_max_iterations": 200}})",
                        Prem8(), times, forcing))
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
  // it equals the radial run to the fixed point's tolerance (1e-10, its
  // solver.step_rtol). Its radial reference model, written beside it, is
  // the radial run to rounding.
  static_cast<void>(Deck());
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  WriteViscosity(dir_ / "radial.nc", model, 1e21, 0.0, true);
  WriteViscosity(dir_ / "uniform.nc", model, 1e21, 0.0);
  const auto rest = std::format(
      R"("times": {{"unit": "kyr", "end": 2, "nodes": 5}},
         "forcing": {{"load": [{}], "tide": [{}]}})",
      kCap, kTide);
  const auto withFile = [&](const std::string& file,
                            const std::string& solver = "") {
    return std::format(R"("model": {{"deck": "ball.deck", "viscosity":
        {{"file": "{}"}}}}, "discretisation": {{"lmax": 8}}, {}{})",
                       file, rest, solver);
  };
  EXPECT_TRUE(Run("deck", std::string{kBall} + ", " + rest)
                  .starts_with("1-D, viscoelastic"));
  const auto radialSummary = Run("radial", withFile("radial.nc"));
  EXPECT_TRUE(radialSummary.starts_with(
      "1-D (the viscosity file is radial), viscoelastic"))
      << radialSummary;
  const auto uniformSummary =
      Run("uniform", withFile("uniform.nc", R"(, "solver": {"step_rtol": 1e-10,
                                                "step_max_iterations": 200})"));
  EXPECT_TRUE(
      uniformSummary.starts_with("3-D (lateral viscosity), viscoelastic"))
      << uniformSummary;
  const auto deck = Output("deck");
  const auto radial = Output("radial");
  const auto uniform = Output("uniform");
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto& want = deck.variables.at(name).values;
    EXPECT_LT(Departure(radial.variables.at(name).values, want), 1e-10) << name;
    const auto departure = Departure(uniform.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-10) << name;
  }
  RecordProperty("lateral_departure", std::format("{:.3e}", worst));
  EXPECT_LT(Departure(uniform.variables.at("displacement_radial_model").values,
                      deck.variables.at("displacement").values),
            1e-10);
  EXPECT_LT(Departure(uniform.variables.at("potential_radial_model").values,
                      deck.variables.at("potential").values),
            1e-10);
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
      R"("times": {"unit": "kyr",
                   "at": [0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]})"};
  const auto run = [&](const std::string& name, const std::string& load,
                       const std::string& tide) {
    static_cast<void>(Run(name, std::format(R"({}, {}, "forcing": {{
            "load": [{{"cap": {{"latitude": 60, "longitude": -85,
                                "width_deg": 20, "amplitude_kg_m2": 1.0e6}},
                       "history": {}}}],
            "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                       "imaginary_m2_s2": 0.5, "history": {}}}]}})",
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
    // a factor of zero leaves the load exactly zero
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
  // present, read from `time_origin` 2 (kyr, the run's unit), likewise;
  // and on a 1-degree grid read by the cubic rule, to the resampling's
  // error (about 3e-8, as the static field term's). Past the file's last
  // date its last field is held where the term says so (`"outside_span":
  // "hold"`), and the read is refused where it does not. A history on a
  // field term multiplies it. The static problem reads the file at its
  // `date` on the file's own axis: the cap at 2 kyr, half of it at 1 kyr,
  // and its summary says so.
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
    static_cast<void>(Run(name, std::format(R"({}, "times": {{"unit": "kyr",
        "end": 3, "nodes": 7}}, "forcing": {{"load": [{}]}})",
                                            kBall, term)));
    return Output(name);
  };
  const auto cap = run("cap", R"({"cap": {"latitude": 60, "longitude": -85,
      "width_deg": 20, "amplitude_kg_m2": 1.0e6},
      "history": [[0, 0], [2, 1]]})");
  const auto own = run(
      "own", R"({"field": {"file": "ramp.nc", "outside_span": "hold"}})");
  const auto shifted = run("shifted", R"({"field": {"file": "ramp_bp.nc",
      "time_origin": 2, "outside_span": "hold"}})");
  const auto resampled = run("resampled", R"({"field": {"file": "fine.nc",
      "variable": "sigma", "outside_span": "hold"}})");
  const auto doubled = run("doubled", R"({"field": {"file": "ramp.nc",
      "outside_span": "hold"}, "history": [[0, 2]]})");
  // without the member the nodes past the file's last date are refused,
  // before the run
  ExpectRefused(app::RunSolve,
                Write("past.json",
                      std::format(R"({{"problem": "quasi_static", {},
                        "times": {{"unit": "kyr", "end": 3, "nodes": 7}},
                        "forcing": {{"load": [{{"field": {{"file":
                        "ramp.nc"}}}}]}}, "rotation": "none",
                        "output": {{"surface": "out/past.nc"}}}})",
                                  kBall)),
                "forcing.load[0].field.file",
                "spans 0 to 2 kyr on the file's own axis, which leaves out "
                "2.5 kyr on the file's own axis");
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

  // the static problem at the file's dates
  const auto capStatic =
      Respond("cap_static", std::format(R"({{"load": [{}]}})", kCap));
  const auto at2 = Respond("at2", R"({"load": [{"field": {"file": "ramp.nc",
                                     "date": 2, "unit": "kyr"}}]})");
  const auto at1 = Respond("at1", R"({"load": [{"field": {"file": "ramp.nc",
                                     "date": 1, "unit": "kyr"}}]})");
  for (const auto* name :
       {"load", "displacement", "tangential_potential", "potential"}) {
    const auto& want = capStatic.variables.at(name).values;
    EXPECT_LT(Departure(at2.variables.at(name).values, want), 1e-12) << name;
    auto half = want;
    for (auto& v : half) v *= 0.5;
    EXPECT_LT(Departure(at1.variables.at(name).values, half), 1e-12) << name;
  }
  const auto summary = app::RunSolve(Write("dated.json", R"({
      "problem": "static", "model": {"deck": "ball.deck"},
      "discretisation": {"lmax": 8},
      "forcing": {"load": [{"field": {"file": "ramp.nc", "date": 2,
                                      "unit": "kyr"}}]},
      "rotation": "none", "output": {"surface": "out/dated.nc"}})"));
  EXPECT_NE(summary.find("; field " + (dir_ / "ramp.nc").string() +
                         ":load (2 kyr); wrote "),
            std::string::npos)
      << summary;
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
        Run(name, std::format(R"({}, "times": {{"unit": "kyr", "end": 1,
                                               "nodes": {}}},
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
  // A 3D run without an ocean under a lateral viscosity and lateral moduli
  // closes each step by the fixed point or by the closure nested over
  // conjugate-gradient solves, to the same tolerance (1e-10 a step): the
  // two agree to it over the run.
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
      R"({}, "times": {{"unit": "kyr", "end": 1, "nodes": 3}},
         "forcing": {{"load": [{}], "tide": [{}]}})",
      Prem8(R"(, "moduli": {"file": "moduli.nc"})"), kCap, kTide);
  const auto fixed = Run("fixed", body + R"(, "solver": {"step_rtol": 1e-10,
      "step_max_iterations": 200})");
  const auto conjugate = Run("conjugate", body + R"(, "solver": {
      "step": "conjugate_gradients", "step_rtol": 1e-10,
      "step_max_iterations": 200})");
  EXPECT_TRUE(fixed.starts_with(
      "3-D (lateral viscosity, lateral moduli), viscoelastic, 3 nodes 0 to 1 "
      "kyr on the run's clock, no ocean, rotation none; load (1 term) + tide "
      "(1 coefficient); wrote "))
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
  // Nodes listed with `at` are the evenly spaced ones when they coincide,
  // to the bit; uneven ones run and are recorded as given. The exchange
  // tables are named by the node's index padded to the width of the last,
  // and exchange_nodes chooses them.
  static_cast<void>(Deck());
  const auto forcing =
      std::format(R"("forcing": {{"load": [{}], "tide": [{}]}})", kCap, kTide);
  static_cast<void>(Run(
      "even", std::format(R"({}, "times": {{"unit": "kyr", "end": 2,
                                           "nodes": 5}}, {})",
                          kBall, forcing)));
  static_cast<void>(Run(
      "listed", std::format(R"({}, "times": {{"unit": "kyr",
                                            "at": [0, 0.5, 1, 1.5, 2]}}, {})",
                            kBall, forcing)));
  const auto even = Output("even");
  const auto listed = Output("listed");
  // the same node times either way, the same transforms: bitwise
  EXPECT_EQ(*even.time, *listed.time);
  for (const auto& [name, field] : even.variables) {
    EXPECT_EQ(field.values, listed.variables.at(name).values) << name;
  }

  // uneven nodes, starting late, with the tables of three of them
  const auto config =
      Write("uneven.json", std::format(R"({{"problem": "quasi_static", {},
                     "times": {{"unit": "kyr", "at": [1, 1.1, 1.4, 2, 3.5, 4,
                     5, 6, 7, 8, 9.5]}}, {}, "rotation": "none",
          "output": {{"surface": "out/uneven.nc",
                      "exchange": "out/uneven_{{node}}_{{node}}.txt",
                      "exchange_nodes": [10, 0, 3, 3]}}}})",
                                       kBall, forcing));
  const auto summary = app::RunSolve(config);
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 11 nodes 1 to 9.5 kyr on the run's clock"))
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
  // Each fault of a viscoelastic config without an ocean is refused naming
  // the member: the times block's forwards form given both ways or
  // neither, out of order or too short; a history out of order or
  // malformed; an elastic deck with no viscosity file; the output's
  // {node}, missing in the quasi-static problem and present in the static
  // one, a volume file, exchange nodes out of range or without their
  // tables; the time step's members and a time origin in the static
  // problem; the kernels block's members at the top level, and a kernels
  // block without its measurements. Once the run reads
  // its deck, a Maxwell deck with no viscosity anywhere.
  static_cast<void>(Deck());
  static_cast<void>(Write("inviscid.deck",
                          "a ball with no viscosity\n"
                          "gia deck for the sl3d tests\n"
                          "0 5500 10000 5500 10000 5500 1 nan\n"
                          "6371000 5500 10000 5500 10000 5500 1 nan\n"));
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto run = [](const fs::path& path) { return app::RunSolve(path); };
  const auto config =
      [&](const std::string& times, const std::string& extra = "",
          const std::string& output =
              R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
          const std::string& model = R"({"deck": "ball.deck"})",
          const std::string& load = kCap) {
        return std::format(R"({{"problem": "{}", "model": {},
                           "discretisation": {{"lmax": 4}}, {}
                           "forcing": {{"load": [{}]}},{} "output": {}}})",
                           times.empty() ? "static" : "quasi_static", model,
                           times, load, extra, output);
      };
  const auto even =
      std::string{R"("times": {"unit": "kyr", "end": 2, "nodes": 3},)"};

  refused(read,
          config(R"("times": {"unit": "kyr", "end": 2, "nodes": 3,
                              "at": [0, 1]},)"),
          "times", "gives the nodes twice");
  refused(read, config(R"("times": {"unit": "kyr", "start": 1},)"), "times",
          "gives no nodes");
  refused(read, config(R"("times": {"unit": "kyr", "end": 2},)"), "times",
          "gives `end` without `nodes`");
  refused(read, config(R"("times": {"unit": "kyr", "end": 2, "nodes": 1},)"),
          "times.nodes", "fewer than two nodes");
  refused(read,
          config(R"("times": {"unit": "kyr", "start": 2, "end": 2,
                              "nodes": 3},)"),
          "times.end", "is not later than `start`");
  refused(read, config(R"("times": {"unit": "kyr", "at": [0, 1, 1]},)"),
          "times.at[2]", "is not later than the node before it");
  refused(read, config(R"("times": {"unit": "kyr", "at": [0]},)"), "times.at",
          "fewer than two nodes");
  refused(read,
          config(R"("times": {"unit": "kyr", "start": 0, "at": [0, 1]},)"),
          "times.start", "is given by `at`");
  refused(read,
          config(R"("times": {"unit": "kyr", "end": 2, "nodes": 3,
                              "step": 1},)"),
          "times.step", "not a recognised member");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history": [[0, 1], [0, 2]]})"),
          "forcing.load[0].history[1][0]", "is not later than the time");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history": [[0, 1, 2]]})"),
          "forcing.load[0].history[0]", "is not a pair");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "ball.deck"})",
                 R"({"cap": {"latitude": 60, "longitude": 0, "width_deg": 10,
                     "amplitude_kg_m2": 1.0}, "history": []})"),
          "forcing.load[0].history", "holds no [time, factor] pair");
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
          "output.volume", "is the static problem's alone");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt",
                               "exchange_nodes": [0, 3]})"),
          "output.exchange_nodes[1]", "is not a node index in [0, 2]");
  refused(read,
          config(even, "", R"({"surface": "o.nc", "exchange_nodes": [0]})"),
          "output.exchange_nodes", "need `output.exchange`");
  refused(read, config("", "", R"({"surface": "o.nc", "exchange_nodes": [0]})"),
          "output.exchange_nodes", "needs the quasi-static problem");
  refused(read,
          config("", R"( "solver": {"scheme": "backward_euler"},)",
                 R"({"surface": "o.nc"})"),
          "solver.scheme", "needs the quasi-static problem");
  refused(read,
          config("", R"( "solver": {"step": "fixed_point"},)",
                 R"({"surface": "o.nc"})"),
          "solver.step", "needs the quasi-static problem");
  refused(read, config(even, R"( "solver": {"step": "picard"},)"),
          "solver.step", "none of");
  refused(read,
          config("", "", R"({"surface": "o.nc"})", R"({"deck": "ball.deck"})",
                 R"({"field": {"file": "f.nc", "time_origin": 2}})"),
          "forcing.load[0].field.time_origin",
          "needs the quasi-static problem");
  for (const auto* member : {"measurements", "adjoint"}) {
    refused(read, config(even, std::format(R"( "{}": {{}},)", member)), member,
            "is not a top-level member: it belongs in `kernels.");
  }
  refused(read, config(even, R"( "record_stride": 2,)"), "record_stride",
          "is not a top-level member: it belongs in "
          "`trajectory.record_stride`");
  // a kernels block is checked, and one without its measurements refused
  refused(read,
          config(even,
                 R"( "kernels": {"parameters": ["viscosity"], "output": "k"},)"),
          "kernels", "has no member");
  refused(run,
          config(even, "", R"({"surface": "o.nc", "exchange": "o_{node}.txt"})",
                 R"({"deck": "inviscid.deck"})"),
          "model.deck", "gives no viscosity");
}

// -- the sea level equation and its kernels -----------------------------------

/// The quasi-static problem over the toy history's window, from 2 kyr BP
/// to the present over `nodes` nodes, and its inputs: the state at the
/// window's end `at`, its topography and its ice, and the ice change, the
/// ice history as one `thickness` term (by default the one toy history for
/// all three, the ice file for the state's ice and the term), and the
/// ocean block's other members `ocean` where given (the shorelines, the
/// calibration).
std::string Window(int nodes, const std::string& topography = "history.nc",
                   const std::string& ice = "", const std::string& ocean = "",
                   const std::string& at = "start") {
  const auto history = ice.empty() ? topography : ice;
  return std::format(
      R"("problem": "quasi_static",
    "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": {}}},
    "forcing": {{"ice_change": [{{"thickness": {{"file": "{}"}}}}]}},
    "ocean": {{{}"at": "{}", "topography": {{"file": "{}"}},
              "ice": {{"file": "{}"}}}})",
      nodes, history, ocean.empty() ? std::string{} : ocean + ", ", at,
      topography, history);
}

/// The ocean block's members of a calibrated run under the given
/// shorelines, for Window with the topography at the window's end.
std::string Calibrated(const std::string& shorelines) {
  return std::format(R"("shorelines": "{}", "calibration": {{"rtol": 1e-8}})",
                     shorelines);
}

TEST_F(AppsTest, Sdirk2IsTheDampedSchemeOfTheRunsAndTheContinuousKernels) {
  // `solver.scheme` "sdirk2": a load switched on at the start and held
  // converges at second order, as ViscoelasticResponseConvergesUnderStepHalving
  // measures the trapezoid, the output naming the scheme; its kernels are
  // the continuous sweep's, stepped by the same stages, the discrete
  // adjoint (the default `kernels.method`) refused naming the choices, and
  // so is a recording with gaps (a stride other than 1); the gradient
  // calibration under migrating shorelines, whose sweep is the discrete
  // adjoint, is refused naming the choices, and taken under frozen ones
  // (no sweep), the fixed point calibrating under migrating ones.
  static_cast<void>(Deck());
  const auto final = [&](int nodes) {
    const auto name = std::format("sdirk2_{}", nodes);
    static_cast<void>(
        Run(name, std::format(R"({}, "times": {{"unit": "kyr", "end": 1,
                                               "nodes": {}}},
            "forcing": {{"load": [{}]}},
            "solver": {{"scheme": "sdirk2"}})",
                              kBall, nodes, kCap)));
    const auto file = Output(name);
    EXPECT_EQ(std::get<std::string>(file.meta.at("scheme")), "sdirk2");
    const auto last = Slice(file, "displacement", file.time->size() - 1);
    return *std::ranges::max_element(
        last, [](double a, double b) { return std::abs(a) < std::abs(b); });
  };
  auto values = std::vector<double>{};
  for (const auto nodes : {9, 17, 33, 65}) values.push_back(final(nodes));
  auto ratios = std::string{};
  for (auto k = std::size_t{2}; k < values.size(); ++k) {
    const auto ratio =
        (values[k - 2] - values[k - 1]) / (values[k - 1] - values[k]);
    ratios += std::format("{}{:.3f}", ratios.empty() ? "" : " ", ratio);
    EXPECT_GT(ratio, 3.5) << k;
    EXPECT_LT(ratio, 4.5) << k;
  }
  // measured: 4.03 and 4.01
  RecordProperty("sdirk2_ratios", ratios);

  WriteToyHistory(dir_ / "history.nc");
  // a dated run's kernels under Sdirk2: the kernels block's method member
  // and a top-level member, each where given
  const auto kernels = [&](const std::string& method, const std::string& top) {
    return std::format(R"({{{}, {}, "rotation": "none",
        "solver": {{"scheme": "sdirk2"}}, {}
        "kernels": {{"parameters": ["ice", "viscosity"], {}
          "measurements": [{{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}}],
          "output": "out/sdirk2_kernels"}}}})",
                       kBall, Window(5), top.empty() ? "" : top + ",",
                       method.empty() ? "" : method + ",");
  };
  constexpr auto kContinuous = R"("method": "continuous")";
  ExpectRefused(
      app::ReadKernelInput, Write("bad.json", kernels("", "")), "kernels",
      "the discrete adjoint (`method` \"discrete\", the default), the "
      "exact transpose of the stepped map, which is not built for "
      "`solver.scheme` \"sdirk2\": give `method` \"continuous\"");
  ExpectRefused(app::ReadKernelInput,
                Write("bad.json", kernels(R"("method": "discrete")", "")),
                "kernels.method",
                "or `solver.scheme` \"exponential_trapezoid\" or "
                "\"backward_euler\" for the discrete adjoint");
  ExpectRefused(
      app::ReadKernelInput,
      Write("bad.json",
            kernels(kContinuous, R"("trajectory": {"record_stride": 2})")),
      "trajectory.record_stride",
      "a run under `solver.scheme` \"sdirk2\" is recorded at every node");
  const auto summary =
      app::RunKernel(Write("sdirk2_kernels.json", kernels(kContinuous, "")));
  EXPECT_NE(summary.find("wrote 1 measurement(s) of kernels"),
            std::string::npos)
      << summary;
  EXPECT_NE(summary.find("continuous"), std::string::npos) << summary;
  // the calibrations under migrating and frozen shorelines, by `method`
  const auto calibrating = [&](const std::string& shorelines,
                               const std::string& method) {
    return Write(
        "sdirk2_calibrated.json",
        std::format(R"({{{}, {}, "rotation": "none",
                      "solver": {{"scheme": "sdirk2"}},
                      "output": {{"surface": "out/sdirk2_calibrated.nc"}}}})",
                    kBall,
                    Window(3, "history.nc", "",
                           std::format(R"("shorelines": "{}", "calibration":
                            {{"method": "{}", "rtol": 1e-6}})",
                                       shorelines, method),
                           "end")));
  };
  ExpectRefused(app::ReadSolveInput, calibrating("migrating", "gradient"),
                "ocean.calibration.method",
                "is the gradient calibration, whose gradient under migrating "
                "shorelines is the discrete adjoint's, which is not built for "
                "`solver.scheme` \"sdirk2\" (its exact derivative runs "
                "through each step's stages): give \"fixed_point\"");
  ExpectRefused(app::ReadKernelInput, calibrating("migrating", "gradient"),
                "ocean.calibration.method", "give \"fixed_point\"");
  EXPECT_NO_THROW(static_cast<void>(
      app::ReadSolveInput(calibrating("frozen", "gradient"))));
  const auto calibrated =
      app::RunSolve(calibrating("migrating", "fixed_point"));
  EXPECT_NE(calibrated.find("initial sea level calibrated"), std::string::npos)
      << calibrated;
}

TEST_F(AppsTest, SolveHoldsTheLandItIsTold) {
  // `ocean.land` names a field that is one where the surface is held as
  // land whatever the flotation: a band of the toy world's southern ocean
  // is then land at every node of a run under migrating shorelines, of a
  // calibrated one, and in the fingerprint, where without the member it
  // is ocean; the summary names the input; a variable under another name
  // gives the same run to the bit. A file without the variable, a date
  // on the member and a value that is not a number are refused.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyLand(dir_ / "land.nc");
  sl3dpp::test::WriteToyLand(dir_ / "held.nc", "held");
  sl3dpp::test::WriteToyLand(dir_ / "hole.nc", "land", true);

  // the ocean function's least and greatest values inside the band, and
  // its least south of it, over every node of a file
  struct Seen {
    double leastInBand = 1.0, greatestInBand = 0.0, leastSouth = 1.0;
  };
  auto seen = [](const pm::SurfaceFile& file) {
    const auto& values = file.variables.at("ocean_function").values;
    const auto ntheta = file.grid->Colatitudes().size();
    const auto nphi = file.grid->NPhi();
    const auto nodes = values.size() / (ntheta * nphi);
    auto out = Seen{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      for (auto i = std::size_t{0}; i < ntheta; ++i) {
        const auto theta = file.grid->Colatitudes()[i];
        for (auto j = std::size_t{0}; j < nphi; ++j) {
          const auto c = values[(n * ntheta + i) * nphi + j];
          if (theta > 2.05 && theta < 2.35) {
            out.leastInBand = std::min(out.leastInBand, c);
            out.greatestInBand = std::max(out.greatestInBand, c);
          } else if (theta > 2.7) {
            out.leastSouth = std::min(out.leastSouth, c);
          }
        }
      }
    }
    return out;
  };
  const auto run = [&](const std::string& name, const std::string& ocean,
                       const std::string& at = "start") {
    return Write(name + ".json",
                 "{" + std::string{kBall} + ", " +
                     Window(5, "history.nc", "", ocean, at) +
                     R"(, "rotation": "none", "output": {"surface": "out/)" +
                     name + R"(.nc"}})");
  };
  const auto migrating = std::string{R"("shorelines": "migrating")"};

  static_cast<void>(app::RunSolve(run("flotation", migrating)));
  const auto free = seen(pm::ReadSurfaceFile(dir_ / "out" / "flotation.nc"));
  EXPECT_EQ(free.leastInBand, 1.0);  // ocean by flotation
  EXPECT_EQ(free.leastSouth, 1.0);

  const auto summary = app::RunSolve(
      run("held", migrating + R"(, "land": {"file": "land.nc"})"));
  EXPECT_NE(summary.find(std::format(", land {}:land (held at every date)",
                                     (dir_ / "land.nc").string())),
            std::string::npos)
      << summary;
  const auto heldFile = pm::ReadSurfaceFile(dir_ / "out" / "held.nc");
  const auto held = seen(heldFile);
  EXPECT_EQ(held.greatestInBand, 0.0);  // land at every node
  EXPECT_EQ(held.leastSouth, 1.0);      // and the ocean beyond it untouched
  // the water goes elsewhere: the run is another run
  {
    const auto& got = heldFile.variables.at("relative_sea_level_change").values;
    const auto other = pm::ReadSurfaceFile(dir_ / "out" / "flotation.nc");
    const auto& want = other.variables.at("relative_sea_level_change").values;
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < want.size(); ++i) {
      scale = std::max(scale, std::abs(want[i]));
      diff = std::max(diff, std::abs(got[i] - want[i]));
    }
    EXPECT_GT(diff, 1e-3 * scale);
  }

  // the variable under another name: the same run
  static_cast<void>(app::RunSolve(run(
      "named",
      migrating + R"(, "land": {"file": "held.nc", "variable": "held"})")));
  // one process, the same inputs and transforms: bitwise
  EXPECT_EQ(pm::ReadSurfaceFile(dir_ / "out" / "named.nc")
                .variables.at("relative_sea_level_change")
                .values,
            heldFile.variables.at("relative_sea_level_change").values);

  // through a calibration, whose runs start again from the ice history
  static_cast<void>(app::RunSolve(
      run("calibrated",
          Calibrated("migrating") + R"(, "land": {"file": "land.nc"})", "end")));
  const auto calibrated =
      seen(pm::ReadSurfaceFile(dir_ / "out" / "calibrated.nc"));
  EXPECT_EQ(calibrated.greatestInBand, 0.0);
  EXPECT_EQ(calibrated.leastSouth, 1.0);

  // the fingerprint
  const auto fingerprint = [&](const std::string& name,
                               const std::string& land) {
    return Write(name + ".json", std::format(R"({{
    "problem": "static", {},
    "forcing": {{"ice_change": [{{"cap_fraction": {{"latitude": 90,
                "longitude": 0, "width_deg": 15, "fraction": -0.1}}}}]}},
    "ocean": {{"date_kyr_bp": 2.0, "topography": {{"file": "history.nc"}},
              "ice": {{"file": "history.nc"}}{}}},
    "rotation": "none",
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, land, name));
  };
  static_cast<void>(app::RunSolve(fingerprint("print", "")));
  EXPECT_EQ(seen(pm::ReadSurfaceFile(dir_ / "out" / "print.nc")).leastInBand,
            1.0);
  const auto printed = app::RunSolve(
      fingerprint("print_held", R"(, "land": {"file": "land.nc"})"));
  EXPECT_NE(printed.find(":land (held at every date)"), std::string::npos)
      << printed;
  EXPECT_EQ(
      seen(pm::ReadSurfaceFile(dir_ / "out" / "print_held.nc")).greatestInBand,
      0.0);

  // refusals
  ExpectRefused(
      app::RunSolve,
      run("absent", migrating + R"(, "land": {"file": "history.nc"})"),
      "ocean.land", "ice_thickness");
  ExpectRefused(app::RunSolve,
                run("dated", migrating + R"(, "land": {"file": "land.nc",
                                              "date_kyr_bp": 0.0})"),
                "date_kyr_bp", "");
  ExpectRefused(app::RunSolve,
                run("hole", migrating + R"(, "land": {"file": "hole.nc"})"),
                "ocean.land", "not a number");
}

TEST_F(AppsTest, SolveFingerprintRunsAToyStateEndToEnd) {
  // In the static problem, solve is the elastic fingerprint of an ice
  // change on the state: written as a surface file of static fields, the
  // ocean function in [0, 1], the far field raised by melting ice, the
  // summary and the metadata naming the state's inputs, the change's term
  // and the shorelines. With the moments of inertia from the model's
  // hydrostatic limit, it is the library's Fingerprint under those
  // parameters, solved here on the same state and change (a cap of ice
  // laid off the pole), to rounding (about 4e-16 here); the moments
  // change the response to a change off the pole (by about 2e-2).
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
  state.variables.emplace("ice_thickness",
                          pm::SurfaceField("ice_thickness", ice,
                                           {.dimensions = pm::units::LENGTH}));
  state.variables.emplace("topography",
                          pm::SurfaceField("topography", topography,
                                           {.dimensions = pm::units::LENGTH}));
  pm::WriteSurfaceFile(dir_ / "state.nc", state);

  const auto config = [&](const std::string& name, const std::string& change,
                          const std::string& extra) {
    return Write(name + ".json", std::format(R"({{
    "problem": "static",
    "model": {{"deck": "ball.deck"}},
    "discretisation": {{"lmax": 8}},
    "forcing": {{"ice_change": [{}]}},
    "ocean": {{"topography": {{"file": "state.nc"}},
              "ice": {{"file": "state.nc"}}}},
    "rotation": "full",{}
    "output": {{"surface": "out/{}.nc"}}}})",
                                             change, extra, name));
  };
  const auto summary = app::RunSolve(
      config("fingerprint", R"({"cap_fraction": {"latitude": 90,
             "longitude": 0, "width_deg": 15, "fraction": -0.1}})",
             ""));
  // the inputs as read: the file, its variable, and static (no time
  // axis); then the change's term
  const auto stateFile = (dir_ / "state.nc").string();
  constexpr auto kTerm =
      "cap at latitude 90, longitude 0, width 15 deg, fraction -0.1 of the "
      "state's ice";
  EXPECT_TRUE(summary.starts_with(std::format(
      "1-D, elastic, static, shorelines frozen, rotation full; ice "
      "{}:ice_thickness "
      "(static), topography {}:topography (static); ice change {}; wrote ",
      stateFile, stateFile, kTerm)))
      << summary;
  EXPECT_TRUE(summary.ends_with("fingerprint.nc")) << summary;

  const auto result = pm::ReadSurfaceFile(dir_ / "out" / "fingerprint.nc");
  for (const auto name :
       {"ice_change", "relative_sea_level_change", "displacement",
        "tangential_potential", "potential", "ocean_function"}) {
    EXPECT_TRUE(result.variables.contains(name)) << name;
  }
  // a radial Earth under an ice change has no toroidal displacement, and
  // frozen shorelines no final ocean function of their own
  for (const auto name :
       {"toroidal_potential", "ocean_function_final", "load_in_water_budget",
        "load_outside_water_budget", "solid_surface_change"}) {
    EXPECT_FALSE(result.variables.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(result.meta.at("shorelines")), "frozen");
  EXPECT_FALSE(result.meta.contains("load"));
  EXPECT_FALSE(result.meta.contains("seismic"));
  EXPECT_TRUE(result.meta.contains("omega_z_rad_s"));
  EXPECT_EQ(std::get<std::string>(result.meta.at("ice")),
            stateFile + ":ice_thickness (static)");
  EXPECT_EQ(std::get<std::string>(result.meta.at("sea_level")),
            "topography " + stateFile + ":topography (static)");
  EXPECT_EQ(std::get<std::string>(result.meta.at("ice_change")), kTerm);
  EXPECT_EQ(result.grid->Kind(), "gauss_legendre");
  const auto& ocean = result.variables.at("ocean_function").values;
  for (const auto value : ocean) {
    EXPECT_GE(value, -0.05);
    EXPECT_LE(value, 1.05);
  }
  // melting ice must raise the far-field sea: somewhere positive
  const auto& change = result.variables.at("relative_sea_level_change").values;
  auto largest = 0.0;
  for (const auto value : change) largest = std::max(largest, value);
  EXPECT_GT(largest, 0.0);

  // a cap of ice laid off the pole, so that the polar wander, and with it
  // the moments of inertia, count: the hydrostatic moments' run is the
  // library's
  constexpr auto kOffPole = R"({"cap_thickness": {"latitude": 70,
      "longitude": 30, "width_deg": 10, "thickness_m": 50}})";
  static_cast<void>(app::RunSolve(config("input", kOffPole, "")));
  const auto hydrostatic = app::RunSolve(
      config("hydrostatic", kOffPole, R"( "moments": "hydrostatic",)"));
  EXPECT_TRUE(
      hydrostatic.starts_with("1-D, elastic, static, shorelines frozen, "
                              "rotation full, moments hydrostatic; ice "))
      << hydrostatic;
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = 8});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  auto options = Fingerprint::Options{};
  options.parameters = WithHydrostaticMoments(SeaLevelParameters{},
                                              FluidTidalNumber(*material, 2));
  const auto love = ComputeLoveNumbers(*material, 8);
  Fingerprint fingerprint(8, love, options);
  const auto history =
      io::FieldHistory::Read(dir_ / "state.nc", fingerprint.Grid());
  ASSERT_TRUE(history);
  const auto ice0 = history->Values("ice_thickness", 0.0);
  fingerprint.SetState(
      ice0, SeaLevelFromTopography(history->Values("topography", 0.0), ice0,
                                   options.parameters));
  const auto cap = app::CapField(fingerprint.Grid(), {70.0, 30.0},
                                 10.0 * std::numbers::pi / 180.0, 1.0);
  auto thickness = std::vector<double>(cap.size());
  for (auto i = std::size_t{0}; i < cap.size(); ++i) {
    thickness[i] = 50.0 * cap[i];
  }
  const auto direct =
      fingerprint.SolveForIceChange(thickness, RotationalFeedback::Full);
  ASSERT_TRUE(direct.converged);
  const auto byCommand = pm::ReadSurfaceFile(dir_ / "out" / "hydrostatic.nc");
  const auto byInput = pm::ReadSurfaceFile(dir_ / "out" / "input.nc");
  auto worst = 0.0;
  for (const auto& [name, want] :
       {std::pair{"relative_sea_level_change", &direct.relativeSeaLevelChange},
        std::pair{"displacement", &direct.displacement},
        std::pair{"potential", &direct.potential}}) {
    const auto departure =
        Departure(byCommand.variables.at(name).values, *want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  const auto moments = Departure(byInput.variables.at("potential").values,
                                 byCommand.variables.at("potential").values);
  RecordProperty("moments_departure", std::format("{:.3e}", moments));
  EXPECT_GT(moments, 1e-6);
}

TEST_F(AppsTest, SolveFingerprintRefusesAStateWithoutTheFields) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  auto state = pm::SurfaceFile{};
  state.grid = pm::Equiangular(9, 18);
  state.source = "sl3dpp tests";
  state.variables.emplace(
      "ice_thickness",
      pm::SurfaceField("ice_thickness", std::vector<double>(9 * 18, 0.0),
                       {.dimensions = pm::units::LENGTH}));
  pm::WriteSurfaceFile(dir_ / "state.nc", state);
  const auto config = Write("run.json", R"({
    "problem": "static",
    "model": {"deck": "ball.deck"},
    "discretisation": {"lmax": 4},
    "forcing": {"ice_change": [{"cap_thickness": {"latitude": 0,
                "longitude": 0, "width_deg": 10, "thickness_m": 1.0}}]},
    "ocean": {"topography": {"file": "state.nc"},
              "ice": {"file": "state.nc"}},
    "output": {"surface": "out/fingerprint.nc"}})");
  ExpectRefused(app::RunSolve, config, "ocean.topography.file",
                "names a variable \"topography\" that " +
                    (dir_ / "state.nc").string() +
                    " does not hold; it holds \"ice_thickness\"");
}

TEST_F(AppsTest, SolveTakesSeismicSourcesWithoutAnOcean) {
  // `forcing.seismic` on a body without an ocean in a run through time: a
  // point moment tensor at a depth gives the library's evolution under the
  // same source (the stress glut held from the start), field for field,
  // with the toroidal potential written; a source with an onset does
  // nothing before it; a fault of one patch is the point double couple at
  // its centroid; the static problem under the source is the co-seismic
  // response, the run's first node (the branchless and the unrelaxed
  // systems, to their discretisations' difference); and a source given
  // both ways is refused.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  const auto run = [&](const std::string& name, const std::string& seismic,
                       const std::string& times =
                           R"("times": {"unit": "kyr", "at": [0, 0.4, 1.0]},)") {
    const auto config =
        Write(name + ".json",
              std::format(R"({{"problem": "quasi_static", {}, {} "forcing":
                     {{"seismic": [{}]}}, "rotation": "none",
                     "output": {{"surface": "out/{}.nc"}}}})",
                          kBall, times, seismic, name));
    return app::RunSolve(config);
  };
  const auto tensor = std::string{R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "moment_tensor_nm": {"rr": 9e20, "tt": -2e20,
      "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}}})"};
  const auto summary = run("point", tensor);
  EXPECT_NE(summary.find("seismic (1 source)"), std::string::npos) << summary;
  const auto file = pm::ReadSurfaceFile(dir_ / "out" / "point.nc");
  ASSERT_TRUE(file.variables.contains("toroidal_potential"));

  // the library's run under the same source
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax);
  const SurfaceGrid grid(kLmax);
  const GRDSolver solver(grid, op.ElasticLoveNumbers(), SeaLevelParameters{});
  const auto& units = solver.SolverScales();
  const auto energy = units.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto surface = material->Radius() * model.ModelScales().Length();
  const auto site = GeographicPoint{30.0, 40.0};
  const auto source = PointSource{
      .at = {.radius = (surface - 300e3) / units.Length(),
             .colatitude = site.CoLatitude(),
             .longitude = site.Longitude()},
      .moment = {.rr = 9e20 / energy, .tt = -2e20 / energy, .pp = -7e20 / energy,
                 .rt = 3e20 / energy, .rp = -7e20 / energy, .tp = 6e20 / energy}};
  auto times = std::vector<double>{};
  for (const auto t : {0.0, 0.4, 1.0}) times.push_back(t * app::KYR / units.Time());
  Evolution evolution(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
  const auto snapshots = evolution.Run(
      times, Evolution::Forcing{.directLoad = {},
                                .tide = {},
                                .sources = [&](double) {
                                  return std::vector<PointSource>{source};
                                },
                                .stressGluts = {}});
  const auto fieldSize = grid.FieldSize();
  const auto check = [&](const char* name, std::size_t n,
                         const std::vector<double>& want, double factor) {
    const auto& got = file.variables.at(name).values;
    auto worst = 0.0, size = 0.0;
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      worst = std::max(worst, std::abs(got[n * fieldSize + i] - factor * want[i]));
      size = std::max(size, std::abs(factor * want[i]));
    }
    EXPECT_GT(size, 0.0) << name;
    EXPECT_LT(worst, 1e-9 * size) << name << " node " << n;
  };
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    check("displacement", n, snapshots[n].displacement, units.Length());
    check("tangential_potential", n, snapshots[n].tangentialPotential,
          units.Length());
    check("toroidal_potential", n, snapshots[n].toroidalPotential, units.Length());
  }

  // an onset at the second node: nothing before it
  static_cast<void>(run("late", R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "strike_deg": 20, "dip_deg": 60, "rake_deg": 90,
      "moment_nm": 1e21}, "onset": 0.4})"));
  const auto late = pm::ReadSurfaceFile(dir_ / "out" / "late.nc");
  const auto& lateU = late.variables.at("displacement").values;
  auto before = 0.0, after = 0.0;
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    before = std::max(before, std::abs(lateU[i]));
    after = std::max(after, std::abs(lateU[fieldSize + i]));
  }
  // exactly: the source's factor is zero before its onset
  EXPECT_EQ(before, 0.0);
  EXPECT_GT(after, 0.0);

  // one patch of a fault is the point double couple at its centroid
  static_cast<void>(run("couple", R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "strike_deg": 20, "dip_deg": 60, "rake_deg": 90,
      "moment_nm": 1e21}})"));
  static_cast<void>(run("fault", R"({"fault": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "strike_deg": 20, "dip_deg": 60, "rake_deg": 90,
      "length_km": 50, "width_km": 20, "moment_nm": 1e21}})"));
  const auto couple = pm::ReadSurfaceFile(dir_ / "out" / "couple.nc");
  const auto fault = pm::ReadSurfaceFile(dir_ / "out" / "fault.nc");
  for (const auto name : {"displacement", "toroidal_potential"}) {
    const auto& a = couple.variables.at(name).values;
    const auto& b = fault.variables.at(name).values;
    auto worst = 0.0, size = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      worst = std::max(worst, std::abs(a[i] - b[i]));
      size = std::max(size, std::abs(a[i]));
    }
    EXPECT_LT(worst, 1e-10 * size) << name;
  }

  // the static problem: the co-seismic response, the run's first node (to
  // about 5e-10 here: the branchless systems' moduli at the nodes against
  // the unrelaxed ones' at the memory points, which a point source's
  // sharp forcing feels more than a load does)
  static_cast<void>(app::RunSolve(Write(
      "elastic.json",
      std::format(R"({{"problem": "static", {}, "forcing": {{"seismic": [{}]}},
          "rotation": "none", "output": {{"surface": "out/elastic.nc"}}}})",
                  kBall, tensor))));
  const auto elastic = pm::ReadSurfaceFile(dir_ / "out" / "elastic.nc");
  for (const auto* name :
       {"displacement", "tangential_potential", "toroidal_potential",
        "potential"}) {
    EXPECT_LT(Departure(elastic.variables.at(name).values, Slice(file, name, 0)),
              1e-8)
        << name;
  }

  // and the refusals
  const auto both = Write(
      "both.json",
      std::format(
          R"({{"problem": "quasi_static", {},
          "times": {{"unit": "kyr", "at": [0, 1]}},
          "forcing": {{"seismic": [
          {{"point": {{"latitude": 0, "longitude": 0, "depth_km": 10,
                      "moment_tensor_nm": {{"rr": 1, "tt": 0, "pp": 0, "rt": 0,
                                           "rp": 0, "tp": 0}},
                      "moment_nm": 1}}}}]}},
          "output": {{"surface": "out/both.nc"}}}})",
          kBall));
  ExpectRefused(app::RunSolve, both, "forcing.seismic[0].point.moment_nm",
                "give one");
}

TEST_F(AppsTest, SolveTakesAnIceHistoryAndAnInitialSeaLevel) {
  // The run's inputs are its state at the window's start, the topography
  // and the ice, and the ice change, an ice history as a `thickness` term:
  // a file holding topography alone serves as the sea level's (the ice it
  // is built with is the state's own), and gives the run of the one file
  // named for all; variables under other names, named by `variable`, give
  // the same run to the bit, and the summary and the output's metadata say
  // what was read. A sea level file without the variable (its own
  // variables listed), an ice file without the ice, the old members
  // (`forcing.state`, `forcing.sea_level`) and values that are not numbers
  // are refused.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  using sl3dpp::test::ToyFields;
  WriteToyHistory(dir_ / "topography.nc", 0.0, false, ToyFields::Topography);
  WriteToyHistory(dir_ / "ice.nc", 0.0, false, ToyFields::Ice);
  WriteToyHistory(dir_ / "renamed.nc", 0.0, false, ToyFields::Both, false,
                  {.ice = "thk", .topography = "topo"});
  const auto run = [&](const std::string& name, const std::string& sea,
                       const std::string& ice) {
    return Write(name + ".json",
                 "{" + std::string{kBall} + ", " + Window(5, sea, ice) +
                     R"(, "rotation": "none", "output": {"surface": "out/)" +
                     name + R"(.nc"}})");
  };
  const auto summary = app::RunSolve(run("one", "history.nc", "history.nc"));
  static_cast<void>(app::RunSolve(run("two", "topography.nc", "ice.nc")));
  const auto renamed = Write("renamed.json", "{" + std::string{kBall} + R"(,
      "problem": "quasi_static",
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
      "forcing": {"ice_change": [{"thickness": {"file": "renamed.nc",
                                                "variable": "thk"}}]},
      "ocean": {"at": "start",
                "topography": {"file": "renamed.nc", "variable": "topo",
                               "format": "planetmodel_surface"},
                "ice": {"file": "renamed.nc", "variable": "thk"}},
      "rotation": "none", "output": {"surface": "out/renamed.nc"}})");
  const auto renamedSummary = app::RunSolve(renamed);
  const auto one = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto two = pm::ReadSurfaceFile(dir_ / "out" / "two.nc");
  const auto other = pm::ReadSurfaceFile(dir_ / "out" / "renamed.nc");
  // the same values read from other files, the same transforms: bitwise
  for (const auto name :
       {"relative_sea_level_change", "displacement", "initial_sea_level"}) {
    EXPECT_EQ(one.variables.at(name).values, two.variables.at(name).values) << name;
    EXPECT_EQ(one.variables.at(name).values, other.variables.at(name).values)
        << name;
  }
  const auto history = (dir_ / "history.nc").string();
  EXPECT_NE(summary.find(std::format(
                "; ice {}:ice_thickness (the window's start, 2 kyr BP), "
                "topography {}:topography (the window's start, 2 kyr BP); "
                "ice change thickness {}:ice_thickness, from the state's "
                "instant to each node's date; wrote ",
                history, history, history)),
            std::string::npos)
      << summary;
  const auto file = (dir_ / "renamed.nc").string();
  EXPECT_NE(renamedSummary.find(std::format(
                "; ice {}:thk (the window's start, 2 kyr BP), topography "
                "{}:topo (the window's start, 2 kyr BP); ice change "
                "thickness {}:thk, from the state's instant to each node's "
                "date; wrote ",
                file, file, file)),
            std::string::npos)
      << renamedSummary;
  EXPECT_EQ(std::get<std::string>(other.meta.at("ice")),
            file + ":thk (the window's start, 2 kyr BP)");
  EXPECT_EQ(std::get<std::string>(other.meta.at("sea_level")),
            "topography " + file + ":topo (the window's start, 2 kyr BP)");
  EXPECT_EQ(std::get<std::string>(other.meta.at("ice_change")),
            "thickness " + file +
                ":thk, from the state's instant to each node's date");

  ExpectRefused(app::RunSolve, run("neither", "ice.nc", "ice.nc"),
                "ocean.topography.file",
                "names a variable \"topography\" that " +
                    (dir_ / "ice.nc").string() +
                    " does not hold; it holds \"ice_thickness\"");
  ExpectRefused(app::RunSolve,
                run("noice", "topography.nc", "topography.nc"),
                "ocean.ice.file",
                "names a variable \"ice_thickness\" that " +
                    (dir_ / "topography.nc").string() +
                    " does not hold; it holds \"topography\"");
  for (const auto* member : {"state", "sea_level"}) {
    const auto old = Write("old.json", "{" + std::string{kBall} +
                                           std::format(R"(,
        "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2, "nodes": 5}},
        "forcing": {{"{}": {{"file": "history.nc"}},
                    "ice_change": [{{"thickness": {{"file": "history.nc"}}}}]}},
        "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                  "ice": {{"file": "history.nc"}}}},
        "rotation": "none", "output": {{"surface": "out/old.nc"}}}})",
                                                       member));
    ExpectRefused(app::RunSolve, old, std::string{"forcing."} + member,
                  "is not a member of `forcing`");
    ExpectRefused(app::RunSolve, old, std::string{"forcing."} + member,
                  "`ocean.sea_level`");
  }
  WriteToyHistory(dir_ / "holed.nc", 0.0, false, ToyFields::Ice, true);
  ExpectRefused(app::RunSolve, run("holed", "topography.nc", "holed.nc"),
                "ocean.ice.file", "not a number");
}

TEST_F(AppsTest, SolveFingerprintTakesItsChangeAsTermsThatAdd) {
  // In the static problem the state is the toy history at 2 kyr BP, its
  // ice and topography, and the ice change a sum of terms. A `thickness`
  // term at the present is the history's change from the state's date:
  // the library's Fingerprint of that change on the same state, to
  // rounding (equal to the bit here), and the fingerprint of the same
  // change given as a `change` file (written here as the differences of
  // the history's own values) to the rounding of the resampling. The
  // fingerprint is linear: two terms, the thickness and a cap of ice laid
  // on land, give the sum of their fingerprints taken one by one, to the
  // solves' tolerance, and the change applied, `ice_change`, is the sum of
  // theirs.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyChange(dir_ / "melt.nc", false);
  const auto config = [&](const std::string& name, const std::string& terms) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "static",
      "forcing": {{"ice_change": [{}]}},
      "ocean": {{"date_kyr_bp": 2, "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, terms, name));
  };
  constexpr auto kThickness =
      R"({"thickness": {"file": "history.nc", "date_kyr_bp": 0}})";
  constexpr auto kCap = R"({"cap_thickness": {"latitude": 30, "longitude": 60,
      "width_deg": 10, "thickness_m": 200}})";
  const auto history = (dir_ / "history.nc").string();
  const auto summary = app::RunSolve(config("thickness", kThickness));
  EXPECT_NE(summary.find(std::format(
                "; ice {}:ice_thickness (2 kyr BP), topography {}:topography "
                "(2 kyr BP); ice change thickness {}:ice_thickness, from the "
                "state's date to 0 kyr BP; wrote ",
                history, history, history)),
            std::string::npos)
      << summary;
  static_cast<void>(
      app::RunSolve(config("change", R"({"change": {"file": "melt.nc"}})")));
  static_cast<void>(app::RunSolve(config("cap", kCap)));
  const auto both = app::RunSolve(
      config("both", std::format("{}, {}", kThickness, kCap)));
  EXPECT_NE(both.find(" + cap at latitude 30, longitude 60, width 10 deg, "
                      "200 m of ice; wrote "),
            std::string::npos)
      << both;
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  // A melting cap has no edge, and takes at each point no more than the
  // ice the state holds: caps deep and wide enough to take it all are
  // one change, the loss of the state's ice, whatever their depth.
  {
    for (const auto* depth : {"1e7", "2e7"}) {
      static_cast<void>(app::RunSolve(config(
          std::string{"gone"} + depth,
          std::format(R"({{"cap_thickness": {{"latitude": 30, "longitude": 60,
              "width_deg": 80, "thickness_m": -{}}}}})",
                      depth))));
    }
    const auto once = read("gone1e7");
    const auto twice = read("gone2e7");
    const auto& taken = once.variables.at("ice_change").values;
    // the same change exactly, the same transforms: bitwise
    EXPECT_EQ(taken, twice.variables.at("ice_change").values);
    EXPECT_EQ(once.variables.at("relative_sea_level_change").values,
              twice.variables.at("relative_sea_level_change").values);
    EXPECT_LT(*std::ranges::min_element(taken), -1.0);
    EXPECT_LE(*std::ranges::max_element(taken), 0.0);
  }
  const auto thickness = read("thickness");
  const auto change = read("change");
  const auto cap = read("cap");
  const auto sum = read("both");
  EXPECT_EQ(std::get<std::string>(sum.meta.at("ice_change")),
            std::format("thickness {}:ice_thickness, from the state's date to "
                        "0 kyr BP; cap at latitude 30, longitude 60, width 10 "
                        "deg, 200 m of ice",
                        history));

  // the library's fingerprint of the history's change on the same state
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = 8});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto love = ComputeLoveNumbers(*material, 8);
  Fingerprint fingerprint(8, love);
  const auto file = io::FieldHistory::Read(dir_ / "history.nc",
                                           fingerprint.Grid());
  ASSERT_TRUE(file);
  const auto ice = file->Values("ice_thickness", -2.0 * app::KYR);
  fingerprint.SetState(
      ice, SeaLevelFromTopography(file->Values("topography", -2.0 * app::KYR),
                                  ice));
  auto melt = file->Values("ice_thickness", 0.0);
  for (auto i = std::size_t{0}; i < melt.size(); ++i) melt[i] -= ice[i];
  const auto direct =
      fingerprint.SolveForIceChange(melt, RotationalFeedback::None);
  ASSERT_TRUE(direct.converged);
  EXPECT_LT(Departure(thickness.variables.at("ice_change").values, melt),
            1e-15);
  auto worst = 0.0;
  for (const auto& [name, want] :
       {std::pair{"relative_sea_level_change", &direct.relativeSeaLevelChange},
        std::pair{"displacement", &direct.displacement},
        std::pair{"potential", &direct.potential}}) {
    const auto departure = Departure(thickness.variables.at(name).values, *want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));

  // the change file's fingerprint, and the terms' sum
  for (const auto* name : {"ice_change", "relative_sea_level_change",
                           "displacement", "potential"}) {
    const auto& one = thickness.variables.at(name).values;
    EXPECT_LT(Departure(change.variables.at(name).values, one), 1e-10) << name;
    auto added = one;
    const auto& other = cap.variables.at(name).values;
    for (auto i = std::size_t{0}; i < added.size(); ++i) added[i] += other[i];
    EXPECT_LT(Departure(sum.variables.at(name).values, added),
              std::string_view{name} == "ice_change" ? 1e-15 : 1e-7)
        << name;
  }
}

TEST_F(AppsTest, SolveRunsItsStateUnderTheIceChange) {
  // A run through time starts from its state at the window's start,
  // `ocean.ice` and the topography, whatever the ice change's files hold
  // there: from a state whose ice is the toy history's at the start made
  // a tenth thicker, under the history as a `thickness` term, the command
  // is the library's Evolution run here from that state under the grounded
  // load of the history's change since the start, to rounding (equal to
  // the bit here). A `change` file on the history's own time axis, holding
  // the same change, gives the same run to the rounding of its
  // resampling.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyChange(dir_ / "change.nc", true);
  {
    // the state, without a time axis: the history's topography and its
    // ice at the start, a tenth thicker
    const auto grid = pm::Equiangular(19, 36);
    auto ice = std::vector<double>{};
    auto topography = std::vector<double>{};
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        topography.push_back(3000.0 * std::cos(theta));
        ice.push_back(theta < 0.45 ? 1.1 * 1500.0 *
                                         std::exp(-std::pow(theta / 0.3, 2))
                                   : 0.0);
      }
    }
    auto state = pm::SurfaceFile{};
    state.grid = grid;
    state.source = "sl3dpp tests";
    state.variables.emplace(
        "ice_thickness",
        pm::SurfaceField("ice_thickness", ice, {.dimensions = pm::units::LENGTH}));
    state.variables.emplace(
        "topography", pm::SurfaceField("topography", topography,
                                       {.dimensions = pm::units::LENGTH}));
    pm::WriteSurfaceFile(dir_ / "state.nc", state);
  }
  const auto config = [&](const std::string& name, const std::string& term) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{}]}},
      "ocean": {{"at": "start", "topography": {{"file": "state.nc"}},
                "ice": {{"file": "state.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, term, name));
  };
  const auto summary = app::RunSolve(
      config("thickness", R"({"thickness": {"file": "history.nc"}})"));
  const auto stateFile = (dir_ / "state.nc").string();
  EXPECT_NE(summary.find(std::format(
                "; ice {}:ice_thickness (static, at the window's start), "
                "topography {}:topography (static, at the window's start); "
                "ice change thickness ",
                stateFile, stateFile)),
            std::string::npos)
      << summary;
  const auto changed = app::RunSolve(
      config("change", R"({"change": {"file": "change.nc"}})"));
  EXPECT_NE(changed.find(std::format(
                "; ice change change {}:ice_thickness_change, read by date; ",
                (dir_ / "change.nc").string())),
            std::string::npos)
      << changed;
  const auto byThickness = pm::ReadSurfaceFile(dir_ / "out" / "thickness.nc");
  const auto byChange = pm::ReadSurfaceFile(dir_ / "out" / "change.nc");

  // the library's run from the state under the history's change
  constexpr auto kLmax = 8;
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax);
  const GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                         SeaLevelParameters{});
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto stateIn = io::FieldHistory::Read(dir_ / "state.nc", solver.Grid());
  const auto history =
      io::FieldHistory::Read(dir_ / "history.nc", solver.Grid());
  ASSERT_TRUE(stateIn);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  auto ice = stateIn->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(stateIn->Values("topography", start), ice,
                                    solver.Parameters());
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  const auto ice0 = history->Values("ice_thickness", start);
  auto times = std::vector<double>{};
  for (auto n = 0; n < 5; ++n) {
    times.push_back(2.0 * app::KYR * n / 4 / scales.Time());
  }
  const auto load = [&](double t) {
    const auto thickness =
        history->Values("ice_thickness", start + t * scales.Time());
    auto out = std::vector<double>(thickness.size());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
               (thickness[i] - ice0[i]) / length;
    }
    return out;
  };
  Evolution evolution(op, solver, state, {.rotation = RotationalFeedback::None});
  const auto snapshots = evolution.Run(times, load);
  ASSERT_EQ(snapshots.size(), 5u);
  auto worst = 0.0;
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "potential"}) {
    const auto factor = std::string_view{name} == "potential"
                            ? scales.Factor(PotentialDimensions)
                            : length;
    auto want = std::vector<double>{};
    for (const auto& snapshot : snapshots) {
      EXPECT_TRUE(snapshot.converged);
      const auto& field = std::string_view{name} == "relative_sea_level_change"
                              ? snapshot.relativeSeaLevelChange
                              : (std::string_view{name} == "displacement"
                                     ? snapshot.displacement
                                     : snapshot.potential);
      for (const auto v : field) want.push_back(v * factor);
    }
    const auto departure =
        Departure(byThickness.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
    EXPECT_LT(Departure(byChange.variables.at(name).values, want), 1e-10)
        << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  // the state's ice, a tenth thicker, is the run's own: the initial sea
  // level is built from it
  auto initial = state.seaLevel;
  for (auto& v : initial) v *= length;
  EXPECT_LT(Departure(byThickness.variables.at("initial_sea_level").values,
                      initial),
            1e-15);
}

TEST_F(AppsTest, SolveTakesLoadsAndSeismicSourcesBesideTheIce) {
  // A run through time under surface loads and a seismic source beside
  // its ice: a cap in the water budget with an amplitude history, a disk
  // outside it that is a layer on the solid surface, and a point source,
  // all present from the first node. Under frozen and under migrating
  // shorelines the command is the library's evolution run here under the
  // same forcing, the tangential and toroidal potentials at every node
  // included; the output says what forced it; and the terms that do
  // not state their properties, a calibrated start and checkpoints under
  // sources are refused. Without an ice change the
  // loads alone force the run.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  constexpr auto kLmax = 8;
  const auto cap = std::string{R"({"cap": {"latitude": 60, "longitude": -85,
      "width_deg": 20, "amplitude_kg_m2": 2000},
      "history_kyr_bp": [[2, 0.5], [0, 1.5]],
      "water_budget": true, "moves_solid_surface": false})"};
  const auto disk = std::string{R"({"disk": {"latitude": -20, "longitude": 100,
      "radius_deg": 25, "amplitude_kg_m2": 4600},
      "water_budget": false, "moves_solid_surface": true,
      "density_kg_m3": 2300})"};
  const auto point = std::string{R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "moment_tensor_nm": {"rr": 9e20, "tt": -2e20,
      "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}}})"};
  const auto config = [&](const std::string& name, const std::string& shorelines,
                          const std::string& forcing,
                          const std::string& extra = "",
                          const std::string& at = "start") {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}]{}}},
      "ocean": {{"shorelines": "{}", "at": "{}",
                 "topography": {{"file": "history.nc"}},
                 "ice": {{"file": "history.nc"}}}},
      "rotation": "none",{} "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, forcing, shorelines, at,
                                             extra, name));
  };
  const auto all = std::format(R"(, "load": [{}, {}], "seismic": [{}])", cap,
                               disk, point);

  // the library's pieces
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax);
  const SurfaceGrid grid(kLmax);
  const GRDSolver solver(grid, op.ElasticLoveNumbers(), SeaLevelParameters{});
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto energy =
      scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto times = std::vector<double>{};
  for (auto n = 0; n < 4; ++n) {
    times.push_back(2.0 * app::KYR * n / 3 / scales.Time());
  }
  const auto rad = std::numbers::pi / 180.0;
  const auto capField = grid.Synthesise(grid.Expand(
      app::CapField(grid, GeographicPoint{60.0, -85.0}, 20.0 * rad, 2000.0)));
  const auto diskField = grid.Synthesise(app::DiskCoefficients(
      grid, GeographicPoint{-20.0, 100.0}, 25.0 * rad, 4600.0));
  const auto kyrOf = [&](double t) { return t * scales.Time() / app::KYR; };
  const auto thicknessAt = [&](double t) {
    return history->Values("ice_thickness", start + t * scales.Time());
  };
  const auto site = GeographicPoint{30.0, 40.0};
  const auto source = PointSource{
      .at = {.radius = (solver.Radius() * length - 300e3) / length,
             .colatitude = site.CoLatitude(),
             .longitude = site.Longitude()},
      .moment = {.rr = 9e20 / energy, .tt = -2e20 / energy, .pp = -7e20 / energy,
                 .rt = 3e20 / energy, .rp = -7e20 / energy, .tp = 6e20 / energy}};

  for (const bool migrating : {false, true}) {
    const auto name = std::string{migrating ? "moving" : "frozen"};
    const auto summary = app::RunSolve(config(name, migrating ? "migrating" : "frozen", all));
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
    EXPECT_EQ(std::get<std::string>(file.meta.at("load")),
              "cap at latitude 60, longitude -85, width 20 deg, 2000 kg m^-2, "
              "history_kyr_bp [[2, 0.5], [0, 1.5]], in the water budget; disk at "
              "latitude -20, longitude 100, radius 25 deg, 4600 kg m^-2, "
              "outside the water budget, a layer on the solid surface of "
              "2300 kg m^-3")
        << summary;
    EXPECT_TRUE(file.meta.contains("seismic"));

    auto forcing = Evolution::Forcing{};
    forcing.directLoad = [&, migrating](double t) {
      auto out = std::vector<double>(capField.size(), 0.0);
      const auto factor = 0.5 + 0.5 * kyrOf(t);
      const auto thickness = thicknessAt(t);
      for (auto i = std::size_t{0}; i < out.size(); ++i) {
        if (!migrating) {
          out[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                   (thickness[i] - ice0[i]) / length;
        }
        out[i] += factor * capField[i] / toLoad;
      }
      return out;
    };
    forcing.directLoadOutsideWaterBudget = [&](double) {
      auto out = diskField;
      for (auto& v : out) v /= toLoad;
      return out;
    };
    forcing.solidSurfaceChange = [&](double) {
      auto out = diskField;
      for (auto& v : out) v /= 2300.0 * length;
      return out;
    };
    forcing.sources = [&](double) { return std::vector<PointSource>{source}; };
    Evolution evolution(op, solver, state, {.rotation = RotationalFeedback::None});
    const auto snapshots =
        migrating ? evolution.RunMigrating(
                        times,
                        [&](double t) {
                          auto out = thicknessAt(t);
                          for (auto i = std::size_t{0}; i < out.size(); ++i) {
                            out[i] = state.iceThickness[i] +
                                     (out[i] - ice0[i]) / length;
                          }
                          return out;
                        },
                        forcing)
                  : evolution.Run(times, forcing);
    ASSERT_EQ(snapshots.size(), 4u);
    for (const auto* variable :
         {"relative_sea_level_change", "displacement", "potential"}) {
      const auto which = std::string_view{variable};
      const auto factor = which == "potential"
                              ? scales.Factor(PotentialDimensions)
                              : length;
      auto want = std::vector<double>{};
      for (const auto& snapshot : snapshots) {
        EXPECT_TRUE(snapshot.converged);
        const auto& field = which == "relative_sea_level_change"
                                ? snapshot.relativeSeaLevelChange
                                : (which == "displacement" ? snapshot.displacement
                                                           : snapshot.potential);
        for (const auto v : field) want.push_back(v * factor);
      }
      EXPECT_LT(Departure(file.variables.at(variable).values, want), 1e-9)
          << variable << ", " << name;
    }
    // the potentials of the tangential displacement at every node, the
    // toroidal one in the frame of no net rotation of the surface
    {
      auto tangential = std::vector<double>{};
      auto toroidal = std::vector<double>{};
      for (const auto& snapshot : snapshots) {
        ASSERT_FALSE(snapshot.toroidalPotential.empty());
        for (const auto v : snapshot.tangentialPotential) {
          tangential.push_back(v * length);
        }
        for (const auto v :
             WithoutNetRotation(grid, snapshot.toroidalPotential)) {
          toroidal.push_back(v * length);
        }
      }
      EXPECT_LT(Departure(file.variables.at("tangential_potential").values,
                          tangential),
                1e-9)
          << name;
      EXPECT_LT(
          Departure(file.variables.at("toroidal_potential").values, toroidal),
          1e-9)
          << name;
    }
    // the forcing beside the ice matters to the run
    static_cast<void>(app::RunSolve(
        config(name + "_ice", migrating ? "migrating" : "frozen", "")));
    const auto bare = pm::ReadSurfaceFile(dir_ / "out" / (name + "_ice.nc"));
    EXPECT_GT(Departure(bare.variables.at("displacement").values,
                        file.variables.at("displacement").values),
              1e-3)
        << name;
    EXPECT_FALSE(bare.meta.contains("load"));
  }

  // -- refusals
  const auto loaded = [&](const std::string& name, const std::string& term) {
    return config(name, "frozen", std::format(R"(, "load": [{}])", term));
  };
  ExpectRefused(app::RunSolve,
                loaded("unsaid", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "moves_solid_surface": false})"),
                "forcing.load[0]", "does not say `water_budget`");
  ExpectRefused(app::RunSolve,
                loaded("unmoved", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "water_budget": true})"),
                "forcing.load[0]", "does not say `moves_solid_surface`");
  ExpectRefused(app::RunSolve,
                loaded("thin", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "water_budget": false, "moves_solid_surface": true})"),
                "forcing.load[0]", "gives no `density_kg_m3`");
  ExpectRefused(app::RunSolve,
                loaded("dense", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "water_budget": false, "moves_solid_surface": false,
                    "density_kg_m3": 2300})"),
                "forcing.load[0].density_kg_m3", "this load is none");
  // times are dates here: the run's clock's names are refused, and so is
  // a history that does not run from the oldest date
  ExpectRefused(app::RunSolve,
                loaded("clocked", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "history_kyr": [[0, 0.5], [2, 1.5]],
                    "water_budget": true, "moves_solid_surface": false})"),
                "forcing.load[0].history_kyr", "`history_kyr_bp`");
  ExpectRefused(app::RunSolve,
                loaded("backwards", R"({"cap": {"latitude": 60, "longitude": -85,
                    "width_deg": 20, "amplitude_kg_m2": 2000},
                    "history_kyr_bp": [[0, 0.5], [2, 1.5]],
                    "water_budget": true, "moves_solid_surface": false})"),
                "forcing.load[0].history_kyr_bp[1][0]", "from the oldest");
  ExpectRefused(
      app::RunSolve,
      config("onset", "frozen",
             R"(, "seismic": [{"point": {"latitude": 30, "longitude": 40,
                 "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                 "rake_deg": 90, "moment_nm": 1e21}, "onset_kyr": 1}])"),
      "forcing.seismic[0].onset_kyr", "`onset_kyr_bp`");
  // a source from 1 kyr BP: nothing before it but the ice's run
  {
    static_cast<void>(app::RunSolve(config(
        "late", "frozen",
        R"(, "seismic": [{"point": {"latitude": 30, "longitude": 40,
            "depth_km": 300, "strike_deg": 20, "dip_deg": 60, "rake_deg": 90,
            "moment_nm": 1e21}, "onset_kyr_bp": 1}])")));
    const auto late = pm::ReadSurfaceFile(dir_ / "out" / "late.nc");
    const auto bare = pm::ReadSurfaceFile(dir_ / "out" / "frozen_ice.nc");
    EXPECT_NE(std::get<std::string>(late.meta.at("seismic")).find("from 1 kyr BP"),
              std::string::npos);
    const auto& got = late.variables.at("displacement").values;
    const auto& ice = bare.variables.at("displacement").values;
    const auto fieldSize = grid.FieldSize();
    // nodes at 2, 4/3, 2/3 and 0 kyr BP, the onset between the second and
    // the third (a pair of the library's own there): the source acts at the
    // last two, the run before them the ice's, to rounding
    auto before = 0.0, after = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < 2 * fieldSize; ++i) {
      before = std::max(before, std::abs(got[i] - ice[i]));
      scale = std::max(scale, std::abs(ice[i]));
    }
    for (auto i = 2 * fieldSize; i < 4 * fieldSize; ++i) {
      after = std::max(after, std::abs(got[i] - ice[i]));
    }
    EXPECT_LT(before, 1e-12 * scale);
    EXPECT_GT(after, 1e-6);
  }
  ExpectRefused(app::RunSolve,
                config("calibrated", "frozen", all, "", "end"), "forcing.load",
                "a calibrated initial sea level does not take");
  ExpectRefused(app::RunSolve,
                config("kept", "frozen", all,
                       R"( "checkpoint": {"directory": "ckpt", "every": 1},)"),
                "forcing.seismic", "is not checkpointed");
  ExpectRefused(
      app::ReadKernelInput,
      Write("kernel.json", "{" + std::string{kBall} + std::format(R"(,
        "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4}},
        "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
                    "load": [{}]}},
        "ocean": {{"at": "end", "topography": {{"file": "history.nc"}},
                  "ice": {{"file": "history.nc"}}}},
        "kernels": {{"measurements": [{{"type": "rotation", "component": "z"}}],
                    "output": "k"}}}})",
                                                                  cap)),
      "forcing.load", "a calibrated initial sea level does not take");
  // without an ice change, the loads and sources alone force the run
  {
    const auto loadsAlone = Write("alone.json", "{" + std::string{kBall} +
                                                    std::format(R"(,
        "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4}},
        "forcing": {{"load": [{}]}},
        "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                  "ice": {{"file": "history.nc"}}}},
        "rotation": "none", "output": {{"surface": "out/alone.nc"}}}})",
                                                                cap));
    const auto read = app::ReadSolveInput(loadsAlone);
    EXPECT_TRUE(read.run.forcing.ice_change.empty());
    EXPECT_EQ(read.run.forcing.load.size(), 1u);
    // under frozen shorelines the run is linear: the cap alone is the run
    // under the ice and the cap less the run under the ice alone, to the
    // solves' tolerance on the ice's response (some hundred times the
    // cap's)
    const auto summary = app::RunSolve(loadsAlone);
    EXPECT_EQ(summary.find("ice change"), std::string::npos) << summary;
    static_cast<void>(app::RunSolve(
        config("ice_cap", "frozen", std::format(R"(, "load": [{}])", cap))));
    const auto alone = pm::ReadSurfaceFile(dir_ / "out" / "alone.nc");
    EXPECT_FALSE(alone.meta.contains("ice_change"));
    const auto both = pm::ReadSurfaceFile(dir_ / "out" / "ice_cap.nc");
    const auto bare = pm::ReadSurfaceFile(dir_ / "out" / "frozen_ice.nc");
    for (const auto* variable : {"relative_sea_level_change", "displacement"}) {
      auto want = both.variables.at(variable).values;
      const auto& ice = bare.variables.at(variable).values;
      for (auto i = std::size_t{0}; i < want.size(); ++i) want[i] -= ice[i];
      EXPECT_LT(Departure(alone.variables.at(variable).values, want), 1e-5)
          << variable;
    }
  }
}

/// The library's radial pieces of the static problem on the test ball at
/// lmax 8: the deck's elastic structure with no relaxing branch, its
/// operator and the sea level solver on its elastic Love numbers.
struct ElasticBall {
  EarthModel model;
  std::unique_ptr<ViscoelasticOperator> op;
  std::unique_ptr<GRDSolver> solver;

  static ElasticBall Make(const fs::path& deck) {
    auto out = ElasticBall{ModelOf(deck, app::ModelFormat::Gia), {}, {}};
    const auto mesh = RadialMesh::Create(out.model, {.lmax = 8});
    const auto material = Material::Create(out.model, mesh.value());
    const auto elastic = Rheology::Create(material.value(), {});
    out.op = std::make_unique<ViscoelasticOperator>(elastic.value(), 8);
    out.solver = std::make_unique<GRDSolver>(
        SurfaceGrid(8), out.op->ElasticLoveNumbers(), SeaLevelParameters{});
    return out;
  }
};

TEST_F(AppsTest, SolveStaticIsTheOneNodeRunUnderAnyForcing) {
  // The static problem under an ice change beside a load in the water
  // budget, a layer on the solid surface outside it, a tide and a point
  // source, on the marine toy state at 2 kyr BP. The ice change is the history's
  // to the present (a `thickness` term) and a cap of ice laid on the
  // southern ocean, thick enough to ground there. Under frozen shorelines
  // the command is the library's one-node Evolution::Run on a branchless
  // rheology under the same forcing, built here, the cap on the ocean
  // loading nothing; under migrating ones it is the one-node RunMigrating,
  // the ice the state's plus the change, the grounded cap moving the
  // shorelines: the final ocean function written is the library's and not
  // the state's. Both are the library's to rounding (about 3e-16 here).
  // The loads applied are written, and the metadata and the
  // summary say what forced the run.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  constexpr auto kCap = R"({"cap": {"latitude": 60, "longitude": -85,
      "width_deg": 20, "amplitude_kg_m2": 2000},
      "water_budget": true, "moves_solid_surface": false})";
  constexpr auto kDisk = R"({"disk": {"latitude": -20, "longitude": 100,
      "radius_deg": 25, "amplitude_kg_m2": 4600},
      "water_budget": false, "moves_solid_surface": true,
      "density_kg_m3": 2300})";
  constexpr auto kPoint = R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "moment_tensor_nm": {"rr": 9e20, "tt": -2e20,
      "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}}})";
  constexpr auto kStaticTide = R"({"degree": 3, "order": 2,
      "amplitude_m2_s2": 40, "imaginary_m2_s2": -25})";
  const auto config = [&](const std::string& name,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "static",
      "forcing": {{"ice_change": [
                     {{"thickness": {{"file": "history.nc", "date_kyr_bp": 0}}}},
                     {{"cap_thickness": {{"latitude": -60, "longitude": 150,
                       "width_deg": 20, "thickness_m": 5000}}}}],
                  "load": [{}, {}], "tide": [{}], "seismic": [{}]}},
      "ocean": {{"date_kyr_bp": 2, "shorelines": "{}",
                 "topography": {{"file": "history.nc"}},
                 "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, kCap, kDisk, kStaticTide,
                                             kPoint, shorelines, name));
  };

  // the library's pieces
  const auto ball = ElasticBall::Make(dir_ / "ball.deck");
  const auto& op = *ball.op;
  const auto& solver = *ball.solver;
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto energy =
      scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  const auto rad = std::numbers::pi / 180.0;
  // the change: the history's to the present and the cap on the ocean
  auto change = history->Values("ice_thickness", 0.0);
  const auto grounding =
      app::CapField(grid, GeographicPoint{-60.0, 150.0}, 20.0 * rad, 1.0);
  for (auto i = std::size_t{0}; i < change.size(); ++i) {
    change[i] -= ice0[i];
    change[i] += 5000.0 * grounding[i];
  }
  const auto capField = grid.Synthesise(grid.Expand(
      app::CapField(grid, GeographicPoint{60.0, -85.0}, 20.0 * rad, 2000.0)));
  const auto diskField = grid.Synthesise(app::DiskCoefficients(
      grid, GeographicPoint{-20.0, 100.0}, 25.0 * rad, 4600.0));
  const auto site = GeographicPoint{30.0, 40.0};
  const auto source = PointSource{
      .at = {.radius = (solver.Radius() * length - 300e3) / length,
             .colatitude = site.CoLatitude(),
             .longitude = site.Longitude()},
      .moment = {.rr = 9e20 / energy, .tt = -2e20 / energy, .pp = -7e20 / energy,
                 .rt = 3e20 / energy, .rp = -7e20 / energy, .tp = 6e20 / energy}};

  for (const bool migrating : {false, true}) {
    const auto name = std::string{migrating ? "moving" : "frozen"};
    const auto summary =
        app::RunSolve(config(name, migrating ? "migrating" : "frozen"));
    EXPECT_TRUE(summary.starts_with(
        migrating
            ? "1-D, elastic, static, shorelines migrating, "
              "rotation none; ice "
            : "1-D, elastic, static, shorelines frozen, rotation none; ice "))
        << summary;
    EXPECT_NE(summary.find("+ cap at latitude -60, longitude 150, width 20 "
                           "deg, 5000 m of ice; load (2 terms) + tide (1 "
                           "coefficient) + seismic (1 source); wrote "),
              std::string::npos)
        << summary;
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
    EXPECT_EQ(std::get<std::string>(file.meta.at("shorelines")),
              migrating ? "migrating" : "frozen");
    EXPECT_EQ(std::get<std::string>(file.meta.at("load")),
              "cap at latitude 60, longitude -85, width 20 deg, 2000 kg m^-2, "
              "in the water budget; disk at latitude -20, longitude 100, "
              "radius 25 deg, 4600 kg m^-2, outside the water budget, a layer "
              "on the solid surface of 2300 kg m^-3");
    EXPECT_NE(std::get<std::string>(file.meta.at("seismic")).find("point source"),
              std::string::npos);

    auto forcing = Evolution::Forcing{};
    forcing.directLoad = [&, migrating](double) {
      auto out = std::vector<double>(capField.size(), 0.0);
      for (auto i = std::size_t{0}; i < out.size(); ++i) {
        if (!migrating) {
          out[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                   change[i] / length;
        }
        out[i] += capField[i] / toLoad;
      }
      return out;
    };
    forcing.directLoadOutsideWaterBudget = [&](double) {
      auto out = diskField;
      for (auto& v : out) v /= toLoad;
      return out;
    };
    forcing.solidSurfaceChange = [&](double) {
      auto out = diskField;
      for (auto& v : out) v /= 2300.0 * length;
      return out;
    };
    forcing.tide = [&](double) {
      auto out = grid.ZeroCoefficients();
      out[SurfaceGrid::Index(3, 2)] =
          Complex{40.0, -25.0} / scales.Factor(PotentialDimensions);
      return out;
    };
    forcing.sources = [&](double) { return std::vector<PointSource>{source}; };
    Evolution evolution(op, solver, state, {.rotation = RotationalFeedback::None});
    const auto one = std::vector<double>{0.0};
    const auto snapshots =
        migrating ? evolution.RunMigrating(
                        one,
                        [&](double) {
                          auto out = state.iceThickness;
                          for (auto i = std::size_t{0}; i < out.size(); ++i) {
                            out[i] += change[i] / length;
                          }
                          return out;
                        },
                        forcing)
                  : evolution.Run(one, forcing);
    ASSERT_EQ(snapshots.size(), 1u);
    const auto& node = snapshots.front();
    ASSERT_TRUE(node.converged);
    ASSERT_FALSE(node.toroidalPotential.empty());
    auto toroidal = WithoutNetRotation(grid, node.toroidalPotential);
    auto worst = 0.0;
    for (const auto& [variable, field, factor] :
         {std::tuple{"relative_sea_level_change", &node.relativeSeaLevelChange,
                     length},
          std::tuple{"displacement", &node.displacement, length},
          std::tuple{"tangential_potential", &node.tangentialPotential, length},
          std::tuple{"toroidal_potential",
                     static_cast<const std::vector<double>*>(&toroidal), length},
          std::tuple{"potential", &node.potential,
                     scales.Factor(PotentialDimensions)}}) {
      auto want = *field;
      for (auto& v : want) v *= factor;
      const auto departure = Departure(file.variables.at(variable).values, want);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-12) << variable << ", " << name;
    }
    RecordProperty(name + "_library_departure", std::format("{:.3e}", worst));
    // the change and the loads applied
    EXPECT_LT(Departure(file.variables.at("ice_change").values, change), 1e-15);
    EXPECT_LT(
        Departure(file.variables.at("load_in_water_budget").values, capField),
        1e-15);
    EXPECT_LT(Departure(file.variables.at("load_outside_water_budget").values,
                        diskField),
              1e-15);
    auto layer = diskField;
    for (auto& v : layer) v /= 2300.0;
    EXPECT_LT(Departure(file.variables.at("solid_surface_change").values, layer),
              1e-12);
    // 0 or 1 by flotation: unequal only at a point on the threshold
    const auto& stateOcean = file.variables.at("ocean_function").values;
    EXPECT_EQ(stateOcean, state.oceanFunction);
    if (migrating) {
      const auto& final = file.variables.at("ocean_function_final").values;
      EXPECT_EQ(final, node.oceanFunction);
      EXPECT_NE(final, stateOcean);
    } else {
      EXPECT_FALSE(file.variables.contains("ocean_function_final"));
    }
  }
  // the grounded cap moves the sea level only under migrating shorelines:
  // under frozen ones ice on the ocean loads nothing
  const auto frozen = pm::ReadSurfaceFile(dir_ / "out" / "frozen.nc");
  const auto moving = pm::ReadSurfaceFile(dir_ / "out" / "moving.nc");
  EXPECT_GT(Departure(moving.variables.at("relative_sea_level_change").values,
                      frozen.variables.at("relative_sea_level_change").values),
            1e-3);
}

TEST_F(AppsTest, SolveStaticUnderAFaultAloneConservesTheOceansWater) {
  // The static problem forced by a fault alone, on the toy state with an
  // ocean, under the full rotational feedback: nothing loads the surface,
  // yet the sea level equation is solved, the ocean's water conserved: the
  // integral of the sea level change over the state's ocean vanishes to
  // the solve's tolerance (about 1e-12 of the integral of its magnitude
  // here), while the change itself does not. The toroidal displacement is
  // written beside the rest, the ice change is zero, and the summary and
  // metadata name the fault.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = Write("fault.json", std::format(R"({{{},
    "problem": "static",
    "forcing": {{"seismic": [{{"fault": {{"latitude": -40, "longitude": 150,
        "depth_km": 20, "strike_deg": 30, "dip_deg": 20, "rake_deg": 90,
        "length_km": 400, "width_km": 100, "moment_nm": 1e22,
        "patches_along_strike": 4, "patches_down_dip": 2}}}}]}},
    "ocean": {{"date_kyr_bp": 2, "topography": {{"file": "history.nc"}},
              "ice": {{"file": "history.nc"}}}},
    "rotation": "full", "output": {{"surface": "out/fault.nc"}}}})",
                                                      kBall));
  const auto summary = app::RunSolve(config);
  EXPECT_EQ(summary.find("ice change"), std::string::npos) << summary;
  EXPECT_NE(summary.find("; seismic (1 source); wrote "), std::string::npos)
      << summary;
  const auto file = pm::ReadSurfaceFile(dir_ / "out" / "fault.nc");
  EXPECT_FALSE(file.meta.contains("ice_change"));
  EXPECT_TRUE(std::get<std::string>(file.meta.at("seismic"))
                  .starts_with("fault centred at latitude -40, longitude 150"));
  for (const auto v : file.variables.at("ice_change").values) {
    EXPECT_EQ(v, 0.0);
  }
  EXPECT_TRUE(file.variables.contains("toroidal_potential"));
  const auto& change = file.variables.at("relative_sea_level_change").values;
  const auto& ocean = file.variables.at("ocean_function").values;
  const SurfaceGrid grid(8);
  auto water = std::vector<double>(change.size());
  auto magnitude = std::vector<double>(change.size());
  for (auto i = std::size_t{0}; i < change.size(); ++i) {
    water[i] = ocean[i] * change[i];
    magnitude[i] = ocean[i] * std::abs(change[i]);
  }
  const auto net = grid.Integrate(water);
  const auto scale = grid.Integrate(magnitude);
  RecordProperty("water_departure", std::format("{:.3e}", std::abs(net) / scale));
  EXPECT_GT(scale, 0.0);
  EXPECT_GT(*std::ranges::max_element(change) -
                *std::ranges::min_element(change),
            1e-6);
  EXPECT_LT(std::abs(net), 1e-9 * scale);
  // the source moves the body
  const auto& displacement = file.variables.at("displacement").values;
  EXPECT_GT(*std::ranges::max_element(displacement), 0.0);
}

TEST_F(AppsTest, SolveStaticOnLaterallyUniformModuliIsTheRadialOne) {
  // A moduli file stored laterally but uniform makes the static problem
  // 3-D; under an ice change, loads inside and outside the water budget, a
  // layer, a tide and a point source, under frozen and under migrating
  // shorelines,
  // the 3-D answer is the radial one to the solves' tolerance, the
  // toroidal displacement included and the final ocean function the same,
  // and the radial model's fields beside it are the radial run's to the
  // bit.
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  const auto zero = [](double, double, double) { return 0.0; };
  ASSERT_TRUE(io::WriteModuli(dir_ / "uniform.nc", model, SurfaceGrid(8), zero,
                              zero,
                              {.name = "uniform moduli", .elementMetres = 500e3}));
  const auto config = [&](const std::string& name, const std::string& moduli,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic"{}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc", "date_kyr_bp": 0}}}},
                       {{"cap_thickness": {{"latitude": -60, "longitude": 150,
                         "width_deg": 20, "thickness_m": 5000}}}}],
        "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                   "amplitude_kg_m2": 2000}},
                   "water_budget": true, "moves_solid_surface": false}},
                 {{"disk": {{"latitude": -20, "longitude": 100, "radius_deg": 25,
                   "amplitude_kg_m2": 4600}},
                   "water_budget": false, "moves_solid_surface": true,
                   "density_kg_m3": 2300}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 50,
                   "imaginary_m2_s2": 20}}],
        "seismic": [{{"point": {{"latitude": 30, "longitude": 40,
                      "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                      "rake_deg": 90, "moment_nm": 1e21}}}}]}},
      "ocean": {{"date_kyr_bp": 2, "shorelines": "{}",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full",
      "output": {{"surface": "out/{}.nc"}}}})",
                                             data,
                                             moduli.empty()
                                                 ? std::string{}
                                                 : std::format(
                                                       R"(, "moduli": {{"file": "{}"}})",
                                                       moduli),
                                             shorelines, name));
  };
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto uniformName = std::format("uniform_{}", shorelines);
    const auto radialName = std::format("radial_{}", shorelines);
    const auto summary =
        app::RunSolve(config(uniformName, "uniform.nc", shorelines));
    EXPECT_TRUE(summary.starts_with(std::format(
        "3-D (lateral moduli), elastic, static, shorelines {}", shorelines)))
        << summary;
    static_cast<void>(app::RunSolve(config(radialName, "", shorelines)));
    const auto uniform = pm::ReadSurfaceFile(dir_ / "out" / (uniformName + ".nc"));
    const auto radial = pm::ReadSurfaceFile(dir_ / "out" / (radialName + ".nc"));
    auto worst = 0.0;
    for (const auto* name :
         {"relative_sea_level_change", "displacement", "tangential_potential",
          "toroidal_potential", "potential"}) {
      const auto departure = Departure(uniform.variables.at(name).values,
                                       radial.variables.at(name).values);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-9) << name << ", " << shorelines;
    }
    // the same radial solve and inputs, the same transforms: bitwise
    for (const auto* name :
         {"relative_sea_level_change", "displacement", "potential"}) {
      EXPECT_EQ(uniform.variables.at(std::string{name} + "_radial_model").values,
                radial.variables.at(name).values)
          << name << ", " << shorelines;
    }
    for (const auto* name :
         {"ice_change", "ocean_function", "load_in_water_budget",
          "load_outside_water_budget", "solid_surface_change"}) {
      EXPECT_EQ(uniform.variables.at(name).values,
                radial.variables.at(name).values)
          << name << ", " << shorelines;
    }
    if (std::string_view{shorelines} == "migrating") {
      // 0 or 1 by flotation: unequal only at a point on the threshold
      EXPECT_EQ(uniform.variables.at("ocean_function_final").values,
                radial.variables.at("ocean_function_final").values);
      EXPECT_NE(radial.variables.at("ocean_function_final").values,
                radial.variables.at("ocean_function").values);
    }
    RecordProperty(std::format("radial_departure_{}", shorelines),
                   std::format("{:.3e}", worst));
  }
}

TEST_F(AppsTest, SolveStaticReadsItsTermsAtTheirDates) {
  // In the static problem a `field` load whose file has a time axis is
  // read at the term's `date_kyr_bp`, kyr before present: the load
  // applied is the file's there (between two stored dates, interpolated
  // as the file says), band-limited to the run's degree, and the metadata
  // say so. Refused, naming what the problem takes: a time in kyr on the
  // file's own axis without its unit (`date_kyr`) and a time origin, a
  // field file
  // with a time axis without its date and a date on one without, the
  // histories and the onsets of a run through time (on a load, a cap of
  // ice change and a source), a history on a file term of the ice change,
  // and a field's date in a run through time, which reads it at each
  // node's date.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  {
    // a load on the toy's time axis: a band of land whose mass grows
    // towards the present
    constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
    const auto grid = pm::Equiangular(19, 36);
    auto values = std::vector<double>{};
    auto time = std::vector<double>{};
    for (const auto date : {2.0, 1.0, 0.0}) {
      time.push_back(-date * kyr);
      for (const auto theta : grid.Colatitudes()) {
        for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
          values.push_back((3.0 - date) * 500.0 *
                           std::exp(-std::pow((theta - 0.8) / 0.2, 2)));
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
    options.dimensions = pm::Dimensions{.mass = 1, .length = -2};
    file.variables.emplace("load", pm::SurfaceField("load", values, options));
    pm::WriteSurfaceFile(dir_ / "load.nc", file);
    auto still = pm::SurfaceFile{};
    still.grid = grid;
    still.source = "sl3dpp tests";
    still.variables.emplace(
        "load", pm::SurfaceField("load", std::vector<double>(19 * 36, 100.0),
                                 {.dimensions = options.dimensions}));
    pm::WriteSurfaceFile(dir_ / "still.nc", still);
  }
  const auto config = [&](const std::string& name, const std::string& forcing,
                          const std::string& problem = "static") {
    const auto timed = problem != "static";
    return Write(name + ".json", std::format(R"({{{},
      "problem": "{}",{}
      "forcing": {},
      "ocean": {{{}, "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, problem,
                                             timed ? R"( "times": {"start_kyr_bp": 2, "nodes": 3},)"
                                                   : "",
                                             forcing,
                                             timed ? R"("at": "start")"
                                                   : R"("date_kyr_bp": 2)",
                                             name));
  };
  const auto field = [](const std::string& members, const std::string& beside = "") {
    return std::format(R"({{"load": [{{"field": {{"file": "load.nc"{}}}{},
        "water_budget": true, "moves_solid_surface": false}}]}})",
                       members, beside);
  };
  static_cast<void>(
      app::RunSolve(config("dated", field(R"(, "date_kyr_bp": 1.5)"))));
  const auto dated = pm::ReadSurfaceFile(dir_ / "out" / "dated.nc");
  const SurfaceGrid grid(8);
  const auto file = io::FieldHistory::Read(dir_ / "load.nc", grid);
  ASSERT_TRUE(file);
  const auto want =
      grid.Synthesise(grid.Expand(file->Values("load", -1.5 * app::KYR)));
  EXPECT_LT(Departure(dated.variables.at("load_in_water_budget").values, want),
            1e-15);
  EXPECT_EQ(std::get<std::string>(dated.meta.at("load")),
            std::format("field {}:load, at 1.5 kyr BP, in the water budget",
                        (dir_ / "load.nc").string()));

  const auto refused = [&](const std::string& name, const std::string& forcing,
                           const std::string& member, const std::string& what,
                           const std::string& problem = "static") {
    ExpectRefused(app::RunSolve, config(name, forcing, problem), member,
                  what);
  };
  refused("own_axis", field(R"(, "date_kyr": 1.5)"),
          "forcing.load[0].field.date_kyr",
          "give `date` with its `unit` (\"unit\": \"kyr\"), or the date "
          "before present, `date_kyr_bp`");
  refused("own_axis_timed", field(R"(, "date_kyr": 1.5)"),
          "forcing.load[0].field.date_kyr",
          "a run through time reads it at each node's date", "quasi_static");
  refused("origin", field(R"(, "time_origin_kyr": 1.5)"),
          "forcing.load[0].field.time_origin_kyr",
          "is read at its date, `date_kyr_bp`");
  refused("undated", field(""), "forcing.load[0].field",
          "give the date it is read at, `date_kyr_bp`");
  ExpectRefused(app::RunSolve,
                config("still", R"({"load": [{"field": {"file": "still.nc",
                    "date_kyr_bp": 1}, "water_budget": false,
                    "moves_solid_surface": false}]})"),
                "forcing.load[0].field.date_kyr_bp",
                "dates a file without a time axis");
  refused("timed_date", field(R"(, "date_kyr_bp": 1.5)"),
          "forcing.load[0].field.date_kyr_bp", "reads it at each node's date",
          "quasi_static");
  refused("load_history", field("", R"(, "history_kyr_bp": [[2, 0], [0, 1]])"),
          "forcing.load[0].history_kyr_bp",
          "is an amplitude history, which needs the quasi-static problem");
  refused("load_clock", field("", R"(, "history_kyr": [[0, 0], [2, 1]])"),
          "forcing.load[0].history_kyr", "`history_kyr_bp`");
  refused("cap_history",
          R"({"ice_change": [{"cap_thickness": {"latitude": 70,
              "longitude": 0, "width_deg": 10, "thickness_m": 1},
              "history_kyr_bp": [[2, 0], [0, 1]]}]})",
          "forcing.ice_change[0].history_kyr_bp",
          "is an amplitude history, which needs the quasi-static problem");
  refused("file_history",
          R"({"ice_change": [{"change": {"file": "load.nc",
              "variable": "load"}, "history_kyr_bp": [[2, 0], [0, 1]]}]})",
          "forcing.ice_change[0].history_kyr_bp",
          "which a cap of ice change alone takes", "quasi_static");
  constexpr auto kPoint = R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "strike_deg": 20, "dip_deg": 60, "rake_deg": 90,
      "moment_nm": 1e21})";
  refused("onset", std::format(R"({{"seismic": [{}, "onset_kyr_bp": 1}}]}})", kPoint),
          "forcing.seismic[0].onset_kyr_bp",
          "is an onset, which needs the quasi-static problem");
  refused("source_history",
          std::format(R"({{"seismic": [{}, "history_kyr_bp": [[2, 1]]}}]}})", kPoint),
          "forcing.seismic[0].history_kyr_bp",
          "is an amplitude history, which needs the quasi-static problem");
  refused("source_clock",
          std::format(R"({{"seismic": [{}, "onset_kyr": 1}}]}})", kPoint),
          "forcing.seismic[0].onset_kyr", "`onset_kyr_bp`");
}

TEST_F(AppsTest, SolveTakesCapsAndAChangeAtTheStartThroughTime) {
  // In a run through time a cap of ice change takes an amplitude history:
  // a fifth of the polar ice melted over the first half of the window and
  // held, from a factor of zero at the window's start, is the library's
  // Evolution run here under that cap's grounded load. A cap without a
  // history is held through the run, so it is present at the window's
  // start: under migrating shorelines the run then solves its start, the
  // library's RunMigrating under an empty forcing beside the ice, built
  // here, the cap grounding ice on the ocean there; both are the
  // library's to rounding (equal to the bit here).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = [&](const std::string& name, const std::string& term,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{}]}},
      "ocean": {{"at": "start", "shorelines": "{}",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, term, shorelines, name));
  };
  const auto summary = app::RunSolve(config(
      "melting",
      R"({"cap_fraction": {"latitude": 90, "longitude": 0, "width_deg": 15,
          "fraction": -0.2}, "history_kyr_bp": [[2, 0], [1, 1]]})",
      "frozen"));
  constexpr auto kMelting =
      "cap at latitude 90, longitude 0, width 15 deg, fraction -0.2 of the "
      "state's ice, history_kyr_bp [[2, 0], [1, 1]]";
  EXPECT_NE(summary.find(std::format("; ice change {}; wrote ", kMelting)),
            std::string::npos)
      << summary;
  static_cast<void>(app::RunSolve(config(
      "grounding",
      R"({"cap_thickness": {"latitude": -60, "longitude": 150,
          "width_deg": 20, "thickness_m": 5000}})",
      "migrating")));
  const auto melting = pm::ReadSurfaceFile(dir_ / "out" / "melting.nc");
  const auto grounding = pm::ReadSurfaceFile(dir_ / "out" / "grounding.nc");
  EXPECT_EQ(std::get<std::string>(melting.meta.at("ice_change")), kMelting);

  // the library's runs from the same state
  constexpr auto kLmax = 8;
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax);
  const GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                         SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto times = std::vector<double>{};
  for (auto n = 0; n < 5; ++n) {
    times.push_back(2.0 * app::KYR * n / 4 / scales.Time());
  }
  const auto rad = std::numbers::pi / 180.0;
  // the history's factor at a time of the run's clock, as the command
  // reads it: 0 at the start, rising to 1 at 1 kyr BP, held
  const auto factorAt = [&](double t) {
    const auto kyr = 2.0 + (start + t * scales.Time()) / app::KYR;
    return std::clamp(kyr, 0.0, 1.0);
  };
  const auto polar = app::CapField(grid, {90.0, 0.0}, 15.0 * rad, 1.0);
  const auto load = [&](double t) {
    const auto factor = factorAt(t);
    auto out = std::vector<double>(polar.size());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      const auto change = -0.2 * polar[i] * ice0[i] * factor;
      out[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) * change /
               length;
    }
    return out;
  };
  const auto onOcean = app::CapField(grid, {-60.0, 150.0}, 20.0 * rad, 1.0);
  const auto thickness = [&](double) {
    auto out = state.iceThickness;
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] += std::max(5000.0 * onOcean[i], -std::max(ice0[i], 0.0)) / length;
    }
    return out;
  };
  Evolution evolution(op, solver, state, {.rotation = RotationalFeedback::None});
  const auto frozen = evolution.Run(times, load);
  const auto moving =
      evolution.RunMigrating(times, thickness, Evolution::Forcing{});
  for (const auto& [file, snapshots, name] :
       {std::tuple{&melting, &frozen, "melting"},
        std::tuple{&grounding, &moving, "grounding"}}) {
    ASSERT_EQ(snapshots->size(), 5u);
    auto worst = 0.0;
    for (const auto* variable :
         {"relative_sea_level_change", "displacement", "potential"}) {
      const auto which = std::string_view{variable};
      const auto factor = which == "potential"
                              ? scales.Factor(PotentialDimensions)
                              : length;
      auto want = std::vector<double>{};
      for (const auto& snapshot : *snapshots) {
        EXPECT_TRUE(snapshot.converged);
        const auto& field = which == "relative_sea_level_change"
                                ? snapshot.relativeSeaLevelChange
                                : (which == "displacement" ? snapshot.displacement
                                                           : snapshot.potential);
        for (const auto v : field) want.push_back(v * factor);
      }
      const auto departure = Departure(file->variables.at(variable).values, want);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-12) << variable << ", " << name;
    }
    RecordProperty(std::string{name} + "_library_departure",
                   std::format("{:.3e}", worst));
  }
  // the melting cap's first node is undisturbed; the grounding cap's start
  // is solved, its shorelines already moved there
  // exactly zero: nothing loads the melting cap's first node
  for (const auto v : Slice(melting, "relative_sea_level_change", 0)) {
    EXPECT_EQ(v, 0.0);
  }
  const auto first = Slice(grounding, "ocean_function", 0);
  // 0 or 1 by flotation: unequal only at a point on the threshold
  EXPECT_EQ(first, moving.front().oceanFunction);
  EXPECT_NE(first, state.oceanFunction);
}

TEST_F(AppsTest, KernelsTakeAChangeAtTheStartWithTheStateHeld) {
  // An ice change present at the window's start, a cap without a history
  // beside the ice history, under either shorelines: the kernel's ice is
  // the state's plus the change, the state held, so its first node's
  // thickness kernel is the derivative with respect to the change there.
  // Checked by solve runs in which a cap of 10 m on land is added at the
  // first node alone (its history 1 there and 0 from the next node on)
  // and the state is not moved: the run is linear in the ice under frozen
  // shorelines, and under migrating ones while no cell flips, so the
  // difference is the kernel's prediction. The reading with the ice the
  // run starts from moved with the first node's (the thickness kernel
  // plus its baseline) is another derivative, and does not match.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kHeld = R"({"cap_thickness": {"latitude": 50, "longitude": 0,
      "width_deg": 8, "thickness_m": 200}})";
  constexpr auto kFirst = R"({"cap_thickness": {"latitude": 45, "longitude": 30,
      "width_deg": 6, "thickness_m": 10}, "history_kyr_bp": [[2, 1], [1.5, 0]]})";
  const auto config = [&](const std::string& name, const std::string& terms,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}},
                                  {}]}},
      "ocean": {{"at": "start", "shorelines": "{}",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete", "parameters": ["ice"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": 10, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, terms, shorelines, name,
                                             name));
  };
  const auto rad = std::numbers::pi / 180.0;
  const auto grid = SurfaceGrid(8);
  const auto bump = app::CapField(grid, {45.0, 30.0}, 6.0 * rad, 10.0);
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto label = std::string{shorelines};
    const auto base = config(label, kHeld, shorelines);
    const auto summary = app::RunKernel(base);
    EXPECT_NE(summary.find("wrote 2 measurement(s) of kernels"),
              std::string::npos)
        << summary;
    static_cast<void>(app::RunSolve(base));
    static_cast<void>(app::RunSolve(config(
        label + "_bumped", std::string{kHeld} + ", " + kFirst, shorelines)));
    const auto plain = pm::ReadSurfaceFile(dir_ / "out" / (label + ".nc"));
    const auto bumped =
        pm::ReadSurfaceFile(dir_ / "out" / (label + "_bumped.nc"));
    if (label == "migrating") {
      EXPECT_EQ(plain.variables.at("ocean_function").values,
                bumped.variables.at("ocean_function").values)
          << "a cell flipped";
    }
    // the bump at the first node alone; the nodes are 2, 1.5, 1, 0.5, 0
    auto deltas = std::vector<std::vector<double>>(5, grid.ZeroField());
    deltas[0] = bump;
    const auto out = dir_ / "out" / (label + "_kernels");
    for (const auto& [name, measured] :
         {std::pair{std::string{"site"},
                    CapMeanOf(bumped, "relative_sea_level_change", 4,
                              {-30.0, 10.0}, 25.0) -
                        CapMeanOf(plain, "relative_sea_level_change", 4,
                                  {-30.0, 10.0}, 25.0)},
          std::pair{std::string{"uplift"},
                    PointValueOf(bumped, "displacement", 2, {55.0, 10.0}) -
                        PointValueOf(plain, "displacement", 2,
                                     {55.0, 10.0})}}) {
      const auto kernels = pm::ReadSurfaceFile(out / (name + "_ice_kernels.nc"));
      EXPECT_EQ(std::get<std::string>(kernels.meta.at("ice_views")),
                "state_held");
      const auto held = NodePairing(kernels, "thickness_dual", deltas);
      const auto moving =
          held + SurfacePairing(
                     kernels,
                     kernels.variables.at("thickness_baseline_dual").values,
                     bump);
      std::cout << std::format(
          "[change at the start, {}, {}] held-state kernel {:.3e} from "
          "solve's difference; with the run's start moved {:.3e}\n",
          label, name, Gap(held, measured), Gap(moving, measured));
      RecordProperty(label + "_" + name + "_held_gap",
                     std::format("{:.3e}", Gap(held, measured)));
      EXPECT_LT(Gap(held, measured), 1e-6) << label << " " << name;
      EXPECT_GT(Gap(moving, measured), 1e-3) << label << " " << name;
    }
  }
}

/// A field on the toy history's grid at the dates of the toy window's five
/// nodes (2, 1.5, 1, 0.5 and 0 kyr BP; WriteToyHistory's grid), as a file
/// of one variable `name` on that time axis, linear between its nodes:
/// `value(k, theta)` at the k-th date and colatitude theta (radians).
void WriteToyNodes(const fs::path& path, const std::string& name,
                   const std::function<double(std::size_t, double)>& value) {
  namespace pm = planetmodel;
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::array{2.0, 1.5, 1.0, 0.5, 0.0};
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  file.time = std::vector<double>{};
  auto values = std::vector<double>{};
  for (auto k = std::size_t{0}; k < dates.size(); ++k) {
    file.time->push_back(-dates[k] * app::KYR);
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        values.push_back(value(k, theta));
      }
    }
  }
  auto options = pm::SurfaceFieldOptions{};
  options.support = "node";
  options.time_interpolation = "linear";
  options.dimensions = pm::units::LENGTH;
  file.variables.emplace(name, pm::SurfaceField(name, values, options));
  pm::WriteSurfaceFile(path, file);
}

TEST_F(AppsTest, SolveHoldsALandThatChangesInTime) {
  // `ocean.land` on a time axis is read at every node's date under
  // migrating shorelines: a band of the toy world's southern ocean held as
  // land at the first two nodes and a narrower one after (`shrinking`)
  // gives, at each node, an ocean function that holds that node's land,
  // the band's edges released to the ocean by flotation from the third
  // node, 1-D and 3-D, from the state at the window's start and through a
  // calibration (the state at its end). A land on a time axis that does
  // not change (`steady`) gives the run of the same land without one, to
  // the bit. Under frozen shorelines the land is the state's instant's
  // alone, the window's start or its end, each the run of that instant's
  // land without a time axis, to the bit. The kernels about a migrating
  // run under the changing land are its derivatives: a cap of ice at the
  // first node alone, which flips no cell, moves the measurements by the
  // kernels' prediction (the run being linear in it there).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto wide = [](double theta) { return theta > 1.9 && theta < 2.5; };
  const auto narrow = [](double theta) {
    return theta > 2.05 && theta < 2.35;
  };
  WriteToyNodes(dir_ / "shrinking.nc", "land",
                [&](std::size_t k, double theta) {
                  return (k <= 1 ? wide(theta) : narrow(theta)) ? 1.0 : 0.0;
                });
  WriteToyNodes(dir_ / "steady.nc", "land", [&](std::size_t, double theta) {
    return wide(theta) ? 1.0 : 0.0;
  });
  // the two masks without a time axis: the first two nodes' and the rest's
  const auto still = [&](const std::string& name,
                         const std::function<bool(double)>& band) {
    const auto grid = pm::Equiangular(19, 36);
    auto held = std::vector<double>{};
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        held.push_back(band(theta) ? 1.0 : 0.0);
      }
    }
    auto file = pm::SurfaceFile{};
    file.grid = grid;
    file.source = "sl3dpp tests";
    file.variables.emplace("land", pm::SurfaceField("land", held, {}));
    pm::WriteSurfaceFile(dir_ / name, file);
  };
  still("wide.nc", wide);
  still("narrow.nc", narrow);
  // the masks on the run's grid, as the run cuts them
  const auto maskOf = [&](const std::string& name) {
    const auto file = io::FieldHistory::Read(dir_ / name, SurfaceGrid(8));
    EXPECT_TRUE(file) << file.error().message;
    auto out = std::vector<char>{};
    for (const auto v : file->Values("land", 0.0)) out.push_back(v > 0.5);
    return out;
  };
  const auto early = maskOf("wide.nc");
  const auto late = maskOf("narrow.nc");
  auto released = std::size_t{0};
  for (auto i = std::size_t{0}; i < early.size(); ++i) {
    if (early[i] && !late[i]) ++released;
  }
  ASSERT_GT(released, 0u);

  const auto config = [&](const std::string& name, const std::string& model,
                          int nodes, const std::string& ocean,
                          const std::string& land, const std::string& at) {
    return Write(name + ".json",
                 std::format(R"({{{}, {}, "rotation": "full"{},
                   "output": {{"surface": "out/{}.nc"}}}})",
                             model,
                             Window(nodes, "history.nc", "",
                                    ocean + R"(, "land": {"file": ")" + land +
                                        R"("})",
                                    at),
                             model == kBall ? R"(, "moments": "hydrostatic")"
                                            : "",
                             name));
  };
  // every node's ocean function holds its land: the first two nodes' the
  // wide band's, the rest the narrow one's, the band's edges ocean there
  const auto expectHeld = [&](const std::string& name, std::size_t nodes,
                              std::size_t firstLate) {
    const auto out = Output(name);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      const auto ocean = Slice(out, "ocean_function", n);
      const auto& held = n < firstLate ? early : late;
      auto edges = 0.0;
      for (auto i = std::size_t{0}; i < ocean.size(); ++i) {
        if (held[i]) EXPECT_EQ(ocean[i], 0.0) << name << " node " << n;
        if (early[i] && !late[i]) edges = std::max(edges, ocean[i]);
      }
      EXPECT_EQ(edges, n < firstLate ? 0.0 : 1.0) << name << " node " << n;
    }
  };
  const auto migrating = std::string{R"("shorelines": "migrating")"};
  const auto summary = app::RunSolve(
      config("shrinking", kBall, 5, migrating, "shrinking.nc", "start"));
  EXPECT_NE(summary.find(std::format(", land {}:land (each node's date)",
                                     (dir_ / "shrinking.nc").string())),
            std::string::npos)
      << summary;
  expectHeld("shrinking", 5, 2);
  // (the water the released edges take depends on the initial sea level
  // there, which slows the calibration's fixed point)
  static_cast<void>(app::RunSolve(
      config("calibrated", kBall, 5,
             R"("shorelines": "migrating",
                "calibration": {"rtol": 1e-8, "iterations": 40})",
             "shrinking.nc", "end")));
  expectHeld("calibrated", 5, 2);
  // 3-D, its nodes at 2, 4/3, 2/3 and 0 kyr BP: the land held at 4/3 is
  // two thirds of the way to the wide band, at 2/3 none
  static_cast<void>(app::RunSolve(
      config("shrinking3d", Prem8(), 4, migrating, "shrinking.nc", "start")));
  expectHeld("shrinking3d", 4, 2);

  // a land on a time axis that does not change: the run without one
  const auto same = [&](const std::string& got, const std::string& want) {
    const auto a = Output(got);
    const auto b = Output(want);
    // the same land read, the same transforms in one process: bitwise
    for (const auto* name : {"relative_sea_level_change", "displacement",
                             "potential", "initial_sea_level"}) {
      EXPECT_EQ(a.variables.at(name).values, b.variables.at(name).values)
          << got << " " << name;
    }
    if (b.variables.contains("ocean_function")) {
      EXPECT_EQ(a.variables.at("ocean_function").values,
                b.variables.at("ocean_function").values)
          << got;
    }
  };
  for (const auto& [model, nodes, label] :
       {std::tuple{std::string{kBall}, 5, std::string{"1d"}},
        std::tuple{Prem8(), 4, std::string{"3d"}}}) {
    static_cast<void>(app::RunSolve(config("steady" + label, model, nodes,
                                           migrating, "steady.nc", "start")));
    static_cast<void>(app::RunSolve(
        config("wide" + label, model, nodes, migrating, "wide.nc", "start")));
    same("steady" + label, "wide" + label);
  }
  // frozen shorelines: the state's instant's land alone
  const auto frozen = std::string{R"("shorelines": "frozen")"};
  static_cast<void>(app::RunSolve(
      config("frozen_start", kBall, 5, frozen, "shrinking.nc", "start")));
  static_cast<void>(app::RunSolve(
      config("frozen_wide", kBall, 5, frozen, "wide.nc", "start")));
  same("frozen_start", "frozen_wide");
  static_cast<void>(app::RunSolve(config("frozen_end", kBall, 5,
                                         Calibrated("frozen"), "shrinking.nc",
                                         "end")));
  static_cast<void>(app::RunSolve(config(
      "frozen_narrow", kBall, 5, Calibrated("frozen"), "narrow.nc", "end")));
  same("frozen_end", "frozen_narrow");

  // the kernels about the migrating run under the changing land
  constexpr auto kHeld = R"({"cap_thickness": {"latitude": 50, "longitude": 0,
      "width_deg": 8, "thickness_m": 200}})";
  constexpr auto kFirst = R"({"cap_thickness": {"latitude": 45, "longitude": 30,
      "width_deg": 6, "thickness_m": 10}, "history_kyr_bp": [[2, 1], [1.5, 0]]})";
  const auto kerneled = [&](const std::string& name, const std::string& terms) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}},
                                  {}]}},
      "ocean": {{"at": "start", "shorelines": "migrating",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}},
                "land": {{"file": "shrinking.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete", "parameters": ["ice"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": 10, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, terms, name, name));
  };
  const auto base = kerneled("land_kernels", kHeld);
  static_cast<void>(app::RunKernel(base));
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(
      kerneled("land_bumped", std::string{kHeld} + ", " + kFirst)));
  const auto plain = Output("land_kernels");
  const auto bumped = Output("land_bumped");
  EXPECT_EQ(plain.variables.at("ocean_function").values,
            bumped.variables.at("ocean_function").values)
      << "a cell flipped";
  const auto rad = std::numbers::pi / 180.0;
  const auto grid = SurfaceGrid(8);
  auto deltas = std::vector<std::vector<double>>(5, grid.ZeroField());
  deltas[0] = app::CapField(grid, {45.0, 30.0}, 6.0 * rad, 10.0);
  for (const auto& [name, measured] :
       {std::pair{std::string{"site"},
                  CapMeanOf(bumped, "relative_sea_level_change", 4,
                            {-30.0, 10.0}, 25.0) -
                      CapMeanOf(plain, "relative_sea_level_change", 4,
                                {-30.0, 10.0}, 25.0)},
        std::pair{std::string{"uplift"},
                  PointValueOf(bumped, "displacement", 2, {55.0, 10.0}) -
                      PointValueOf(plain, "displacement", 2, {55.0, 10.0})}}) {
    const auto kernels =
        pm::ReadSurfaceFile(dir_ / "out" / "land_kernels_kernels" /
                            (name + "_ice_kernels.nc"));
    const auto predicted = NodePairing(kernels, "thickness_dual", deltas);
    std::cout << std::format(
        "[a land changing in time, {}] kernel against solve's difference "
        "{:.3e}\n",
        name, Gap(predicted, measured));
    RecordProperty(name + "_gap",
                   std::format("{:.3e}", Gap(predicted, measured)));
    EXPECT_LT(Gap(predicted, measured), 1e-6) << name;
  }
}

TEST_F(AppsTest, KernelsOfACalibratedRunAgainstTheRunsOfItsIce) {
  // A calibrated run (the state at the window's end, `"at": "end"`) of the
  // toy history whose polar ice floats off as it thins, under frozen and
  // under migrating shorelines (the equator's ring of cells flooded by the
  // end, checked), its kernels in the ice: the views `run_start_moves`.
  // Each field is a derivative with the initial sea level HELD, and a
  // solve run re-calibrates it, so the change of J between two solve runs
  // is the thickness kernels paired with the change of the ice plus
  // `initial_sea_level_dual` paired with the change of the calibrated
  // initial sea level (the solve files' `initial_sea_level`). Under frozen
  // shorelines that dual is zero, and the thickness kernels alone are the
  // derivative of the re-calibrated run; under migrating ones it is not
  // (the first node's check below misses by a quarter without it). The
  // runs are linear in the ice and the calibration's target while no cell
  // flips (checked under migrating shorelines), so each one-sided
  // difference is the derivative to the solves' and the calibration's
  // tolerances. Against the base run, solve runs of:
  // - the ice at the first node alone, a 10-m band of grounded ice at
  //   2 kyr BP: `thickness_dual` there, which moves the ice the run starts
  //   from with it (the baseline folded in);
  // - the ice at 1 kyr BP alone: `thickness_dual` there, and `rate_dual`
  //   (the first node's ice held) paired with the intervals' increments;
  // - the state's ice raised by the band with the ice change held: the ice
  //   at every node and the run's start move alike, so the run's loads do
  //   not change (nor J, beyond rounding), and the sum over the nodes of
  //   `thickness_dual` is zero to rounding;
  // - the calibration's target alone, the topography at the end lowered
  //   by a metre about the equator: `initial_sea_level_dual` alone (zero
  //   change under frozen shorelines, and of the uplift at 1 kyr BP,
  //   before any cell floods).
  // `thickness_baseline_dual` (the start's ice with every node's held) is
  // no perturbation a config makes, its run starting from the first
  // node's ice: it is the part of the first node's kernel that moves the
  // start.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  // the toy's marine topography with the equator a metre below sea level
  // (`depth`), so that its flooding by the end is not decided at a sea
  // level of zero
  const auto topography = [](double depth) {
    return [depth](std::size_t, double theta) {
      const auto basin = std::exp(-std::pow(theta / 0.3, 2));
      const auto equator =
          std::exp(-std::pow((theta - std::numbers::pi / 2) / 0.15, 2));
      return 3000.0 * std::cos(theta) - 3500.0 * basin - depth * equator;
    };
  };
  WriteToyNodes(dir_ / "topography.nc", "topography", topography(1.0));
  WriteToyNodes(dir_ / "target.nc", "topography", topography(2.0));
  // the ice at one node alone, the band at that node's date
  constexpr auto kBand = 10.0;
  const auto band = [](double theta) {
    return kBand * std::exp(-std::pow((theta - 0.6) / 0.2, 2));
  };
  for (const auto& [name, node] : {std::pair{"first.nc", std::size_t{0}},
                                   std::pair{"middle.nc", std::size_t{2}}}) {
    WriteToyNodes(dir_ / name, "ice_thickness",
                  [&, node](std::size_t k, double theta) {
                    return k == node ? band(theta) : 0.0;
                  });
  }
  // the state's ice raised by the band at the end (WriteToyHistory's bump
  // is the band above), the history's own ice otherwise
  WriteToyHistory(dir_ / "state.nc", kBand, true, sl3dpp::test::ToyFields::Ice);
  const auto config = [&](const std::string& name,
                          const std::string& shorelines,
                          const std::string& extra, const std::string& ice,
                          const std::string& topography) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}{}]}},
      "ocean": {{"at": "end", "shorelines": "{}",
                "calibration": {{"rtol": 1e-12, "iterations": 60}},
                "topography": {{"file": "{}"}}, "ice": {{"file": "{}"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete", "parameters": ["ice"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": 10, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, extra, shorelines,
                                             topography, ice, name, name));
  };
  const auto term = [](const std::string& file) {
    return std::format(R"(, {{"thickness": {{"file": "{}"}}}})", file);
  };
  // the perturbations on the solver grid, as the runs read them
  const auto grid = SurfaceGrid(8);
  const auto first = io::FieldHistory::Read(dir_ / "first.nc", grid);
  const auto middle = io::FieldHistory::Read(dir_ / "middle.nc", grid);
  const auto state = io::FieldHistory::Read(dir_ / "state.nc", grid);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(first && middle && state && history);
  auto stateBand = state->Values("ice_thickness", 0.0);
  {
    const auto plain = history->Values("ice_thickness", 0.0);
    for (auto i = std::size_t{0}; i < stateBand.size(); ++i) {
      stateBand[i] -= plain[i];
    }
  }
  const auto valueOf = [](const pm::SurfaceFile& file,
                          const std::string& name) {
    return name == "site" ? CapMeanOf(file, "relative_sea_level_change", 4,
                                      {-30.0, 10.0}, 25.0)
                          : PointValueOf(file, "displacement", 2, {55.0, 10.0});
  };
  struct Perturbed {
    std::string name, extra, ice, topography;
  };
  const auto perturbations = std::array{
      Perturbed{"first", term("first.nc"), "history.nc", "topography.nc"},
      Perturbed{"middle", term("middle.nc"), "history.nc", "topography.nc"},
      Perturbed{"state", "", "state.nc", "topography.nc"},
      Perturbed{"target", "", "history.nc", "target.nc"}};
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto label = std::string{shorelines};
    const auto base = config(label, label, "", "history.nc", "topography.nc");
    const auto summary = app::RunKernel(base);
    EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
        << summary;
    static_cast<void>(app::RunSolve(base));
    const auto run = pm::ReadSurfaceFile(dir_ / "out" / (label + ".nc"));
    for (const auto& p : perturbations) {
      static_cast<void>(app::RunSolve(
          config(label + "_" + p.name, label, p.extra, p.ice, p.topography)));
    }
    if (label == "migrating") {
      EXPECT_NE(Slice(run, "ocean_function", 0),
                Slice(run, "ocean_function", 4))
          << "no cell flipped through the run";
    }
    const auto out = dir_ / "out" / (label + "_kernels");
    for (const auto* name : {"site", "uplift"}) {
      const auto measurement = std::string{name};
      const auto value = valueOf(run, measurement);
      EXPECT_LT(Gap(MeasuredValue(out, measurement), value), 1e-10)
          << label << " " << measurement;
      const auto kernels =
          pm::ReadSurfaceFile(out / (measurement + "_ice_kernels.nc"));
      EXPECT_EQ(std::get<std::string>(kernels.meta.at("ice_views")),
                "run_start_moves");
      const auto& sea = kernels.variables.at("initial_sea_level_dual").values;
      if (label == "frozen") {
        EXPECT_TRUE(std::ranges::all_of(sea, [](double v) { return v == 0.0; }))
            << measurement;
      }
      const auto thicknessAt = [&](std::size_t n) {
        return FieldAt(kernels, "thickness_dual", n);
      };
      const auto& rates = kernels.variables.at("rate_dual").values;
      const auto rateAt = [&](std::size_t k) {
        const auto size = grid.FieldSize();
        return std::vector<double>(
            rates.begin() + static_cast<std::ptrdiff_t>(k * size),
            rates.begin() + static_cast<std::ptrdiff_t>((k + 1) * size));
      };
      for (const auto& p : perturbations) {
        const auto moved =
            pm::ReadSurfaceFile(dir_ / "out" / (label + "_" + p.name + ".nc"));
        if (label == "migrating") {
          EXPECT_EQ(moved.variables.at("ocean_function").values,
                    run.variables.at("ocean_function").values)
              << p.name << ": a cell flipped";
        }
        const auto measured = valueOf(moved, measurement) - value;
        // the thickness kernels paired with the ice the run's nodes took
        auto held = 0.0;
        auto rated = std::optional<double>{};
        if (p.name == "first") {
          held =
              SurfacePairing(kernels, thicknessAt(0),
                             first->Values("ice_thickness", -2.0 * app::KYR));
        } else if (p.name == "middle") {
          const auto bump = middle->Values("ice_thickness", -1.0 * app::KYR);
          held = SurfacePairing(kernels, thicknessAt(2), bump);
          // the increments across the intervals: +bump into 1 kyr BP,
          // -bump out of it
          rated = SurfacePairing(kernels, rateAt(1), bump) -
                  SurfacePairing(kernels, rateAt(2), bump);
        } else if (p.name == "state") {
          for (auto n = std::size_t{0}; n < 5; ++n) {
            held += SurfacePairing(kernels, thicknessAt(n), stateBand);
          }
        }
        // and the initial sea level dual with the re-calibration's change
        auto calibrated = moved.variables.at("initial_sea_level").values;
        const auto& before = run.variables.at("initial_sea_level").values;
        for (auto i = std::size_t{0}; i < calibrated.size(); ++i) {
          calibrated[i] -= before[i];
        }
        const auto bySea = SurfacePairing(kernels, sea, calibrated);
        const auto predicted = held + bySea;
        // the gap from the change, a change that vanishes held to 1e-9 of J
        // (the rounding of pairings of J's order that cancel)
        const auto scale = std::max(std::abs(measured), 1e-9 * std::abs(value));
        const auto gapOf = [&](double got) {
          return std::abs(got - measured) / scale;
        };
        std::cout << std::format(
            "[calibrated, {}, {}, {}] change {:.3e} m against J {:.3e} m; "
            "the kernels {:.3e} from it, without the initial sea level's "
            "{:.3e}{}\n",
            label, measurement, p.name, measured, value, gapOf(predicted),
            gapOf(held),
            rated ? std::format("; the rates {:.3e}", gapOf(*rated + bySea))
                  : std::string{});
        RecordProperty(label + "_" + measurement + "_" + p.name + "_gap",
                       std::format("{:.3e}", gapOf(predicted)));
        EXPECT_LT(gapOf(predicted), 1e-6)
            << label << " " << measurement << " " << p.name;
        if (rated) {
          EXPECT_LT(gapOf(*rated + bySea), 1e-6) << label << " " << measurement;
        }
        const auto vanishes = p.name == "state" ||
                              (p.name == "target" &&
                               (label == "frozen" || measurement == "uplift"));
        if (vanishes) {
          EXPECT_LT(std::abs(measured), 1e-12 * std::abs(value))
              << label << " " << measurement << " " << p.name;
        } else {
          EXPECT_GT(std::abs(measured), 1e-6 * std::abs(value))
              << label << " " << measurement << " " << p.name;
        }
      }
    }
  }
}

TEST_F(AppsTest, Solve1DRunsAndARestartedRunMatchesAnUninterruptedOne) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{kBall} + ", " + Window(5) +
                    R"(, "rotation": "full", "moments": "hydrostatic")";
  const auto whole = Write(
      "whole.json", "{" + body + R"(, "output": {"surface": "out/whole.nc"}})");
  auto summary = app::RunSolve(whole);
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 2 to 0 kyr BP, shorelines frozen, rotation "
      "full, moments hydrostatic; ice "))
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  ASSERT_TRUE(reference.time);
  EXPECT_EQ(reference.time->size(), 5u);
  EXPECT_EQ(
      pm::detail::TextOf(reference.variables.at("relative_sea_level_change").attributes,
                         "time_support")
          .value_or(""),
      "node");

  // the same run through wall-time stops: each attempt checkpoints and
  // quits, the reruns resume, and the final output is the reference's
  const auto pieces = Write(
      "pieces.json", "{" + body +
                         R"(, "checkpoint": {"directory": "ckpt", "every": 2,
                              "max_walltime_min": 1e-9},
              "output": {"surface": "out/pieces.nc"}})");
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    summary = app::RunSolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
    EXPECT_NE(summary.find("checkpointed"), std::string::npos) << summary;
  }
  EXPECT_GE(attempts, 1);
  EXPECT_NE(summary.find("resumed"), std::string::npos);
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  ASSERT_TRUE(rebuilt.time);
  // one process, the same transforms, the state read back exactly: bitwise
  EXPECT_EQ(*rebuilt.time, *reference.time);
  for (const auto name : {"relative_sea_level_change", "displacement",
                          "tangential_potential", "potential"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
  // a radial run under the ice alone has no toroidal displacement
  EXPECT_FALSE(reference.variables.contains("toroidal_potential"));
  EXPECT_EQ(rebuilt.meta.at("omega_z_rad_s"), reference.meta.at("omega_z_rad_s"));

  // the checkpoints belong to the run, not to the config's text: the same
  // run with another output, no wall-time limit and other spacing adopts
  // them...
  const auto same = Write("same.json", "{\n" + body +
                                           R"(,
              "checkpoint": {"directory": "ckpt", "every": 2},
              "output": {"surface": "out/same.nc"}
          })");
  summary = app::RunSolve(same);
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  // resumed in one process from the same checkpoints: bitwise
  EXPECT_EQ(pm::ReadSurfaceFile(dir_ / "out" / "same.nc")
                .variables.at("relative_sea_level_change")
                .values,
            reference.variables.at("relative_sea_level_change").values);

  // ...while another computation, or the same config over a changed input
  // file, is refused
  const auto refused = [](const fs::path& config) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::RunSolve(config));
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
                                    R"(, "solver": {"rtol": 1e-11},
                        "checkpoint": {"directory": "ckpt", "every": 2},
                        "output": {"surface": "out/tighter.nc"}})"));
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
              "output": {"surface": "out/over.nc"}})");
  summary = app::RunSolve(over);
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
              "output": {"surface": "out/again.nc"}})");
  summary = app::RunSolve(again);
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;

  // a checkpoint of another format (its magic's version another) is
  // refused, not read; starting over replaces it
  fs::create_directories(dir_ / "older");
  {
    auto out = std::ofstream(dir_ / "older" / "node_000002.ckpt",
                             std::ios::binary);
    const auto magic = std::uint64_t{0x534c3344434b5031};  // "SL3DCKP1"
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  }
  const auto older = Write("older.json", "{" + body + R"(,
              "checkpoint": {"directory": "older", "every": 2},
              "output": {"surface": "out/older.nc"}})");
  EXPECT_THROW(
      {
        try {
          static_cast<void>(app::RunSolve(older));
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string{error.what()}.find(
                        "is a checkpoint of another format"),
                    std::string::npos)
              << error.what();
          throw;
        }
      },
      std::runtime_error);
  const auto fresh = Write("fresh.json", "{" + body + R"(, "restart": false,
              "checkpoint": {"directory": "older", "every": 2},
              "output": {"surface": "out/fresh.nc"}})");
  summary = app::RunSolve(fresh);
  EXPECT_NE(summary.find("replacing 1 earlier checkpoint(s)"),
            std::string::npos)
      << summary;
  EXPECT_TRUE(fs::exists(dir_ / "out" / "fresh.nc"));
}

TEST_F(AppsTest, SolveCalibratesTheInitialSeaLevel) {
  // With the sea level at the window's end ("at": "end") the run's initial
  // sea level is calibrated so that it reaches the state file's sea level
  // there: exactly under frozen shorelines (one linear step), to the
  // calibration's tolerance under migrating ones; a resumed calibrating
  // run reads its calibrated state back instead of calibrating again, and
  // ends where the uninterrupted run does. The calibration's members take
  // their defaults without an `ocean.calibration` block.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  // a calibrated run under the given shorelines
  const auto body = [](const std::string& shorelines) {
    return std::string{kBall} + ", " +
           Window(5, "history.nc", "", Calibrated(shorelines), "end") +
           R"(, "rotation": "full", "moments": "hydrostatic")";
  };

  // the target: the state file's present-day sea level on the solver grid
  const auto grid = SurfaceGrid(8);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto target =
      SeaLevelFromTopography(history->Values("topography", 0.0),
                             history->Values("ice_thickness", 0.0));
  auto worstOf = [&](const pm::SurfaceFile& file) {
    const auto& s0 = file.variables.at("initial_sea_level").values;
    const auto& slc = file.variables.at("relative_sea_level_change").values;
    const auto last = slc.size() - target.size();
    auto worst = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < target.size(); ++i) {
      worst = std::max(worst, std::abs(s0[i] + slc[last + i] - target[i]));
      scale = std::max(scale, std::abs(target[i]));
    }
    return worst / scale;
  };

  const auto frozen =
      Write("frozen.json", "{" + body("frozen") +
                               R"(, "output": {"surface": "out/frozen.nc"}})");
  auto summary = app::RunSolve(frozen);
  EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "frozen.nc");
  EXPECT_LT(worstOf(reference), 1e-9);
  // frozen shorelines hold the state's ocean function: one static field
  EXPECT_EQ(reference.variables.at("ocean_function").values.size(),
            grid.FieldSize());
  // without the calibration block, its defaults
  const auto defaults = Write(
      "defaults.json",
      "{" + std::string{kBall} + ", " +
          Window(5, "history.nc", "", R"("shorelines": "frozen")", "end") +
          R"(, "rotation": "full", "moments": "hydrostatic",
              "output": {"surface": "out/defaults.nc"}})");
  EXPECT_NE(app::RunSolve(defaults).find("initial sea level calibrated"),
            std::string::npos);
  EXPECT_LT(worstOf(pm::ReadSurfaceFile(dir_ / "out" / "defaults.nc")), 1e-9);

  const auto migrating = Write(
      "migrating.json", "{" + body("migrating") +
                            R"(, "output": {"surface": "out/migrating.nc"}})");
  summary = app::RunSolve(migrating);
  EXPECT_NE(summary.find("shorelines migrating, initial sea level calibrated"),
            std::string::npos)
      << summary;
  const auto moving = pm::ReadSurfaceFile(dir_ / "out" / "migrating.nc");
  EXPECT_LT(worstOf(moving), 1e-6);
  EXPECT_TRUE(moving.variables.contains("ocean_function"));

  // through wall-time stops: the calibration runs once, at the start
  const auto pieces = Write("pieces.json", "{" + body("migrating") +
                                               R"(,
              "checkpoint": {"directory": "ckpt", "every": 2,
                             "max_walltime_min": 1e-9},
              "output": {"surface": "out/pieces.nc"}})");
  for (auto attempts = 0; attempts < 8; ++attempts) {
    summary = app::RunSolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
  }
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  EXPECT_TRUE(fs::exists(dir_ / "ckpt" / "initial_state.bin"));
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  EXPECT_NE(std::get<std::string>(rebuilt.meta.at("initial_sea_level"))
                .find("read back"),
            std::string::npos);
  // one process, the same transforms, the state read back exactly: bitwise
  for (const auto name :
       {"relative_sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              moving.variables.at(name).values)
        << name;
  }

  // starting over calibrates afresh, keeping its own state
  const auto over = Write("over.json", "{" + body("migrating") +
                                           R"(, "restart": false,
              "checkpoint": {"directory": "ckpt2", "every": 2},
              "output": {"surface": "out/over.nc"}})");
  static_cast<void>(app::RunSolve(over));
  ASSERT_TRUE(fs::exists(dir_ / "ckpt2" / "initial_state.bin"));
  static_cast<void>(app::RunSolve(over));
  EXPECT_TRUE(fs::exists(dir_ / "ckpt2" / "initial_state.bin"));

  // the kernels about the calibrated migrating run
  const auto kernels = Write("kernels.json", "{" + body("migrating") +
                                                 R"(, "kernels": {
              "method": "discrete",
              "parameters": ["ice", "viscosity"],
              "measurements": [{"type": "relative_sea_level_point", "name": "site",
                                "latitude": -30.0, "longitude": 20.0,
                                "date_kyr_bp": 0.0}],
              "output": "out/kernels"}})");
  summary = app::RunKernel(kernels);
  EXPECT_TRUE(fs::exists(dir_ / "out" / "kernels" / "site_ice_kernels.nc"))
      << summary;

  // a state file that ends before the window's end gives no target
  auto beyond = body("frozen");
  beyond.replace(beyond.find(R"("end_kyr_bp": 0.0)"), 17,
                 R"("end_kyr_bp": -0.5)");
  const auto future = Write("future.json", "{" + beyond +
                                               R"(, "output": {"surface":
                                                  "out/future.nc"}})");
  try {
    static_cast<void>(app::RunSolve(future));
    ADD_FAILURE() << "a target outside the state file was accepted";
  } catch (const planetmodel::Error& error) {
    EXPECT_NE(std::string{error.what()}.find(
                  "ocean.topography.file: spans 2 to 0 kyr BP, which leaves "
                  "out -0.5 kyr BP, the window's end, where the state is "
                  "read, the calibration's target"),
              std::string::npos)
        << error.what();
  }

  // a migrating solve that cannot meet its tolerance is refused, not
  // finished quietly with its last iterate
  const auto strict = Write("strict.json", "{" + body("migrating") +
                                               R"(, "solver": {"rtol": 1e-17},
                                    "output": {"surface": "out/strict.nc"}})");
  try {
    static_cast<void>(app::RunSolve(strict));
    ADD_FAILURE() << "an unconverged migrating run was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("did not converge"),
              std::string::npos)
        << error.what();
  }
}

TEST_F(AppsTest, The3DRunsCalibrateUnderMigratingShorelines) {
  // A 3-D run of solve and the 3-D kernels take migrating shorelines as
  // the 1-D ones do: the calibrated run reaches the state file's present-day
  // sea level and records its ocean functions, through wall-time stops as
  // uninterrupted, and the kernels, by either sweep, come about its own
  // shorelines. The
  // history's polar ice sits in a basin below sea level and floats off as
  // it thins.
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  const auto body =
      Prem8() + ", " +
      Window(4, "history.nc", "", Calibrated("migrating"), "end") + R"(,
    "rotation": "full")";
  const auto grid = SurfaceGrid(8);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto target =
      SeaLevelFromTopography(history->Values("topography", 0.0),
                             history->Values("ice_thickness", 0.0));

  const auto whole = Write(
      "whole.json", "{" + body + R"(, "output": {"surface": "out/whole.nc"}})");
  auto summary = app::RunSolve(whole);
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity), viscoelastic, 4 nodes 2 to 0 kyr BP, "
      "shorelines migrating, initial sea level calibrated, rotation full; "
      "ice "))
      << summary;
  EXPECT_NE(summary.find(":ice_thickness (the window's end, 0 kyr BP), "
                         "topography "),
            std::string::npos)
      << summary;
  EXPECT_NE(summary.find(":topography (the window's end, 0 kyr BP); ice "
                         "change thickness "),
            std::string::npos)
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  EXPECT_EQ(std::get<std::string>(reference.meta.at("shorelines")),
            "migrating");
  ASSERT_TRUE(reference.variables.contains("ocean_function"));
  const auto& s0 = reference.variables.at("initial_sea_level").values;
  const auto& slc = reference.variables.at("relative_sea_level_change").values;
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
              "output": {"surface": "out/pieces.nc"}})");
  for (auto attempts = 0; attempts < 8; ++attempts) {
    summary = app::RunSolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
  }
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  // one process, the same transforms, the state read back exactly: bitwise
  for (const auto name :
       {"relative_sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }

  // by the default (discrete) sweep and by the continuous one
  for (const auto& [name, method] :
       {std::pair{std::string{"kernels"}, std::string{}},
        std::pair{std::string{"continuous_kernels"},
                  std::string{R"("method": "continuous", )"}}}) {
    const auto kernels = Write(name + ".json", "{" + body + R"(, "kernels": {)" +
                                                   method + R"(
              "parameters": ["ice", "viscosity"],
              "measurements": [{"type": "relative_sea_level_point", "name": "site",
                                "latitude": -30.0, "longitude": 20.0,
                                "date_kyr_bp": 0.0}],
              "output": "out/)" + name + R"("}})");
    summary = app::RunKernel(kernels);
    const auto ice =
        pm::ReadSurfaceFile(dir_ / "out" / name / "site_ice_kernels.nc");
    // about moving shorelines the initial sea level kernel is not zero
    const auto& dual = ice.variables.at("initial_sea_level_dual").values;
    EXPECT_TRUE(std::ranges::any_of(dual, [](double v) { return v != 0.0; }))
        << summary;
  }
}

TEST_F(AppsTest, Solve3DRunsAndARestartedRunMatchesAnUninterruptedOne) {
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto body = Prem8() + ", " + Window(4) + R"(, "rotation": "full")";
  const auto whole =
      Write("whole3d.json",
            "{" + body + R"(, "output": {"surface": "out/whole.nc"}})");
  auto summary = app::RunSolve(whole);
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity), viscoelastic, 4 nodes 2 to 0 kyr BP, "
      "shorelines frozen, rotation full; ice "))
      << summary;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  ASSERT_TRUE(reference.time);
  EXPECT_EQ(reference.time->size(), 4u);
  EXPECT_EQ(reference.meta.at("viscosity"),
            pm::netcdf::Attribute{data + "/viscosity_log.nc"});

  const auto pieces =
      Write("pieces3d.json",
            "{" + body +
                R"(, "checkpoint": {"directory": "ckpt3d", "every": 1,
                              "max_walltime_min": 1e-9},
              "output": {"surface": "out/pieces.nc"}})");
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    summary = app::RunSolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
    EXPECT_NE(summary.find("checkpointed"), std::string::npos) << summary;
  }
  EXPECT_GE(attempts, 1);
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  // one process, the same transforms, the state read back exactly: bitwise
  for (const auto name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "toroidal_potential", "potential"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
  // a 3-D run's toroidal displacement: none at the radial start, some
  // after it
  {
    const auto& toroidal = reference.variables.at("toroidal_potential").values;
    const auto size = toroidal.size() / reference.time->size();
    EXPECT_TRUE(std::all_of(toroidal.begin(),
                            toroidal.begin() + static_cast<std::ptrdiff_t>(size),
                            [](double v) { return v == 0.0; }));
    EXPECT_TRUE(std::any_of(toroidal.begin() + static_cast<std::ptrdiff_t>(size),
                            toroidal.end(), [](double v) { return v != 0.0; }));
  }
  EXPECT_EQ(rebuilt.meta.at("omega_z_rad_s"),
            reference.meta.at("omega_z_rad_s"));

  // the same config with a kernels block is a kernel config too
  const auto kernels = app::ReadKernelInput(
      Write("kernel.json", "{" + body + R"(, "kernels": {
              "parameters": ["ice", "viscosity"], "measurements": [
              {"type": "rotation", "component": "z"}],
              "output": "out/kernels"}})"));
  EXPECT_EQ(kernels.kernels.measurements.size(), 1u);
}

TEST_F(AppsTest, SolvePicks1DOr3DFromTheModelBlock) {
  // The model block alone decides between the 1-D and the 3-D run. On the
  // ball, the deck's viscosity column runs 1-D; a viscosity file holding
  // the same 1e21 Pa s as a field of the radius alone is radial and runs
  // 1-D on the file's viscosity, the deck's run to rounding (about 8e-13:
  // the same viscosity by another route); held on a grid, laterally
  // uniform, it runs the 3-D evolution with a zero lateral perturbation,
  // the radial run to the solves' tolerances (1e-9; about 7e-12 here).
  // The 3-D run writes its radial reference model beside it, the file's
  // radial reference stepped by the 1-D run from the same state: the
  // radial file's run to the rounding of the two files' viscosities (about
  // 6e-12 here), its angular velocity too; and each
  // node's closure, its passes, where the 1-D runs write `direct` steps of
  // none. The kernels take the radial file too, the deck's to rounding.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  WriteViscosity(dir_ / "radial.nc", model, 1e21, 0.0, true);
  WriteViscosity(dir_ / "uniform.nc", model, 1e21, 0.0);
  const auto modelOf = [](const std::string& file) {
    return file.empty() ? std::string{R"("model": {"deck": "ball.deck"})"}
                        : std::format(R"("model": {{"deck": "ball.deck",
                              "viscosity": {{"file": "{}"}}}})",
                                      file);
  };
  const auto rest = ", \"discretisation\": {\"lmax\": 8}, " + Window(5) +
                    R"(, "rotation": "full", "moments": "hydrostatic")";
  const auto run = [&](const std::string& name, const std::string& file) {
    return app::RunSolve(
        Write(name + ".json", "{" + modelOf(file) + rest +
                                  std::format(R"(, "output": {{"surface":
                                      "out/{}.nc"}}}})",
                                              name)));
  };
  const auto deckSummary = run("deck", "");
  EXPECT_TRUE(deckSummary.starts_with("1-D, viscoelastic")) << deckSummary;
  const auto radialSummary = run("radial", "radial.nc");
  EXPECT_TRUE(radialSummary.starts_with(
      "1-D (the viscosity file is radial), viscoelastic"))
      << radialSummary;
  const auto uniformSummary = run("uniform", "uniform.nc");
  EXPECT_TRUE(
      uniformSummary.starts_with("3-D (lateral viscosity), viscoelastic"))
      << uniformSummary;
  const auto deck = pm::ReadSurfaceFile(dir_ / "out" / "deck.nc");
  const auto radial = pm::ReadSurfaceFile(dir_ / "out" / "radial.nc");
  const auto uniform = pm::ReadSurfaceFile(dir_ / "out" / "uniform.nc");
  auto worstRadial = 0.0, worstUniform = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    const auto& want = deck.variables.at(name).values;
    const auto r = Departure(radial.variables.at(name).values, want);
    const auto u = Departure(uniform.variables.at(name).values, want);
    worstRadial = std::max(worstRadial, r);
    worstUniform = std::max(worstUniform, u);
    EXPECT_LT(r, 1e-10) << name;
    EXPECT_LT(u, 1e-9) << name;
  }
  RecordProperty("radial_departure", std::format("{:.3e}", worstRadial));
  RecordProperty("lateral_departure", std::format("{:.3e}", worstUniform));
  auto worstReference = 0.0;
  for (const auto& [reference, field] :
       {std::pair{"relative_sea_level_change_radial_model",
                  "relative_sea_level_change"},
        std::pair{"displacement_radial_model", "displacement"},
        std::pair{"potential_radial_model", "potential"}}) {
    const auto departure = Departure(uniform.variables.at(reference).values,
                                     radial.variables.at(field).values);
    worstReference = std::max(worstReference, departure);
    EXPECT_LT(departure, 1e-10) << reference;
    EXPECT_FALSE(radial.variables.contains(reference)) << reference;
  }
  {
    // the three components together (the toy's x and y are zero)
    auto got = std::vector<double>{}, want = std::vector<double>{};
    for (const auto* axis : {"x", "y", "z"}) {
      const auto& mine = std::get<std::vector<double>>(
          uniform.meta.at(std::format("omega_{}_radial_model_rad_s", axis)));
      const auto& theirs = std::get<std::vector<double>>(
          radial.meta.at(std::format("omega_{}_rad_s", axis)));
      got.insert(got.end(), mine.begin(), mine.end());
      want.insert(want.end(), theirs.begin(), theirs.end());
    }
    EXPECT_LT(Departure(got, want), 1e-9);
  }
  RecordProperty("reference_departure", std::format("{:.3e}", worstReference));
  EXPECT_EQ(std::get<std::string>(radial.meta.at("step_solver")), "direct");
  EXPECT_EQ(std::get<std::vector<long long>>(radial.meta.at("step_iterations")),
            std::vector<long long>(5, 0));
  EXPECT_EQ(std::get<std::string>(uniform.meta.at("step_solver")),
            "fixed_point");
  const auto& passes =
      std::get<std::vector<long long>>(uniform.meta.at("step_iterations"));
  ASSERT_EQ(passes.size(), 5u);
  EXPECT_TRUE(std::all_of(passes.begin() + 1, passes.end(),
                          [](long long n) { return n > 0; }));
  EXPECT_EQ(std::get<std::string>(uniform.meta.at("scheme")),
            "exponential_trapezoid");

  // the 1-D kernels on the radial file
  const auto kernels = [&](const std::string& name, const std::string& file) {
    static_cast<void>(app::RunKernel(
        Write(name + ".json",
              "{" + modelOf(file) + rest +
                  std::format(R"(, "kernels": {{
                "parameters": ["ice", "viscosity"], "measurements": [
                {{"type": "relative_sea_level_point", "name": "site",
                  "latitude": -30.0, "longitude": 20.0}}],
                "output": "out/{}"}}}})",
                              name))));
    return pm::ReadSurfaceFile(dir_ / "out" / name / "site_ice_kernels.nc");
  };
  const auto fromDeck = kernels("kernels_deck", "");
  const auto fromFile = kernels("kernels_radial", "radial.nc");
  EXPECT_LT(Departure(fromFile.variables.at("thickness_dual").values,
                      fromDeck.variables.at("thickness_dual").values),
            1e-10);
}

TEST_F(AppsTest, SolveUnderLateralModuliMatchesTheLibrary) {
  // model.moduli reaches the 3-D evolution as Viscoelastic3D::Options::
  // moduli: the command under the prem8 deck's lateral viscosity and a
  // toy lateral shear modulus is Evolution3D run here on the same
  // operator, background state and ice load, to rounding (the same
  // algorithm: equal to the bit here). Without the moduli the run differs
  // (by about 3e-3 relative).
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  ASSERT_TRUE(io::WriteModuli(dir_ / "moduli.nc", model, SurfaceGrid(8),
                              [](double, double theta, double phi) {
                                return 0.05 * std::sin(theta) * std::cos(phi);
                              },
                              {},
                              {.name = "toy moduli", .elementMetres = 500e3}));
  const auto rest = ", " + Window(4) + R"(, "rotation": "full")";
  const auto summary = app::RunSolve(
      Write("moduli.json", "{" + Prem8(R"(, "moduli": {"file": "moduli.nc"})") +
                               rest +
                               R"(, "output": {"surface": "out/moduli.nc"}})"));
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity, lateral moduli), viscoelastic, 4 nodes 2 to 0 "
      "kyr BP, shorelines frozen, rotation full; ice "))
      << summary;
  static_cast<void>(app::RunSolve(Write(
      "plain.json",
      "{" + Prem8() + rest + R"(, "output": {"surface": "out/plain.nc"}})")));
  const auto byCommand = pm::ReadSurfaceFile(dir_ / "out" / "moduli.nc");
  const auto plain = pm::ReadSurfaceFile(dir_ / "out" / "plain.nc");
  EXPECT_EQ(std::get<std::string>(byCommand.meta.at("moduli")),
            (dir_ / "moduli.nc").string());

  // the library's run: the background from the state at 2 kyr BP, the
  // grounded ice load relative to it, at the window's four nodes
  constexpr auto kLmax = 8;
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto viscosity =
      io::ModelViscosity::Read(data + "/viscosity_log.nc", model);
  ASSERT_TRUE(viscosity);
  const auto rheology = Rheology::Maxwell(*material, viscosity->Viscosity());
  ASSERT_TRUE(rheology);
  const auto moduli = io::ModelModuli::Read(dir_ / "moduli.nc", model);
  ASSERT_TRUE(moduli);
  auto options = Viscoelastic3D::Options{};
  options.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
  const Viscoelastic3D op(*rheology, kLmax, viscosity->Delta(), options);
  const GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLove(),
                              SeaLevelParameters{});
  const auto history =
      io::FieldHistory::Read(dir_ / "history.nc", solver.Grid());
  ASSERT_TRUE(history);
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto times = std::vector<double>{};
  for (auto n = 0; n < 4; ++n) {
    times.push_back(2.0 * app::KYR * n / 3 / scales.Time());
  }
  const auto load = [&](double t) {
    const auto thickness =
        history->Values("ice_thickness", start + t * scales.Time());
    auto out = std::vector<double>(thickness.size());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
               (thickness[i] - ice0[i]) / length;
    }
    return out;
  };
  Evolution3D evolution(op, solver, state);
  const auto snapshots = evolution.Run(times, load);
  ASSERT_EQ(snapshots.size(), 4u);
  // the histories node after node, SI (the first node's zero: no load yet)
  auto worst = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    const auto factor = std::string_view{name} == "potential"
                            ? scales.Factor(PotentialDimensions)
                            : length;
    auto want = std::vector<double>{};
    for (const auto& snapshot : snapshots) {
      EXPECT_TRUE(snapshot.converged);
      const auto& field = std::string_view{name} == "relative_sea_level_change"
                              ? snapshot.relativeSeaLevelChange
                              : (std::string_view{name} == "displacement"
                                     ? snapshot.displacement
                                     : snapshot.potential);
      for (const auto v : field) want.push_back(v * factor);
    }
    const auto departure = Departure(byCommand.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  const auto moved =
      Departure(plain.variables.at("relative_sea_level_change").values,
                byCommand.variables.at("relative_sea_level_change").values);
  RecordProperty("moduli_departure", std::format("{:.3e}", moved));
  EXPECT_GT(moved, 1e-6);
}

TEST_F(AppsTest, Solve3DStepSolversAgree) {
  // A 3-D run closes each step's combined fixed point by plain passes or
  // over conjugate-gradient solves nested inside (solver.step), to the
  // same tolerance (solver.step_rtol, 1e-10 here): the two agree to it
  // over the run (about 4e-12).
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto body = Prem8() + ", " + Window(4) + R"(, "rotation": "full")";
  const auto run = [&](const std::string& name, const std::string& step) {
    return app::RunSolve(
        Write(name + ".json",
              "{" + body +
                  std::format(R"(, "solver": {{"step": "{}", "step_rtol": 1e-10,
                "step_max_iterations": 200}},
                "output": {{"surface": "out/{}.nc"}}}})",
                              step, name)));
  };
  static_cast<void>(run("fixed", "fixed_point"));
  static_cast<void>(run("conjugate", "conjugate_gradients"));
  const auto a = pm::ReadSurfaceFile(dir_ / "out" / "fixed.nc");
  const auto b = pm::ReadSurfaceFile(dir_ / "out" / "conjugate.nc");
  auto worst = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    const auto departure =
        Departure(b.variables.at(name).values, a.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-9) << name;
  }
  RecordProperty("solver_departure", std::format("{:.3e}", worst));
}

/// A fingerprint config on the prem8 deck at lmax 8, the moduli member and
/// the solver block given where they are: an off-pole cap of 50 m of ice
/// laid on the toy history's present day (its ice and topography at
/// 0 kyr BP), with the full rotational feedback.
std::string Prem8Fingerprint(const std::string& name,
                             const std::string& moduli = "",
                             const std::string& solver = "") {
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  return std::format(
      R"({{
    "problem": "static",
    "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic"{}}},
    "discretisation": {{"lmax": 8}},
    "forcing": {{"ice_change": [{{"cap_thickness": {{"latitude": 70,
                                  "longitude": 30, "width_deg": 10,
                                  "thickness_m": 50}}}}]}},
    "ocean": {{"date_kyr_bp": 0, "topography": {{"file": "history.nc"}},
              "ice": {{"file": "history.nc"}}}},
    "rotation": "full",{}
    "output": {{"surface": "out/{}.nc"}}}})",
      data,
      moduli.empty() ? std::string{}
                     : std::format(R"(, "moduli": {{"file": "{}"}})", moduli),
      solver.empty() ? std::string{} : std::format(R"( "solver": {},)", solver),
      name);
}

TEST_F(AppsTest, SolveFingerprintUnderLateralModuliMatchesTheLibrary) {
  // In the static problem, a laterally varying moduli file makes the
  // fingerprint 3-D: the command is Evolution3D::Instantaneous on
  // the stepper of a branchless rheology under the same moduli, solved
  // here on the same state and load, to rounding (equal to the bit here),
  // with the metadata of the 3-D solve. Its *_radial_model fields are the
  // radial fingerprint of the same change, a run without the moduli to the
  // bit, and the lateral moduli (10% in shear, 5% in bulk) move the
  // fingerprint measurably (about 2e-2 of its largest sea level change).
  // A run through time cannot pin
  // it: the command measures a history's load from the window's start, so
  // the first node of a run through time carries no load; the library's
  // gates hold Instantaneous to Run's first node.
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  const auto shear = [](double, double theta, double phi) {
    return 0.1 * std::sin(theta) * std::cos(phi);
  };
  const auto bulk = [](double, double theta, double) {
    return 0.05 * std::cos(theta);
  };
  ASSERT_TRUE(io::WriteModuli(dir_ / "moduli.nc", model, SurfaceGrid(8), shear,
                              bulk,
                              {.name = "toy moduli", .elementMetres = 500e3}));
  const auto summary = app::RunSolve(
      Write("lateral.json", Prem8Fingerprint("lateral", "moduli.nc")));
  EXPECT_TRUE(
      summary.starts_with("3-D (lateral moduli), elastic, static, shorelines "
                          "frozen, rotation full; ice "))
      << summary;
  EXPECT_NE(summary.find(":ice_thickness (0 kyr BP), topography "),
            std::string::npos)
      << summary;
  EXPECT_NE(summary.find("fixed-point iterations, residual "),
            std::string::npos)
      << summary;
  static_cast<void>(
      app::RunSolve(Write("radial.json", Prem8Fingerprint("radial"))));
  const auto byCommand = pm::ReadSurfaceFile(dir_ / "out" / "lateral.nc");
  const auto radial = pm::ReadSurfaceFile(dir_ / "out" / "radial.nc");
  for (const auto* name :
       {"ice_change", "relative_sea_level_change", "displacement", "potential",
        "ocean_function", "relative_sea_level_change_radial_model",
        "displacement_radial_model", "potential_radial_model"}) {
    EXPECT_TRUE(byCommand.variables.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(byCommand.meta.at("moduli")),
            (dir_ / "moduli.nc").string());
  EXPECT_EQ(byCommand.meta.at("lateral"),
            pm::netcdf::Attribute{std::vector<long long>{1}});
  EXPECT_EQ(std::get<std::string>(byCommand.meta.at("moduli_reference")),
            "lateral_mean");
  EXPECT_EQ(std::get<std::string>(byCommand.meta.at("step_solver")),
            "fixed_point");
  const auto iterations =
      std::get<std::vector<long long>>(byCommand.meta.at("step_iterations"));
  const auto residual =
      std::get<std::vector<double>>(byCommand.meta.at("step_residual"));
  EXPECT_GT(iterations.at(0), 0);
  EXPECT_LE(residual.at(0), 1e-9);

  // the radial model's fields are the radial fingerprint's, to the bit: the
  // same radial solve in one process, the same transforms
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(byCommand.variables.at(std::string{name} + "_radial_model").values,
              radial.variables.at(name).values)
        << name;
  }
  for (const auto* axis : {"x", "y", "z"}) {
    EXPECT_EQ(byCommand.meta.at(std::format("omega_{}_radial_model_rad_s", axis)),
              radial.meta.at(std::format("omega_{}_rad_s", axis)))
        << axis;
  }

  // the library's instantaneous response on the same operator, state and
  // load
  constexpr auto kLmax = 8;
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto elastic = Rheology::Create(*material, {});
  ASSERT_TRUE(elastic);
  const auto moduli = io::ModelModuli::Read(dir_ / "moduli.nc", model);
  ASSERT_TRUE(moduli);
  auto options = Viscoelastic3D::Options{};
  options.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
  const Viscoelastic3D op(*elastic, kLmax,
                          [](double, double, double) { return 0.0; }, options);
  const GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLove(),
                              SeaLevelParameters{});
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", solver.Grid());
  ASSERT_TRUE(history);
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  auto ice = history->Values("ice_thickness", 0.0);
  auto sea = SeaLevelFromTopography(history->Values("topography", 0.0), ice,
                                    solver.Parameters());
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  const auto cap = app::CapField(solver.Grid(), {70.0, 30.0},
                                 10.0 * std::numbers::pi / 180.0, 1.0);
  auto load = std::vector<double>(cap.size());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    const auto thickness = 50.0 * cap[i];
    load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
              thickness / length;
  }
  Evolution3D evolution(op, solver, state);
  const auto response = evolution.Instantaneous(load);
  ASSERT_TRUE(response.converged);
  EXPECT_EQ(iterations.at(0), response.coupledIterations);
  auto worst = 0.0;
  for (const auto& [name, field, factor] :
       {std::tuple{"relative_sea_level_change", &response.relativeSeaLevelChange, length},
        std::tuple{"displacement", &response.displacement, length},
        std::tuple{"potential", &response.potential,
                   scales.Factor(PotentialDimensions)}}) {
    auto want = *field;
    for (auto& v : want) v *= factor;
    const auto departure = Departure(byCommand.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  const auto moved =
      Departure(byCommand.variables.at("relative_sea_level_change").values,
                radial.variables.at("relative_sea_level_change").values);
  RecordProperty("moduli_departure", std::format("{:.3e}", moved));
  EXPECT_GT(moved, 1e-5);
  EXPECT_LT(moved, 0.1);
}

TEST_F(AppsTest, SolveFingerprintOnLaterallyUniformModuliIsTheRadialOne) {
  // A moduli file stored laterally but uniform (its lateral part zero)
  // makes the fingerprint 3-D, and the 3-D fingerprint is the radial one
  // to the solves' tolerance (5e-16 here: the operator's elastic Love
  // numbers and the radial solver's agree, and the zero-length step's
  // fixed point adds nothing).
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  const auto zero = [](double, double, double) { return 0.0; };
  ASSERT_TRUE(io::WriteModuli(dir_ / "uniform.nc", model, SurfaceGrid(8), zero,
                              zero,
                              {.name = "uniform moduli", .elementMetres = 500e3}));
  const auto summary = app::RunSolve(
      Write("uniform.json", Prem8Fingerprint("uniform", "uniform.nc")));
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral moduli), elastic, static, shorelines frozen"))
      << summary;
  static_cast<void>(
      app::RunSolve(Write("radial.json", Prem8Fingerprint("radial"))));
  const auto uniform = pm::ReadSurfaceFile(dir_ / "out" / "uniform.nc");
  const auto radial = pm::ReadSurfaceFile(dir_ / "out" / "radial.nc");
  auto worst = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    const auto departure = Departure(uniform.variables.at(name).values,
                                     radial.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-9) << name;
    // the same radial solve and inputs, the same transforms: bitwise
    EXPECT_EQ(uniform.variables.at(std::string{name} + "_radial_model").values,
              radial.variables.at(name).values)
        << name;
  }
  EXPECT_EQ(uniform.variables.at("ocean_function").values,
            radial.variables.at("ocean_function").values);
  EXPECT_EQ(uniform.variables.at("ice_change").values,
            radial.variables.at("ice_change").values);
  RecordProperty("radial_departure", std::format("{:.3e}", worst));
}

TEST_F(AppsTest, SolveFingerprintStepSolversAgree) {
  // The 3-D fingerprint closes its zero-length step by the fixed point or
  // by the sea level closure nested over conjugate-gradient solves
  // (solver.step), to the same tolerance (solver.step_rtol, 1e-10 here):
  // the two agree to it (about 2e-12).
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", model, SurfaceGrid(8),
      [](double, double theta, double phi) {
        return 0.1 * std::sin(theta) * std::cos(phi);
      },
      {}, {.name = "toy moduli", .elementMetres = 500e3}));
  const auto run = [&](const std::string& name, const std::string& step) {
    return app::RunSolve(Write(
        name + ".json",
        Prem8Fingerprint(name, "moduli.nc",
                         std::format(R"({{"step": "{}", "step_rtol": 1e-10,
                                     "step_max_iterations": 200}})",
                                     step))));
  };
  const auto fixed = run("fixed", "fixed_point");
  EXPECT_NE(fixed.find("fixed-point iterations"), std::string::npos) << fixed;
  const auto conjugate = run("conjugate", "conjugate_gradients");
  EXPECT_NE(conjugate.find("conjugate-gradient iterations in"),
            std::string::npos)
      << conjugate;
  const auto a = pm::ReadSurfaceFile(dir_ / "out" / "fixed.nc");
  const auto b = pm::ReadSurfaceFile(dir_ / "out" / "conjugate.nc");
  EXPECT_EQ(std::get<std::string>(b.meta.at("step_solver")),
            "conjugate_gradients");
  EXPECT_GT(std::get<std::vector<long long>>(b.meta.at("solve_iterations")).at(0),
            0);
  auto worst = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    const auto departure =
        Departure(b.variables.at(name).values, a.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-9) << name;
  }
  RecordProperty("solver_departure", std::format("{:.3e}", worst));
}

/// A cap and a disk as load terms, their properties left to the run.
constexpr auto kCapTerm = R"({"cap": {"latitude": 60, "longitude": -85,
    "width_deg": 20, "amplitude_kg_m2": 1.0e6})";
constexpr auto kDiskTerm = R"({"disk": {"latitude": -20, "longitude": 100,
    "radius_deg": 25, "amplitude_kg_m2": 4600})";

/// A point moment tensor and a tide coefficient, open for members beside
/// them (an onset, a history).
constexpr auto kPointOpen = R"({"point": {"latitude": 30, "longitude": 40,
    "depth_km": 300, "moment_tensor_nm": {"rr": 9e20, "tt": -2e20,
    "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}}, )";
constexpr auto kTideOpen = R"({"degree": 2, "order": 1,
    "amplitude_m2_s2": 1.0, "imaginary_m2_s2": 0.5, )";

/// The library's evolution of the ball's Maxwell deck at lmax 8 on a body
/// without an ocean and without rotational feedback (Evolution from
/// GRDSolver::NoOceanState), at the nodes `timesOf` gives for the solver's
/// time unit of so many seconds (formed as the command forms them, so that
/// the steps are the command's to the bit), under the cap of kCapTerm
/// times `capFactor`, band-limited, the tide coefficient of kTideOpen
/// times `tideFactor` and the point source of kPointOpen from `onset` kyr
/// on, each factor a function of kyr from the start: the fields a run
/// without an ocean writes, SI, node after node.
std::map<std::string, std::vector<double>> LibraryRunWithoutAnOcean(
    const fs::path& deck,
    const std::function<std::vector<double>(double)>& timesOf,
    const std::function<double(double)>& capFactor,
    const std::function<double(double)>& tideFactor, double onset) {
  const auto model = ModelOf(deck, app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = 8});
  EXPECT_TRUE(mesh);
  const auto material = Material::Create(model, mesh.value());
  EXPECT_TRUE(material);
  const auto rheology = Rheology::FromModel(material.value(), model);
  EXPECT_TRUE(rheology);
  const ViscoelasticOperator op(rheology.value(), 8);
  const SurfaceGrid grid(8);
  const GRDSolver solver(grid, op.ElasticLoveNumbers(), SeaLevelParameters{});
  const auto& units = solver.SolverScales();
  const auto rate = units.Time() / app::KYR;
  const auto kyrOf = [rate](double t) { return t * rate; };
  const auto cap = grid.Synthesise(grid.Expand(app::CapField(
      grid, {60.0, -85.0}, 20.0 * std::numbers::pi / 180.0, 1.0e6)));
  const auto toLoad = units.Factor(Dimensions{.mass = 1, .length = -2});
  const auto toPotential = units.Factor(PotentialDimensions);
  const auto energy =
      units.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto surface = material->Radius() * model.ModelScales().Length();
  const auto site = GeographicPoint{30.0, 40.0};
  // an onset at a node after the start is a jump of the forcing there:
  // the node repeated, the source absent from its left limit
  // (Evolution::Forcing::sourcesBefore), the window's node its right
  auto times = std::vector<double>{};
  auto right = std::vector<std::size_t>{};
  for (const auto t : timesOf(units.Time())) {
    if (!times.empty() && std::abs(kyrOf(t) - onset) <= 1e-12) times.push_back(t);
    right.push_back(times.size());
    times.push_back(t);
  }
  const auto source = [&](double t, bool before) {
    const auto kyr = kyrOf(t);
    const auto on =
        (before ? kyr > onset + 1e-12 : kyr >= onset - 1e-12) ? 1.0 : 0.0;
    return std::vector<PointSource>{PointSource{
        .at = {.radius = (surface - 300e3) / units.Length(),
               .colatitude = site.CoLatitude(),
               .longitude = site.Longitude()},
        .moment = {.rr = on * 9e20 / energy,
                   .tt = on * -2e20 / energy,
                   .pp = on * -7e20 / energy,
                   .rt = on * 3e20 / energy,
                   .rp = on * -7e20 / energy,
                   .tp = on * 6e20 / energy}}};
  };
  Evolution evolution(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
  const auto run = evolution.Run(
      times,
      Evolution::Forcing{
          .directLoad =
              [&](double t) {
                auto out = cap;
                for (auto& v : out) v *= capFactor(kyrOf(t)) / toLoad;
                return out;
              },
          .tide =
              [&](double t) {
                auto out = grid.ZeroCoefficients();
                out[SurfaceGrid::Index(2, 1)] = tideFactor(kyrOf(t)) *
                                                Complex{1.0, 0.5} / toPotential;
                return out;
              },
          .sources = [&](double t) { return source(t, false); },
          .stressGluts = {},
          .sourcesBefore = [&](double t) { return source(t, true); }});
  EXPECT_EQ(run.size(), times.size());
  auto out = std::map<std::string, std::vector<double>>{};
  for (const auto k : right) {
    const auto& snapshot = run[k];
    EXPECT_TRUE(snapshot.converged);
    for (const auto& [name, field, factor] :
         {std::tuple{"displacement", &snapshot.displacement, units.Length()},
          std::tuple{"tangential_potential", &snapshot.tangentialPotential,
                     units.Length()},
          std::tuple{"toroidal_potential", &snapshot.toroidalPotential,
                     units.Length()},
          std::tuple{"potential", &snapshot.potential, toPotential}}) {
      auto& values = out[name];
      for (const auto v : *field) values.push_back(v * factor);
    }
  }
  return out;
}

TEST_F(AppsTest, SolveWithoutAnOceanIsTheElasticResponse) {
  // Without an `ocean` block the body has no ocean: the static problem
  // under a cap and a disk, their properties left at their defaults (no
  // water budget, no layer), without rotational feedback, is the solid
  // body's elastic response to the load: the library's static solves on
  // the radial reference systems under the same load, field by field, to
  // rounding (about 2e-15 here: the same unrelaxed solves by two routes).
  // The relative sea level change is then -(u + phi / g); the load applied
  // is the cap as sampled and the disk's synthesis, and outside the water
  // budget band-limited; the water budget changes nothing, and a load laid
  // on the solid surface lowers the relative sea level change by the
  // layer's thickness, the body's response the same. The file holds no
  // field or metadata of a state, its metadata and the summary saying
  // there is no ocean.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  const auto config = [&](const std::string& name, const std::string& disk) {
    return Write(name + ".json", std::format(R"({{{}, "problem": "static",
      "forcing": {{"load": [{}}}, {}]}}, "rotation": "none",
      "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, kCapTerm, disk, name));
  };
  const auto summary = app::RunSolve(config("dry", std::string{kDiskTerm} + "}"));
  EXPECT_EQ(
      summary,
      std::format("1-D, elastic, static, no ocean, rotation none; load (2 "
                  "terms); wrote {}",
                  (dir_ / "out" / "dry.nc").string()));
  static_cast<void>(app::RunSolve(config(
      "budget", std::string{kDiskTerm} + R"(, "water_budget": true})")));
  static_cast<void>(app::RunSolve(config(
      "layered", std::string{kDiskTerm} +
                     R"(, "moves_solid_surface": true, "density_kg_m3": 2300})")));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto dry = read("dry");
  // the library's elastic response to the cap and the disk
  const SurfaceGrid grid(8);
  const auto degree = std::numbers::pi / 180.0;
  const auto capField = app::CapField(grid, {60.0, -85.0}, 20.0 * degree, 1e6);
  const auto disk =
      app::DiskCoefficients(grid, {-20.0, 100.0}, 25.0 * degree, 4600.0);
  auto load = grid.Expand(capField);
  for (auto i = std::size_t{0}; i < load.size(); ++i) load[i] += disk[i];
  const auto want = RadialElastic(dir_ / "ball.deck", app::ModelFormat::Gia, 8,
                                  load, grid.ZeroCoefficients());
  auto worst = 0.0;
  for (const auto& [name, coefficients] :
       {std::pair{"displacement", &want.U},
        std::pair{"tangential_potential", &want.V},
        std::pair{"potential", &want.Phi}}) {
    const auto departure = Departure(dry.variables.at(name).values,
                                     grid.Synthesise(*coefficients));
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));

  // the relative sea level change, -(u + phi / g)
  const auto ball = ElasticBall::Make(dir_ / "ball.deck");
  const auto& scales = ball.solver->SolverScales();
  const auto g = ball.solver->SurfaceGravity() *
                 scales.Factor(Dimensions{.length = 1, .time = -2});
  const auto& u = dry.variables.at("displacement").values;
  const auto& phi = dry.variables.at("potential").values;
  auto equipotential = std::vector<double>(u.size());
  for (auto i = std::size_t{0}; i < u.size(); ++i) {
    equipotential[i] = -(u[i] + phi[i] / g);
  }
  EXPECT_LT(Departure(dry.variables.at("relative_sea_level_change").values,
                      equipotential),
            1e-12);

  // the loads applied, as given and band-limited; nothing of a state
  auto given = capField;
  const auto diskField = grid.Synthesise(disk);
  for (auto i = std::size_t{0}; i < given.size(); ++i) given[i] += diskField[i];
  EXPECT_LT(Departure(dry.variables.at("load").values, given), 1e-14);
  EXPECT_LT(Departure(dry.variables.at("load_outside_water_budget").values,
                      grid.Synthesise(grid.Expand(given))),
            1e-12);
  for (const auto* name : {"load_in_water_budget", "solid_surface_change",
                           "ice_change", "ocean_function",
                           "ocean_function_final", "toroidal_potential"}) {
    EXPECT_FALSE(dry.variables.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(dry.meta.at("ocean")), "none");
  EXPECT_EQ(std::get<std::string>(dry.meta.at("rotation")), "none");
  for (const auto* name : {"ice", "sea_level", "shorelines", "ice_change"}) {
    EXPECT_FALSE(dry.meta.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(dry.meta.at("load")),
            "cap at latitude 60, longitude -85, width 20 deg, 1e+06 kg m^-2, "
            "outside the water budget; disk at latitude -20, longitude 100, "
            "radius 25 deg, 4600 kg m^-2, outside the water budget");

  // the water budget has no ocean to change
  const auto budget = read("budget");
  for (const auto* name : {"relative_sea_level_change", "displacement",
                           "tangential_potential", "potential"}) {
    EXPECT_LT(Departure(budget.variables.at(name).values,
                        dry.variables.at(name).values),
              1e-14)
        << name;
  }
  EXPECT_TRUE(budget.variables.contains("load_in_water_budget"));

  // a layer: its thickness off the relative sea level change, and nothing
  // more
  const auto layered = read("layered");
  const auto& layer = layered.variables.at("solid_surface_change").values;
  auto lowered = dry.variables.at("relative_sea_level_change").values;
  for (auto i = std::size_t{0}; i < lowered.size(); ++i) lowered[i] -= layer[i];
  EXPECT_LT(Departure(layered.variables.at("relative_sea_level_change").values,
                      lowered),
            1e-12);
  EXPECT_GT(*std::ranges::max_element(layer), 1.0);
  for (const auto* name :
       {"displacement", "tangential_potential", "potential"}) {
    EXPECT_LT(Departure(layered.variables.at(name).values,
                        dry.variables.at(name).values),
              1e-14)
        << name;
  }
}

TEST_F(AppsTest, SolveWithoutAnOceanIsTheLibrarysRunThroughTime) {
  // The quasi-static problem without an ocean and without rotational
  // feedback: a cap with an amplitude history, a tide coefficient with its
  // own and a point source from a date on, over a window from 2 kyr BP to
  // the present in half-kyr steps, is the library's evolution on a body
  // without an ocean under the same forcing (the histories' and the
  // onset's dates turned to kyr from the window's start, the onset's node
  // repeated, the source absent from its left limit), built here, node
  // by node, to rounding (about 4e-15 here), the tangential and toroidal
  // potentials included; the load applied at each node is the cap times
  // its factor there, and each exchange table holds the expansion of its
  // node's fields under the problem's named scalars, the node's time from
  // the window's start (`time_kyr`) and its date (`date_kyr_bp`). Without
  // the source the run is checkpointed, a run resumed from its
  // checkpoints is the uninterrupted one to the bit (its tables too), and
  // the same run with an ocean is refused them.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = [&](const std::string& name, const std::string& rest) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      {}, "rotation": "none", "output": {{"surface": "out/{}.nc",
                                          "exchange": "out/{}_{{node}}.txt"}}}})",
                                             kBall, rest, name, name));
  };
  const auto summary = app::RunSolve(config(
      "dry", std::format(R"("forcing": {{
      "load": [{}, "history_kyr_bp": [[2, 0.5], [0.5, 1.5]]}}],
      "tide": [{}"history_kyr_bp": [[2, 1], [0, -1]]}}],
      "seismic": [{}"onset_kyr_bp": 1.5}}]}})",
                         kCapTerm, kTideOpen, kPointOpen)));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 2 to 0 kyr BP, no ocean, rotation none; "
      "load (1 term) + tide (1 coefficient) + seismic (1 source); wrote "))
      << summary;
  const auto dry = pm::ReadSurfaceFile(dir_ / "out" / "dry.nc");
  ASSERT_TRUE(dry.time);
  ASSERT_EQ(dry.time->size(), 5u);

  // the library's evolution under the same forcing, at t kyr from the
  // window's start: the cap's factor 0.5 + t / 1.5 to 1.5 at 1.5 kyr, the
  // tide's 1 - t, the source from 0.5 kyr
  const auto want = LibraryRunWithoutAnOcean(
      dir_ / "ball.deck",
      [](double scale) {
        // a dated window's evenly spaced nodes: its span times n / 4
        auto out = std::vector<double>{};
        for (auto n = 0; n < 5; ++n) out.push_back(2.0 * app::KYR * n / 4 / scale);
        return out;
      },
      [](double t) { return std::min(0.5 + t / 1.5, 1.5); },
      [](double t) { return 1.0 - t; }, 0.5);
  auto worst = 0.0;
  for (const auto& [name, values] : want) {
    const auto departure = Departure(dry.variables.at(name).values, values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  const SurfaceGrid grid(8);
  // the load at each node, the cap as sampled times its factor there
  const auto sampled = app::CapField(grid, {60.0, -85.0},
                                     20.0 * std::numbers::pi / 180.0, 1.0e6);
  for (auto n = std::size_t{0}; n < 5; ++n) {
    auto want = sampled;
    for (auto& v : want) v *= std::min(0.5 + 0.5 * n / 1.5, 1.5);
    EXPECT_LT(Departure(Slice(dry, "load", n), want), 1e-14) << n;
  }
  // each table: its node's fields expanded, under the problem's scalars
  auto tables = 0.0;
  for (auto n = std::size_t{0}; n < 5; ++n) {
    const auto table =
        ReadExchange(dir_ / "out" / std::format("dry_{}.txt", n));
    tables = std::max(
        {tables, Departure(table.U, grid.Expand(Slice(dry, "displacement", n))),
         Departure(table.Phi, grid.Expand(Slice(dry, "potential", n)))});
    EXPECT_EQ(table.scalars.at("node"), std::to_string(n));
    EXPECT_EQ(table.scalars.at("time_kyr"), std::format("{:.17g}", 0.5 * n));
    EXPECT_EQ(table.scalars.at("date_kyr_bp"),
              std::format("{:.17g}", 2.0 - 0.5 * n));
    for (const auto& [name, value] :
         {std::pair{"lmax", "8"}, std::pair{"lateral", "0"},
          std::pair{"load_terms", "1"}, std::pair{"tide_coefficients", "1"},
          std::pair{"cap_latitude_deg", "60"},
          std::pair{"cap_amplitude_kg_m2", "1000000"},
          std::pair{"tide_amplitude_m2_s2", "1 0.5"}}) {
      EXPECT_EQ(table.scalars.at(name), value) << name;
    }
  }
  EXPECT_LT(tables, 1e-12);
  RecordProperty("table_departure", std::format("{:.3e}", tables));
  EXPECT_EQ(std::get<std::string>(dry.meta.at("tide")),
            "tide degree 2 order 1, 1 + 0.5 i m^2 s^-2, history_kyr_bp [[2, "
            "1], [0, -1]]");
  EXPECT_EQ(std::get<std::string>(dry.meta.at("ocean")), "none");
  for (const auto* name : {"initial_sea_level", "ocean_function"}) {
    EXPECT_FALSE(dry.variables.contains(name)) << name;
  }
  for (const auto* name : {"ice", "sea_level", "shorelines",
                           "initial_sea_level"}) {
    EXPECT_FALSE(dry.meta.contains(name)) << name;
  }

  // checkpointed without the source, and resumed
  const auto loads = std::format(R"("forcing": {{
      "load": [{}, "history_kyr_bp": [[2, 0.5], [0.5, 1.5]],
                "water_budget": true, "moves_solid_surface": false}}],
      "tide": [{}"history_kyr_bp": [[2, 1], [0, -1]]}}]}})",
                                 kCapTerm, kTideOpen);
  static_cast<void>(app::RunSolve(config("whole", loads)));
  const auto pieces = config(
      "pieces", loads + R"(, "checkpoint": {"directory": "ckpt", "every": 1,
                                            "max_walltime_min": 1e-9})");
  auto attempts = 0;
  auto part = std::string{};
  for (; attempts < 8; ++attempts) {
    part = app::RunSolve(pieces);
    if (part.find("wrote") != std::string::npos) break;
    EXPECT_NE(part.find("checkpointed"), std::string::npos) << part;
  }
  EXPECT_GE(attempts, 1);
  EXPECT_NE(part.find("resumed"), std::string::npos) << part;
  const auto whole = pm::ReadSurfaceFile(dir_ / "out" / "whole.nc");
  const auto rebuilt = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  // one process, the same transforms, the state read back exactly: bitwise
  for (const auto* name : {"relative_sea_level_change", "displacement",
                           "tangential_potential", "potential", "load"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values, whole.variables.at(name).values)
        << name;
  }
  for (auto n = 0; n < 5; ++n) {
    const auto table = [&](const std::string& name) {
      const auto read = ReadExchange(dir_ / "out" /
                                     std::format("{}_{}.txt", name, n));
      return std::tuple{read.scalars, read.U, read.Phi};
    };
    EXPECT_EQ(table("pieces"), table("whole")) << n;
  }
  const auto wet =
      config("wet", loads + R"(, "ocean": {"at": "start",
                 "topography": {"file": "history.nc"},
                 "ice": {"file": "history.nc"}},
                 "checkpoint": {"directory": "ckpt", "every": 1})");
  EXPECT_THROW(
      {
        try {
          static_cast<void>(app::RunSolve(wet));
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string{error.what()}.find("different run"),
                    std::string::npos)
              << error.what();
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(AppsTest, SolveTakesATideWithAnOcean) {
  // A prescribed tide, `forcing.tide`, beside the ice change of the toy
  // state, under the full rotational feedback: in the static problem the
  // command is the library's one-node Evolution::Run on a branchless
  // rheology under the ice's grounded load and the tide, and through time
  // its Evolution run under the history's load and the tide ramped up by
  // its history, with the hydrostatic moments, both built here, to
  // rounding (equal to the bit here), the angular velocity included. The
  // tide matters to the run, and the summary and metadata say it is
  // there.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kTides = R"([
      {"degree": 2, "order": 1, "amplitude_m2_s2": 500, "imaginary_m2_s2": 200},
      {"degree": 2, "order": 0, "amplitude_m2_s2": -300}])";
  constexpr auto kRampedTides = R"([
      {"degree": 2, "order": 1, "amplitude_m2_s2": 500, "imaginary_m2_s2": 200,
       "history_kyr_bp": [[2, 0], [0, 1]]},
      {"degree": 2, "order": 0, "amplitude_m2_s2": -300,
       "history_kyr_bp": [[2, 0], [0, 1]]}])";
  const auto tideAt = [](const GRDSolver& solver, double factor) {
    const auto& grid = solver.Grid();
    const auto potential = solver.SolverScales().Factor(PotentialDimensions);
    auto out = grid.ZeroCoefficients();
    out[SurfaceGrid::Index(2, 1)] = factor * Complex{500.0, 200.0} / potential;
    out[SurfaceGrid::Index(2, 0)] = factor * Complex{-300.0, 0.0} / potential;
    return out;
  };
  const auto compare = [](const pm::SurfaceFile& file, const GRDSolver& solver,
                          const auto& snapshots, const std::string& label) {
    const auto& scales = solver.SolverScales();
    const auto length = scales.Length();
    auto worst = 0.0;
    for (const auto* name : {"relative_sea_level_change", "displacement",
                             "tangential_potential", "potential"}) {
      const auto which = std::string_view{name};
      const auto factor =
          which == "potential" ? scales.Factor(PotentialDimensions) : length;
      auto want = std::vector<double>{};
      for (const auto& snapshot : snapshots) {
        EXPECT_TRUE(snapshot.converged);
        const auto& field =
            which == "relative_sea_level_change" ? snapshot.relativeSeaLevelChange
            : which == "displacement"            ? snapshot.displacement
            : which == "tangential_potential"    ? snapshot.tangentialPotential
                                                 : snapshot.potential;
        for (const auto v : field) want.push_back(v * factor);
      }
      const auto departure = Departure(file.variables.at(name).values, want);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-12) << name << ", " << label;
    }
    for (const auto& [name, k] : {std::pair{"omega_x_rad_s", 0},
                                  std::pair{"omega_y_rad_s", 1},
                                  std::pair{"omega_z_rad_s", 2}}) {
      auto want = std::vector<double>{};
      for (const auto& snapshot : snapshots) {
        want.push_back(snapshot.angularVelocity[k] / scales.Time());
      }
      EXPECT_LT(Departure(std::get<std::vector<double>>(file.meta.at(name)),
                          want),
                1e-12)
          << name << ", " << label;
    }
    return worst;
  };

  // the static problem
  {
    const auto config = Write("static.json", std::format(R"({{{},
      "problem": "static",
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc",
                                                  "date_kyr_bp": 0}}}}],
                  "tide": {}}},
      "ocean": {{"date_kyr_bp": 2, "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "output": {{"surface": "out/static.nc"}}}})",
                                                         kBall, kTides));
    const auto summary = app::RunSolve(config);
    EXPECT_NE(summary.find("; tide (2 coefficients); wrote "),
              std::string::npos)
        << summary;
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / "static.nc");
    EXPECT_EQ(std::get<std::string>(file.meta.at("tide")),
              "tide degree 2 order 1, 500 + 200 i m^2 s^-2; tide degree 2 "
              "order 0, -300 + 0 i m^2 s^-2");
    const auto ball = ElasticBall::Make(dir_ / "ball.deck");
    const auto& solver = *ball.solver;
    const auto& grid = solver.Grid();
    const auto length = solver.SolverScales().Length();
    const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
    ASSERT_TRUE(history);
    const auto ice0 = history->Values("ice_thickness", -2.0 * app::KYR);
    auto sea = SeaLevelFromTopography(
        history->Values("topography", -2.0 * app::KYR), ice0,
        solver.Parameters());
    auto ice = ice0;
    for (auto& v : ice) v /= length;
    for (auto& v : sea) v /= length;
    const auto state = solver.MakeState(std::move(ice), std::move(sea));
    const auto now = history->Values("ice_thickness", 0.0);
    auto load = std::vector<double>(now.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                (now[i] - ice0[i]) / length;
    }
    Evolution evolution(*ball.op, solver, state,
                        {.rotation = RotationalFeedback::Full});
    const auto snapshots = evolution.Run(
        std::vector<double>{0.0},
        Evolution::Forcing{.directLoad = [&](double) { return load; },
                           .tide = [&](double) { return tideAt(solver, 1.0); }});
    RecordProperty("static_departure",
                   std::format("{:.3e}", compare(file, solver, snapshots,
                                                 "static")));
  }

  // through time, the tide ramped up from nothing at the window's start
  const auto body = std::string{kBall} + ", " + Window(4) +
                    R"(, "rotation": "full", "moments": "hydrostatic")";
  auto tidal = body;
  const auto term = std::string{R"([{"thickness": {"file": "history.nc"}}])"};
  tidal.replace(tidal.find(term), term.size(),
                term + R"(, "tide": )" + kRampedTides);
  static_cast<void>(app::RunSolve(Write(
      "bare.json", "{" + body + R"(, "output": {"surface": "out/bare.nc"}})")));
  const auto summary = app::RunSolve(Write(
      "tidal.json", "{" + tidal + R"(, "output": {"surface": "out/tidal.nc"}})"));
  EXPECT_NE(summary.find("; tide (2 coefficients); wrote "), std::string::npos)
      << summary;
  const auto file = pm::ReadSurfaceFile(dir_ / "out" / "tidal.nc");
  const auto bare = pm::ReadSurfaceFile(dir_ / "out" / "bare.nc");
  EXPECT_GT(Departure(file.variables.at("displacement").values,
                      bare.variables.at("displacement").values),
            1e-3);
  EXPECT_FALSE(bare.meta.contains("tide"));
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = 8});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, 8);
  const GRDSolver solver(
      SurfaceGrid(8), op.ElasticLoveNumbers(),
      WithHydrostaticMoments(SeaLevelParameters{},
                             FluidTidalNumber(*material, 2)));
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto times = std::vector<double>{};
  for (auto n = 0; n < 4; ++n) {
    times.push_back(2.0 * app::KYR * n / 3 / scales.Time());
  }
  Evolution evolution(op, solver, state,
                      {.rotation = RotationalFeedback::Full});
  const auto snapshots = evolution.Run(
      times,
      Evolution::Forcing{
          .directLoad =
              [&](double t) {
                const auto thickness =
                    history->Values("ice_thickness", start + t * scales.Time());
                auto out = std::vector<double>(thickness.size());
                for (auto i = std::size_t{0}; i < out.size(); ++i) {
                  out[i] = solver.IceDensity() *
                           (1.0 - state.oceanFunction[i]) *
                           (thickness[i] - ice0[i]) / length;
                }
                return out;
              },
          .tide =
              [&](double t) {
                return tideAt(solver, t * scales.Time() / app::KYR / 2.0);
              }});
  ASSERT_EQ(snapshots.size(), 4u);
  RecordProperty("library_departure",
                 std::format("{:.3e}", compare(file, solver, snapshots,
                                               "through time")));
}

TEST_F(AppsTest, SolveTakesATideUnderMigratingShorelines) {
  // A prescribed tide, ramped up from nothing at the window's start,
  // beside the marine toy history's ice under migrating shorelines: the
  // 1-D run is the library's RunMigrating under the tide to rounding (the
  // ocean functions to the bit; TestMigratingTide holds the shorelines a
  // tide moves), the tide matters to the run, and the
  // summary and metadata say it is there. A run stopped on the wall clock
  // and resumed from its checkpoints is the uninterrupted one to the bit,
  // in 1-D and in 3-D. The tide enters the run's hashes: a run with it is
  // another run to the checkpoints and to the trajectory store.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  constexpr auto kRampedTides = R"([
      {"degree": 2, "order": 1, "amplitude_m2_s2": 50, "imaginary_m2_s2": 20,
       "history_kyr_bp": [[2, 0], [0, 1]]},
      {"degree": 3, "order": 0, "amplitude_m2_s2": -30,
       "history_kyr_bp": [[2, 0], [0, 1]]}])";
  const auto withTide = [&](std::string body) {
    const auto term = std::string{R"([{"thickness": {"file": "history.nc"}}])"};
    body.replace(body.find(term), term.size(),
                 term + R"(, "tide": )" + kRampedTides);
    return body;
  };
  const auto migrating = std::string{R"("shorelines": "migrating")"};
  const auto bare = std::string{kBall} + ", " +
                    Window(4, "history.nc", "", migrating) +
                    R"(, "rotation": "full", "moments": "hydrostatic")";
  const auto tidal = withTide(bare);
  static_cast<void>(app::RunSolve(Write(
      "bare.json", "{" + bare + R"(, "output": {"surface": "out/bare.nc"}})")));
  const auto summary = app::RunSolve(Write(
      "tidal.json", "{" + tidal + R"(, "output": {"surface": "out/tidal.nc"}})"));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 4 nodes 2 to 0 kyr BP, shorelines migrating, "
      "rotation full, moments hydrostatic; ice "))
      << summary;
  EXPECT_NE(summary.find("; tide (2 coefficients); wrote "), std::string::npos)
      << summary;
  const auto file = pm::ReadSurfaceFile(dir_ / "out" / "tidal.nc");
  const auto without = pm::ReadSurfaceFile(dir_ / "out" / "bare.nc");
  EXPECT_GT(Departure(file.variables.at("relative_sea_level_change").values,
                      without.variables.at("relative_sea_level_change").values),
            1e-3);
  EXPECT_TRUE(file.meta.contains("tide"));
  EXPECT_FALSE(without.meta.contains("tide"));

  // the library's run, built here
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = 8});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, 8);
  const GRDSolver solver(
      SurfaceGrid(8), op.ElasticLoveNumbers(),
      WithHydrostaticMoments(SeaLevelParameters{},
                             FluidTidalNumber(*material, 2)));
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto potential = scales.Factor(PotentialDimensions);
  const auto history = io::FieldHistory::Read(dir_ / "history.nc", grid);
  ASSERT_TRUE(history);
  const auto start = -2.0 * app::KYR;
  const auto ice0 = history->Values("ice_thickness", start);
  auto sea = SeaLevelFromTopography(history->Values("topography", start), ice0,
                                    solver.Parameters());
  auto ice = ice0;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto times = std::vector<double>{};
  for (auto n = 0; n < 4; ++n) {
    times.push_back(2.0 * app::KYR * n / 3 / scales.Time());
  }
  Evolution evolution(op, solver, state,
                      {.rotation = RotationalFeedback::Full});
  const auto snapshots = evolution.RunMigrating(
      times,
      [&](double t) {
        auto out = history->Values("ice_thickness", start + t * scales.Time());
        for (auto& v : out) v /= length;
        return out;
      },
      Evolution::Forcing{.tide = [&](double t) {
        const auto ramp = t * scales.Time() / app::KYR / 2.0;
        auto out = grid.ZeroCoefficients();
        out[SurfaceGrid::Index(2, 1)] = ramp * Complex{50.0, 20.0} / potential;
        out[SurfaceGrid::Index(3, 0)] = ramp * Complex{-30.0, 0.0} / potential;
        return out;
      }});
  ASSERT_EQ(snapshots.size(), 4u);
  auto worst = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement",
                           "tangential_potential", "potential"}) {
    const auto which = std::string_view{name};
    const auto factor = which == "potential" ? potential : length;
    auto want = std::vector<double>{};
    for (const auto& snapshot : snapshots) {
      ASSERT_TRUE(snapshot.converged);
      const auto& field =
          which == "relative_sea_level_change" ? snapshot.relativeSeaLevelChange
          : which == "displacement"            ? snapshot.displacement
          : which == "tangential_potential"    ? snapshot.tangentialPotential
                                               : snapshot.potential;
      for (const auto v : field) want.push_back(v * factor);
    }
    const auto departure = Departure(file.variables.at(name).values, want);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
  auto oceans = std::vector<double>{};
  for (const auto& snapshot : snapshots) {
    oceans.insert(oceans.end(), snapshot.oceanFunction.begin(),
                  snapshot.oceanFunction.end());
  }
  // 0 or 1 by flotation: unequal only at a point on the threshold
  EXPECT_EQ(file.variables.at("ocean_function").values, oceans);
  RecordProperty("library_departure", std::format("{:.3e}", worst));

  // stopped on the wall clock at every node and resumed: the uninterrupted
  // run, 1-D and 3-D
  const auto tidal3d = withTide(Prem8() + ", " +
                                Window(4, "history.nc", "", migrating) +
                                R"(, "rotation": "full")");
  for (const auto& [name, body] :
       {std::pair{std::string{"1d"}, tidal}, std::pair{std::string{"3d"}, tidal3d}}) {
    static_cast<void>(app::RunSolve(Write(
        "whole_" + name + ".json",
        "{" + body + R"(, "output": {"surface": "out/whole_)" + name +
            R"(.nc"}})")));
    const auto pieces = Write(
        "pieces_" + name + ".json",
        "{" + body + R"(, "checkpoint": {"directory": "ckpt_)" + name +
            R"(", "every": 1, "max_walltime_min": 1e-9},
            "output": {"surface": "out/pieces_)" +
            name + R"(.nc"}})");
    auto said = std::string{};
    for (auto attempts = 0; attempts < 8; ++attempts) {
      said = app::RunSolve(pieces);
      if (said.find("wrote") != std::string::npos) break;
    }
    EXPECT_NE(said.find("resumed"), std::string::npos) << said;
    const auto reference =
        pm::ReadSurfaceFile(dir_ / "out" / ("whole_" + name + ".nc"));
    const auto rebuilt =
        pm::ReadSurfaceFile(dir_ / "out" / ("pieces_" + name + ".nc"));
    // one process, the same transforms, the state read back exactly: bitwise
    for (const auto* variable :
         {"relative_sea_level_change", "displacement", "tangential_potential",
          "potential", "ocean_function"}) {
      EXPECT_EQ(rebuilt.variables.at(variable).values,
                reference.variables.at(variable).values)
          << variable << ", " << name;
    }
    EXPECT_EQ(rebuilt.meta.at("omega_x_rad_s"), reference.meta.at("omega_x_rad_s"))
        << name;
  }

  // the hashes: a run with the tide is another run
  const auto hashesOf = [&](const std::string& body) {
    const auto input = app::ReadSolveInput(
        Write("hashed.json",
              "{" + body + R"(, "output": {"surface": "out/hashed.nc"}})"));
    auto hash = app::Fnv1a{};
    app::AddRunMembers(hash, input.run, false);
    return std::pair{hash.Value(), app::TrajectoryHashOf(input.run, false)};
  };
  const auto [plainCheckpoint, plainStore] = hashesOf(bare);
  const auto [tideCheckpoint, tideStore] = hashesOf(tidal);
  EXPECT_NE(tideCheckpoint, plainCheckpoint);
  EXPECT_NE(tideStore, plainStore);
}

TEST_F(AppsTest, SolveWithoutAnOceanIn3DMatchesTheLibrary) {
  // Without an ocean, the 3-D runs: through time on the prem8 deck's
  // lateral viscosity under a cap with a history and a tide, the dated run
  // is the same run counting forwards in kyr (the histories' dates turned
  // to times; that form is the library's stepper,
  // ViscoelasticResponseMatchesTheLibraryStepper), node by node to
  // rounding (to the bit here), its toroidal displacement written, zero
  // at the radial start, and so are the load at each node, the radial
  // reference model's fields, each step's closure (its passes, their
  // conjugate-gradient iterations and its residual, the passes the same)
  // and the exchange tables; and the static problem on lateral moduli, the
  // zero-length step's fixed point against the library's
  // conjugate-gradient elastic solve under the same moduli
  // (Viscoelastic3D::SolveElastic), to their tolerances (1e-11 here), with
  // the radial model's response beside it, the library's static solves on
  // the radial reference systems to rounding, its exchange table (the
  // expansion of the 3-D fields) and its volume file (the elastic solve
  // under the static solve's loads, at the radial reference's memory
  // nodes) likewise.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kSolver = R"("solver": {"step_rtol": 1e-11,
                                         "step_max_iterations": 200})";
  static_cast<void>(Run(
      "kyr", std::format(R"({}, "times": {{"unit": "kyr", "end": 1.5,
                                         "nodes": 4}},
      "forcing": {{"load": [{}, "history": [[0, 0.5], [1, 1]]}}],
                  "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                            "imaginary_m2_s2": 0.5}}]}}, {})",
                         Prem8(), kCapTerm, kSolver)));
  const auto summary = app::RunSolve(Write(
      "dry.json", std::format(R"({{{}, "problem": "quasi_static",
      "times": {{"start_kyr_bp": 1.5, "end_kyr_bp": 0, "nodes": 4}},
      "forcing": {{"load": [{}, "history_kyr_bp": [[1.5, 0.5], [0.5, 1]]}}],
                  "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                            "imaginary_m2_s2": 0.5}}]}},
      "rotation": "none", {}, "output": {{"surface": "out/dry.nc",
                                         "exchange": "out/dry_{{node}}.txt",
                                         "exchange_nodes": [3, 0]}}}})",
                              Prem8(), kCapTerm, kSolver)));
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity), viscoelastic, 4 nodes 1.5 to 0 kyr BP, no "
      "ocean, rotation none; load (1 term) + tide (1 coefficient); wrote "))
      << summary;

  const auto dry = pm::ReadSurfaceFile(dir_ / "out" / "dry.nc");
  const auto forwards = Output("kyr");
  auto worst = 0.0;
  for (const auto* name : kResponseFields) {
    const auto departure = Departure(dry.variables.at(name).values,
                                     forwards.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-10) << name;
  }
  RecordProperty("lateral_departure", std::format("{:.3e}", worst));
  for (const auto* name : {"load", "displacement_radial_model",
                           "potential_radial_model"}) {
    EXPECT_LT(Departure(dry.variables.at(name).values,
                        forwards.variables.at(name).values),
              1e-12)
        << name;
  }
  EXPECT_EQ(dry.meta.at("step_iterations"), forwards.meta.at("step_iterations"));
  {
    // the summary sums the steps' passes, the start's left out
    const auto& passes =
        std::get<std::vector<long long>>(forwards.meta.at("step_iterations"));
    EXPECT_TRUE(summary.ends_with(std::format(
        " and 2 exchange tables ({} fixed-point iterations over 3 steps)",
        passes[1] + passes[2] + passes[3])))
        << summary;
  }
  EXPECT_EQ(dry.meta.at("solve_iterations"),
            forwards.meta.at("solve_iterations"));
  // the last passes' residuals, within the tolerance both (rounding moves
  // a residual at the tolerance by more than it moves the fields)
  for (const auto* file : {&dry, &forwards}) {
    const auto& residuals =
        std::get<std::vector<double>>(file->meta.at("step_residual"));
    EXPECT_EQ(residuals.size(), 4u);
    EXPECT_TRUE(std::ranges::all_of(
        residuals, [](double r) { return r >= 0.0 && r <= 1e-11; }));
  }
  for (const auto* name : {"scheme", "step_solver", "lateral", "load_terms",
                           "tide_coefficients", "load_latitude_deg",
                           "tide_imaginary_m2_s2"}) {
    EXPECT_EQ(dry.meta.at(name), forwards.meta.at(name)) << name;
  }
  EXPECT_FALSE(fs::exists(dir_ / "out" / "dry_1.txt"));
  auto tables = 0.0;
  for (const auto n : {0, 3}) {
    tables = std::max(
        tables,
        CompareTables(dir_ / "out" / std::format("dry_{}.txt", n),
                      dir_ / "out" / std::format("kyr_{}.txt", n),
                      {"date_kyr_bp"}));
  }
  EXPECT_LT(tables, 1e-10);
  RecordProperty("table_departure", std::format("{:.3e}", tables));
  {
    const auto& toroidal = dry.variables.at("toroidal_potential").values;
    const auto size = toroidal.size() / dry.time->size();
    EXPECT_TRUE(std::all_of(toroidal.begin(),
                            toroidal.begin() + static_cast<std::ptrdiff_t>(size),
                            [](double v) { return v == 0.0; }));
    EXPECT_TRUE(std::any_of(toroidal.begin() + static_cast<std::ptrdiff_t>(size),
                            toroidal.end(), [](double v) { return v != 0.0; }));
  }

  // the static problem on lateral moduli
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto model =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  ASSERT_TRUE(io::WriteModuli(dir_ / "moduli.nc", model, SurfaceGrid(8),
                              [](double, double theta, double phi) {
                                return 0.1 * std::sin(theta) * std::cos(phi);
                              },
                              [](double, double theta, double) {
                                return 0.05 * std::cos(theta);
                              },
                              {.name = "toy moduli", .elementMetres = 500e3}));
  const auto lateral = std::format(
      R"("model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                   "moduli": {{"file": "moduli.nc"}}}},
                   "discretisation": {{"lmax": 8}})",
      data);
  const auto forcing = std::format(
      R"("forcing": {{"load": [{}}}],
                  "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                            "imaginary_m2_s2": 0.5}}]}})",
      kCapTerm);
  const auto outputs = [](const std::string& name) {
    return std::format(R"("output": {{"surface": "out/{0}.nc",
        "exchange": "out/{0}.txt", "volume": "out/{0}_volume.nc"}})",
                       name);
  };
  const auto staticSummary = app::RunSolve(Write(
      "static.json", std::format(R"({{"problem": "static", {}, {}, {},
      "rotation": "none", {}}})",
                                 lateral, forcing, kSolver, outputs("static"))));
  EXPECT_TRUE(
      staticSummary.starts_with("3-D (lateral moduli), elastic, static, no "
                                "ocean, rotation none; load (1 term) "
                                "+ tide (1 coefficient); wrote "))
      << staticSummary;
  EXPECT_NE(staticSummary.find(std::format(
                "wrote {0}.nc and {0}.txt and {0}_volume.nc (",
                (dir_ / "out" / "static").string())),
            std::string::npos)
      << staticSummary;
  // the library's elastic responses to the cap and the tide, lateral and
  // radial
  const SurfaceGrid grid(8);
  const auto load = grid.Expand(app::CapField(
      grid, {60.0, -85.0}, 20.0 * std::numbers::pi / 180.0, 1.0e6));
  auto tide = grid.ZeroCoefficients();
  tide[SurfaceGrid::Index(2, 1)] = {1.0, 0.5};
  const auto deck = fs::path{data} / "prem8.elastic.deck";
  const auto want =
      LateralElastic(deck, app::ModelFormat::Elastic, 8, dir_ / "moduli.nc",
                     load, tide, 1e-11, dir_ / "out" / "want_volume.nc");
  const auto radial =
      RadialElastic(deck, app::ModelFormat::Elastic, 8, load, tide);
  const auto still = Output("static");
  auto worstStatic = 0.0;
  for (const auto& [name, coefficients] :
       {std::pair{"displacement", &want.U},
        std::pair{"tangential_potential", &want.V},
        std::pair{"potential", &want.Phi}}) {
    const auto departure = Departure(still.variables.at(name).values,
                                     grid.Synthesise(*coefficients));
    worstStatic = std::max(worstStatic, departure);
    EXPECT_LT(departure, 1e-9) << name;
  }
  RecordProperty("static_departure", std::format("{:.3e}", worstStatic));
  for (const auto& [name, coefficients] :
       {std::pair{"displacement_radial_model", &radial.U},
        std::pair{"potential_radial_model", &radial.Phi}}) {
    EXPECT_LT(Departure(still.variables.at(name).values,
                        grid.Synthesise(*coefficients)),
              1e-12)
        << name;
  }
  const auto table = ReadExchange(dir_ / "out" / "static.txt");
  const auto staticTable =
      std::max(Departure(table.U, grid.Expand(still.variables.at("displacement")
                                                  .values)),
               Departure(table.Phi,
                         grid.Expand(still.variables.at("potential").values)));
  EXPECT_LT(staticTable, 1e-12);
  RecordProperty("static_table_departure", std::format("{:.3e}", staticTable));
  EXPECT_LT(std::max(Departure(table.U, want.U), Departure(table.Phi, want.Phi)),
            1e-9);
  const auto volume = VolumeDeparture(dir_ / "out" / "static_volume.nc",
                                      dir_ / "out" / "want_volume.nc");
  EXPECT_LT(volume, 1e-9);
  RecordProperty("static_volume_departure", std::format("{:.3e}", volume));
}

TEST_F(AppsTest, SolveWritesItsStaticOutputsWithoutAnOcean) {
  // Without an ocean and without rotational feedback, the static problem
  // under a cap and a degree-2 tide coefficient on the radial ball writes
  // its exchange table, its volume file and its surface file's metadata:
  // the table's named scalars (the body's radius and surface gravity, the
  // counts, the single cap's and tide's), its header naming the problem,
  // the config and each term, and its coefficients those of the library's
  // static solves on the radial reference systems under the same load and
  // tide, to rounding (about 3e-15 here); the volume file's fields the
  // response inside the body of the same solves, to rounding (about
  // 3e-13); the surface file's `load` the cap as sampled, and its metadata
  // the counts, the single terms, `lateral` and the closure of a 1-D solve
  // (`direct`, zero). The summary lists the files.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  const auto forcing = std::format(R"("forcing": {{"load": [{}}}],
      "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
                "imaginary_m2_s2": 0.5}}]}})",
                                   kCapTerm);
  const auto outputs = [](const std::string& name) {
    return std::format(R"("output": {{"surface": "out/{0}.nc",
        "exchange": "out/{0}.txt", "volume": "out/{0}_volume.nc"}})",
                       name);
  };
  const auto summary = app::RunSolve(Write(
      "dry.json", std::format(R"({{"problem": "static", {}, {},
      "rotation": "none", {}}})",
                              kBall, forcing, outputs("dry"))));
  EXPECT_EQ(summary,
            std::format(
                "1-D, elastic, static, no ocean, rotation none; load (1 term) "
                "+ tide (1 coefficient); wrote {0}.nc and {0}.txt and "
                "{0}_volume.nc",
                (dir_ / "out" / "dry").string()));

  // the library's elastic response to the cap and the tide
  const SurfaceGrid grid(8);
  const auto cap = app::CapField(grid, {60.0, -85.0},
                                 20.0 * std::numbers::pi / 180.0, 1.0e6);
  auto tide = grid.ZeroCoefficients();
  tide[SurfaceGrid::Index(2, 1)] = {1.0, 0.5};
  const auto want = RadialElastic(dir_ / "ball.deck", app::ModelFormat::Gia, 8,
                                  grid.Expand(cap), tide,
                                  dir_ / "out" / "want_volume.nc");

  // the table: its coefficients, its named scalars and its header
  const auto table = ReadExchange(dir_ / "out" / "dry.txt");
  const auto departure =
      std::max(Departure(table.U, want.U), Departure(table.Phi, want.Phi));
  EXPECT_LT(departure, 1e-12);
  RecordProperty("table_departure", std::format("{:.3e}", departure));
  const auto ball = ElasticBall::Make(dir_ / "ball.deck");
  const auto& scales = ball.solver->SolverScales();
  EXPECT_NEAR(std::stod(table.scalars.at("radius")), 6371000.0, 1e-6);
  EXPECT_NEAR(std::stod(table.scalars.at("surface_gravity")),
              ball.solver->SurfaceGravity() *
                  scales.Factor(Dimensions{.length = 1, .time = -2}),
              1e-12);
  EXPECT_NEAR(std::stod(table.scalars.at("cap_width_deg")), 20.0, 1e-13);
  for (const auto& [name, value] :
       {std::pair{"lmax", "8"}, std::pair{"lateral", "0"},
        std::pair{"load_terms", "1"}, std::pair{"tide_coefficients", "1"},
        std::pair{"cap_latitude_deg", "60"},
        std::pair{"cap_longitude_deg", "-85"},
        std::pair{"cap_amplitude_kg_m2", "1000000"},
        std::pair{"tide_degree", "2"}, std::pair{"tide_order", "1"},
        std::pair{"tide_amplitude_m2_s2", "1 0.5"},
        std::pair{"columns", "l m U_re U_im Phi_re Phi_im"}}) {
    EXPECT_EQ(table.scalars.at(name), value) << name;
  }
  EXPECT_EQ(table.scalars.size(), 14u);
  const auto header = HeaderOf(dir_ / "out" / "dry.txt");
  ASSERT_EQ(header.size(), 5u);
  EXPECT_TRUE(header[0].starts_with(
      "# elastic surface response from sl3dpp solve (radial elastic model, "
      "no ocean, rotation none): the m >= 0 spherical harmonic coefficients "
      "of the radial displacement U (m) and the potential perturbation Phi "
      "(m^2 s^-2), "))
      << header[0];
  EXPECT_TRUE(header[1].starts_with("# code sl3dpp version ")) << header[1];
  EXPECT_EQ(header[2], std::format("# config {} model {} moduli none",
                                   (dir_ / "dry.json").string(),
                                   (dir_ / "ball.deck").string()));
  EXPECT_EQ(header[3],
            "# forcing cap at latitude 60, longitude -85, width 20 deg, "
            "1e+06 kg m^-2, outside the water budget");
  EXPECT_EQ(header[4],
            "# forcing tide degree 2 order 1, 1 + 0.5 i m^2 s^-2");

  // the volume file: the library's response inside the body
  const auto volume = VolumeDeparture(dir_ / "out" / "dry_volume.nc",
                                      dir_ / "out" / "want_volume.nc");
  EXPECT_LT(volume, 1e-11);
  RecordProperty("volume_departure", std::format("{:.3e}", volume));

  // the surface file: the load as sampled, the counts and single terms
  const auto dry = pm::ReadSurfaceFile(dir_ / "out" / "dry.nc");
  // the cap sampled by the same function, no transform: bitwise
  EXPECT_EQ(dry.variables.at("load").values, cap);
  const auto integers = [&](const char* name) {
    return std::get<std::vector<long long>>(dry.meta.at(name));
  };
  const auto numbers = [&](const char* name) {
    return std::get<std::vector<double>>(dry.meta.at(name));
  };
  EXPECT_EQ(integers("load_terms"), std::vector<long long>{1});
  EXPECT_EQ(integers("tide_coefficients"), std::vector<long long>{1});
  EXPECT_EQ(numbers("load_latitude_deg"), std::vector<double>{60.0});
  EXPECT_EQ(numbers("load_longitude_deg"), std::vector<double>{-85.0});
  EXPECT_NEAR(numbers("load_width_deg").at(0), 20.0, 1e-13);
  EXPECT_EQ(numbers("load_amplitude_kg_m2"), std::vector<double>{1.0e6});
  EXPECT_EQ(integers("tide_degree"), std::vector<long long>{2});
  EXPECT_EQ(integers("tide_order"), std::vector<long long>{1});
  EXPECT_EQ(numbers("tide_amplitude_m2_s2"), std::vector<double>{1.0});
  EXPECT_EQ(numbers("tide_imaginary_m2_s2"), std::vector<double>{0.5});
  EXPECT_EQ(integers("lateral"), std::vector<long long>{0});
  EXPECT_EQ(std::get<std::string>(dry.meta.at("step_solver")), "direct");
  for (const auto* name : {"step_iterations", "solve_iterations"}) {
    EXPECT_EQ(integers(name), std::vector<long long>{0}) << name;
  }
  EXPECT_EQ(numbers("step_residual"), std::vector<double>{0.0});
}

TEST_F(AppsTest, SolveWritesItsTablesAndVolumeWithAnOcean) {
  // With an ocean the exchange table holds the relative sea level change's
  // coefficients beside the displacement's and the potential's (columns
  // SL_re SL_im), each the expansion of the surface file's field (to the
  // bit here), the state's inputs and the ice change's terms in its
  // header: in the static problem on the toy state under an ice change, a
  // load in the water budget and a tide, with the full rotational
  // feedback; and through time at each node chosen. Without the rotation,
  // the static problem's volume file is the response inside the body to
  // the total load the solve reached (the ice's grounded load, the load
  // and the ocean's) and the tide: the library's static solves on the
  // radial reference systems under that load, to rounding (about 1e-13
  // here).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const SurfaceGrid grid(8);
  // a table's columns against a file's fields, expanded
  const auto against = [&](const ExchangeTable& table, const fs::path& path,
                           const pm::SurfaceFile& file, std::size_t node) {
    auto sl = std::vector<Complex>{};
    auto in = std::ifstream(path);
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || !std::isdigit(static_cast<unsigned char>(line.front()))) {
        continue;
      }
      auto row = std::istringstream(line);
      auto l = 0, m = 0;
      auto u = std::array<double, 6>{};
      row >> l >> m;
      for (auto& v : u) row >> v;
      sl.emplace_back(u[4], u[5]);
    }
    const auto field = [&](const std::string& name) {
      return file.time ? Slice(file, name, node) : file.variables.at(name).values;
    };
    return std::max({Departure(table.U, grid.Expand(field("displacement"))),
                     Departure(table.Phi, grid.Expand(field("potential"))),
                     Departure(sl, grid.Expand(field("relative_sea_level_change")))});
  };
  const auto config = [&](const std::string& name, const std::string& rotation,
                          const std::string& extra) {
    return Write(name + ".json", std::format(R"({{{}, "problem": "static",
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc",
                                                  "date_kyr_bp": 0}}}}],
                  "load": [{}, "water_budget": true,
                            "moves_solid_surface": false}}],
                  "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 500,
                            "imaginary_m2_s2": 200}}]}},
      "ocean": {{"date_kyr_bp": 2, "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "{}", "output": {{"surface": "out/{}.nc",
                                   "exchange": "out/{}.txt"{}}}}})",
                                             kBall, kCapTerm, rotation, name,
                                             name, extra));
  };
  static_cast<void>(app::RunSolve(config("wet", "full", "")));
  const auto wet = pm::ReadSurfaceFile(dir_ / "out" / "wet.nc");
  const auto table = ReadExchange(dir_ / "out" / "wet.txt");
  auto worst = against(table, dir_ / "out" / "wet.txt", wet, 0);
  const auto header = HeaderOf(dir_ / "out" / "wet.txt");
  EXPECT_TRUE(header.front().starts_with(
      "# elastic surface response from sl3dpp solve (radial elastic "
      "model, shorelines frozen, rotation full): the m >= 0 spherical "
      "harmonic coefficients of the radial displacement U (m), the potential "
      "perturbation Phi (m^2 s^-2) and the relative sea level change SL (m), "))
      << header.front();
  EXPECT_EQ(table.scalars.at("columns"), "l m U_re U_im Phi_re Phi_im SL_re SL_im");
  EXPECT_EQ(table.scalars.at("load_terms"), "1");
  EXPECT_EQ(table.scalars.at("tide_degree"), "2");
  EXPECT_TRUE(std::ranges::any_of(header, [](const std::string& line) {
    return line.starts_with("# state ice ") &&
           line.find(":ice_thickness (2 kyr BP), topography ") !=
               std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(header, [](const std::string& line) {
    return line.starts_with("# forcing thickness ");
  }));

  // through time, the tables of the nodes chosen
  static_cast<void>(app::RunSolve(
      Write("timed.json", "{" + std::string{kBall} + ", " + Window(3) +
                              R"(, "moments": "hydrostatic",
                                  "output": {"surface": "out/timed.nc",
                                  "exchange": "out/timed_{node}.txt",
                                  "exchange_nodes": [2]}})")));
  const auto timed = pm::ReadSurfaceFile(dir_ / "out" / "timed.nc");
  EXPECT_FALSE(fs::exists(dir_ / "out" / "timed_0.txt"));
  const auto last = ReadExchange(dir_ / "out" / "timed_2.txt");
  worst = std::max(worst, against(last, dir_ / "out" / "timed_2.txt", timed, 2));
  EXPECT_EQ(last.scalars.at("node"), "2");
  EXPECT_EQ(last.scalars.at("time_kyr"), "2");
  EXPECT_EQ(last.scalars.at("date_kyr_bp"), "0");
  EXPECT_LT(worst, 1e-12);
  RecordProperty("table_departure", std::format("{:.3e}", worst));

  // the volume file without the rotation, against the library's response
  // inside the body under the total load the solve reached
  const auto summary = app::RunSolve(config(
      "still", "none", R"(, "volume": "out/still_volume.nc")"));
  EXPECT_NE(summary.find(std::format("and {}", (dir_ / "out" /
                                                "still_volume.nc").string())),
            std::string::npos)
      << summary;
  const auto still = pm::ReadSurfaceFile(dir_ / "out" / "still.nc");
  const auto& ocean = still.variables.at("ocean_function").values;
  const auto& change = still.variables.at("ice_change").values;
  const auto& budget = still.variables.at("load_in_water_budget").values;
  const auto& sea = still.variables.at("relative_sea_level_change").values;
  const auto parameters = SeaLevelParameters{};
  auto total = std::vector<double>(ocean.size());
  for (auto i = std::size_t{0}; i < total.size(); ++i) {
    total[i] = parameters.ice_density * (1.0 - ocean[i]) * change[i] +
               budget[i] + parameters.water_density * ocean[i] * sea[i];
  }
  // the total load's coefficients, real at m = 0 as a real field's are
  auto coefficients = grid.Expand(total);
  for (auto l = 0; l <= 8; ++l) {
    auto& c = coefficients[SurfaceGrid::Index(l, 0)];
    c = c.real();
  }
  auto tide = grid.ZeroCoefficients();
  tide[SurfaceGrid::Index(2, 1)] = {500.0, 200.0};
  static_cast<void>(RadialElastic(dir_ / "ball.deck", app::ModelFormat::Gia, 8,
                                  coefficients, tide,
                                  dir_ / "out" / "total_volume.nc"));
  const auto volume = VolumeDeparture(dir_ / "out" / "still_volume.nc",
                                      dir_ / "out" / "total_volume.nc");
  EXPECT_LT(volume, 1e-11);
  RecordProperty("volume_departure", std::format("{:.3e}", volume));
}

TEST_F(AppsTest, SolveRefusesWhatItsOutputsCannotDo) {
  // The output block's members beside the surface file, held to the
  // problem and refused naming the member: in the static problem a table
  // per node (`{node}`) and `exchange_nodes`; through time an exchange
  // path without `{node}`, `exchange_nodes` without `exchange` or past the
  // last node, and the volume file; the volume file under seismic
  // sources; and optional fields or a smoothing the block cannot say.
  static_cast<void>(Deck());
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };
  const auto still = [&](const std::string& forcing, const std::string& output) {
    return std::format(R"({{{}, "problem": "static", "forcing": {},
        "output": {}}})",
                       kBall, forcing, output);
  };
  const auto timed = [&](const std::string& output) {
    return std::format(R"({{{}, "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2, "end_kyr_bp": 0, "nodes": 3}},
        "forcing": {{"load": [{}}}]}}, "output": {}}})",
                       kBall, kCapTerm, output);
  };
  const auto cap = std::format(R"({{"load": [{}}}]}})", kCapTerm);
  refused(read, still(cap, R"({"surface": "o.nc", "exchange": "o_{node}.txt"})"),
          "output.exchange", "holds `{node}`, which names a table per time node");
  refused(read, still(cap, R"({"surface": "o.nc", "exchange": "o.txt",
                               "exchange_nodes": [0]})"),
          "output.exchange_nodes", "needs the quasi-static problem");
  refused(read, timed(R"({"surface": "o.nc", "exchange": "o.txt"})"),
          "output.exchange", "lacks `{node}`: a run through time writes a table");
  refused(read, timed(R"({"surface": "o.nc", "exchange_nodes": [0]})"),
          "output.exchange_nodes", "need `output.exchange`");
  refused(read, timed(R"({"surface": "o.nc", "exchange": "o_{node}.txt",
                          "exchange_nodes": [1, 3]})"),
          "output.exchange_nodes[1]", "is not a node index in [0, 2]");
  refused(read, timed(R"({"surface": "o.nc", "volume": "v.nc"})"),
          "output.volume", "is the static problem's alone");
  refused(read,
          still(R"({"seismic": [{"point": {"latitude": 30, "longitude": 40,
                     "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                     "rake_deg": 90, "moment_nm": 1e21}}]})",
                R"({"surface": "o.nc", "volume": "v.nc"})"),
          "output.volume", "is not written under seismic sources");
  // the optional fields and the smoothing: a list of known words, none
  // twice; a block of a known kind and a positive width, both required
  refused(read, still(cap, R"({"surface": "o.nc",
                               "fields": "horizontal_displacement"})"),
          "output.fields", "is not a list");
  refused(read, still(cap, R"({"surface": "o.nc", "fields": ["east"]})"),
          "output.fields[0]",
          "is \"east\", which is none of \"horizontal_displacement\"");
  refused(read, still(cap, R"({"surface": "o.nc",
                               "fields": ["horizontal_displacement",
                                          "horizontal_displacement"]})"),
          "output.fields[1]", "is listed twice");
  refused(read, still(cap, R"({"surface": "o.nc", "smoothing": 2})"),
          "output.smoothing", "is not a block");
  refused(read, still(cap, R"({"surface": "o.nc",
                               "smoothing": {"kind": "boxcar", "width_deg": 2}})"),
          "output.smoothing.kind",
          "is \"boxcar\", which is none of \"gaussian\"");
  refused(read, still(cap, R"({"surface": "o.nc", "smoothing": {"width_deg": 2}})"),
          "output.smoothing", "has no member \"kind\"");
  refused(read, still(cap, R"({"surface": "o.nc",
                               "smoothing": {"kind": "gaussian"}})"),
          "output.smoothing", "has no member \"width_deg\"");
  refused(read, timed(R"({"surface": "o.nc",
                          "smoothing": {"kind": "gaussian", "width_deg": 0}})"),
          "output.smoothing.width_deg", "is not a positive angle");
  refused(read, timed(R"({"surface": "o.nc",
                          "smoothing": {"kind": "gaussian", "width_deg": 2,
                                        "fwhm_deg": 4}})"),
          "output.smoothing.fwhm_deg", "is not a recognised member");
}

TEST_F(AppsTest, SolveWritesTheHorizontalDisplacementAndSmoothedFields) {
  // `output.fields` and `output.smoothing` add to the surface file and
  // change nothing already in it. On a radial ball under a surface load,
  // which has no toroidal part: the east and north components are the
  // surface gradient of V, curl-free with divergence -l (l + 1) V_lm
  // degree by degree (their spin-weighted expansions); each smoothed field
  // read at grid points and between them is the cap mean the library's
  // measurement of that width reads there, to rounding at every width (2
  // degrees at lmax 32 and 96 too); the smoothed east and north components are the
  // gradient of the smoothed V; the metadata state the convention, the
  // kind, the width and the degree weights. Through time under a point
  // source, which has a toroidal part, every node's components are the
  // library's displacement field of V and W, and its smoothed fields are
  // its cap means. A displacement_vector_cap measurement is the smoothed
  // east, north and upward components at its centre, at every width (2,
  // 6 and 20 degrees), next to either pole and between, with a toroidal
  // part and without. With an ocean under frozen shorelines a run through
  // time writes the state's ocean function, one static field, which is a
  // migrating run's first node.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 32;
  constexpr auto kWidthDeg = 20.0;
  const auto plain = std::string{R"("surface": "out/plain.nc")"};
  const auto asked = std::format(
      R"("surface": "out/asked.nc", "fields": ["horizontal_displacement"],
         "smoothing": {{"kind": "gaussian", "width_deg": {}}})",
      kWidthDeg);
  const auto still = [&](const std::string& name, const std::string& output) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        std::format(R"({{"problem": "static", "model": {{"deck": "ball.deck"}},
            "discretisation": {{"lmax": {}}}, "forcing": {{"load": [{}]}},
            "rotation": "none", "output": {{{}}}}})",
                    kLmax, kCap, output))));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto bare = still("plain", plain);
  const auto file = still("asked", asked);
  // the fields already there, bitwise as they were: the added ones come after
  for (const auto& [name, field] : bare.variables) {
    ASSERT_TRUE(file.variables.contains(name)) << name;
    EXPECT_EQ(file.variables.at(name).values, field.values) << name;
    for (const auto& [key, value] : field.attributes) {
      if (key == "_FillValue") continue;  // not a number, unequal to itself
      EXPECT_EQ(file.variables.at(name).attributes.at(key), value)
          << name << " " << key;
    }
  }
  for (const auto& [key, value] : bare.meta) {
    ASSERT_TRUE(file.meta.contains(key)) << key;
    if (key != "config") EXPECT_EQ(file.meta.at(key), value) << key;
  }
  for (const auto* name :
       {"displacement_east", "displacement_north", "displacement_smoothed",
        "tangential_potential_smoothed", "potential_smoothed",
        "relative_sea_level_change_smoothed", "displacement_east_smoothed",
        "displacement_north_smoothed"}) {
    ASSERT_TRUE(file.variables.contains(name)) << name;
    EXPECT_EQ(file.variables.at(name).Text("time_support"), "static") << name;
  }
  // the loads applied and the ocean function are not smoothed; a ball
  // without seismic sources has no toroidal potential
  for (const auto* name : {"load_smoothed", "load_in_water_budget_smoothed",
                           "toroidal_potential_smoothed", "ocean_function"}) {
    EXPECT_FALSE(file.variables.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(file.meta.at("smoothing_kind")), "gaussian");
  EXPECT_EQ(std::get<std::vector<double>>(file.meta.at("smoothing_width_deg")),
            std::vector<double>{kWidthDeg});
  // the same quadrature at the same width: bitwise
  const auto weights =
      CapMeanWeights(kLmax, kWidthDeg * std::numbers::pi / 180.0);
  EXPECT_EQ(
      std::get<std::vector<double>>(file.meta.at("smoothing_degree_weights")),
      weights);
  for (const auto* key : {"smoothing", "horizontal_displacement"}) {
    EXPECT_NE(std::get<std::string>(file.meta.at(key)).find("east"),
              std::string::npos)
        << key;
  }

  const SurfaceGrid grid(kLmax, {.nmax = 1});
  const auto& V = file.variables.at("tangential_potential").values;
  const auto& Vs = file.variables.at("tangential_potential_smoothed").values;
  // the components' potentials from their spin-weighted expansions:
  // u^(+-)_lm = sqrt(l (l + 1) / 2) (a +- i b)_lm, u = grad_1 a - r x grad_1 b
  const auto potentials = [&](const std::vector<double>& east,
                              const std::vector<double>& north) {
    auto plus = std::vector<Complex>(grid.FieldSize());
    auto minus = std::vector<Complex>(grid.FieldSize());
    constexpr Complex i(0.0, 1.0);
    for (auto k = std::size_t{0}; k < grid.FieldSize(); ++k) {
      const auto theta = -north[k];
      const auto phi = east[k];
      plus[k] = -(theta - i * phi) / std::numbers::sqrt2;
      minus[k] = (theta + i * phi) / std::numbers::sqrt2;
    }
    const auto p = grid.ExpandSpin(1, plus);
    const auto q = grid.ExpandSpin(-1, minus);
    auto a = grid.ZeroCoefficients(), b = grid.ZeroCoefficients();
    for (auto l = 1; l <= kLmax; ++l) {
      const auto omega = std::sqrt(0.5 * l * (l + 1.0));
      for (auto m = 0; m <= l; ++m) {
        const auto pk = p[SurfaceGrid::SpinIndex(1, l, m)];
        const auto qk = q[SurfaceGrid::SpinIndex(-1, l, m)];
        a[SurfaceGrid::Index(l, m)] = (pk + qk) / (2.0 * omega);
        b[SurfaceGrid::Index(l, m)] = (pk - qk) / (2.0 * i * omega);
      }
    }
    return std::pair{a, b};
  };
  for (const auto& [east, north, potential] :
       {std::tuple{"displacement_east", "displacement_north", &V},
        std::tuple{"displacement_east_smoothed", "displacement_north_smoothed",
                   &Vs}}) {
    const auto [a, b] = potentials(file.variables.at(east).values,
                                   file.variables.at(north).values);
    const auto want = grid.Expand(*potential);
    auto divergence = 0.0, curl = 0.0, size = 0.0;
    for (auto l = 1; l <= kLmax; ++l) {
      const auto ll = l * (l + 1.0);
      for (auto m = 0; m <= l; ++m) {
        const auto k = SurfaceGrid::Index(l, m);
        size = std::max(size, ll * std::abs(want[k]));
        divergence = std::max(divergence, ll * std::abs(a[k] - want[k]));
        curl = std::max(curl, ll * std::abs(b[k]));
      }
    }
    EXPECT_GT(size, 0.0) << east;
    EXPECT_LT(divergence, 1e-12 * size) << east;
    EXPECT_LT(curl, 1e-12 * size) << east;
  }

  // A displacement_vector_cap measurement is the displacement smoothed as
  // a vector read at the cap's centre: its east, north and upward
  // components each the file's smoothed field there, to rounding, at grid
  // points next to either pole and between them.
  const auto vectorCaps = [&](const pm::SurfaceFile& f, std::size_t node,
                              int lmax, double widthDeg) {
    const SurfaceGrid g(lmax);
    const auto ring = g.Longitudes().size();
    const auto rows = g.CoLatitudes().size();
    const auto east = FieldAt(f, "displacement_east_smoothed", node);
    const auto north = FieldAt(f, "displacement_north_smoothed", node);
    const auto up = FieldAt(f, "displacement_smoothed", node);
    auto scale = 0.0;
    for (const auto* c : {&east, &north, &up}) {
      for (const auto v : *c) scale = std::max(scale, std::abs(v));
    }
    ASSERT_GT(scale, 0.0);
    for (const auto [it, ip] :
         {std::pair{std::size_t{0}, ring / 3}, std::pair{rows / 3, ring / 7},
          std::pair{rows / 2, ring * 3 / 4}, std::pair{rows - 1, ring / 2}}) {
      const auto at = it * ring + ip;
      const auto site = GeographicPoint::FromSpherical(g.CoLatitudes()[it],
                                                       g.Longitudes()[ip]);
      for (const auto& [direction, field, label] :
           {std::tuple{SurfaceDirection{.east = 1.0}, &east, "east"},
            std::tuple{SurfaceDirection{.north = 1.0}, &north, "north"},
            std::tuple{SurfaceDirection{.up = 1.0}, &up, "up"}}) {
        EXPECT_NEAR(VectorCapOf(f, node, site, widthDeg, direction, lmax),
                    (*field)[at], 1e-11 * scale)
            << label << " at " << site.latitude << ", " << site.longitude
            << ", lmax " << lmax << ", width " << widthDeg << ", node " << node;
      }
    }
  };
  vectorCaps(file, 0, kLmax, kWidthDeg);

  // the smoothed fields are the cap means at grid points and between them
  // (a grid point's index, or none for a point between them)
  const auto nphi = grid.Longitudes().size();
  auto sites = std::vector<std::pair<GeographicPoint, std::optional<std::size_t>>>{};
  for (const auto [it, ip] : {std::pair{3, 5}, std::pair{10, 40},
                              std::pair{20, 17}, std::pair{30, 60}}) {
    sites.emplace_back(GeographicPoint::FromSpherical(grid.CoLatitudes()[it],
                                                      grid.Longitudes()[ip]),
                       static_cast<std::size_t>(it) * nphi + ip);
  }
  sites.emplace_back(GeographicPoint{55.0, -80.0}, std::nullopt);
  sites.emplace_back(GeographicPoint{-32.05, 115.74}, std::nullopt);
  for (const auto* name : {"displacement", "tangential_potential", "potential",
                           "relative_sea_level_change"}) {
    const auto smoothed = std::string{name} + "_smoothed";
    const auto& values = file.variables.at(smoothed).values;
    const auto scale =
        std::ranges::max(values, {}, [](double v) { return std::abs(v); });
    for (const auto& [site, at] : sites) {
      const auto mean = CapMeanOf(file, name, 0, site, kWidthDeg, kLmax);
      const auto value =
          at ? values[*at] : PointValueOf(file, smoothed, 0, site, kLmax);
      EXPECT_NEAR(value, mean, 1e-12 * std::abs(scale))
          << smoothed << " at " << site.latitude << ", " << site.longitude;
    }
  }
  // and at a width no grid of the band resolves, 2 degrees at lmax 32 and
  // at lmax 96: the measurement is the smoothed field at its centre still
  for (const auto lmax : {32, 96}) {
    const auto name = std::format("narrow_{}", lmax);
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        std::format(R"({{"problem": "static", "model": {{"deck": "ball.deck"}},
            "discretisation": {{"lmax": {}}}, "forcing": {{"load": [{}]}},
            "rotation": "none", "output": {{"surface": "out/{}.nc",
            "fields": ["horizontal_displacement"],
            "smoothing": {{"kind": "gaussian", "width_deg": 2}}}}}})",
                    lmax, kCap, name))));
    const auto narrow = pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
    vectorCaps(narrow, 0, lmax, 2.0);
    const SurfaceGrid fine(lmax);
    const auto ring = fine.Longitudes().size();
    for (const auto* field : {"displacement", "potential", "relative_sea_level_change"}) {
      const auto smoothed = std::string{field} + "_smoothed";
      const auto& values = narrow.variables.at(smoothed).values;
      const auto scale =
          std::ranges::max(values, {}, [](double v) { return std::abs(v); });
      // at grid points under the load and beside it, and between them
      for (const auto [it, ip] : {std::pair{lmax / 6, ring * 3 / 4},
                                  std::pair{lmax / 3, ring / 5}}) {
        const auto site = GeographicPoint::FromSpherical(
            fine.CoLatitudes()[static_cast<std::size_t>(it)], fine.Longitudes()[ip]);
        EXPECT_NEAR(values[static_cast<std::size_t>(it) * ring + ip],
                    CapMeanOf(narrow, field, 0, site, 2.0, lmax),
                    1e-12 * std::abs(scale))
            << smoothed << " at lmax " << lmax;
      }
      const auto site = GeographicPoint{57.3, -83.1};
      EXPECT_NEAR(PointValueOf(narrow, smoothed, 0, site, lmax),
                  CapMeanOf(narrow, field, 0, site, 2.0, lmax),
                  1e-11 * std::abs(scale))
          << smoothed << " at lmax " << lmax;
    }
  }

  // through time under a point source: a toroidal part, every node
  const auto timed = [&](const std::string& name, const std::string& output) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        std::format(R"({{"problem": "quasi_static", "model": {{"deck": "ball.deck"}},
            "discretisation": {{"lmax": 24}},
            "times": {{"unit": "kyr", "at": [0, 0.4, 1.0]}},
            "forcing": {{"seismic": [{{"point": {{"latitude": 30,
                "longitude": 40, "depth_km": 300, "strike_deg": 20,
                "dip_deg": 60, "rake_deg": 30, "moment_nm": 1e21}}}}]}},
            "rotation": "none", "output": {{{}}}}})",
                    output))));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto quiet = timed("plain_run", R"("surface": "out/plain_run.nc")");
  const auto run = timed(
      "asked_run",
      R"("surface": "out/asked_run.nc", "fields": ["horizontal_displacement"],
         "smoothing": {"kind": "gaussian", "width_deg": 20})");
  const auto sharper = timed(
      "sharper_run",
      R"("surface": "out/sharper_run.nc", "fields": ["horizontal_displacement"],
         "smoothing": {"kind": "gaussian", "width_deg": 6})");
  // as without the outputs asked for, bitwise
  for (const auto& [name, field] : quiet.variables) {
    EXPECT_EQ(run.variables.at(name).values, field.values) << name;
  }
  ASSERT_TRUE(run.variables.contains("toroidal_potential_smoothed"));
  const SurfaceGrid small(24, {.nmax = 1});
  const auto nodes = run.time->size();
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    vectorCaps(run, n, 24, 20.0);
    vectorCaps(sharper, n, 24, 6.0);
    const auto u =
        DisplacementField(small, {}, Slice(run, "tangential_potential", n),
                          Slice(run, "toroidal_potential", n));
    EXPECT_LT(Departure(Slice(run, "displacement_east", n), u.east), 1e-14) << n;
    EXPECT_LT(Departure(Slice(run, "displacement_north", n), u.north), 1e-14)
        << n;
    const auto us = DisplacementField(
        small, {}, Slice(run, "tangential_potential_smoothed", n),
        Slice(run, "toroidal_potential_smoothed", n));
    EXPECT_LT(Departure(Slice(run, "displacement_east_smoothed", n), us.east),
              1e-14)
        << n;
    EXPECT_LT(Departure(Slice(run, "displacement_north_smoothed", n), us.north),
              1e-14)
        << n;
    for (const auto* name : {"displacement", "toroidal_potential"}) {
      const auto site = GeographicPoint{25.0, 50.0};
      const auto value = PointValueOf(run, std::string{name} + "_smoothed", n,
                                      site, 24);
      const auto mean = CapMeanOf(run, name, n, site, 20.0, 24);
      const auto scale = std::ranges::max(Slice(run, name, n), {}, [](double v) {
        return std::abs(v);
      });
      EXPECT_NEAR(value, mean, 1e-12 * std::abs(scale)) << name << " node " << n;
    }
  }

  // with an ocean under frozen shorelines, the state's ocean function, one
  // static field whatever the members: a migrating run's first node, which
  // is the state itself
  WriteToyHistory(dir_ / "history.nc");
  const auto ocean = [&](const std::string& name, const std::string& members) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        "{" + std::string{kBall} + ", " + Window(3, "history.nc", "", members) +
            R"(, "rotation": "none", "output": {"surface": "out/)" + name +
            R"(.nc"}})")));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto frozen = ocean("frozen", "");
  const auto moving = ocean("moving", R"("shorelines": "migrating")");
  const auto& held = frozen.variables.at("ocean_function");
  EXPECT_EQ(held.Text("time_support"), "static");
  EXPECT_NE(held.Text("long_name")->find("frozen shorelines hold"),
            std::string::npos);
  EXPECT_EQ(held.values, Slice(moving, "ocean_function", 0));
  EXPECT_EQ(moving.variables.at("ocean_function").Text("time_support"), "node");
}

TEST_F(AppsTest, HorizontalDisplacementOfOneHarmonicIsItsGradient) {
  // The east and north components of the potentials V = Re Y_32 and, apart,
  // W = Re Y_32 (up to a constant, sin^2 cos(theta) cos(2 phi)) against
  // their gradients in closed form: V's grad_1 V, east (1 / sin) d_phi V,
  // north -d_theta V; W's -r x grad_1 W, east -d_theta W, north
  // -(1 / sin) d_phi W.
  const SurfaceGrid grid(12);
  const auto f = [](double theta, double phi) {
    return std::sin(theta) * std::sin(theta) * std::cos(theta) *
           std::cos(2.0 * phi);
  };
  const auto dTheta = [](double theta, double phi) {
    const auto s = std::sin(theta), c = std::cos(theta);
    return (2.0 * s * c * c - s * s * s) * std::cos(2.0 * phi);
  };
  const auto dPhiOverSin = [](double theta, double phi) {
    return -2.0 * std::sin(theta) * std::cos(theta) * std::sin(2.0 * phi);
  };
  const auto field = grid.FromFunction(f);
  const auto spheroidal = app::HorizontalOf(grid, field, {});
  const auto toroidal = app::HorizontalOf(grid, {}, field);
  const auto nphi = grid.Longitudes().size();
  auto worst = 0.0;
  for (auto it = std::size_t{0}; it < grid.CoLatitudes().size(); ++it) {
    for (auto ip = std::size_t{0}; ip < nphi; ++ip) {
      const auto theta = grid.CoLatitudes()[it];
      const auto phi = grid.Longitudes()[ip];
      const auto k = it * nphi + ip;
      worst = std::max(
          {worst, std::abs(spheroidal.east[k] - dPhiOverSin(theta, phi)),
           std::abs(spheroidal.north[k] + dTheta(theta, phi)),
           std::abs(toroidal.east[k] + dTheta(theta, phi)),
           std::abs(toroidal.north[k] + dPhiOverSin(theta, phi))});
    }
  }
  EXPECT_LT(worst, 1e-12);
}

TEST_F(AppsTest, SolveRefusesWhatAnOceanOrATideCannotDo) {
  // Refused, naming the member: an ice change without an ocean (its water
  // has nowhere to go), a layer without its density without an ocean as
  // with one, a tide under a calibrated start, a tide's history in the
  // static problem and a history in kyr without its unit (`history_kyr`)
  // on a tide; the kernels in the ice and of the sea level without an
  // ocean.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };
  constexpr auto kTide = R"({"degree": 2, "amplitude_m2_s2": 1.0)";
  const auto dryStatic = [&](const std::string& forcing) {
    return std::format(R"({{{}, "problem": "static", "forcing": {},
        "output": {{"surface": "o.nc"}}}})",
                       kBall, forcing);
  };
  refused(read,
          dryStatic(R"({"ice_change": [{"cap_thickness": {"latitude": 70,
              "longitude": 0, "width_deg": 10, "thickness_m": 1}}]})"),
          "forcing.ice_change",
          "the config has no `ocean` block: without an ocean the ice's water "
          "has nowhere to go");
  refused(read,
          dryStatic(std::format(R"({{"load": [{},
              "moves_solid_surface": true}}]}})",
                                kCapTerm)),
          "forcing.load[0]", "gives no `density_kg_m3`");
  refused(read,
          dryStatic(std::format(R"({{"tide": [{}, "history_kyr_bp": [[0, 1]]}}]}})",
                                kTide)),
          "forcing.tide[0].history_kyr_bp", "needs the quasi-static problem");
  refused(read,
          dryStatic(std::format(R"({{"tide": [{}, "history_kyr": [[0, 1]]}}]}})",
                                kTide)),
          "forcing.tide[0].history_kyr", "`history_kyr_bp`");
  const auto timed = [&](const std::string& forcing, const std::string& ocean) {
    return std::format(R"({{{}, "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2, "nodes": 3}}, "forcing": {},
        "ocean": {}, "output": {{"surface": "o.nc"}}}})",
                       kBall, forcing, ocean);
  };
  const auto withTide = std::format(
      R"({{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
          "tide": [{}, "history_kyr": [[0, 1]]}}]}})",
      kTide);
  refused(read,
          timed(withTide, R"({"at": "start", "topography": {"file": "history.nc"},
                              "ice": {"file": "history.nc"}})"),
          "forcing.tide[0].history_kyr", "`history_kyr_bp`");
  const auto tide = std::format(
      R"({{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
          "tide": [{}}}]}})",
      kTide);
  refused(read,
          timed(tide, R"({"at": "end", "topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "forcing.tide", "a calibrated initial sea level does not take");
  // the kernels: those in the ice and of the sea level need an ocean
  const auto kernel = [&](const std::string& forcing, const std::string& ocean,
                          const std::string& members) {
    return std::format(R"({{{}, "problem": "quasi_static",
        "times": {{"start_kyr_bp": 2, "nodes": 3}}, "forcing": {},{}
        "kernels": {{{}, "output": "k"}}}})",
                       kBall, forcing, ocean, members);
  };
  constexpr auto kLoad = R"({"load": [{"cap": {"latitude": 60,
      "longitude": -85, "width_deg": 20, "amplitude_kg_m2": 1.0e6}}]})";
  constexpr auto kLod =
      R"("measurements": [{"type": "rotation", "component": "z"}])";
  refused(app::ReadKernelInput,
          kernel(kLoad, "", std::format(R"("parameters": ["ice"], {})", kLod)),
          "kernels.parameters[0]",
          "is the kernels in the ice, which a run without an ocean");
  refused(app::ReadKernelInput,
          kernel(kLoad, "",
                 R"("parameters": ["load"], "measurements": [
                     {"type": "relative_sea_level_point", "latitude": 0,
                      "longitude": 0}])"),
          "kernels.measurements[0].type",
          "is a measurement of the sea level, which a run without an ocean");
  // without `parameters` the words the run takes are listed, without an
  // ocean none of the ice's
  refused(app::ReadKernelInput, kernel(kLoad, "", kLod), "kernels",
          "list them, of \"load\", \"tide\", \"viscosity\" and \"moduli\"");
  // under migrating shorelines the words the run takes include the tide's
  refused(app::ReadKernelInput,
          kernel(R"({"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                 R"( "ocean": {"at": "start", "shorelines": "migrating",
              "topography": {"file": "history.nc"},
              "ice": {"file": "history.nc"}},)",
                 kLod),
          "kernels",
          "list them, of \"ice\", \"load\", \"tide\", \"viscosity\" and "
          "\"moduli\"");
}

TEST_F(AppsTest, SolveAndKernelsRefuseWhatTheyCannotDo) {
  // Each fault of a solve or kernel config is refused naming the member:
  // the problem missing, unknown or at odds with the times block; the
  // degree given at the top level, its discretisation block missing, not
  // an object, holding another member or a degree below 1; times in
  // kyr without a unit in the times block; the members only the
  // quasi-static problem takes (the ocean's calibration, the time step, a
  // viscosity file, checkpoints, the state's `at`) in the static one, and
  // the 3-D step's there without a moduli file, and the static problem's
  // (the state's date, a dated term) in the quasi-static one; the ocean
  // block's sea level missing or given twice, its ice missing, its `at`
  // missing or unknown, a calibration with the state at the start, an
  // unknown format; an ice change without an ocean block; no forcing at
  // all, an empty ice change with nothing beside it, or a term naming no
  // kind or several; the old members'
  // names (the ice as `forcing.ice`, a date or an `at` on the sea level);
  // a member of the commands' blocks given at the top level; an unknown
  // member in every block; solve's members in a kernel config, checked as
  // solve checks them. Once the run reads its files: the state's date held
  // to its files' time axes, a term's file to its kind (a thickness
  // without a time axis, a change dated against its file), a change or a
  // cap with the state at the window's end, a change that takes more ice
  // than the state holds (in the static problem and at a node of a run
  // through time), and the 3-D step's members on a radial moduli file in
  // the static problem.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  // a static ice change: a cap of ice laid on the state
  const auto change = std::string{
      R"({"ice_change": [{"cap_thickness": {"latitude": 70, "longitude": 0,
                          "width_deg": 10, "thickness_m": 1}}]})"};
  const auto dated = std::string{
      R"({"date_kyr_bp": 0, "topography": {"file": "history.nc"},
          "ice": {"file": "history.nc"}})"};
  // a fingerprint config, the forcing and the ocean replaced where given
  const auto fingerprint =
      [&](const std::string& extra,
          const std::string& model = R"({"deck": "ball.deck"})",
          const std::string& forcing = "", const std::string& ocean = "") {
        return std::format(
            R"({{"problem": "static", "model": {},
                 "discretisation": {{"lmax": 4}},
        "forcing": {}, "ocean": {},{} "output": {{"surface": "o.nc"}}}})",
            model, forcing.empty() ? change : forcing,
            ocean.empty() ? dated : ocean, extra);
      };
  // a run through time
  const auto start = std::string{
      R"({"at": "start", "topography": {"file": "history.nc"},
          "ice": {"file": "history.nc"}})"};
  const auto history = std::string{
      R"({"ice_change": [{"thickness": {"file": "history.nc"}}]})"};
  const auto timed = [&](const std::string& extra,
                         const std::string& times =
                             R"("times": {"start_kyr_bp": 2,
                             "nodes": 3})",
                         const std::string& forcing = "",
                         const std::string& ocean = "") {
    return std::format(R"({{"problem": "quasi_static",
        "model": {{"deck": "ball.deck"}}, "discretisation": {{"lmax": 4}}, {},
        "forcing": {}, "ocean": {},{} "output": {{"surface": "o.nc"}}}})",
                       times, forcing.empty() ? history : forcing,
                       ocean.empty() ? start : ocean, extra);
  };
  constexpr auto kTimes = R"("times": {"start_kyr_bp": 2, "nodes": 3})";
  const auto kernel = [&](const std::string& model, const std::string& extra) {
    return std::format(R"({{"problem": "quasi_static", "model": {},
        "discretisation": {{"lmax": 4}},
        "times": {{"start_kyr_bp": 2, "nodes": 3}},
        "forcing": {},
        "ocean": {},{}
        "kernels": {{"parameters": ["ice", "viscosity"],
                    "measurements": [{{"type": "rotation", "component": "z"}}],
                    "output": "k"}}}})",
                       model, history, start, extra);
  };
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };

  // the problem: required, one of two words, and held to the times block
  auto missing = timed("");
  missing.replace(missing.find(R"("problem": "quasi_static",)"),
                  std::string_view{R"("problem": "quasi_static",)"}.size(), "");
  refused(read, missing, "bad.json", "has no `problem`");
  {
    auto unsaid = kernel(R"({"deck": "ball.deck"})", "");
    unsaid.replace(unsaid.find(R"("problem": "quasi_static",)"),
                   std::string_view{R"("problem": "quasi_static",)"}.size(),
                   "");
    refused(app::ReadKernelInput, unsaid, "bad.json", "has no `problem`");
  }
  auto unknown = timed("");
  unknown.replace(unknown.find("quasi_static"), 12, "dynamic");
  refused(read, unknown, "problem", "none of \"static\", \"quasi_static\"");
  auto staticTimed = timed("");
  staticTimed.replace(staticTimed.find("quasi_static"), 12, "static");
  refused(read, staticTimed, "times", "the static problem does not take");
  auto timeless = fingerprint("");
  timeless.replace(timeless.find(R"("static")"), 8, R"("quasi_static")");
  refused(read, timeless, "problem", "needs a `times` block");

  // the degree: the discretisation block's, required, holding it alone
  {
    constexpr auto kBlock =
        std::string_view{R"("discretisation": {"lmax": 4})"};
    const auto degree = [&](const std::string& text,
                            const std::string& member) {
      auto out = text;
      out.replace(out.find(kBlock), kBlock.size(), member);
      return out;
    };
    // solve's config, and a kernel config read by solve
    for (const auto& text :
         {timed(""), kernel(R"({"deck": "ball.deck"})", "")}) {
      refused(read, degree(text, R"("lmax": 4)"), "lmax",
              "is not a top-level member: it belongs in "
              "`discretisation.lmax`");
      auto without = text;
      without.erase(without.find(kBlock), kBlock.size() + 1);  // and its ','
      refused(read, without, "bad.json", "has no `discretisation` block");
      refused(read, degree(text, R"("discretisation": [4])"),
              "discretisation", "is not a block");
      refused(read, degree(text, R"("discretisation": {"lmax": 4, "nr": 8})"),
              "discretisation.nr", "not a recognised member");
      refused(read, degree(text, R"("discretisation": {"lmax": 0})"),
              "discretisation.lmax", "is not a degree");
    }
    refused(app::ReadKernelInput,
            degree(kernel(R"({"deck": "ball.deck"})", ""), R"("lmax": 4)"),
            "lmax",
            "is not a top-level member: it belongs in `discretisation.lmax`");
  }

  // the times block's form
  refused(read, timed("", R"("times": {"end_kyr": 2, "nodes": 3})"),
          "times.end_kyr", "times on the run's clock are `end` with `unit`");
  auto listed = kernel(R"({"deck": "ball.deck"})", "");
  listed.replace(listed.find(R"("start_kyr_bp": 2,)"),
                 std::string_view{R"("start_kyr_bp": 2,)"}.size(),
                 R"("at_kyr": [0, 1],)");
  refused(app::ReadKernelInput, listed, "times.at_kyr", "counts forwards");
  refused(read, timed("", R"("times": {"start_kyr_bp": 0, "end_kyr_bp": 2,
                                       "nodes": 3})"),
          "times.end_kyr_bp", "is not later than start_kyr_bp");
  refused(read, timed("", R"("times": {"start_kyr_bp": 2, "nodes": 1})"),
          "times.nodes", "fewer than two nodes");

  // what only the quasi-static problem takes, in the static one
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"calibration": {}, "date_kyr_bp": 0,
                          "topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "ocean.calibration", "needs the quasi-static problem");
  refused(read, fingerprint(R"( "solver": {"scheme": "backward_euler"},)"),
          "solver.scheme", "needs the quasi-static problem");
  refused(read, fingerprint(R"( "solver": {"step_rtol": 1e-8},)"),
          "solver.step_rtol", "needs the quasi-static problem");
  refused(read, fingerprint(R"( "solver": {"step": "conjugate_gradients"},)"),
          "solver.step", "or lateral moduli (`model.moduli`)");
  // with a moduli file the time step is still refused
  refused(read,
          fingerprint(R"( "solver": {"scheme": "backward_euler"},)",
                      R"({"deck": "ball.deck", "moduli": {"file": "m.nc"}})"),
          "solver.scheme", "needs the quasi-static problem");
  refused(read, fingerprint("", R"({"deck": "ball.deck",
                             "viscosity": {"file": "v.nc"}})"),
          "model.viscosity", "needs the quasi-static problem");
  refused(read,
          fingerprint(R"( "checkpoint": {"directory": "c", "every": 1},)"),
          "checkpoint", "needs the quasi-static problem");
  refused(read, fingerprint(R"( "restart": false,)"), "restart",
          "needs the quasi-static problem");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"at": "start", "topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "ocean.at", "which the static problem does not have");
  // and the static problem's members in the quasi-static one
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc",
                                                  "date_kyr_bp": 0}}]})"),
          "forcing.ice_change[0].thickness.date_kyr_bp",
          "reads it at each node's date");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "date_kyr_bp": 0,
                    "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.date_kyr_bp", "dates the state, as the static problem");

  // the ocean block: the state's sea level and ice, and its instant
  refused(read,
          timed("", kTimes, "",
                R"({"topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "ocean", "has no `at`");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "middle", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.at", "none of \"start\", \"end\"");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"},
                    "sea_level": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "ocean", "gives both `sea_level` and `topography`");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "shorelines": "frozen",
                    "ice": {"file": "history.nc"}})"),
          "ocean", "gives no sea level");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"}})"),
          "ocean", "gives no ice, `ice`");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"date_kyr_bp": 0,
                          "topography": {"file": "history.nc"}})"),
          "ocean", "gives no ice, `ice`");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"},
                    "calibration": {"iterations": 4}})"),
          "ocean.calibration", "\"at\": \"end\"");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start",
                    "topography": {"file": "history.nc", "format": "netcdf"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.topography.format", "none of \"planetmodel_surface\"");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc", "variable": ""}})"),
          "ocean.ice.variable", "is empty");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc",
                                                  "variable": ""}}]})"),
          "forcing.ice_change[0].thickness.variable", "is empty");
  // without an ocean, an ice change is refused: its water has nowhere to go
  refused(read,
          std::format(R"({{"problem": "quasi_static",
              "model": {{"deck": "ball.deck"}},
              "discretisation": {{"lmax": 4}}, {},
              "forcing": {}, "output": {{"surface": "o.nc"}}}})",
                      kTimes, history),
          "forcing.ice_change", "the config has no `ocean` block");

  // the forcing: some term of some kind, each of one kind
  refused(read, fingerprint("", R"({"deck": "ball.deck"})", "{}"), "forcing",
          "holds no forcing");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", R"({"ice_change": []})"),
          "forcing.ice_change", "holds no term, and nothing else forces");
  refused(read, timed("", kTimes, "{}"), "forcing", "holds no forcing");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [], "load": [], "seismic": []})"),
          "forcing.ice_change", "or seismic source (`seismic`)");
  refused(read, timed("", kTimes, R"({"ice_change": [{}]})"),
          "forcing.ice_change[0]", "names no term");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc"},
                                    "change": {"file": "history.nc"}}]})"),
          "forcing.ice_change[0]", "names several terms");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"melt": {"file": "history.nc"}}]})"),
          "forcing.ice_change[0].melt", "not a recognised member");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"thickness":
                          {"file": "history.nc"}}]})"),
          "forcing.ice_change[0].thickness", "has no `date_kyr_bp`");

  // the old members, named with the new
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc"}}],
                    "sea_level": {"file": "history.nc"}})"),
          "forcing.sea_level", "`ocean.sea_level`");
  for (const auto& [ice, what] :
       {std::pair{R"({"ice": {"file": "history.nc"}})", "`ocean.ice`"},
        std::pair{R"({"ice": {"file": "history.nc"}})", "`forcing.ice_change`"},
        std::pair{R"({"ice": {"file": "history.nc", "date_kyr_bp": 0,
                     "type": "cap_thickness", "latitude": 70,
                     "longitude": 0, "width_deg": 10, "thickness_m": 1}})",
                  "`forcing.ice_change`"}}) {
    refused(read, timed("", kTimes, ice), "forcing.ice", what);
    refused(read, fingerprint("", R"({"deck": "ball.deck"})", ice),
            "forcing.ice", what);
  }
  refused(read,
          timed("", kTimes, "",
                R"({"topography": {"file": "history.nc", "at": "start"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.topography.at", "is the block's, `ocean.at`");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start",
                    "topography": {"file": "history.nc", "date_kyr_bp": 0},
                    "ice": {"file": "history.nc"}})"),
          "ocean.topography.date_kyr_bp", "the state's date is the block's");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"topography": {"file": "history.nc",
                                         "date_kyr_bp": 0},
                          "ice": {"file": "history.nc"}})"),
          "ocean.topography.date_kyr_bp",
          "the state's date is the block's, `ocean.date_kyr_bp`");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc", "date_kyr_bp": 0}})"),
          "ocean.ice.date_kyr_bp",
          "the state's date is the block's, `ocean.date_kyr_bp`");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"date_kyr_bp": 0,
                          "topography": {"file": "history.nc", "at": "start"},
                          "ice": {"file": "history.nc"}})"),
          "ocean.topography.at", "`ocean.date_kyr_bp`");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "end", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"},
                    "calibrate_initial_sea_level": {}})"),
          "ocean.calibrate_initial_sea_level", "`ocean.calibration`");

  // the blocks' members at the top level
  for (const auto& [member, block] :
       {std::pair{R"("sea_level": {"file": "history.nc"})",
                  "`ocean.sea_level`"},
        std::pair{R"("topography": {"file": "history.nc"})",
                  "`ocean.topography`"},
        std::pair{R"("ice": {"file": "history.nc"})", "`ocean.ice`"},
        std::pair{R"("calibrate_initial_sea_level": {})",
                  "`ocean.calibration`"},
        std::pair{R"("ice_history": {"file": "history.nc"})",
                  "`forcing.ice_change`"},
        std::pair{R"("ice_change": [])", "`forcing.ice_change`"},
        std::pair{R"("at": "start")", "`ocean.at`"},
        std::pair{R"("coupled": {"rtol": 1e-8})", "`solver`"},
        std::pair{R"("shorelines": "migrating")", "`ocean.shorelines`"},
        std::pair{R"("rtol": 1e-9)", "`solver.rtol`"},
        std::pair{R"("model_format": "elastic")", "`model.format`"}}) {
    const auto name =
        std::string{member}.substr(1, std::string{member}.find('"', 1) - 1);
    refused(read, timed(std::format(" {},", member)), name,
            std::string{"is not a top-level member: it belongs in "} + block);
    refused(app::ReadKernelInput,
            kernel(R"({"deck": "ball.deck"})", std::format(" {},", member)),
            name, "it belongs in");
  }
  refused(read,
          std::format(R"({{"problem": "quasi_static", "model": "ball.deck",
              "discretisation": {{"lmax": 4}}, {}, "forcing": {}, "ocean": {},
              "output": {{"surface": "o.nc"}}}})",
                      kTimes, history, start),
          "model", "is not a block");
  refused(read,
          std::format(R"({{"problem": "quasi_static",
              "model": {{"deck": "ball.deck"}},
              "discretisation": {{"lmax": 4}}, {},
              "forcing": {}, "ocean": {}, "output": "o.nc"}})",
                      kTimes, history, start),
          "output", "is not a block");

  // an unknown member in every block
  refused(read, timed(R"( "colour": "blue",)"), "colour",
          "not a recognised member");
  refused(read,
          timed("", R"("times": {"start_kyr_bp": 2, "nodes": 3, "step": 1})"),
          "times.step", "not a recognised member");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc"}}],
                    "sediment": {}})"),
          "forcing.sediment", "not a recognised member");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start",
                    "topography": {"file": "history.nc", "fiel": "x"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.topography.fiel", "not a recognised member");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc", "fiel": "x"}})"),
          "ocean.ice.fiel", "not a recognised member");
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc",
                                                  "name": "x"}}]})"),
          "forcing.ice_change[0].thickness.name", "not a recognised member");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"cap_thickness": {"latitude": 70,
                          "longitude": 0, "width_deg": 10, "thickness_m": 1,
                          "radius_deg": 3}}]})"),
          "forcing.ice_change[0].cap_thickness.radius_deg",
          "not a recognised member");
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"cap_fraction": {"latitude": 70,
                          "longitude": 0, "width_deg": 10,
                          "thickness_m": 1}}]})"),
          "forcing.ice_change[0].cap_fraction.thickness_m",
          "not a recognised member");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "start", "shoreline": "migrating",
                    "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "ocean.shoreline", "not a recognised member");
  refused(read,
          timed("", kTimes, "",
                R"({"at": "end", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"},
                    "calibration": {"runs": 3}})"),
          "ocean.calibration.runs", "not a recognised member");
  refused(read, timed(R"( "solver": {"max_iterations": 3},)"),
          "solver.max_iterations", "not a recognised member");
  refused(read, timed(R"( "solver": {"step": "picard"},)"), "solver.step",
          "none of");
  refused(read, timed(R"( "checkpoint": {"directory": "c", "every": 1,
                                   "stride": 2},)"),
          "checkpoint.stride", "not a recognised member");
  refused(read,
          std::format(R"({{"problem": "quasi_static",
              "model": {{"deck": "ball.deck", "visc": {{}}}},
              "discretisation": {{"lmax": 4}},
              {}, "forcing": {}, "ocean": {},
              "output": {{"surface": "o.nc"}}}})",
                      kTimes, history, start),
          "model.visc", "not a recognised member");
  refused(read,
          std::format(R"({{"problem": "quasi_static",
              "model": {{"deck": "ball.deck"}},
              "discretisation": {{"lmax": 4}}, {},
              "forcing": {}, "ocean": {},
              "output": {{"surface": "o.nc", "exchnage": "o_{{node}}.txt"}}}})",
                      kTimes, history, start),
          "output.exchnage", "not a recognised member");

  // the kernel config's own: a times block as solve's, and solve's
  // members checked as solve checks them, the static problem's refusal of
  // checkpoints included
  {
    auto untimed = kernel(R"({"deck": "ball.deck"})", "");
    untimed.replace(untimed.find(R"("times": {"start_kyr_bp": 2, "nodes": 3},)"),
                    std::string_view{
                        R"("times": {"start_kyr_bp": 2, "nodes": 3},)"}
                        .size(),
                    "");
    refused(app::ReadKernelInput, untimed, "problem", "needs a `times` block");
  }
  refused(app::ReadKernelInput,
          kernel(R"({"deck": "ball.deck"})",
                 R"( "checkpoint": {"directory": "c", "every": 1,
                                           "stride": 2},)"),
          "checkpoint.stride", "not a recognised member");
  {
    auto still = kernel(R"({"deck": "ball.deck"})",
                        R"( "checkpoint": {"directory": "c", "every": 1},)");
    still.replace(still.find("quasi_static"), 12, "static");
    still.replace(still.find(R"("times": {"start_kyr_bp": 2, "nodes": 3},)"),
                  std::string_view{
                      R"("times": {"start_kyr_bp": 2, "nodes": 3},)"}
                      .size(),
                  "");
    still.replace(still.find(start), start.size(), dated);
    still.replace(still.find(history), history.size(), change);
    refused(app::ReadKernelInput, still, "checkpoint",
            "the static problem is one solve, with nothing to checkpoint");
  }

  // once the files are read: the state's date held to its files (required
  // of a state with a time axis, refused of one without), a term's file
  // held to its kind (a thickness needs a time axis, a static change file
  // its date where it has one), a variable the file lacks refused naming
  // those it holds, a change that takes more ice than the state holds, a
  // change present at the start under migrating shorelines, and the 3-D
  // step's members on a radial moduli file in the static problem
  {
    namespace pm = planetmodel;
    auto file = pm::SurfaceFile{};
    file.grid = pm::Equiangular(9, 18);
    file.source = "sl3dpp tests";
    for (const auto* name : {"ice_thickness", "topography"}) {
      file.variables.emplace(
          name, pm::SurfaceField(name, std::vector<double>(9 * 18, 100.0),
                                 {.dimensions = pm::units::LENGTH}));
    }
    pm::WriteSurfaceFile(dir_ / "static.nc", file);
  }
  const auto still = std::string{
      R"({"topography": {"file": "static.nc"}, "ice": {"file": "static.nc"}})"};
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"date_kyr_bp": 0,
                          "topography": {"file": "static.nc"},
                          "ice": {"file": "static.nc"}})"),
          "ocean.date_kyr_bp", "dates a state whose files have no time axis");
  // the ice's file has a time axis, the topography's none
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"topography": {"file": "static.nc"},
                          "ice": {"file": "history.nc"}})"),
          "ocean",
          "holds a file with a time axis (" + (dir_ / "history.nc").string() +
              "): give the state's date, `date_kyr_bp`");
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "ocean", "give the state's date, `date_kyr_bp`");
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"change": {"file": "static.nc",
                          "variable": "ice_thickness", "date_kyr_bp": 0}}]})"),
          "forcing.ice_change[0].change.date_kyr_bp",
          "dates a file without a time axis");
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"change": {"file": "history.nc",
                          "variable": "ice_thickness"}}]})"),
          "forcing.ice_change[0].change", "names a file with a time axis");
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"thickness": {"file": "static.nc",
                          "date_kyr_bp": 0}}]})"),
          "forcing.ice_change[0].thickness",
          "names a file without a time axis");
  refused(app::RunSolve,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "static.nc"}}]})"),
          "forcing.ice_change[0].thickness",
          "whose change from the state's instant is identically zero");
  // a thickness term on a state without a date, and a change taken as it
  // stands with the state at the window's end, refused once read
  refused(read,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"thickness": {"file": "history.nc",
                          "date_kyr_bp": 0}}]})",
                      still),
          "forcing.ice_change[0].thickness",
          "measures its change from the state's date");
  sl3dpp::test::WriteToyChange(dir_ / "change.nc", true);
  refused(read,
          timed("", kTimes,
                R"({"ice_change": [{"change": {"file": "change.nc"}}]})",
                R"({"at": "end", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc"}})"),
          "forcing.ice_change[0].change",
          "the ice changes are then thickness histories, `thickness` terms");
  for (const auto* cap :
       {R"({"cap_thickness": {"latitude": 70, "longitude": 0,
            "width_deg": 10, "thickness_m": 1}})",
        R"({"cap_fraction": {"latitude": 70, "longitude": 0,
            "width_deg": 10, "fraction": -0.1},
            "history_kyr_bp": [[2, 0], [0, 1]]})"}) {
    const auto kind = std::string{cap}.substr(2, std::string{cap}.find('"', 2) - 2);
    refused(read,
            timed("", kTimes, std::format(R"({{"ice_change": [{}]}})", cap),
                  R"({"at": "end", "topography": {"file": "history.nc"},
                      "ice": {"file": "history.nc"}})"),
            "forcing.ice_change[0]." + kind,
            "is a cap of ice change, which the state at the window's end");
  }
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})", "",
                      R"({"date_kyr_bp": 0,
                          "topography": {"file": "history.nc",
                                         "variable": "topo"},
                          "ice": {"file": "history.nc"}})"),
          "ocean.topography.variable",
          "names a variable \"topo\" that " + (dir_ / "history.nc").string() +
              " does not hold; it holds \"ice_thickness\", \"topography\"");
  refused(app::RunSolve,
          timed("", kTimes, "",
                R"({"at": "start", "topography": {"file": "history.nc"},
                    "ice": {"file": "history.nc", "variable": "thk"}})"),
          "ocean.ice.variable", "it holds \"ice_thickness\", \"topography\"");
  refused(app::RunSolve,
          timed("", kTimes,
                R"({"ice_change": [{"thickness": {"file": "history.nc",
                                                  "variable": "thk"}}]})"),
          "forcing.ice_change[0].thickness.variable",
          "it holds \"ice_thickness\", \"topography\"");
  // more ice taken than the state holds: a file's melt of four times the
  // history's, in the static problem and at a node of the run
  sl3dpp::test::WriteToyChange(dir_ / "melted.nc", false, 4.0);
  refused(app::RunSolve,
          fingerprint("", R"({"deck": "ball.deck"})",
                      R"({"ice_change": [{"change": {"file": "melted.nc"}}]})"),
          "forcing.ice_change", "removes more ice than the state holds, by ");
  sl3dpp::test::WriteToyChange(dir_ / "melt.nc", true, 4.0);
  for (const auto* what :
       {"removes more ice than the state holds, by ", " kyr BP)"}) {
    refused(app::RunSolve,
            timed("", kTimes,
                  R"({"ice_change": [{"change": {"file": "melt.nc"}}]})"),
            "forcing.ice_change", what);
  }
  // a change held through the run is present at its start: a load there
  // under frozen shorelines
  sl3dpp::test::WriteToyChange(dir_ / "held.nc", false);
  const auto held =
      std::string{R"({"ice_change": [{"change": {"file": "held.nc"}}]})"};
  {
    const auto frozen = app::RunSolve(
        Write("frozen.json", timed(R"( "rotation": "none",)", kTimes, held)));
    EXPECT_NE(frozen.find(std::format("; ice change change {}:"
                                      "ice_thickness_change, read by date; ",
                                      (dir_ / "held.nc").string())),
              std::string::npos)
        << frozen;
  }
  const auto prem8 =
      ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic);
  ASSERT_TRUE(io::WriteModuli(dir_ / "radial_moduli.nc", prem8, SurfaceGrid(4),
                              {}, {},
                              {.name = "radial moduli", .elementMetres = 500e3}));
  for (const auto* member :
       {R"("step_rtol": 1e-8)", R"("step": "conjugate_gradients")",
        R"("transform_degrees": [3])", R"("cache_mb": 4)"}) {
    const auto name = std::string{member}.substr(1, std::string{member}.find('"', 1) - 1);
    refused(app::RunSolve,
            fingerprint(std::format(R"( "solver": {{{}}},)", member),
                        std::format(R"({{"deck": "{}/prem8.elastic.deck",
                                       "format": "elastic",
                                       "moduli": {{"file": "radial_moduli.nc"}}}})",
                                    data)),
            "solver." + name,
            "3-D step, which the static problem on a radial Earth");
  }
}

TEST_F(AppsTest, KernelsMatchFiniteDifferencesOfTheForwardValue) {
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "state.nc");
  // the perturbed histories bracket a bump of ice laid on bare land at the
  // final date, 20 m either side of 20 m, so that neither takes ice the
  // land does not hold: the run is linear in the ice under frozen
  // shorelines, so the difference is the kernels' about any history
  WriteToyHistory(dir_ / "plus.nc", 40.0);
  WriteToyHistory(dir_ / "minus.nc", 0.0);
  auto config_with = [&](const std::string& history, const std::string& name,
                         const std::string& extra) {
    return Write(name, std::string{R"({
      "problem": "quasi_static",
      "model": {"deck": "ball.deck"},
      "discretisation": {"lmax": 8},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 9},
      "ocean": {"at": "start", "topography": {"file": "state.nc"},
                "ice": {"file": ")"} +
                           history + R"("}},
      "forcing": {"ice_change": [{"thickness": {"file": ")" +
                           history + R"("}}]},
      "rotation": "full", "moments": "hydrostatic",
      "kernels": {
        "method": "discrete",
        "parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0},
          {"type": "stokes_coefficient", "name": "j2", "degree": 2,
           "order": 0, "date_kyr_bp": 0.0},
          {"type": "rotation", "name": "lod", "component": "z",
           "date_kyr_bp": 0.0},
          {"type": "decay_time", "name": "rebound", "latitude": 80.0,
           "longitude": 0.0, "width_deg": 15.0, "window_start_kyr_bp": 1.0,
           "window_end_kyr_bp": 0.0, "time_uncertainty_kyr": 0.0,
           "value_uncertainty_m": 0.1}],
        "output": ")" + extra +
                           R"("}})");
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

  auto summary = app::RunKernel(
      config_with("state.nc", "kernels.json", "out_kernels"));
  EXPECT_NE(summary.find("4 measurement(s)"), std::string::npos);
  // what the config was taken to mean and the inputs as read, in the
  // summary and the values' header
  const auto state = (dir_ / "state.nc").string();
  EXPECT_TRUE(summary.starts_with(std::format(
      "1-D, viscoelastic, 9 nodes 2 to 0 kyr BP, shorelines frozen, rotation "
      "full, moments hydrostatic; ice {}:ice_thickness (the window's start, 2 "
      "kyr BP), topography {}:topography (the window's start, 2 kyr BP); ice "
      "change thickness {}:ice_thickness, from the state's instant to each "
      "node's date; wrote 4 measurement(s) of kernels to ",
      state, state, state)))
      << summary;
  EXPECT_TRUE(summary.ends_with(" (ice, viscosity; discrete adjoint)"))
      << summary;
  const auto out = dir_ / "out_kernels";
  for (const auto name :
       {"site_ice_kernels.nc", "j2_ice_kernels.nc", "lod_ice_kernels.nc",
        "rebound_ice_kernels.nc", "site_viscosity_kernel.txt",
        "measurements.txt"}) {
    EXPECT_TRUE(fs::exists(out / name)) << name;
  }
  {
    auto in = std::ifstream(out / "measurements.txt");
    const auto text = std::string(std::istreambuf_iterator<char>(in), {});
    EXPECT_NE(text.find(std::format(
                  "\n# ice_change thickness {}:ice_thickness, from the "
                  "state's instant to each node's date\n",
                  state)),
              std::string::npos)
        << text;
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
      app::RunKernel(config_with("plus.nc", "plus.json", "out_plus")));
  static_cast<void>(
      app::RunKernel(config_with("minus.nc", "minus.json", "out_minus")));
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
  const auto body = Prem8() + ", " + Window(4) + R"(, "rotation": "full")";
  const auto measurement = std::string{R"(
    "parameters": ["ice", "viscosity"],
    "measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
       "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}])"};
  const auto config =
      Write("kernel3d.json", "{" + body + R"(, "kernels": {)" + measurement +
                                 R"(, "output": "out_kernel3d"}})");
  const auto summary = app::RunKernel(config);
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

  // the disk-backed trajectory: the recorded states written to the
  // trajectory store during the run and fetched back by the sweep give the
  // same kernels exactly, and the store stays for later sweeps
  const auto spilled_config = Write(
      "kernel3d_spill.json",
      "{" + body + R"(, "trajectory": {"record_stride": 2, "directory": "spill"},
    "kernels": {)" + measurement +
          R"(, "output": "out_kernel3d_spill"}})");
  const auto in_memory_config = Write(
      "kernel3d_mem.json", "{" + body +
                               R"(, "trajectory": {"record_stride": 2},
    "kernels": {)" + measurement +
                               R"(, "output": "out_kernel3d_mem"}})");
  static_cast<void>(app::RunKernel(in_memory_config));
  const auto spilled_summary = app::RunKernel(spilled_config);
  EXPECT_NE(spilled_summary.find("; the forward run recorded to the trajectory "
                                 "store "),
            std::string::npos)
      << spilled_summary;
  EXPECT_TRUE(fs::exists(dir_ / "spill" / "rank_0000" / "node_000000.state"));
  EXPECT_TRUE(fs::exists(dir_ / "spill" / "rank_0000" / "node_000002.state"));
  EXPECT_FALSE(fs::exists(dir_ / "spill" / "rank_0000" / "node_000001.state"));
  EXPECT_TRUE(io::TrajectoryStore::ManifestOf(dir_ / "spill").complete);
  const auto memory =
      pm::ReadSurfaceFile(dir_ / "out_kernel3d_mem" / "site_ice_kernels.nc");
  const auto disk =
      pm::ReadSurfaceFile(dir_ / "out_kernel3d_spill" / "site_ice_kernels.nc");
  // one process, the same sweep, the states read back exactly: bitwise
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

TEST_F(AppsTest, ContinuousKernelsStateTheirStepsAgainstTheShortestRelaxation) {
  // `kernel` by the continuous sweep through time states the run's longest
  // step over the shortest relaxation time of the model as built, and
  // where that time is (RelaxationSite): in its summary, in the header of
  // measurements.txt and in its kernel files' metadata. By the discrete
  // sweep, the default, and in the static problem, which has no steps, it
  // states the method alone. On the homogeneous ball (mu = 5500^3 Pa, eta
  // = 1e21 Pa s) the time is eta / mu at every point, the ties going to the
  // innermost, against steps of 0.5 kyr; on its own clock in years the soft
  // ball's (1e19 Pa s) against 4 yr; under the prem8 deck's lateral
  // viscosity the shortest over the volume, its place on the sphere named,
  // the field files' metadata holding it too.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  WriteToyHistory(dir_ / "history.nc");
  const auto mu = 5500.0 * 5500.0 * 5500.0;
  const auto kernels = [](const std::string& method, const std::string& name,
                          const std::string& parameters,
                          const std::string& measurement) {
    return std::format(
        R"("kernels": {{{}"parameters": {}, "measurements": [{}],
          "output": "out/{}"}})",
        method.empty() ? std::string{}
                       : std::format(R"("method": "{}", )", method),
        parameters, measurement, name);
  };
  constexpr auto kSite = R"({"type": "relative_sea_level_cap", "name": "site",
      "latitude": -30, "longitude": 10, "width_deg": 25})";
  const auto config = [&](const std::string& name, const std::string& body,
                          const std::string& method,
                          const std::string& parameters,
                          const std::string& measurement) {
    return Write(name + ".json", "{" + body + ", " +
                                     kernels(method, name, parameters,
                                             measurement) +
                                     "}");
  };
  const auto out = [&](const std::string& name, const std::string& file) {
    return dir_ / "out" / name / file;
  };
  const auto number = [](const pm::SurfaceFile& file, const std::string& key) {
    return std::get<std::vector<double>>(file.meta.at(key)).front();
  };
  const auto header = [&](const std::string& name) {
    auto in = std::ifstream(out(name, "measurements.txt"));
    for (std::string line; std::getline(in, line);) {
      if (line.starts_with("# adjoint ")) return line;
    }
    return std::string{};
  };
  constexpr auto kRatio = "longest_step_over_shortest_relaxation_time";

  // the ball, dated, 0.5 kyr steps
  const auto ball =
      std::string{kBall} + ", " + Window(5) + R"(, "rotation": "none")";
  const auto summary = app::RunKernel(
      config("continuous", ball, "continuous", R"(["ice", "viscosity"])", kSite));
  const auto tau = 1e21 / mu;
  const auto ratio = 0.5 * app::KYR / tau;
  const auto clause = std::format(
      "(ice, viscosity; continuous adjoint, its longest step {:.3g} times the "
      "shortest relaxation time ({:.3g} kyr, at ",
      ratio, tau / app::KYR);
  EXPECT_NE(summary.find(clause), std::string::npos) << summary;
  EXPECT_NE(summary.find("in layer 0, 0 to 6371 km deep))"), std::string::npos)
      << summary;
  EXPECT_EQ(summary.find("latitude"), std::string::npos) << summary;
  const auto file = pm::ReadSurfaceFile(out("continuous", "site_ice_kernels.nc"));
  EXPECT_EQ(std::get<std::string>(file.meta.at("adjoint_method")),
            "continuous");
  // the window's steps and the deck's velocities, each formed otherwise:
  // to rounding (measured 5.1e-16 and 4.8e-16)
  const auto ratioGap = Gap(number(file, kRatio), ratio);
  const auto timeGap = Gap(number(file, "shortest_relaxation_time_s"), tau);
  RecordProperty("ratio_gap", std::format("{:.2e}", ratioGap));
  RecordProperty("time_gap", std::format("{:.2e}", timeGap));
  EXPECT_LT(ratioGap, 1e-13);
  EXPECT_LT(timeGap, 1e-13);
  // the innermost memory point, in the element at the centre (measured
  // 500 km from it: the graded mesh's elements widen with depth)
  EXPECT_LT(number(file, "shortest_relaxation_radius_m"), 1e6);
  EXPECT_NE(std::get<std::string>(file.meta.at("shortest_relaxation_at"))
                .find("in layer 0, 0 to 6371 km deep"),
            std::string::npos);
  EXPECT_TRUE(header("continuous").starts_with(
      "# adjoint continuous, its longest step "))
      << header("continuous");

  // the default: the discrete sweep, the method alone
  const auto discrete = app::RunKernel(
      config("discrete", ball, "", R"(["ice", "viscosity"])", kSite));
  EXPECT_NE(discrete.find("(ice, viscosity; discrete adjoint)"),
            std::string::npos)
      << discrete;
  const auto plain = pm::ReadSurfaceFile(out("discrete", "site_ice_kernels.nc"));
  EXPECT_EQ(std::get<std::string>(plain.meta.at("adjoint_method")), "discrete");
  EXPECT_FALSE(plain.meta.contains(kRatio));
  EXPECT_EQ(header("discrete"), "# adjoint discrete");

  // on its own clock in years: the soft ball's time against a step of 4 yr
  constexpr auto kUplift = R"({"type": "displacement_point", "name": "uplift",
      "latitude": 55, "longitude": -80, "time": 6})";
  const auto clocked = std::format(
      R"("problem": "quasi_static", "model": {{"deck": "soft.deck"}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [0, 1, 2.5, 3, 6, 10]}},
      "forcing": {{"load": [{}]}}, "rotation": "none")",
      kCap);
  const auto years =
      app::RunKernel(config("clocked", clocked, "continuous", R"(["load"])",
                            kUplift));
  const auto soft = 1e19 / mu;
  const auto year = app::KYR / 1e3;
  EXPECT_NE(years.find(std::format("(load; continuous adjoint, its longest "
                                   "step {:.3g} times the shortest relaxation "
                                   "time ({:.3g} yr, at ",
                                   4.0 * year / soft, soft / year)),
            std::string::npos)
      << years;
  const auto load = pm::ReadSurfaceFile(out("clocked", "uplift_load_kernels.nc"));
  // to rounding (measured 2.1e-16)
  const auto clockedGap = Gap(number(load, kRatio), 4.0 * year / soft);
  RecordProperty("clocked_gap", std::format("{:.2e}", clockedGap));
  EXPECT_LT(clockedGap, 1e-13);

  // the static problem: no steps
  const auto still = app::RunKernel(config(
      "static", std::string{kBall} + R"(, "problem": "static",
        "forcing": {"load": [)" + std::string{kCap} + R"(]}, "rotation": "none")",
      "continuous", R"(["load"])",
      R"({"type": "displacement_point", "name": "uplift", "latitude": 55,
          "longitude": -80})"));
  EXPECT_NE(still.find("(load; continuous adjoint)"), std::string::npos)
      << still;
  EXPECT_FALSE(pm::ReadSurfaceFile(out("static", "uplift_load_kernels.nc"))
                   .meta.contains(kRatio));

  // 3-D, under the prem8 deck's lateral viscosity: 2/3 kyr steps against
  // the shortest time over the volume, and where on the sphere
  const auto lateral = app::RunKernel(config(
      "lateral", Prem8() + ", " + Window(4) + R"(, "rotation": "none")",
      "continuous", R"(["ice", "viscosity"])", kSite));
  EXPECT_NE(lateral.find(", at latitude "), std::string::npos) << lateral;
  const auto surface = pm::ReadSurfaceFile(out("lateral", "site_ice_kernels.nc"));
  const auto steps = number(surface, kRatio);
  // the ratio times the time is the longest step, to rounding (measured
  // 0)
  const auto lateralGap =
      Gap(steps * number(surface, "shortest_relaxation_time_s"),
          2.0 / 3.0 * app::KYR);
  const auto& where =
      std::get<std::string>(surface.meta.at("shortest_relaxation_at"));
  RecordProperty("lateral_gap", std::format("{:.2e}", lateralGap));
  RecordProperty("lateral_at", where);
  EXPECT_LT(lateralGap, 1e-13);
  EXPECT_NE(where.find(", at latitude "), std::string::npos) << where;
  // the field file's numeric metadata, the same numbers
  const auto field =
      pm::ReadModelFile(out("lateral", "site_viscosity_kernels.nc"));
  EXPECT_EQ(std::get<std::vector<double>>(field.Meta().at(kRatio)).front(),
            steps);
}

// A measurement is held to the run when the config is read: a date off the
// times grid, a coefficient beyond the band or a decay window too short for
// the fit costs a read, not the forward run before the sweeps.
TEST_F(AppsTest, KernelMeasurementsAreCheckedWhenTheConfigIsRead) {
  const auto config = [&](const std::string& measurement) {
    return Write("kernel_check.json", std::string{R"({
      "model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8}, )"} +
                                          Window(5) + R"(,
      "kernels": {"parameters": ["ice", "viscosity"],
                  "measurements": [)" + measurement + R"(],
      "output": "out"}})");
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
  refuse(R"({"type": "relative_sea_level_point", "latitude": 10, "longitude": 20,
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
  EXPECT_EQ(input.kernels.measurements.size(), 2u);
}

TEST_F(AppsTest, The1DRunsRefuseThe3DStep) {
  // The solver members of the 3-D step (its closure, budget, transforms
  // and cache) are refused by a run on a radial model: solve's 1-D run
  // and the kernels of one.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body =
      std::string{R"("model": {"deck": "ball.deck"},
                     "discretisation": {"lmax": 4}, )"} + Window(3);
  for (const auto& member : {std::string{R"("step_rtol": 1e-8)"},
                             std::string{R"("step": "fixed_point")"},
                             std::string{R"("cache_mb": 4)"}}) {
    const auto name = member.substr(1, member.find('"', 1) - 1);
    const auto solver = std::format(R"(, "solver": {{{}}})", member);
    ExpectRefused(
        app::RunSolve,
        Write("sea_level_1d.json",
              "{" + body + solver + R"(, "output": {"surface": "out.nc"}})"),
        "solver." + name, "3-D step, which a 1-D run");
    ExpectRefused(
        app::RunKernel,
        Write("kernel.json",
              "{" + body + solver + R"(, "kernels": {
                  "parameters": ["ice", "viscosity"], "measurements": [
                  {"type": "rotation", "component": "z"}], "output": "out"}})"),
        "solver." + name, "3-D step, which a 1-D run");
  }
}

TEST_F(AppsTest, TransformDegreesAreReadAndChecked) {
  const auto config = [&](std::string_view degrees) {
    return Write("transform.json",
                 std::string{R"({
      "model": {"deck": "ball.deck"}, "discretisation": {"lmax": 4}, )"} +
                     Window(3) + R"(,
      "solver": {"transform_degrees": )" +
                     std::string{degrees} +
                     R"(}, "output": {"surface": "out.nc"}})");
  };
  EXPECT_EQ(
      app::ReadSolveInput(config("[2, 3]")).run.solver.transform_degrees,
      (std::vector<int>{2, 3}));
  for (const auto* bad : {"[5]", "[1]", "[2.5]"}) {
    EXPECT_THROW(
        {
          try {
            static_cast<void>(app::ReadSolveInput(config(bad)));
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
  // solve's and kernel's solver.cache_mb: a count of MiB that
  // reaches the operator's options as bytes, 0 or absent leaving the
  // library's default
  static_cast<void>(Deck());
  const auto seaLevel = [&](std::string_view member) {
    return Write("cache.json", std::string{R"({
      "model": {"deck": "ball.deck"}, "discretisation": {"lmax": 4}, )"} +
                                   Window(3) + R"(,
      "solver": {)" + std::string{member} +
                                   R"(}, "output": {"surface": "out.nc"}})");
  };
  const auto absent = app::ReadSolveInput(seaLevel(""));
  const auto zero = app::ReadSolveInput(seaLevel(R"("cache_mb": 0)"));
  const auto set = app::ReadSolveInput(seaLevel(R"("cache_mb": 32)"));
  EXPECT_EQ(absent.run.solver.cache_mb, 0u);
  EXPECT_EQ(zero.run.solver.cache_mb, 0u);
  EXPECT_EQ(set.run.solver.cache_mb, 32u);
  const auto earth =
      app::BuildEarth(absent.run, planetmodel::ReadJsonFile(seaLevel("")));
  const auto rheology = Rheology::Create(earth.material, {});
  ASSERT_TRUE(rheology);
  const auto library = Viscoelastic3D::Options{}.cache_bytes;
  EXPECT_EQ(app::OperatorOptionsOf(absent.run, earth, *rheology).cache_bytes,
            library);
  EXPECT_EQ(app::OperatorOptionsOf(zero.run, earth, *rheology).cache_bytes,
            library);
  EXPECT_EQ(app::OperatorOptionsOf(set.run, earth, *rheology).cache_bytes,
            std::size_t{32} << 20);

  // refused naming the member: negative, fractional, or not a number
  for (const auto& [bad, what] :
       {std::pair{"-1", "is not a cache size in MiB"},
        std::pair{"2.5", "is not an integer"},
        std::pair{R"("32")", "expected a number"}}) {
    const auto member = std::string{R"("cache_mb": )"} + bad;
    ExpectRefused(app::ReadSolveInput, seaLevel(member), "cache_mb", what);
  }
}

/// The data lines of a text table (those not starting with '#').
std::vector<std::string> DataLines(const fs::path& path) {
  auto in = std::ifstream(path);
  auto out = std::vector<std::string>{};
  for (auto line = std::string{}; std::getline(in, line);) {
    if (!line.starts_with('#')) out.push_back(line);
  }
  return out;
}

/// The kernel variables of a model file, by part and name.
std::map<std::pair<std::size_t, std::string>, std::vector<double>> KernelValues(
    const fs::path& path) {
  const auto file = planetmodel::ReadModelFile(path);
  auto out = std::map<std::pair<std::size_t, std::string>, std::vector<double>>{};
  for (auto p = std::size_t{0}; p < file.Parts().size(); ++p) {
    for (const auto& [name, variable] : file.Parts()[p].variables) {
      if (name.starts_with("K_")) out[{p, name}] = variable.values;
    }
  }
  return out;
}

TEST_F(AppsTest, KernelOfARadialRunWritesLateralKernels) {
  // `kernels.lateral` on a radial run: the kernels in the viscosity and the
  // moduli as fields over the memory nodes' grids, written as a 3-D run's
  // are, beside the radial profiles — which are, data line for data line,
  // those of the run without the member, by either sweep; from a
  // recording into the trajectory store, and swept again from it with the
  // same values to the bit; the grids' members of `solver` taken, which a
  // radial run refuses otherwise, and left out of the run's hashes, so
  // that the store serves the sweep at other transforms (a lateral run's
  // store, at its own alone). Refused: `lateral` true naming neither the
  // viscosity nor the moduli, and false on a model with lateral structure.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{R"("model": {"deck": "ball.deck"},
    "discretisation": {"lmax": 4}, )"} + Window(3) + R"(, "rotation": "axial")";
  const auto kernels = [&](const std::string& extra, const std::string& output) {
    return std::string{R"("kernels": {)"} + extra + R"(
      "parameters": ["ice", "viscosity", "moduli"],
      "measurements": [
        {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}],
      "output": ")" + output + R"("})";
  };
  const auto lateral = Write(
      "lateral.json", "{" + body + R"(, "trajectory": {"directory": "store"},
    "solver": {"transform_degrees": [3], "cache_mb": 1}, )" +
                          kernels(R"("lateral": true,)", "out_lateral") + "}");
  const auto radial =
      Write("radial.json", "{" + body + ", " + kernels("", "out_radial") + "}");
  const auto first = app::RunKernel(lateral);
  EXPECT_NE(first.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << first;
  static_cast<void>(app::RunKernel(radial));
  const auto out = dir_ / "out_lateral";
  // the same radial sweep, the lateral fields read off it: bitwise
  for (const auto* name : {"site_viscosity_kernel.txt", "site_moduli_kernel.txt"}) {
    EXPECT_EQ(DataLines(out / name), DataLines(dir_ / "out_radial" / name)) << name;
  }
  EXPECT_FALSE(fs::exists(dir_ / "out_radial" / "site_viscosity_kernels.nc"));
  const auto viscosity = KernelValues(out / "site_viscosity_kernels.nc");
  const auto moduli = KernelValues(out / "site_moduli_kernels.nc");
  auto held = std::set<std::string>{};
  for (const auto* file : {&viscosity, &moduli}) {
    for (const auto& [key, values] : *file) {
      if (!key.second.ends_with("_poles")) held.insert(key.second);
    }
  }
  EXPECT_EQ(held, (std::set<std::string>{"K_ln_eta_0", "K_kappa", "K_mu"}));
  // by the continuous sweep (the runs in memory) likewise: its profiles
  // its radial run's, data line for data line
  static_cast<void>(app::RunKernel(Write(
      "lateral_continuous.json",
      "{" + body + R"(, "solver": {"transform_degrees": [3], "cache_mb": 1}, )" +
          kernels(R"("lateral": true, "method": "continuous",)",
                  "out_lateral_continuous") +
          "}")));
  static_cast<void>(app::RunKernel(Write(
      "radial_continuous.json",
      "{" + body + ", " +
          kernels(R"("method": "continuous",)", "out_radial_continuous") +
          "}")));
  for (const auto* name : {"site_viscosity_kernel.txt", "site_moduli_kernel.txt"}) {
    EXPECT_EQ(DataLines(dir_ / "out_lateral_continuous" / name),
              DataLines(dir_ / "out_radial_continuous" / name))
        << name;
  }
  EXPECT_FALSE(
      KernelValues(dir_ / "out_lateral_continuous" / "site_viscosity_kernels.nc")
          .empty());
  // swept again from the complete store, the same to the bit
  fs::rename(out, dir_ / "out_lateral_first");
  const auto again = app::RunKernel(lateral);
  EXPECT_NE(again.find("; the forward run read back from the trajectory store "),
            std::string::npos)
      << again;
  EXPECT_EQ(KernelValues(out / "site_viscosity_kernels.nc"), viscosity);
  EXPECT_EQ(KernelValues(out / "site_moduli_kernels.nc"), moduli);
  // The transforms' degrees are the sweep's alone on a radial run: they
  // leave the run's hashes as they are, and the store recorded at one
  // serves a sweep at another, whose kernels are, to the bit, those of
  // the run at that one in memory. A lateral run's hashes change with
  // them, and its store serves the run it recorded alone.
  const auto atDegrees = [&](const std::string& name, const std::string& model,
                             const std::string& degrees,
                             const std::string& trajectory) {
    return Write(name + ".json",
                 "{" + model + trajectory +
                     R"(, "solver": {"transform_degrees": )" + degrees + "}, " +
                     kernels(R"("lateral": true,)", "out_" + name) + "}");
  };
  constexpr auto kStore = R"(, "trajectory": {"directory": "store"})";
  const auto hashes = [&](const fs::path& path, bool lateralRun) {
    const auto input = app::ReadKernelInput(path);
    auto hash = app::Fnv1a{};
    app::AddRunMembers(hash, input.run, lateralRun);
    return std::pair{hash.Value(),
                     app::TrajectoryHashOf(input.run, lateralRun)};
  };
  const auto two = atDegrees("two", body, "[2]", kStore);
  EXPECT_EQ(hashes(two, false), hashes(lateral, false));
  const auto fromStore = app::RunKernel(two);
  EXPECT_NE(
      fromStore.find("; the forward run read back from the trajectory store "),
      std::string::npos)
      << fromStore;
  static_cast<void>(app::RunKernel(atDegrees("two_memory", body, "[2]", "")));
  // the same forward states and sweep transforms in one process: bitwise
  for (const auto* name :
       {"site_viscosity_kernels.nc", "site_moduli_kernels.nc"}) {
    EXPECT_EQ(KernelValues(dir_ / "out_two" / name),
              KernelValues(dir_ / "out_two_memory" / name))
        << name;
  }
  const auto lateralBody = Prem8() + ", " + Window(3);
  constexpr auto kLateralStore =
      R"(, "trajectory": {"directory": "lateral_store"})";
  const auto three =
      atDegrees("lateral_three", lateralBody, "[3]", kLateralStore);
  const auto twoLateral =
      atDegrees("lateral_two", lateralBody, "[2]", kLateralStore);
  EXPECT_NE(hashes(three, true).first, hashes(twoLateral, true).first);
  EXPECT_NE(hashes(three, true).second, hashes(twoLateral, true).second);
  static_cast<void>(app::RunKernel(three));
  ExpectRefused(app::RunKernel, twoLateral, "trajectory.directory",
                "holds the trajectory store of another run");

  ExpectRefused(app::RunKernel,
                Write("bad_parameters.json",
                      "{" + body + R"(, "kernels": {"lateral": true,
      "parameters": ["ice"],
      "measurements": [{"type": "rotation", "name": "lod", "component": "z"}],
      "output": "out_bad"}})"),
                "kernels.lateral", "names neither");
  ExpectRefused(app::RunKernel,
                Write("bad_transforms.json",
                      "{" + body + R"(, "solver": {"transform_degrees": [3]}, )" +
                          kernels("", "out_bad") + "}"),
                "solver.transform_degrees", "3-D step, which a 1-D run");
  const auto lateralModel = Prem8() + ", " + Window(3);
  ExpectRefused(app::RunKernel,
                Write("bad_model.json",
                      "{" + lateralModel + ", " +
                          kernels(R"("lateral": false,)", "out_bad") + "}"),
                "kernels.lateral", "is false, and the model is 3-D");
}

TEST_F(AppsTest, TransformDegreesOnARadialRunAreTheLateralKernels) {
  // The grids' members of the `solver` block (`transform_degrees`,
  // `cache_mb`) on a radial model whose config asks for lateral kernels
  // (`kernels.lateral` true): `solve` takes them and leaves them unused,
  // as it does the `kernels` block, its run the run without them to the
  // bit, in the static problem and through time; the static problem's
  // `kernel` sweeps its lateral kernels on the grids they cut, its moduli
  // kernels then those of the coarser grids at depth. Without lateral
  // kernels a radial run refuses them, `solve` and `kernel` alike.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto staticRun = [&](const std::string& name, const std::string& solver,
                             bool lateral) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static", "model": {{"deck": "ball.deck"}},
      "discretisation": {{"lmax": 8}}, "rotation": "none",
      "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
          "width_deg": 20, "amplitude_kg_m2": 1.0e6}}}}]}},
      {}"output": {{"surface": "out/{}.nc"}},
      "kernels": {{"lateral": {}, "parameters": ["load", "moduli"],
        "measurements": [{{"type": "displacement_cap", "name": "uplift",
          "latitude": 55, "longitude": -80, "width_deg": 10}}],
        "output": "out/{}_kernels"}}}})",
                                             solver, name, lateral, name));
  };
  constexpr auto kCut =
      R"("solver": {"transform_degrees": [2, 4], "cache_mb": 1}, )";
  static_cast<void>(app::RunSolve(staticRun("plain", "", true)));
  static_cast<void>(app::RunSolve(staticRun("cut", kCut, true)));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  // a radial run's transforms take neither member: bitwise
  for (const auto& [name, variable] : read("plain").variables) {
    EXPECT_EQ(read("cut").variables.at(name).values, variable.values) << name;
  }
  static_cast<void>(app::RunKernel(staticRun("plain", "", true)));
  static_cast<void>(app::RunKernel(staticRun("cut", kCut, true)));
  const auto plain =
      KernelValues(dir_ / "out" / "plain_kernels" / "uplift_moduli_kernels.nc");
  const auto cut =
      KernelValues(dir_ / "out" / "cut_kernels" / "uplift_moduli_kernels.nc");
  EXPECT_FALSE(plain.empty());
  EXPECT_NE(cut, plain);
  ExpectRefused(app::RunSolve, staticRun("radial", kCut, false),
                "solver.transform_degrees", "3-D step");
  ExpectRefused(app::RunKernel, staticRun("radial", kCut, false),
                "solver.transform_degrees", "3-D step");
  // through time
  const auto body = std::string{R"("model": {"deck": "ball.deck"},
    "discretisation": {"lmax": 4}, )"} +
                    Window(3) + R"(, "rotation": "axial")";
  const auto timed = [&](const std::string& name, const std::string& solver,
                         bool lateral) {
    return Write(name + ".json",
                 "{" + body + ", " + solver +
                     std::format(R"("output": {{"surface": "out/{}.nc"}},
      "kernels": {{"lateral": {}, "parameters": ["viscosity"],
        "measurements": [{{"type": "relative_sea_level_cap", "name": "site",
          "latitude": -30.0, "longitude": 10.0, "width_deg": 25.0,
          "date_kyr_bp": 0.0}}],
        "output": "out/{}_kernels"}}}})",
                                 name, lateral, name));
  };
  static_cast<void>(app::RunSolve(timed("timed_plain", "", true)));
  static_cast<void>(app::RunSolve(timed("timed_cut", kCut, true)));
  // a radial run's transforms take neither member: bitwise
  for (const auto& [name, variable] : read("timed_plain").variables) {
    EXPECT_EQ(read("timed_cut").variables.at(name).values, variable.values)
        << name;
  }
  ExpectRefused(app::RunSolve, timed("timed_radial", kCut, false),
                "solver.transform_degrees", "3-D step, which a 1-D run");
}

TEST_F(AppsTest, Kernel1DRecordsItsTrajectoryToo) {
  // the 1-D run is known from the model block, the deck's; its recording
  // goes to the trajectory store like a 3-D one's, every state of it
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = Write("kernel1d_spill.json", std::string{R"({
    "model": {"deck": "ball.deck"},
    "discretisation": {"lmax": 4}, )"} + Window(3) +
                                                       R"(,
    "rotation": "none", "trajectory": {"directory": "spill"},
    "kernels": {
      "parameters": ["ice", "viscosity"],
      "measurements": [
        {"type": "rotation", "name": "lod", "component": "z",
         "date_kyr_bp": 0.0}],
      "output": "out"}})");
  const auto summary = app::RunKernel(config);
  EXPECT_NE(summary.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << summary;
  const auto manifest = io::TrajectoryStore::ManifestOf(dir_ / "spill");
  EXPECT_TRUE(manifest.complete);
  EXPECT_EQ(manifest.dimension, 1);
  EXPECT_EQ(manifest.kept, (std::vector<bool>{true, true, true}));
}

/// A point source and a tide coefficient as `seismic` and `tide` terms,
/// their timing members appended by the caller.
/// Numbers as a JSON list, each to the full precision of a double.
std::string ListOf(const std::vector<double>& values) {
  auto out = std::string{"["};
  for (auto k = std::size_t{0}; k < values.size(); ++k) {
    out += std::format("{}{:.17g}", k == 0 ? "" : ", ", values[k]);
  }
  return out + "]";
}

TEST_F(AppsTest, SolveOnItsOwnClockIsTheLibrarysRunThroughTime) {
  // A run counting forwards on its own clock (`times.unit` with `at`) on a
  // body without an ocean and without rotational feedback is the library's
  // evolution on the same uneven nodes under the same forcing, its times
  // taken as given: a cap with an amplitude history (`history`), a tide
  // coefficient with its own and a point source from an onset (`onset`, a
  // node: the library's grid repeats it, the source absent from its left
  // limit), node by node to rounding (about 4e-15 here: in kyr the two step
  // through the same times to the bit), the tangential and toroidal
  // potentials included. The same run in years, every time a thousand
  // times larger, agrees with it to the rounding of its node times, which
  // a step's response amplifies (about 3e-12 here: on this ball a one-ulp
  // change of a node moves the fields by up to 7e-13), and so does a run
  // of forty days in days with its run in kyr (to about 1e-16 here); and
  // the dated run of the same listed nodes, their dates counted back from
  // 3 kyr BP, to the rounding of its node times likewise. The file's time
  // axis is the run's clock in seconds, its metadata the nodes in the
  // run's unit (`times`) and the unit's name (`time_unit`), the terms are
  // described in the unit, and the summary states the window in it. The
  // exchange tables in kyr hold the expansion of each node's fields and
  // the node's time, `time_kyr` the node itself, as the dated run's do,
  // and in years name the node's time in the unit (`time_yr`).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  // the run on the run's own clock in `unit`, its times `scale` times kyr's
  const auto forwards = [&](const std::string& name, const std::string& unit,
                            const std::vector<double>& at, double scale) {
    auto times = at;
    for (auto& t : times) t *= scale;
    const auto config = Write(
        name + ".json",
        std::format(R"({{{}, "problem": "quasi_static",
      "times": {{"unit": "{}", "at": {}}},
      "forcing": {{"load": [{}, "history": [[0, 0.5], [{:.17g}, 1.5]]}}],
                  "tide": [{}"history": [[0, 1], [{:.17g}, -1]]}}],
                  "seismic": [{}"onset": {:.17g}}}]}},
      "rotation": "none", "output": {{"surface": "out/{}.nc",
                                     "exchange": "out/{}_{{node}}.txt"}}}})",
                    kBall, unit, ListOf(times), kCapTerm, 1.5 * scale,
                    kTideOpen, 2.0 * scale, kPointOpen, 0.5 * scale, name,
                    name));
    return app::RunSolve(config);
  };
  const auto nodes = std::vector<double>{0.0, 0.2, 0.5, 1.1, 2.0, 3.0};
  static_cast<void>(forwards("kyr", "kyr", nodes, 1.0));
  const auto summary = forwards("yr", "yr", nodes, 1000.0);
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 6 nodes 0 to 3000 yr on the run's clock, no "
      "ocean, rotation none; load (1 term) + tide (1 coefficient) + seismic "
      "(1 source); wrote "))
      << summary;
  // the dated run of the same nodes, from 3 kyr BP, its terms' times
  // turned to dates
  static_cast<void>(app::RunSolve(Write("dated.json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"at_kyr_bp": [3, 2.8, 2.5, 1.9, 1, 0]}},
      "forcing": {{"load": [{}, "history_kyr_bp": [[3, 0.5], [1.5, 1.5]]}}],
                  "tide": [{}"history_kyr_bp": [[3, 1], [1, -1]]}}],
                  "seismic": [{}"onset_kyr_bp": 2.5}}]}},
      "rotation": "none", "output": {{"surface": "out/dated.nc",
                                     "exchange": "out/dated_{{node}}.txt"}}}})",
                                                          kBall, kCapTerm,
                                                          kTideOpen,
                                                          kPointOpen))));
  const auto want = LibraryRunWithoutAnOcean(
      dir_ / "ball.deck",
      [&](double scale) {
        // a run's own clock in kyr: each node times the unit over the scale
        auto out = std::vector<double>{};
        for (const auto t : nodes) out.push_back(t * (app::KYR / scale));
        return out;
      },
      [](double t) { return std::clamp(0.5 + t / 1.5, 0.5, 1.5); },
      [](double t) { return std::max(1.0 - t, -1.0); }, 0.5);
  const auto kyr = Output("kyr");
  const auto yr = Output("yr");
  const auto dated = Output("dated");
  auto worst = 0.0, units = 0.0, forms = 0.0;
  for (const auto* name : {"displacement", "tangential_potential",
                           "toroidal_potential", "potential"}) {
    const auto departure =
        Departure(kyr.variables.at(name).values, want.at(name));
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
    const auto scaled =
        Departure(yr.variables.at(name).values, kyr.variables.at(name).values);
    units = std::max(units, scaled);
    EXPECT_LT(scaled, 1e-10) << name;
    const auto byDate = Departure(dated.variables.at(name).values,
                                  kyr.variables.at(name).values);
    forms = std::max(forms, byDate);
    EXPECT_LT(byDate, 1e-10) << name;
  }
  RecordProperty("library_departure", std::format("{:.3e}", worst));
  RecordProperty("yr_departure", std::format("{:.3e}", units));
  RecordProperty("dated_departure", std::format("{:.3e}", forms));
  const SurfaceGrid grid(8);
  auto tables = 0.0;
  for (auto n = std::size_t{0}; n < 6; ++n) {
    const auto table = ReadExchange(dir_ / "out" / std::format("kyr_{}.txt", n));
    tables = std::max(
        {tables, Departure(table.U, grid.Expand(Slice(kyr, "displacement", n))),
         Departure(table.Phi, grid.Expand(Slice(kyr, "potential", n)))});
    EXPECT_EQ(table.scalars.at("time_kyr"), std::format("{:.17g}", nodes[n]));
    // the dated run's time from its start, 3 kyr less a date, to rounding
    EXPECT_LT(CompareTables(dir_ / "out" / std::format("dated_{}.txt", n),
                            dir_ / "out" / std::format("kyr_{}.txt", n),
                            {"date_kyr_bp", "time_kyr"}),
              1e-10);
    EXPECT_NEAR(std::stod(ReadExchange(dir_ / "out" /
                                       std::format("dated_{}.txt", n))
                              .scalars.at("time_kyr")),
                nodes[n], 1e-12);
    const auto inYears =
        ReadExchange(dir_ / "out" / std::format("yr_{}.txt", n)).scalars;
    EXPECT_FALSE(inYears.contains("time_kyr"));
    EXPECT_EQ(inYears.at("time_yr"),
              std::format("{:.17g}", std::vector{0.0, 200.0, 500.0, 1100.0,
                                                 2000.0, 3000.0}[n]));
  }
  EXPECT_LT(tables, 1e-12);
  RecordProperty("table_departure", std::format("{:.3e}", tables));

  // the file: the run's clock in seconds, the nodes in the unit
  ASSERT_TRUE(yr.time);
  ASSERT_EQ(yr.time->size(), nodes.size());
  for (auto n = std::size_t{0}; n < nodes.size(); ++n) {
    EXPECT_EQ((*yr.time)[n], nodes[n] * 1000.0 * (app::KYR / 1000.0)) << n;
  }
  EXPECT_EQ(std::get<std::string>(yr.meta.at("time_unit")), "yr");
  EXPECT_EQ(std::get<std::vector<double>>(yr.meta.at("times")),
            (std::vector<double>{0.0, 200.0, 500.0, 1100.0, 2000.0, 3000.0}));
  EXPECT_FALSE(yr.meta.contains("dates_kyr_bp"));
  EXPECT_EQ(std::get<std::string>(yr.meta.at("tide")),
            "tide degree 2 order 1, 1 + 0.5 i m^2 s^-2, history in yr [[0, "
            "1], [2000, -1]]");
  EXPECT_NE(std::get<std::string>(yr.meta.at("seismic")).find(", from 500 yr"),
            std::string::npos);
  EXPECT_NE(std::get<std::string>(yr.meta.at("load"))
                .find(", history in yr [[0, 0.5], [1500, 1.5]]"),
            std::string::npos);

  // forty days in days against the same run in kyr, the histories' times
  // and the onset the kyr run's scaled alike
  const auto days = std::vector<double>{0.0, 1.0, 3.0, 10.0, 40.0};
  static_cast<void>(forwards("days", "day", days, 1.0));
  static_cast<void>(forwards("days_in_kyr", "kyr", days, 86400.0 / app::KYR));
  const auto inDays = Output("days");
  const auto inKyr = Output("days_in_kyr");
  auto weeks = 0.0;
  for (const auto* name : {"displacement", "tangential_potential",
                           "toroidal_potential", "potential"}) {
    const auto departure = Departure(inDays.variables.at(name).values,
                                     inKyr.variables.at(name).values);
    weeks = std::max(weeks, departure);
    EXPECT_LT(departure, 1e-11) << name;
  }
  RecordProperty("day_departure", std::format("{:.3e}", weeks));
  EXPECT_EQ(std::get<std::string>(inDays.meta.at("time_unit")), "day");
}

TEST_F(AppsTest, SolveReadsAFieldOnItsOwnClockAndOnItsOwnAxis) {
  // A `field` load whose file has a time axis is read, in a run counting
  // forwards, on the run's clock at t - `time_origin` in the run's unit:
  // the file's axis counted back from the present, a cap ramped up over
  // the two kyr to the present, read from a time origin of 2 kyr at uneven
  // nodes (the file's last field held past it, `"outside_span": "hold"`),
  // is the cap term under the matching history ramped up over the
  // run's first two kyr, to rounding, and the same in years from 2000 yr
  // agrees with it to the rounding of its node times (about 4e-12;
  // SolveOnItsOwnClockIsTheLibrarysRunThroughTime); the summary lists the
  // field's input as read. In the static problem the file is read at
  // `date` in its `unit` on the file's own axis, the same load as at
  // `date_kyr_bp` of minus that date, to the bit in kyr, yr and days.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  const SurfaceGrid grid(kLmax);
  const auto field = app::CapField(grid, {60.0, -85.0},
                                   20.0 * std::numbers::pi / 180.0, 1.0e6);
  {
    auto file = pm::SurfaceFile{};
    file.grid = pm::AngularGrid(
        std::vector<double>(grid.CoLatitudes().begin(),
                            grid.CoLatitudes().end()),
        std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
        "gauss_legendre", kLmax,
        std::vector<double>(grid.CoLatitudeWeights().begin(),
                            grid.CoLatitudeWeights().end()));
    file.source = "sl3dpp tests";
    file.time = std::vector<double>{-2.0 * app::KYR, 0.0};
    auto values = std::vector<double>(field.size(), 0.0);
    values.insert(values.end(), field.begin(), field.end());
    file.variables.emplace(
        "load", pm::SurfaceField("load", values,
                                 {.support = "node",
                                  .time_interpolation = "linear",
                                  .dimensions = pm::Dimensions{.mass = 1,
                                                               .length = -2}}));
    pm::WriteSurfaceFile(dir_ / "ramp_bp.nc", file);
  }
  static_cast<void>(Run("cap", std::format(R"({},
      "times": {{"unit": "kyr", "at": [0, 0.5, 1.2, 2, 2.5, 3.5]}},
      "forcing": {{"load": [{}, "history": [[0, 0], [2, 1]]}}]}})",
                                           kBall, kCapTerm)));
  const auto onClock = [&](const std::string& name, const std::string& unit,
                           double scale) {
    auto at = std::vector<double>{0.0, 0.5, 1.2, 2.0, 2.5, 3.5};
    for (auto& t : at) t *= scale;
    return app::RunSolve(Write(
        name + ".json",
        std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"unit": "{}", "at": {}}},
      "forcing": {{"load": [{{"field": {{"file": "ramp_bp.nc",
                                         "time_origin": {:.17g},
                                         "outside_span": "hold"}}}}]}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                    kBall, unit, ListOf(at), 2.0 * scale, name)));
  };
  const auto summary = onClock("clock", "kyr", 1.0);
  EXPECT_NE(summary.find(std::format(
                "; load (1 term); field {}:load (time origin 2 kyr); wrote ",
                (dir_ / "ramp_bp.nc").string())),
            std::string::npos)
      << summary;
  static_cast<void>(onClock("years", "yr", 1000.0));
  const auto cap = Output("cap");
  const auto clock = Output("clock");
  const auto years = Output("years");
  auto worst = 0.0, units = 0.0;
  for (const auto* name : {"displacement", "tangential_potential", "potential",
                           "load"}) {
    const auto departure = Departure(clock.variables.at(name).values,
                                     cap.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-12) << name;
    // in years, to the rounding of the node times (the clock test above)
    const auto scaled = Departure(years.variables.at(name).values,
                                  clock.variables.at(name).values);
    units = std::max(units, scaled);
    EXPECT_LT(scaled, 1e-10) << name;
  }
  RecordProperty("cap_departure", std::format("{:.3e}", worst));
  RecordProperty("yr_departure", std::format("{:.3e}", units));
  EXPECT_EQ(std::get<std::string>(years.meta.at("load")),
            std::format("field {}:load, time origin 2000 yr, outside the "
                        "water budget",
                        (dir_ / "ramp_bp.nc").string()));

  // the static problem at a date on the file's own axis
  const auto still = [&](const std::string& name, const std::string& when) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        std::format(R"({{{},
      "problem": "static",
      "forcing": {{"load": [{{"field": {{"file": "ramp_bp.nc", {}}}}}]}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                    kBall, when, name))));
    return Output(name);
  };
  const auto dated = still("dated", R"("date_kyr_bp": 1.5)");
  EXPECT_EQ(std::get<std::string>(dated.meta.at("load")),
            std::format("field {}:load, at 1.5 kyr BP, outside the water "
                        "budget",
                        (dir_ / "ramp_bp.nc").string()));
  for (const auto& [name, when, text] :
       {std::tuple{"own_kyr", R"("date": -1.5, "unit": "kyr")", "-1.5 kyr"},
        std::tuple{"own_yr", R"("date": -1500, "unit": "yr")", "-1500 yr"},
        std::tuple{"own_day", R"("date": -547875, "unit": "day")",
                   "-547875 day"}}) {
    // the three dates are the same whole number of seconds: bitwise
    const auto own = still(name, when);
    for (const auto* field_name :
         {"load_outside_water_budget", "displacement", "tangential_potential",
          "potential", "relative_sea_level_change"}) {
      EXPECT_EQ(own.variables.at(field_name).values,
                dated.variables.at(field_name).values)
          << name << ", " << field_name;
    }
    EXPECT_EQ(std::get<std::string>(own.meta.at("load")),
              std::format("field {}:load, at {}, outside the water budget",
                          (dir_ / "ramp_bp.nc").string(), text));
  }
  // a quarter of the cap: the file's ramp at 1.5 kyr BP
  auto quarter = grid.Synthesise(grid.Expand(field));
  for (auto& v : quarter) v *= 0.25;
  EXPECT_LT(Departure(dated.variables.at("load_outside_water_budget").values,
                      quarter),
            1e-14);
}

TEST_F(AppsTest, SolveReadsADatedFieldAtEachNodesDate) {
  // A `field` load whose file has a time axis is read, in a dated run, at
  // each node's date on the calendar's axis by the file's own time rule:
  // a cap of water in the budget on the file's dates (none at 2.5 kyr BP,
  // the cap at 1.5, four tenths of it from 0.25 to the present, linear
  // between them; the file's last date held to the present, `"outside_span":
  // "hold"`), written on the run's own grid, is the cap term under
  // the history of the same dates, to rounding, beside the toy state's
  // ice history under frozen and under migrating shorelines; the summary
  // and the outputs' metadata say the file is read by date.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kLmax = 8;
  const SurfaceGrid grid(kLmax);
  {
    const auto cap = app::CapField(grid, {-20.0, 100.0},
                                   25.0 * std::numbers::pi / 180.0, 3000.0);
    auto file = pm::SurfaceFile{};
    file.grid = pm::AngularGrid(
        std::vector<double>(grid.CoLatitudes().begin(),
                            grid.CoLatitudes().end()),
        std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
        "gauss_legendre", kLmax,
        std::vector<double>(grid.CoLatitudeWeights().begin(),
                            grid.CoLatitudeWeights().end()));
    file.source = "sl3dpp tests";
    file.time =
        std::vector<double>{-2.5 * app::KYR, -1.5 * app::KYR, -0.25 * app::KYR};
    auto values = std::vector<double>{};
    for (const auto factor : {0.0, 1.0, 0.4}) {
      for (const auto v : cap) values.push_back(factor * v);
    }
    file.variables.emplace(
        "load", pm::SurfaceField(
                    "load", values,
                    {.support = "node",
                     .time_interpolation = "linear",
                     .dimensions = pm::Dimensions{.mass = 1, .length = -2}}));
    pm::WriteSurfaceFile(dir_ / "water.nc", file);
  }
  const auto config = [&](const std::string& name, const std::string& term,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "load": [{{{}, "water_budget": true,
                  "moves_solid_surface": false}}]}},
      "ocean": {{"at": "start", "shorelines": "{}",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, term, shorelines, name));
  };
  constexpr auto kCap = R"("cap": {"latitude": -20, "longitude": 100,
      "width_deg": 25, "amplitude_kg_m2": 3000},
      "history_kyr_bp": [[2.5, 0], [1.5, 1], [0.25, 0.4]])";
  constexpr auto kField =
      R"("field": {"file": "water.nc", "outside_span": "hold"})";
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto label = std::string{shorelines};
    static_cast<void>(app::RunSolve(config(label + "_cap", kCap, label)));
    const auto summary = app::RunSolve(config(label + "_field", kField, label));
    EXPECT_NE(summary.find(std::format(
                  "; load (1 term); field {}:load (each node's date); wrote ",
                  (dir_ / "water.nc").string())),
              std::string::npos)
        << summary;
    const auto cap = Output(label + "_cap");
    const auto field = Output(label + "_field");
    EXPECT_EQ(std::get<std::string>(field.meta.at("load")),
              std::format("field {}:load, read by date, in the water budget",
                          (dir_ / "water.nc").string()));
    auto worst = 0.0;
    for (const auto* name :
         {"load_in_water_budget", "relative_sea_level_change", "displacement",
          "tangential_potential", "potential"}) {
      const auto departure = Departure(field.variables.at(name).values,
                                       cap.variables.at(name).values);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-12) << label << ", " << name;
    }
    RecordProperty(label + "_cap_departure", std::format("{:.3e}", worst));
    if (label == "migrating") {
      // 0 or 1 by flotation: unequal only at a point on the threshold
      EXPECT_EQ(field.variables.at("ocean_function").values,
                cap.variables.at("ocean_function").values);
    }
    // the load at the nodes: half the cap at 2 kyr BP, the cap at 1.5, the
    // ramp down to four tenths at 0.25 kyr BP, held to the present
    const auto& applied = field.variables.at("load_in_water_budget").values;
    const auto size = grid.FieldSize();
    auto peak = 0.0;
    for (auto i = std::size_t{0}; i < size; ++i) {
      peak = std::max(peak, std::abs(applied[size + i]));
    }
    for (const auto& [node, factor] :
         {std::pair{0, 0.5}, std::pair{1, 1.0}, std::pair{2, 0.76},
          std::pair{3, 0.52}, std::pair{4, 0.4}}) {
      auto most = 0.0;
      for (auto i = std::size_t{0}; i < size; ++i) {
        most = std::max(
            most, std::abs(applied[static_cast<std::size_t>(node) * size + i]));
      }
      EXPECT_NEAR(most, factor * peak, 1e-12 * peak)
          << label << ", node " << node;
    }
  }
}

TEST_F(AppsTest, SolveTakesListedNodesDatedOrOnItsOwnClock) {
  // Dates listed in `at_kyr_bp` that are the evenly spaced ones give the
  // run of `start_kyr_bp` and `nodes` to the bit, under the toy history
  // with an ocean, and its metadata list the dates given. A genuinely
  // uneven dated run is the run counting forwards on its own clock over
  // the same nodes, the state read at the window's start date
  // (`ocean.date_kyr_bp` there), a cap melting a fifth of the polar ice
  // under its history and a change held through the run, under frozen and
  // migrating shorelines alike, to the rounding of the node times, formed
  // from dates in the one and from times in the other, which the steps
  // amplify (about 2.4e-12 here; the ocean functions equal).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyChange(dir_ / "held.nc", false);
  const auto config = [&](const std::string& name, const std::string& times,
                          const std::string& forcing,
                          const std::string& ocean) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static", "times": {},
      "forcing": {}, "ocean": {},
      "rotation": "full", "moments": "hydrostatic",
      "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, times, forcing, ocean,
                                             name));
  };
  constexpr auto kHistory =
      R"({"ice_change": [{"thickness": {"file": "history.nc"}}]})";
  constexpr auto kAtStart = R"({"at": "start",
      "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
  static_cast<void>(app::RunSolve(config(
      "even", R"({"start_kyr_bp": 2, "nodes": 5})", kHistory, kAtStart)));
  const auto summary = app::RunSolve(config(
      "listed", R"({"at_kyr_bp": [2, 1.5, 1, 0.5, 0]})", kHistory, kAtStart));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 2 to 0 kyr BP, shorelines frozen"))
      << summary;
  const auto even = Output("even");
  const auto listed = Output("listed");
  // the node times are whole seconds, formed exactly either way: bitwise
  EXPECT_EQ(*listed.time, *even.time);
  for (const auto& [name, values] : even.variables) {
    EXPECT_EQ(listed.variables.at(name).values, values.values) << name;
  }
  EXPECT_EQ(std::get<std::vector<double>>(listed.meta.at("dates_kyr_bp")),
            (std::vector<double>{2.0, 1.5, 1.0, 0.5, 0.0}));

  // uneven: dated against forwards
  constexpr auto kCap = R"({"cap_fraction": {"latitude": 90, "longitude": 0,
      "width_deg": 15, "fraction": -0.2}, )";
  constexpr auto kHeld = R"({"change": {"file": "held.nc"}})";
  auto worst = 0.0;
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto dated = std::format("dated_{}", shorelines);
    const auto clock = std::format("clock_{}", shorelines);
    static_cast<void>(app::RunSolve(config(
        dated, R"({"at_kyr_bp": [2, 1.7, 1.2, 0.5, 0]})",
        std::format(R"({{"ice_change": [{}"history_kyr_bp": [[2, 0], [1, 1]]}},
                                       {}]}})",
                    kCap, kHeld),
        std::format(R"({{"at": "start", "shorelines": "{}",
          "topography": {{"file": "history.nc"}},
          "ice": {{"file": "history.nc"}}}})",
                    shorelines))));
    const auto said = app::RunSolve(config(
        clock, R"({"unit": "kyr", "at": [0, 0.3, 0.8, 1.5, 2]})",
        std::format(R"({{"ice_change": [{}"history": [[0, 0], [1, 1]]}},
                                       {}]}})",
                    kCap, kHeld),
        std::format(R"({{"at": "start", "date_kyr_bp": 2,
          "shorelines": "{}", "topography": {{"file": "history.nc"}},
          "ice": {{"file": "history.nc"}}}})",
                    shorelines)));
    EXPECT_NE(said.find(std::format(
                  "ice {}:ice_thickness (2 kyr BP), topography {}:topography "
                  "(2 kyr BP); ice change cap at latitude 90, longitude 0, "
                  "width 15 deg, fraction -0.2 of the state's ice, history "
                  "in kyr [[0, 0], [1, 1]] + change {}:ice_thickness_change, "
                  "held through the run; wrote ",
                  (dir_ / "history.nc").string(),
                  (dir_ / "history.nc").string(), (dir_ / "held.nc").string())),
              std::string::npos)
        << said;
    const auto a = Output(dated);
    const auto b = Output(clock);
    for (const auto* name : {"relative_sea_level_change", "displacement",
                             "tangential_potential", "potential",
                             "initial_sea_level"}) {
      const auto departure =
          Departure(b.variables.at(name).values, a.variables.at(name).values);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-10) << name << ", " << shorelines;
    }
    if (std::string_view{shorelines} == "migrating") {
      // 0 or 1 by flotation: unequal only at a point on the threshold
      EXPECT_EQ(b.variables.at("ocean_function").values,
                a.variables.at("ocean_function").values);
    }
    EXPECT_EQ(std::get<std::vector<double>>(a.meta.at("dates_kyr_bp")),
              (std::vector<double>{2.0, 1.7, 1.2, 0.5, 0.0}));
    EXPECT_EQ(std::get<std::vector<double>>(b.meta.at("times")),
              (std::vector<double>{0.0, 0.3, 0.8, 1.5, 2.0}));
  }
  RecordProperty("dated_departure", std::format("{:.3e}", worst));
}

TEST_F(AppsTest, ReadsOutsideAFilesSpanAreRefusedUnlessHeld) {
  // Every read of a field on a time axis outside its file's span (the toy
  // history's, 2 to 0 kyr BP) is refused, naming the member, the file's
  // span, the date asked and the read: the static problem's state at its
  // date, a dated run's state at its start, a thickness term at the
  // state's instant, a calibrated run's state at its end. Each input that says
  // `"outside_span": "hold"` holds its file's first or last stored values
  // instead, as without the member the runs did: the static problem's
  // state at 3.5 kyr BP is its state at 2 kyr BP, to the bit. A date
  // within 1e-9 of the span (of its length or its ends' magnitudes) is the
  // end itself: a file whose first date is 1e-12 short of the window's
  // start is read there without the member, and one 1e-6 short is
  // refused; a file of the same dates in years runs as the one in seconds,
  // to the bit. `outside_span` is refused on a static field, where it does
  // nothing, and enters neither hash: a run's checkpoints resume under the
  // member.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kHold = R"(, "outside_span": "hold")";
  const auto fingerprint = [&](const std::string& name, double date,
                               const std::string& topography,
                               const std::string& ice) {
    return Write(name + ".json", std::format(R"({{
    "problem": "static", {},
    "forcing": {{"ice_change": [{{"cap_fraction": {{"latitude": 90,
                "longitude": 0, "width_deg": 15, "fraction": -0.1}}}}]}},
    "ocean": {{"date_kyr_bp": {}, "topography": {{"file": "history.nc"{}}},
              "ice": {{"file": "history.nc"{}}}}},
    "rotation": "none",
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, date, topography, ice,
                                             name));
  };
  static_cast<void>(app::RunSolve(fingerprint("oldest", 2.0, "", "")));
  ExpectRefused(app::RunSolve, fingerprint("older", 3.5, "", ""),
                "ocean.topography.file",
                std::format("spans 2 to 0 kyr BP, which leaves out 3.5 kyr "
                            "BP, the state's date, `ocean.date_kyr_bp`, where "
                            "the state is read: give a file that covers it, "
                            "or \"outside_span\": \"hold\""));
  // each input holds its own ends
  ExpectRefused(app::RunSolve, fingerprint("older", 3.5, kHold, ""),
                "ocean.ice.file", "which leaves out 3.5 kyr BP");
  static_cast<void>(app::RunSolve(fingerprint("older", 3.5, kHold, kHold)));
  // the same stored values read, the same transforms: bitwise
  for (const auto* name : {"relative_sea_level_change", "ocean_function"}) {
    EXPECT_EQ(Output("older").variables.at(name).values,
              Output("oldest").variables.at(name).values)
        << name;
  }

  const auto window = [&](const std::string& name, const std::string& times,
                          const std::string& at, const std::string& state,
                          const std::string& term,
                          const std::string& file = "history.nc") {
    return Write(name + ".json", std::format(R"({{
    "problem": "quasi_static", {}, "times": {{{}}},
    "forcing": {{"ice_change": [{{"thickness": {{"file": "{}"{}}}}}]}},
    "ocean": {{"at": "{}", "topography": {{"file": "{}"{}}},
              "ice": {{"file": "{}"{}}}}},
    "rotation": "none",
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, times, file, term, at, file,
                                             state, file, state, name));
  };
  constexpr auto kEarly = R"("start_kyr_bp": 3.0, "end_kyr_bp": 0.0, "nodes": 4)";
  ExpectRefused(app::RunSolve, window("early", kEarly, "start", "", ""),
                "ocean.topography.file",
                "spans 2 to 0 kyr BP, which leaves out 3 kyr BP, the window's "
                "start, where the state is read");
  ExpectRefused(app::RunSolve, window("early", kEarly, "start", kHold, ""),
                "forcing.ice_change[0].thickness.file",
                "spans 2 to 0 kyr BP, which leaves out 3 kyr BP, the "
                "state's instant, from which its change is measured");
  const auto early =
      app::RunSolve(window("early", kEarly, "start", kHold, kHold));
  EXPECT_NE(early.find("(the window's start, 3 kyr BP)"), std::string::npos)
      << early;
  constexpr auto kLate = R"("start_kyr_bp": 2.0, "end_kyr_bp": -0.5, "nodes": 3)";
  ExpectRefused(app::RunSolve, window("late", kLate, "end", "", ""),
                "ocean.topography.file",
                "spans 2 to 0 kyr BP, which leaves out -0.5 kyr BP, the "
                "window's end, where the state is read, the calibration's "
                "target");
  // held, the calibration's target is the file's last state
  const auto late = app::RunSolve(window("late", kLate, "end", kHold, kHold));
  EXPECT_NE(late.find("initial sea level calibrated"), std::string::npos)
      << late;

  // the span's tolerance: the toy history's first date moved later by a
  // relative 1e-12 of the window's start, and by 1e-6; and its dates in
  // years
  const auto moved = [&](const std::string& name, double shift,
                         double unit) {
    auto time = std::vector<double>{-2.0 * app::KYR * (1.0 - shift),
                                    -1.0 * app::KYR, 0.0};
    for (auto& t : time) t /= unit;
    sl3dpp::test::WriteToyHistoryOn(dir_ / name, time, unit);
  };
  moved("near.nc", 1e-12, 1.0);
  moved("far.nc", 1e-6, 1.0);
  moved("years.nc", 0.0, app::KYR / 1000.0);
  constexpr auto kEven = R"("start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5)";
  static_cast<void>(
      app::RunSolve(window("near", kEven, "start", "", "", "near.nc")));
  ExpectRefused(app::RunSolve, window("far", kEven, "start", "", "", "far.nc"),
                "ocean.topography.file",
                "which leaves out 2 kyr BP, the window's start");
  static_cast<void>(
      app::RunSolve(window("seconds", kEven, "start", "", "")));
  static_cast<void>(
      app::RunSolve(window("years", kEven, "start", "", "", "years.nc")));
  // the axis's years scale back to the same seconds exactly: bitwise
  for (const auto* name : {"relative_sea_level_change", "displacement"}) {
    EXPECT_EQ(Output("years").variables.at(name).values,
              Output("seconds").variables.at(name).values)
        << name;
  }

  // the member on a static field does nothing, and is refused
  sl3dpp::test::WriteToyLand(dir_ / "land.nc");
  ExpectRefused(
      app::RunSolve,
      Write("static_land.json",
            std::format(R"({{"problem": "quasi_static", {}, "times": {{{}}},
              "forcing": {{"ice_change": [{{"thickness": {{"file":
                "history.nc"}}}}]}},
              "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}},
                "land": {{"file": "land.nc", "outside_span": "hold"}}}},
              "rotation": "none", "output": {{"surface": "out/l.nc"}}}})",
                        kBall, kEven)),
      "ocean.land.outside_span",
      std::format("{} has no time axis", (dir_ / "land.nc").string()));
  sl3dpp::test::WriteToyHistoryOn(
      dir_ / "with_land.nc", {-2.0 * app::KYR, -1.0 * app::KYR, 0.0}, 1.0,
      true);
  ExpectRefused(
      app::RunSolve,
      Write("static_field.json",
            std::format(R"({{"problem": "quasi_static", {}, "times": {{{}}},
              "forcing": {{"ice_change": [{{"thickness": {{"file":
                "history.nc"}}}}]}},
              "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}},
                "land": {{"file": "with_land.nc", "outside_span": "refuse"}}}},
              "rotation": "none", "output": {{"surface": "out/l.nc"}}}})",
                        kBall, kEven)),
      "ocean.land.outside_span",
      std::format("\"land\" of {} is static",
                  (dir_ / "with_land.nc").string()));

  // neither hash takes the member: a run stopped at its wall-time limit
  // resumes with it
  const auto hashesOf = [&](const fs::path& path) {
    const auto input = app::ReadSolveInput(path);
    auto hash = app::Fnv1a{};
    app::AddRunMembers(hash, input.run, false);
    return std::pair{hash.Value(), app::TrajectoryHashOf(input.run, false)};
  };
  const auto stopping = [&](const std::string& name, const std::string& hold,
                            const std::string& limit) {
    return Write(name + ".json", std::format(R"({{
    "problem": "quasi_static", {}, "times": {{{}}},
    "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"{}}}}}]}},
    "ocean": {{"at": "start", "topography": {{"file": "history.nc"{}}},
              "ice": {{"file": "history.nc"{}}}}},
    "rotation": "none",
    "checkpoint": {{"directory": "ckpt", "every": 1{}}},
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, kEven, hold, hold, hold,
                                             limit, name));
  };
  constexpr auto kLimit = R"(, "max_walltime_min": 1e-9)";
  EXPECT_EQ(hashesOf(stopping("plain", "", kLimit)),
            hashesOf(stopping("held", kHold, "")));
  const auto stopped = app::RunSolve(stopping("plain", "", kLimit));
  EXPECT_NE(stopped.find("at the wall-time limit"), std::string::npos)
      << stopped;
  const auto resumed = app::RunSolve(stopping("held", kHold, ""));
  EXPECT_NE(resumed.find(", resumed"), std::string::npos) << resumed;
  // one process, the same transforms, the state read back exactly: bitwise
  EXPECT_EQ(Output("held").variables.at("relative_sea_level_change").values,
            Output("seconds").variables.at("relative_sea_level_change").values);
}

TEST_F(AppsTest, SolveOnItsOwnClockReadsItsStateAtItsDate) {
  // A run counting forwards has no calendar of its own: its state is a
  // snapshot of its files at `ocean.date_kyr_bp`. The toy history's slice
  // at 1 kyr BP, named through the file with a time axis and that date,
  // gives the run of the same slice written as files without one, under a
  // melting cap and a surface load, each with a history in years, to
  // rounding (equal to the bit here); the summary and the metadata give
  // the slice read.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  {
    // the history's slice at 1 kyr BP, its second, as static fields
    const auto history = pm::ReadSurfaceFile(dir_ / "history.nc");
    auto file = pm::SurfaceFile{};
    file.grid = history.grid;
    file.source = "sl3dpp tests";
    for (const auto* name : {"ice_thickness", "topography"}) {
      const auto& values = history.variables.at(name).values;
      const auto size = values.size() / history.time->size();
      file.variables.emplace(
          name, pm::SurfaceField(
                    name,
                    std::vector<double>(
                        values.begin() + static_cast<std::ptrdiff_t>(size),
                        values.begin() + static_cast<std::ptrdiff_t>(2 * size)),
                    {.dimensions = pm::units::LENGTH}));
    }
    pm::WriteSurfaceFile(dir_ / "slice.nc", file);
  }
  const auto config = [&](const std::string& name, const std::string& ocean) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"unit": "yr", "at": [0, 100, 300, 1000]}},
      "forcing": {{
        "ice_change": [{{"cap_fraction": {{"latitude": 90, "longitude": 0,
                          "width_deg": 15, "fraction": -0.2}},
                        "history": [[0, 0], [500, 1]]}}],
        "load": [{{"cap": {{"latitude": -40, "longitude": 120,
                            "width_deg": 10, "amplitude_kg_m2": 2e4}},
                   "history": [[0, 0], [300, 1]],
                   "water_budget": true, "moves_solid_surface": false}}]}},
      "ocean": {}, "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, ocean, name));
  };
  const auto summary = app::RunSolve(config(
      "dated", R"({"at": "start", "date_kyr_bp": 1,
                   "topography": {"file": "history.nc"},
                   "ice": {"file": "history.nc"}})"));
  EXPECT_TRUE(summary.starts_with(std::format(
      "1-D, viscoelastic, 4 nodes 0 to 1000 yr on the run's clock, "
      "shorelines frozen, rotation none; ice {}:ice_thickness (1 kyr BP), "
      "topography {}:topography (1 kyr BP); ice change cap",
      (dir_ / "history.nc").string(), (dir_ / "history.nc").string())))
      << summary;
  static_cast<void>(app::RunSolve(
      config("still", R"({"at": "start", "topography": {"file": "slice.nc"},
                          "ice": {"file": "slice.nc"}})")));
  const auto dated = Output("dated");
  const auto still = Output("still");
  EXPECT_EQ(std::get<std::string>(still.meta.at("ice")),
            std::format("{}:ice_thickness (static)",
                        (dir_ / "slice.nc").string()));
  auto worst = 0.0;
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "potential", "initial_sea_level"}) {
    const auto departure = Departure(dated.variables.at(name).values,
                                     still.variables.at(name).values);
    worst = std::max(worst, departure);
    EXPECT_LT(departure, 1e-13) << name;
  }
  RecordProperty("slice_departure", std::format("{:.3e}", worst));
  // the cap and the load act: the run is not the state at rest
  auto moved = 0.0;
  for (const auto v : Slice(dated, "relative_sea_level_change", 3)) {
    moved = std::max(moved, std::abs(v));
  }
  EXPECT_GT(moved, 1.0);
}

TEST_F(AppsTest, SolveOnItsOwnClockRestartsFromItsCheckpoints) {
  // A run counting forwards, stopped at its checkpoints by the wall-time
  // limit and resumed, reproduces the uninterrupted run to the bit, its
  // time axis included. Its checkpoints belong to its unit: the same nodes
  // and times stated in kyr are another run, refused them.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = [&](const std::string& unit, double scale) {
    return std::format(R"({},
      "problem": "quasi_static",
      "times": {{"unit": "{}", "at": [0, {:.17g}, {:.17g}, {:.17g}]}},
      "forcing": {{
        "ice_change": [{{"cap_fraction": {{"latitude": 90, "longitude": 0,
                          "width_deg": 15, "fraction": -0.2}},
                        "history": [[0, 0], [{:.17g}, 1]]}}]}},
      "ocean": {{"at": "start", "date_kyr_bp": 2,
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic")",
                       kBall, unit, 100 * scale, 300 * scale, 1000 * scale,
                       500 * scale);
  };
  static_cast<void>(app::RunSolve(Write(
      "whole.json",
      "{" + body("yr", 1.0) + R"(, "output": {"surface": "out/whole.nc"}})")));
  const auto pieces = Write(
      "pieces.json", "{" + body("yr", 1.0) +
                         R"(, "checkpoint": {"directory": "ckpt", "every": 1,
                              "max_walltime_min": 1e-9},
              "output": {"surface": "out/pieces.nc"}})");
  auto attempts = 0;
  auto summary = std::string{};
  for (; attempts < 8; ++attempts) {
    summary = app::RunSolve(pieces);
    if (summary.find("wrote") != std::string::npos) break;
    EXPECT_NE(summary.find("checkpointed"), std::string::npos) << summary;
  }
  EXPECT_GE(attempts, 1);
  EXPECT_NE(summary.find("resumed"), std::string::npos) << summary;
  const auto whole = Output("whole");
  const auto rebuilt = Output("pieces");
  // one process, the same transforms, the state read back exactly: bitwise
  EXPECT_EQ(*rebuilt.time, *whole.time);
  for (const auto* name : {"relative_sea_level_change", "displacement",
                           "tangential_potential", "potential"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              whole.variables.at(name).values)
        << name;
  }
  EXPECT_EQ(rebuilt.meta.at("times"), whole.meta.at("times"));
  EXPECT_EQ(rebuilt.meta.at("omega_z_rad_s"), whole.meta.at("omega_z_rad_s"));
  const auto inKyr = Write(
      "in_kyr.json", "{" + body("kyr", 1e-3) +
                         R"(, "checkpoint": {"directory": "ckpt", "every": 1},
              "output": {"surface": "out/in_kyr.nc"}})");
  EXPECT_THROW(
      {
        try {
          static_cast<void>(app::RunSolve(inKyr));
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string{error.what()}.find("different run"),
                    std::string::npos)
              << error.what();
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(AppsTest, SolveRefusesWhatEachFormOfItsTimesCannotSay) {
  // Each fault of the two forms of the times is refused naming the
  // member: a run counting forwards without its unit or with an unknown
  // one, members of the two forms together, times in kyr without a unit
  // (`start_kyr`, `history_kyr`, `onset_kyr`, `time_origin_kyr`), either
  // form's nodes given twice, out of order or too few; the terms' times in
  // the other form's names (histories, onsets, a field's time origin and
  // dates), and the static problem's own (a `date` without its `unit`,
  // with `date_kyr_bp`, a `unit` alone, a time origin); in a run counting
  // forwards a `thickness` term and a `change` file with a time axis (a
  // dated ice history), the state at the window's end, and the state's
  // date held to its files; a kernel measurement's instant in the other
  // form or off the nodes. A change
  // taking more ice than the state holds names its time in the run's
  // unit.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyChange(dir_ / "change.nc", true);
  {
    namespace pm = planetmodel;
    auto file = pm::SurfaceFile{};
    file.grid = pm::Equiangular(9, 18);
    file.source = "sl3dpp tests";
    for (const auto* name : {"ice_thickness", "topography", "load"}) {
      file.variables.emplace(
          name, pm::SurfaceField(name, std::vector<double>(9 * 18, 100.0),
                                 {.dimensions = pm::units::LENGTH}));
    }
    pm::WriteSurfaceFile(dir_ / "static.nc", file);
  }
  const auto read = [](const fs::path& path) {
    return app::ReadSolveInput(path);
  };
  const auto refused = [&](auto&& run, const std::string& text,
                           const std::string& member, const std::string& what) {
    ExpectRefused(run, Write("bad.json", text), member, what);
  };
  constexpr auto kForwards = R"({"unit": "yr", "at": [0, 100, 300]})";
  constexpr auto kDated = R"({"start_kyr_bp": 2, "nodes": 3})";
  constexpr auto kCapLoad = R"({"cap": {"latitude": 60, "longitude": -85,
      "width_deg": 20, "amplitude_kg_m2": 1.0e6}, )";
  // a run without an ocean under the times and the forcing given
  const auto dry = [&](const std::string& times, const std::string& forcing) {
    return std::format(R"({{"problem": "quasi_static",
        "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 4}}, "times": {},
        "forcing": {}, "output": {{"surface": "o.nc"}}}})",
                       times, forcing);
  };
  // a run with an ocean, its state in `ocean`
  const auto wet = [&](const std::string& times, const std::string& forcing,
                       const std::string& ocean) {
    return std::format(R"({{"problem": "quasi_static",
        "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 4}}, "times": {},
        "forcing": {}, "ocean": {}, "output": {{"surface": "o.nc"}}}})",
                       times, forcing, ocean);
  };
  const auto load = std::format(R"({{"load": [{}"history": [[0, 1]]}}]}})",
                                kCapLoad);
  const auto times = [&](const std::string& block) { return dry(block, load); };
  // the load with the properties a run with an ocean is told
  const auto wetLoad = std::format(R"({{"load": [{}"history": [[0, 1]],
      "water_budget": false, "moves_solid_surface": false}}]}})",
                                   kCapLoad);

  // the times block's two forms
  refused(read, times(R"({"at": [0, 100]})"), "times",
          "gives no `unit` for its times: \"kyr\", \"yr\" or \"day\"");
  refused(read, times(R"({"unit": "month", "at": [0, 1]})"), "times.unit",
          "none of \"kyr\", \"yr\", \"day\"");
  refused(read, times(R"({"start_kyr_bp": 2, "unit": "kyr", "end": 1,
                          "nodes": 3})"),
          "times.unit",
          "beside `start_kyr_bp`, a date before present: the times are given "
          "in one of two forms, dates before present (`start_kyr_bp`, "
          "`end_kyr_bp` and `nodes`, or `at_kyr_bp`) or times counted "
          "forwards on the run's own clock (`unit` with `start`, `end` and "
          "`nodes`, or `at`)");
  refused(read, times(R"({"start_kyr": 0, "end_kyr": 2, "nodes": 3})"),
          "times.start_kyr", "times on the run's clock are `start` with `unit`");
  refused(read, times(R"({"nodes": 3})"), "times", "gives no window");
  refused(read, times(R"({"at_kyr_bp": [2, 1, 1]})"), "times.at_kyr_bp[2]",
          "strictly decreasing in kyr BP");
  refused(read, times(R"({"at_kyr_bp": [2]})"), "times.at_kyr_bp",
          "fewer than two nodes");
  refused(read, times(R"({"start_kyr_bp": 2, "at_kyr_bp": [2, 1]})"),
          "times.start_kyr_bp", "is given by `at_kyr_bp`");
  refused(read, times(R"({"at_kyr_bp": [2, 1], "nodes": 2})"), "times",
          "gives the nodes twice");
  refused(read, times(R"({"unit": "yr", "at": [0, 100, 50]})"), "times.at[2]",
          "strictly increasing");
  refused(read, times(R"({"unit": "yr", "start": 0, "at": [0, 100]})"),
          "times.start", "is given by `at`");
  refused(read, times(R"({"unit": "yr", "end": 100, "at": [0, 100]})"),
          "times", "gives the nodes twice");
  refused(read, times(R"({"unit": "yr"})"), "times", "gives no nodes");
  refused(read, times(R"({"unit": "yr", "nodes": 3})"), "times",
          "gives `nodes` without `end`");
  refused(read, times(R"({"unit": "yr", "start": 5, "end": 5, "nodes": 3})"),
          "times.end", "is not later than `start`");
  refused(read, times(R"({"unit": "yr", "end": 5, "nodes": 1})"),
          "times.nodes", "fewer than two nodes");
  refused(read, times(R"({"unit": "yr", "at": [0, 1], "step": 1})"),
          "times.step", "not a recognised member");

  // the terms' times, each form naming its own
  refused(read,
          dry(kForwards, std::format(R"({{"load": [{}"history_kyr_bp":
                                       [[2, 1]]}}]}})",
                                     kCapLoad)),
          "forcing.load[0].history_kyr_bp",
          "is a date before present, the dated form of `times`; this run "
          "counts forwards on its own clock, in its `unit` (`times.unit`): "
          "`history`");
  refused(read,
          dry(kForwards, std::format(R"({{"load": [{}"history_kyr":
                                       [[0, 1]]}}]}})",
                                     kCapLoad)),
          "forcing.load[0].history_kyr",
          "is a time in kyr without its unit; this run counts forwards on its "
          "own clock, in its `unit` (`times.unit`): `history`");
  refused(read,
          dry(kForwards, std::format(R"({{"tide": [{}"history_kyr_bp":
                                       [[2, 1]]}}]}})",
                                     kTideOpen)),
          "forcing.tide[0].history_kyr_bp", "`history`");
  refused(read,
          dry(kForwards,
              std::format(R"({{"seismic": [{}"onset_kyr_bp": 1}}]}})",
                          kPointOpen)),
          "forcing.seismic[0].onset_kyr_bp", "`onset`");
  refused(read,
          dry(kForwards, std::format(R"({{"seismic": [{}"onset_kyr": 1}}]}})",
                                     kPointOpen)),
          "forcing.seismic[0].onset_kyr", "`onset`");
  refused(read,
          dry(kForwards, R"({"load": [{"field": {"file": "static.nc",
                              "time_origin_kyr": 2}}]})"),
          "forcing.load[0].field.time_origin_kyr", "`time_origin`");
  refused(read,
          dry(kForwards, R"({"load": [{"field": {"file": "static.nc",
                              "date_kyr_bp": 2}}]})"),
          "forcing.load[0].field.date_kyr_bp", "at t - `time_origin`");
  refused(read,
          dry(kForwards, R"({"load": [{"field": {"file": "static.nc",
                              "date": 2, "unit": "yr"}}]})"),
          "forcing.load[0].field.date", "at t - `time_origin`");
  refused(read,
          dry(kDated, std::format(R"({{"load": [{}"history": [[0, 1]]}}]}})",
                                  kCapLoad)),
          "forcing.load[0].history",
          "the form of a `times` block counting forwards with a `unit`; this "
          "run's times are dates, kyr before present: `history_kyr_bp`");
  refused(read,
          dry(kDated,
              std::format(R"({{"seismic": [{}"onset": 1}}]}})", kPointOpen)),
          "forcing.seismic[0].onset", "`onset_kyr_bp`");
  refused(read,
          dry(kDated, R"({"load": [{"field": {"file": "static.nc",
                           "time_origin": 2}}]})"),
          "forcing.load[0].field.time_origin", "is read by date");
  refused(read,
          dry(kDated, R"({"load": [{"field": {"file": "static.nc",
                           "date": 2, "unit": "yr"}}]})"),
          "forcing.load[0].field.date", "reads it at each node's date");
  refused(read,
          dry(kDated, R"({"load": [{"field": {"file": "static.nc",
                           "unit": "yr"}}]})"),
          "forcing.load[0].field.unit", "reads the file at each node's date");
  // the static problem's: no history, onset or time origin; a date on the
  // file's own axis with its unit
  const auto fixed = [&](const std::string& forcing) {
    return std::format(R"({{"problem": "static",
        "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 4}}, "forcing": {},
        "output": {{"surface": "o.nc"}}}})",
                       forcing);
  };
  refused(read,
          fixed(std::format(R"({{"load": [{}"history": [[0, 1]]}}]}})",
                            kCapLoad)),
          "forcing.load[0].history", "which needs the quasi-static problem");
  refused(read,
          fixed(std::format(R"({{"seismic": [{}"onset": 1}}]}})", kPointOpen)),
          "forcing.seismic[0].onset", "which needs the quasi-static problem");
  refused(read,
          fixed(R"({"load": [{"field": {"file": "static.nc",
                    "time_origin": 2}}]})"),
          "forcing.load[0].field.time_origin",
          "which needs the quasi-static problem");
  refused(read,
          fixed(R"({"load": [{"field": {"file": "static.nc", "date": 2}}]})"),
          "forcing.load[0].field.date",
          "needs its `unit` beside it: \"kyr\", \"yr\" or \"day\"");
  refused(read,
          fixed(R"({"load": [{"field": {"file": "static.nc", "date": 2,
                    "unit": "yr", "date_kyr_bp": 1}}]})"),
          "forcing.load[0].field.date",
          "dates the file twice, beside `date_kyr_bp`");
  refused(read,
          fixed(R"({"load": [{"field": {"file": "static.nc",
                    "unit": "yr"}}]})"),
          "forcing.load[0].field.unit", "which the field does not give");
  refused(read,
          fixed(R"({"load": [{"field": {"file": "static.nc", "date": 2,
                    "unit": "week"}}]})"),
          "forcing.load[0].field.unit", "none of \"kyr\", \"yr\", \"day\"");
  refused(app::RunSolve,
          fixed(R"({"load": [{"field": {"file": "static.nc", "date": 2,
                    "unit": "yr"}}]})"),
          "forcing.load[0].field.date", "dates a file without a time axis");

  // the ice and the state of a run counting forwards
  constexpr auto kState = R"({"at": "start", "date_kyr_bp": 2,
      "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
  refused(read,
          wet(kForwards,
              R"({"ice_change": [{"thickness": {"file": "history.nc"}}]})",
              kState),
          "forcing.ice_change[0].thickness",
          "a dated history, which makes the problem a dated one. Give the "
          "dated form of `times`");
  refused(read,
          wet(kForwards,
              R"({"ice_change": [{"cap_thickness": {"latitude": 70,
                  "longitude": 0, "width_deg": 10, "thickness_m": 1},
                  "history_kyr_bp": [[2, 0], [0, 1]]}]})",
              kState),
          "forcing.ice_change[0].history_kyr_bp", "`history`");
  refused(read,
          wet(kForwards,
              R"({"ice_change": [{"change": {"file": "change.nc",
                  "date_kyr_bp": 1}}]})",
              kState),
          "forcing.ice_change[0].change.date_kyr_bp",
          "holds the file, one field without a time axis");
  refused(app::RunSolve,
          wet(kForwards,
              R"({"ice_change": [{"change": {"file": "change.nc"}}]})", kState),
          "forcing.ice_change[0].change",
          "a dated history of the change, which makes the problem a dated "
          "one: give the dated form of `times`");
  refused(read,
          wet(kForwards, wetLoad,
              R"({"at": "end", "topography": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}})"),
          "ocean.at",
          "Give the state at the start, \"start\", or the dated form of "
          "`times`");
  refused(read,
          wet(kForwards, wetLoad,
              R"({"at": "start", "topography": {"file": "history.nc"},
                  "ice": {"file": "history.nc", "date_kyr_bp": 2}})"),
          "ocean.ice.date_kyr_bp", "`ocean.date_kyr_bp`");
  refused(app::RunSolve,
          wet(kForwards, wetLoad,
              R"({"at": "start", "topography": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}})"),
          "ocean", "give the state's date, `date_kyr_bp`");
  refused(app::RunSolve,
          wet(kForwards, wetLoad,
              R"({"at": "start", "date_kyr_bp": 2,
                  "topography": {"file": "static.nc"},
                  "ice": {"file": "static.nc"}})"),
          "ocean.date_kyr_bp", "dates a state whose files have no time axis");
  // a dated run's state stays at an end of its window
  refused(read,
          wet(kDated,
              std::format(R"({{"load": [{}"water_budget": false,
                  "moves_solid_surface": false}}]}})",
                          kCapLoad),
              R"({"at": "start", "date_kyr_bp": 2,
                  "topography": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}})"),
          "ocean.date_kyr_bp", "a dated run puts the state at an end");
  // more ice taken than the state holds, at a time of the run's clock
  for (const auto* what :
       {"removes more ice than the state holds, by ", ", 300 yr)"}) {
    refused(app::RunSolve,
            wet(kForwards,
                R"({"ice_change": [{"cap_fraction": {"latitude": 90,
                    "longitude": 0, "width_deg": 15, "fraction": -1},
                    "history": [[0, 0], [300, 10]]}]})",
                kState),
            "forcing.ice_change", what);
  }

  // a measurement's instant in the form of the run's times: `time` on the
  // run's clock, in its unit, `date_kyr_bp` before present, each a node
  const auto kernel = [&](const std::string& block,
                          const std::string& measurement) {
    const auto dated = block.find("at_kyr_bp") != std::string::npos;
    return std::format(R"({{"problem": "quasi_static",
        "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 4}}, "times": {},
        "forcing": {}, "kernels": {{"parameters": ["viscosity"],
                                  "measurements": [{}], "output": "k"}}}})",
                       block,
                       dated ? std::format(R"({{"load": [{}"history_kyr_bp":
                                               [[2, 1]]}}]}})",
                                           kCapLoad)
                             : load,
                       measurement);
  };
  refused(app::ReadKernelInput,
          kernel(kForwards, R"({"type": "rotation", "component": "z",
                                "date_kyr_bp": 0})"),
          "kernels.measurements[0].date_kyr_bp",
          "the run counts forwards on its own clock: give the measurement's "
          "time on it, `time`, in yr");
  refused(app::ReadKernelInput,
          kernel(kForwards, R"({"type": "rotation", "component": "z",
                                "time": 200})"),
          "kernels.measurements[0].time",
          "200 yr is not a node of the times grid");
  refused(app::ReadKernelInput,
          kernel(R"({"at_kyr_bp": [2, 1, 0]})",
                 R"({"type": "rotation", "component": "z", "time": 1})"),
          "kernels.measurements[0].time",
          "the run's nodes are dates before present: give the measurement's "
          "date, `date_kyr_bp`");
  refused(app::ReadKernelInput,
          kernel(R"({"at_kyr_bp": [2, 1.5, 0]})",
                 R"({"type": "rotation", "component": "z",
                     "date_kyr_bp": 1})"),
          "kernels.measurements[0].date_kyr_bp",
          "1 kyr BP is not a node of the times grid");
  // the listed nodes and the run's clock are taken; a time defaults to the
  // last node, a date to the present
  EXPECT_EQ(app::ReadKernelInput(
                Write("clock.json",
                      kernel(kForwards, R"({"type": "rotation",
                                            "component": "z", "time": 100},
                                           {"type": "rotation",
                                            "component": "x"})")))
                .kernels.measurements[1]
                .node,
            2u);
  EXPECT_EQ(app::ReadKernelInput(
                Write("listed.json",
                      kernel(R"({"at_kyr_bp": [2, 0.5, 0]})",
                             R"({"type": "rotation", "component": "z",
                                 "date_kyr_bp": 0.5}, {"type": "rotation",
                                 "component": "x"})")))
                .kernels.measurements[0]
                .node,
            1u);
}

// -- the kernels of every problem, against solve --------------------------------

TEST_F(AppsTest, KernelsOfTheStaticProblemWithoutAnOcean) {
  // The static problem on the ball without an ocean under a cap load, one
  // config for solve and for kernel: the kernels of a radial displacement
  // at a point and of a vector displacement over a cap (its east, north and
  // up components), in the load and in the moduli. Their values are
  // solve's. The run is linear in the load, so each value is its load
  // kernel paired with the load solve applied; the moduli kernels, paired
  // with the moduli (the table's own columns), against central differences
  // of solve on the ball with its moduli scaled by 1 +- 1e-4. The two
  // methods agree to rounding, there being no step to commute.
  namespace pm = planetmodel;
  static_cast<void>(Write("ball.deck", BallDeck(0.0)));
  static_cast<void>(Write("stiff.deck", BallDeck(1e-4)));
  static_cast<void>(Write("soft.deck", BallDeck(-1e-4)));
  const auto config = [&](const std::string& name, const std::string& deck,
                          const std::string& method) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static", "model": {{"deck": "{}"}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
          "width_deg": 20, "amplitude_kg_m2": 1.0e6}}}}]}},
      "rotation": "none", "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["load", "moduli"],
        "measurements": [
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": -80}},
          {{"type": "displacement_vector_cap", "name": "drift",
            "latitude": 45, "longitude": -70, "width_deg": 10,
            "direction": {{"east": 1, "north": 0.5, "up": 0.2}}}}],
        "output": "out/{}_kernels"}}}})",
                                             deck, name, method, name));
  };
  const auto base = config("continuous", "ball.deck", "continuous");
  const auto summary = app::RunKernel(base);
  EXPECT_TRUE(summary.starts_with(
      "1-D, elastic, static, no ocean, rotation none; load (1 term); wrote 2 "
      "measurement(s) of kernels to "))
      << summary;
  EXPECT_TRUE(summary.ends_with("(load, moduli; continuous adjoint)"))
      << summary;
  static_cast<void>(app::RunKernel(config("discrete", "ball.deck", "discrete")));
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(config("stiff", "stiff.deck", "discrete")));
  static_cast<void>(app::RunSolve(config("soft", "soft.deck", "discrete")));
  const auto run = pm::ReadSurfaceFile(dir_ / "out" / "continuous.nc");
  const auto stiff = pm::ReadSurfaceFile(dir_ / "out" / "stiff.nc");
  const auto soft = pm::ReadSurfaceFile(dir_ / "out" / "soft.nc");
  const auto direction = SurfaceDirection{.east = 1.0, .north = 0.5, .up = 0.2};
  const auto valueOf = [&](const pm::SurfaceFile& file,
                           const std::string& name) {
    return name == "uplift"
               ? PointValueOf(file, "displacement", 0, {55.0, -80.0})
               : VectorCapOf(file, 0, {45.0, -70.0}, 10.0, direction);
  };
  const auto out = dir_ / "out" / "continuous_kernels";
  const auto applied = run.variables.at("load_outside_water_budget").values;
  for (const auto* name : {"uplift", "drift"}) {
    const auto value = valueOf(run, name);
    EXPECT_LT(Gap(MeasuredValue(out, name), value), 1e-10) << name;
    // the load
    const auto kernels =
        pm::ReadSurfaceFile(out / (std::string{name} + "_load_kernels.nc"));
    EXPECT_FALSE(kernels.time.has_value());
    EXPECT_FALSE(kernels.variables.contains("load_outside_water_budget_dual"));
    const auto& dual = kernels.variables.at("load_dual").values;
    const auto paired = SurfacePairing(kernels, dual, applied);
    // the moduli
    const auto rows = TableRows(out / (std::string{name} + "_moduli_kernel.txt"));
    auto predicted = 0.0;
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 6u);
      predicted += row[1] * (row[4] * row[2] + row[5] * row[3]);
    }
    const auto measured = (valueOf(stiff, name) - valueOf(soft, name)) / 2e-4;
    std::cout << std::format(
        "[static, no ocean, {}] load kernel {:.3e} from the value; moduli "
        "{:.3e} from solve's differences\n",
        name, Gap(paired, value), Gap(predicted, measured));
    RecordProperty(std::string{name} + "_load_gap",
                   std::format("{:.3e}", Gap(paired, value)));
    RecordProperty(std::string{name} + "_moduli_gap",
                   std::format("{:.3e}", Gap(predicted, measured)));
    EXPECT_LT(Gap(paired, value), 1e-9) << name;
    EXPECT_LT(Gap(predicted, measured), 1e-6) << name;
    // the discrete method's, to rounding
    const auto other = dir_ / "out" / "discrete_kernels";
    EXPECT_LT(
        Departure(pm::ReadSurfaceFile(
                      other / (std::string{name} + "_load_kernels.nc"))
                      .variables.at("load_dual")
                      .values,
                  dual),
        1e-9)
        << name;
    const auto discrete =
        TableRows(other / (std::string{name} + "_moduli_kernel.txt"));
    auto kd = std::vector<double>{}, kc = std::vector<double>{};
    for (auto i = std::size_t{0}; i < rows.size(); ++i) {
      for (const auto c : {4, 5}) {
        kc.push_back(rows[i][static_cast<std::size_t>(c)]);
        kd.push_back(discrete[i][static_cast<std::size_t>(c)]);
      }
    }
    EXPECT_LT(Departure(kd, kc), 1e-9) << name;
  }
}

TEST_F(AppsTest, KernelsOfTheStaticProblemWithAnIceChange) {
  // The static problem on the toy state at the present under a cap of ice
  // laid on its land, under frozen and under migrating shorelines: the
  // kernels in the ice of a sea level cap mean and of a potential at a
  // point. Each value is solve's; solve runs with a second cap of 10 m
  // added to the change, the state not moved, differ by the thickness
  // kernel's prediction: the run is linear in the ice under frozen
  // shorelines and, while no cell flips, under migrating ones. The kernel
  // is the derivative with respect to the change; the library's one-node
  // thickness view, its baseline folded in, is identically zero under
  // frozen shorelines.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kChange = R"({"cap_thickness": {"latitude": 50,
      "longitude": 0, "width_deg": 8, "thickness_m": 200}})";
  constexpr auto kBump = R"({"cap_thickness": {"latitude": 45,
      "longitude": 30, "width_deg": 6, "thickness_m": 10}})";
  const auto config = [&](const std::string& name, const std::string& terms,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "static", "forcing": {{"ice_change": [{}]}},
      "ocean": {{"date_kyr_bp": 0, "shorelines": "{}",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"parameters": ["ice"], "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": 20}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, terms, shorelines, name,
                                             name));
  };
  const auto rad = std::numbers::pi / 180.0;
  const auto bump = app::CapField(SurfaceGrid(8), {45.0, 30.0}, 6.0 * rad, 10.0);
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto label = std::string{shorelines};
    const auto base = config(label, kChange, shorelines);
    // the default method, the discrete sweep
    const auto summary = app::RunKernel(base);
    EXPECT_NE(summary.find("(ice; discrete adjoint)"), std::string::npos)
        << summary;
    static_cast<void>(app::RunSolve(base));
    static_cast<void>(app::RunSolve(config(
        label + "_bumped", std::string{kChange} + ", " + kBump, shorelines)));
    const auto plain = pm::ReadSurfaceFile(dir_ / "out" / (label + ".nc"));
    const auto bumped =
        pm::ReadSurfaceFile(dir_ / "out" / (label + "_bumped.nc"));
    if (label == "migrating") {
      EXPECT_EQ(plain.variables.at("ocean_function_final").values,
                bumped.variables.at("ocean_function_final").values)
          << "a cell flipped";
    }
    const auto out = dir_ / "out" / (label + "_kernels");
    const auto siteOf = [](const pm::SurfaceFile& file) {
      return CapMeanOf(file, "relative_sea_level_change", 0, {-30.0, 10.0},
                       25.0);
    };
    const auto geoidOf = [](const pm::SurfaceFile& file) {
      return PointValueOf(file, "potential", 0, {60.0, 20.0});
    };
    for (const auto& [name, value, measured] :
         {std::tuple{std::string{"site"}, siteOf(plain),
                     siteOf(bumped) - siteOf(plain)},
          std::tuple{std::string{"geoid"}, geoidOf(plain),
                     geoidOf(bumped) - geoidOf(plain)}}) {
      EXPECT_LT(Gap(MeasuredValue(out, name), value), 1e-10)
          << label << " " << name;
      const auto kernels =
          pm::ReadSurfaceFile(out / (name + "_ice_kernels.nc"));
      EXPECT_FALSE(kernels.variables.contains("rate_dual"));
      const auto predicted = SurfacePairing(
          kernels, kernels.variables.at("thickness_dual").values, bump);
      std::cout << std::format(
          "[static, {}, {}] ice kernel {:.3e} from solve's difference\n", label,
          name, Gap(predicted, measured));
      RecordProperty(label + "_" + name + "_gap",
                     std::format("{:.3e}", Gap(predicted, measured)));
      EXPECT_LT(Gap(predicted, measured), 1e-6) << label << " " << name;
    }
  }
}

TEST_F(AppsTest, AnOnsetAtANodeHoldsTheCoSeismicResponse) {
  // A point source under the ball of a 1e19 Pa s mantle without an ocean
  // whose onset is a node after the start, a cap load and a tide acting
  // through it: the library repeats the node (its left limit without the
  // source), and the outputs hold one record per node of the window, the
  // onset's the co-seismic response. Under the source alone the run from
  // the onset on is the run started at the onset, to rounding, nothing
  // moving before it. Under all three the kernels are over the window's
  // nodes: the measurement dated at the onset reads the co-seismic field;
  // the moment kernels from the onset on predict a solve run with the
  // tensor's member rt raised (the source's moment at its onset node the
  // co-seismic one); the load kernels, at the onset the two sides' summed,
  // predict a solve run with the cap doubled, and the tide's a run with
  // the tide's real part doubled.
  namespace pm = planetmodel;
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  const auto config = [&](const std::string& name, const std::string& times,
                          double rt, double cap, double tide) {
    const auto loads =
        cap == 0.0
            ? std::string{}
            : std::format(R"(, "load": [{{"cap": {{"latitude": 35,
                "longitude": 45, "width_deg": 15, "amplitude_kg_m2": {:.17g}}},
                "history": [[0, 0.5], [10, 1.5]]}}],
              "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": {:.17g},
                "imaginary_m2_s2": 0.5}}])",
                          cap, tide);
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static", "model": {{"deck": "soft.deck"}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [{}]}},
      "forcing": {{"seismic": [{{"point": {{"latitude": 30, "longitude": 40,
          "depth_km": 100, "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20,
          "pp": -7e20, "rt": {:.17g}, "rp": -7e20, "tp": 6e20}}}},
          "onset": 2.5}}]{}}},
      "rotation": "none", "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete",
        "parameters": ["load", "tide", "moment"],
        "measurements": [
          {{"type": "displacement_point", "name": "coseismic",
            "latitude": 28, "longitude": 40, "time": 2.5}},
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 33, "longitude": 42,
            "direction": {{"east": 1, "north": -0.5}}, "time": 6}}],
        "output": "out/{}_kernels"}}}})",
                                             times, rt, loads, name, name));
  };
  const auto window = std::string{"0, 1, 2.5, 3, 6, 10"};
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };

  // -- the source alone, against the run started at its onset
  const auto summary =
      app::RunSolve(config("late", window, 3e20, 0.0, 0.0));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 6 nodes 0 to 10 yr on the run's clock"))
      << summary;
  static_cast<void>(
      app::RunSolve(config("started", "2.5, 3, 6, 10", 3e20, 0.0, 0.0)));
  const auto late = read("late");
  const auto started = read("started");
  ASSERT_EQ(late.time->size(), 6u);
  for (auto n = std::size_t{0}; n < 2; ++n) {
    for (const auto v : Slice(late, "displacement", n)) EXPECT_EQ(v, 0.0) << n;
  }
  auto worst = 0.0;
  for (auto k = std::size_t{0}; k < 4; ++k) {
    for (const auto* name : {"displacement", "tangential_potential",
                             "potential", "toroidal_potential"}) {
      worst = std::max(worst, Departure(Slice(late, name, 2 + k),
                                        Slice(started, name, k)));
    }
  }
  RecordProperty("onset_against_started", std::format("{:.3e}", worst));
  // measured: 1.5e-12 (the potential's smallest coefficients)
  EXPECT_LT(worst, 1e-10);

  // -- the kernels under the source, the load and the tide
  const auto base = config("run", window, 3e20, 1e4, 1.0);
  static_cast<void>(app::RunKernel(base));
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(config("raised", window, 4e20, 1e4, 1.0)));
  static_cast<void>(app::RunSolve(config("heavier", window, 3e20, 2e4, 1.0)));
  static_cast<void>(app::RunSolve(config("real", window, 3e20, 1e4, 2.0)));
  const auto run = read("run");
  ASSERT_EQ(run.time->size(), 6u);
  const auto direction = SurfaceDirection{.east = 1.0, .north = -0.5};
  const auto valueOf = [&](const pm::SurfaceFile& file, const std::string& name) {
    return name == "drift"
               ? VectorPointOf(file, 4, {33.0, 42.0}, direction)
               : PointValueOf(file, "displacement", 2, {28.0, 40.0});
  };
  auto applied = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < 6; ++n) applied.push_back(FieldAt(run, "load", n));
  const auto out = dir_ / "out" / "run_kernels";
  for (const auto* name : {"coseismic", "drift"}) {
    const auto label = std::string{name};
    const auto value = valueOf(run, label);
    // the measurement at the onset reads the co-seismic field
    EXPECT_LT(Gap(MeasuredValue(out, label), value), 1e-10) << label;
    const auto rows = TableRows(out / (label + "_moment_kernels.txt"));
    ASSERT_EQ(rows.size(), 6u);
    auto moment = 0.0;
    for (const auto& row : rows) {
      if (row[1] >= 2.5) moment += row[6] * 1e20;
    }
    const auto loads = pm::ReadSurfaceFile(out / (label + "_load_kernels.nc"));
    ASSERT_EQ(loads.time->size(), 6u);
    const auto load = NodePairing(loads, "load_dual", applied);
    auto tide = 0.0;
    for (const auto& row : TableRows(out / (label + "_tide_kernels.txt"))) {
      if (row[2] == 2.0 && row[3] == 1.0) tide += row[4] * 1.0;
    }
    for (const auto& [term, predicted] :
         {std::pair{std::string{"raised"}, moment},
          std::pair{std::string{"heavier"}, load},
          std::pair{std::string{"real"}, tide}}) {
      const auto measured = valueOf(read(term), label) - value;
      ASSERT_NE(measured, 0.0) << label << ", " << term;
      RecordProperty(label + "_" + term + "_gap",
                     std::format("{:.3e}", Gap(predicted, measured)));
      // measured: 3.6e-11 at worst (the moment's at the onset); the moment
      // kernel with the left side's added at the onset misses by 0.36
      EXPECT_LT(Gap(predicted, measured), 1e-8) << label << ", " << term;
    }
  }
}

TEST_F(AppsTest, AnOnsetBetweenNodesIsAPairOfTheLibrarysOwn) {
  // A seismic onset strictly between two nodes of the window: the program
  // steps a pair of nodes of its own at the onset (the left limit without
  // the source, the right with it) and writes the window's nodes alone. At
  // every node of the window the run is the run whose window has a node at
  // the onset, which it repeats: the two step over one grid, to rounding
  // (the onset's solver time formed from the clock or from the node); and
  // from the onset on it is the run started at the onset, whose steps
  // after it are the same. Over 1-D runs on their own clock without an
  // ocean, under the source alone and beside a load and a tide; several
  // terms (two onsets between one pair of nodes, one of them two terms',
  // another between the next pair, one at a node, one before the window's
  // start, which acts from the start, and one after its end, which never
  // acts; the trajectory store holds the library's nodes); dated runs with
  // an ocean, frozen and migrating under a land on a time axis, which the
  // run reads at the onset's date too; a 3-D run; and a run without
  // gravity. `kernel` refuses such a run, naming the earliest such onset,
  // and an onset within the tolerance of a node is that node.
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  // the largest departure over the response fields `names` of run `got`'s
  // nodes from run `want`'s, each pair (got's node, want's node) over the
  // largest magnitude of want's field there; a field zero in want there
  // must be zero in got
  const auto response =
      std::vector<std::string>{"displacement", "tangential_potential",
                               "potential", "toroidal_potential"};
  const auto departure =
      [&](const std::string& got, const std::string& want,
          const std::vector<std::pair<std::size_t, std::size_t>>& nodes,
          const std::vector<std::string>& names) {
        const auto a = Output(got);
        const auto b = Output(want);
        auto out = 0.0;
        for (const auto& [i, j] : nodes) {
          for (const auto& name : names) {
            const auto x = Slice(a, name, i);
            const auto y = Slice(b, name, j);
            auto diff = 0.0, size = 0.0;
            for (auto k = std::size_t{0}; k < y.size(); ++k) {
              diff = std::max(diff, std::abs(x[k] - y[k]));
              size = std::max(size, std::abs(y[k]));
            }
            out = std::max(out, size > 0.0 ? diff / size
                                : diff > 0.0
                                    ? std::numeric_limits<double>::infinity()
                                    : 0.0);
          }
        }
        return out;
      };
  // a point source under the ball at a latitude, from an onset where given
  const auto point = [](double latitude, std::optional<double> onset,
                        std::string_view name = "onset") {
    return std::format(
        R"({{"point": {{"latitude": {}, "longitude": 40, "depth_km": 100,
            "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20, "pp": -7e20,
            "rt": 3e20, "rp": -7e20, "tp": 6e20}}}}{}}})",
        latitude, onset ? std::format(R"(, "{}": {:.17g})", name, *onset) : "");
  };
  // a run on its own clock in years without an ocean, its nodes `at`, on
  // the soft ball with the model block's further members `model` (a 3-D
  // run's viscosity file, its step's tolerances then given)
  const auto clocked = [&](const std::string& name, const std::string& at,
                           const std::string& seismic,
                           const std::string& more = "",
                           const std::string& model = "",
                           const std::string& top = "") {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "soft.deck"{}}}, "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [{}]}},
      "forcing": {{"seismic": [{}]{}}},
      "rotation": "none",
      "solver": {{"rtol": 1e-12{}}}{},
      "output": {{"surface": "out/{}.nc"}}}})",
                                             model, at, seismic, more,
                                             model.empty()
                                                 ? ""
                                                 : R"(, "step_rtol": 1e-12,
                                                   "step_max_iterations": 200)",
                                             top, name));
  };
  const auto window = std::string{"0, 1, 3, 6, 10"};
  const auto withNode = std::string{"0, 1, 2, 3, 6, 10"};
  // the window's nodes against the run with a node at the onset (2 yr)
  const auto shared = std::vector<std::pair<std::size_t, std::size_t>>{
      {0, 0}, {1, 1}, {2, 3}, {3, 4}, {4, 5}};
  const auto gate = [&](const std::string& label, double gap, double bound) {
    std::cout << std::format("[onset between nodes, {}] {:.3e}\n", label, gap);
    RecordProperty(label, std::format("{:.3e}", gap));
    EXPECT_LT(gap, bound) << label;
  };

  // -- the source alone: the window's nodes alone written, nothing before
  // the onset, the run with a node there, the run started at it
  const auto summary =
      app::RunSolve(clocked("between", window, point(30, 2.0)));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 5 nodes 0 to 10 yr on the run's clock"))
      << summary;
  const auto between = Output("between");
  ASSERT_EQ(between.time->size(), 5u);
  for (auto n = std::size_t{0}; n < 2; ++n) {
    for (const auto v : Slice(between, "displacement", n)) {
      EXPECT_EQ(v, 0.0) << n;
    }
  }
  static_cast<void>(app::RunSolve(clocked("node", withNode, point(30, 2.0))));
  static_cast<void>(
      app::RunSolve(clocked("started", "2, 3, 6, 10", point(30, 2.0))));
  // measured: 0 (one grid, the onset's solver time the node's to the bit
  // here); the bound is rounding's, the two formed otherwise
  gate("alone_against_node", departure("between", "node", shared, response),
       1e-12);
  // measured: 0 (the steps' lengths alike to the bit here)
  gate("alone_against_started",
       departure("between", "started", {{2, 1}, {3, 2}, {4, 3}}, response),
       1e-10);

  // -- beside a load and a tide, which the run reads at the onset too
  const auto loads = std::string{R"(, "load": [{"cap": {"latitude": 35,
      "longitude": 45, "width_deg": 15, "amplitude_kg_m2": 1e4},
      "history": [[0, 0.5], [10, 1.5]]}],
      "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0,
      "imaginary_m2_s2": 0.5}])"};
  static_cast<void>(
      app::RunSolve(clocked("loaded", window, point(30, 2.0), loads)));
  static_cast<void>(
      app::RunSolve(clocked("loaded_node", withNode, point(30, 2.0), loads)));
  gate("loaded_against_node",
       departure("loaded", "loaded_node", shared, response), 1e-12);

  // -- several terms: onsets at 1.5 and 2 yr between the nodes at 1 and 3
  // (2 yr two terms'), at 4.5 yr between 3 and 6, at the node at 6, before
  // the start and after the end
  const auto several = [&](std::optional<double> before, bool late) {
    auto out = point(10, 2.0) + ", " + point(20, 1.5) + ", " + point(30, 2.0) +
               ", " + point(40, 4.5) + ", " + point(50, 6.0) + ", " +
               point(60, before);
    if (late) out += ", " + point(70, 12.0);
    return out;
  };
  static_cast<void>(app::RunSolve(
      clocked("several", window, several(-1.0, true), "", "",
              R"(, "trajectory": {"directory": "store_several"})")));
  static_cast<void>(app::RunSolve(clocked(
      "several_node", "0, 1, 1.5, 2, 3, 4.5, 6, 10", several(-1.0, true))));
  // measured: 1.2e-13 (the solver time of 1.5 yr one ulp from the node's)
  gate("several_against_node",
       departure("several", "several_node",
                 {{0, 0}, {1, 1}, {2, 4}, {3, 6}, {4, 7}}, response),
       1e-12);
  // the library's grid: the window's five nodes, a pair at 1.5, 2 and 4.5
  // yr each, the node at 6 yr repeated
  const auto manifest = io::TrajectoryStore::ManifestOf(dir_ / "store_several");
  ASSERT_EQ(manifest.times.size(), 12u);
  const auto& t = manifest.times;
  for (auto k = std::size_t{1}; k < t.size(); ++k) {
    const auto pair = k == 3 || k == 5 || k == 8 || k == 10;
    EXPECT_EQ(t[k] == t[k - 1], pair) << k;
    EXPECT_GE(t[k], t[k - 1]) << k;
  }
  for (const auto& [k, yr] :
       {std::pair{2, 1.5}, std::pair{4, 2.0}, std::pair{6, 3.0},
        std::pair{7, 4.5}, std::pair{9, 6.0}}) {
    EXPECT_NEAR(t[static_cast<std::size_t>(k)] / t[1], yr, 1e-12) << k;
  }
  // an onset before the start acts from it, one after the end never acts:
  // the run without the onset and without the term
  static_cast<void>(app::RunSolve(
      clocked("several_plain", window, several(std::nullopt, false))));
  gate("several_against_plain",
       departure("several", "several_plain",
                 {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, response),
       1e-12);

  // -- `kernel` refuses an onset between nodes, naming the earliest; an
  // onset within the tolerance of a node is that node
  const auto kernels = std::string{R"(, "kernels": {"parameters": ["moment"],
      "measurements": [{"type": "displacement_point", "name": "uplift",
        "latitude": 28, "longitude": 40}], "output": "out/kernels"})"};
  ExpectRefused(
      app::ReadKernelInput,
      clocked("several_kernels", window, several(-1.0, true), "", "", kernels),
      "forcing.seismic[1].onset",
      "falls between the window's nodes 1 yr and 3 yr, where solve "
      "takes the earthquake at a node of its own, and the kernels "
      "about a run with an onset between nodes are not built");
  ExpectRefused(app::ReadKernelInput,
                clocked("outside_kernels", window, point(30, 3.0 + 1e-10), "",
                        "", kernels),
                "forcing.seismic[0].onset", "give the onset as a node");
  const auto tolerant =
      clocked("tolerant", window, point(30, 3.0 + 1e-12), "", "", kernels);
  EXPECT_NO_THROW(static_cast<void>(app::ReadKernelInput(tolerant)));
  static_cast<void>(app::RunSolve(tolerant));
  static_cast<void>(app::RunSolve(clocked("at_node", window, point(30, 3.0))));
  gate("tolerant_against_node",
       departure("tolerant", "at_node",
                 {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}, response),
       1e-12);

  // -- 3-D, under a lateral viscosity
  WriteViscosity(dir_ / "viscosity.nc",
                 ModelOf(dir_ / "soft.deck", app::ModelFormat::Gia), 1e19,
                 0.05);
  const auto lateral =
      std::string{R"(, "viscosity": {"file": "viscosity.nc"})"};
  const auto summary3d =
      app::RunSolve(clocked("lateral", window, point(30, 2.0), "", lateral));
  EXPECT_TRUE(
      summary3d.starts_with("3-D (lateral viscosity), viscoelastic, 5 "
                            "nodes 0 to 10 yr on the run's clock"))
      << summary3d;
  static_cast<void>(app::RunSolve(
      clocked("lateral_node", withNode, point(30, 2.0), "", lateral)));
  // measured: 0 (the coupled iterations alike, one grid)
  gate("lateral_against_node",
       departure("lateral", "lateral_node", shared, response), 1e-10);

  // -- without gravity: the displacement's fields, the body having no
  // potential perturbation
  static_cast<void>(app::RunSolve(clocked("weightless", window, point(30, 2.0),
                                          "", "", R"(, "gravity": "none")")));
  static_cast<void>(
      app::RunSolve(clocked("weightless_node", withNode, point(30, 2.0), "", "",
                            R"(, "gravity": "none")")));
  gate(
      "weightless_against_node",
      departure("weightless", "weightless_node", shared,
                {"displacement", "tangential_potential", "toroidal_potential"}),
      1e-12);

  // -- dated, with an ocean under the toy ice history: frozen, and
  // migrating under a land that shrinks between 1.5 and 1 kyr BP, read at
  // the onset's date (1.3 kyr BP) as at a node's
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  WriteToyNodes(dir_ / "shrinking.nc", "land", [](std::size_t k, double theta) {
    const auto wide = theta > 1.9 && theta < 2.5;
    const auto narrow = theta > 2.05 && theta < 2.35;
    return (k <= 1 ? wide : narrow) ? 1.0 : 0.0;
  });
  const auto dated = [&](const std::string& name, const std::string& at,
                         const std::string& ocean,
                         const std::string& more = "") {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static", "times": {{"at_kyr_bp": [{}]}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "seismic": [{}]}},
      "ocean": {{"at": "start", {}"topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}}{},
      "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, at,
                                             point(30, 1.3, "onset_kyr_bp"),
                                             ocean, more, name));
  };
  const auto ocean = std::vector<std::string>{
      "displacement", "tangential_potential", "potential", "toroidal_potential",
      "relative_sea_level_change"};
  const auto datedShared = std::vector<std::pair<std::size_t, std::size_t>>{
      {0, 0}, {1, 1}, {2, 3}, {3, 4}, {4, 5}};
  for (const auto& [label, members] :
       {std::pair{std::string{"frozen"}, std::string{}},
        std::pair{std::string{"migrating"},
                  std::string{R"("shorelines": "migrating",
                    "land": {"file": "shrinking.nc"}, )"}}}) {
    static_cast<void>(
        app::RunSolve(dated(label, "2, 1.5, 1, 0.5, 0", members)));
    static_cast<void>(app::RunSolve(
        dated(label + "_node", "2, 1.5, 1.3, 1, 0.5, 0", members)));
    auto names = ocean;
    if (label == "migrating") names.push_back("ocean_function");
    // measured: 4.2e-12 frozen, 4.8e-12 migrating (the onset's solver time
    // from the clock, the listed node's from its date: a step's response
    // moves with the last bit of its length)
    gate(label + "_against_node",
         departure(label, label + "_node", datedShared, names), 1e-10);
  }
  ExpectRefused(app::ReadKernelInput,
                dated("dated_kernels", "2, 1.5, 1, 0.5, 0", "", kernels),
                "forcing.seismic[0].onset_kyr_bp",
                "falls between the window's nodes 1.5 kyr BP and 1 kyr BP");
}

TEST_F(AppsTest, KernelsOfAPostSeismicRunOnItsOwnClock) {
  // A point source under the ball of a 1e19 Pa s mantle without an ocean,
  // its relaxation over ten years on the run's own clock at uneven nodes:
  // the kernels in the source's moment tensor and in the viscosity of a
  // vector displacement at the surface at a time of the clock and of the
  // radial displacement at the last node. Their values are solve's. The
  // run is linear in the source, which acts from its onset, the node at 1
  // yr (repeated by the library), at every node from there: a solve run
  // with the tensor's member rt raised differs from the base run by the
  // moment kernels' prediction. The viscosity kernels against central
  // differences of solve on the ball with its viscosity scaled by
  // 1 +- 1e-4, through the zero-length step at the onset.
  namespace pm = planetmodel;
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  static_cast<void>(Write("thick.deck", BallDeck(0.0, 1e19 * (1.0 + 1e-4))));
  static_cast<void>(Write("thin.deck", BallDeck(0.0, 1e19 * (1.0 - 1e-4))));
  const auto config = [&](const std::string& name, const std::string& deck,
                          double rt) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static", "model": {{"deck": "{}"}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [0, 1, 2.5, 3, 6, 10]}},
      "forcing": {{"seismic": [{{"point": {{"latitude": 30, "longitude": 40,
          "depth_km": 100, "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20,
          "pp": -7e20, "rt": {:.17g}, "rp": -7e20, "tp": 6e20}}}},
          "onset": 1}}]}},
      "rotation": "none", "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete", "parameters": ["moment", "viscosity"],
        "measurements": [
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 33, "longitude": 42,
            "direction": {{"east": 1, "north": -0.5}}, "time": 6}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 28,
            "longitude": 40}}],
        "output": "out/{}_kernels"}}}})",
                                             deck, rt, name, name));
  };
  const auto base = config("run", "soft.deck", 3e20);
  const auto summary = app::RunKernel(base);
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 6 nodes 0 to 10 yr on the run's clock, no ocean, "
      "rotation none; seismic (1 source); wrote 2 measurement(s) of kernels"))
      << summary;
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(config("raised", "soft.deck", 4e20)));
  static_cast<void>(app::RunSolve(config("thick", "thick.deck", 3e20)));
  static_cast<void>(app::RunSolve(config("thin", "thin.deck", 3e20)));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto run = read("run");
  const auto raised = read("raised");
  const auto thick = read("thick");
  const auto thin = read("thin");
  const auto direction = SurfaceDirection{.east = 1.0, .north = -0.5};
  const auto valueOf = [&](const pm::SurfaceFile& file,
                           const std::string& name) {
    return name == "drift"
               ? VectorPointOf(file, 4, {33.0, 42.0}, direction)
               : PointValueOf(file, "displacement", 5, {28.0, 40.0});
  };
  const auto out = dir_ / "out" / "run_kernels";
  for (const auto* name : {"drift", "uplift"}) {
    const auto value = valueOf(run, name);
    EXPECT_LT(Gap(MeasuredValue(out, name), value), 1e-10) << name;
    // the moment kernels: the source from its onset, the nodes from 1 yr
    const auto rows = TableRows(out / (std::string{name} + "_moment_kernels.txt"));
    ASSERT_EQ(rows.size(), 6u);
    auto predicted = 0.0;
    for (const auto& row : rows) {
      ASSERT_EQ(row.size(), 9u);
      if (row[1] >= 1.0) predicted += row[6] * 1e20;
    }
    const auto measured = valueOf(raised, name) - value;
    // the viscosity kernels against the deck's viscosity scaled
    auto paired = 0.0;
    for (const auto& row : TableRows(out / (std::string{name} +
                                            "_viscosity_kernel.txt"))) {
      ASSERT_EQ(row.size(), 3u);
      paired += row[2] * row[1] * 1e19;
    }
    const auto differenced = (valueOf(thick, name) - valueOf(thin, name)) / 2e-4;
    std::cout << std::format(
        "[post-seismic, {}] moment kernel {:.3e} from solve's difference; "
        "viscosity {:.3e} from solve's differences\n",
        name, Gap(predicted, measured), Gap(paired, differenced));
    RecordProperty(std::string{name} + "_moment_gap",
                   std::format("{:.3e}", Gap(predicted, measured)));
    RecordProperty(std::string{name} + "_viscosity_gap",
                   std::format("{:.3e}", Gap(paired, differenced)));
    EXPECT_LT(Gap(predicted, measured), 1e-8) << name;
    EXPECT_LT(Gap(paired, differenced), 1e-5) << name;
  }
  // the drift at 6 yr has no kernel at the nodes after it
  for (const auto& row : TableRows(out / "drift_moment_kernels.txt")) {
    if (row[1] <= 6.0) continue;
    for (auto c = std::size_t{3}; c < 9; ++c) EXPECT_EQ(row[c], 0.0);
  }
}

TEST_F(AppsTest, KernelsOfAPostSeismicRunIn3D) {
  // A point source under the ball of a 1e19 Pa s mantle with a lateral
  // log-viscosity of 0.05 cos(theta), without an ocean, over six years on
  // the run's own clock from its onset at the node at 1 yr (repeated by the
  // library): the 3-D kernels in the source's moment tensor and in the log
  // viscosity of a vector displacement at a time of the clock.
  // Its value is solve's; a solve run with the tensor's member rt raised
  // differs by the moment kernels' prediction; and the log-viscosity
  // kernels paired with a uniform perturbation (the file's
  // `uniform_log_viscosity_pairing`) against central differences of solve
  // with the file's viscosity scaled by 1 +- 1e-3, which is that
  // perturbation exactly, and paired with the run's lateral one
  // (`lateral_log_viscosity_pairing`) against central differences of
  // solve with the file's anomaly scaled by 1 +- 1e-3, which samples the
  // anomaly through the file's lateral rule in the viscosity, not its
  // logarithm, and so moves the sampled log anomaly by its scale to about
  // 1.3e-2 times the anomaly's amplitude. The discrete sweep is the exact
  // transpose; the continuous one is held to its commute error at these
  // steps, which are near the Maxwell time.
  namespace pm = planetmodel;
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  const auto model = ModelOf(dir_ / "soft.deck", app::ModelFormat::Gia);
  constexpr auto kAnomaly = 0.05;
  for (const auto& [name, eta, amplitude] :
       {std::tuple{"viscosity.nc", 1e19, kAnomaly},
        std::tuple{"wide.nc", 1e19, kAnomaly * (1.0 + 1e-3)},
        std::tuple{"narrow.nc", 1e19, kAnomaly * (1.0 - 1e-3)},
        std::tuple{"heavy.nc", 1e19 * (1.0 + 1e-3), kAnomaly},
        std::tuple{"light.nc", 1e19 * (1.0 - 1e-3), kAnomaly}}) {
    WriteViscosity(dir_ / name, model, eta, amplitude);
  }
  const auto config = [&](const std::string& name, const std::string& file,
                          double rt, const std::string& method) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "soft.deck", "viscosity": {{"file": "{}"}}}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "at": [0, 1, 3, 6]}},
      "forcing": {{"seismic": [{{"point": {{"latitude": 30, "longitude": 40,
          "depth_km": 100, "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20,
          "pp": -7e20, "rt": {:.17g}, "rp": -7e20, "tp": 6e20}}}},
          "onset": 1}}]}},
      "rotation": "none",
      "solver": {{"rtol": 1e-12, "step_rtol": 1e-12,
                 "step_max_iterations": 200}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["moment", "viscosity"],
        "measurements": [
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 33, "longitude": 42,
            "direction": {{"east": 1, "north": -0.5, "up": 0.3}},
            "time": 3}}],
        "output": "out/{}_kernels"}}}})",
                                             file, rt, name, method, name));
  };
  const auto direction =
      SurfaceDirection{.east = 1.0, .north = -0.5, .up = 0.3};
  const auto valueOf = [&](const std::string& name) {
    return VectorPointOf(pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc")),
                         2, {33.0, 42.0}, direction);
  };
  const auto base = config("discrete", "viscosity.nc", 3e20, "discrete");
  const auto summary = app::RunKernel(base);
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity), viscoelastic, 4 nodes 0 to 6 yr on the run's "
      "clock, no ocean"))
      << summary;
  static_cast<void>(app::RunKernel(config("continuous", "viscosity.nc", 3e20,
                                          "continuous")));
  static_cast<void>(app::RunSolve(base));
  for (const auto& [name, file, rt] :
       {std::tuple{"raised", "viscosity.nc", 4e20},
        std::tuple{"wide", "wide.nc", 3e20},
        std::tuple{"narrow", "narrow.nc", 3e20},
        std::tuple{"heavy", "heavy.nc", 3e20},
        std::tuple{"light", "light.nc", 3e20}}) {
    static_cast<void>(app::RunSolve(config(name, file, rt, "discrete")));
  }
  const auto value = valueOf("discrete");
  const auto raisedBy = valueOf("raised") - value;
  // d ln eta = ln(1 +- 1e-3) everywhere
  const auto uniformly =
      (valueOf("heavy") - valueOf("light")) /
      (std::log(1.0 + 1e-3) - std::log(1.0 - 1e-3));
  const auto laterally = (valueOf("wide") - valueOf("narrow")) / 2e-3;
  for (const auto* method : {"discrete", "continuous"}) {
    const auto label = std::string{method};
    const auto out = dir_ / "out" / (label + "_kernels");
    EXPECT_LT(Gap(MeasuredValue(out, "drift"), value), 1e-10) << method;
    auto predicted = 0.0;
    for (const auto& row : TableRows(out / "drift_moment_kernels.txt")) {
      if (row[1] >= 1.0) predicted += row[6] * 1e20;
    }
    const auto kernels = pm::ReadModelFile(out / "drift_viscosity_kernels.nc");
    const auto pairing = [&](const std::string& key) {
      return std::get<std::vector<double>>(kernels.Meta().at(key)).front();
    };
    const auto uniform = pairing("uniform_log_viscosity_pairing");
    const auto lateral = pairing("lateral_log_viscosity_pairing");
    std::cout << std::format(
        "[post-seismic, 3-D, {}] moment kernel {:.3e} from solve's "
        "difference; log viscosity {:.3e} (uniform), {:.3e} (lateral) from "
        "solve's differences\n",
        label, Gap(predicted, raisedBy), Gap(uniform, uniformly),
        Gap(lateral, laterally));
    RecordProperty(label + "_moment_gap",
                   std::format("{:.3e}", Gap(predicted, raisedBy)));
    RecordProperty(label + "_uniform_viscosity_gap",
                   std::format("{:.3e}", Gap(uniform, uniformly)));
    RecordProperty(label + "_lateral_viscosity_gap",
                   std::format("{:.3e}", Gap(lateral, laterally)));
    const auto loose = label == "continuous";
    EXPECT_LT(Gap(predicted, raisedBy), loose ? 0.05 : 1e-8) << method;
    EXPECT_LT(Gap(uniform, uniformly), loose ? 0.1 : 5e-6) << method;
    EXPECT_LT(Gap(lateral, laterally), loose ? 0.1 : 2e-2 * kAnomaly)
        << method;
  }
}

TEST_F(AppsTest, KernelsOnTheRunsClockAreTheDatedRunsKernels) {
  // One run stated twice: dated, from 2 kyr BP to the present, and on its
  // own clock in years, from 0 to 2000 yr with the state dated at 2 kyr
  // BP, under a held change present from the start: a sea level cap mean
  // at a node and the decay time of a site's series over a window, each
  // instant in its form. The values and the ice kernels agree to the last
  // bits of the node times (a decay time's in seconds, its tau in each
  // run's unit).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  sl3dpp::test::WriteToyChange(dir_ / "held.nc", false);
  const auto config = [&](const std::string& name, const std::string& times,
                          const std::string& ocean,
                          const std::string& measurements) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static", "times": {},
      "forcing": {{"ice_change": [{{"change": {{"file": "held.nc"}}}}]}},
      "ocean": {{{}"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "solver": {{"rtol": 1e-12}},
      "kernels": {{"method": "discrete", "parameters": ["ice"],
        "measurements": [{}], "output": "out/{}"}}}})",
                                             kBall, times, ocean,
                                             measurements, name));
  };
  static_cast<void>(app::RunKernel(config(
      "dated", R"({"start_kyr_bp": 2, "end_kyr_bp": 0, "nodes": 9})", "",
      R"({"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
          "longitude": 10, "width_deg": 25, "date_kyr_bp": 0.5},
         {"type": "decay_time", "name": "rebound", "latitude": 80,
          "longitude": 0, "width_deg": 15, "window_start_kyr_bp": 1.5,
          "time_uncertainty_kyr": 0.1, "value_uncertainty_m": 0.1})")));
  const auto summary = app::RunKernel(config(
      "clock", R"({"unit": "yr", "start": 0, "end": 2000, "nodes": 9})",
      R"("date_kyr_bp": 2, )",
      R"({"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
          "longitude": 10, "width_deg": 25, "time": 1500},
         {"type": "decay_time", "name": "rebound", "latitude": 80,
          "longitude": 0, "width_deg": 15, "window_start": 500,
          "time_uncertainty": 100, "value_uncertainty_m": 0.1})"));
  EXPECT_NE(summary.find("9 nodes 0 to 2000 yr on the run's clock"),
            std::string::npos)
      << summary;
  const auto dated = dir_ / "out" / "dated";
  const auto clock = dir_ / "out" / "clock";
  for (const auto* name : {"site", "rebound"}) {
    EXPECT_LT(Gap(MeasuredValue(clock, name), MeasuredValue(dated, name)),
              1e-10)
        << name;
    const auto a =
        pm::ReadSurfaceFile(clock / (std::string{name} + "_ice_kernels.nc"));
    const auto b =
        pm::ReadSurfaceFile(dated / (std::string{name} + "_ice_kernels.nc"));
    for (const auto* field : {"thickness_dual", "rate_dual",
                              "instantaneous_dual"}) {
      EXPECT_LT(Departure(a.variables.at(field).values,
                          b.variables.at(field).values),
                1e-9)
          << name << " " << field;
    }
    EXPECT_EQ(std::get<std::string>(a.meta.at("time_unit")), "yr");
  }
  // the decay's tau in each run's unit
  const auto tauOf = [](const fs::path& directory) {
    auto in = std::ifstream(directory / "measurements.txt");
    for (std::string line; std::getline(in, line);) {
      if (!line.starts_with("rebound ")) continue;
      auto words = std::istringstream(line);
      auto name = std::string{}, type = std::string{};
      auto value = 0.0, tau = 0.0;
      words >> name >> type >> value >> tau;
      return tau;
    }
    return 0.0;
  };
  EXPECT_LT(Gap(tauOf(clock), 1000.0 * tauOf(dated)), 1e-10);
}

TEST_F(AppsTest, KernelsOfLoadsAndATideBesideTheIce) {
  // The toy state through time over listed dates, frozen shorelines, under
  // the ice history, a cap of sediment outside the water budget laid as a
  // layer on the solid surface (ramped up from the start), a disk of water
  // in the budget and a tide coefficient: the kernels in the ice, the
  // loads and the tide of a sea level cap mean at the present and of a
  // potential at a point at an earlier node. Their values are solve's. The
  // run is linear in each term: solve runs with the sediment, the water,
  // the tide's real part or its imaginary part doubled differ from the
  // base run by the kernels paired with the term (the loads as solve
  // applied them, by their properties, at every node).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = [&](const std::string& name, double sediment,
                          double water, double real, double imaginary) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"at_kyr_bp": [2, 1.6, 1, 0.3, 0]}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "load": [
          {{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                    "amplitude_kg_m2": {:.17g}}},
           "water_budget": false, "moves_solid_surface": true,
           "density_kg_m3": 2000, "history_kyr_bp": [[2, 0], [0.5, 1]]}},
          {{"disk": {{"latitude": -20, "longitude": 100, "radius_deg": 25,
                     "amplitude_kg_m2": {:.17g}}},
           "water_budget": true, "moves_solid_surface": false}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": {:.17g},
                  "imaginary_m2_s2": {:.17g}}}]}},
      "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete",
        "parameters": ["ice", "load", "tide"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": 20, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, sediment, water, real,
                                             imaginary, name, name));
  };
  const auto base = config("run", 5e4, 3000, 1.0, 0.5);
  const auto summary = app::RunKernel(base);
  EXPECT_NE(summary.find("5 nodes 2 to 0 kyr BP, shorelines frozen"),
            std::string::npos)
      << summary;
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(config("sediment", 1e5, 3000, 1.0, 0.5)));
  static_cast<void>(app::RunSolve(config("water", 5e4, 6000, 1.0, 0.5)));
  static_cast<void>(app::RunSolve(config("real", 5e4, 3000, 2.0, 0.5)));
  static_cast<void>(app::RunSolve(config("imaginary", 5e4, 3000, 1.0, 1.0)));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto run = read("run");
  const auto valueOf = [](const pm::SurfaceFile& file, const std::string& name) {
    return name == "site" ? CapMeanOf(file, "relative_sea_level_change", 4,
                                      {-30.0, 10.0}, 25.0)
                          : PointValueOf(file, "potential", 2, {60.0, 20.0});
  };
  // the loads solve applied at every node, by their properties
  const auto applied = [&](const std::string& field) {
    auto out = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < 5; ++n) out.push_back(FieldAt(run, field, n));
    return out;
  };
  const auto out = dir_ / "out" / "run_kernels";
  for (const auto* name : {"site", "geoid"}) {
    const auto label = std::string{name};
    const auto value = valueOf(run, label);
    EXPECT_LT(Gap(MeasuredValue(out, label), value), 1e-10) << label;
    const auto loads = pm::ReadSurfaceFile(out / (label + "_load_kernels.nc"));
    EXPECT_EQ(std::get<std::vector<double>>(loads.meta.at("dates_kyr_bp")),
              (std::vector<double>{2, 1.6, 1, 0.3, 0}));
    const auto sediment =
        NodePairing(loads, "load_outside_water_budget_dual",
                    applied("load_outside_water_budget")) +
        NodePairing(loads, "solid_surface_change_dual",
                    applied("solid_surface_change"));
    const auto water =
        NodePairing(loads, "load_dual", applied("load_in_water_budget"));
    auto real = 0.0, imaginary = 0.0;
    for (const auto& row : TableRows(out / (label + "_tide_kernels.txt"))) {
      if (row[2] != 2.0 || row[3] != 1.0) continue;
      real += row[4] * 1.0;
      imaginary += row[5] * 0.5;
    }
    for (const auto& [term, predicted] :
         {std::pair{std::string{"sediment"}, sediment},
          std::pair{std::string{"water"}, water},
          std::pair{std::string{"real"}, real},
          std::pair{std::string{"imaginary"}, imaginary}}) {
      const auto measured = valueOf(read(term), label) - value;
      std::cout << std::format(
          "[loads and a tide, {}] {} kernel {:.3e} from solve's difference\n",
          label, term, Gap(predicted, measured));
      RecordProperty(label + "_" + term + "_gap",
                     std::format("{:.3e}", Gap(predicted, measured)));
      EXPECT_LT(Gap(predicted, measured), 1e-7) << label << " " << term;
    }
  }
}

TEST_F(AppsTest, KernelsOfATideUnderMigratingShorelines) {
  // The marine toy history under migrating shorelines beside a tide
  // coefficient, 1-D and 3-D: kernel gives the kernels in the tide of a
  // sea level cap mean at the present and of a potential at a point at an
  // earlier node, about the run's own shorelines, by the discrete sweep.
  // Their values are solve's, and so are their derivatives: the run is
  // linear in the tide while no cell flips, so solve runs with the tide's
  // real part or its imaginary part doubled, whose ocean functions are the
  // base run's, differ from it by the kernels paired with the term. The
  // runs are at lmax 7, whose grid has no node on the toy world's
  // equatorial coast, at sea level, where any change flips a cell.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  const auto config = [&](const std::string& name, const std::string& model,
                          double real, double imaginary) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"at_kyr_bp": [2, 1.6, 1, 0.3, 0]}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": {:.17g},
                  "imaginary_m2_s2": {:.17g}}}]}},
      "ocean": {{"at": "start", "shorelines": "migrating",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12{}}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete",
        "parameters": ["ice", "tide"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": 20, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             model, real, imaginary,
                                             model.find("viscosity") ==
                                                     std::string::npos
                                                 ? ""
                                                 : R"(, "step_rtol": 1e-12,
                                                      "step_max_iterations": 200)",
                                             name, name));
  };
  const auto atSeven = [](std::string block) {
    const auto eight = std::string{R"("lmax": 8)"};
    block.replace(block.find(eight), eight.size(), R"("lmax": 7)");
    return block;
  };
  for (const auto& [dimension, model] :
       {std::pair{std::string{"1d"}, atSeven(kBall)},
        std::pair{std::string{"3d"}, atSeven(Prem8())}}) {
    const auto at = [&](const std::string& name) { return dimension + "_" + name; };
    const auto base = config(at("run"), model, 20.0, 10.0);
    const auto summary = app::RunKernel(base);
    EXPECT_NE(summary.find("5 nodes 2 to 0 kyr BP, shorelines migrating"),
              std::string::npos)
        << summary;
    static_cast<void>(app::RunSolve(base));
    static_cast<void>(app::RunSolve(config(at("real"), model, 40.0, 10.0)));
    static_cast<void>(app::RunSolve(config(at("imaginary"), model, 20.0, 20.0)));
    const auto read = [&](const std::string& name) {
      return pm::ReadSurfaceFile(dir_ / "out" / (at(name) + ".nc"));
    };
    const auto run = read("run");
    for (const auto* moved : {"real", "imaginary"}) {
      EXPECT_EQ(read(moved).variables.at("ocean_function").values,
                run.variables.at("ocean_function").values)
          << dimension << " " << moved << ": a cell flipped";
    }
    const auto valueOf = [](const pm::SurfaceFile& file, const std::string& name) {
      return name == "site" ? CapMeanOf(file, "relative_sea_level_change", 4,
                                        {-30.0, 10.0}, 25.0, 7)
                            : PointValueOf(file, "potential", 2, {60.0, 20.0}, 7);
    };
    const auto out = dir_ / "out" / (at("run") + "_kernels");
    for (const auto* name : {"site", "geoid"}) {
      const auto label = dimension + " " + name;
      const auto value = valueOf(run, name);
      EXPECT_LT(Gap(MeasuredValue(out, name), value), 1e-10) << label;
      auto real = 0.0, imaginary = 0.0;
      for (const auto& row : TableRows(out / (std::string{name} + "_tide_kernels.txt"))) {
        if (row[2] != 2.0 || row[3] != 1.0) continue;
        real += row[4] * 20.0;
        imaginary += row[5] * 10.0;
      }
      for (const auto& [term, predicted] :
           {std::pair{std::string{"real"}, real},
            std::pair{std::string{"imaginary"}, imaginary}}) {
        const auto measured = valueOf(read(term), name) - value;
        std::cout << std::format(
            "[a tide under migrating shorelines, {}] {} kernel {:.3e} from solve's "
            "difference\n",
            label, term, Gap(predicted, measured));
        RecordProperty(dimension + "_" + name + "_" + term + "_gap",
                       std::format("{:.3e}", Gap(predicted, measured)));
        EXPECT_LT(Gap(predicted, measured), 1e-7) << label << " " << term;
      }
    }
  }
}

/// The factor of an amplitude history at a date (pairs [date, factor] from
/// the oldest), linear between its dates and held beyond them.
double FactorAt(const std::vector<std::array<double, 2>>& history,
                double date) {
  if (date >= history.front()[0]) return history.front()[1];
  for (auto k = std::size_t{1}; k < history.size(); ++k) {
    if (date >= history[k][0]) {
      const auto& [d0, f0] = history[k - 1];
      const auto& [d1, f1] = history[k];
      return f0 + (f1 - f0) * (d0 - date) / (d0 - d1);
    }
  }
  return history.back()[1];
}

TEST_F(AppsTest, ContinuousTideAndMomentKernelsConverge) {
  // The continuous sweep's kernels in a prescribed tide and in a point
  // source's moment tensor, 1-D, on two runs: the toy state through time
  // under frozen shorelines and its ice history beside a tide coefficient
  // whose amplitude history rises and falls (its kink at a node of every
  // grid), measured in a sea level cap mean at the present and a
  // potential at 1 kyr BP; and a post-seismic run of the ball of a 1e19
  // Pa s mantle without an ocean on its own clock, a point source held
  // from the start, measured in a vector displacement at 4 yr and an
  // uplift at the end. Each run is linear in the tide and the source, so
  // a solve run with the coefficient's real or imaginary part, or the
  // tensor's member rt, raised differs from the base run by the exact
  // derivative of the discrete run; the continuous kernels carry their
  // commute error against it, which falls at second order as the steps
  // are halved (by 3.5 to 4 here, from 7e-2 at most on the coarsest grid
  // to 6e-3 on the finest), held to the library gate's ratios over three
  // grids, 2 and then 3, and its 5 per cent on the finest.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  WriteToyHistory(dir_ / "history.nc");
  const auto tideHistory =
      std::vector<std::array<double, 2>>{{2.0, 0.3}, {1.0, 1.0}, {0.0, 0.5}};
  const auto tide = [&](const std::string& name, int nodes, double real,
                        double imaginary, const std::string& method) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2, "end_kyr_bp": 0, "nodes": {}}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": {:.17g},
                  "imaginary_m2_s2": {:.17g},
                  "history_kyr_bp": [[2, 0.3], [1, 1], [0, 0.5]]}}]}},
      "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["tide"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": 20, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                                             kBall, nodes, real, imaginary,
                                             name, method, name));
  };
  const auto seismic = [&](const std::string& name, int nodes, double rt,
                           const std::string& method) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static", "model": {{"deck": "soft.deck"}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "yr", "start": 0, "end": 8, "nodes": {}}},
      "forcing": {{"seismic": [{{"point": {{"latitude": 30, "longitude": 40,
          "depth_km": 100, "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20,
          "pp": -7e20, "rt": {:.17g}, "rp": -7e20, "tp": 6e20}}}}}}]}},
      "rotation": "none", "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["moment"],
        "measurements": [
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 33, "longitude": 42,
            "direction": {{"east": 1, "north": -0.5}}, "time": 4}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 28,
            "longitude": 40}}],
        "output": "out/{}_kernels"}}}})",
                                             nodes, rt, name, method, name));
  };
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  // each kernel's gap at each grid
  auto gaps = std::map<std::string, std::vector<double>>{};
  for (const auto nodes : {5, 9, 17}) {
    const auto tag = std::format("n{}", nodes);
    const auto last = static_cast<std::size_t>(nodes - 1);
    // the tide
    const auto tideRun = tag + "_tide";
    const auto summary =
        app::RunKernel(tide(tideRun, nodes, 1.0, 0.5, "continuous"));
    EXPECT_NE(summary.find("(tide; continuous adjoint, its longest step "),
              std::string::npos)
        << summary;
    static_cast<void>(
        app::RunSolve(tide(tideRun, nodes, 1.0, 0.5, "continuous")));
    static_cast<void>(
        app::RunSolve(tide(tideRun + "_real", nodes, 2.0, 0.5, "continuous")));
    static_cast<void>(app::RunSolve(
        tide(tideRun + "_imaginary", nodes, 1.0, 1.0, "continuous")));
    const auto tideValue = [&](const pm::SurfaceFile& file,
                               const std::string& name) {
      return name == "site"
                 ? CapMeanOf(file, "relative_sea_level_change", last,
                             {-30.0, 10.0}, 25.0)
                 : PointValueOf(file, "potential", last / 2, {60.0, 20.0});
    };
    for (const auto* name : {"site", "geoid"}) {
      const auto label = std::string{name};
      const auto out = dir_ / "out" / (tideRun + "_kernels");
      const auto value = tideValue(read(tideRun), label);
      EXPECT_LT(Gap(MeasuredValue(out, label), value), 1e-10) << label;
      // a raised part acts at each node by the history's factor there
      auto real = 0.0, imaginary = 0.0;
      for (const auto& row : TableRows(out / (label + "_tide_kernels.txt"))) {
        if (row[2] != 2.0 || row[3] != 1.0) continue;
        const auto factor = FactorAt(tideHistory, row[1]);
        real += row[4] * factor * 1.0;
        imaginary += row[5] * factor * 0.5;
      }
      gaps["tide real, " + label].push_back(
          Gap(real, tideValue(read(tideRun + "_real"), label) - value));
      gaps["tide imaginary, " + label].push_back(Gap(
          imaginary, tideValue(read(tideRun + "_imaginary"), label) - value));
    }
    // the moment
    const auto sourceRun = tag + "_source";
    static_cast<void>(
        app::RunKernel(seismic(sourceRun, nodes, 3e20, "continuous")));
    static_cast<void>(
        app::RunSolve(seismic(sourceRun, nodes, 3e20, "continuous")));
    static_cast<void>(app::RunSolve(
        seismic(sourceRun + "_raised", nodes, 4e20, "continuous")));
    const auto sourceValue = [&](const pm::SurfaceFile& file,
                                 const std::string& name) {
      return name == "drift"
                 ? VectorPointOf(file, last / 2, {33.0, 42.0},
                                 {.east = 1.0, .north = -0.5})
                 : PointValueOf(file, "displacement", last, {28.0, 40.0});
    };
    for (const auto* name : {"drift", "uplift"}) {
      const auto label = std::string{name};
      const auto out = dir_ / "out" / (sourceRun + "_kernels");
      const auto value = sourceValue(read(sourceRun), label);
      EXPECT_LT(Gap(MeasuredValue(out, label), value), 1e-10) << label;
      // the source acts at every node, held from the start
      auto predicted = 0.0;
      for (const auto& row : TableRows(out / (label + "_moment_kernels.txt"))) {
        predicted += row[6] * 1e20;
      }
      gaps["moment, " + label].push_back(Gap(
          predicted, sourceValue(read(sourceRun + "_raised"), label) - value));
    }
  }
  for (const auto& [what, gap] : gaps) {
    ASSERT_EQ(gap.size(), 3u);
    std::cout << std::format(
        "[continuous, 1-D] {} kernel {:.3e}, {:.3e}, {:.3e} from solve's "
        "differences at 5, 9 and 17 nodes\n",
        what, gap[0], gap[1], gap[2]);
    RecordProperty(what,
                   std::format("{:.3e} {:.3e} {:.3e}", gap[0], gap[1], gap[2]));
    EXPECT_LT(gap[2], 0.05) << what;
    EXPECT_GT(gap[0] / gap[1], 2.0) << what << ": " << gap[0] << " " << gap[1];
    EXPECT_GT(gap[1] / gap[2], 3.0) << what << ": " << gap[1] << " " << gap[2];
  }
}

TEST_F(AppsTest, LoadKernelsUnderMigratingShorelines) {
  // The toy state through time over listed dates under migrating
  // shorelines, its polar ice in a basin below sea level floating off as it
  // thins (the shorelines move through the run, checked), the ice history
  // beside three loads: a disk of water in the
  // budget held from the start, a cap of mass outside the budget on land
  // and a layer of sediment on the ocean floor (outside the budget, moving
  // the solid surface), the last two ramped up from the start. The kernels
  // in the loads of a sea level cap mean at the present and of a potential
  // at a point at an earlier node. The run is linear in the loads while no
  // cell flips, so the central differences of solve runs with one term
  // scaled by 1 +- 0.1, between which no cell flips (checked), are the
  // derivative, and equal the kernels paired with the term's change as
  // solve applied it, by its properties, at every node: the water with the
  // load kernel, the mass with the kernel outside the budget, the layer
  // with that kernel in its mass and the solid surface's in its height.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  constexpr auto kWater = 3000.0, kMass = 2e4, kSediment = 5e4;
  const auto config = [&](const std::string& name, double water, double mass,
                          double sediment) {
    return Write(name + ".json",
                 std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"at_kyr_bp": [2, 1.6, 1, 0.3, 0]}},
      "forcing": {{
        "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "load": [
          {{"disk": {{"latitude": -20, "longitude": 100, "radius_deg": 25,
                     "amplitude_kg_m2": {:.17g}}},
           "water_budget": true, "moves_solid_surface": false}},
          {{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                    "amplitude_kg_m2": {:.17g}}},
           "water_budget": false, "moves_solid_surface": false,
           "history_kyr_bp": [[2, 0], [0.5, 1]]}},
          {{"cap": {{"latitude": -45, "longitude": -60, "width_deg": 15,
                    "amplitude_kg_m2": {:.17g}}},
           "water_budget": false, "moves_solid_surface": true,
           "density_kg_m3": 2000, "history_kyr_bp": [[2, 0], [1, 1]]}}]}},
      "ocean": {{"at": "start", "shorelines": "migrating",
                "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "discrete", "parameters": ["ice", "load"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": 20, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                             kBall, water, mass, sediment, name, name));
  };
  const auto base = config("run", kWater, kMass, kSediment);
  const auto summary = app::RunKernel(base);
  EXPECT_NE(summary.find("5 nodes 2 to 0 kyr BP, shorelines migrating"),
            std::string::npos)
      << summary;
  static_cast<void>(app::RunSolve(base));
  for (const auto& [name, water, mass, sediment] :
       {std::tuple{"water_up", 1.1 * kWater, kMass, kSediment},
        std::tuple{"water_down", 0.9 * kWater, kMass, kSediment},
        std::tuple{"mass_up", kWater, 1.1 * kMass, kSediment},
        std::tuple{"mass_down", kWater, 0.9 * kMass, kSediment},
        std::tuple{"sediment_up", kWater, kMass, 1.1 * kSediment},
        std::tuple{"sediment_down", kWater, kMass, 0.9 * kSediment}}) {
    static_cast<void>(app::RunSolve(config(name, water, mass, sediment)));
  }
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto run = read("run");
  EXPECT_NE(Slice(run, "ocean_function", 0), Slice(run, "ocean_function", 4))
      << "no cell flipped through the run";
  const auto valueOf = [](const pm::SurfaceFile& file,
                          const std::string& name) {
    return name == "site" ? CapMeanOf(file, "relative_sea_level_change", 4,
                                      {-30.0, 10.0}, 25.0)
                          : PointValueOf(file, "potential", 2, {60.0, 20.0});
  };
  // a load solve applied, by its properties: its half change between two
  // runs at every node
  const auto halfChange = [](const pm::SurfaceFile& up,
                             const pm::SurfaceFile& down,
                             const std::string& field) {
    auto out = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < 5; ++n) {
      auto a = FieldAt(up, field, n);
      const auto b = FieldAt(down, field, n);
      for (auto i = std::size_t{0}; i < a.size(); ++i)
        a[i] = 0.5 * (a[i] - b[i]);
      out.push_back(std::move(a));
    }
    return out;
  };
  const auto out = dir_ / "out" / "run_kernels";
  for (const auto* term : {"water", "mass", "sediment"}) {
    const auto label = std::string{term};
    const auto up = read(label + "_up");
    const auto down = read(label + "_down");
    EXPECT_EQ(up.variables.at("ocean_function").values,
              run.variables.at("ocean_function").values)
        << label << ": a cell flipped";
    EXPECT_EQ(down.variables.at("ocean_function").values,
              run.variables.at("ocean_function").values)
        << label << ": a cell flipped";
    for (const auto* name : {"site", "geoid"}) {
      const auto measurement = std::string{name};
      const auto loads =
          pm::ReadSurfaceFile(out / (measurement + "_load_kernels.nc"));
      auto predicted = 0.0;
      if (label == "water") {
        predicted = NodePairing(loads, "load_dual",
                                halfChange(up, down, "load_in_water_budget"));
      } else {
        predicted =
            NodePairing(loads, "load_outside_water_budget_dual",
                        halfChange(up, down, "load_outside_water_budget"));
        if (label == "sediment") {
          predicted +=
              NodePairing(loads, "solid_surface_change_dual",
                          halfChange(up, down, "solid_surface_change"));
        }
      }
      const auto measured =
          0.5 * (valueOf(up, measurement) - valueOf(down, measurement));
      std::cout << std::format(
          "[loads under migrating shorelines, {}] {} kernel {:.3e} from "
          "solve's differences\n",
          measurement, label, Gap(predicted, measured));
      RecordProperty(measurement + "_" + label + "_gap",
                     std::format("{:.3e}", Gap(predicted, measured)));
      EXPECT_LT(Gap(predicted, measured), 1e-7) << measurement << " " << label;
    }
  }
  for (const auto* name : {"site", "geoid"}) {
    EXPECT_LT(Gap(MeasuredValue(out, name), valueOf(run, name)), 1e-10) << name;
  }
}

TEST_F(AppsTest, KernelsOfTheStaticProblemOnLateralModuli) {
  // The static problem without an ocean on the ball with lateral shear and
  // bulk moduli, under a cap load and a point source: the 3-D kernels of a
  // radial displacement over a cap, of a vector displacement at a point
  // and of one smoothed over a cap (their toroidal parts the lateral
  // structure's), in the load, the source's
  // moment tensor and the moduli. Their values are solve's; each is its
  // load and moment kernels paired with the load applied and the tensor,
  // the run linear in both, and a solve run with the tensor's member rt
  // raised differs by the moment kernel's prediction; and the moduli
  // kernels paired with the run's own lateral perturbation (the file's
  // `lateral_moduli_pairing`) against central differences of solve with
  // that perturbation scaled by 1 +- 1e-3.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  for (const auto& [name, scale] :
       {std::pair{"moduli.nc", 1.0}, std::pair{"stiff.nc", 1.0 + 1e-3},
        std::pair{"soft.nc", 1.0 - 1e-3}}) {
    ASSERT_TRUE(io::WriteModuli(
        dir_ / name, model, SurfaceGrid(8),
        [scale](double, double theta, double) {
          return scale * 0.1 * std::cos(theta);
        },
        [scale](double, double theta, double phi) {
          return scale * 0.05 * std::sin(theta) * std::cos(phi);
        },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  const auto config = [&](const std::string& name, const std::string& moduli,
                          double rt = 3e20) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "ball.deck", "moduli": {{"file": "{}"}}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
          "width_deg": 20, "amplitude_kg_m2": 1.0e6}}}}],
        "seismic": [{{"point": {{"latitude": 45, "longitude": -70,
          "depth_km": 100, "moment_tensor_nm": {{"rr": 9e20, "tt": -2e20,
          "pp": -7e20, "rt": {:.17g}, "rp": -7e20, "tp": 6e20}}}}}}]}},
      "rotation": "none",
      "solver": {{"rtol": 1e-12, "step_rtol": 1e-12,
                 "step_max_iterations": 200}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"parameters": ["load", "moment", "moduli"],
        "measurements": [
          {{"type": "displacement_cap", "name": "uplift", "latitude": 55,
            "longitude": -80, "width_deg": 10}},
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 40, "longitude": -60,
            "direction": {{"east": 1, "north": 1}}}},
          {{"type": "displacement_vector_cap", "name": "swept",
            "latitude": 40, "longitude": -60, "width_deg": 10,
            "direction": {{"east": 1, "north": 1, "up": 0.2}}}}],
        "output": "out/{}_kernels"}}}})",
                                             moduli, rt, name, name));
  };
  const auto base = config("run", "moduli.nc");
  const auto summary = app::RunKernel(base);
  EXPECT_TRUE(
      summary.starts_with("3-D (lateral moduli), elastic, static, no ocean"))
      << summary;
  static_cast<void>(app::RunSolve(base));
  static_cast<void>(app::RunSolve(config("stiff", "stiff.nc")));
  static_cast<void>(app::RunSolve(config("soft", "soft.nc")));
  static_cast<void>(app::RunSolve(config("raised", "moduli.nc", 1.03e22)));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto run = read("run");
  const auto stiff = read("stiff");
  const auto soft = read("soft");
  const auto raised = read("raised");
  const auto valueOf = [](const pm::SurfaceFile& file,
                          const std::string& name) {
    if (name == "uplift") {
      return CapMeanOf(file, "displacement", 0, {55.0, -80.0}, 10.0);
    }
    return name == "drift"
               ? VectorPointOf(file, 0, {40.0, -60.0},
                               {.east = 1.0, .north = 1.0})
               : VectorCapOf(file, 0, {40.0, -60.0}, 10.0,
                             {.east = 1.0, .north = 1.0, .up = 0.2});
  };
  EXPECT_TRUE(run.variables.contains("toroidal_potential"));
  const auto out = dir_ / "out" / "run_kernels";
  const auto& applied = run.variables.at("load_outside_water_budget").values;
  for (const auto* name : {"uplift", "drift", "swept"}) {
    const auto label = std::string{name};
    const auto value = valueOf(run, label);
    EXPECT_LT(Gap(MeasuredValue(out, label), value), 1e-9) << label;
    // the run is linear in the load and the source together: J is their
    // kernels paired with them
    const auto loads = pm::ReadSurfaceFile(out / (label + "_load_kernels.nc"));
    const auto moments = TableRows(out / (label + "_moment_kernels.txt"));
    EXPECT_EQ(moments.size(), 1u);
    const auto& D = moments.front();
    const auto bySource = D[3] * 9e20 + D[4] * -2e20 + D[5] * -7e20 +
                          D[6] * 3e20 + D[7] * -7e20 + D[8] * 6e20;
    const auto paired =
        SurfacePairing(loads, loads.variables.at("load_dual").values, applied) +
        bySource;
    const auto raisedBy = valueOf(raised, label) - value;
    EXPECT_LT(Gap(D[6] * 1e22, raisedBy), 1e-8) << label;
    const auto moduli = pm::ReadModelFile(out / (label + "_moduli_kernels.nc"));
    const auto predicted = std::get<std::vector<double>>(
                               moduli.Meta().at("lateral_moduli_pairing"))
                               .front();
    auto held = false;
    for (const auto& part : moduli.Parts()) {
      held = held || (part.variables.contains("K_kappa") &&
                      part.variables.contains("K_mu"));
    }
    EXPECT_TRUE(held) << label;
    const auto measured = (valueOf(stiff, label) - valueOf(soft, label)) / 2e-3;
    std::cout << std::format(
        "[static, lateral moduli, {}] load and moment kernels {:.3e} from the "
        "value; moment {:.3e} from solve's difference; moduli {:.3e} from "
        "solve's differences\n",
        label, Gap(paired, value), Gap(D[6] * 1e22, raisedBy),
        Gap(predicted, measured));
    RecordProperty(label + "_load_gap", std::format("{:.3e}", Gap(paired, value)));
    RecordProperty(label + "_moduli_gap",
                   std::format("{:.3e}", Gap(predicted, measured)));
    EXPECT_LT(Gap(paired, value), 1e-8) << label;
    EXPECT_LT(Gap(predicted, measured), 1e-5) << label;
  }
}

/// A layer of LayeredDeck: its radii, its moduli (Pa) and its viscosity
/// (Pa s; not a number where it is elastic).
struct DeckLayer {
  double bottom, top, kappa, mu, viscosity;
};

/// The layers of LayeredDeck: an inner body to 3000 km of viscosity
/// 1e22 Pa s, a mantle to 5700 km of 1e21 Pa s and an elastic lithosphere
/// to the ball's surface, each uniform, of the ball's density and each of
/// its own shear modulus; the mantle's shear modulus scaled by
/// 1 + `mantleMu` with its viscosity alike (its relaxation time held, and
/// its bulk modulus), the lithosphere's bulk modulus by
/// 1 + `lithosphereKappa` (its shear modulus held).
std::array<DeckLayer, 3> LayersOf(double mantleMu, double lithosphereKappa) {
  constexpr auto rho = 5500.0;
  const auto kappaOf = [](double vp, double vs) {
    return rho * (vp * vp - 4.0 / 3.0 * vs * vs);
  };
  return {DeckLayer{0.0, 3000e3, kappaOf(10000.0, 5500.0),
                    rho * 5500.0 * 5500.0, 1e22},
          DeckLayer{3000e3, 5700e3, kappaOf(10000.0, 5200.0),
                    (1.0 + mantleMu) * rho * 5200.0 * 5200.0,
                    (1.0 + mantleMu) * 1e21},
          DeckLayer{5700e3, 6371e3,
                    (1.0 + lithosphereKappa) * kappaOf(9000.0, 5000.0),
                    rho * 5000.0 * 5000.0, std::nan("")}};
}

/// The GIA deck of LayersOf: two knots per layer, a repeated radius at
/// each boundary, the wave speeds from the moduli (rho vs^2 = mu,
/// rho vp^2 = kappa + 4 mu / 3).
std::string LayeredDeck(double mantleMu = 0.0, double lithosphereKappa = 0.0) {
  constexpr auto rho = 5500.0;
  auto out = std::string{"layered test ball\ngia deck for the sl3d tests\n"};
  for (const auto& layer : LayersOf(mantleMu, lithosphereKappa)) {
    const auto vs = std::sqrt(layer.mu / rho);
    const auto vp = std::sqrt((layer.kappa + 4.0 / 3.0 * layer.mu) / rho);
    const auto viscosity = std::isnan(layer.viscosity)
                               ? std::string{"nan"}
                               : std::format("{:.17g}", layer.viscosity);
    for (const auto r : {layer.bottom, layer.top}) {
      out += std::format("{} 5500 {:.17g} {:.17g} {:.17g} {:.17g} 1 {}\n", r,
                         vp, vs, vp, vs, viscosity);
    }
  }
  return out;
}

/// The pairing of a 1-D moduli kernel table with a relative perturbation
/// of one layer's modulus: the rows at that layer's memory points (told
/// by the layer's own shear modulus, which each row lists, a boundary's
/// point once per element), sum W_m3 K mu_Pa for the shear modulus
/// (`shear`), sum W_m3 K kappa_Pa for the bulk. `rows` counts the rows
/// paired.
double LayerPairing(const fs::path& table, const DeckLayer& layer, bool shear,
                    std::size_t* rows = nullptr) {
  auto out = 0.0;
  auto count = std::size_t{0};
  for (const auto& row : TableRows(table)) {
    EXPECT_EQ(row.size(), 6u);
    if (std::abs(row[3] - layer.mu) > 1e-9 * layer.mu) continue;
    out += row[1] * (shear ? row[5] * row[3] : row[4] * row[2]);
    ++count;
  }
  if (rows != nullptr) *rows = count;
  return out;
}

TEST_F(AppsTest, ModuliKernelsOfARunThroughTime) {
  // The toy state through time under frozen shorelines on a layered radial
  // Earth (LayersOf): the kernels in the moduli of a sea level cap mean at
  // the present and of an uplift at 1 kyr BP. They are with respect to
  // kappa and the unrelaxed mu with the relaxation times held, so each is
  // paired with a relative change of one layer's modulus and checked
  // against central differences of solve on decks scaled by 1 +- 1e-4
  // there: the mantle's shear modulus with its viscosity alike (its
  // relaxation time held), and the elastic lithosphere's bulk modulus. The
  // discrete sweep is the derivative of the run to the differences' error
  // (under 3e-7 here); the continuous one carries its commute error, 2e-2
  // at most at 250-yr steps (about a Maxwell time of the mantle), falling
  // under step halving by 3.3 to 3.9 here, held to the library gate's
  // ratio, 1.5.
  namespace pm = planetmodel;
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kStep = 1e-4;
  for (const auto& [name, mu, kappa] :
       {std::tuple{"layered", 0.0, 0.0}, std::tuple{"stiff", kStep, 0.0},
        std::tuple{"soft", -kStep, 0.0}, std::tuple{"firm", 0.0, kStep},
        std::tuple{"loose", 0.0, -kStep}}) {
    static_cast<void>(
        Write(std::string{name} + ".deck", LayeredDeck(mu, kappa)));
  }
  const auto layers = LayersOf(0.0, 0.0);
  const auto& mantle = layers[1];
  const auto& lithosphere = layers[2];
  const auto config = [&](const std::string& name, const std::string& deck,
                          int nodes, const std::string& method) {
    return Write(name + ".json",
                 std::format(R"({{
      "model": {{"deck": "{}.deck"}}, "discretisation": {{"lmax": 8}}, {},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["moduli"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": 10, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                             deck, Window(nodes), name, method, name));
  };
  const auto valueOf = [&](const std::string& run, const std::string& name,
                           int nodes) {
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / (run + ".nc"));
    const auto last = static_cast<std::size_t>(nodes - 1);
    return name == "site"
               ? CapMeanOf(file, "relative_sea_level_change", last,
                           {-30.0, 10.0}, 25.0)
               : PointValueOf(file, "displacement", last / 2, {55.0, 10.0});
  };
  // the continuous gaps at each resolution, per measurement and layer
  auto continuous = std::map<std::string, std::vector<double>>{};
  for (const auto nodes : {9, 17}) {
    const auto tag = std::format("n{}", nodes);
    const auto methods =
        nodes == 9 ? std::vector<std::string>{"discrete", "continuous"}
                   : std::vector<std::string>{"continuous"};
    for (const auto& method : methods) {
      const auto summary =
          app::RunKernel(config(tag + "_" + method, "layered", nodes, method));
      // the continuous sweep states its steps against the shortest
      // relaxation time
      EXPECT_NE(summary.find(std::format("(moduli; {} adjoint{}", method,
                                         method == "continuous"
                                             ? ", its longest step "
                                             : ")")),
                std::string::npos)
          << summary;
    }
    static_cast<void>(app::RunSolve(config(tag, "layered", nodes, "discrete")));
    for (const auto* deck : {"stiff", "soft", "firm", "loose"}) {
      static_cast<void>(
          app::RunSolve(config(tag + "_" + deck, deck, nodes, "discrete")));
    }
    for (const auto* name : {"site", "uplift"}) {
      const auto label = std::string{name};
      const auto byMu = (valueOf(tag + "_stiff", label, nodes) -
                         valueOf(tag + "_soft", label, nodes)) /
                        (2.0 * kStep);
      const auto byKappa = (valueOf(tag + "_firm", label, nodes) -
                            valueOf(tag + "_loose", label, nodes)) /
                           (2.0 * kStep);
      for (const auto& method : methods) {
        const auto out = dir_ / "out" / (tag + "_" + method + "_kernels");
        EXPECT_LT(Gap(MeasuredValue(out, label), valueOf(tag, label, nodes)),
                  1e-10)
            << label << " " << method;
        const auto table = out / (label + "_moduli_kernel.txt");
        auto paired = std::size_t{0};
        const auto mu = LayerPairing(table, mantle, true, &paired);
        EXPECT_GT(paired, 0u);
        const auto kappa = LayerPairing(table, lithosphere, false, &paired);
        EXPECT_GT(paired, 0u);
        std::cout << std::format(
            "[moduli through time, 1-D, {} nodes, {}, {}] mantle mu {:.3e}, "
            "lithosphere kappa {:.3e} from solve's differences\n",
            nodes, method, label, Gap(mu, byMu), Gap(kappa, byKappa));
        RecordProperty(std::format("{}_{}_{}_mu_gap", tag, method, label),
                       std::format("{:.3e}", Gap(mu, byMu)));
        RecordProperty(std::format("{}_{}_{}_kappa_gap", tag, method, label),
                       std::format("{:.3e}", Gap(kappa, byKappa)));
        if (method == "discrete") {
          EXPECT_LT(Gap(mu, byMu), 1e-6) << label;
          EXPECT_LT(Gap(kappa, byKappa), 1e-6) << label;
        } else {
          continuous[label + " mu"].push_back(Gap(mu, byMu));
          continuous[label + " kappa"].push_back(Gap(kappa, byKappa));
        }
      }
    }
  }
  for (const auto& [what, gaps] : continuous) {
    ASSERT_EQ(gaps.size(), 2u);
    EXPECT_LT(gaps[0], 0.1) << what;
    EXPECT_GT(gaps[0] / gaps[1], 1.5)
        << what << ": " << gaps[0] << " " << gaps[1];
  }
}

TEST_F(AppsTest, ModuliKernelsOfARunThroughTimeOnLateralModuli) {
  // The toy state through time under frozen shorelines on the ball with
  // the deck's radial viscosity and lateral shear and bulk moduli, a 3-D
  // run: the kernels in the moduli of a sea level cap mean at the present
  // and of an uplift at 1 kyr BP, paired with the run's own lateral
  // perturbation (the file's `lateral_moduli_pairing`), against central
  // differences of solve with that perturbation scaled by 1 +- 1e-3 (the
  // branch moduli and the relaxation times scale as the kernels hold
  // them). The discrete sweep is the derivative of the run to the
  // differences' error (under 1e-6 here); the continuous one carries its
  // commute error, 5e-2 at most at these 500-yr steps, held to a tenth
  // (its fall under step halving is the library gate's,
  // DiscreteAdjoint3D.ContinuousMaterialKernelsAgreeUnderLateralModuli; it
  // falls by 3.3 to 3.9 here at 250-yr steps).
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  for (const auto& [name, scale] :
       {std::pair{"moduli.nc", 1.0}, std::pair{"stiff.nc", 1.0 + 1e-3},
        std::pair{"soft.nc", 1.0 - 1e-3}}) {
    ASSERT_TRUE(io::WriteModuli(
        dir_ / name, model, SurfaceGrid(8),
        [scale](double, double theta, double) {
          return scale * 0.1 * std::cos(theta);
        },
        [scale](double, double theta, double phi) {
          return scale * 0.05 * std::sin(theta) * std::cos(phi);
        },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  const auto config = [&](const std::string& name, const std::string& moduli,
                          const std::string& method) {
    return Write(name + ".json",
                 std::format(R"({{
      "model": {{"deck": "ball.deck", "moduli": {{"file": "{}"}}}},
      "discretisation": {{"lmax": 8}}, {},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-12, "step_rtol": 1e-12,
                 "step_max_iterations": 200}},
      "output": {{"surface": "out/{}.nc"}},
      "kernels": {{"method": "{}", "parameters": ["moduli"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}},
          {{"type": "displacement_point", "name": "uplift", "latitude": 55,
            "longitude": 10, "date_kyr_bp": 1}}],
        "output": "out/{}_kernels"}}}})",
                             moduli, Window(5), name, method, name));
  };
  const auto valueOf = [&](const std::string& run, const std::string& name) {
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / (run + ".nc"));
    return name == "site" ? CapMeanOf(file, "relative_sea_level_change", 4,
                                      {-30.0, 10.0}, 25.0)
                          : PointValueOf(file, "displacement", 2, {55.0, 10.0});
  };
  for (const auto* method : {"discrete", "continuous"}) {
    const auto summary = app::RunKernel(config(method, "moduli.nc", method));
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral moduli), viscoelastic, 5 nodes 2 to 0 kyr BP, "
        "shorelines frozen"))
        << summary;
  }
  static_cast<void>(app::RunSolve(config("run", "moduli.nc", "discrete")));
  static_cast<void>(app::RunSolve(config("stiff", "stiff.nc", "discrete")));
  static_cast<void>(app::RunSolve(config("soft", "soft.nc", "discrete")));
  for (const auto* name : {"site", "uplift"}) {
    const auto label = std::string{name};
    const auto measured =
        (valueOf("stiff", label) - valueOf("soft", label)) / 2e-3;
    for (const auto* method : {"discrete", "continuous"}) {
      const auto out = dir_ / "out" / (std::string{method} + "_kernels");
      EXPECT_LT(Gap(MeasuredValue(out, label), valueOf("run", label)), 1e-9)
          << label << " " << method;
      const auto moduli =
          pm::ReadModelFile(out / (label + "_moduli_kernels.nc"));
      const auto predicted = std::get<std::vector<double>>(
                                 moduli.Meta().at("lateral_moduli_pairing"))
                                 .front();
      std::cout << std::format(
          "[moduli through time, 3-D, {}, {}] lateral moduli {:.3e} from "
          "solve's differences\n",
          method, label, Gap(predicted, measured));
      RecordProperty(std::format("{}_{}_moduli_gap", method, label),
                     std::format("{:.3e}", Gap(predicted, measured)));
      EXPECT_LT(Gap(predicted, measured),
                std::string_view{method} == "discrete" ? 1e-5 : 0.1)
          << label << " " << method;
    }
  }
}

TEST_F(AppsTest, OneConfigServesSolveAndKernel) {
  // A solve config with a kernels block: solve checks the block, a fault
  // in it refused before anything runs, and writes what it writes without
  // it; kernel checks solve's output, checkpoint and restart, a fault in
  // them refused, and writes its kernels alone, no surface file and no
  // checkpoint.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto body = std::string{kBall} + ", " + Window(3) +
                    R"(, "rotation": "none",
      "checkpoint": {"directory": "ckpt", "every": 1}, "restart": false)";
  const auto config = [&](const std::string& name, const std::string& output,
                          const std::string& kernels) {
    return Write(name + ".json",
                 std::format(R"({{{}, "output": {}, "kernels": {}}})", body,
                             output, kernels));
  };
  constexpr auto kOutput =
      R"({"surface": "out/run.nc", "exchange": "out/run_{node}.txt"})";
  constexpr auto kKernels = R"({"parameters": ["ice", "load"],
      "measurements": [{"type": "rotation", "name": "lod",
                        "component": "z"}],
      "output": "out/kernels"})";
  const auto both = config("both", kOutput, kKernels);
  const auto solved = app::RunSolve(both);
  EXPECT_NE(solved.find("; wrote "), std::string::npos) << solved;
  EXPECT_EQ(solved.find("kernels"), std::string::npos) << solved;
  EXPECT_TRUE(fs::exists(dir_ / "out" / "run.nc"));
  EXPECT_TRUE(fs::exists(dir_ / "out" / "run_2.txt"));
  EXPECT_FALSE(fs::exists(dir_ / "out" / "kernels"));
  fs::remove_all(dir_ / "out");
  fs::remove_all(dir_ / "ckpt");
  const auto kerneled = app::RunKernel(both);
  EXPECT_NE(kerneled.find("wrote 1 measurement(s) of kernels"),
            std::string::npos)
      << kerneled;
  EXPECT_TRUE(fs::exists(dir_ / "out" / "kernels" / "lod_ice_kernels.nc"));
  EXPECT_TRUE(fs::exists(dir_ / "out" / "kernels" / "lod_load_kernels.nc"));
  EXPECT_FALSE(fs::exists(dir_ / "out" / "run.nc"));
  EXPECT_FALSE(fs::exists(dir_ / "ckpt"));
  // the faults each command finds in the other's members
  ExpectRefused(app::RunSolve,
                config("bad_kernels", kOutput,
                       R"({"parameters": ["stress"], "measurements": [
                           {"type": "rotation", "component": "z"}],
                           "output": "k"})"),
                "kernels.parameters[0]", "none of");
  ExpectRefused(app::RunSolve,
                config("unnamed_kernels", kOutput,
                       R"({"measurements": [
                           {"type": "rotation", "component": "z"}],
                           "output": "k"})"),
                "kernels", "names no `parameters`");
  ExpectRefused(app::RunSolve,
                config("seismic_kernels", kOutput,
                       R"({"parameters": ["seismic"], "measurements": [
                           {"type": "rotation", "component": "z"}],
                           "output": "k"})"),
                "kernels.parameters[0]", "are \"moment\"");
  ExpectRefused(app::RunSolve,
                config("bad_measurement", kOutput,
                       R"({"parameters": ["ice", "viscosity"],
                           "measurements": [{"type": "rotation",
                           "component": "z", "date_kyr_bp": 0.7}],
                           "output": "k"})"),
                "kernels.measurements[0].date_kyr_bp",
                "0.7 kyr BP is not a node of the times grid");
  ExpectRefused(app::RunKernel,
                config("bad_output", R"({"surface": "o.nc", "exchnage": "x"})",
                       kKernels),
                "output.exchnage", "not a recognised member");
  ExpectRefused(app::RunKernel,
                config("bad_exchange", R"({"surface": "o.nc",
                                          "exchange": "o.txt"})",
                       kKernels),
                "output.exchange", "{node}");
}

TEST_F(AppsTest, KernelRefusesWhatItCannotGive) {
  // Each fault of a kernel config is refused naming the member: the
  // kernels block's members at the top level, naming it, and a path for
  // the top-level `output`; no kernels block, or one that is not an
  // object or holds an unknown member; an unknown method or kernel, a
  // kernel named twice, none named, one the run cannot give (the moment
  // tensors' without seismic sources, the viscosity's in the static
  // problem), the forcing's word for the sources, no `parameters` (the
  // words the run takes listed); no measurements; a measurement's instant in
  // the static problem, a decay time there, a decay window in the other
  // form's members, a vector displacement along no direction; the
  // recording's members in the kernels block, naming the trajectory
  // block's; a stride in the static problem, and under seismic sources one
  // other than 1.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto refused = [&](const std::string& text, const std::string& member,
                           const std::string& what) {
    ExpectRefused(app::ReadKernelInput, Write("bad.json", text), member, what);
  };
  constexpr auto kLod = R"({"type": "rotation", "component": "z"})";
  // a dated run with an ocean, the kernels block's members given
  const auto wet = [&](const std::string& members) {
    return std::format(R"({{{}, {}, "rotation": "none", {}}})", kBall,
                       Window(5), members);
  };
  const auto block = [&](const std::string& members) {
    return wet(std::format(R"("kernels": {{{}, "output": "k"}})", members));
  };
  const auto measured = [&](const std::string& measurement,
                            const std::string& members = "") {
    return block(std::format(
        R"("parameters": ["ice", "viscosity"], {}"measurements": [{}])",
        members, measurement));
  };
  // the static problem without an ocean under a load
  const auto still = [&](const std::string& members) {
    return std::format(R"({{{}, "problem": "static",
        "forcing": {{"load": [{}}}]}}, "rotation": "none",
        "kernels": {{{}, "output": "k"}}}})",
                       kBall, kCapTerm, members);
  };
  // a run through time without an ocean under a point source
  const auto sourced = [&](const std::string& members) {
    return std::format(R"({{{}, "problem": "quasi_static",
        "times": {{"unit": "yr", "at": [0, 1, 2, 4]}},
        "forcing": {{"seismic": [{}"onset": 1}}]}}, "rotation": "none",
        "kernels": {{{}, "output": "k"}}}})",
                       kBall, kPointOpen, members);
  };

  for (const auto& [member, place] :
       {std::pair{R"("adjoint": "discrete")", "kernels.method"},
        std::pair{R"("measurements": [])", "kernels.measurements"},
        std::pair{R"("record_stride": 2)", "trajectory.record_stride"},
        std::pair{R"("parameters": ["ice"])", "kernels.parameters"}}) {
    const auto name =
        std::string{member}.substr(1, std::string{member}.find('"', 1) - 1);
    refused(wet(std::format(R"({}, "kernels": {{"measurements": [{}],
                "output": "k"}})",
                            member, kLod)),
            name, std::format("it belongs in `{}`", place));
  }
  refused(wet(std::format(R"("output": "k", "kernels": {{"measurements": [{}],
              "output": "k"}})",
                          kLod)),
          "output", "is a path: the kernel files' directory is "
                    "`kernels.output`");
  refused(wet(R"("output": {"surface": "o.nc"})"), "bad.json",
          "has no `kernels` block");
  refused(wet(R"("kernels": ["ice"])"), "kernels", "is not a block");
  refused(block(std::format(R"("measurements": [{}], "parameter": ["ice"])",
                            kLod)),
          "kernels.parameter", "not a recognised member");
  refused(block(std::format(R"("method": "exact", "measurements": [{}])", kLod)),
          "kernels.method", "none of \"continuous\", \"discrete\"");
  refused(block(std::format(R"("parameters": ["ice", "stress"],
                               "measurements": [{}])",
                            kLod)),
          "kernels.parameters[1]", "none of");
  refused(block(std::format(R"("parameters": ["load", "ice", "load"],
                               "measurements": [{}])",
                            kLod)),
          "kernels.parameters[2]", "names a kernel named before");
  refused(block(std::format(R"("parameters": [], "measurements": [{}])", kLod)),
          "kernels.parameters", "names no kernels");
  refused(block(std::format(R"("parameters": ["moment"],
                               "measurements": [{}])",
                            kLod)),
          "kernels.parameters[0]",
          "the moment tensors of the run's seismic sources, and the run has "
          "none");
  refused(sourced(std::format(R"("parameters": ["seismic", "load"],
                                 "measurements": [{}])",
                              kLod)),
          "kernels.parameters[0]",
          "is the forcing's word for the sources: their kernels, in the "
          "point moment tensors, are \"moment\"");
  refused(still(std::format(R"("parameters": ["viscosity"],
                               "measurements": [{}])",
                            kLod)),
          "kernels.parameters[0]",
          "is the kernels in the viscosity, which needs the quasi-static "
          "problem");
  refused(still(std::format(R"("measurements": [{}])", kLod)), "kernels",
          "names no `parameters`, the kernels to write: list them, of "
          "\"load\", \"tide\" and \"moduli\" (the words this run takes)");
  refused(block(std::format(R"("measurements": [{}])", kLod)), "kernels",
          "list them, of \"ice\", \"load\", \"tide\", \"viscosity\" and "
          "\"moduli\" (the words this run takes)");
  refused(sourced(std::format(R"("measurements": [{}])", kLod)), "kernels",
          "list them, of \"load\", \"tide\", \"moment\", \"viscosity\" "
          "and \"moduli\" (the words this run takes)");
  refused(block(R"("parameters": ["ice"], "measurements": [])"),
          "kernels.measurements", "holds no measurements");
  // the measurements against the problem and the times
  refused(still(R"("parameters": ["load"], "measurements": [
                   {"type": "rotation", "component": "z", "date_kyr_bp": 0}])"),
          "kernels.measurements[0].date_kyr_bp",
          "dates a measurement of the static problem");
  refused(still(R"("parameters": ["load"], "measurements": [
                   {"type": "displacement_point", "latitude": 0,
                    "longitude": 0, "time": 1}])"),
          "kernels.measurements[0].time",
          "dates a measurement of the static problem");
  refused(still(R"("parameters": ["load"], "measurements": [
                   {"type": "decay_time", "latitude": 80, "longitude": 0,
                    "width_deg": 15, "window_start_kyr_bp": 1,
                    "time_uncertainty_kyr": 0, "value_uncertainty_m": 1}])"),
          "kernels.measurements[0].type",
          "is the decay time of a series through time, which needs the "
          "quasi-static problem");
  refused(measured(R"({"type": "decay_time", "latitude": 80, "longitude": 0,
                       "width_deg": 15, "window_start": 1,
                       "time_uncertainty_kyr": 0, "value_uncertainty_m": 1})"),
          "kernels.measurements[0].window_start",
          "the run's nodes are dates before present: give "
          "`window_start_kyr_bp`");
  refused(sourced(R"("parameters": ["moment"], "measurements": [
                     {"type": "displacement_point", "latitude": 0,
                      "longitude": 0, "time": 1},
                     {"type": "decay_time", "latitude": 80, "longitude": 0,
                      "width_deg": 15, "window_start_kyr_bp": 1,
                      "time_uncertainty_kyr": 0, "value_uncertainty_m": 1}])"),
          "kernels.measurements[1].type",
          "is a measurement of the sea level, which a run without an ocean");
  refused(measured(R"({"type": "displacement_vector_point", "latitude": 0,
                       "longitude": 0,
                       "direction": {"east": 0, "north": 0}})"),
          "kernels.measurements[0].direction", "is the zero vector");
  refused(measured(R"({"type": "displacement_vector_cap", "latitude": 0,
                       "longitude": 0, "width_deg": 5,
                       "direction": {"west": 1}})"),
          "kernels.measurements[0].direction.west", "not a recognised member");
  // the recording: the kernels block's members are the trajectory block's
  refused(block(std::format(R"("record_stride": 2, "measurements": [{}])",
                            kLod)),
          "kernels.record_stride",
          "is the top-level `trajectory` block's, shared with solve: give "
          "`trajectory.record_stride`");
  refused(block(std::format(R"("trajectory": {{"directory": "t"}},
                               "measurements": [{}])",
                            kLod)),
          "kernels.trajectory",
          "is the top-level `trajectory` block's, shared with solve: give "
          "`trajectory.directory`");
  // the trajectory block: a stride in the static problem, under seismic
  // sources one other than 1, a member it does not know
  const auto recorded = [](const std::string& text, const std::string& block) {
    auto out = text;
    out.insert(1, std::format(R"("trajectory": {}, )", block));
    return out;
  };
  refused(recorded(still(std::format(R"("parameters": ["load"],
                                        "measurements": [{}])",
                                     kLod)),
                   R"({"record_stride": 1})"),
          "trajectory.record_stride",
          "is a stride through the nodes of a run through time, which needs "
          "the quasi-static problem");
  refused(recorded(sourced(std::format(R"("measurements": [{}])", kLod)),
                   R"({"record_stride": 2})"),
          "trajectory.record_stride",
          "is 2, and a run under seismic sources (`forcing.seismic`) is "
          "recorded at every node");
  refused(recorded(sourced(std::format(R"("measurements": [{}])", kLod)),
                   R"({"record_stride": 0})"),
          "trajectory.record_stride", "is not a positive stride");
  refused(recorded(sourced(std::format(R"("measurements": [{}])", kLod)),
                   R"({"directory": "t", "stride": 1})"),
          "trajectory.stride", "not a recognised member");
  refused(recorded(sourced(std::format(R"("measurements": [{}])", kLod)),
                   R"("t")"),
          "trajectory", "is not a block");
  // a directory under seismic sources, and in the static problem, is taken
  const auto stored = app::ReadKernelInput(Write(
      "stored.json",
      recorded(sourced(std::format(
                   R"("parameters": ["viscosity"], "measurements": [{}])",
                   kLod)),
               R"({"directory": "t"})")));
  EXPECT_EQ(stored.trajectory.directory, dir_ / "t");
  EXPECT_EQ(stored.trajectory.record_stride, 1);
  // and what is taken: the moment tensors' kernels, on the run's clock
  const auto taken = app::ReadKernelInput(Write(
      "taken.json",
      sourced(std::format(R"("parameters": ["moment", "load", "moduli"],
                             "measurements": [{}])",
                          kLod))));
  EXPECT_EQ(taken.kernels.parameters,
            (std::vector{app::KernelParameter::Load,
                         app::KernelParameter::Moment,
                         app::KernelParameter::Moduli}));
  EXPECT_EQ(taken.kernels.measurements.front().node, 3u);
}

// -- the trajectory store -------------------------------------------------------

/// Two kernel runs' outputs, file for file: the surface files' fields and
/// axes, the model files' fields and pairings, and the tables' rows, each
/// bitwise (their headers, which name the config and the time, aside):
/// the callers' runs are one process's, the same transforms throughout.
void ExpectSameKernels(const fs::path& got, const fs::path& want) {
  namespace pm = planetmodel;
  auto files = 0;
  for (const auto& entry : fs::directory_iterator(want)) {
    const auto name = entry.path().filename().string();
    const auto other = got / name;
    ASSERT_TRUE(fs::exists(other)) << other;
    ++files;
    if (name.ends_with("_viscosity_kernels.nc") ||
        name.ends_with("_moduli_kernels.nc")) {
      const auto a = pm::ReadModelFile(other);
      const auto b = pm::ReadModelFile(entry.path());
      ASSERT_EQ(a.Parts().size(), b.Parts().size()) << name;
      for (auto p = std::size_t{0}; p < b.Parts().size(); ++p) {
        for (const auto& [field, variable] : b.Parts()[p].variables) {
          EXPECT_EQ(a.Parts()[p].variables.at(field).values, variable.values)
              << name << " " << field;
        }
      }
      for (const auto& [key, value] : b.Meta()) {
        if (key.find("pairing") == std::string::npos) continue;
        EXPECT_TRUE(a.Meta().at(key) == value) << name << " " << key;
      }
    } else if (name.ends_with(".nc")) {
      const auto a = pm::ReadSurfaceFile(other);
      const auto b = pm::ReadSurfaceFile(entry.path());
      EXPECT_EQ(a.time, b.time) << name;
      EXPECT_EQ(a.instant, b.instant) << name;
      ASSERT_EQ(a.variables.size(), b.variables.size()) << name;
      for (const auto& [field, variable] : b.variables) {
        EXPECT_EQ(a.variables.at(field).values, variable.values)
            << name << " " << field;
      }
    } else {
      const auto rows = [](const fs::path& path) {
        auto in = std::ifstream(path);
        auto out = std::vector<std::string>{};
        for (std::string line; std::getline(in, line);) {
          if (!line.starts_with('#')) out.push_back(line);
        }
        return out;
      };
      const auto b = rows(entry.path());
      EXPECT_FALSE(b.empty()) << name;
      EXPECT_EQ(rows(other), b) << name;
    }
  }
  EXPECT_GT(files, 1) << want;
}

TEST_F(AppsTest, KernelsFromARecordedTrajectoryAreTheRunsInMemory) {
  // `solve` records its forward run into the trajectory store of
  // `trajectory.directory`, and `kernel`, reading the same config, sweeps
  // from it without running the forward problem: its kernels and values
  // are, bitwise, those of `kernel` running the forward problem itself in
  // memory at the same stride. Over a covering set: 1-D and 3-D, dated
  // and on the run's own clock, with an ocean and without, frozen and
  // migrating (calibrated), the static problem and runs through time,
  // under loads, a tide and seismic sources, by the discrete and the
  // continuous sweeps.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  static_cast<void>(Write("soft.deck", BallDeck(0.0, 1e19)));
  const auto soft = ModelOf(dir_ / "soft.deck", app::ModelFormat::Gia);
  WriteViscosity(dir_ / "viscosity.nc", soft, 1e19, 0.05);
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia),
      SurfaceGrid(8),
      [](double, double theta, double) { return 0.1 * std::cos(theta); },
      [](double, double theta, double phi) {
        return 0.05 * std::sin(theta) * std::cos(phi);
      },
      {.name = "toy moduli", .elementMetres = 500e3}));
  // each case: the run's members, the kernels block's (but its output) and
  // the stride (0: the static problem, which takes none)
  struct Case {
    std::string name, body, kernels;
    int stride;
  };
  const auto point = std::string{R"({"point": {"latitude": 30,
      "longitude": 40, "depth_km": 100, "moment_tensor_nm": {"rr": 9e20,
      "tt": -2e20, "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}},
      "onset": 1})"};
  const auto atNode = point.substr(0, point.rfind('1')) + "2.5}";
  const auto cases = std::vector<Case>{
      {"dated",
       std::string{kBall} + ", " + Window(9) +
           R"(, "rotation": "full", "moments": "hydrostatic")",
       R"("method": "discrete", "parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
           "longitude": 10, "width_deg": 25},
          {"type": "stokes_coefficient", "name": "j2", "degree": 2,
           "order": 0, "date_kyr_bp": 1.0},
          {"type": "decay_time", "name": "rebound", "latitude": 80,
           "longitude": 0, "width_deg": 15, "window_start_kyr_bp": 1.0,
           "time_uncertainty_kyr": 0.0, "value_uncertainty_m": 0.1}])",
       3},
      {"migrating",
       std::string{kBall} + ", " +
           Window(5, "history.nc", "", Calibrated("migrating"), "end") +
           R"(, "rotation": "full", "moments": "hydrostatic")",
       R"("method": "continuous", "parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
           "longitude": 10, "width_deg": 25, "date_kyr_bp": 0.5}])",
       2},
      {"lateral", Prem8() + ", " + Window(4) + R"(, "rotation": "full")",
       R"("method": "continuous", "parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
           "longitude": 10, "width_deg": 25}])",
       2},
      {"loads", std::string{kBall} + R"(, "problem": "quasi_static",
        "times": {"at_kyr_bp": [2, 1.6, 1, 0.3, 0]},
        "forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}],
          "load": [{"cap": {"latitude": 60, "longitude": -85,
            "width_deg": 20, "amplitude_kg_m2": 5e4}, "water_budget": false,
            "moves_solid_surface": true, "density_kg_m3": 2000,
            "history_kyr_bp": [[2, 0], [0.5, 1]]},
            {"disk": {"latitude": -20, "longitude": 100, "radius_deg": 25,
              "amplitude_kg_m2": 3000}, "water_budget": true,
              "moves_solid_surface": false}],
          "tide": [)" + std::string{kTide} + R"(]},
        "ocean": {"at": "start", "topography": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}},
        "rotation": "full", "moments": "hydrostatic")",
       R"("method": "discrete", "parameters": ["ice", "load", "tide"],
        "measurements": [
          {"type": "potential_point", "name": "geoid", "latitude": 60,
           "longitude": 20, "date_kyr_bp": 1}])",
       2},
      {"postseismic", R"("problem": "quasi_static",
        "model": {"deck": "soft.deck"}, "discretisation": {"lmax": 8},
        "times": {"unit": "yr", "at": [0, 1, 2.5, 3, 6, 10]},
        "forcing": {"seismic": [)" + point + R"(]}, "rotation": "none")",
       R"("method": "discrete", "parameters": ["moment", "viscosity"],
        "measurements": [
          {"type": "displacement_vector_point", "name": "drift",
           "latitude": 33, "longitude": 42,
           "direction": {"east": 1, "north": -0.5}, "time": 6}])",
       1},
      // the onsets at nodes (an onset between nodes has no kernels): the
      // store holds the library's repeat; here the continuous sweep, and
      // a measurement at the onset
      {"postseismic_onset", R"("problem": "quasi_static",
        "model": {"deck": "soft.deck"}, "discretisation": {"lmax": 8},
        "times": {"unit": "yr", "at": [0, 1, 2.5, 3, 6, 10]},
        "forcing": {"seismic": [)" + atNode + R"(]}, "rotation": "none")",
       R"("method": "continuous", "parameters": ["moment", "viscosity"],
        "measurements": [
          {"type": "displacement_vector_point", "name": "drift",
           "latitude": 33, "longitude": 42,
           "direction": {"east": 1, "north": -0.5}, "time": 2.5}])",
       1},
      {"postseismic3d", R"("problem": "quasi_static",
        "model": {"deck": "soft.deck", "viscosity": {"file": "viscosity.nc"}},
        "discretisation": {"lmax": 8},
        "times": {"unit": "yr", "at": [0, 1, 3, 6]},
        "forcing": {"seismic": [)" + point + R"(]}, "rotation": "none",
        "solver": {"step_rtol": 1e-12, "step_max_iterations": 200})",
       R"("method": "continuous", "parameters": ["moment", "viscosity"],
        "measurements": [
          {"type": "displacement_vector_point", "name": "drift",
           "latitude": 33, "longitude": 42,
           "direction": {"east": 1, "north": -0.5, "up": 0.3},
           "time": 3}])",
       1},
      {"static", std::string{kBall} + R"(, "problem": "static",
        "forcing": {"load": [)" + std::string{kCap} + R"(]},
        "rotation": "none")",
       R"("method": "discrete", "parameters": ["load", "moduli"],
        "measurements": [
          {"type": "displacement_vector_cap", "name": "drift",
           "latitude": 45, "longitude": -70, "width_deg": 10,
           "direction": {"east": 1, "north": 0.5, "up": 0.2}}])",
       0},
      {"static3d", R"("problem": "static",
        "model": {"deck": "ball.deck", "moduli": {"file": "moduli.nc"}},
        "discretisation": {"lmax": 8},
        "forcing": {"load": [)" + std::string{kCap} +
                       R"(], "seismic": [)" + point.substr(0, point.rfind(',')) +
                       R"(}]}, "rotation": "none",
        "solver": {"step_rtol": 1e-12, "step_max_iterations": 200})",
       R"("method": "continuous", "parameters": ["load", "moment", "moduli"],
        "measurements": [
          {"type": "displacement_point", "name": "uplift", "latitude": 55,
           "longitude": -80}])",
       0}};
  for (const auto& c : cases) {
    const auto stride =
        c.stride > 0 ? std::format(R"("record_stride": {}, )", c.stride)
                     : std::string{};
    const auto memory = Write(
        c.name + "_memory.json",
        std::format(R"({{{}, "trajectory": {{{}}},
          "kernels": {{{}, "output": "out/{}_memory"}}}})",
                    c.body, stride.empty() ? "" : stride.substr(0, stride.size() - 2),
                    c.kernels, c.name));
    const auto stored = Write(
        c.name + "_stored.json",
        std::format(R"({{{}, "trajectory": {{{}"directory": "store_{}"}},
          "output": {{"surface": "out/{}.nc"}},
          "kernels": {{{}, "output": "out/{}_stored"}}}})",
                    c.body, stride, c.name, c.name, c.kernels, c.name));
    static_cast<void>(app::RunKernel(memory));
    const auto solved = app::RunSolve(stored);
    EXPECT_NE(solved.find(std::format("; recorded the trajectory to the store {}",
                                      (dir_ / ("store_" + c.name)).string())),
              std::string::npos)
        << c.name << ": " << solved;
    const auto swept = app::RunKernel(stored);
    EXPECT_NE(swept.find(std::format(
                  "; the forward run read back from the trajectory store {}",
                  (dir_ / ("store_" + c.name)).string())),
              std::string::npos)
        << c.name << ": " << swept;
    ExpectSameKernels(dir_ / "out" / (c.name + "_stored"),
                      dir_ / "out" / (c.name + "_memory"));
  }
}

TEST_F(AppsTest, KernelRecordsItsTrajectoryAndSweepsItAgain) {
  // `kernel` with `trajectory.directory` naming nothing yet runs the
  // forward problem, records it there and leaves the store; a second
  // `kernel` run of the same run, for other measurements, sweeps from the
  // store without running the forward problem (the summary says so, and
  // no file of the store is written again), and each run's kernels are,
  // bitwise, those of the run in memory.
  WriteToyHistory(dir_ / "history.nc");
  const auto body = Prem8() + ", " + Window(5) + R"(, "rotation": "full")";
  const auto config = [&](const std::string& name,
                          const std::string& trajectory,
                          const std::string& measurement) {
    return Write(name + ".json",
                 std::format(R"({{{}, "trajectory": {{{}}},
                   "kernels": {{"method": "discrete",
                   "parameters": ["ice", "viscosity"], "measurements": [{}],
                   "output": "out/{}"}}}})",
                             body, trajectory, measurement, name));
  };
  constexpr auto kSite = R"({"type": "relative_sea_level_cap", "name": "site",
      "latitude": -30, "longitude": 10, "width_deg": 25})";
  constexpr auto kGeoid = R"({"type": "potential_point", "name": "geoid",
      "latitude": 60, "longitude": 20, "date_kyr_bp": 1})";
  constexpr auto kStored = R"("record_stride": 2, "directory": "store")";
  const auto store = dir_ / "store";
  const auto first = app::RunKernel(config("first", kStored, kSite));
  EXPECT_NE(first.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << first;
  const auto manifest = io::TrajectoryStore::ManifestOf(store);
  EXPECT_TRUE(manifest.complete);
  EXPECT_EQ(manifest.kept,
            (std::vector<bool>{true, false, true, false, true}));
  auto written = std::map<fs::path, fs::file_time_type>{};
  for (const auto& entry : fs::recursive_directory_iterator(store)) {
    written.emplace(entry.path(), fs::last_write_time(entry.path()));
  }
  const auto second = app::RunKernel(config("second", kStored, kGeoid));
  EXPECT_NE(second.find("; the forward run read back from the trajectory "
                        "store "),
            std::string::npos)
      << second;
  for (const auto& entry : fs::recursive_directory_iterator(store)) {
    EXPECT_EQ(fs::last_write_time(entry.path()), written.at(entry.path()))
        << entry.path();
  }
  static_cast<void>(
      app::RunKernel(config("first_memory", R"("record_stride": 2)", kSite)));
  static_cast<void>(
      app::RunKernel(config("second_memory", R"("record_stride": 2)", kGeoid)));
  ExpectSameKernels(dir_ / "out" / "first", dir_ / "out" / "first_memory");
  ExpectSameKernels(dir_ / "out" / "second", dir_ / "out" / "second_memory");
}

TEST_F(AppsTest, TheTrajectoryStoreIsTheRunsOwn) {
  // A store is read and recorded by its own run alone: `solve` leaves a
  // complete store of its run as it is; a store of another run (another
  // config), one recorded at another stride, and a directory holding what
  // no store writes are refused, nothing in them removed; an incomplete
  // store of the run (a run stopped at the wall-time limit) is recorded
  // whole by `kernel`, which a resumption then finds complete and leaves
  // as it is, as a run starting over does (a resumption continuing the
  // store is ARecordingRunResumesItsStore's).
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = [&](const std::string& name, const std::string& rtol,
                          const std::string& trajectory,
                          const std::string& extra = "") {
    return Write(name + ".json",
                 std::format(R"({{{}, {}, "rotation": "none",
                   "solver": {{"rtol": {}}}, "trajectory": {{{}}}{},
                   "output": {{"surface": "out/{}.nc"}},
                   "kernels": {{"parameters": ["ice", "viscosity"],
                     "measurements": [{{"type": "rotation",
                     "name": "lod", "component": "z"}}], "output": "out/k_{}"}}}})",
                             kBall, Window(5), rtol, trajectory, extra, name,
                             name));
  };
  constexpr auto kStore = R"("record_stride": 2, "directory": "store")";
  const auto store = dir_ / "store";
  const auto run = config("run", "1e-10", kStore);
  EXPECT_NE(app::RunSolve(run).find("; recorded the trajectory to the store "),
            std::string::npos);
  const auto again = app::RunSolve(run);
  EXPECT_NE(again.find(std::format("; the trajectory store {} holds this run "
                                   "complete (",
                                   store.string())),
            std::string::npos)
      << again;
  EXPECT_TRUE(again.ends_with("), left as it is")) << again;
  // another run, another stride, files no store writes
  ExpectRefused(app::RunKernel, config("other", "1e-11", kStore),
                "trajectory.directory",
                "holds the trajectory store of another run");
  ExpectRefused(app::RunSolve, config("other", "1e-11", kStore),
                "trajectory.directory",
                "holds the trajectory store of another run");
  ExpectRefused(app::RunKernel,
                config("strided", "1e-10",
                       R"("record_stride": 3, "directory": "store")"),
                "trajectory.directory",
                "holds this run recorded at record stride 2, and this run's "
                "is 3");
  EXPECT_TRUE(io::TrajectoryStore::ManifestOf(store).complete);
  fs::create_directories(dir_ / "cluttered");
  static_cast<void>(Write("cluttered/notes.txt", "not a record"));
  ExpectRefused(app::RunSolve,
                config("cluttered", "1e-10", R"("directory": "cluttered")"),
                "trajectory.directory",
                "holds what a trajectory store does not write");
  EXPECT_TRUE(fs::exists(dir_ / "cluttered" / "notes.txt"));
  // stopped at the wall-time limit after a checkpoint, the store is left
  // incomplete; a kernel run of the same run records it whole, and the
  // resumption finds it complete
  const auto checkpointed =
      R"(, "checkpoint": {"directory": "ckpt", "every": 1,
                          "max_walltime_min": 1e-9})";
  const auto stopped = app::RunSolve(
      config("stopped", "1e-10", R"("directory": "partial")", checkpointed));
  EXPECT_NE(stopped.find("at the wall-time limit"), std::string::npos)
      << stopped;
  EXPECT_NE(stopped.find(std::format("the trajectory store {} left incomplete "
                                     "at node 0, the resumption recording on "
                                     "from there",
                                     (dir_ / "partial").string())),
            std::string::npos)
      << stopped;
  EXPECT_FALSE(io::TrajectoryStore::ManifestOf(dir_ / "partial").complete);
  const auto recorded = app::RunKernel(
      config("stopped", "1e-10", R"("directory": "partial")", checkpointed));
  EXPECT_NE(recorded.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << recorded;
  EXPECT_TRUE(io::TrajectoryStore::ManifestOf(dir_ / "partial").complete);
  const auto onwards = app::RunSolve(config(
      "stopped", "1e-10", R"("directory": "partial")",
      R"(, "checkpoint": {"directory": "ckpt", "every": 1})"));
  EXPECT_NE(onwards.find(", resumed"), std::string::npos) << onwards;
  EXPECT_NE(onwards.find("holds this run complete"), std::string::npos)
      << onwards;
  // starting over records it too, a complete store of the run being left
  // as it is
  const auto over = app::RunSolve(
      config("stopped", "1e-10", R"("directory": "partial")",
             R"(, "checkpoint": {"directory": "ckpt", "every": 1},
                  "restart": false)"));
  EXPECT_NE(over.find("holds this run complete"), std::string::npos) << over;
}

/// Expects two directories to hold the same files, byte for byte.
void ExpectSameFiles(const fs::path& got, const fs::path& want) {
  const auto listed = [](const fs::path& root) {
    auto out = std::map<fs::path, std::string>{};
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      if (!entry.is_regular_file()) continue;
      auto in = std::ifstream(entry.path(), std::ios::binary);
      out.emplace(fs::relative(entry.path(), root),
                  std::string(std::istreambuf_iterator<char>(in), {}));
    }
    return out;
  };
  const auto a = listed(got);
  const auto b = listed(want);
  EXPECT_EQ(a.size(), b.size()) << got << " against " << want;
  for (const auto& [name, bytes] : b) {
    const auto found = a.find(name);
    ASSERT_NE(found, a.end()) << name;
    EXPECT_TRUE(found->second == bytes) << name;
  }
}

TEST_F(AppsTest, ARecordingRunResumesItsStore) {
  // A run recording into the trajectory store and stopped at the
  // wall-time limit after every node's checkpoint (one invocation a node)
  // continues its incomplete store from the checkpoint it resumes from:
  // the store it finishes is, file by file and byte by byte, the
  // uninterrupted run's, at a record stride the resumptions' nodes do not
  // all divide (the kept nodes count from the run's first), 1-D and 3-D,
  // frozen and migrating (calibrated, its start read back), and `kernel`
  // sweeps from it bitwise the kernels of the uninterrupted store; the
  // runs' outputs are the uninterrupted run's. A record the stopped run
  // left past its checkpoint, or half written, is replaced. A resumption
  // whose store lacks a record before its checkpoint is refused with the
  // ways on, nothing removed, and so is one that would begin a store at
  // its checkpoint.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kKernels = R"("parameters": ["ice", "viscosity"],
      "measurements": [{"type": "relative_sea_level_cap", "name": "site",
        "latitude": -30, "longitude": 10, "width_deg": 25}])";
  const auto config = [&](const std::string& name, const std::string& body,
                          const std::string& store,
                          const std::string& checkpoint) {
    return Write(name + ".json",
                 std::format(R"({{{}, "trajectory": {{"record_stride": 2,
                   "directory": "{}"}}{}, "output": {{"surface": "out/{}.nc"}},
                   "kernels": {{{}, "output": "out/k_{}"}}}})",
                             body, store, checkpoint, name, kKernels, name));
  };
  constexpr auto kStopping = R"(, "checkpoint": {{"directory": "ckpt_{}",
      "every": 1, "max_walltime_min": 1e-9}})";
  // runs a config until it finishes, the store checked incomplete at each
  // stop, the first at node `first`; the summaries
  const auto pieces = [&](const fs::path& path, const fs::path& store,
                          int first) {
    auto summaries = std::vector<std::string>{};
    for (auto attempt = 0; attempt < 10; ++attempt) {
      summaries.push_back(app::RunSolve(path));
      const auto& summary = summaries.back();
      if (summary.find("; wrote ") != std::string::npos) break;
      EXPECT_NE(summary.find(std::format(
                    "; the trajectory store {} left incomplete at node {}, ",
                    store.string(), first + attempt)),
                std::string::npos)
          << summary;
      EXPECT_FALSE(io::TrajectoryStore::ManifestOf(store).complete);
    }
    return summaries;
  };
  const auto cases = std::vector<std::pair<std::string, std::string>>{
      {"frozen1d", std::string{kBall} + ", " + Window(5) +
                       R"(, "rotation": "full", "moments": "hydrostatic")"},
      {"migrating1d",
       std::string{kBall} + ", " +
           Window(5, "history.nc", "", Calibrated("migrating"), "end") +
           R"(, "rotation": "full", "moments": "hydrostatic")"},
      {"frozen3d", Prem8() + ", " + Window(4) + R"(, "rotation": "full")"},
      {"migrating3d",
       Prem8() + ", " +
           Window(4, "history.nc", "", R"("shorelines": "migrating")") +
           R"(, "rotation": "full")"}};
  for (const auto& [name, body] : cases) {
    const auto whole = config(name + "_whole", body, "whole_" + name, "");
    const auto parted =
        config(name + "_parts", body, "parts_" + name,
               std::format(kStopping, name));
    static_cast<void>(app::RunSolve(whole));
    const auto store = dir_ / ("parts_" + name);
    auto summaries = std::vector<std::string>{};
    if (name == "frozen1d") {
      // stopped at node 0: a record past it and a half-written one, as a
      // run killed after its checkpoint leaves them, are replaced
      summaries.push_back(app::RunSolve(parted));
      fs::copy_file(store / "node_000000.surface",
                    store / "node_000003.surface");
      fs::copy_file(store / "node_000000.surface",
                    store / ".node_000001.surface.partial");
      fs::copy_file(store / "rank_0000" / "node_000000.state",
                    store / "rank_0000" / "node_000002.state");
      const auto rest = pieces(parted, store, 1);
      summaries.insert(summaries.end(), rest.begin(), rest.end());
    } else {
      summaries = pieces(parted, store, 0);
    }
    ASSERT_GE(summaries.size(), 3u) << name;
    const auto& last = summaries.back();
    EXPECT_NE(last.find(", resumed"), std::string::npos) << last;
    EXPECT_NE(last.find(std::format("; recorded the trajectory to the store {} "
                                    "(",
                                    store.string())),
              std::string::npos)
        << last;
    EXPECT_NE(last.find(std::format(", continuing it from node {}",
                                    summaries.size() - 2)),
              std::string::npos)
        << last;
    EXPECT_TRUE(io::TrajectoryStore::ManifestOf(store).complete);
    // one process, the same transforms, the state read back exactly: bitwise
    ExpectSameFiles(store, dir_ / ("whole_" + name));
    for (const auto* field : {"relative_sea_level_change", "displacement"}) {
      EXPECT_EQ(Output(name + "_parts").variables.at(field).values,
                Output(name + "_whole").variables.at(field).values)
          << name << " " << field;
    }
    const auto swept = app::RunKernel(parted);
    EXPECT_NE(swept.find("; the forward run read back from the trajectory "
                         "store "),
              std::string::npos)
        << swept;
    static_cast<void>(app::RunKernel(whole));
    ExpectSameKernels(dir_ / "out" / ("k_" + name + "_parts"),
                      dir_ / "out" / ("k_" + name + "_whole"));
  }

  // a store short of a record before the checkpoint: refused, the ways on
  // named, nothing removed
  const auto shortened = config("short", cases.front().second, "short",
                                std::format(kStopping, "short"));
  for (auto attempt = 0; attempt < 3; ++attempt) {
    static_cast<void>(app::RunSolve(shortened));
  }
  fs::remove(dir_ / "short" / "node_000001.surface");
  ExpectRefused(app::RunSolve, shortened, "trajectory.directory",
                "holds this run's incomplete trajectory store without node "
                "1's surface record whole");
  ExpectRefused(app::RunSolve, shortened, "trajectory.directory",
                "Set \"restart\": false to start the run over and record the "
                "store whole (it is replaced), or drop `trajectory.directory` "
                "to resume without recording");
  EXPECT_TRUE(fs::exists(dir_ / "short" / "node_000002.surface"));
  // a store a resumption would begin at its checkpoint: refused
  const auto begun = config("begun", cases.front().second, "begun",
                            std::format(kStopping, "short"));
  ExpectRefused(app::RunSolve, begun, "trajectory.directory",
                "holds no trajectory store, and the run resumes from the "
                "checkpoint of node 2");
}

TEST_F(AppsTest, TheRunHashesTakeACapsNumbersWhole) {
  // A cap of ice change is described to six digits, and its numbers enter
  // the run's hashes whole beside the description: two runs whose caps
  // differ in the eighth digit of any of them (its place, its width, its
  // thickness or fraction, a date or a factor of its history), the same
  // run described, are two runs to the checkpoints and to the trajectory
  // store alike; solve refuses the one's checkpoints and its store to the
  // other.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto config = [&](const std::string& name, const std::string& cap,
                          const std::string& extra = "") {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "quasi_static",
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}},
                                  {}]}},
      "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "none"{},
      "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, cap, extra, name));
  };
  const auto thickness =
      [](const std::string& latitude, const std::string& longitude,
         const std::string& width, const std::string& amount,
         const std::string& date, const std::string& factor) {
        return std::format(
            R"({{"cap_thickness": {{"latitude": {}, "longitude": {},
            "width_deg": {}, "thickness_m": {}}},
            "history_kyr_bp": [[2, {}], [{}, 0]]}})",
            latitude, longitude, width, amount, factor, date);
      };
  const auto fraction = [](const std::string& amount) {
    return std::format(
        R"({{"cap_fraction": {{"latitude": 90, "longitude": 0,
            "width_deg": 15, "fraction": {}}}}})",
        amount);
  };
  // a run's two hashes and its cap's description
  struct Hashes {
    std::uint64_t checkpoint;
    std::string trajectory, described;
  };
  const auto hashesOf = [&](const fs::path& path) {
    const auto input = app::ReadSolveInput(path);
    auto hash = app::Fnv1a{};
    app::AddRunMembers(hash, input.run, false);
    return Hashes{hash.Value(), app::TrajectoryHashOf(input.run, false),
                  app::Describe(input.run.forcing.ice_change.at(1))};
  };
  const auto base =
      hashesOf(config("base", thickness("45", "30", "6", "10", "1.5", "1")));
  for (const auto& [name, cap] :
       {std::pair{"latitude",
                  thickness("45.000001", "30", "6", "10", "1.5", "1")},
        std::pair{"longitude",
                  thickness("45", "30.000001", "6", "10", "1.5", "1")},
        std::pair{"width",
                  thickness("45", "30", "6.0000001", "10", "1.5", "1")},
        std::pair{"thickness",
                  thickness("45", "30", "6", "10.000001", "1.5", "1")},
        std::pair{"date", thickness("45", "30", "6", "10", "1.5000001", "1")},
        std::pair{"factor",
                  thickness("45", "30", "6", "10", "1.5", "1.0000001")}}) {
    const auto other = hashesOf(config(name, cap));
    EXPECT_EQ(other.described, base.described) << name;
    EXPECT_NE(other.checkpoint, base.checkpoint) << name;
    EXPECT_NE(other.trajectory, base.trajectory) << name;
  }
  {
    const auto plain = hashesOf(config("fraction", fraction("-0.2")));
    const auto other =
        hashesOf(config("fraction_other", fraction("-0.20000001")));
    EXPECT_EQ(other.described, plain.described);
    EXPECT_NE(other.checkpoint, plain.checkpoint);
    EXPECT_NE(other.trajectory, plain.trajectory);
  }
  // and solve: the checkpoints and the store of the one are refused to the
  // other
  constexpr auto kKept = R"(, "checkpoint": {"directory": "ckpt", "every": 2},
      "trajectory": {"directory": "store"})";
  static_cast<void>(app::RunSolve(
      config("kept", thickness("45", "30", "6", "10", "1.5", "1"), kKept)));
  const auto other = config(
      "kept_other", thickness("45", "30", "6", "10.000001", "1.5", "1"), kKept);
  try {
    static_cast<void>(app::RunSolve(other));
    ADD_FAILURE() << "another run resumed the checkpoints";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("different run"),
              std::string::npos)
        << error.what();
  }
  ExpectRefused(
      app::RunSolve,
      config("store_other", thickness("45", "30", "6", "10.000001", "1.5", "1"),
             R"(, "trajectory": {"directory": "store"})"),
      "trajectory.directory", "holds the trajectory store of another run");
}

TEST_F(AppsTest, SolveWithoutGravity) {
  // `"gravity": "none"`: a seismic run through time on the ball without
  // an ocean is the library's evolution on the operator without gravity,
  // field for field, its files holding no potential and no sea level
  // change and its exchange tables the displacement alone; the static
  // problem is the run's first node; on a PREM-like deck with a fluid core
  // the answer is not the run with gravity's; and what needs gravity is
  // refused, naming its member.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  const auto tensor = std::string{R"({"point": {"latitude": 30, "longitude": 40,
      "depth_km": 300, "moment_tensor_nm": {"rr": 9e20, "tt": -2e20,
      "pp": -7e20, "rt": 3e20, "rp": -7e20, "tp": 6e20}}})"};
  const auto config = [&](const std::string& name, const std::string& extra,
                          const std::string& forcing = "",
                          const std::string& problem = "quasi_static") {
    const auto times =
        problem == "static"
            ? std::string{}
            : std::string{R"("times": {"unit": "kyr", "at": [0, 0.4, 1.0]},)"};
    return Write(name + ".json",
                 std::format(R"({{"problem": "{}", {}, {} "forcing":
                     {{"seismic": [{}]{}}}, "gravity": "none"{},
                     "output": {{"surface": "out/{}.nc",
                                 "exchange": "out/{}{}.txt"}}}})",
                             problem, kBall, times, tensor, forcing, extra,
                             name, name, problem == "static" ? "" : "_{node}"));
  };
  const auto summary = app::RunSolve(config("none", ""));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 3 nodes 0 to 1 kyr on the run's clock, no ocean, "
      "no gravity, rotation none; seismic (1 source); wrote"))
      << summary;
  const auto file = Output("none");
  for (const auto* name :
       {"displacement", "tangential_potential", "toroidal_potential"}) {
    EXPECT_TRUE(file.variables.contains(name)) << name;
  }
  EXPECT_FALSE(file.variables.contains("potential"));
  EXPECT_FALSE(file.variables.contains("relative_sea_level_change"));
  EXPECT_EQ(std::get<std::string>(file.meta.at("gravity")), "none");
  {
    const auto table = ReadExchange(dir_ / "out" / "none_2.txt");
    EXPECT_EQ(table.scalars.at("columns"), "l m U_re U_im");
    const auto header = HeaderOf(dir_ / "out" / "none_2.txt");
    EXPECT_NE(header.front().find("no ocean, no gravity, rotation none"),
              std::string::npos)
        << header.front();
  }

  // the library's run without gravity under the same source
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax, {.gravitating = false});
  const SurfaceGrid grid(kLmax);
  const GRDSolver solver(grid, ComputeLoveNumbers(*material, kLmax),
                         SeaLevelParameters{});
  const auto& units = solver.SolverScales();
  const auto energy =
      units.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto surface = material->Radius() * model.ModelScales().Length();
  const auto site = GeographicPoint{30.0, 40.0};
  const auto source =
      PointSource{.at = {.radius = (surface - 300e3) / units.Length(),
                         .colatitude = site.CoLatitude(),
                         .longitude = site.Longitude()},
                  .moment = {.rr = 9e20 / energy,
                             .tt = -2e20 / energy,
                             .pp = -7e20 / energy,
                             .rt = 3e20 / energy,
                             .rp = -7e20 / energy,
                             .tp = 6e20 / energy}};
  // the run's own times, as it forms them
  const auto times =
      app::ReadSolveInput(dir_ / "none.json").run.times->Times(units.Time());
  Evolution evolution(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
  const auto snapshots =
      evolution.Run(times, Evolution::Forcing{.sources = [&](double) {
                      return std::vector<PointSource>{source};
                    }});
  const auto fieldSize = grid.FieldSize();
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    for (const auto& [name, want] :
         {std::pair{"displacement", &snapshots[n].displacement},
          std::pair{"tangential_potential", &snapshots[n].tangentialPotential},
          std::pair{"toroidal_potential", &snapshots[n].toroidalPotential}}) {
      auto scaled = *want;
      if (std::string_view{name} == "toroidal_potential") {
        scaled = WithoutNetRotation(grid, scaled);
      }
      for (auto& v : scaled) v *= units.Length();
      ASSERT_EQ(scaled.size(), fieldSize);
      EXPECT_LT(Departure(Slice(file, name, n), scaled), 1e-12)
          << name << " node " << n;
    }
  }

  // the static problem without gravity: the run's first node, to the
  // branchless and the unrelaxed systems' difference
  static_cast<void>(app::RunSolve(config("elastic", "", "", "static")));
  const auto elastic = Output("elastic");
  EXPECT_FALSE(elastic.variables.contains("potential"));
  EXPECT_FALSE(elastic.variables.contains("relative_sea_level_change"));
  for (const auto* name :
       {"displacement", "tangential_potential", "toroidal_potential"}) {
    EXPECT_LT(
        Departure(elastic.variables.at(name).values, Slice(file, name, 0)),
        1e-8)
        << name;
  }

  // on a PREM-like deck with a fluid core, the static answer with gravity
  // and without differ
  const auto prem =
      std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck";
  const auto premConfig = [&](const std::string& name,
                              const std::string& gravity) {
    return Write(name + ".json",
                 std::format(R"({{"problem": "static",
                     "model": {{"deck": "{}", "format": "elastic"}},
                     "discretisation": {{"lmax": 8}},
                     "forcing": {{"seismic": [{}]}},
                     "gravity": "{}", "rotation": "none",
                     "output": {{"surface": "out/{}.nc"}}}})",
                             prem.string(), tensor, gravity, name));
  };
  static_cast<void>(app::RunSolve(premConfig("prem_full", "full")));
  static_cast<void>(app::RunSolve(premConfig("prem_none", "none")));
  const auto full = Output("prem_full");
  const auto none = Output("prem_none");
  const auto apart = Departure(none.variables.at("displacement").values,
                               full.variables.at("displacement").values);
  RecordProperty("prem_gravity_effect", std::format("{:.3e}", apart));
  EXPECT_GT(apart, 1e-2);

  // the hashes: no member, or "full", hash as a run always has; "none"
  // is another run
  {
    const auto hashesOf = [&](const std::string& gravity) {
      const auto path = Write(
          "hash.json",
          std::format(R"({{"problem": "quasi_static", {},
              "times": {{"unit": "kyr", "at": [0, 0.4, 1.0]}},
              "forcing": {{"seismic": [{}]}}{} "rotation": "none",
              "output": {{"surface": "out/hash.nc"}}}})",
                      kBall, tensor,
                      gravity.empty()
                          ? ","
                          : std::format(R"(, "gravity": "{}",)", gravity)));
      const auto input = app::ReadSolveInput(path);
      auto hash = app::Fnv1a{};
      app::AddRunMembers(hash, input.run, false);
      return std::pair{hash.Value(), app::TrajectoryHashOf(input.run, false)};
    };
    const auto absent = hashesOf("");
    const auto full = hashesOf("full");
    const auto none = hashesOf("none");
    EXPECT_EQ(full, absent);
    EXPECT_NE(none.first, absent.first);
    EXPECT_NE(none.second, absent.second);
  }

  // what needs gravity is refused, naming its member
  WriteToyHistory(dir_ / "history.nc");
  const auto refused = [&](const std::string& extra, const std::string& forcing,
                           const std::string& member, const std::string& what) {
    ExpectRefused(app::RunSolve, config("bad", extra, forcing), member, what);
  };
  refused(R"(, "ocean": {"at": "start", "topography": {"file": "history.nc"},
             "ice": {"file": "history.nc"}})",
          "", "ocean", "a run without gravity");
  refused("", R"(, "tide": [{"degree": 2, "amplitude_m2_s2": 1.0}])",
          "forcing.tide", "a run without gravity");
  refused(R"(, "rotation": "full")", "", "rotation", "a rotational feedback");
  refused(R"(, "rotation": "axial")", "", "rotation", "a rotational feedback");
  refused(R"(, "moments": "hydrostatic")", "", "moments",
          "a run without gravity");
  refused(R"(, "trajectory": {"record_stride": 1})", "", "trajectory",
          "kernels");
  refused(R"(, "kernels": {"measurements": []})", "", "kernels", "kernels");
  ExpectRefused(
      app::RunSolve,
      Write("word.json", std::format(R"({{"problem": "static", {}, "forcing":
                          {{"seismic": [{}]}}, "gravity": "off",
                          "output": {{"surface": "out/word.nc"}}}})",
                                     kBall, tensor)),
      "gravity", "\"off\"");
  ExpectRefused(
      app::RunKernel,
      Write("kernel.json", std::format(R"({{"problem": "static", {}, "forcing":
                          {{"seismic": [{}]}}, "gravity": "none",
                          "kernels": {{"measurements": []}}}})",
                                       kBall, tensor)),
      "gravity", "kernels");
  // a 3-D model
  const auto read = sl3dpp::Deck::Read(dir_ / "ball.deck",
                                       app::FormatOf(app::ModelFormat::Gia));
  ASSERT_TRUE(read);
  const auto ball = EarthModel::FromDeck(*read);
  ASSERT_TRUE(ball);
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", *ball, SurfaceGrid(8),
      [](double, double theta, double) { return 0.1 * std::cos(theta); }, {},
      {.name = "toy moduli", .elementMetres = 500e3}));
  ExpectRefused(app::RunSolve,
                Write("lateral.json", std::format(R"({{"problem": "static",
                "model": {{"deck": "ball.deck", "moduli": {{"file": "moduli.nc"}}}},
                "discretisation": {{"lmax": 8}},
                "forcing": {{"seismic": [{}]}}, "gravity": "none",
                "output": {{"surface": "out/lateral.nc"}}}})",
                                                  tensor)),
                "model.moduli.file", "a 3-D run without gravity");
}

TEST_F(AppsTest, SolveWithoutGravityUnderLoads) {
  // `"gravity": "none"` under surface loads, which act by their weight, a
  // normal traction. A run through time under a coefficient table without
  // degree 1 is the library's evolution without gravity under the same
  // load, field for field, its files holding the loads applied beside the
  // response and no potential or sea level change; a pair of antipodal
  // caps laid as a layer is the pair's weight alone, its height written
  // beside it; the static problem is the traction numbers times the
  // load's weight; the same run with gravity differs, and hashes as
  // another run; a run under loads resumes from its checkpoints to the
  // uninterrupted run; loads with a degree-1 part, through time and
  // static, are answered as the loads less it (the runs under their
  // projections given without one, to rounding), the run saying so in one
  // line on standard error; and the volume file is refused.
  namespace pm = planetmodel;
  static_cast<void>(Deck());
  constexpr auto kLmax = 8;
  // a load's coefficients, kg m^-2: degrees 0, 2 and 3
  static_cast<void>(Write("load_lm.txt",
                          "# a load without degree 1\n"
                          "0 0 2.0e5 0\n"
                          "2 0 -1.5e5 0\n"
                          "2 1 4.0e4 -2.5e4\n"
                          "3 2 -3.0e4 1.0e4\n"));
  const auto table = std::string{R"({"coefficients": {"file": "load_lm.txt"}})"};
  const auto history = std::string{R"(, "history": [[0, 0.5], [0.4, 1.0]])"};
  const auto termOf = [](const std::string& term, const std::string& more) {
    return term.substr(0, term.size() - 1) + more + "}";
  };
  const auto config = [&](const std::string& name, const std::string& loads,
                          const std::string& extra = R"(, "gravity": "none")",
                          const std::string& times =
                              R"("times": {"unit": "kyr", "at": [0, 0.4, 1.0]},)") {
    const auto timed = !times.empty();
    return Write(name + ".json",
                 std::format(R"({{"problem": "{}", {}, {} "forcing":
                     {{"load": [{}]}}, "rotation": "none"{},
                     "output": {{"surface": "out/{}.nc",
                                 "exchange": "out/{}{}.txt"}}}})",
                             timed ? "quasi_static" : "static", kBall, times,
                             loads, extra, name, name, timed ? "_{node}" : ""));
  };
  const auto summary =
      app::RunSolve(config("loaded", termOf(table, history)));
  EXPECT_TRUE(summary.starts_with(
      "1-D, viscoelastic, 3 nodes 0 to 1 kyr on the run's clock, no ocean, "
      "no gravity, rotation none; load (1 term); wrote"))
      << summary;
  const auto file = Output("loaded");
  for (const auto* name : {"displacement", "tangential_potential", "load",
                           "load_outside_water_budget"}) {
    EXPECT_TRUE(file.variables.contains(name)) << name;
  }
  for (const auto* name :
       {"potential", "relative_sea_level_change", "toroidal_potential",
        "load_in_water_budget", "solid_surface_change"}) {
    EXPECT_FALSE(file.variables.contains(name)) << name;
  }
  EXPECT_EQ(std::get<std::string>(file.meta.at("gravity")), "none");
  EXPECT_EQ(std::get<std::string>(file.meta.at("ocean")), "none");
  {
    const auto exchange = ReadExchange(dir_ / "out" / "loaded_2.txt");
    EXPECT_EQ(exchange.scalars.at("columns"), "l m U_re U_im");
    EXPECT_EQ(exchange.scalars.at("load_terms"), "1");
    const auto header = HeaderOf(dir_ / "out" / "loaded_2.txt");
    EXPECT_NE(header.front().find("no ocean, no gravity, rotation none"),
              std::string::npos)
        << header.front();
  }

  // the library's run without gravity under the same load
  const auto model = ModelOf(dir_ / "ball.deck", app::ModelFormat::Gia);
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax});
  ASSERT_TRUE(mesh);
  const auto material = Material::Create(model, *mesh);
  ASSERT_TRUE(material);
  const auto rheology = Rheology::FromModel(*material, model);
  ASSERT_TRUE(rheology);
  const ViscoelasticOperator op(*rheology, kLmax, {.gravitating = false});
  const SurfaceGrid grid(kLmax);
  const auto love = ComputeLoveNumbers(*material, kLmax);
  const GRDSolver solver(grid, love, SeaLevelParameters{});
  const auto& units = solver.SolverScales();
  const auto toLoad = units.Factor(Dimensions{.mass = 1, .length = -2});
  auto sigma = grid.ZeroCoefficients();  // SI
  sigma[SurfaceGrid::Index(0, 0)] = 2.0e5;
  sigma[SurfaceGrid::Index(2, 0)] = -1.5e5;
  sigma[SurfaceGrid::Index(2, 1)] = Complex(4.0e4, -2.5e4);
  sigma[SurfaceGrid::Index(3, 2)] = Complex(-3.0e4, 1.0e4);
  auto given = sigma;
  for (auto& c : given) c /= toLoad;
  const auto field = grid.Synthesise(given);
  // the run's own times, as it forms them
  const auto times =
      app::ReadSolveInput(dir_ / "loaded.json").run.times->Times(units.Time());
  ASSERT_EQ(times.size(), 3u);
  const auto factors = std::vector<double>{0.5, 1.0, 1.0};
  {
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    const auto snapshots = evolution.Run(
        times, Evolution::Forcing{.directLoadOutsideWaterBudget =
                                      [&](double t) {
                                        const auto n = static_cast<std::size_t>(
                                            std::ranges::find(times, t) -
                                            times.begin());
                                        auto out = field;
                                        for (auto& v : out) v *= factors.at(n);
                                        return out;
                                      }});
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      for (const auto& [name, want] :
           {std::pair{"displacement", &snapshots[n].displacement},
            std::pair{"tangential_potential",
                      &snapshots[n].tangentialPotential}}) {
        auto scaled = *want;
        for (auto& v : scaled) v *= units.Length();
        const auto gap = Departure(Slice(file, name, n), scaled);
        worst = std::max(worst, gap);
        EXPECT_LT(gap, 1e-12) << name << " node " << n;
      }
      // the loads applied, as given
      auto load = field;
      for (auto& v : load) v *= factors[n] * toLoad;
      EXPECT_LT(Departure(Slice(file, "load", n), load), 1e-12) << n;
    }
    RecordProperty("library_departure", std::format("{:.3e}", worst));
  }

  // a pair of antipodal caps laid as a layer: its weight alone acts, its
  // height written beside it
  {
    const auto cap = [](double latitude, double longitude,
                        const std::string& more) {
      return std::format(R"({{"cap": {{"latitude": {}, "longitude": {},
          "width_deg": 20, "amplitude_kg_m2": 1.0e6}}{}}})",
                         latitude, longitude, more);
    };
    constexpr auto kLayer =
        R"(, "moves_solid_surface": true, "density_kg_m3": 917)";
    static_cast<void>(app::RunSolve(
        config("layer", cap(30, 40, kLayer) + ", " + cap(-30, 220, kLayer))));
    static_cast<void>(app::RunSolve(
        config("weight", cap(30, 40, "") + ", " + cap(-30, 220, ""))));
    const auto layer = Output("layer");
    const auto weight = Output("weight");
    EXPECT_TRUE(layer.variables.contains("solid_surface_change"));
    EXPECT_FALSE(weight.variables.contains("solid_surface_change"));
    // without gravity a layer's height is only written: the weight's run
    for (const auto* name : {"displacement", "tangential_potential", "load"}) {
      EXPECT_EQ(layer.variables.at(name).values,
                weight.variables.at(name).values)
          << name;
    }
    const auto height = Slice(layer, "solid_surface_change", 2);
    const auto mass = Slice(layer, "load_outside_water_budget", 2);
    auto worst = 0.0;
    for (auto i = std::size_t{0}; i < height.size(); ++i) {
      worst = std::max(worst, std::abs(height[i] * 917.0 - mass[i]));
    }
    EXPECT_LT(worst, 1e-9 * 1.0e6);
  }

  // the static problem: the traction numbers times the load's weight
  {
    static_cast<void>(app::RunSolve(config("static", table, R"(, "gravity": "none")", "")));
    const auto exchange = ReadExchange(dir_ / "out" / "static.txt");
    EXPECT_EQ(exchange.scalars.at("columns"), "l m U_re U_im");
    const auto numbers = ComputeTractionNumbers(*material, kLmax, false);
    const auto stat = Output("static");
    const auto U = grid.Expand(stat.variables.at("displacement").values);
    auto want = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        if (l != 1) want[i] = love.surface_gravity * numbers.h_u[l] * sigma[i];
      }
    }
    const auto gap = Departure(U, want);
    RecordProperty("static_traction_numbers", std::format("{:.3e}", gap));
    EXPECT_LT(gap, 1e-12);
  }

  // with gravity the same run differs, and is another run to the hashes
  static_cast<void>(app::RunSolve(
      config("full", termOf(table, history), R"(, "gravity": "full")")));
  const auto apart = Departure(Slice(file, "displacement", 2),
                               Slice(Output("full"), "displacement", 2));
  RecordProperty("ball_gravity_effect", std::format("{:.3e}", apart));
  EXPECT_GT(apart, 1e-2);
  {
    const auto hashesOf = [&](const std::string& extra) {
      const auto input =
          app::ReadSolveInput(config("hash", termOf(table, history), extra));
      auto hash = app::Fnv1a{};
      app::AddRunMembers(hash, input.run, false);
      return std::pair{hash.Value(), app::TrajectoryHashOf(input.run, false)};
    };
    const auto absent = hashesOf("");
    const auto full = hashesOf(R"(, "gravity": "full")");
    const auto none = hashesOf(R"(, "gravity": "none")");
    EXPECT_EQ(full, absent);
    EXPECT_NE(none.first, full.first);
    EXPECT_NE(none.second, full.second);
  }

  // a run under loads resumes from its checkpoints: each attempt
  // checkpoints and quits at the wall-time limit, and the last is the
  // uninterrupted run
  {
    constexpr auto kFive =
        R"("times": {"unit": "kyr", "at": [0, 0.2, 0.4, 0.7, 1.0]},)";
    static_cast<void>(app::RunSolve(config(
        "whole", termOf(table, history), R"(, "gravity": "none")", kFive)));
    const auto pieces = config("pieces", termOf(table, history),
                               R"(, "gravity": "none",
        "checkpoint": {"directory": "ckpt_none", "every": 1,
                       "max_walltime_min": 1e-9})",
                               kFive);
    auto said = std::string{};
    auto attempts = 0;
    for (; attempts < 8; ++attempts) {
      said = app::RunSolve(pieces);
      if (said.find("wrote") != std::string::npos) break;
      EXPECT_NE(said.find("checkpointed"), std::string::npos) << said;
    }
    EXPECT_GE(attempts, 1);
    EXPECT_NE(said.find("resumed"), std::string::npos) << said;
    // one process, the same transforms, the state read back exactly: bitwise
    for (const auto* name : {"displacement", "tangential_potential"}) {
      EXPECT_EQ(Output("pieces").variables.at(name).values,
                Output("whole").variables.at(name).values)
          << name;
    }
  }

  // loads with a degree-1 part, through time and static: answered as the
  // loads less it, the run saying how much it removed in one line on
  // standard error; a load without one says nothing
  {
    // a run's line on standard error, and its surface file
    const auto run = [&](const fs::path& config) {
      testing::internal::CaptureStderr();
      static_cast<void>(app::RunSolve(config));
      return testing::internal::GetCapturedStderr();
    };
    // a coefficient table of a field's expansion on the run's grid, less
    // its degree-1 part
    const auto projected = [&](const std::string& name,
                               const std::vector<double>& field) {
      const auto c = grid.Expand(field);
      auto text = std::string{"# a load less its degree-1 part\n"};
      for (auto l = 0; l <= kLmax; ++l) {
        if (l == 1) continue;
        for (auto m = 0; m <= l; ++m) {
          const auto v = c[SurfaceGrid::Index(l, m)];
          text +=
              std::format("{} {} {:.17g} {:.17g}\n", l, m, v.real(), v.imag());
        }
      }
      static_cast<void>(Write(name, text));
      return std::format(R"({{"coefficients": {{"file": "{}"}}}})", name);
    };
    const auto same = [&](const std::string& a, const std::string& b,
                          std::size_t nodes) {
      const auto x = Output(a);
      const auto y = Output(b);
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        for (const auto* name : {"displacement", "tangential_potential"}) {
          const auto gap =
              nodes == 1 ? Departure(x.variables.at(name).values,
                                     y.variables.at(name).values)
                         : Departure(Slice(x, name, n), Slice(y, name, n));
          EXPECT_LT(gap, 1e-12) << a << " " << name << " node " << n;
        }
      }
    };
    constexpr auto kPolar = R"({"cap": {"latitude": 90, "longitude": 0,
        "width_deg": 20, "amplitude_kg_m2": 1.0e6}})";
    const auto said = run(config("polar", kPolar));
    EXPECT_NE(said.find("sl3d: forcing.load: the loads' degree-1 part ("),
              std::string::npos)
        << said;
    EXPECT_NE(said.find(" of their size at "), std::string::npos) << said;
    EXPECT_NE(said.find("it is removed"), std::string::npos) << said;
    EXPECT_EQ(std::ranges::count(said, '\n'), 1) << said;
    const auto polar = Output("polar");
    EXPECT_EQ(run(config("polar_less",
                         projected("polar_lm.txt", Slice(polar, "load", 0)))),
              "");
    same("polar", "polar_less", 3);
    // the static problem
    EXPECT_NE(run(config("polar_static", kPolar, R"(, "gravity": "none")", ""))
                  .find("the loads' degree-1 part ("),
              std::string::npos);
    static_cast<void>(run(
        config("polar_static_less",
               projected("polar_static_lm.txt",
                         Output("polar_static").variables.at("load").values),
               R"(, "gravity": "none")", "")));
    same("polar_static", "polar_static_less", 1);
    // a coefficient table with a degree-1 row
    static_cast<void>(
        Write("tilted_lm.txt", "0 0 2.0e5 0\n1 1 1.0e4 0\n2 1 4.0e4 -2.5e4\n"));
    static_cast<void>(Write("level_lm.txt", "0 0 2.0e5 0\n2 1 4.0e4 -2.5e4\n"));
    EXPECT_NE(
        run(config("tilted", R"({"coefficients": {"file": "tilted_lm.txt"}})"))
            .find("the loads' degree-1 part ("),
        std::string::npos);
    static_cast<void>(
        run(config("level", R"({"coefficients": {"file": "level_lm.txt"}})")));
    same("tilted", "level", 3);
  }
  ExpectRefused(
      app::RunSolve,
      Write("volume.json",
            std::format(R"({{"problem": "static", {}, "forcing":
                {{"load": [{}]}}, "gravity": "none",
                "output": {{"surface": "out/volume.nc",
                            "volume": "out/volume_body.nc"}}}})",
                        kBall, table)),
      "output.volume", "without gravity");
  // a degree-1 row of zero is no degree-1 part
  static_cast<void>(Write("zero_lm.txt", "0 0 2.0e5 0\n1 0 0 0\n2 0 1e5 0\n"));
  EXPECT_NO_THROW(static_cast<void>(app::RunSolve(config(
      "zero", R"({"coefficients": {"file": "zero_lm.txt"}})"))));
}

TEST_F(AppsTest, TheRunHashesTakeEveryForwardMember) {
  // Every member a config takes, changed alone from a base config to
  // another valid value, against the run's two hashes: the checkpoints'
  // (AddRunMembers) and the trajectory store's (TrajectoryHashOf). A member
  // that decides the forward run changes both, or the store's alone where
  // a run cannot be checkpointed (the seismic sources, the static problem);
  // a member that steers the run, sets the reverse sweep or names what is
  // written changes neither (`kernels`, `output`, `checkpoint`,
  // `trajectory`, `restart`, `solver.cache_mb`, a file input's and a
  // fault file's `outside_span`, `solver.transform_degrees` on a radial
  // run, and the paths). Two spellings of one run hash alike where its
  // numbers are formed alike (a format's two names, a default given, a
  // run's own clock listed at its evenly spaced times, a static field's
  // date before present given on the file's own axis in kyr, or in yr
  // where the two products are one double) and apart where they are
  // formed otherwise (evenly spaced dates listed, that date in yr where
  // the products differ in the last bit, a point source's moment tensor
  // given for its strike, dip and rake). The hashes read their files'
  // bytes alone: the files here are stand-ins, a line of text each.
  static_cast<void>(Deck());
  constexpr auto kFiles = std::array{
      "history.nc", "thick.nc", "change.nc", "ice.nc",    "sea.nc",
      "land.nc",    "load.nc",  "table.txt", "fault.txt", "moduli.nc",
      "viscosity.nc"};
  fs::create_directories(dir_ / "moved");
  for (const auto* name : kFiles) {
    const auto bytes = std::string{"stand-in for "} + name + "\n";
    static_cast<void>(Write(name, bytes));
    static_cast<void>(Write(std::string{"moved/"} + name, bytes));
  }
  fs::copy_file(dir_ / "ball.deck", dir_ / "moved" / "ball.deck");

  // the base configs' terms, each a unique text the rows edit
  const auto thickness = std::string{
      R"({"thickness": {"file": "thick.nc", "variable": "ice_thickness"}})"};
  const auto change = std::string{
      R"({"change": {"file": "change.nc",)"
      R"( "variable": "ice_thickness_change"}})"};
  const auto iceCap = std::string{
      R"({"cap_thickness": {"latitude": 45, "longitude": 30, "width_deg": 6,
          "thickness_m": 10}, "history_kyr_bp": [[1.9, 1], [1.5, 0]]})"};
  const auto iceFraction = std::string{
      R"({"cap_fraction": {"latitude": 80, "longitude": 3, "width_deg": 15,
          "fraction": -0.2}})"};
  const auto loadCap = std::string{
      R"({"cap": {"latitude": 11, "longitude": 21, "width_deg": 5,
          "amplitude_kg_m2": 100}, "history_kyr_bp": [[1.8, 0], [1.2, 1]],
          "water_budget": true, "moves_solid_surface": false})"};
  const auto loadDisk = std::string{
      R"({"disk": {"latitude": -12, "longitude": 42, "radius_deg": 7,
          "amplitude_kg_m2": 200}, "water_budget": false,
          "moves_solid_surface": true, "density_kg_m3": 2500})"};
  const auto loadField = std::string{
      R"({"field": {"file": "load.nc", "variable": "load"},
          "water_budget": false, "moves_solid_surface": false})"};
  const auto loadTable = std::string{
      R"({"coefficients": {"file": "table.txt"}, "water_budget": true,
          "moves_solid_surface": false})"};
  const auto tide = std::string{
      R"({"degree": 2, "order": 1, "amplitude_m2_s2": 1.5,
          "imaginary_m2_s2": 0.25, "history_kyr_bp": [[1.7, 0], [0.7, 1]]})"};
  const auto point = std::string{
      R"({"point": {"latitude": 5, "longitude": 6, "depth_km": 30,
          "moment_tensor_nm": {"rr": 1e19, "tt": 2e19, "pp": 3e19,
                               "rt": 4e19, "rp": 5e19, "tp": 6e19}},
          "onset_kyr_bp": 1.6, "history_kyr_bp": [[1.6, 1], [0.6, 2]]})"};
  const auto fault = std::string{
      R"({"fault": {"latitude": -20, "longitude": 285, "depth_km": 25,
          "strike_deg": 10, "dip_deg": 20, "rake_deg": 90, "length_km": 100,
          "width_km": 50, "moment_nm": 1e21, "patches_along_strike": 2,
          "patches_down_dip": 3}})"};
  const auto faultFile = std::string{
      R"({"fault_file": {"file": "fault.txt", "cells_along_strike": 4,
          "cells_down_dip": 5, "points": 2, "place_by": "depth",
          "outside_span": "hold"}})"};
  const auto burgers = std::string{
      R"({"depth_km": [0, 3185.5], "kelvin_viscosity": 1e19,
          "kelvin_modulus_fraction": 0.5})"};
  const auto oceanDated = std::string{
      R"("ocean": {"at": "start",
         "topography": {"file": "history.nc", "variable": "topography"},
         "ice": {"file": "history.nc", "variable": "ice_thickness"},
         "land": {"file": "land.nc", "variable": "land"},
         "shorelines": "migrating"})"};
  const auto solverDated = std::string{
      R"("solver": {"rtol": 1e-9, "step_rtol": 1e-8, "step_max_iterations": 40,
         "transform_degrees": [8, 6], "cache_mb": 4,
         "scheme": "exponential_trapezoid", "step": "conjugate_gradients"})"};
  const auto measurement = std::string{
      R"({"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
          "longitude": 10, "width_deg": 25, "date_kyr_bp": 0})"};
  const auto calibration = std::string{
      R"(, "calibration": {"method": "gradient", "iterations": 7,)"
      R"( "rtol": 1e-5})"};
  const auto forwardsLoad = std::string{
      R"({"field": {"file": "load.nc", "time_origin": 10,
          "outside_span": "hold"}, "history": [[0, 1], [100, 2]],
          "water_budget": true, "moves_solid_surface": true,
          "density_kg_m3": 1000})"};
  const auto forwardsPoint = std::string{
      R"({"point": {"latitude": 10, "longitude": 20, "depth_km": 50,
          "strike_deg": 30, "dip_deg": 40, "rake_deg": 80, "moment_nm": 1e21},
          "onset": 5, "history": [[5, 1], [50, 0.5]]})"};
  const auto forwardsFaultFile = std::string{
      R"({"fault_file": {"file": "fault.txt", "cells_along_strike": 3,
          "cells_down_dip": 1, "time_origin": 2, "outside_span": "hold"}})"};
  const auto staticLoadDated = std::string{
      R"({"field": {"file": "load.nc", "date_kyr_bp": 1.25},
          "water_budget": false, "moves_solid_surface": false})"};
  const auto staticLoadOwnAxis = std::string{
      R"({"field": {"file": "load.nc", "date": 3, "unit": "day"},
          "water_budget": true, "moves_solid_surface": false})"};
  // the static family's point as the moment tensor its strike, dip, rake
  // and moment form, each component to the bit
  const auto staticTensor = [] {
    const auto m = DoubleCouple(11.0, 45.0, 91.0);
    const auto scaled = [](double v) { return std::format("{:.17g}", v * 1e20); };
    return std::format(
        R"("moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
           "rp": {}, "tp": {}}}}})",
        scaled(m.rr), scaled(m.tt), scaled(m.pp), scaled(m.rt), scaled(m.rp),
        scaled(m.tp));
  }();

  // the families of base configs: the problem and the times' form, each
  // with the members it takes; `lateral` as the run's operator is
  // (a run through the 3-D operator, whose transforms its hashes take)
  struct Family {
    std::string text;
    bool lateral = false;
  };
  const auto families = std::map<std::string, Family>{
      // a dated run on lateral moduli and Burgers layers, every forcing
      {"dated",
       {R"({"problem": "quasi_static",
       "model": {"deck": "ball.deck", "format": "maxwell",
         "moduli": {"file": "moduli.nc", "reference": "lateral_mean"},
         "burgers": [)" + burgers + R"(]},
       "discretisation": {"lmax": 8},
       "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
       "forcing": {"ice_change": [)" + thickness + ", " + change + ", " +
            iceCap + ", " + iceFraction + R"(],
         "load": [)" + loadCap + ", " + loadDisk + ", " + loadField + ", " +
            loadTable + R"(],
         "tide": [)" + tide + R"(],
         "seismic": [)" + point + ", " + fault + ", " + faultFile + R"(]},
       )" + oceanDated + R"(,
       "rotation": "full", "moments": "hydrostatic",
       )" + solverDated + R"(,
       "output": {"surface": "out/dated.nc",
                  "exchange": "out/dated_{node}.txt"},
       "trajectory": {"directory": "store_dated"}})",
        true}},
      // a calibrated radial run on listed dates, checkpointed, recorded and
      // swept for lateral kernels
      {"calibrated",
       {R"({"problem": "quasi_static",
       "model": {"deck": "ball.deck",
         "viscosity": {"file": "viscosity.nc", "reference": "geometric_mean"}},
       "discretisation": {"lmax": 8},
       "times": {"at_kyr_bp": [2.0, 1.5, 0.5, 0.0]},
       "forcing": {"ice_change": [
         {"thickness": {"file": "ice.nc", "variable": "thickness"}}]},
       "ocean": {"at": "end", "sea_level": {"file": "sea.nc", "variable": "sl"},
         "ice": {"file": "ice.nc", "variable": "thickness"})" +
            calibration + R"(},
       "rotation": "axial",
       "solver": {"rtol": 1e-10, "scheme": "backward_euler",
                  "transform_degrees": [6], "cache_mb": 3},
       "checkpoint": {"directory": "ckpt", "every": 2, "max_walltime_min": 30},
       "restart": true,
       "output": {"surface": "out/calibrated.nc"},
       "trajectory": {"directory": "store_calibrated", "record_stride": 2},
       "kernels": {"method": "discrete", "lateral": true,
         "parameters": ["ice", "viscosity", "moduli"],
         "measurements": [)" + measurement + R"(],
         "output": "out_kernels"}})",
        false}},
      // a run on its own clock, on a viscosity file and lateral moduli
      {"forwards",
       {R"({"problem": "quasi_static",
       "model": {"deck": "ball.deck", "format": "maxwell",
         "viscosity": {"file": "viscosity.nc", "reference": "file_radial"},
         "moduli": {"file": "moduli.nc", "reference": "file_radial"}},
       "discretisation": {"lmax": 8},
       "times": {"unit": "yr", "start": 0, "end": 100, "nodes": 5},
       "forcing": {"ice_change": [{"change": {"file": "change.nc"}},
           {"cap_fraction": {"latitude": 81, "longitude": 4, "width_deg": 16,
             "fraction": -0.3}, "history": [[0, 0], [50, 1]]}],
         "load": [)" + forwardsLoad + R"(],
         "tide": [{"degree": 3, "order": 2, "amplitude_m2_s2": 2.0,
           "imaginary_m2_s2": -1.0, "history": [[0, 1], [100, 0]]}],
         "seismic": [)" + forwardsPoint + ", " + forwardsFaultFile + R"(]},
       "ocean": {"at": "start", "date_kyr_bp": 0.5,
         "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}},
       "rotation": "none", "moments": "input", "gravity": "full",
       "solver": {"step_rtol": 1e-9, "step_max_iterations": 60,
                  "transform_degrees": [8], "step": "fixed_point"},
       "output": {"surface": "out/forwards.nc"}})",
        true}},
      // the static problem on lateral moduli
      {"static",
       {R"({"problem": "static",
       "model": {"deck": "ball.deck", "format": "maxwell",
                 "moduli": {"file": "moduli.nc"}},
       "discretisation": {"lmax": 8},
       "forcing": {"ice_change": [
           {"thickness": {"file": "thick.nc", "date_kyr_bp": 1.0}},
           {"change": {"file": "change.nc", "date_kyr_bp": 0.5}},
           {"cap_thickness": {"latitude": 46, "longitude": 31, "width_deg": 7,
             "thickness_m": -10}}],
         "load": [)" + staticLoadDated + ", " + staticLoadOwnAxis + R"(],
         "tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0}],
         "seismic": [{"point": {"latitude": 7, "longitude": 8, "depth_km": 35,
           "strike_deg": 11, "dip_deg": 45, "rake_deg": 91,
           "moment_nm": 1e20}}]},
       "ocean": {"date_kyr_bp": 2.0, "topography": {"file": "history.nc"},
         "ice": {"file": "history.nc"}, "land": {"file": "land.nc"},
         "shorelines": "migrating"},
       "rotation": "full",
       "solver": {"rtol": 1e-9, "step_rtol": 1e-7, "step_max_iterations": 30,
                  "step": "conjugate_gradients", "transform_degrees": [8, 4],
                  "cache_mb": 2},
       "output": {"surface": "out/static.nc",
                  "exchange": "out/static.txt"}})",
        true}},
      // a run without gravity, and so without an ocean
      {"no gravity",
       {R"({"problem": "quasi_static", "model": {"deck": "ball.deck"},
       "discretisation": {"lmax": 8},
       "times": {"unit": "day", "at": [0, 1, 3, 10]},
       "forcing": {
         "load": [{"cap": {"latitude": 0, "longitude": 0, "width_deg": 10,
           "amplitude_kg_m2": 1000}, "moves_solid_surface": true,
           "density_kg_m3": 3000}],
         "seismic": [{"fault": {"latitude": 1, "longitude": 2, "depth_km": 20,
           "strike_deg": 0, "dip_deg": 30, "rake_deg": 90, "length_km": 40,
           "width_km": 20, "moment_nm": 1e20}}]},
       "gravity": "none", "rotation": "none",
       "output": {"surface": "out/no_gravity.nc"}})",
        false}},
      // a plain checkpointed run under a load, with an ocean
      {"plain",
       {R"({"problem": "quasi_static", "model": {"deck": "ball.deck"},
       "discretisation": {"lmax": 8},
       "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 3},
       "forcing": {"load": [{"cap": {"latitude": 12, "longitude": 22,
         "width_deg": 8, "amplitude_kg_m2": 50}, "water_budget": true,
         "moves_solid_surface": false}]},
       "ocean": {"at": "start", "topography": {"file": "history.nc"},
                 "ice": {"file": "history.nc"}},
       "rotation": "full",
       "checkpoint": {"directory": "ckpt_plain", "every": 1},
       "output": {"surface": "out/plain.nc"}})",
        false}}};

  // what a row expects of the hashes: both change, the store's changes
  // (a run that cannot be checkpointed), or neither does
  enum class Expect { Both, Store, Neither };
  using Edits = std::vector<std::pair<std::string, std::string>>;
  struct Row {
    std::string family, member;
    Edits edits;
    Expect expect = Expect::Both;
    std::string file{};  ///< a file of the run, given other bytes instead
  };
  // an edit within a term: the term, and the term with `from` (once in
  // it) replaced by `to`
  const auto in = [](const std::string& term, std::string_view from,
                     std::string_view to) {
    auto edited = term;
    const auto at = edited.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    EXPECT_EQ(edited.find(from, at + 1), std::string::npos) << from;
    if (at != std::string::npos) edited.replace(at, from.size(), to);
    return std::pair{term, edited};
  };
  const auto drop = [](const std::string& term) {
    return std::pair{", " + term, std::string{}};
  };
  const auto both = Expect::Both;
  const auto store = Expect::Store;
  const auto neither = Expect::Neither;
  const auto rows = std::vector<Row>{
      // the model
      {"dated", "model.deck's bytes", {}, both, "ball.deck"},
      {"dated", "model.deck moved", {{R"("deck": "ball.deck")",
                                      R"("deck": "moved/ball.deck")"}},
       neither},
      {"dated", "model.format \"gia\", \"maxwell\"'s other name",
       {{R"("format": "maxwell")", R"("format": "gia")"}}, neither},
      {"dated", "model.format given as its default",
       {{R"(, "format": "maxwell")", ""}}, neither},
      {"static", "model.format \"mineos\"",
       {{R"("format": "maxwell")", R"("format": "mineos")"}}, store},
      {"static", "model.format \"elastic\"",
       {{R"("format": "maxwell")", R"("format": "elastic")"}}, store},
      {"static", "model.format \"prony\"",
       {{R"("format": "maxwell")", R"("format": "prony")"}}, store},
      {"static", "model.format \"burgers\"",
       {{R"("format": "maxwell")", R"("format": "burgers")"}}, store},
      {"dated", "model.moduli.file's bytes", {}, both, "moduli.nc"},
      {"dated", "model.moduli.file moved",
       {{R"("file": "moduli.nc")", R"("file": "moved/moduli.nc")"}}, neither},
      {"dated", "model.moduli.reference",
       {{R"("reference": "lateral_mean")", R"("reference": "file_radial")"}},
       both},
      {"dated", "model.moduli.reference given as its default",
       {{R"(, "reference": "lateral_mean")", ""}}, neither},
      {"dated", "model.moduli dropped",
       {{R"("moduli": {"file": "moduli.nc", "reference": "lateral_mean"},)",
         ""}},
       both},
      {"forwards", "model.moduli.reference",
       {{R"("moduli.nc", "reference": "file_radial")",
         R"("moduli.nc", "reference": "lateral_mean")"}},
       both},
      {"calibrated", "model.viscosity.file's bytes", {}, both, "viscosity.nc"},
      {"calibrated", "model.viscosity.file moved",
       {{R"("file": "viscosity.nc")", R"("file": "moved/viscosity.nc")"}},
       neither},
      {"calibrated", "model.viscosity.reference",
       {{R"("reference": "geometric_mean")", R"("reference": "file_radial")"}},
       both},
      {"calibrated", "model.viscosity.reference given as its default",
       {{R"(, "reference": "geometric_mean")", ""}}, neither},
      {"calibrated", "model.viscosity dropped",
       {{R"(,
         "viscosity": {"file": "viscosity.nc", "reference": "geometric_mean"})",
         ""}},
       both},
      {"forwards", "model.viscosity.reference",
       {{R"("viscosity.nc", "reference": "file_radial")",
         R"("viscosity.nc", "reference": "geometric_mean")"}},
       both},
      {"dated", "model.burgers top", {in(burgers, "[0,", "[100,")}, both},
      {"dated", "model.burgers bottom", {in(burgers, "3185.5", "3185.6")},
       both},
      {"dated", "model.burgers kelvin_viscosity",
       {in(burgers, "1e19", "1.0000001e19")}, both},
      {"dated", "model.burgers kelvin_modulus_fraction",
       {in(burgers, "0.5}", "0.50000001}")}, both},
      {"dated", "model.burgers a second region",
       {{burgers, burgers + R"(, {"depth_km": [3185.5, 6000],
          "kelvin_viscosity": 1e18, "kelvin_modulus_fraction": 0.1})"}},
       both},
      {"dated", "model.burgers dropped",
       {{R"(,
         "burgers": [)" + burgers + "]",
         ""}},
       both},
      // the degree
      {"dated", "discretisation.lmax", {{R"("lmax": 8)", R"("lmax": 9)"}},
       both},
      // the times
      {"dated", "times.start_kyr_bp",
       {{R"("start_kyr_bp": 2.0)", R"("start_kyr_bp": 2.5)"}}, both},
      {"dated", "times.start_kyr_bp in the ninth digit",
       {{R"("start_kyr_bp": 2.0)", R"("start_kyr_bp": 2.00000001)"}}, both},
      {"dated", "times.end_kyr_bp",
       {{R"("end_kyr_bp": 0.0)", R"("end_kyr_bp": 0.5)"}}, both},
      {"dated", "times.end_kyr_bp given as its default",
       {{R"("end_kyr_bp": 0.0, )", ""}}, neither},
      {"dated", "times.nodes", {{R"("nodes": 5)", R"("nodes": 6)"}}, both},
      {"dated", "times listed at the evenly spaced dates",
       {{R"("start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5)",
         R"("at_kyr_bp": [2.0, 1.5, 1.0, 0.5, 0.0])"}},
       both},
      {"calibrated", "times.at_kyr_bp, a node",
       {{"[2.0, 1.5, 0.5, 0.0]", "[2.0, 1.5, 0.6, 0.0]"}}, both},
      {"calibrated", "times.at_kyr_bp, a node more",
       {{"[2.0, 1.5, 0.5, 0.0]", "[2.0, 1.5, 1.0, 0.5, 0.0]"}}, both},
      {"forwards", "times.unit", {{R"("unit": "yr")", R"("unit": "day")"}},
       both},
      {"forwards", "times.start", {{R"("start": 0,)", R"("start": 1,)"}},
       both},
      {"forwards", "times.start given as its default",
       {{R"("start": 0, )", ""}}, neither},
      {"forwards", "times.end", {{R"("end": 100,)", R"("end": 101,)"}}, both},
      {"forwards", "times.nodes", {{R"("nodes": 5)", R"("nodes": 6)"}}, both},
      {"forwards", "times listed at the evenly spaced times",
       {{R"("start": 0, "end": 100, "nodes": 5)",
         R"("at": [0, 25, 50, 75, 100])"}},
       neither},
      {"no gravity", "times.at, a node", {{"[0, 1, 3, 10]", "[0, 1, 4, 10]"}},
       both},
      {"no gravity", "times.unit", {{R"("unit": "day")", R"("unit": "yr")"}},
       both},
      // the ice change
      {"dated", "ice_change thickness's file's bytes", {}, both, "thick.nc"},
      {"dated", "ice_change thickness's file moved",
       {in(thickness, "thick.nc", "moved/thick.nc")}, neither},
      {"dated", "ice_change thickness's variable",
       {in(thickness, R"("variable": "ice_thickness")",
           R"("variable": "ice")")},
       both},
      {"dated", "ice_change thickness's variable given as its default",
       {in(thickness, R"(, "variable": "ice_thickness")", "")}, neither},
      {"dated", "ice_change thickness's format given as its default",
       {in(thickness, R"("thick.nc",)",
           R"("thick.nc", "format": "planetmodel_surface",)")},
       neither},
      {"dated", "ice_change thickness's outside_span",
       {in(thickness, R"("ice_thickness"})",
           R"("ice_thickness", "outside_span": "hold"})")},
       neither},
      {"dated", "ice_change thickness taken as a change",
       {in(thickness, R"("thickness":)", R"("change":)")}, both},
      {"dated", "ice_change change's file's bytes", {}, both, "change.nc"},
      {"dated", "ice_change change's variable",
       {in(change, R"("ice_thickness_change")", R"("change")")}, both},
      {"dated", "ice_change cap_thickness latitude",
       {in(iceCap, "45,", "45.000001,")}, both},
      {"dated", "ice_change cap_thickness longitude",
       {in(iceCap, "30,", "30.000001,")}, both},
      {"dated", "ice_change cap_thickness width_deg",
       {in(iceCap, "6,", "6.0000001,")}, both},
      {"dated", "ice_change cap_thickness thickness_m",
       {in(iceCap, "10}", "10.000001}")}, both},
      {"dated", "ice_change cap_thickness history date",
       {in(iceCap, "[1.9,", "[1.9000001,")}, both},
      {"dated", "ice_change cap_thickness history factor",
       {in(iceCap, "[1.5, 0]", "[1.5, 1e-7]")}, both},
      {"dated", "ice_change cap_thickness history dropped",
       {in(iceCap, R"(, "history_kyr_bp": [[1.9, 1], [1.5, 0]])", "")}, both},
      {"dated", "ice_change cap_thickness as cap_fraction",
       {in(iceCap, R"("cap_thickness")", R"("cap_fraction")"),
        {R"("thickness_m": 10})", R"("fraction": 10})"}},
       both},
      {"dated", "ice_change cap_fraction fraction",
       {in(iceFraction, "-0.2", "-0.20000001")}, both},
      {"dated", "ice_change cap_fraction latitude",
       {in(iceFraction, "80,", "80.000001,")}, both},
      {"forwards", "ice_change cap_fraction history time",
       {{"[50, 1]]", "[51, 1]]"}}, both},
      {"forwards", "ice_change cap_fraction history factor",
       {{"[[0, 0], [50, 1]]", "[[0, 1e-7], [50, 1]]"}}, both},
      {"forwards", "ice_change change's outside_span",
       {{R"({"change": {"file": "change.nc"}})",
         R"({"change": {"file": "change.nc", "outside_span": "hold"}})"}},
       neither},
      {"dated", "ice_change terms in another order",
       {drop(iceFraction),
        {R"("ice_change": [)", R"("ice_change": [)" + iceFraction + ", "}},
       both},
      {"dated", "ice_change a term dropped", {drop(iceFraction)}, both},
      // the surface loads
      {"dated", "load cap latitude", {in(loadCap, "11,", "11.000001,")}, both},
      {"dated", "load cap longitude", {in(loadCap, "21,", "21.000001,")},
       both},
      {"dated", "load cap width_deg", {in(loadCap, "5,", "5.0000001,")}, both},
      {"dated", "load cap amplitude_kg_m2",
       {in(loadCap, "100}", "100.00001}")}, both},
      {"dated", "load cap history date", {in(loadCap, "[1.8,", "[1.8000001,")},
       both},
      {"dated", "load cap history factor",
       {in(loadCap, "[1.2, 1]", "[1.2, 1.0000001]")}, both},
      {"dated", "load cap history dropped",
       {in(loadCap, R"( "history_kyr_bp": [[1.8, 0], [1.2, 1]],)", "")}, both},
      {"dated", "load cap water_budget",
       {in(loadCap, R"("water_budget": true)", R"("water_budget": false)")},
       both},
      {"dated", "load cap moves_solid_surface",
       {in(loadCap, R"("moves_solid_surface": false)",
           R"("moves_solid_surface": true, "density_kg_m3": 2000)")},
       both},
      {"dated", "load disk latitude", {in(loadDisk, "-12,", "-12.000001,")},
       both},
      {"dated", "load disk radius_deg", {in(loadDisk, "7,", "7.0000001,")},
       both},
      {"dated", "load disk amplitude_kg_m2",
       {in(loadDisk, "200}", "200.00001}")}, both},
      {"dated", "load disk density_kg_m3",
       {in(loadDisk, "2500}", "2500.0001}")}, both},
      {"dated", "load disk moves_solid_surface",
       {in(loadDisk, R"(true, "density_kg_m3": 2500)", "false")}, both},
      {"dated", "load field's file's bytes", {}, both, "load.nc"},
      {"dated", "load field's file moved",
       {in(loadField, "load.nc", "moved/load.nc")}, neither},
      {"dated", "load field's variable",
       {in(loadField, R"("variable": "load")", R"("variable": "mass")")},
       both},
      {"dated", "load field's variable given as its default",
       {in(loadField, R"(, "variable": "load")", "")}, neither},
      {"dated", "load field's outside_span",
       {in(loadField, R"("load"})", R"("load", "outside_span": "hold"})")},
       neither},
      {"dated", "load field water_budget",
       {in(loadField, R"("water_budget": false)", R"("water_budget": true)")},
       both},
      {"dated", "load coefficients' table's bytes", {}, both, "table.txt"},
      {"dated", "load coefficients' table moved",
       {in(loadTable, "table.txt", "moved/table.txt")}, neither},
      {"dated", "load terms in another order",
       {{loadCap + ", " + loadDisk, loadDisk + ", " + loadCap}}, both},
      {"dated", "load a term given twice",
       {{loadTable, loadTable + ", " + loadTable}}, both},
      {"dated", "load a term dropped", {drop(loadTable)}, both},
      {"forwards", "load field time_origin",
       {in(forwardsLoad, R"("time_origin": 10)", R"("time_origin": 11)")},
       both},
      {"forwards", "load field time_origin dropped",
       {in(forwardsLoad, R"( "time_origin": 10,)", "")}, both},
      {"forwards", "load field's outside_span",
       {in(forwardsLoad, R"(,
          "outside_span": "hold")",
           "")},
       neither},
      {"forwards", "load history time",
       {in(forwardsLoad, "[100, 2]", "[90, 2]")}, both},
      {"forwards", "load density_kg_m3",
       {in(forwardsLoad, "1000}", "1001}")}, both},
      {"no gravity", "load cap amplitude_kg_m2",
       {{"1000}", "1000.0001}"}}, both},
      {"no gravity", "load density_kg_m3", {{"3000}", "3001}"}}, both},
      {"no gravity", "load moves_solid_surface",
       {{R"("moves_solid_surface": true,
           "density_kg_m3": 3000)",
         R"("moves_solid_surface": false)"}},
       both},
      {"no gravity", "load water_budget given as its default",
       {{R"("moves_solid_surface": true,)",
         R"("water_budget": false, "moves_solid_surface": true,)"}},
       neither},
      {"static", "load field date_kyr_bp",
       {in(staticLoadDated, "1.25", "1.3")}, store},
      {"static", "load field date on the file's axis",
       {in(staticLoadOwnAxis, R"("date": 3,)", R"("date": 3.5,)")}, store},
      {"static", "load field unit", {in(staticLoadOwnAxis, "day", "yr")},
       store},
      // a date before present and the same instant on the file's own axis
      // resolve to one time (StaticReadTimeOf): in kyr always (the
      // negation is exact), in yr where the two products are one double
      // (1.25 kyr and 1250 yr are exact; 0.017 kyr and 17 yr are not,
      // below)
      {"static",
       "load field date_kyr_bp given as a date on the file's own axis in kyr",
       {in(staticLoadDated, R"("date_kyr_bp": 1.25)",
           R"("date": -1.25, "unit": "kyr")")},
       neither},
      {"static",
       "load field date_kyr_bp given as a date on the file's own axis in yr",
       {in(staticLoadDated, R"("date_kyr_bp": 1.25)",
           R"("date": -1250, "unit": "yr")")},
       neither},
      // the tide
      {"dated", "tide degree", {in(tide, R"("degree": 2)", R"("degree": 3)")},
       both},
      {"dated", "tide order", {in(tide, R"("order": 1)", R"("order": 2)")},
       both},
      {"dated", "tide amplitude_m2_s2", {in(tide, "1.5,", "1.5000001,")},
       both},
      {"dated", "tide imaginary_m2_s2", {in(tide, "0.25,", "0.2500001,")},
       both},
      {"dated", "tide history date", {in(tide, "[1.7,", "[1.7000001,")},
       both},
      {"dated", "tide history factor", {in(tide, "[0.7, 1]", "[0.7, 1.5]")},
       both},
      {"dated", "tide history dropped",
       {in(tide, R"(, "history_kyr_bp": [[1.7, 0], [0.7, 1]])", "")},
       both},
      {"dated", "tide a coefficient more",
       {{tide, tide + R"(, {"degree": 4, "order": 0, "amplitude_m2_s2": 1})"}},
       both},
      {"dated", "tide dropped",
       {{R"(,
         "tide": [)" + tide + "]",
         ""}},
       both},
      {"forwards", "tide history time", {{"[100, 0]", "[90, 0]"}}, both},
      {"static", "tide amplitude_m2_s2",
       {{R"("amplitude_m2_s2": 1.0)", R"("amplitude_m2_s2": 1.1)"}}, store},
      {"static", "tide order and imaginary_m2_s2 given as their defaults",
       {{R"("amplitude_m2_s2": 1.0)",
         R"("amplitude_m2_s2": 1.0, "imaginary_m2_s2": 0)"},
        {R"(, "order": 0)", ""}},
       neither},
      // the seismic sources: the store's alone, a run under them is not
      // checkpointed
      {"dated", "seismic point latitude",
       {in(point, R"("latitude": 5)", R"("latitude": 5.000001)")}, store},
      {"dated", "seismic point longitude",
       {in(point, R"("longitude": 6)", R"("longitude": 6.000001)")}, store},
      {"dated", "seismic point depth_km",
       {in(point, R"("depth_km": 30)", R"("depth_km": 30.000001)")}, store},
      {"dated", "seismic point moment rr", {in(point, "1e19", "1.0000001e19")},
       store},
      {"dated", "seismic point moment tt", {in(point, "2e19", "2.0000001e19")},
       store},
      {"dated", "seismic point moment pp", {in(point, "3e19", "3.0000001e19")},
       store},
      {"dated", "seismic point moment rt", {in(point, "4e19", "4.0000001e19")},
       store},
      {"dated", "seismic point moment rp", {in(point, "5e19", "5.0000001e19")},
       store},
      {"dated", "seismic point moment tp", {in(point, "6e19", "6.0000001e19")},
       store},
      {"dated", "seismic point onset_kyr_bp",
       {in(point, R"("onset_kyr_bp": 1.6)", R"("onset_kyr_bp": 1.65)")},
       store},
      {"dated", "seismic point onset dropped",
       {in(point, R"("onset_kyr_bp": 1.6, )", "")}, store},
      {"dated", "seismic point history date", {in(point, "[0.6,", "[0.65,")},
       store},
      {"dated", "seismic point history factor",
       {in(point, "[0.6, 2]", "[0.6, 3]")}, store},
      {"dated", "seismic point history dropped",
       {in(point, R"(, "history_kyr_bp": [[1.6, 1], [0.6, 2]])", "")}, store},
      {"forwards", "seismic point strike_deg",
       {in(forwardsPoint, R"("strike_deg": 30)", R"("strike_deg": 31)")},
       store},
      {"forwards", "seismic point dip_deg",
       {in(forwardsPoint, R"("dip_deg": 40)", R"("dip_deg": 41)")}, store},
      {"forwards", "seismic point rake_deg",
       {in(forwardsPoint, R"("rake_deg": 80)", R"("rake_deg": 81)")}, store},
      {"forwards", "seismic point moment_nm",
       {in(forwardsPoint, "1e21", "1.0000001e21")}, store},
      {"forwards", "seismic point onset",
       {in(forwardsPoint, R"("onset": 5)", R"("onset": 6)")}, store},
      // between nodes (the library's grid holds a pair at it) and at one
      {"forwards", "seismic point onset moved onto a node",
       {in(forwardsPoint, R"("onset": 5)", R"("onset": 25)")}, store},
      {"forwards", "seismic point history time",
       {in(forwardsPoint, "[50, 0.5]", "[51, 0.5]")}, store},
      {"static", "seismic point strike_deg",
       {{R"("strike_deg": 11)", R"("strike_deg": 12)"}}, store},
      {"static", "seismic point moment_nm", {{"1e20", "1.0000001e20"}}, store},
      // the sources hash as given: a moment tensor and the double couple
      // that forms it are two spellings apart, even where the tensor is
      // the double couple's to the bit (a store is then not reused, never
      // misread)
      {"static", "seismic point given as the moment tensor it forms",
       {{R"("strike_deg": 11, "dip_deg": 45, "rake_deg": 91,
           "moment_nm": 1e20})",
         staticTensor}},
       store},
      {"dated", "seismic fault latitude",
       {in(fault, R"("latitude": -20)", R"("latitude": -20.000001)")}, store},
      {"dated", "seismic fault longitude",
       {in(fault, R"("longitude": 285)", R"("longitude": 285.000001)")}, store},
      {"dated", "seismic fault depth_km",
       {in(fault, R"("depth_km": 25)", R"("depth_km": 25.000001)")}, store},
      {"dated", "seismic fault strike_deg",
       {in(fault, R"("strike_deg": 10)", R"("strike_deg": 10.000001)")}, store},
      {"dated", "seismic fault dip_deg",
       {in(fault, R"("dip_deg": 20)", R"("dip_deg": 20.000001)")}, store},
      {"dated", "seismic fault rake_deg",
       {in(fault, R"("rake_deg": 90)", R"("rake_deg": 90.000001)")}, store},
      {"dated", "seismic fault length_km",
       {in(fault, R"("length_km": 100)", R"("length_km": 100.00001)")}, store},
      {"dated", "seismic fault width_km",
       {in(fault, R"("width_km": 50)", R"("width_km": 50.000001)")}, store},
      {"dated", "seismic fault moment_nm", {in(fault, "1e21", "1.0000001e21")},
       store},
      {"dated", "seismic fault patches_along_strike",
       {in(fault, R"("patches_along_strike": 2)",
           R"("patches_along_strike": 3)")},
       store},
      {"dated", "seismic fault patches_down_dip",
       {in(fault, R"("patches_down_dip": 3)", R"("patches_down_dip": 4)")},
       store},
      {"no gravity", "seismic fault length_km", {{"40,", "41,"}}, store},
      {"dated", "seismic fault_file's bytes", {}, store, "fault.txt"},
      {"dated", "seismic fault_file moved",
       {in(faultFile, "fault.txt", "moved/fault.txt")}, neither},
      {"dated", "seismic fault_file cells_along_strike",
       {in(faultFile, R"("cells_along_strike": 4)",
           R"("cells_along_strike": 5)")},
       store},
      {"dated", "seismic fault_file cells_down_dip",
       {in(faultFile, R"("cells_down_dip": 5)", R"("cells_down_dip": 6)")},
       store},
      {"dated", "seismic fault_file points",
       {in(faultFile, R"("points": 2)", R"("points": 3)")}, store},
      {"dated", "seismic fault_file place_by",
       {in(faultFile, R"("place_by": "depth")", R"("place_by": "radius")")},
       store},
      {"dated", "seismic fault_file place_by dropped",
       {in(faultFile, R"( "place_by": "depth",)", "")}, store},
      {"forwards", "seismic fault_file time_origin",
       {in(forwardsFaultFile, R"("time_origin": 2)", R"("time_origin": 3)")},
       store},
      {"forwards", "seismic fault_file points given as its default",
       {in(forwardsFaultFile, R"("cells_down_dip": 1,)",
           R"("cells_down_dip": 1, "points": 1,)")},
       neither},
      // a fault file's `outside_span`, as a file input's: where "refuse"
      // runs, no read lies outside the span and "hold" reads alike
      {"forwards", "seismic fault_file outside_span",
       {in(forwardsFaultFile, R"("outside_span": "hold")",
           R"("outside_span": "refuse")")},
       neither},
      {"dated", "seismic fault_file outside_span dropped",
       {in(faultFile, R"(,
          "outside_span": "hold")", "")},
       neither},
      {"dated", "seismic terms in another order",
       {{point + ", " + fault, fault + ", " + point}}, store},
      {"dated", "seismic a term dropped", {drop(faultFile)}, store},
      // the ocean
      {"dated", "ocean's state's bytes", {}, both, "history.nc"},
      {"dated", "ocean's state moved",
       {{R"("topography": {"file": "history.nc")",
         R"("topography": {"file": "moved/history.nc")"},
        {R"("ice": {"file": "history.nc")",
         R"("ice": {"file": "moved/history.nc")"}},
       neither},
      {"dated", "ocean.ice in another file",
       {{R"("ice": {"file": "history.nc")", R"("ice": {"file": "ice.nc")"}},
       both},
      {"dated", "ocean.topography's variable",
       {{R"("variable": "topography")", R"("variable": "relief")"}}, both},
      {"dated", "ocean.topography's variable given as its default",
       {{R"(, "variable": "topography")", ""}}, neither},
      {"dated", "ocean.ice's variable",
       {{R"("history.nc", "variable": "ice_thickness")",
         R"("history.nc", "variable": "thickness")"}},
       both},
      {"dated", "ocean.ice's variable given as its default",
       {{R"("history.nc", "variable": "ice_thickness")", R"("history.nc")"}},
       neither},
      {"dated", "ocean.ice's outside_span",
       {{R"("history.nc", "variable": "ice_thickness")",
         R"("history.nc", "variable": "ice_thickness",
           "outside_span": "hold")"}},
       neither},
      {"dated", "ocean.land's bytes", {}, both, "land.nc"},
      {"dated", "ocean.land's variable",
       {{R"("variable": "land")", R"("variable": "held")"}}, both},
      {"dated", "ocean.land's outside_span",
       {{R"("variable": "land")", R"("variable": "land",
           "outside_span": "hold")"}},
       neither},
      {"dated", "ocean.land dropped",
       {{R"(
         "land": {"file": "land.nc", "variable": "land"},)",
         ""}},
       both},
      {"dated", "ocean.shorelines",
       {{R"("shorelines": "migrating")", R"("shorelines": "frozen")"}}, both},
      {"plain", "ocean.shorelines given as its default",
       {{R"("at": "start",)", R"("at": "start", "shorelines": "frozen",)"}},
       neither},
      {"plain", "ocean dropped",
       {{R"(,
       "ocean": {"at": "start", "topography": {"file": "history.nc"},
                 "ice": {"file": "history.nc"}})",
         ""}},
       both},
      {"calibrated", "ocean.sea_level's bytes", {}, both, "sea.nc"},
      {"calibrated", "ocean.sea_level's variable",
       {{R"("variable": "sl")", R"("variable": "sea")"}}, both},
      {"calibrated", "ocean.sea_level as a topography",
       {{R"("sea_level": {)", R"("topography": {)"}}, both},
      {"calibrated", "ocean.ice's bytes", {}, both, "ice.nc"},
      {"calibrated", "ocean.at \"start\"",
       {{R"("at": "end")", R"("at": "start")"}, {calibration, ""}}, both},
      {"calibrated", "ocean.calibration.method",
       {{R"("method": "gradient")", R"("method": "fixed_point")"}}, both},
      {"calibrated", "ocean.calibration.iterations",
       {{R"("iterations": 7)", R"("iterations": 8)"}}, both},
      {"calibrated", "ocean.calibration.rtol",
       {{R"("rtol": 1e-5)", R"("rtol": 2e-5)"}}, both},
      {"forwards", "ocean.date_kyr_bp",
       {{R"("date_kyr_bp": 0.5)", R"("date_kyr_bp": 0.6)"}}, both},
      {"forwards", "ocean.date_kyr_bp dropped",
       {{R"( "date_kyr_bp": 0.5,)", ""}}, both},
      {"static", "ocean.date_kyr_bp",
       {{R"("date_kyr_bp": 2.0)", R"("date_kyr_bp": 2.1)"}}, store},
      {"static", "ocean.shorelines",
       {{R"("shorelines": "migrating")", R"("shorelines": "frozen")"}}, store},
      {"static", "ocean.land dropped",
       {{R"(, "land": {"file": "land.nc"})", ""}}, store},
      {"static", "ice_change thickness's date_kyr_bp",
       {{R"("thick.nc", "date_kyr_bp": 1.0)",
         R"("thick.nc", "date_kyr_bp": 1.1)"}},
       store},
      {"static", "ice_change change's date_kyr_bp",
       {{R"("change.nc", "date_kyr_bp": 0.5)",
         R"("change.nc", "date_kyr_bp": 0.6)"}},
       store},
      {"static", "ice_change change's date_kyr_bp dropped",
       {{R"("change.nc", "date_kyr_bp": 0.5)", R"("change.nc")"}}, store},
      {"static", "ice_change cap_thickness thickness_m",
       {{"-10}", "-10.000001}"}}, store},
      // the rotation, the moments and gravity
      {"dated", "rotation \"axial\"",
       {{R"("rotation": "full")", R"("rotation": "axial")"}}, both},
      {"dated", "rotation \"none\"",
       {{R"("rotation": "full")", R"("rotation": "none")"}}, both},
      {"calibrated", "rotation \"full\"",
       {{R"("rotation": "axial")", R"("rotation": "full")"}}, both},
      {"forwards", "rotation \"axial\"",
       {{R"("rotation": "none")", R"("rotation": "axial")"}}, both},
      {"static", "rotation \"none\"",
       {{R"("rotation": "full")", R"("rotation": "none")"}}, store},
      {"no gravity", "rotation given as its default",
       {{R"(, "rotation": "none")", ""}}, neither},
      {"dated", "moments \"input\"",
       {{R"("moments": "hydrostatic")", R"("moments": "input")"}}, both},
      {"forwards", "moments given as its default",
       {{R"( "moments": "input",)", ""}}, neither},
      {"forwards", "gravity given as its default",
       {{R"( "gravity": "full",)", ""}}, neither},
      {"no gravity", "gravity \"full\"", {{R"("gravity": "none", )", ""}},
       both},
      // the solves
      {"dated", "solver.rtol", {{R"("rtol": 1e-9)", R"("rtol": 1e-10)"}},
       both},
      {"dated", "solver.rtol given as its default",
       {{R"("rtol": 1e-9, )", ""}}, neither},
      {"dated", "solver.step_rtol",
       {{R"("step_rtol": 1e-8)", R"("step_rtol": 1e-9)"}}, both},
      {"dated", "solver.step_max_iterations",
       {{R"("step_max_iterations": 40)", R"("step_max_iterations": 41)"}},
       both},
      {"dated", "solver.scheme",
       {{R"("scheme": "exponential_trapezoid")",
         R"("scheme": "backward_euler")"}},
       both},
      {"dated", "solver.scheme \"sdirk2\"",
       {{R"("scheme": "exponential_trapezoid")", R"("scheme": "sdirk2")"}},
       both},
      {"dated", "solver.scheme given as its default",
       {{R"(
         "scheme": "exponential_trapezoid",)",
         ""}},
       neither},
      {"dated", "solver.step",
       {{R"("step": "conjugate_gradients")", R"("step": "fixed_point")"}},
       both},
      {"forwards", "solver.step given as its default",
       {{R"(, "step": "fixed_point")", ""}}, neither},
      {"dated", "solver.transform_degrees on a lateral run",
       {{"[8, 6]", "[8, 5]"}}, both},
      {"dated", "solver.transform_degrees dropped on a lateral run",
       {{R"(
         "transform_degrees": [8, 6],)",
         ""}},
       both},
      {"forwards", "solver.transform_degrees on a lateral run",
       {{R"("transform_degrees": [8])", R"("transform_degrees": [7])"}}, both},
      {"calibrated", "solver.transform_degrees on a radial run",
       {{R"("transform_degrees": [6])", R"("transform_degrees": [5])"}},
       neither},
      {"dated", "solver.cache_mb",
       {{R"("cache_mb": 4)", R"("cache_mb": 1)"}}, neither},
      {"dated", "solver.cache_mb dropped", {{R"( "cache_mb": 4,)", ""}},
       neither},
      {"calibrated", "solver.rtol", {{R"("rtol": 1e-10)", R"("rtol": 1e-11)"}},
       both},
      {"calibrated", "solver.scheme",
       {{R"("scheme": "backward_euler")",
         R"("scheme": "exponential_trapezoid")"}},
       both},
      {"static", "solver.rtol", {{R"("rtol": 1e-9)", R"("rtol": 1e-8)"}},
       store},
      {"static", "solver.step_rtol",
       {{R"("step_rtol": 1e-7)", R"("step_rtol": 1e-8)"}}, store},
      {"static", "solver.step_max_iterations",
       {{R"("step_max_iterations": 30)", R"("step_max_iterations": 31)"}},
       store},
      {"static", "solver.step",
       {{R"("step": "conjugate_gradients")", R"("step": "fixed_point")"}},
       store},
      {"static", "solver.transform_degrees on a lateral run",
       {{"[8, 4]", "[8, 3]"}}, store},
      {"static", "solver.cache_mb",
       {{R"("cache_mb": 2)", R"("cache_mb": 8)"}}, neither},
      // what steers the run, what the sweep reads and what is written
      {"dated", "output.surface",
       {{R"("surface": "out/dated.nc")", R"("surface": "elsewhere.nc")"}},
       neither},
      {"dated", "output.exchange dropped",
       {{R"(,
                  "exchange": "out/dated_{node}.txt")",
         ""}},
       neither},
      {"dated", "output.fields and output.smoothing",
       {{R"("surface": "out/dated.nc",)",
         R"("surface": "out/dated.nc",
           "fields": ["horizontal_displacement"],
           "smoothing": {"kind": "gaussian", "width_deg": 3},)"}},
       neither},
      {"dated", "output.exchange_nodes",
       {{R"("exchange": "out/dated_{node}.txt")",
         R"("exchange": "out/dated_{node}.txt", "exchange_nodes": [0, 2])"}},
       neither},
      {"dated", "trajectory.directory",
       {{R"("directory": "store_dated")", R"("directory": "elsewhere")"}},
       neither},
      {"dated", "trajectory dropped",
       {{R"(,
       "trajectory": {"directory": "store_dated"})",
         ""}},
       neither},
      {"calibrated", "trajectory.record_stride",
       {{R"("record_stride": 2)", R"("record_stride": 1)"}}, neither},
      {"calibrated", "checkpoint.directory",
       {{R"("directory": "ckpt")", R"("directory": "elsewhere")"}}, neither},
      {"calibrated", "checkpoint.every",
       {{R"("every": 2)", R"("every": 3)"}}, neither},
      {"calibrated", "checkpoint.max_walltime_min",
       {{R"("max_walltime_min": 30)", R"("max_walltime_min": 60)"}}, neither},
      {"calibrated", "restart", {{R"("restart": true)", R"("restart": false)"}},
       neither},
      {"calibrated", "solver.cache_mb",
       {{R"("cache_mb": 3)", R"("cache_mb": 5)"}}, neither},
      {"calibrated", "kernels.method",
       {{R"("method": "discrete")", R"("method": "continuous")"}}, neither},
      {"calibrated", "kernels.method dropped, its default",
       {{R"("method": "discrete", )", ""}}, neither},
      {"calibrated", "kernels.parameters",
       {{R"(["ice", "viscosity", "moduli"])", R"(["ice", "viscosity"])"}},
       neither},
      {"calibrated", "kernels.measurements",
       {in(measurement, R"("width_deg": 25)", R"("width_deg": 20)")}, neither},
      {"calibrated", "kernels.output",
       {{R"("output": "out_kernels")", R"("output": "elsewhere")"}}, neither},
      {"plain", "checkpoint dropped",
       {{R"(,
       "checkpoint": {"directory": "ckpt_plain", "every": 1})",
         ""}},
       neither}};

  // each family's two hashes, then each row's against its family's
  const auto hashesOf = [&](const std::string& text, bool lateral) {
    const auto input = app::ReadSolveInput(Write("hashed.json", text));
    auto hash = app::Fnv1a{};
    app::AddRunMembers(hash, input.run, lateral);
    return std::pair{hash.Value(), app::TrajectoryHashOf(input.run, lateral)};
  };
  auto bases =
      std::map<std::string, std::pair<std::uint64_t, std::string>>{};
  for (const auto& [name, family] : families) {
    bases.emplace(name, hashesOf(family.text, family.lateral));
  }
  for (const auto& row : rows) {
    const auto label = row.family + ": " + row.member;
    const auto& family = families.at(row.family);
    auto text = family.text;
    auto edited = true;
    for (const auto& [from, to] : row.edits) {
      const auto at = text.find(from);
      if (at == std::string::npos || text.find(from, at + 1) != text.npos) {
        ADD_FAILURE() << label << ": the edit's text is not once in the "
                      << "config: " << from;
        edited = false;
        break;
      }
      text.replace(at, from.size(), to);
    }
    if (!edited) continue;
    if (!row.file.empty()) {
      static_cast<void>(Write(row.file, "other bytes\n"));
    }
    auto hashes = std::pair<std::uint64_t, std::string>{};
    try {
      hashes = hashesOf(text, family.lateral);
    } catch (const std::exception& error) {
      ADD_FAILURE() << label << ": " << error.what();
      continue;
    }
    if (!row.file.empty()) {
      fs::copy_file(dir_ / "moved" / row.file, dir_ / row.file,
                    fs::copy_options::overwrite_existing);
    }
    const auto& base = bases.at(row.family);
    const auto checkpointMoved = hashes.first != base.first;
    const auto storeMoved = hashes.second != base.second;
    switch (row.expect) {
      case Expect::Both:
        EXPECT_TRUE(checkpointMoved) << label << ": the checkpoints' hash";
        EXPECT_TRUE(storeMoved) << label << ": the store's hash";
        break;
      case Expect::Store:
        EXPECT_TRUE(storeMoved) << label << ": the store's hash";
        break;
      case Expect::Neither:
        EXPECT_FALSE(checkpointMoved) << label << ": the checkpoints' hash";
        EXPECT_FALSE(storeMoved) << label << ": the store's hash";
        break;
    }
  }
  // a date before present and the same instant on the file's own axis in
  // yr whose products differ in the last bit (0.017 kyr, 17 yr) are two
  // runs, and hash apart; at 1.25 kyr and 1250 yr they are one (a row)
  {
    const auto& base = families.at("static");
    const auto at = [&](std::string_view date) {
      auto text = base.text;
      const auto from = std::string_view{R"("date_kyr_bp": 1.25)"};
      text.replace(text.find(from), from.size(), date);
      return hashesOf(text, base.lateral);
    };
    const auto before = at(R"("date_kyr_bp": 0.017)");
    const auto axis = at(R"("date": -17, "unit": "yr")");
    EXPECT_NE(-0.017 * app::KYR, -17.0 * (app::KYR / 1000.0));
    EXPECT_NE(before.second, axis.second);
    // the present, its zero of either sign one time
    EXPECT_EQ(at(R"("date_kyr_bp": 0)").second,
              at(R"("date": 0, "unit": "yr")").second);
  }
  RecordProperty("rows", static_cast<int>(rows.size()));
}

}  // namespace
