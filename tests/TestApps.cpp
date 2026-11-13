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
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numbers>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/json.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
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

/// The mean displacement of a solve surface file over a Gaussian cap
/// along a direction in the local frame at each point (the radial,
/// tangential and toroidal potentials paired with DisplacementMeasurement's
/// duals of the cap-mean weight).
double VectorCapOf(const planetmodel::SurfaceFile& file, std::size_t node,
                   GeographicPoint site, double width_deg,
                   SurfaceDirection direction, int lmax = 8) {
  const SurfaceGrid grid(lmax, {.nmax = 1});
  const auto cap =
      CapMeanDual(grid, 1.0, site, width_deg * std::numbers::pi / 180.0);
  const auto along = [&](double component) {
    auto out = cap;
    for (auto& v : out) v *= component;
    return out;
  };
  const auto m = DisplacementMeasurement(grid, {.east = along(direction.east),
                                                .north = along(direction.north),
                                                .up = along(direction.up)});
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
  EXPECT_EQ(read, 54);
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

TEST_F(AppsTest, SolveIsBitwiseIndependentOfTheCache) {
  // cache_mb reaches every transform grid of the static problem on lateral
  // moduli (the sea level grid, the 3D operator's coupling and the volume
  // file's radial reference coupling), and the run is bit for bit the
  // same under any setting. At lmax 24 the smallest
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
  for (const auto& [name, field] : standard.variables) {
    EXPECT_EQ(field.values, small.variables.at(name).values) << name;
  }
  for (const auto* name :
       {"step_iterations", "solve_iterations", "step_residual"}) {
    EXPECT_EQ(standard.meta.at(name), small.meta.at(name)) << name;
  }

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

TEST_F(AppsTest, SolveRefusesWhatItsTermsCannotSay) {
  // Each fault of a config without an ocean is refused naming the member:
  // an unknown member anywhere, a load term of two kinds or none, an
  // amplitude history or a viscosity in the static problem, the problem
  // missing, unknown or at odds with the times block, a times block
  // without its end, an empty forcing, the old layout's members, a
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
  for (const auto* name : {"load", "displacement", "potential"}) {
    const auto first = Slice(ramp, name, 0);
    EXPECT_TRUE(std::ranges::all_of(first, [](double v) { return v == 0.0; }))
        << name;
  }
  const auto last = Slice(ramp, "displacement", 2);
  EXPECT_TRUE(std::ranges::any_of(last, [](double v) { return v != 0.0; }));
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
  // date its last field is held. A history on a field term multiplies it.
  // The static problem reads the file at its `date` on the file's own
  // axis: the cap at 2 kyr, half of it at 1 kyr, and its summary says so.
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
  const auto own = run("own", R"({"field": {"file": "ramp.nc"}})");
  const auto shifted = run(
      "shifted", R"({"field": {"file": "ramp_bp.nc", "time_origin": 2}})");
  const auto resampled = run(
      "resampled", R"({"field": {"file": "fine.nc", "variable": "sigma"}})");
  const auto doubled = run(
      "doubled", R"({"field": {"file": "ramp.nc"}, "history": [[0, 2]]})");
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
  // laid off the pole), to rounding (equal to the bit here); the moments
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
    // nodes at 2, 4/3, 2/3 and 0 kyr BP: the source acts at the last two
    auto before = 0.0, after = 0.0;
    for (auto i = std::size_t{0}; i < 2 * fieldSize; ++i) {
      before = std::max(before, std::abs(got[i] - ice[i]));
    }
    for (auto i = 2 * fieldSize; i < 4 * fieldSize; ++i) {
      after = std::max(after, std::abs(got[i] - ice[i]));
    }
    EXPECT_LT(before, 1e-12);
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
  // budget, a layer on the solid surface outside it and a point source,
  // on the marine toy state at 2 kyr BP. The ice change is the history's
  // to the present (a `thickness` term) and a cap of ice laid on the
  // southern ocean, thick enough to ground there. Under frozen shorelines
  // the command is the library's one-node Evolution::Run on a branchless
  // rheology under the same forcing, built here, the cap on the ocean
  // loading nothing; under migrating ones it is the one-node RunMigrating,
  // the ice the state's plus the change, the grounded cap moving the
  // shorelines: the final ocean function written is the library's and not
  // the state's. Both are the library's to rounding (equal to the bit
  // here). The loads applied are written, and the metadata and the
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
  const auto config = [&](const std::string& name,
                          const std::string& shorelines) {
    return Write(name + ".json", std::format(R"({{{},
      "problem": "static",
      "forcing": {{"ice_change": [
                     {{"thickness": {{"file": "history.nc", "date_kyr_bp": 0}}}},
                     {{"cap_thickness": {{"latitude": -60, "longitude": 150,
                       "width_deg": 20, "thickness_m": 5000}}}}],
                  "load": [{}, {}], "seismic": [{}]}},
      "ocean": {{"date_kyr_bp": 2, "shorelines": "{}",
                 "topography": {{"file": "history.nc"}},
                 "ice": {{"file": "history.nc"}}}},
      "rotation": "none", "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, kCap, kDisk, kPoint,
                                             shorelines, name));
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
                           "deg, 5000 m of ice; load (2 terms) + seismic (1 "
                           "source); wrote "),
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
              1e-15);
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
  // layer and a point source, under frozen and under migrating shorelines,
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
  for (const auto v : Slice(melting, "relative_sea_level_change", 0)) {
    EXPECT_EQ(v, 0.0);
  }
  const auto first = Slice(grounding, "ocean_function", 0);
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
                "calibration": {{"rtol": 1e-13, "iterations": 60}},
                "topography": {{"file": "{}"}}, "ice": {{"file": "{}"}}}},
      "rotation": "full", "moments": "hydrostatic",
      "solver": {{"rtol": 1e-13}},
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
  EXPECT_FALSE(reference.variables.contains("ocean_function"));
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
    EXPECT_NE(std::string{error.what()}.find("leaves out the window's end"),
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
  // uninterrupted, and the kernels come about its own shorelines. The
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
  for (const auto name :
       {"relative_sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_EQ(rebuilt.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }

  const auto kernels = Write("kernels.json", "{" + body + R"(, "kernels": {
              "parameters": ["ice", "viscosity"],
              "measurements": [{"type": "relative_sea_level_point", "name": "site",
                                "latitude": -30.0, "longitude": 20.0,
                                "date_kyr_bp": 0.0}],
              "output": "out/kernels"}})");
  summary = app::RunKernel(kernels);
  const auto ice =
      pm::ReadSurfaceFile(dir_ / "out" / "kernels" / "site_ice_kernels.nc");
  // about moving shorelines the initial sea level kernel is not zero
  const auto& dual = ice.variables.at("initial_sea_level_dual").values;
  EXPECT_TRUE(std::ranges::any_of(dual, [](double v) { return v != 0.0; }))
      << summary;
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
    EXPECT_LT(r, 1e-11) << name;
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
    EXPECT_LT(departure, 1e-11) << reference;
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
    EXPECT_LT(Departure(got, want), 1e-10);
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

  // the radial model's fields are the radial fingerprint's, to the bit
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
  const auto times = timesOf(units.Time());
  Evolution evolution(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
  const auto snapshots = evolution.Run(
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
          .sources =
              [&](double t) {
                const auto on = kyrOf(t) >= onset - 1e-12 ? 1.0 : 0.0;
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
              },
          .stressGluts = {}});
  EXPECT_EQ(snapshots.size(), times.size());
  auto out = std::map<std::string, std::vector<double>>{};
  for (const auto& snapshot : snapshots) {
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
  // onset's dates turned to kyr from the window's start), built here, node
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
  // last node, and the volume file; and the volume file under seismic
  // sources.
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
}

TEST_F(AppsTest, SolveRefusesWhatAnOceanOrATideCannotDo) {
  // Refused, naming the member: an ice change without an ocean (its water
  // has nowhere to go), a layer without its density without an ocean as
  // with one, a tide under migrating shorelines (the library's migrating
  // runs take none) and under a calibrated start, a tide's history in the
  // static problem and a history in kyr without its unit (`history_kyr`)
  // on a tide; the kernels in the ice and of the sea level without an
  // ocean, and the tide's under migrating shorelines.
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
          timed(tide, R"({"at": "start", "shorelines": "migrating",
                          "topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "forcing.tide", "migrating shorelines");
  refused(read,
          timed(tide, R"({"at": "end", "topography": {"file": "history.nc"},
                          "ice": {"file": "history.nc"}})"),
          "forcing.tide", "a calibrated initial sea level does not take");
  // the kernels: those in the ice and of the sea level need an ocean, and
  // the tide's are not given under migrating shorelines
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
  refused(app::ReadKernelInput,
          kernel(R"({"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                 R"( "ocean": {"at": "start", "shorelines": "migrating",
              "topography": {"file": "history.nc"},
              "ice": {"file": "history.nc"}},)",
                 std::format(R"("parameters": ["tide"], {})", kLod)),
          "kernels.parameters[0]",
          "is the kernels in a tidal potential, which a run under migrating "
          "shorelines");
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
  // coefficient with its own and a point source from an onset (`onset`),
  // node by node to rounding (about 4e-15 here: in kyr the two step
  // through the same times to the bit), the tangential and toroidal
  // potentials included. The same run in years, every time a thousand
  // times larger, agrees with it to the rounding of its node times, which
  // a step's response amplifies (about 3e-12 here: on this ball a one-ulp
  // change of a node moves the fields by up to 7e-13), and so does a run
  // of forty days in days with its run in kyr (equal to the bit here); and
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
    EXPECT_LT(scaled, 1e-11) << name;
    const auto byDate = Departure(dated.variables.at(name).values,
                                  kyr.variables.at(name).values);
    forms = std::max(forms, byDate);
    EXPECT_LT(byDate, 1e-11) << name;
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
              1e-11);
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
  // nodes, is the cap term under the matching history ramped up over the
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
                                         "time_origin": {:.17g}}}}}]}},
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
    EXPECT_LT(scaled, 1e-11) << name;
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
  // between them), written on the run's own grid, is the cap term under
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
  constexpr auto kField = R"("field": {"file": "water.nc"})";
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
      EXPECT_LT(departure, 1e-11) << name << ", " << shorelines;
    }
    if (std::string_view{shorelines} == "migrating") {
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

TEST_F(AppsTest, SolveHoldsAStatesFilesBeyondTheirSpanButAtACalibratedEnd) {
  // A state read outside its files' time span (the toy history's, 2 to 0
  // kyr BP): their end values are held at the static problem's date and at
  // a dated run's start, the run going ahead on them, and refused at the
  // window's end of a calibrating run (`"at": "end"`), the calibration's
  // target. A characterisation of the reads as they stand.
  static_cast<void>(Deck());
  WriteToyHistory(dir_ / "history.nc");
  const auto fingerprint = [&](const std::string& name, double date) {
    return Write(name + ".json", std::format(R"({{
    "problem": "static", {},
    "forcing": {{"ice_change": [{{"cap_fraction": {{"latitude": 90,
                "longitude": 0, "width_deg": 15, "fraction": -0.1}}}}]}},
    "ocean": {{"date_kyr_bp": {}, "topography": {{"file": "history.nc"}},
              "ice": {{"file": "history.nc"}}}},
    "rotation": "none",
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, date, name));
  };
  static_cast<void>(app::RunSolve(fingerprint("oldest", 2.0)));
  static_cast<void>(app::RunSolve(fingerprint("older", 3.5)));
  for (const auto* name : {"relative_sea_level_change", "ocean_function"}) {
    EXPECT_EQ(Output("older").variables.at(name).values,
              Output("oldest").variables.at(name).values)
        << name;
  }
  const auto window = [&](const std::string& name, const std::string& times,
                          const std::string& at) {
    return Write(name + ".json", std::format(R"({{
    "problem": "quasi_static", {}, "times": {{{}}},
    "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}]}},
    "ocean": {{"at": "{}", "topography": {{"file": "history.nc"}},
              "ice": {{"file": "history.nc"}}}},
    "rotation": "none",
    "output": {{"surface": "out/{}.nc"}}}})",
                                             kBall, times, at, name));
  };
  const auto early = app::RunSolve(window(
      "early", R"("start_kyr_bp": 3.0, "end_kyr_bp": 0.0, "nodes": 4)",
      "start"));
  EXPECT_NE(early.find("(the window's start, 3 kyr BP)"), std::string::npos)
      << early;
  ExpectRefused(
      app::RunSolve,
      window("late", R"("start_kyr_bp": 2.0, "end_kyr_bp": -0.5, "nodes": 3)",
             "end"),
      "ocean.topography.file", "leaves out the window's end");
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
    const auto summary = app::RunKernel(base);
    EXPECT_NE(summary.find("(ice; continuous adjoint)"), std::string::npos)
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

TEST_F(AppsTest, KernelsOfAPostSeismicRunOnItsOwnClock) {
  // A point source under the ball of a 1e19 Pa s mantle without an ocean,
  // its relaxation over ten years on the run's own clock at uneven nodes:
  // the kernels in the source's moment tensor and in the viscosity of a
  // vector displacement at the surface at a time of the clock and of the
  // radial displacement at the last node. Their values are solve's. The
  // run is linear in the source, which acts from its onset at every later
  // node: a solve run with the tensor's member rt raised differs from the
  // base run by the moment kernels' prediction. The viscosity kernels
  // against central differences of solve on the ball with its viscosity
  // scaled by 1 +- 1e-4.
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
          "onset": 0.5}}]}},
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
      if (row[1] > 0.5) predicted += row[6] * 1e20;
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
  // the run's own clock: the 3-D kernels in the source's moment tensor and
  // in the log viscosity of a vector displacement at a time of the clock.
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
          "onset": 0.5}}]}},
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
      if (row[1] > 0.5) predicted += row[6] * 1e20;
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
    EXPECT_TRUE(summary.ends_with("(tide; continuous adjoint)")) << summary;
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
  // radial displacement over a cap and of a vector displacement at a point
  // (its toroidal part the lateral structure's), in the load, the source's
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
            "direction": {{"east": 1, "north": 1}}}}],
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
  static_cast<void>(app::RunSolve(config("raised", "moduli.nc", 4e20)));
  const auto read = [&](const std::string& name) {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto run = read("run");
  const auto stiff = read("stiff");
  const auto soft = read("soft");
  const auto raised = read("raised");
  const auto valueOf = [](const pm::SurfaceFile& file, const std::string& name) {
    return name == "uplift"
               ? CapMeanOf(file, "displacement", 0, {55.0, -80.0}, 10.0)
               : VectorPointOf(file, 0, {40.0, -60.0},
                               {.east = 1.0, .north = 1.0});
  };
  EXPECT_TRUE(run.variables.contains("toroidal_potential"));
  const auto out = dir_ / "out" / "run_kernels";
  const auto& applied = run.variables.at("load_outside_water_budget").values;
  for (const auto* name : {"uplift", "drift"}) {
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
    EXPECT_LT(Gap(D[6] * 1e20, raisedBy), 1e-8) << label;
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
        label, Gap(paired, value), Gap(D[6] * 1e20, raisedBy),
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
      EXPECT_TRUE(
          summary.ends_with(std::format("(moduli; {} adjoint)", method)))
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
        "forcing": {{"seismic": [{}"onset": 0.5}}]}}, "rotation": "none",
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
/// bitwise (their headers, which name the config and the time, aside).
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
      "onset": 0.5})"};
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
       R"("parameters": ["ice", "viscosity"], "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
           "longitude": 10, "width_deg": 25, "date_kyr_bp": 0.5}])",
       2},
      {"lateral", Prem8() + ", " + Window(4) + R"(, "rotation": "full")",
       R"("parameters": ["ice", "viscosity"], "measurements": [
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
      {"postseismic3d", R"("problem": "quasi_static",
        "model": {"deck": "soft.deck", "viscosity": {"file": "viscosity.nc"}},
        "discretisation": {"lmax": 8},
        "times": {"unit": "yr", "at": [0, 1, 3, 6]},
        "forcing": {"seismic": [)" + point + R"(]}, "rotation": "none",
        "solver": {"step_rtol": 1e-12, "step_max_iterations": 200})",
       R"("parameters": ["moment", "viscosity"], "measurements": [
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
       R"("parameters": ["load", "moment", "moduli"], "measurements": [
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
  // no store writes are refused, nothing in them removed; a run resuming
  // from a checkpoint is refused a store it would record (its nodes before
  // the checkpoint are not run again), while one starting over records it;
  // an incomplete store of the run (a run stopped at the wall-time limit)
  // is replaced by the next recording.
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
  // incomplete; the resumption is refused it, and a kernel run of the same
  // run records it whole
  const auto checkpointed =
      R"(, "checkpoint": {"directory": "ckpt", "every": 1,
                          "max_walltime_min": 1e-9})";
  const auto stopped = app::RunSolve(
      config("stopped", "1e-10", R"("directory": "partial")", checkpointed));
  EXPECT_NE(stopped.find("at the wall-time limit"), std::string::npos)
      << stopped;
  EXPECT_NE(stopped.find(std::format("the trajectory store {} left incomplete",
                                     (dir_ / "partial").string())),
            std::string::npos)
      << stopped;
  EXPECT_FALSE(io::TrajectoryStore::ManifestOf(dir_ / "partial").complete);
  ExpectRefused(
      app::RunSolve,
      config("stopped", "1e-10", R"("directory": "partial")", checkpointed),
      "trajectory.directory",
      "names a trajectory store this run would record, and the run resumes "
      "from the checkpoint of node ");
  const auto recorded = app::RunKernel(
      config("stopped", "1e-10", R"("directory": "partial")", checkpointed));
  EXPECT_NE(recorded.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << recorded;
  EXPECT_TRUE(io::TrajectoryStore::ManifestOf(dir_ / "partial").complete);
  // starting over records it too, a complete store of the run being left
  // as it is
  const auto over = app::RunSolve(
      config("stopped", "1e-10", R"("directory": "partial")",
             R"(, "checkpoint": {"directory": "ckpt", "every": 1},
                  "restart": false)"));
  EXPECT_NE(over.find("holds this run complete"), std::string::npos) << over;
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
    app::AddRunMembers(hash, input.run);
    return Hashes{hash.Value(), app::TrajectoryHashOf(input.run),
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
  auto times = std::vector<double>{};
  for (const auto t : {0.0, 0.4, 1.0})
    times.push_back(t * app::KYR / units.Time());
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
      app::AddRunMembers(hash, input.run);
      return std::pair{hash.Value(), app::TrajectoryHashOf(input.run)};
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
  refused("", R"(, "load": [{"cap": {"latitude": 90, "longitude": 0,
             "width_deg": 20, "amplitude_kg_m2": 1.0e6}}])",
          "forcing.load", "acts by its weight");
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

}  // namespace
