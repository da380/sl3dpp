// `sl3d tangent` and the `perturbation` block: the tangent-linear run of
// the problem `solve` solves, along a direction given as the inputs are.
// Its files are the central differences of pairs of `solve` runs whose
// configs have the perturbation's members added to their inputs, plus and
// minus (which is what makes the block a mirror of the inputs), per kind of
// direction, 1-D and 3-D; its measurements' changes are `kernel`'s kernels
// of the same config paired with the perturbation; about a calibrated run
// it is the total derivative, the initial sea level calibrated again, or
// the partial one as asked; and what it cannot do is refused naming the
// member.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <planetmodel/deck.hpp>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "AppFixtures.hpp"
#include "Commands.hpp"
#include "Common.hpp"
#include "sl3dpp/Deck.hpp"
#include "Forcing.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace {

namespace fs = std::filesystem;
namespace pm = planetmodel;
using namespace sl3dpp;
using sl3dpp::test::WriteToyHistory;

constexpr auto kLmax = 8;

class TangentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("sl3dpp_test_tangent_" +
            std::string{::testing::UnitTest::GetInstance()
                            ->current_test_info()
                            ->name()});
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

  /// A surface file a run wrote under out/.
  pm::SurfaceFile Out(const std::string& name) const {
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  }

  fs::path dir_;
};

/// The homogeneous test ball's Maxwell deck with its moduli scaled by
/// 1 + `moduli` (the wave speeds by its square root) and its viscosity
/// `viscosity` Pa s.
std::string BallDeck(double moduli = 0.0, double viscosity = 1e21) {
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

/// The ball's moduli, Pa.
constexpr auto kBallMu = 5500.0 * 5500.0 * 5500.0;
constexpr auto kBallKappa =
    5500.0 * (10000.0 * 10000.0 - 4.0 / 3.0 * 5500.0 * 5500.0);

EarthModel ModelOf(const fs::path& deck,
                   app::ModelFormat format = app::ModelFormat::Gia) {
  const auto read = sl3dpp::Deck::Read(deck, app::FormatOf(format));
  EXPECT_TRUE(read) << read.error().message;
  auto model = EarthModel::FromDeck(*read);
  EXPECT_TRUE(model) << model.error().message;
  return std::move(*model);
}

/// A planetmodel model file over a model's solid layers holding the fields
/// given, each a function of (r, theta, phi) in SI metres and radians;
/// with `radial`, stored as functions of the radius alone (read at the
/// north pole). A field of a direction in the viscosity, which the
/// vocabulary does not name, is written with its spec (LogViscositySpec);
/// every other under the vocabulary's.
using Field3 = std::function<double(double, double, double)>;
void WriteModelFields(const fs::path& path, const EarthModel& model,
                      const std::map<std::string, Field3>& fields, bool radial,
                      const pm::AngularGrid& grid = pm::GaussLegendre(8)) {
  const auto& boundaries = model.Boundaries();
  auto file = pm::ModelFile(
      pm::Skeleton(std::vector<double>(boundaries.begin(), boundaries.end())));
  file.SetName("sl3dpp tangent tests");
  const auto& skeleton = file.GetSkeleton();
  const auto layers = static_cast<int>(skeleton.NLayers());
  file.AddPart(0, layers - 1, pm::ElementEdges(skeleton, 0, layers - 1, 500e3),
               5, grid);
  for (const auto& [name, field] : fields) {
    auto sources = std::map<int, pm::LayerSource>{};
    for (auto layer = 0; layer < layers; ++layer) {
      if (model.IsFluid(static_cast<std::size_t>(layer))) continue;
      sources.emplace(layer, radial ? pm::LayerSource::Radial([field](double r) {
                                        return field(r, 0.0, 0.0);
                                      })
                                    : pm::LayerSource::Lateral(field));
    }
    if (const auto spec = test::LogViscositySpec(name)) {
      file.AddField(name, *spec, sources);
    } else {
      file.AddField(name, sources);
    }
  }
  pm::WriteModelFile(path, file);
}

/// A field on the toy grid at the five dates 2, 1.5, 1, 0.5 and 0 kyr BP,
/// linear between them: `value(k, theta, phi)` at the k-th date; or, with
/// `fixed`, one static field, `value(0, theta, phi)`.
void WriteToyField(
    const fs::path& path, const std::string& name,
    const std::function<double(std::size_t, double, double)>& value,
    bool fixed = false, const pm::Dimensions& dims = pm::units::LENGTH) {
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::array{2.0, 1.5, 1.0, 0.5, 0.0};
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  auto values = std::vector<double>{};
  if (!fixed) file.time = std::vector<double>{};
  for (auto k = std::size_t{0}; k < (fixed ? 1u : dates.size()); ++k) {
    if (!fixed) file.time->push_back(-dates[k] * app::KYR);
    for (const auto theta : grid.Colatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        values.push_back(value(k, theta, phi));
      }
    }
  }
  auto options = pm::SurfaceFieldOptions{};
  if (!fixed) {
    options.support = "node";
    options.time_interpolation = "linear";
  }
  options.dimensions = dims;
  file.variables.emplace(name, pm::SurfaceField(name, values, options));
  pm::WriteSurfaceFile(path, file);
}

/// The shape of the toy history's polar ice cap (WriteToyHistory), one at
/// the pole: a change of the ice in proportion to it takes ice only where
/// there is some.
double ToyCap(double theta) {
  return theta < 0.45 ? std::exp(-std::pow(theta / 0.3, 2)) : 0.0;
}

/// A smooth bump about a colatitude and longitude.
double Bump(double theta, double phi, double theta0, double phi0,
            double width) {
  const auto c = std::cos(theta) * std::cos(theta0) +
                 std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// A field of a surface file at a node (a static field: any node).
std::vector<double> FieldAt(const pm::SurfaceFile& file,
                            const std::string& name, std::size_t node) {
  const auto& variable = file.variables.at(name);
  const auto& values = variable.values;
  const auto support = variable.Text("time_support").value_or("static");
  if (!file.time || support == "static") return values;
  const auto size = values.size() / file.time->size();
  return {values.begin() + static_cast<std::ptrdiff_t>(node * size),
          values.begin() + static_cast<std::ptrdiff_t>((node + 1) * size)};
}

double MaxAbs(std::span<const double> values) {
  auto out = 0.0;
  for (const auto v : values) out = std::max(out, std::abs(v));
  return out;
}

double Gap(double got, double want) {
  return std::abs(got - want) / std::abs(want);
}

/// What a tangent file is held against: the central difference of two
/// solve files, (plus - minus) / (2 eps), field by field. Returns the
/// largest departure of a field of `names` over the largest magnitude of
/// the difference of that field (over all its nodes), and the same for the
/// angular velocity in the metadata; a field whose difference vanishes to
/// 1e-12 of `floor` (the scale of the fields of its kind) must vanish in
/// the tangent file to that too.
struct Agreement {
  double worst = 0.0;
  std::string where;
};
Agreement AgainstDifference(const pm::SurfaceFile& tangent,
                            const pm::SurfaceFile& plus,
                            const pm::SurfaceFile& minus, double eps,
                            std::vector<std::string> names = {}) {
  if (names.empty()) {
    for (const auto& [name, variable] : tangent.variables) {
      names.push_back(name);
    }
  }
  auto out = Agreement{};
  const auto hold = [&out](const std::string& name,
                           std::span<const double> got,
                           std::span<const double> up,
                           std::span<const double> down, double eps_) {
    EXPECT_EQ(got.size(), up.size()) << name;
    if (got.size() != up.size() || up.size() != down.size()) return;
    auto scale = 0.0, diff = 0.0, size = 0.0;
    for (auto i = std::size_t{0}; i < got.size(); ++i) {
      const auto want = (up[i] - down[i]) / (2.0 * eps_);
      scale = std::max(scale, std::abs(want));
      diff = std::max(diff, std::abs(got[i] - want));
      size = std::max(size, std::abs(up[i]) / eps_);
    }
    // a difference at the rounding of its terms says nothing
    if (scale <= 1e-11 * size) {
      EXPECT_LE(MaxAbs(got), 1e-9 * std::max(size, 1e-300)) << name;
      return;
    }
    if (diff / scale > out.worst) {
      out.worst = diff / scale;
      out.where = name;
    }
  };
  for (const auto& name : names) {
    EXPECT_TRUE(plus.variables.contains(name)) << name;
    if (!plus.variables.contains(name)) continue;
    hold(name, tangent.variables.at(name).values,
         plus.variables.at(name).values, minus.variables.at(name).values, eps);
  }
  // the angular velocity as one vector: a component the problem's
  // symmetry leaves at zero is rounding in every run
  const auto omega = [](const pm::SurfaceFile& file) {
    auto all = std::vector<double>{};
    for (const auto* key :
         {"omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s"}) {
      const auto& part = std::get<std::vector<double>>(file.meta.at(key));
      all.insert(all.end(), part.begin(), part.end());
    }
    return all;
  };
  hold("omega_rad_s", omega(tangent), omega(plus), omega(minus), eps);
  return out;
}

/// The fields of the response a solve file holds, which a tangent file
/// holds the first-order changes of.
std::vector<std::string> ResponseOf(const pm::SurfaceFile& file) {
  auto out = std::vector<std::string>{};
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "toroidal_potential", "potential"}) {
    if (file.variables.contains(name)) out.emplace_back(name);
  }
  return out;
}

/// The surface pairing int K delta dS of two fields on a kernel file's
/// grid over the sphere of the file's `radius_m`.
double SurfacePairing(const pm::SurfaceFile& kernels,
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

/// The rows of numbers of a text table, its `#` lines left out.
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

/// The first-order changes of the measurements in a tangent run's table
/// (rows `name type value_si change_si`), by name; and the forward values.
struct Changes {
  std::map<std::string, double> change, value;
};
Changes ChangesIn(const fs::path& table) {
  auto in = std::ifstream(table);
  EXPECT_TRUE(in) << table;
  auto out = Changes{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.starts_with("#")) continue;
    auto words = std::istringstream(line);
    auto name = std::string{}, type = std::string{};
    auto value = 0.0, change = 0.0;
    words >> name >> type >> value >> change;
    out.value[name] = value;
    out.change[name] = change;
  }
  return out;
}

/// A piecewise-linear history's factor at a date (kyr BP), from pairs
/// `[date, factor]` of decreasing dates, held beyond its ends.
double FactorAt(const std::vector<std::array<double, 2>>& history,
                double date) {
  if (date >= history.front()[0]) return history.front()[1];
  if (date <= history.back()[0]) return history.back()[1];
  for (auto k = std::size_t{1}; k < history.size(); ++k) {
    if (date < history[k][0]) continue;
    const auto s =
        (history[k - 1][0] - date) / (history[k - 1][0] - history[k][0]);
    return history[k - 1][1] + s * (history[k][1] - history[k - 1][1]);
  }
  return history.back()[1];
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
        } catch (const pm::Error& error) {
          const auto message = std::string{error.what()};
          EXPECT_NE(message.find(member), std::string::npos) << message;
          EXPECT_NE(message.find(what), std::string::npos) << message;
          throw;
        }
      },
      pm::Error)
      << config;
}

/// A config of blocks: the members given, each `"name": value` text, in an
/// object.
std::string Config(const std::vector<std::string>& members) {
  auto out = std::string{"{"};
  for (const auto& member : members) {
    if (member.empty()) continue;
    out += (out.size() > 1 ? ",\n" : "\n") + member;
  }
  return out + "\n}";
}

/// The `output` block of solve's and the `tangent` block, under out/:
/// `tangent` the tangent block's other members, and `asked` the members
/// both output blocks take beside their surface file (`fields`,
/// `smoothing`).
std::string Outputs(const std::string& name, const std::string& tangent = "",
                    const std::string& asked = "") {
  return std::format(R"("output": {{"surface": "out/{}.nc"{}}},
  "tangent": {{"output": {{"surface": "out/{}_tangent.nc"{}}}{}}})",
                     name, asked, name, asked, tangent);
}

constexpr auto kBall = R"("model": {"deck": "ball.deck"},
  "discretisation": {"lmax": 8})";
constexpr auto kDated = R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5})";
constexpr auto kTight = R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12})";

/// A number of a kernel model file's metadata.
double MetaOf(const fs::path& path, const std::string& key) {
  return std::get<std::vector<double>>(pm::ReadModelFile(path).Meta().at(key))
      .front();
}

/// The 3-D runs' Earth: the ball with a viscosity that varies with
/// latitude, eta = 1e21 exp(kAnomaly cos(theta)) times `more`, and lateral
/// moduli, mu(r) (1 + kShear cos(theta)) and kappa(r) (1 + kBulk sin(theta)
/// cos(phi)), each plus `more`.
constexpr auto kAnomaly = 0.3;
constexpr auto kShear = 0.1;
constexpr auto kBulk = 0.05;
using Angular = std::function<double(double, double)>;

/// The angular grid a 3-D run samples its lateral structure on at degree
/// `lmax` (the coupling's, at the library's oversampling): a model file
/// written on it is read at its own points, with no lateral interpolation
/// between a file and the run, so that a file of a field and the field
/// are one thing to the run.
SurfaceGrid SamplingGrid(int lmax = kLmax) {
  return SurfaceGrid(lmax, {.nmax = 2, .oversampling = 1.5});
}
pm::AngularGrid SamplingGridOfFiles(int lmax = kLmax) {
  const auto grid = SamplingGrid(lmax);
  return pm::GaussLegendre(
      static_cast<int>(grid.CoLatitudes().size()) - 1,
      static_cast<int>(grid.Longitudes().size()));
}

void WriteLateralViscosity(const fs::path& path, const EarthModel& model,
                           const Field3& more = {}) {
  WriteModelFields(path, model,
                   {{"viscosity",
                     [more](double r, double theta, double phi) {
                       return 1e21 *
                              std::exp(kAnomaly * std::cos(theta) +
                                       (more ? more(r, theta, phi) : 0.0));
                     }}},
                   false, SamplingGridOfFiles());
}
void WriteLateralModuli(const fs::path& path, const EarthModel& model,
                        const Angular& shear = {}, const Angular& bulk = {}) {
  ASSERT_TRUE(io::WriteModuli(
      path, model, SamplingGrid(),
      [shear](double, double theta, double phi) {
        return kShear * std::cos(theta) + (shear ? shear(theta, phi) : 0.0);
      },
      [bulk](double, double theta, double phi) {
        return kBulk * std::sin(theta) * std::cos(phi) +
               (bulk ? bulk(theta, phi) : 0.0);
      },
      {.name = "toy moduli", .elementMetres = 500e3}));
}

constexpr auto kTight3D = R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12, "step_rtol": 1e-12, "step_max_iterations": 200)";

TEST_F(TangentTest, TheForcingsDirectionsAreSolvesDifferences) {
  // A radial run through time of the toy ice history under frozen
  // shorelines, with a surface load and a tide beside the ice. Each kind of
  // direction of the forcing, and all of them together, against the central
  // difference of two solve runs whose forcing lists have the
  // perturbation's terms appended, scaled by plus and minus one (the
  // problem is linear in its forcing under frozen shorelines, so the
  // difference is the derivative at any size, to the rounding of the
  // subtraction):
  //  - the ice: a `change` file that is not zero at the first node, and a
  //    cap of a fraction of the state's ice with a history;
  //  - the loads: a cap in the water budget with a history, and a disk
  //    outside it that is a layer on the solid surface;
  //  - the tide: a coefficient with a history.
  // The optional and smoothed fields of the output block are held with the
  // rest: the horizontal displacement and every field smoothed over a cap.
  static_cast<void>(Write("ball.deck", BallDeck()));
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kAsked = R"(, "fields": ["horizontal_displacement"],
    "smoothing": {"kind": "gaussian", "width_deg": 30})";
  // the change of the ice: over the toy cap, where there is ice to take,
  // at every node (the first included), turned by the scale
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(dir_ / name, "ice_thickness_change",
                  [scale](std::size_t k, double theta, double phi) {
                    return scale * (10.0 + 5.0 * static_cast<double>(k)) *
                           ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
                  });
  }
  const auto ice = [](double s) {
    return std::format(
        R"({{"change": {{"file": "{}"}}}},
        {{"cap_fraction": {{"latitude": 80, "longitude": 30, "width_deg": 15,
                           "fraction": {}}},
          "history_kyr_bp": [[2, 0.2], [1, 1.0], [0, 0.5]]}})",
        s > 0 ? "change.nc" : "change_minus.nc", 0.1 * s);
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}},
          "water_budget": true, "moves_solid_surface": false,
          "history_kyr_bp": [[2, 0], [0.5, 1]]}},
        {{"disk": {{"latitude": -40, "longitude": 200, "radius_deg": 25,
                    "amplitude_kg_m2": {}}},
          "water_budget": false, "moves_solid_surface": true,
          "density_kg_m3": 2500}})",
        2.0e4 * s, 5.0e4 * s);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 2, "order": 1, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}, "history_kyr_bp": [[2, 1], [0, 0.3]]}})",
        0.8 * s, -0.4 * s);
  };
  struct Added {
    std::string ice, load, tide;
  };
  // the run's own forcing, with terms appended to its lists
  const auto forcing = [&](const Added& added) {
    return std::format(
        R"("forcing": {{
      "ice_change": [{{"thickness": {{"file": "history.nc"}}}}{}],
      "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                         "amplitude_kg_m2": 1.0e5}},
                "water_budget": true, "moves_solid_surface": false}}{}],
      "tide": [{{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0}}{}]}})",
        added.ice.empty() ? "" : ", " + added.ice,
        added.load.empty() ? "" : ", " + added.load,
        added.tide.empty() ? "" : ", " + added.tide);
  };
  constexpr auto kOcean = R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
  const auto solved = [&](const std::string& name, const Added& added) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json", Config({kBall, kDated, forcing(added), kOcean, kTight,
                                Outputs(name, "", kAsked)}))));
    return Out(name);
  };
  // the perturbation: the same terms, as its own lists
  const auto perturbation = [&](const Added& added) {
    auto lists = std::vector<std::string>{};
    if (!added.ice.empty()) {
      lists.push_back(R"("ice_change": [)" + added.ice + "]");
    }
    if (!added.load.empty()) lists.push_back(R"("load": [)" + added.load + "]");
    if (!added.tide.empty()) lists.push_back(R"("tide": [)" + added.tide + "]");
    auto joined = std::string{};
    for (const auto& list : lists) joined += (joined.empty() ? "" : ", ") + list;
    return R"("perturbation": {"forcing": {)" + joined + "}}";
  };
  const auto tangent = [&](const std::string& name, const Added& added) {
    const auto summary = app::RunTangent(Write(
        name + ".json",
        Config({kBall, kDated, forcing({}), kOcean, kTight,
                Outputs(name, "", kAsked), perturbation(added)})));
    EXPECT_NE(summary.find("; wrote "), std::string::npos) << summary;
    EXPECT_NE(summary.find("(the tangent-linear run along the perturbation's "),
              std::string::npos)
        << summary;
    return Out(name + "_tangent");
  };
  const auto of = [&](double s, int kinds) {
    return Added{.ice = kinds & 1 ? ice(s) : "",
                 .load = kinds & 2 ? load(s) : "",
                 .tide = kinds & 4 ? tide(s) : ""};
  };
  const auto base = solved("base", {});
  auto sum = std::map<std::string, std::vector<double>>{};
  for (const auto& [label, kinds] :
       {std::pair{"ice", 1}, std::pair{"load", 2}, std::pair{"tide", 4},
        std::pair{"all", 7}}) {
    const auto name = std::string{label};
    const auto file = tangent("t_" + name, of(1.0, kinds));
    const auto plus = solved("p_" + name, of(1.0, kinds));
    const auto minus = solved("m_" + name, of(-1.0, kinds));
    // every field of the tangent file is one of solve's
    for (const auto& [field, variable] : file.variables) {
      EXPECT_TRUE(plus.variables.contains(field)) << name << " " << field;
      EXPECT_TRUE(variable.Text("long_name")
                      .value_or("")
                      .starts_with("the first-order change along the "
                                   "perturbation of: "))
          << field;
    }
    EXPECT_FALSE(file.variables.contains("ocean_function"));
    EXPECT_TRUE(file.variables.contains("displacement_east"));
    EXPECT_TRUE(file.variables.contains("displacement_smoothed"));
    EXPECT_TRUE(std::get<std::string>(file.meta.at("tangent"))
                    .starts_with("every field is the first-order change, "));
    const auto agreement = AgainstDifference(file, plus, minus, 1.0);
    std::cout << std::format(
        "[tangent, 1-D, frozen, {}] {} fields; the worst {:.3e} from solve's "
        "difference ({})\n",
        name, file.variables.size(), agreement.worst, agreement.where);
    RecordProperty(name + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-8) << name << " " << agreement.where;
    // the members together are one direction, their sum
    for (const auto& field : ResponseOf(file)) {
      const auto& values = file.variables.at(field).values;
      if (name == "all") {
        auto worst = 0.0;
        for (auto i = std::size_t{0}; i < values.size(); ++i) {
          worst = std::max(worst, std::abs(values[i] - sum.at(field)[i]));
        }
        EXPECT_LT(worst, 1e-9 * MaxAbs(values)) << field;
      } else {
        auto& total = sum[field];
        if (total.empty()) total.assign(values.size(), 0.0);
        for (auto i = std::size_t{0}; i < values.size(); ++i) {
          total[i] += values[i];
        }
      }
    }
  }
  static_cast<void>(base);
}

TEST_F(TangentTest, ADirectionNeedNotBeOfATermTheRunHas) {
  // The perturbation's terms are changes of the forcing, not of the run's
  // own terms: a run under an ice history alone takes a direction in a
  // load, a tide and a seismic source (acting from the window's start),
  // none of which it has, and its files then hold what a run with those
  // terms would, the toroidal potential among them. Against the difference
  // of two solve runs with the terms given, plus and minus; radial, and
  // under a lateral viscosity.
  const auto deck = Write("ball.deck", BallDeck());
  WriteLateralViscosity(dir_ / "eta.nc", ModelOf(deck));
  WriteToyHistory(dir_ / "history.nc");
  const auto lists = [](double s) {
    return std::format(
        R"("load": [{{"cap": {{"latitude": 20, "longitude": 120,
        "width_deg": 20, "amplitude_kg_m2": {}}}, "water_budget": false,
        "moves_solid_surface": false}}],
      "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": {}}}],
      "seismic": [{{"point": {{"latitude": 30, "longitude": 40,
        "depth_km": 300, "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {},
        "rt": {}, "rp": {}, "tp": {}}}}}}}])",
        3.0e4 * s, 0.7 * s, 1e20 * s, -2e20 * s, 1e20 * s, 3e20 * s, -1e20 * s,
        2e20 * s);
  };
  for (const auto& [tag, model, solver] :
       {std::tuple{"radial", R"("deck": "ball.deck")", kTight},
        std::tuple{"lateral",
                   R"("deck": "ball.deck", "viscosity": {"file": "eta.nc"})",
                   R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12, "step_rtol": 1e-12, "step_max_iterations": 200})"}}) {
    const auto name = std::string{tag};
    const auto config = [&](const std::string& run, double s,
                            const std::string& perturbation = "") {
      return Write(
          run + ".json",
          Config({std::format(R"("model": {{{}}},
  "discretisation": {{"lmax": 8}})",
                              model),
                  kDated,
                  std::format(R"("forcing": {{"ice_change": [
    {{"thickness": {{"file": "history.nc"}}}}]{}}})",
                              s == 0.0 ? "" : ", " + lists(s)),
                  R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                  solver, Outputs(run), perturbation}));
    };
    const auto solved = [&](const std::string& run, double s) {
      static_cast<void>(app::RunSolve(config(run, s)));
      return Out(run);
    };
    static_cast<void>(app::RunTangent(config(
        name + "_t", 0.0,
        R"("perturbation": {"forcing": {)" + lists(1.0) + "}}")));
    const auto tangent = Out(name + "_t_tangent");
    EXPECT_TRUE(tangent.variables.contains("toroidal_potential"));
    EXPECT_TRUE(tangent.variables.contains("load_outside_water_budget"));
    const auto plus = solved(name + "_p", 1.0);
    auto names = std::vector<std::string>{};
    for (const auto& [field, variable] : tangent.variables) {
      names.push_back(field);
    }
    const auto agreement = AgainstDifference(
        tangent, plus, solved(name + "_m", -1.0), 1.0, names);
    std::cout << std::format(
        "[tangent, terms the run has none of, {}] {} fields; the worst "
        "{:.3e} from solve's difference ({})\n",
        name, names.size(), agreement.worst, agreement.where);
    RecordProperty(name + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-8) << name << " " << agreement.where;
  }
}

TEST_F(TangentTest, UnderMigratingShorelinesItIsAboutTheRunsOwn) {
  // The toy history through time under migrating shorelines, the state at
  // the window's start, on a topography whose equator stands a few metres
  // above the sea: the melt floods it, and the shorelines move through
  // the run. The tangent-linear run holds the
  // run's own shorelines, so it is the difference of solve runs that flip
  // no cell of the base run's (checked), which are linear in what moves
  // them until one does:
  //  - the ice: a `change` file, not zero at the first node, where the run
  //    then solves its start;
  //  - the state's sea level, given as the state gives it, a topography:
  //    its change is minus the sea level's, and the file's
  //    `initial_sea_level` holds it;
  //  - a load in the water budget and a tide.
  static_cast<void>(Write("ball.deck", BallDeck()));
  WriteToyHistory(dir_ / "history.nc");
  // the state's topography as a file of its own, and it moved
  const auto topography = [](double scale) {
    return [scale](std::size_t, double theta, double phi) {
      return 3000.0 * std::cos(theta) +
             5.0 * std::exp(-std::pow((theta - std::numbers::pi / 2) / 0.15,
                                      2)) +
             scale * 0.5 * Bump(theta, phi, 1.9, 2.0, 0.6);
    };
  };
  WriteToyField(dir_ / "topography.nc", "topography", topography(0.0));
  WriteToyField(dir_ / "topography_plus.nc", "topography", topography(1.0));
  WriteToyField(dir_ / "topography_minus.nc", "topography", topography(-1.0));
  WriteToyField(dir_ / "d_topography.nc", "topography",
                [](std::size_t, double theta, double phi) {
                  return 0.5 * Bump(theta, phi, 1.9, 2.0, 0.6);
                });
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(dir_ / name, "ice_thickness_change",
                  [scale](std::size_t k, double theta, double phi) {
                    return scale * (0.2 + 0.1 * static_cast<double>(k)) *
                           ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
                  });
  }
  struct Case {
    std::string name, ice = "", load = "", tide = "",
                      topography = "topography.nc", perturbation = "";
  };
  const auto config = [&](const Case& c) {
    const auto forcing = std::format(
        R"("forcing": {{
      "ice_change": [{{"thickness": {{"file": "history.nc"}}}}{}]{}{}}})",
        c.ice.empty() ? "" : ", " + c.ice,
        c.load.empty() ? "" : R"(, "load": [)" + c.load + "]",
        c.tide.empty() ? "" : R"(, "tide": [)" + c.tide + "]");
    return Write(c.name + ".json",
                 Config({kBall, kDated, forcing,
                         std::format(R"("ocean": {{"at": "start",
      "shorelines": "migrating", "topography": {{"file": "{}"}},
      "ice": {{"file": "history.nc"}}}})",
                                     c.topography),
                         kTight, Outputs(c.name), c.perturbation}));
  };
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": -30, "longitude": 120, "width_deg": 25,
                    "amplitude_kg_m2": {}}},
          "water_budget": true, "moves_solid_surface": false,
          "history_kyr_bp": [[2, 0.3], [0, 1]]}})",
        200.0 * s);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 2, "order": 1, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}}})",
        0.2 * s, 0.1 * s);
  };
  static_cast<void>(app::RunSolve(config({.name = "base"})));
  const auto base = Out("base");
  ASSERT_NE(FieldAt(base, "ocean_function", 0), FieldAt(base, "ocean_function", 4))
      << "no cell flipped through the run";
  const auto solved = [&](const Case& c) {
    static_cast<void>(app::RunSolve(config(c)));
    auto file = Out(c.name);
    EXPECT_EQ(file.variables.at("ocean_function").values,
              base.variables.at("ocean_function").values)
        << c.name << ": a cell flipped";
    return file;
  };
  struct Kind {
    std::string name;
    Case plus, minus, direction;
  };
  const auto kinds = std::vector<Kind>{
      {"ice",
       {.name = "p_ice", .ice = ice(1.0)},
       {.name = "m_ice", .ice = ice(-1.0)},
       {.name = "t_ice",
        .perturbation = R"("perturbation": {"forcing": {"ice_change": [)" +
                        ice(1.0) + "]}}"}},
      {"sea_level",
       {.name = "p_sea", .topography = "topography_plus.nc"},
       {.name = "m_sea", .topography = "topography_minus.nc"},
       {.name = "t_sea",
        .perturbation = R"("perturbation": {"ocean": {"topography":
            {"file": "d_topography.nc"}}})"}},
      {"load_and_tide",
       {.name = "p_load", .load = load(1.0), .tide = tide(1.0)},
       {.name = "m_load", .load = load(-1.0), .tide = tide(-1.0)},
       {.name = "t_load",
        .perturbation = R"("perturbation": {"forcing": {"load": [)" +
                        load(1.0) + R"(], "tide": [)" + tide(1.0) + "]}}"}}};
  for (const auto& kind : kinds) {
    static_cast<void>(app::RunTangent(config(kind.direction)));
    const auto file = Out(kind.direction.name + "_tangent");
    const auto plus = solved(kind.plus);
    const auto minus = solved(kind.minus);
    EXPECT_FALSE(file.variables.contains("ocean_function"));
    const auto agreement = AgainstDifference(file, plus, minus, 1.0);
    std::cout << std::format(
        "[tangent, 1-D, migrating, {}] {} fields; the worst {:.3e} from "
        "solve's difference ({})\n",
        kind.name, file.variables.size(), agreement.worst, agreement.where);
    RecordProperty(kind.name + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-7) << kind.name << " " << agreement.where;
    if (kind.name == "sea_level") {
      EXPECT_GT(MaxAbs(file.variables.at("initial_sea_level").values), 0.1);
      EXPECT_GT(MaxAbs(file.variables.at("relative_sea_level_change").values),
                0.0);
    }
  }
}

TEST_F(TangentTest, TheEarthsDirectionsAreSolvesDifferences) {
  // The toy history through time on the ball, frozen shorelines, and the
  // directions in the Earth's parameters, each against the central
  // difference of two solve runs on the Earth moved by plus and minus a
  // small step of the direction (1e-4 of the viscosity's, 1e-3 of the
  // moduli's, a ten-thousandth of the moduli):
  //  - the viscosity: a radial model file of the change of ln viscosity, a
  //    function of the radius, against runs on viscosity files of the
  //    viscosity times exp(+-1e-4 of it);
  //  - the moduli: a radial model file of the changes of kappa and mu, a
  //    tenth of each modulus, against runs on decks of the moduli scaled.
  //    The moduli move with the relaxation times held, as `model.moduli`
  //    moves them, so the decks' viscosity is scaled with the shear
  //    modulus.
  // The differences' own error (their remainder at second order in the
  // step, the solves' rounding over it) is what is left.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kStep = 1e-4;
  // the change of ln viscosity, of the radius
  const auto change = [](double r, double, double) {
    return 0.3 + 0.7 * std::pow(r / 6371000.0, 2);
  };
  WriteModelFields(dir_ / "d_eta.nc", model, {{"ln_viscosity", change}}, true);
  for (const auto& [name, scale] : {std::pair{"eta.nc", 0.0},
                                    std::pair{"eta_plus.nc", kStep},
                                    std::pair{"eta_minus.nc", -kStep}}) {
    WriteModelFields(dir_ / name, model,
                     {{"viscosity",
                       [scale, change](double r, double theta, double phi) {
                         return 1e21 * std::exp(scale * change(r, theta, phi));
                       }}},
                     true);
  }
  constexpr auto kTenth = 0.1;
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu", [](double, double, double) { return kTenth * kBallMu; }},
                    {"kappa",
                     [](double, double, double) { return kTenth * kBallKappa; }}},
                   true);
  constexpr auto kModuliStep = 1e-3;
  for (const auto& [name, scale] : {std::pair{"stiff.deck", kModuliStep},
                                    std::pair{"soft.deck", -kModuliStep}}) {
    static_cast<void>(Write(name, BallDeck(kTenth * scale,
                                           1e21 * (1.0 + kTenth * scale))));
  }
  constexpr auto kForcing = R"("forcing": {
    "ice_change": [{"thickness": {"file": "history.nc"}}]})";
  constexpr auto kOcean = R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
  const auto config = [&](const std::string& name, const std::string& model_,
                          const std::string& perturbation = "") {
    return Write(name + ".json",
                 Config({std::format(R"("model": {{{}}},
  "discretisation": {{"lmax": 8}})",
                                     model_),
                         kDated, kForcing, kOcean, kTight, Outputs(name),
                         perturbation}));
  };
  const auto solved = [&](const std::string& name, const std::string& model_) {
    static_cast<void>(app::RunSolve(config(name, model_)));
    return Out(name);
  };
  const auto file = [](const std::string& viscosity) {
    return std::format(R"("deck": "ball.deck", "viscosity": {{"file": "{}"}})",
                       viscosity);
  };
  {
    const auto summary = app::RunTangent(
        config("t_eta", file("eta.nc"),
               R"("perturbation": {"model": {"viscosity":
                   {"file": "d_eta.nc"}}})"));
    EXPECT_TRUE(summary.starts_with("1-D (the viscosity file is radial), "))
        << summary;
    const auto tangent = Out("t_eta_tangent");
    const auto agreement = AgainstDifference(
        tangent, solved("p_eta", file("eta_plus.nc")),
        solved("m_eta", file("eta_minus.nc")), kStep);
    std::cout << std::format(
        "[tangent, 1-D, ln viscosity] the worst {:.3e} from solve's "
        "difference ({})\n",
        agreement.worst, agreement.where);
    RecordProperty("viscosity_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-6) << agreement.where;
    EXPECT_EQ(std::get<std::string>(tangent.meta.at("perturbation_viscosity")),
              (dir_ / "d_eta.nc").string() + " (the change of ln viscosity)");
  }
  {
    static_cast<void>(app::RunTangent(
        config("t_moduli", R"("deck": "ball.deck")",
               R"("perturbation": {"model": {"moduli":
                   {"file": "d_moduli.nc"}}})")));
    const auto agreement = AgainstDifference(
        Out("t_moduli_tangent"), solved("p_moduli", R"("deck": "stiff.deck")"),
        solved("m_moduli", R"("deck": "soft.deck")"), kModuliStep);
    std::cout << std::format(
        "[tangent, 1-D, moduli] the worst {:.3e} from solve's difference "
        "({})\n",
        agreement.worst, agreement.where);
    RecordProperty("moduli_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-6) << agreement.where;
  }
}

TEST_F(TangentTest, TheStaticProblemAndABodyWithoutAnOcean) {
  // The paths the program exposes beside the run through time with an
  // ocean, each a tangent-linear run of one of solve's:
  //  - the static problem with an ocean (one node), frozen shorelines and
  //    migrating: a change of the ice, a load and a tide, and under frozen
  //    shorelines the moduli;
  //  - a body without an ocean, static, and through time on its own clock
  //    under an earthquake at a node of its window: a load, a tide and a
  //    seismic source with the earthquake's onset.
  // Each against the central difference of two solve runs with the
  // perturbation's terms appended to their lists, plus and minus (the
  // moduli's on decks scaled by a ten-thousandth, whose difference of the
  // static problem's angular velocity, which the moduli barely move, keeps
  // six digits where its fields keep eight).
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(
        dir_ / name, "ice_thickness_change",
        [scale](std::size_t, double theta, double phi) {
          return scale * 20.0 * ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
        },
        true);
  }
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto load = [](double s, const std::string& properties) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}{}}})",
        3.0e4 * s, properties);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 3, "order": 2, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}}})",
        0.6 * s, 0.2 * s);
  };
  constexpr auto kBudget = R"(, "water_budget": true,
    "moves_solid_surface": false)";
  const auto added = [](const std::string& terms) {
    return terms.empty() ? std::string{} : ", " + terms;
  };

  // the state's topography as a file of its own, the equator a little
  // above the sea, where the melt floods it; and it moved
  for (const auto& [name, scale] :
       {std::pair{"topography.nc", 0.0}, std::pair{"topography_plus.nc", 1.0},
        std::pair{"topography_minus.nc", -1.0}}) {
    WriteToyField(
        dir_ / name, "topography",
        [scale](std::size_t, double theta, double phi) {
          return 3000.0 * std::cos(theta) +
                 0.5 * std::exp(-std::pow(
                           (theta - std::numbers::pi / 2) / 0.15, 2)) +
                 scale * 0.1 * Bump(theta, phi, 1.6, 2.0, 0.6);
        },
        true);
  }
  WriteToyField(
      dir_ / "d_topography.nc", "topography",
      [](std::size_t, double theta, double phi) {
        return 0.1 * Bump(theta, phi, 1.6, 2.0, 0.6);
      },
      true);

  // -- the static problem with an ocean
  for (const auto* shorelines : {"frozen", "migrating"}) {
    const auto tag = std::string{shorelines};
    const auto config = [&](const std::string& name, double s,
                            const std::string& perturbation = "",
                            const std::string& deck_ = "ball.deck",
                            const std::string& topography = "topography.nc") {
      const auto forcing = std::format(
          R"("forcing": {{
        "ice_change": [{{"cap_fraction": {{"latitude": 85, "longitude": 0,
                                         "width_deg": 20, "fraction": -0.3}}}}{}],
        "load": [{}{}], "tide": [{}{}]}})",
          s == 0.0 ? "" : added(ice(s)), load(1.0, kBudget),
          s == 0.0 ? "" : added(load(s, kBudget)), tide(1.0),
          s == 0.0 ? "" : added(tide(s)));
      return Write(
          tag + "_" + name + ".json",
          Config({std::format(R"("problem": "static",
  "model": {{"deck": "{}"}}, "discretisation": {{"lmax": 8}})",
                              deck_),
                  forcing,
                  std::format(R"("ocean": {{"date_kyr_bp": 2,
    "shorelines": "{}", "topography": {{"file": "{}"}},
    "ice": {{"file": "history.nc"}}}})",
                              tag, topography),
                  kTight, Outputs(tag + "_" + name), perturbation}));
    };
    static_cast<void>(app::RunSolve(config("base", 0.0)));
    const auto base = Out(tag + "_base");
    const auto solved = [&](const std::string& name, double s,
                            const std::string& deck_ = "ball.deck",
                            const std::string& topography = "topography.nc") {
      static_cast<void>(
          app::RunSolve(config(name, s, "", deck_, topography)));
      auto file = Out(tag + "_" + name);
      if (tag == "migrating") {
        EXPECT_EQ(file.variables.at("ocean_function_final").values,
                  base.variables.at("ocean_function_final").values)
            << name << ": a cell flipped";
      }
      return file;
    };
    const auto summary = app::RunTangent(config(
        "t", 0.0,
        R"("perturbation": {"forcing": {"ice_change": [)" + ice(1.0) +
            R"(], "load": [)" + load(1.0, kBudget) + R"(], "tide": [)" +
            tide(1.0) + "]}}"));
    EXPECT_TRUE(summary.starts_with("1-D, elastic, static, ")) << summary;
    const auto tangent = Out(tag + "_t_tangent");
    EXPECT_FALSE(tangent.time);
    EXPECT_TRUE(tangent.variables.contains("ice_change"));
    EXPECT_FALSE(tangent.variables.contains("ocean_function_final"));
    const auto agreement =
        AgainstDifference(tangent, solved("p", 1.0), solved("m", -1.0), 1.0);
    std::cout << std::format(
        "[tangent, 1-D, static, {}] {} fields; the worst {:.3e} from solve's "
        "difference ({})\n",
        tag, tangent.variables.size(), agreement.worst, agreement.where);
    RecordProperty("static_" + tag + "_gap",
                   std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-8) << tag << " " << agreement.where;
    if (tag == "migrating") {
      // the state's sea level, by its topography: it moves the solve
      // through the cells the melt floods
      ASSERT_NE(base.variables.at("ocean_function").values,
                base.variables.at("ocean_function_final").values)
          << "no cell flipped";
      static_cast<void>(app::RunTangent(config(
          "t_sea", 0.0,
          R"("perturbation": {"ocean": {"topography":
              {"file": "d_topography.nc"}}})")));
      const auto sea = Out(tag + "_t_sea_tangent");
      const auto moved = AgainstDifference(
          sea, solved("p_sea", 0.0, "ball.deck", "topography_plus.nc"),
          solved("m_sea", 0.0, "ball.deck", "topography_minus.nc"), 1.0,
          ResponseOf(sea));
      EXPECT_GT(MaxAbs(sea.variables.at("relative_sea_level_change").values),
                0.0);
      std::cout << std::format(
          "[tangent, 1-D, static, migrating, the state's sea level] the "
          "worst {:.3e} from solve's difference ({})\n",
          moved.worst, moved.where);
      RecordProperty("static_sea_level_gap",
                     std::format("{:.3e}", moved.worst));
      EXPECT_LT(moved.worst, 1e-6) << moved.where;
    }
    if (tag != "frozen") continue;
    // the moduli of the static problem's elastic Earth
    constexpr auto kStep = 1e-3;
    constexpr auto kTenth = 0.1;
    WriteModelFields(
        dir_ / "d_moduli.nc", model,
        {{"mu", [](double, double, double) { return kTenth * kBallMu; }},
         {"kappa",
          [](double, double, double) { return kTenth * kBallKappa; }}},
        true);
    static_cast<void>(Write("stiff.deck", BallDeck(kTenth * kStep)));
    static_cast<void>(Write("soft.deck", BallDeck(-kTenth * kStep)));
    static_cast<void>(app::RunTangent(config(
        "t_moduli", 0.0,
        R"("perturbation": {"model": {"moduli": {"file": "d_moduli.nc"}}})")));
    const auto moduli = AgainstDifference(
        Out(tag + "_t_moduli_tangent"), solved("stiff", 0.0, "stiff.deck"),
        solved("soft", 0.0, "soft.deck"), kStep, ResponseOf(base));
    std::cout << std::format(
        "[tangent, 1-D, static, moduli] the worst {:.3e} from solve's "
        "difference ({})\n",
        moduli.worst, moduli.where);
    RecordProperty("static_moduli_gap", std::format("{:.3e}", moduli.worst));
    EXPECT_LT(moduli.worst, 5e-6) << moduli.where;
  }

  // -- a body without an ocean: static, and through time on its own clock
  // under an earthquake at a node
  const auto source = [](double s, const std::string& onset) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 30,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}{}}})",
        1e20 * s, -2e20 * s, 1e20 * s, 3e20 * s, -1e20 * s, 2e20 * s, onset);
  };
  for (const auto* problem : {"static", "quasi_static"}) {
    const auto tag = std::string{"dry_"} + problem;
    const auto timed = std::string{problem} == "quasi_static";
    const auto onset = timed ? std::string{R"(, "onset": 100)"} : std::string{};
    const auto history =
        timed ? std::string{R"(, "history": [[0, 0.2], [300, 1]])"}
              : std::string{};
    const auto config = [&](const std::string& name, double s,
                            const std::string& perturbation = "") {
      const auto forcing = std::format(
          R"("forcing": {{
        "load": [{}{}], "tide": [{}{}],
        "seismic": [{{"fault": {{"latitude": 35, "longitude": 45, "depth_km": 20,
            "strike_deg": 30, "dip_deg": 40, "rake_deg": 90, "length_km": 200,
            "width_km": 80, "moment_nm": 1e21}}{}}}{}]}})",
          load(1.0, ""), s == 0.0 ? "" : added(load(s, history)), tide(1.0),
          s == 0.0 ? "" : added(tide(s)), onset,
          s == 0.0 ? "" : added(source(s, onset)));
      return Write(
          tag + "_" + name + ".json",
          Config({std::format(R"("problem": "{}", {})", problem, kBall),
                  timed ? R"("times": {"unit": "yr", "at": [0, 100, 200, 400]})"
                        : "",
                  forcing, R"("rotation": "full", "moments": "hydrostatic")",
                  Outputs(tag + "_" + name), perturbation}));
    };
    const auto solved = [&](const std::string& name, double s) {
      static_cast<void>(app::RunSolve(config(name, s)));
      return Out(tag + "_" + name);
    };
    const auto summary = app::RunTangent(config(
        "t", 0.0,
        R"("perturbation": {"forcing": {"load": [)" + load(1.0, history) +
            R"(], "tide": [)" + tide(1.0) + R"(], "seismic": [)" +
            source(1.0, onset) + "]}}"));
    EXPECT_NE(summary.find(", no ocean, "), std::string::npos) << summary;
    const auto tangent = Out(tag + "_t_tangent");
    EXPECT_TRUE(tangent.variables.contains("toroidal_potential"));
    EXPECT_FALSE(tangent.variables.contains("initial_sea_level"));
    const auto agreement =
        AgainstDifference(tangent, solved("p", 1.0), solved("m", -1.0), 1.0);
    std::cout << std::format(
        "[tangent, 1-D, no ocean, {}] {} fields; the worst {:.3e} from "
        "solve's difference ({})\n",
        problem, tangent.variables.size(), agreement.worst, agreement.where);
    RecordProperty(tag + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-8) << tag << " " << agreement.where;
  }
}

TEST_F(TangentTest, KernelsPairedWithThePerturbationAreItsChanges) {
  // One config, `kernel` and `tangent`: the kernels of each measurement,
  // paired with the perturbation as the kernel files say they pair, are
  // the first-order changes `tangent` writes for it. A radial run through
  // time of the toy history under frozen shorelines, with a load, a tide
  // and an earthquake at a node beside the ice, by the two-stage step (the
  // default) and by the trapezoid, the discrete adjoint and the discrete
  // tangent-linear run, each the transpose of the other, for measurements
  // of every kind (a cap mean, point values, a vector displacement, a
  // Stokes coefficient, the rotation, and the decay time of a site's
  // series, a fitted datum whose change is its fit's linearisation). The
  // perturbation moves everything at once, and each kind alone:
  //  - the ice by a `change` file, not zero at the first node;
  //  - a load in the water budget with a history and a layer outside it;
  //  - a tide coefficient with a history;
  //  - the earthquake's own moment tensor, from its onset;
  //  - ln viscosity and the moduli by radial model files, functions of
  //    the radius.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kRadius = 6371000.0;
  const auto dates = std::array{2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0};
  constexpr auto kOnset = 1.5;

  // the perturbation's members, and what each is at a node of the run
  WriteToyField(dir_ / "d_ice.nc", "ice_thickness_change",
                [](std::size_t k, double theta, double phi) {
                  return (10.0 + 5.0 * static_cast<double>(k)) * ToyCap(theta) *
                         (1.0 + 0.5 * std::cos(phi));
                });
  const auto loadHistory =
      std::vector<std::array<double, 2>>{{2.0, 0.2}, {0.5, 1.0}};
  const auto tideHistory =
      std::vector<std::array<double, 2>>{{2.0, 1.0}, {0.0, 0.3}};
  constexpr auto kCap = 2.0e4, kLayer = 5.0e4, kDensity = 2500.0;
  constexpr auto kTideRe = 0.8, kTideIm = -0.4;
  const auto moment =
      std::array{1e19, -2e19, 1e19, 3e19, -1e19, 2e19};  // rr tt pp rt rp tp
  const auto logViscosity = [](double r) {
    return 0.3 + 0.7 * std::pow(r / kRadius, 2);
  };
  const auto shear = [](double r) {
    return kBallMu * (0.05 + 0.1 * std::pow(r / kRadius, 2));
  };
  const auto bulk = [](double r) {
    return kBallKappa * (0.1 - 0.05 * (r / kRadius));
  };
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity", [&](double r, double, double) {
                       return logViscosity(r);
                     }}},
                   true);
  WriteModelFields(
      dir_ / "d_moduli.nc", model,
      {{"mu", [&](double r, double, double) { return shear(r); }},
       {"kappa", [&](double r, double, double) { return bulk(r); }}},
      true);
  const auto members = std::map<std::string, std::string>{
      {"ice", R"("forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}]})"},
      {"load", std::format(R"("forcing": {{"load": [
        {{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 25,
                  "amplitude_kg_m2": {}}},
         "water_budget": true, "moves_solid_surface": false,
         "history_kyr_bp": [[2, 0.2], [0.5, 1]]}},
        {{"cap": {{"latitude": -40, "longitude": 200, "width_deg": 30,
                  "amplitude_kg_m2": {}}},
         "water_budget": false, "moves_solid_surface": true,
         "density_kg_m3": {}}}]}})",
                           kCap, kLayer, kDensity)},
      {"tide", std::format(R"("forcing": {{"tide": [
        {{"degree": 2, "order": 1, "amplitude_m2_s2": {},
         "imaginary_m2_s2": {}, "history_kyr_bp": [[2, 1], [0, 0.3]]}}]}})",
                           kTideRe, kTideIm)},
      {"moment", std::format(R"("forcing": {{"seismic": [
        {{"point": {{"latitude": 30, "longitude": 40, "depth_km": 30,
          "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                               "rp": {}, "tp": {}}}}},
         "onset_kyr_bp": {}}}]}})",
                             moment[0], moment[1], moment[2], moment[3],
                             moment[4], moment[5], kOnset)},
      {"viscosity", R"("model": {"viscosity": {"file": "d_eta.nc"}})"},
      {"moduli", R"("model": {"moduli": {"file": "d_moduli.nc"}})"}};
  // everything at once: the forcing's lists and the model's members merged
  const auto everything = std::format(
      R"("forcing": {{"ice_change": [{{"change": {{"file": "d_ice.nc"}}}}],
      {}, {}, {}}}, "model": {{"viscosity": {{"file": "d_eta.nc"}},
                              "moduli": {{"file": "d_moduli.nc"}}}})",
      members.at("load").substr(12, members.at("load").size() - 13),
      members.at("tide").substr(12, members.at("tide").size() - 13),
      members.at("moment").substr(12, members.at("moment").size() - 13));

  constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10, "date_kyr_bp": 1},
      {"type": "displacement_vector_point", "name": "east", "latitude": 32,
       "longitude": 45, "direction": {"east": 1, "north": 0.5},
       "date_kyr_bp": 1.5},
      {"type": "stokes_coefficient", "name": "c21", "degree": 2, "order": 1,
       "date_kyr_bp": 0.5},
      {"type": "rotation", "name": "wobble", "component": "x"},
      {"type": "decay_time", "name": "decay", "latitude": -30,
       "longitude": 10, "width_deg": 25, "window_start_kyr_bp": 1,
       "time_uncertainty_kyr": 0.01, "value_uncertainty_m": 0.001}])";
  const auto config = [&](const std::string& name, const std::string& scheme,
                          const std::string& perturbation) {
    return Write(
        name + ".json",
        Config({kBall, R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 9})",
                std::format(R"("forcing": {{
    "ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
    "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                       "amplitude_kg_m2": 1.0e5}},
              "water_budget": true, "moves_solid_surface": false}}],
    "tide": [{{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0}}],
    "seismic": [{{"point": {{"latitude": 30, "longitude": 40, "depth_km": 30,
        "strike_deg": 20, "dip_deg": 35, "rake_deg": 80, "moment_nm": 1e21}},
      "onset_kyr_bp": {}}}]}})",
                            kOnset),
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::format(R"("rotation": "full", "moments": "hydrostatic",
  "solver": {{"rtol": 1e-12, "scheme": "{}"}})",
                            scheme),
                Outputs(name),
                std::format(R"("kernels": {{
    "parameters": ["ice", "load", "tide", "moment", "viscosity", "moduli"],
    {}, "output": "out/{}_kernels"}})",
                            kMeasurements, name),
                R"("perturbation": {)" + perturbation + "}"}));
  };

  const auto grid = SurfaceGrid(kLmax);
  const auto change = io::FieldHistory::Read(dir_ / "d_ice.nc", grid);
  ASSERT_TRUE(change);
  const auto radians = std::numbers::pi / 180.0;
  const auto cap = app::CapField(grid, {20.0, 120.0}, 25.0 * radians, kCap);
  const auto layer = app::CapField(grid, {-40.0, 200.0}, 30.0 * radians, kLayer);
  auto height = layer;
  for (auto& v : height) v /= kDensity;

  // the kernels of a measurement paired with each kind of the perturbation
  const auto paired = [&](const fs::path& out, const std::string& name) {
    auto by = std::map<std::string, double>{};
    const auto ice = pm::ReadSurfaceFile(out / (name + "_ice_kernels.nc"));
    const auto loads = pm::ReadSurfaceFile(out / (name + "_load_kernels.nc"));
    for (auto n = std::size_t{0}; n < dates.size(); ++n) {
      by["ice"] += SurfacePairing(
          ice, FieldAt(ice, "thickness_dual", n),
          change->Values("ice_thickness_change", -dates[n] * app::KYR));
      by["load"] +=
          FactorAt(loadHistory, dates[n]) *
              SurfacePairing(loads, FieldAt(loads, "load_dual", n), cap) +
          SurfacePairing(
              loads, FieldAt(loads, "load_outside_water_budget_dual", n),
              layer) +
          SurfacePairing(loads, FieldAt(loads, "solid_surface_change_dual", n),
                         height);
    }
    // rows: node date l m dJ_damplitude dJ_dimaginary
    for (const auto& row : TableRows(out / (name + "_tide_kernels.txt"))) {
      if (row[2] != 2.0 || row[3] != 1.0) continue;
      by["tide"] += FactorAt(tideHistory, row[1]) *
                    (kTideRe * row[4] + kTideIm * row[5]);
    }
    // rows: node date point dJ_dMrr dJ_dMtt dJ_dMpp dJ_dMrt dJ_dMrp dJ_dMtp
    for (const auto& row : TableRows(out / (name + "_moment_kernels.txt"))) {
      if (row[1] > kOnset) continue;  // before the onset the source is absent
      for (auto c = std::size_t{0}; c < moment.size(); ++c) {
        by["moment"] += moment[c] * row[3 + c];
      }
    }
    // rows: r_m K_branch_0 W_m3, the kernel per Pa s of the viscosity
    for (const auto& row : TableRows(out / (name + "_viscosity_kernel.txt"))) {
      by["viscosity"] += row[2] * row[1] * 1e21 * logViscosity(row[0]);
    }
    // rows: r_m W_m3 kappa_Pa mu_Pa K_kappa K_mu
    for (const auto& row : TableRows(out / (name + "_moduli_kernel.txt"))) {
      by["moduli"] += row[1] * (row[4] * bulk(row[0]) + row[5] * shear(row[0]));
    }
    return by;
  };

  for (const auto* scheme : {"sdirk2", "exponential_trapezoid"}) {
    const auto tag = std::string{scheme};
    static_cast<void>(app::RunKernel(config(tag, tag, everything)));
    const auto out = dir_ / "out" / (tag + "_kernels");
    auto worst = std::map<std::string, double>{};
    auto tables = std::map<std::string, Changes>{};
    for (const auto& [kind, member] : members) {
      const auto name = tag + "_" + kind;
      static_cast<void>(app::RunTangent(config(name, tag, member)));
      tables[kind] =
          ChangesIn(dir_ / "out" / (name + "_tangent_measurements.txt"));
    }
    const auto summary =
        app::RunTangent(config(tag + "_all", tag, everything));
    EXPECT_NE(summary.find("; the measurements' first-order changes in "),
              std::string::npos)
        << summary;
    tables["all"] =
        ChangesIn(dir_ / "out" / (tag + "_all_tangent_measurements.txt"));
    for (const auto* measurement :
         {"site", "uplift", "east", "c21", "wobble", "decay"}) {
      const auto name = std::string{measurement};
      const auto by = paired(out, name);
      auto total = 0.0;
      for (const auto& [kind, predicted] : by) {
        total += predicted;
        const auto got = tables.at(kind).change.at(name);
        // a kind that cannot move the measurement (the earthquake a
        // measurement before it) pairs to nothing with it
        const auto gap = predicted == 0.0 && got == 0.0
                             ? 0.0
                             : Gap(got, predicted);
        worst[kind] = std::max(worst[kind], gap);
        EXPECT_LT(gap, 1e-8) << tag << " " << name << " " << kind << ": "
                             << got << " against " << predicted;
      }
      const auto got = tables.at("all").change.at(name);
      worst["all"] = std::max(worst["all"], Gap(got, total));
      EXPECT_LT(Gap(got, total), 1e-8) << tag << " " << name;
      // the table's forward value is the kernel run's
      EXPECT_GT(std::abs(tables.at("all").value.at(name)), 0.0);
    }
    auto line = std::format("[tangent against kernels, 1-D, {}]", tag);
    for (const auto& [kind, gap] : worst) {
      line += std::format(" {} {:.2e}", kind, gap);
      RecordProperty(tag + "_" + kind + "_gap", std::format("{:.3e}", gap));
    }
    std::cout << line << "\n";
  }
}

TEST_F(TangentTest, OfACalibratedRunItIsTheTotalDerivative) {
  // A calibrated run (the state at the window's end) under migrating
  // shorelines: the marine toy history on a topography whose equator the
  // melt floods. `solve` finds the initial sea level again for every run,
  // so the derivative of what it computes has the change of the calibrated
  // initial sea level in it, and that is the tangent-linear run `tangent`
  // makes by default, the change found by the linearised calibration.
  //  - Against central differences of solve runs that each calibrate again
  //    and flip no cell of the base run's: along a thickness history (the
  //    first node's ice moving the run's start), along the target (the
  //    state's topography), and along ln viscosity. The file's
  //    `initial_sea_level` is the change of the calibrated one.
  //  - Against `kernel`'s total kernels of the same config, paired with
  //    the perturbation: the thickness kernels with the ice, the viscosity
  //    kernel with ln viscosity, and `target_sea_level_dual` with the
  //    target's change.
  //  - The partial run, the option stated, against the partial kernels:
  //    the calibrated initial sea level is then held, and the block's sea
  //    level member is its change, which `initial_sea_level_dual` pairs
  //    with.
  //  - Under frozen shorelines the linearised calibration is one run.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc", 0.0, true);
  const auto equator = [](double theta) {
    return std::exp(-std::pow((theta - std::numbers::pi / 2) / 0.15, 2));
  };
  // the change of the target's topography: the equator's ring and a bump
  // of the southern sea floor
  const auto moved = [equator](double theta, double phi) {
    return -0.2 * equator(theta) + 0.3 * Bump(theta, phi, 2.2, 2.0, 0.5);
  };
  constexpr auto kTargetStep = 0.5;
  for (const auto& [name, scale] :
       {std::pair{"topography.nc", 0.0},
        std::pair{"topography_plus.nc", kTargetStep},
        std::pair{"topography_minus.nc", -kTargetStep}}) {
    WriteToyField(dir_ / name, "topography",
                  [=](std::size_t, double theta, double phi) {
                    return 3000.0 * std::cos(theta) -
                           3500.0 * std::exp(-std::pow(theta / 0.3, 2)) -
                           equator(theta) + scale * moved(theta, phi);
                  });
  }
  WriteToyField(dir_ / "d_topography.nc", "topography",
                [=](std::size_t, double theta, double phi) {
                  return moved(theta, phi);
                });
  // a thickness history over the cap, zero at the window's end
  const auto amounts = std::array{10.0, 8.0, 6.0, 3.0, 0.0};
  for (const auto& [name, scale] :
       {std::pair{"d_ice.nc", 1.0}, std::pair{"d_ice_minus.nc", -1.0}}) {
    WriteToyField(dir_ / name, "ice_thickness",
                  [=](std::size_t k, double theta, double phi) {
                    return scale * amounts[k] * ToyCap(theta) *
                           (1.0 + 0.5 * std::cos(phi));
                  });
  }
  constexpr auto kStep = 1e-3;
  static_cast<void>(Write("up.deck", BallDeck(0.0, 1e21 * std::exp(kStep))));
  static_cast<void>(Write("down.deck", BallDeck(0.0, 1e21 * std::exp(-kStep))));
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);

  struct Case {
    std::string name, shorelines = "migrating", deck = "ball.deck", ice = "",
                      topography = "topography.nc", perturbation = "",
                      tangent = "", kernels = "";
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({std::format(R"("model": {{"deck": "{}"}},
  "discretisation": {{"lmax": 8}})",
                            c.deck),
                kDated,
                std::format(R"("forcing": {{"ice_change": [
    {{"thickness": {{"file": "history.nc"}}}}{}]}})",
                            c.ice.empty() ? "" : ", " + c.ice),
                std::format(R"("ocean": {{"at": "end", "shorelines": "{}",
    "calibration": {{"rtol": 1e-12, "iterations": 60}},
    "topography": {{"file": "{}"}}, "ice": {{"file": "history.nc"}}}})",
                            c.shorelines, c.topography),
                kTight, Outputs(c.name, c.tangent),
                std::format(R"("kernels": {{"parameters": ["ice", "viscosity"]{},
    "measurements": [
      {{"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
        "longitude": 10, "width_deg": 25}},
      {{"type": "displacement_point", "name": "uplift", "latitude": 55,
        "longitude": 10, "date_kyr_bp": 1}}],
    "output": "out/{}_kernels"}})",
                            c.kernels, c.name),
                c.perturbation.empty()
                    ? ""
                    : R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto term = [](const std::string& file) {
    return std::format(R"({{"thickness": {{"file": "{}"}}}})", file);
  };
  const auto ofIce =
      R"("forcing": {"ice_change": [)" + term("d_ice.nc") + "]}";
  constexpr auto kOfTarget =
      R"("ocean": {"topography": {"file": "d_topography.nc"}})";
  constexpr auto kOfViscosity =
      R"("model": {"viscosity": {"file": "d_eta.nc"}})";
  constexpr auto kPartial = R"(, "calibration": {"derivative": "partial"})";

  static_cast<void>(app::RunSolve(config({.name = "base"})));
  const auto base = Out("base");
  ASSERT_NE(FieldAt(base, "ocean_function", 0),
            FieldAt(base, "ocean_function", 4))
      << "no cell flipped through the run";
  const auto solved = [&](const Case& c) {
    static_cast<void>(app::RunSolve(config(c)));
    auto file = Out(c.name);
    EXPECT_EQ(file.variables.at("ocean_function").values,
              base.variables.at("ocean_function").values)
        << c.name << ": a cell flipped";
    return file;
  };

  // -- against solve's differences, each run calibrating again
  struct Kind {
    std::string name, perturbation;
    Case plus, minus;
    double step, tolerance;
  };
  const auto kinds = std::vector<Kind>{
      {"ice", ofIce, {.name = "p_ice", .ice = term("d_ice.nc")},
       {.name = "m_ice", .ice = term("d_ice_minus.nc")}, 1.0, 1e-7},
      {"target", kOfTarget,
       {.name = "p_target", .topography = "topography_plus.nc"},
       {.name = "m_target", .topography = "topography_minus.nc"}, kTargetStep,
       1e-7},
      {"viscosity", kOfViscosity, {.name = "p_eta", .deck = "up.deck"},
       {.name = "m_eta", .deck = "down.deck"}, kStep, 1e-5}};
  for (const auto& kind : kinds) {
    const auto name = "t_" + kind.name;
    const auto summary =
        app::RunTangent(config({.name = name, .perturbation = kind.perturbation}));
    EXPECT_NE(summary.find("; total derivative, the initial sea level "
                           "calibrated again, its change in "),
              std::string::npos)
        << summary;
    const auto tangent = Out(name + "_tangent");
    EXPECT_EQ(std::get<std::string>(tangent.meta.at("calibration_derivative")),
              "total");
    const auto runs = std::get<std::vector<double>>(
                          tangent.meta.at("calibration_tangent_runs"))
                          .front();
    const auto agreement = AgainstDifference(tangent, solved(kind.plus),
                                             solved(kind.minus), kind.step);
    std::cout << std::format(
        "[tangent, calibrated, total, {}] the linearised calibration in {} "
        "tangent-linear runs; the worst {:.3e} from the difference of solve "
        "runs calibrating again ({})\n",
        kind.name, runs, agreement.worst, agreement.where);
    RecordProperty(kind.name + "_gap", std::format("{:.3e}", agreement.worst));
    RecordProperty(kind.name + "_runs", std::format("{}", runs));
    EXPECT_LT(agreement.worst, kind.tolerance)
        << kind.name << " " << agreement.where;
    // the calibrated initial sea level moves
    EXPECT_GT(MaxAbs(tangent.variables.at("initial_sea_level").values), 0.0);
    // the partial run, alone, is not that derivative
    static_cast<void>(app::RunTangent(config({.name = name + "_partial",
                                              .perturbation = kind.perturbation,
                                              .tangent = kPartial})));
    const auto partial = Out(name + "_partial_tangent");
    EXPECT_EQ(std::get<std::string>(partial.meta.at("calibration_derivative")),
              "partial");
    EXPECT_FALSE(partial.meta.contains("calibration_tangent_runs"));
    if (kind.name != "target") {
      const auto missed =
          AgainstDifference(partial, solved(kind.plus), solved(kind.minus),
                            kind.step, {"relative_sea_level_change"});
      EXPECT_GT(missed.worst, 1e-2) << kind.name;
    }
  }

  // -- against the kernels of the same config, total and partial
  const auto grid = SurfaceGrid(kLmax);
  const auto ice = io::FieldHistory::Read(dir_ / "d_ice.nc", grid);
  const auto topography = io::FieldHistory::Read(dir_ / "d_topography.nc", grid);
  ASSERT_TRUE(ice && topography);
  // the sea level's change is minus the topography's
  auto seaLevel = topography->Values("topography", 0.0);
  for (auto& v : seaLevel) v = -v;
  const auto dates = std::array{2.0, 1.5, 1.0, 0.5, 0.0};
  const auto everything = ofIce + ", " + kOfTarget + ", " + kOfViscosity;
  for (const auto& [tag, shorelines, option, kernelOption] :
       {std::tuple{"total", "migrating", "", ""},
        std::tuple{"partial", "migrating", kPartial, kPartial},
        std::tuple{"frozen", "frozen", "", ""}}) {
    const auto name = std::string{tag};
    const auto both = config({.name = name,
                              .shorelines = shorelines,
                              .perturbation = everything,
                              .tangent = option,
                              .kernels = kernelOption});
    static_cast<void>(app::RunKernel(both));
    const auto summary = app::RunTangent(both);
    const auto file = Out(name + "_tangent");
    if (name == "frozen") {
      // one run: the initial sea level moves nothing
      EXPECT_EQ(std::get<std::vector<double>>(
                    file.meta.at("calibration_tangent_runs")),
                std::vector<double>{1.0});
      EXPECT_TRUE(summary.find("its change in") == std::string::npos)
          << summary;
    }
    const auto changes =
        ChangesIn(dir_ / "out" / (name + "_tangent_measurements.txt"));
    const auto out = dir_ / "out" / (name + "_kernels");
    for (const auto* measurement : {"site", "uplift"}) {
      const auto label = std::string{measurement};
      const auto kernels = pm::ReadSurfaceFile(out / (label + "_ice_kernels.nc"));
      auto predicted = 0.0;
      // the ice at a node less its value at the window's end, the first
      // node's moving the run's start
      const auto last = ice->Values("ice_thickness", 0.0);
      for (auto n = std::size_t{0}; n < dates.size(); ++n) {
        auto delta = ice->Values("ice_thickness", -dates[n] * app::KYR);
        for (auto i = std::size_t{0}; i < delta.size(); ++i) {
          delta[i] -= last[i];
        }
        predicted +=
            SurfacePairing(kernels, FieldAt(kernels, "thickness_dual", n), delta);
      }
      predicted += SurfacePairing(
          kernels,
          kernels.variables
              .at(name == "partial" ? "initial_sea_level_dual"
                                    : "target_sea_level_dual")
              .values,
          seaLevel);
      for (const auto& row : TableRows(out / (label + "_viscosity_kernel.txt"))) {
        predicted += row[2] * row[1] * 1e21;
      }
      const auto got = changes.change.at(label);
      std::cout << std::format(
          "[tangent against kernels, calibrated, {}, {}] change {:.6e}; "
          "{:.3e} from the kernels' pairing\n",
          name, label, got, Gap(got, predicted));
      RecordProperty(name + "_" + label + "_gap",
                     std::format("{:.3e}", Gap(got, predicted)));
      EXPECT_LT(Gap(got, predicted), 1e-7) << name << " " << label;
    }
  }

  // -- the block's member, held to the run
  const auto refused = [&](const std::string& name, const std::string& option,
                           const std::string& member, const std::string& what) {
    ExpectRefused(app::RunTangent,
                  config({.name = name, .perturbation = ofIce, .tangent = option}),
                  member, what);
  };
  refused("bad_word", R"(, "calibration": {"derivative": "all"})",
          "tangent.calibration.derivative", "none of \"total\", \"partial\"");
  refused("bad_iterations", R"(, "calibration": {"iterations": 0})",
          "tangent.calibration.iterations", "fewer than one run");
  refused("partial_rtol",
          R"(, "calibration": {"derivative": "partial", "rtol": 1e-6})",
          "tangent.calibration.rtol",
          "the partial run (`derivative`: \"partial\") does not take");
  // an iteration that runs out is refused, not written
  refused("short", R"(, "calibration": {"iterations": 2})",
          "tangent.calibration",
          "did not find the change of the calibrated initial sea level in");
  EXPECT_FALSE(fs::exists(dir_ / "out" / "short_tangent.nc"));
  // a calibrated run takes its ice changes as thickness histories, and
  // nothing beside the ice: so does a perturbation of it
  ExpectRefused(app::RunTangent,
                config({.name = "beside",
                        .perturbation = R"("forcing": {"tide": [{"degree": 2,
                            "amplitude_m2_s2": 1.0}]})"}),
                "perturbation.forcing.tide",
                "a calibrated initial sea level does not take");
  ExpectRefused(app::RunTangent,
                config({.name = "cap",
                        .perturbation = R"("forcing": {"ice_change": [
                            {"cap_fraction": {"latitude": 80, "longitude": 0,
                              "width_deg": 10, "fraction": 0.1}}]})"}),
                "perturbation.forcing.ice_change[0].cap_fraction",
                "the state at the window's end");
}

TEST_F(TangentTest, In3DItsDirectionsAreSolvesDifferences) {
  // A 3-D run through time: the toy history on the ball under a lateral
  // viscosity and lateral moduli, frozen shorelines. The tangent-linear run
  // against the central difference of two 3-D solve runs:
  //  - the ice, a `change` file, its terms appended plus and minus;
  //  - ln viscosity, a model file of a change that varies with the radius
  //    and laterally, the runs on viscosity files of the viscosity times
  //    exp(+-1e-3 of it);
  //  - the moduli, a model file of lateral changes of mu and kappa, the
  //    runs on moduli files with the changes added, 1e-3 of them (the
  //    relaxation times held, as the file's own perturbation holds them).
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(dir_ / name, "ice_thickness_change",
                  [scale](std::size_t k, double theta, double phi) {
                    return scale * (10.0 + 5.0 * static_cast<double>(k)) *
                           ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
                  });
  }
  constexpr auto kStep = 1e-3;
  const auto logViscosity = [](double r, double theta, double phi) {
    return 0.5 + std::pow(r / 6371000.0, 2) * std::sin(theta) * std::cos(phi);
  };
  WriteModelFields(dir_ / "d_eta.nc", model, {{"ln_viscosity", logViscosity}},
                   false, SamplingGridOfFiles());
  WriteLateralViscosity(dir_ / "eta.nc", model);
  for (const auto& [name, scale] :
       {std::pair{"eta_plus.nc", kStep}, std::pair{"eta_minus.nc", -kStep}}) {
    WriteLateralViscosity(dir_ / name, model,
                          [=](double r, double theta, double phi) {
                            return scale * logViscosity(r, theta, phi);
                          });
  }
  // lateral changes of the moduli, relative to the ball's: of no lateral
  // mean, so that a moduli file with them added moves the run's moduli by
  // them (the file's own lateral mean is its reference)
  const auto shear = [](double theta, double phi) {
    return 0.2 * std::sin(theta) * std::sin(phi);
  };
  const auto bulk = [](double theta, double) { return 0.1 * std::cos(theta); };
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu",
                     [=](double, double theta, double phi) {
                       return kBallMu * shear(theta, phi);
                     }},
                    {"kappa",
                     [=](double, double theta, double phi) {
                       return kBallKappa * bulk(theta, phi);
                     }}},
                   false, SamplingGridOfFiles());
  WriteLateralModuli(dir_ / "moduli.nc", model);
  for (const auto& [name, scale] : {std::pair{"moduli_plus.nc", kStep},
                                    std::pair{"moduli_minus.nc", -kStep}}) {
    WriteLateralModuli(
        dir_ / name, model,
        [=](double theta, double phi) { return scale * shear(theta, phi); },
        [=](double theta, double phi) { return scale * bulk(theta, phi); });
  }
  struct Case {
    std::string name, viscosity = "eta.nc", moduli = "moduli.nc", ice = "",
                      perturbation = "";
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({std::format(R"("model": {{"deck": "ball.deck",
    "viscosity": {{"file": "{}"}}, "moduli": {{"file": "{}"}}}},
  "discretisation": {{"lmax": 8}})",
                            c.viscosity, c.moduli),
                kDated,
                std::format(R"("forcing": {{"ice_change": [
    {{"thickness": {{"file": "history.nc"}}}}{}]}})",
                            c.ice.empty() ? "" : ", " + c.ice),
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::string{kTight3D} + "}", Outputs(c.name),
                c.perturbation.empty()
                    ? ""
                    : R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto solved = [&](const Case& c) {
    static_cast<void>(app::RunSolve(config(c)));
    return Out(c.name);
  };
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  struct Kind {
    std::string name, perturbation;
    Case plus, minus;
    double step, tolerance;
  };
  const auto kinds = std::vector<Kind>{
      {"ice", R"("forcing": {"ice_change": [)" + ice(1.0) + "]}",
       {.name = "p_ice", .ice = ice(1.0)}, {.name = "m_ice", .ice = ice(-1.0)},
       1.0, 1e-8},
      {"viscosity", R"("model": {"viscosity": {"file": "d_eta.nc"}})",
       {.name = "p_eta", .viscosity = "eta_plus.nc"},
       {.name = "m_eta", .viscosity = "eta_minus.nc"}, kStep, 1e-5},
      {"moduli", R"("model": {"moduli": {"file": "d_moduli.nc"}})",
       {.name = "p_moduli", .moduli = "moduli_plus.nc"},
       {.name = "m_moduli", .moduli = "moduli_minus.nc"}, kStep, 1e-5}};
  for (const auto& kind : kinds) {
    const auto summary = app::RunTangent(
        config({.name = "t_" + kind.name, .perturbation = kind.perturbation}));
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity, lateral moduli), viscoelastic, "))
        << summary;
    const auto tangent = Out("t_" + kind.name + "_tangent");
    EXPECT_EQ(std::get<std::vector<long long>>(tangent.meta.at("lateral")),
              std::vector<long long>{1});
    // the radial reference model's fields are solve's alone
    EXPECT_FALSE(tangent.variables.contains("displacement_radial_model"));
    const auto plus = solved(kind.plus);
    const auto minus = solved(kind.minus);
    auto names = std::vector<std::string>{};
    for (const auto& [name, variable] : tangent.variables) {
      names.push_back(name);
    }
    const auto agreement =
        AgainstDifference(tangent, plus, minus, kind.step, names);
    std::cout << std::format(
        "[tangent, 3-D, {}] {} fields; the worst {:.3e} from solve's "
        "difference ({})\n",
        kind.name, names.size(), agreement.worst, agreement.where);
    RecordProperty(kind.name + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, kind.tolerance)
        << kind.name << " " << agreement.where;
  }
}

TEST_F(TangentTest, In3DUnderAnEarthquake) {
  // A 3-D body without an ocean on its own clock, under a lateral
  // viscosity and an earthquake at a node of its window: the
  // tangent-linear run along a seismic source of that onset and a load is
  // the difference of two 3-D solve runs with the terms appended, plus and
  // minus.
  const auto deck = Write("ball.deck", BallDeck());
  WriteLateralViscosity(dir_ / "eta.nc", ModelOf(deck));
  const auto source = [](double s) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 300,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}, "onset": 100}})",
        1e20 * s, -2e20 * s, 1e20 * s, 3e20 * s, -1e20 * s, 2e20 * s);
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}, "history": [[0, 0.2], [300, 1]]}})",
        3.0e4 * s);
  };
  const auto config = [&](const std::string& name, double s,
                          const std::string& perturbation = "") {
    return Write(
        name + ".json",
        Config({R"("problem": "quasi_static",
  "model": {"deck": "ball.deck", "viscosity": {"file": "eta.nc"}},
  "discretisation": {"lmax": 8},
  "times": {"unit": "yr", "at": [0, 100, 200, 400]})",
                std::format(R"("forcing": {{"load": [{}{}],
    "seismic": [{{"fault": {{"latitude": 35, "longitude": 45, "depth_km": 200,
        "strike_deg": 30, "dip_deg": 40, "rake_deg": 90, "length_km": 200,
        "width_km": 80, "moment_nm": 1e21}}, "onset": 100}}{}]}})",
                            load(1.0), s == 0.0 ? "" : ", " + load(s),
                            s == 0.0 ? "" : ", " + source(s)),
                R"("rotation": "none",
  "solver": {"step_rtol": 1e-12, "step_max_iterations": 200})",
                Outputs(name), perturbation}));
  };
  const auto solved = [&](const std::string& name, double s) {
    static_cast<void>(app::RunSolve(config(name, s)));
    return Out(name);
  };
  const auto summary = app::RunTangent(config(
      "t", 0.0,
      R"("perturbation": {"forcing": {"load": [)" + load(1.0) +
          R"(], "seismic": [)" + source(1.0) + "]}}"));
  EXPECT_TRUE(summary.starts_with("3-D (lateral viscosity), viscoelastic, "))
      << summary;
  const auto tangent = Out("t_tangent");
  const auto agreement = AgainstDifference(
      tangent, solved("p", 1.0), solved("m", -1.0), 1.0,
      {"displacement", "tangential_potential", "toroidal_potential",
       "potential", "relative_sea_level_change", "load"});
  std::cout << std::format(
      "[tangent, 3-D, an earthquake] the worst {:.3e} from solve's "
      "difference ({})\n",
      agreement.worst, agreement.where);
  RecordProperty("gap", std::format("{:.3e}", agreement.worst));
  EXPECT_LT(agreement.worst, 1e-7) << agreement.where;
}

TEST_F(TangentTest, In3DKernelsPairedWithThePerturbationAreItsChanges) {
  // The 3-D run of In3DItsDirectionsAreSolvesDifferences, one config for
  // `kernel` and `tangent`, by the two-stage step and by the trapezoid: the
  // kernels of each measurement paired with the perturbation are the
  // changes `tangent` writes. The perturbation moves the ice by a `change`
  // file, ln viscosity by a uniform part and a multiple of the run's own
  // lateral part, and the moduli by a multiple of the run's own lateral
  // perturbation, which the kernel files' metadata pair their fields with
  // (`uniform_log_viscosity_pairing`, `lateral_log_viscosity_pairing`,
  // `lateral_moduli_pairing`). And the same of a calibrated 3-D run under
  // migrating shorelines, its total kernels and its total tangent-linear
  // run, with the target moved.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteToyField(dir_ / "d_ice.nc", "ice_thickness_change",
                [](std::size_t k, double theta, double phi) {
                  return (10.0 + 5.0 * static_cast<double>(k)) * ToyCap(theta) *
                         (1.0 + 0.5 * std::cos(phi));
                });
  WriteLateralViscosity(dir_ / "eta.nc", model);
  WriteLateralModuli(dir_ / "moduli.nc", model);
  constexpr auto kUniform = 0.4, kOwn = 1.5, kOwnModuli = 0.7;
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity",
                     [](double, double theta, double) {
                       return kUniform + kOwn * kAnomaly * std::cos(theta);
                     }}},
                   false, SamplingGridOfFiles());
  WriteModelFields(
      dir_ / "d_moduli.nc", model,
      {{"mu",
        [](double, double theta, double) {
          return kOwnModuli * kBallMu * kShear * std::cos(theta);
        }},
       {"kappa",
        [](double, double theta, double phi) {
          return kOwnModuli * kBallKappa * kBulk * std::sin(theta) *
                 std::cos(phi);
        }}},
      false, SamplingGridOfFiles());
  constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10, "date_kyr_bp": 1}])";
  const auto grid = SurfaceGrid(kLmax);
  const auto dates = std::array{2.0, 1.5, 1.0, 0.5, 0.0};
  const auto change = io::FieldHistory::Read(dir_ / "d_ice.nc", grid);
  ASSERT_TRUE(change);

  for (const auto* scheme : {"sdirk2", "exponential_trapezoid"}) {
    const auto tag = std::string{scheme};
    const auto config = Write(
        tag + ".json",
        Config({R"("model": {"deck": "ball.deck",
    "viscosity": {"file": "eta.nc"}, "moduli": {"file": "moduli.nc"}},
  "discretisation": {"lmax": 8})",
                kDated, R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::format(R"({}, "scheme": "{}"}})", kTight3D, tag),
                Outputs(tag),
                std::format(R"("kernels": {{
    "parameters": ["ice", "viscosity", "moduli"], {},
    "output": "out/{}_kernels"}})",
                            kMeasurements, tag),
                R"("perturbation": {
    "forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}]},
    "model": {"viscosity": {"file": "d_eta.nc"},
              "moduli": {"file": "d_moduli.nc"}}})"}));
    static_cast<void>(app::RunKernel(config));
    static_cast<void>(app::RunTangent(config));
    const auto changes =
        ChangesIn(dir_ / "out" / (tag + "_tangent_measurements.txt"));
    const auto out = dir_ / "out" / (tag + "_kernels");
    for (const auto* measurement : {"site", "uplift"}) {
      const auto name = std::string{measurement};
      const auto ice = pm::ReadSurfaceFile(out / (name + "_ice_kernels.nc"));
      auto predicted = 0.0;
      for (auto n = std::size_t{0}; n < dates.size(); ++n) {
        predicted += SurfacePairing(
            ice, FieldAt(ice, "thickness_dual", n),
            change->Values("ice_thickness_change", -dates[n] * app::KYR));
      }
      const auto viscosity = out / (name + "_viscosity_kernels.nc");
      predicted += kUniform * MetaOf(viscosity, "uniform_log_viscosity_pairing") +
                   kOwn * MetaOf(viscosity, "lateral_log_viscosity_pairing") +
                   kOwnModuli * MetaOf(out / (name + "_moduli_kernels.nc"),
                                       "lateral_moduli_pairing");
      const auto got = changes.change.at(name);
      std::cout << std::format(
          "[tangent against kernels, 3-D, {}, {}] change {:.6e}; {:.3e} from "
          "the kernels' pairing\n",
          tag, name, got, Gap(got, predicted));
      RecordProperty(tag + "_" + name + "_gap",
                     std::format("{:.3e}", Gap(got, predicted)));
      EXPECT_LT(Gap(got, predicted), 1e-8) << tag << " " << name;
    }
  }

  // -- a calibrated 3-D run under migrating shorelines, the total
  // derivatives of both commands
  WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  const auto equator = [](double theta) {
    return std::exp(-std::pow((theta - std::numbers::pi / 2) / 0.15, 2));
  };
  WriteToyField(dir_ / "topography.nc", "topography",
                [=](std::size_t, double theta, double) {
                  return 3000.0 * std::cos(theta) -
                         3500.0 * std::exp(-std::pow(theta / 0.3, 2)) -
                         equator(theta);
                });
  WriteToyField(dir_ / "d_topography.nc", "topography",
                [=](std::size_t, double theta, double phi) {
                  return -0.2 * equator(theta) +
                         0.3 * Bump(theta, phi, 2.2, 2.0, 0.5);
                });
  const auto amounts = std::array{10.0, 8.0, 6.0, 3.0, 0.0};
  WriteToyField(dir_ / "d_thickness.nc", "ice_thickness",
                [=](std::size_t k, double theta, double phi) {
                  return amounts[k] * ToyCap(theta) *
                         (1.0 + 0.5 * std::cos(phi));
                });
  const auto calibrated = Write(
      "calibrated.json",
      Config({R"("model": {"deck": "ball.deck",
    "viscosity": {"file": "eta.nc"}}, "discretisation": {"lmax": 8})",
              kDated, R"("forcing": {"ice_change": [
    {"thickness": {"file": "marine.nc"}}]})",
              R"("ocean": {"at": "end", "shorelines": "migrating",
    "calibration": {"rtol": 1e-12, "iterations": 60},
    "topography": {"file": "topography.nc"}, "ice": {"file": "marine.nc"}})",
              std::string{kTight3D} + "}", Outputs("calibrated"),
              R"("kernels": {"parameters": ["ice", "viscosity"],
    "measurements": [{"type": "relative_sea_level_cap", "name": "site",
      "latitude": -30, "longitude": 10, "width_deg": 25}],
    "output": "out/calibrated_kernels"})",
              R"("perturbation": {
    "forcing": {"ice_change": [{"thickness": {"file": "d_thickness.nc"}}]},
    "model": {"viscosity": {"file": "d_eta.nc"}},
    "ocean": {"topography": {"file": "d_topography.nc"}}})"}));
  static_cast<void>(app::RunKernel(calibrated));
  const auto summary = app::RunTangent(calibrated);
  EXPECT_NE(summary.find("; total derivative, the initial sea level "
                         "calibrated again, its change in "),
            std::string::npos)
      << summary;
  const auto thickness = io::FieldHistory::Read(dir_ / "d_thickness.nc", grid);
  const auto topography = io::FieldHistory::Read(dir_ / "d_topography.nc", grid);
  ASSERT_TRUE(thickness && topography);
  auto seaLevel = topography->Values("topography", 0.0);
  for (auto& v : seaLevel) v = -v;
  const auto changes =
      ChangesIn(dir_ / "out" / "calibrated_tangent_measurements.txt");
  const auto out = dir_ / "out" / "calibrated_kernels";
  const auto runs = std::get<std::vector<double>>(
                        Out("calibrated_tangent")
                            .meta.at("calibration_tangent_runs"))
                        .front();
  for (const auto* measurement : {"site"}) {
    const auto name = std::string{measurement};
    const auto ice = pm::ReadSurfaceFile(out / (name + "_ice_kernels.nc"));
    auto predicted = SurfacePairing(
        ice, ice.variables.at("target_sea_level_dual").values, seaLevel);
    const auto last = thickness->Values("ice_thickness", 0.0);
    for (auto n = std::size_t{0}; n < dates.size(); ++n) {
      auto delta = thickness->Values("ice_thickness", -dates[n] * app::KYR);
      for (auto i = std::size_t{0}; i < delta.size(); ++i) delta[i] -= last[i];
      predicted +=
          SurfacePairing(ice, FieldAt(ice, "thickness_dual", n), delta);
    }
    const auto viscosity = out / (name + "_viscosity_kernels.nc");
    predicted += kUniform * MetaOf(viscosity, "uniform_log_viscosity_pairing") +
                 kOwn * MetaOf(viscosity, "lateral_log_viscosity_pairing");
    const auto got = changes.change.at(name);
    std::cout << std::format(
        "[tangent against kernels, 3-D, calibrated, {}] the linearised "
        "calibration in {} runs; change {:.6e}; {:.3e} from the total "
        "kernels' pairing\n",
        name, runs, got, Gap(got, predicted));
    RecordProperty("calibrated_" + name + "_gap",
                   std::format("{:.3e}", Gap(got, predicted)));
    EXPECT_LT(Gap(got, predicted), 1e-7) << name;
  }
}

TEST_F(TangentTest, ItStepsTheJumpsItsRunSteps) {
  // A direction is read at the nodes the library steps the run over: at a
  // jump of the run's forcing, its left limit and its right. A body without
  // an ocean on its own clock under an earthquake BETWEEN two nodes of its
  // window, where solve steps a pair of nodes of its own: the
  // tangent-linear run along a seismic source of that onset, a load whose
  // history steps there and a tide is the difference of two solve runs. A
  // direction that jumps where the run does not is refused at its member:
  // the run has no pair of nodes for it.
  static_cast<void>(Write("ball.deck", BallDeck()));
  const auto source = [](double s, double onset) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 30,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}, "onset": {}}})",
        1e20 * s, -2e20 * s, 1e20 * s, 3e20 * s, -1e20 * s, 2e20 * s, onset);
  };
  const auto load = [](double s, double step) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}},
          "history": [[0, 0.2], [{}, 1]], "time_interpolation": "previous"}})",
        3.0e4 * s, step);
  };
  const auto tide = [](double s) {
    return std::format(R"({{"degree": 2, "order": 2, "amplitude_m2_s2": {},
                           "history": [[0, 0], [400, 1]]}})",
                       0.5 * s);
  };
  const auto config = [&](const std::string& name, const std::string& added,
                          const std::string& perturbation = "") {
    return Write(
        name + ".json",
        Config({R"("problem": "quasi_static")", kBall,
                R"("times": {"unit": "yr", "at": [0, 100, 200, 400]})",
                std::format(R"("forcing": {{
    "seismic": [{{"fault": {{"latitude": 35, "longitude": 45, "depth_km": 20,
        "strike_deg": 30, "dip_deg": 40, "rake_deg": 90, "length_km": 200,
        "width_km": 80, "moment_nm": 1e21}}, "onset": 150}}]{}}})",
                            added),
                R"("rotation": "full", "moments": "hydrostatic")",
                Outputs(name), perturbation}));
  };
  const auto lists = [&](double s, double onset, double step) {
    return std::format(R"("load": [{}], "tide": [{}], "seismic": [{}])",
                       load(s, step), tide(s), source(s, onset));
  };
  // the perturbed runs: the run's own source list with the direction's
  // source appended, and the direction's other lists
  const auto solved = [&](const std::string& name, double s) {
    static_cast<void>(app::RunSolve(Write(
        name + ".json",
        Config({R"("problem": "quasi_static")", kBall,
                R"("times": {"unit": "yr", "at": [0, 100, 200, 400]})",
                std::format(R"("forcing": {{"load": [{}], "tide": [{}],
    "seismic": [{{"fault": {{"latitude": 35, "longitude": 45, "depth_km": 20,
        "strike_deg": 30, "dip_deg": 40, "rake_deg": 90, "length_km": 200,
        "width_km": 80, "moment_nm": 1e21}}, "onset": 150}}, {}]}})",
                            load(s, 150), tide(s), source(s, 150)),
                R"("rotation": "full", "moments": "hydrostatic")",
                Outputs(name)}))));
    return Out(name);
  };
  static_cast<void>(app::RunTangent(config(
      "t", "", R"("perturbation": {"forcing": {)" + lists(1.0, 150, 150) + "}}")));
  const auto agreement = AgainstDifference(Out("t_tangent"), solved("p", 1.0),
                                           solved("m", -1.0), 1.0);
  std::cout << std::format(
      "[tangent, a jump between nodes] the worst {:.3e} from solve's "
      "difference ({})\n",
      agreement.worst, agreement.where);
  RecordProperty("gap", std::format("{:.3e}", agreement.worst));
  EXPECT_LT(agreement.worst, 1e-8) << agreement.where;
  // a jump of the direction's own, at a node where the run's forcing has
  // none, and between two nodes
  ExpectRefused(app::RunTangent,
                config("at_a_node", "",
                       R"("perturbation": {"forcing": {)" +
                           lists(1.0, 200, 150) + "}}"),
                "perturbation.forcing.seismic[0].onset",
                "jumps at 200 yr, where the run's own forcing does not");
  ExpectRefused(app::RunTangent,
                config("between", "",
                       R"("perturbation": {"forcing": {)" +
                           lists(1.0, 150, 300) + "}}"),
                "perturbation.forcing.load[0].history",
                "jumps at 300 yr, where the run's own forcing does not");
}

TEST_F(TangentTest, OneConfigServesSolveKernelAndTangent) {
  // A config holding `output`, `kernels`, `perturbation` and `tangent`
  // serves the three commands, each checking the others' blocks and using
  // its own: solve writes its file and records the run into the trajectory
  // store, and kernel and tangent sweep and run from that store without
  // the forward run; the perturbation and the tangent block are out of the
  // run's hash. The tangent-linear run from the store is the run in memory
  // to the bit. A fault in a block is found whichever command runs.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity", [](double r, double, double) {
                       return 0.3 + 0.7 * std::pow(r / 6371000.0, 2);
                     }}},
                   true);
  constexpr auto kLoad = R"("forcing": {"load": [{"cap": {"latitude": 20,
      "longitude": 120, "width_deg": 20, "amplitude_kg_m2": 3.0e4},
      "water_budget": true, "moves_solid_surface": false}]})";
  constexpr auto kViscosity = R"("model": {"viscosity": {"file": "d_eta.nc"}})";
  struct Case {
    std::string name, perturbation, tangent = "", store = "";
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({kBall, kDated,
                R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                kTight,
                c.store.empty() ? ""
                                : std::format(
                                      R"("trajectory": {{"directory": "{}"}})",
                                      c.store),
                Outputs(c.name, c.tangent),
                std::format(R"("kernels": {{"parameters": ["load", "viscosity"],
    "measurements": [
      {{"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
        "longitude": 10, "width_deg": 25}}],
    "output": "out/{}_kernels"}})",
                            c.name),
                R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto both = std::string{kLoad} + ", " + kViscosity;

  // -- one config, three commands, one recorded run
  const auto one = config({.name = "one", .perturbation = both, .store = "store"});
  const auto solved = app::RunSolve(one);
  EXPECT_NE(solved.find("recorded the trajectory to the store "),
            std::string::npos)
      << solved;
  const auto forward = Out("one");
  EXPECT_TRUE(forward.variables.contains("ocean_function"));
  EXPECT_FALSE(forward.meta.contains("tangent"));
  const auto swept = app::RunKernel(one);
  EXPECT_NE(swept.find("; the forward run read back from the trajectory store "),
            std::string::npos)
      << swept;
  const auto ran = app::RunTangent(one);
  EXPECT_NE(ran.find("; the forward run read back from the trajectory store "),
            std::string::npos)
      << ran;
  EXPECT_NE(ran.find("(the tangent-linear run along the perturbation's load, "
                     "viscosity)"),
            std::string::npos)
      << ran;
  // solve's file is still the forward run's
  EXPECT_EQ(Out("one").variables.at("displacement").values,
            forward.variables.at("displacement").values);
  const auto fromStore = Out("one_tangent");
  EXPECT_TRUE(fromStore.meta.contains("perturbation_load"));
  EXPECT_TRUE(fromStore.meta.contains("perturbation_viscosity"));
  // the same run in memory, and another direction from the same store
  static_cast<void>(
      app::RunTangent(config({.name = "memory", .perturbation = both})));
  const auto inMemory = Out("memory_tangent");
  for (const auto& name : ResponseOf(inMemory)) {
    EXPECT_EQ(fromStore.variables.at(name).values,
              inMemory.variables.at(name).values)
        << name;
  }
  // a sparse recording (every second node kept, the states between rebuilt
  // by the run's own steps) gives the same run, to the rebuild's rounding
  {
    auto sparse = std::string{};
    {
      auto in = std::ifstream(config({.name = "sparse", .perturbation = both}));
      sparse.assign(std::istreambuf_iterator<char>(in), {});
    }
    sparse.insert(sparse.rfind('}'),
                  R"(, "trajectory": {"record_stride": 2})");
    static_cast<void>(app::RunTangent(Write("sparse.json", sparse)));
    const auto rebuilt = Out("sparse_tangent");
    for (const auto& name : ResponseOf(inMemory)) {
      const auto& got = rebuilt.variables.at(name).values;
      const auto& want = inMemory.variables.at(name).values;
      auto worst = 0.0;
      for (auto i = std::size_t{0}; i < want.size(); ++i) {
        worst = std::max(worst, std::abs(got[i] - want[i]));
      }
      EXPECT_LT(worst, 1e-9 * MaxAbs(want)) << name;
    }
  }
  const auto other = app::RunTangent(
      config({.name = "other", .perturbation = kLoad, .store = "store"}));
  EXPECT_NE(other.find("; the forward run read back from the trajectory store "),
            std::string::npos)
      << other;

  // -- a fault in a block is found by every command
  const auto bad = config({.name = "bad",
                           .perturbation = R"("model": {"deck": "ball.deck"})"});
  for (const auto& run : {app::RunSolve, app::RunKernel, app::RunTangent}) {
    ExpectRefused(run, bad, "perturbation.model.deck",
                  "is not a perturbation's member");
  }
  // the continuous run along the viscosity is not built for the two-stage
  // scheme the config steps by, whichever command reads the blocks
  const auto badTangent = config({.name = "bad_tangent", .perturbation = both,
                                  .tangent = R"(, "method": "continuous")"});
  for (const auto& run : {app::RunSolve, app::RunKernel, app::RunTangent}) {
    ExpectRefused(run, badTangent, "tangent.method",
                  "is not built for the two-stage scheme");
  }
}

TEST_F(TangentTest, TheStaticProblemOnLateralModuli) {
  // The static problem on lateral moduli, a 3-D run of one node: the
  // tangent-linear run along a change of the ice and a multiple of the
  // run's own lateral moduli, against the central difference of two 3-D
  // solve runs (the ice's terms appended, the moduli file's perturbation
  // scaled), and its measurements' changes against `kernel`'s kernels
  // paired with the perturbation.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kStep = 1e-3;
  constexpr auto kOwn = 0.7;
  for (const auto& [name, scale] :
       {std::pair{"moduli.nc", 0.0}, std::pair{"moduli_plus.nc", kStep},
        std::pair{"moduli_minus.nc", -kStep}}) {
    WriteLateralModuli(
        dir_ / name, model,
        [=](double theta, double) {
          return scale * kOwn * kShear * std::cos(theta);
        },
        [=](double theta, double phi) {
          return scale * kOwn * kBulk * std::sin(theta) * std::cos(phi);
        });
  }
  WriteModelFields(
      dir_ / "d_moduli.nc", model,
      {{"mu",
        [](double, double theta, double) {
          return kOwn * kBallMu * kShear * std::cos(theta);
        }},
       {"kappa",
        [](double, double theta, double phi) {
          return kOwn * kBallKappa * kBulk * std::sin(theta) * std::cos(phi);
        }}},
      false, SamplingGridOfFiles());
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(
        dir_ / name, "ice_thickness_change",
        [scale](std::size_t, double theta, double phi) {
          return scale * 20.0 * ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
        },
        true);
  }
  struct Case {
    std::string name, moduli = "moduli.nc", ice = "", perturbation = "";
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({std::format(R"("problem": "static",
  "model": {{"deck": "ball.deck", "moduli": {{"file": "{}"}}}},
  "discretisation": {{"lmax": 8}})",
                            c.moduli),
                std::format(R"("forcing": {{"ice_change": [
    {{"cap_fraction": {{"latitude": 85, "longitude": 0, "width_deg": 20,
                       "fraction": -0.3}}}}{}]}})",
                            c.ice.empty() ? "" : ", " + c.ice),
                R"("ocean": {"date_kyr_bp": 2,
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::string{kTight3D} + "}", Outputs(c.name),
                R"("kernels": {"parameters": ["ice", "moduli"],
    "measurements": [{"type": "relative_sea_level_cap", "name": "site",
      "latitude": -30, "longitude": 10, "width_deg": 25}],
    "output": "out/)" + c.name + R"(_kernels"})",
                c.perturbation.empty()
                    ? ""
                    : R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto solved = [&](const Case& c) {
    static_cast<void>(app::RunSolve(config(c)));
    return Out(c.name);
  };
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto ofIce = R"("forcing": {"ice_change": [)" + ice(1.0) + "]}";
  constexpr auto kOfModuli = R"("model": {"moduli": {"file": "d_moduli.nc"}})";
  {
    const auto summary =
        app::RunTangent(config({.name = "t_ice", .perturbation = ofIce}));
    EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), elastic, static, "))
        << summary;
    const auto agreement = AgainstDifference(
        Out("t_ice_tangent"), solved({.name = "p_ice", .ice = ice(1.0)}),
        solved({.name = "m_ice", .ice = ice(-1.0)}), 1.0);
    std::cout << std::format(
        "[tangent, 3-D, static, ice] the worst {:.3e} from solve's difference "
        "({})\n",
        agreement.worst, agreement.where);
    RecordProperty("ice_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-8) << agreement.where;
  }
  {
    static_cast<void>(
        app::RunTangent(config({.name = "t_moduli", .perturbation = kOfModuli})));
    const auto tangent = Out("t_moduli_tangent");
    const auto agreement = AgainstDifference(
        tangent, solved({.name = "p_moduli", .moduli = "moduli_plus.nc"}),
        solved({.name = "m_moduli", .moduli = "moduli_minus.nc"}), kStep,
        ResponseOf(tangent));
    std::cout << std::format(
        "[tangent, 3-D, static, moduli] the worst {:.3e} from solve's "
        "difference ({})\n",
        agreement.worst, agreement.where);
    RecordProperty("moduli_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-5) << agreement.where;
  }
  const auto both =
      config({.name = "both", .perturbation = ofIce + ", " + kOfModuli});
  static_cast<void>(app::RunKernel(both));
  static_cast<void>(app::RunTangent(both));
  const auto out = dir_ / "out" / "both_kernels";
  const auto kernels = pm::ReadSurfaceFile(out / "site_ice_kernels.nc");
  const auto change =
      io::FieldHistory::Read(dir_ / "change.nc", SurfaceGrid(kLmax));
  ASSERT_TRUE(change);
  const auto predicted =
      SurfacePairing(kernels, kernels.variables.at("thickness_dual").values,
                     change->Values("ice_thickness_change", 0.0)) +
      MetaOf(out / "site_moduli_kernels.nc", "lateral_moduli_pairing") * kOwn;
  const auto got = ChangesIn(dir_ / "out" / "both_tangent_measurements.txt")
                       .change.at("site");
  std::cout << std::format(
      "[tangent against kernels, 3-D, static] change {:.6e}; {:.3e} from the "
      "kernels' pairing\n",
      got, Gap(got, predicted));
  RecordProperty("kernels_gap", std::format("{:.3e}", Gap(got, predicted)));
  EXPECT_LT(Gap(got, predicted), 1e-8);
}

TEST_F(TangentTest, RefusesWhatItCannotDo) {
  // The refusals of the `perturbation` and `tangent` blocks, each naming
  // the member: a member with no counterpart in the run, a direction the
  // library has not built, a file that is not the run's body or does not
  // cover its window, and the blocks' own faults.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  // a change on the toy history's three dates, 2, 1 and 0 kyr BP
  WriteToyField(
      dir_ / "static.nc", "ice_thickness_change",
      [](std::size_t, double theta, double) { return ToyCap(theta); }, true);
  WriteModelFields(dir_ / "d_radial.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }},
                    {"mu", [](double, double, double) { return 1e9; }}},
                   true);
  WriteModelFields(dir_ / "d_lateral.nc", model,
                   {{"ln_viscosity",
                     [](double, double theta, double) { return std::cos(theta); }},
                    {"mu",
                     [](double, double theta, double) {
                       return 1e9 * std::cos(theta);
                     }}},
                   false);
  // a model file of another body, and one without the fields
  static_cast<void>(Write(
      "small.deck",
      "a smaller ball\ngia deck\n0 5500 10000 5500 10000 5500 1 1e21\n"
      "6000000 5500 10000 5500 10000 5500 1 1e21\n"));
  WriteModelFields(dir_ / "d_small.nc", ModelOf(dir_ / "small.deck"),
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  WriteModelFields(dir_ / "d_density.nc", model,
                   {{"rho", [](double, double, double) { return 1.0; }}}, true);
  const auto one = [](double, double, double) { return 1.0; };
  // the viscosity itself, a parameter a burgers deck states, a branch a
  // Maxwell body does not have, and no field at all
  WriteModelFields(dir_ / "d_named.nc", model, {{"viscosity", one}}, true);
  WriteModelFields(dir_ / "d_stated.nc", model,
                   {{"ln_viscosity", one}, {"kelvin_viscosity", one}}, true);
  WriteModelFields(dir_ / "d_branch.nc", model,
                   {{"ln_viscosity_branch_1", one}}, true);
  WriteModelFields(dir_ / "d_moduli_only.nc", model, {{"mu", one}}, true);
  // a ball of two layers, the upper a Burgers body by `model.burgers` and
  // the lower a Maxwell one
  static_cast<void>(Write(
      "two.deck",
      "a ball of two layers\ngia deck\n0 5500 10000 5500 10000 5500 1 1e21\n"
      "3185500 5500 10000 5500 10000 5500 1 1e21\n"
      "3185500 5500 10000 5500 10000 5500 1 1e21\n"
      "6371000 5500 10000 5500 10000 5500 1 1e21\n"));
  WriteModelFields(dir_ / "d_two.nc", ModelOf(dir_ / "two.deck"),
                   {{"ln_viscosity_branch_1", one}}, true);
  constexpr auto kLoad = R"("forcing": {"load": [{"cap": {"latitude": 20,
      "longitude": 120, "width_deg": 20, "amplitude_kg_m2": 3.0e4},
      "water_budget": true, "moves_solid_surface": false}]})";
  struct Case {
    std::string name, perturbation = "", tangent = "", problem = kDated,
                      ocean = R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                      forcing = R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                      model = R"("model": {"deck": "ball.deck"},
  "discretisation": {"lmax": 8})",
                      extra = R"("rotation": "full",
  "moments": "hydrostatic")";
    // whether the `tangent` block is the usual one with `tangent` appended
    // to its output's members (or `tangent` is the block itself, or none),
    // and whether a perturbation is given (a load, where none is stated)
    bool blocks = true, perturbed = true;
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({c.model, c.problem, c.forcing, c.ocean,
                R"("output": {"surface": "out/x.nc"})",
                c.blocks ? std::format(
                               R"("tangent": {{"output": {{"surface":
    "out/x_tangent.nc"{}}}}})",
                               c.tangent)
                         : c.tangent,
                !c.perturbed ? ""
                             : R"("perturbation": )" +
                                   (c.perturbation.empty()
                                        ? "{" + std::string{kLoad} + "}"
                                        : c.perturbation),
                c.extra}));
  };
  const auto refused = [&](const Case& c, const std::string& member,
                           const std::string& what) {
    ExpectRefused(app::RunTangent, config(c), member, what);
  };
  const auto of = [](const std::string& members) { return "{" + members + "}"; };
  constexpr auto kStatic = R"("problem": "static")";
  constexpr auto kStaticOcean = R"("ocean": {"date_kyr_bp": 2,
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
  constexpr auto kDry = R"("forcing": {"load": [{"cap": {"latitude": 20,
      "longitude": 120, "width_deg": 20, "amplitude_kg_m2": 3.0e4}}]})";

  // -- the blocks themselves
  refused({.name = "no_perturbation", .perturbed = false},
          "no_perturbation.json", "has no `perturbation` block");
  refused({.name = "no_tangent", .blocks = false}, "no_tangent.json",
          "has no `tangent` block");
  refused({.name = "not_a_block", .perturbation = R"("load")"}, "perturbation",
          "is not a block");
  refused({.name = "empty", .perturbation = "{}"}, "perturbation",
          "holds no direction");
  refused({.name = "unknown", .perturbation = R"({"ice_change": []})"},
          "perturbation.ice_change", "not a recognised member");
  refused({.name = "no_terms", .perturbation = R"({"forcing": {}})"},
          "perturbation.forcing", "holds no forcing");
  refused({.name = "no_output", .tangent = R"("tangent": {})", .blocks = false},
          "tangent", "has no `output`");
  refused({.name = "exchange", .tangent = R"(, "exchange": "out/x_{node}.txt")"},
          "tangent.output.exchange", "are not built");
  refused({.name = "volume", .tangent = R"(, "volume": "out/x_volume.nc")"},
          "tangent.output.volume", "are not built");
  refused({.name = "calibration",
           .tangent = R"("tangent": {"output": {"surface": "out/x_tangent.nc"},
               "calibration": {"derivative": "partial"}})",
           .blocks = false},
          "tangent.calibration", "this run does not calibrate");

  // -- a member with no counterpart in the run
  refused({.name = "gravity", .ocean = "", .forcing = kDry,
           .extra = R"("gravity": "none")"},
          "gravity", "the tangent-linear run of a run without gravity is not "
                     "built");
  refused({.name = "dry_ice",
           .perturbation = of(R"("forcing": {"ice_change": [
               {"change": {"file": "static.nc"}}]})"),
           .ocean = "", .forcing = kDry},
          "perturbation.forcing.ice_change", "the config has no `ocean` block");
  refused({.name = "dry_sea",
           .perturbation = of(R"("ocean": {"sea_level": {"file": "static.nc"}})"),
           .ocean = "", .forcing = kDry},
          "perturbation.ocean", "the run has no ocean");
  refused({.name = "static_viscosity",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_radial.nc"}})"),
           .problem = kStatic, .ocean = kStaticOcean,
           .forcing = R"("forcing": {"ice_change": [{"cap_fraction": {
               "latitude": 85, "longitude": 0, "width_deg": 20,
               "fraction": -0.3}}]})"},
          "perturbation.model.viscosity", "the static problem's Earth is elastic");
  refused({.name = "static_history",
           .perturbation = of(R"("forcing": {"tide": [{"degree": 2,
               "amplitude_m2_s2": 1, "history_kyr_bp": [[2, 0], [0, 1]]}]})"),
           .problem = kStatic, .ocean = kStaticOcean,
           .forcing = R"("forcing": {"ice_change": [{"cap_fraction": {
               "latitude": 85, "longitude": 0, "width_deg": 20,
               "fraction": -0.3}}]})"},
          "perturbation.forcing.tide[0].history_kyr_bp", "quasi-static");
  refused({.name = "empty_measurements",
           .tangent = R"("tangent": {"output": {"surface": "out/x_tangent.nc"},
               "measurements": []})",
           .blocks = false},
          "tangent.measurements", "holds no measurements");
  refused({.name = "method",
           .tangent = R"("tangent": {"output": {"surface": "out/x_tangent.nc"},
               "method": "exact"})",
           .blocks = false},
          "tangent.method", "exact");
  // the continuous run along the viscosity under the default scheme
  refused({.name = "continuous_sdirk2",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_radial.nc"}})"),
           .tangent = R"("tangent": {"output": {"surface": "out/x_tangent.nc"},
               "method": "continuous"})",
           .blocks = false},
          "tangent.method",
          "state the one-stage scheme in the `solver` block (`scheme`: "
          "\"exponential_trapezoid\"), or take the discrete method");
  refused({.name = "other_member",
           .perturbation = of(R"("ocean": {"sea_level": {"file": "static.nc"}})")},
          "perturbation.ocean.sea_level",
          "give the change of that, `topography`");
  refused({.name = "state_ice",
           .perturbation = of(R"("ocean": {"ice": {"file": "static.nc"}})")},
          "perturbation.ocean.ice", "is not built");
  refused({.name = "land",
           .perturbation = of(R"("ocean": {"land": {"file": "static.nc"}})")},
          "perturbation.ocean.land", "has no direction");
  refused({.name = "reference",
           .perturbation = of(R"("model": {"moduli": {"file": "d_radial.nc",
               "reference": "file_radial"}})")},
          "perturbation.model.moduli.reference",
          "a perturbation's field is the change as it stands");

  // -- a direction the library has not built, found once its file is read
  refused({.name = "lateral_viscosity",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_lateral.nc"}})")},
          "perturbation.model.viscosity.file",
          "the tangent-linear run of a radial run along lateral structure is "
          "not built");
  refused({.name = "lateral_moduli",
           .perturbation = of(R"("model": {"moduli": {"file": "d_lateral.nc"}})")},
          "perturbation.model.moduli.file",
          "the tangent-linear run of a radial run along lateral structure is "
          "not built");

  // -- files that are not the run's
  refused({.name = "other_body",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_small.nc"}})")},
          "perturbation.model.viscosity.file", "is not the model's");
  refused({.name = "other_field",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_density.nc"}})")},
          "perturbation.model.viscosity.file",
          "holds a field `rho`, which a perturbation of the viscosity does not "
          "read");
  refused({.name = "no_field",
           .perturbation =
               of(R"("model": {"viscosity": {"file": "d_moduli_only.nc"}})")},
          "perturbation.model.viscosity.file",
          "holds none of the fields a perturbation of the viscosity is read "
          "from");
  refused({.name = "named_viscosity",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_named.nc"}})")},
          "perturbation.model.viscosity.file",
          "read from the field `ln_viscosity`");
  refused({.name = "stated_parameter",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_stated.nc"}})")},
          "perturbation.model.viscosity.file",
          "holds a field `kelvin_viscosity`, a parameter a prony or burgers "
          "deck (or `model.burgers`) states: a direction in those parameters");
  refused({.name = "no_such_branch",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_branch.nc"}})")},
          "perturbation.model.viscosity.file",
          "holds `ln_viscosity_branch_1`, and the run's rheology has 1 branch,");
  refused({.name = "absent_branch",
           .perturbation = of(R"("model": {"viscosity": {"file": "d_two.nc"}})"),
           .model = R"("model": {"deck": "two.deck", "burgers": [
               {"depth_km": [0, 3185.5], "kelvin_viscosity": 1e19,
                "kelvin_modulus_fraction": 0.5}]},
  "discretisation": {"lmax": 8})"},
          "perturbation.model.viscosity.file",
          "where the rheology relaxes and has no branch 1");
  refused({.name = "no_moduli",
           .perturbation = of(R"("model": {"moduli": {"file": "d_density.nc"}})")},
          "perturbation.model.moduli.file", "holds neither `mu` nor `kappa`");
  // the toy history spans 2 to 0 kyr BP: a window beyond it is not covered
  refused({.name = "outside",
           .perturbation = of(R"("forcing": {"ice_change": [
               {"change": {"file": "history.nc", "variable": "ice_thickness"}}]})"),
           .problem = R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": -1.0, "nodes": 4})",
           .ocean = R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
           .forcing = R"("forcing": {"ice_change": [{"cap_fraction": {
               "latitude": 85, "longitude": 0, "width_deg": 20,
               "fraction": -0.3}}]})"},
          "perturbation.forcing.ice_change[0].change.file", "span");
  refused({.name = "no_variable",
           .perturbation = of(R"("ocean": {"topography": {"file": "static.nc"}})")},
          "perturbation.ocean.topography", "topography");

  // -- a 3-D run takes a change of the moduli where its lateral structure
  // lives: under a lateral viscosity alone, in its viscous mantle, and not
  // in its elastic lithosphere
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  WriteModelFields(dir_ / "d_prem.nc",
                   ModelOf(data + "/prem8.elastic.deck",
                           app::ModelFormat::Elastic),
                   {{"mu", [](double, double, double) { return 1e9; }}}, true);
  refused({.name = "elastic_lid",
           .perturbation = of(R"("model": {"moduli": {"file": "d_prem.nc"}})"),
           .problem = R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 2})",
           .model = std::format(R"("model": {{"deck": "{}/prem8.elastic.deck",
    "format": "elastic", "viscosity": {{"file": "{}/viscosity_log.nc"}}}},
  "discretisation": {{"lmax": 8}})",
                                data, data),
           .extra = R"("rotation": "full")"},
          "perturbation.model.moduli.file",
          "in an element where the 3-D run carries no memory nodes");
}

/// The lidded ball of the deck tests: a viscous interior under an elastic
/// lid 100 km thick, its elastic values the test ball's; `interior` the
/// further columns, each its value in the interior and NaN in the lid.
pm::Deck LiddedBall(const std::vector<std::pair<std::string, double>>& interior,
                    std::vector<std::string> header) {
  constexpr auto kNan = std::numeric_limits<double>::quiet_NaN();
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

/// The lidded ball's Burgers body (a steady viscosity of 1e21 Pa s, a
/// Kelvin element of 1e20 Pa s and half the shear modulus) and its two
/// branches, the slower first as the rheology numbers them: strengths and
/// times (s), planetmodel's conversion.prony.burgers_terms of it.
constexpr auto kSteady = 1e21;
constexpr auto kKelvin = 1e20;
constexpr auto kBurgersStrengths =
    std::array{0.3047166335287642, 0.6952833664712358};
constexpr auto kBurgersTimes = std::array{18850363032.74964, 383295870.33074903};

/// Writes the lidded ball as a burgers deck, its two viscosities scaled.
void WriteBurgersBall(const fs::path& path, double scale = 1.0) {
  pm::WriteDeck(path,
                LiddedBall({{"viscosity", scale * kSteady},
                            {"kelvin_viscosity", scale * kKelvin},
                            {"kelvin_modulus_fraction", 0.5}},
                           pm::BurgersHeader("lidded ball")),
                pm::BURGERS());
}

/// Writes the lidded ball as a prony deck of the branches given, each a
/// strength and a time (s).
void WritePronyBall(const fs::path& path,
                    const std::vector<std::array<double, 2>>& branches) {
  auto interior = std::vector<std::pair<std::string, double>>{};
  for (auto k = std::size_t{0}; k < branches.size(); ++k) {
    interior.emplace_back(std::format("relaxation_strength_{}", k + 1),
                          branches[k][0]);
    interior.emplace_back(std::format("relaxation_time_{}", k + 1),
                          branches[k][1]);
  }
  pm::WriteDeck(
      path,
      LiddedBall(interior, pm::PronyHeader("lidded ball", branches.size())),
      pm::PRONY());
}

/// The largest departure of two files' response fields, each over the
/// first's largest value.
double DepartureOf(const pm::SurfaceFile& a, const pm::SurfaceFile& b) {
  auto worst = 0.0;
  for (const auto& name : ResponseOf(a)) {
    const auto& x = a.variables.at(name).values;
    const auto& y = b.variables.at(name).values;
    EXPECT_EQ(x.size(), y.size()) << name;
    if (x.size() != y.size()) continue;
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      diff = std::max(diff, std::abs(x[i] - y[i]));
    }
    const auto scale = MaxAbs(x);
    worst = std::max(worst, scale > 0.0 ? diff / scale : diff);
  }
  return worst;
}

TEST_F(TangentTest, OnABurgersOrPronyBodyTheViscosityIsItsBranches) {
  // A direction in the viscosity on the rheologies of several branches,
  // the toy history on the lidded ball as a burgers deck and as a prony
  // deck of two branches (strengths 0.7 and 0.3, times 4e8 and 2e10 s):
  //  - `ln_viscosity`, every branch's relaxation time moved alike, against
  //    the central difference of two solve runs on decks with every time
  //    scaled by 1 +- 1e-4 (a burgers deck's two viscosities, a prony
  //    deck's two times; the numbers exact in the decks' text);
  //  - `ln_viscosity_branch_<b>`, one branch alone, against solve runs on
  //    prony decks with that branch's time scaled: the prony deck's branch
  //    1, and the burgers deck's branch 0, which a burgers deck cannot
  //    move alone, against prony decks of the burgers deck's own branches
  //    (kBurgersStrengths and kBurgersTimes), the slower one's time scaled;
  //  - and the changes of two measurements (the block's own
  //    `tangent.measurements`), along a direction of `ln_viscosity` and a
  //    branch's own field together, each a function of the radius, against
  //    `kernel`'s radial kernels a branch paired with it: the sum over the
  //    rows of W K_branch_<b> eta_b delta ln eta_b, with eta_b = mu_b tau_b
  //    the branch's viscosity (its strength times the shear modulus times
  //    its time) and delta ln eta_b the sum of the two fields.
  constexpr auto kStep = 1e-4;
  const auto half = 0.5 * (std::log(1.0 + kStep) - std::log(1.0 - kStep));
  constexpr auto kBranches =
      std::array{std::array{0.7, 4e8}, std::array{0.3, 2e10}};
  WriteToyHistory(dir_ / "history.nc");
  WriteBurgersBall(dir_ / "burgers.deck");
  WriteBurgersBall(dir_ / "burgers_plus.deck", 1.0 + kStep);
  WriteBurgersBall(dir_ / "burgers_minus.deck", 1.0 - kStep);
  const auto scaled = [&](double first, double second) {
    return std::vector<std::array<double, 2>>{
        {kBranches[0][0], kBranches[0][1] * first},
        {kBranches[1][0], kBranches[1][1] * second}};
  };
  WritePronyBall(dir_ / "prony.deck", scaled(1.0, 1.0));
  WritePronyBall(dir_ / "prony_plus.deck", scaled(1.0 + kStep, 1.0 + kStep));
  WritePronyBall(dir_ / "prony_minus.deck", scaled(1.0 - kStep, 1.0 - kStep));
  WritePronyBall(dir_ / "prony_plus_1.deck", scaled(1.0, 1.0 + kStep));
  WritePronyBall(dir_ / "prony_minus_1.deck", scaled(1.0, 1.0 - kStep));
  // the burgers deck's branches as a prony deck, the slower one's time
  // scaled. A deck holds 8 digits (planetmodel's writer, `%16.8g`), which
  // do not hold these times scaled: the times are taken as written, and
  // the step in ln tau is theirs
  const auto written = [](double x) {
    return std::stod(std::format("{:.8g}", x));
  };
  const auto slower = std::array{written(kBurgersTimes[0] * (1.0 + kStep)),
                                 written(kBurgersTimes[0] * (1.0 - kStep))};
  const auto slowerHalf = 0.5 * (std::log(slower[0]) - std::log(slower[1]));
  for (const auto& [name, time] :
       {std::pair{"burgers_plus_0.deck", slower[0]},
        std::pair{"burgers_minus_0.deck", slower[1]}}) {
    WritePronyBall(dir_ / name, {{kBurgersStrengths[0], time},
                                 {kBurgersStrengths[1], kBurgersTimes[1]}});
  }
  const auto model =
      ModelOf(dir_ / "burgers.deck", app::ModelFormat::Burgers);
  const auto one = [](double, double, double) { return 1.0; };
  const auto all = [](double r) { return 0.3 + 0.7 * std::pow(r / 6371e3, 2); };
  const auto own = [](double r) { return 1.0 - 0.5 * (r / 6371e3); };
  WriteModelFields(dir_ / "d_all.nc", model, {{"ln_viscosity", one}}, true);
  WriteModelFields(dir_ / "d_one.nc", model, {{"ln_viscosity_branch_1", one}},
                   true);
  WriteModelFields(dir_ / "d_zero.nc", model, {{"ln_viscosity_branch_0", one}},
                   true);
  for (const auto b : {0, 1}) {
    WriteModelFields(
        dir_ / std::format("d_both_{}.nc", b), model,
        {{"ln_viscosity", [&](double r, double, double) { return all(r); }},
         {std::format("ln_viscosity_branch_{}", b),
          [&](double r, double, double) { return own(r); }}},
        true);
  }
  constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10, "date_kyr_bp": 1}])";
  // a config on a deck: with a direction, tangent's blocks (its own
  // measurements); without, solve's alone or kernel's block
  const auto config = [&](const std::string& name, const std::string& deck,
                          const std::string& format,
                          const std::string& direction = "",
                          bool kernels = false) {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{"deck": "{}", "format": "{}"}},
  "discretisation": {{"lmax": 8}})",
                            deck, format),
                kDated,
                R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                kTight,
                Outputs(name, std::string{", "} + kMeasurements),
                kernels ? std::format(R"("kernels": {{
    "parameters": ["viscosity"], {}, "output": "out/{}_kernels"}})",
                                      kMeasurements, name)
                        : std::string{},
                direction.empty()
                    ? std::string{}
                    : std::format(R"("perturbation": {{"model": {{"viscosity":
    {{"file": "{}"}}}}}})",
                                  direction)}));
  };
  const auto solved = [&](const std::string& name, const std::string& deck,
                          const std::string& format) {
    static_cast<void>(app::RunSolve(config(name, deck, format)));
    return Out(name);
  };
  // the tangent-linear run on the deck of `format`, against solve runs on
  // decks of `differenced` (the same format, or prony) a step `step` in
  // ln tau apart each way
  const auto against = [&](const std::string& name, const std::string& format,
                           const std::string& direction,
                           const std::string& plus, const std::string& minus,
                           const std::string& differenced = "",
                           std::optional<double> step = std::nullopt) {
    static_cast<void>(app::RunTangent(
        config(name, format + ".deck", format, direction)));
    const auto& by = differenced.empty() ? format : differenced;
    const auto agreement = AgainstDifference(
        Out(name + "_tangent"), solved(name + "_plus", plus, by),
        solved(name + "_minus", minus, by), step.value_or(half));
    std::cout << std::format(
        "[tangent, 1-D, {} deck, {}] the worst {:.3e} from solve's "
        "difference ({})\n",
        format, direction, agreement.worst, agreement.where);
    RecordProperty(name + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-6) << name << " " << agreement.where;
  };
  against("burgers_all", "burgers", "d_all.nc", "burgers_plus.deck",
          "burgers_minus.deck");
  against("prony_all", "prony", "d_all.nc", "prony_plus.deck",
          "prony_minus.deck");
  against("prony_one", "prony", "d_one.nc", "prony_plus_1.deck",
          "prony_minus_1.deck");
  against("burgers_zero", "burgers", "d_zero.nc", "burgers_plus_0.deck",
          "burgers_minus_0.deck", "prony", slowerHalf);

  // the kernels a branch paired with the direction
  const auto viscosities = std::map<std::string, std::array<double, 2>>{
      {"burgers",
       {kBurgersStrengths[0] * kBallMu * kBurgersTimes[0],
        kBurgersStrengths[1] * kBallMu * kBurgersTimes[1]}},
      {"prony",
       {kBranches[0][0] * kBallMu * kBranches[0][1],
        kBranches[1][0] * kBallMu * kBranches[1][1]}}};
  for (const auto& [format, eta] : viscosities) {
    static_cast<void>(app::RunKernel(
        config(format + "_k", format + ".deck", format, "", true)));
    const auto out = dir_ / "out" / (format + "_k_kernels");
    for (const auto b : {std::size_t{0}, std::size_t{1}}) {
      const auto name = std::format("{}_both_{}", format, b);
      static_cast<void>(app::RunTangent(config(
          name, format + ".deck", format, std::format("d_both_{}.nc", b))));
      const auto table = dir_ / "out" / (name + "_tangent_measurements.txt");
      const auto changes = ChangesIn(table);
      {
        auto in = std::ifstream(table);
        const auto text = std::string(std::istreambuf_iterator<char>(in), {});
        EXPECT_NE(text.find("# the measurements of `tangent.measurements`"),
                  std::string::npos)
            << text;
      }
      for (const auto* measurement : {"site", "uplift"}) {
        const auto label = std::string{measurement};
        // rows: r_m K_branch_0 K_branch_1 W_m3
        auto predicted = 0.0;
        for (const auto& row :
             TableRows(out / (label + "_viscosity_kernel.txt"))) {
          ASSERT_EQ(row.size(), 4u);
          for (const auto k : {std::size_t{0}, std::size_t{1}}) {
            predicted += row[3] * row[1 + k] * eta[k] *
                         (all(row[0]) + (k == b ? own(row[0]) : 0.0));
          }
        }
        const auto got = changes.change.at(label);
        std::cout << std::format(
            "[tangent against kernels, 1-D, {} deck, ln_viscosity and branch "
            "{}, {}] change {:.6e}; {:.3e} from the branches' kernels' "
            "pairing\n",
            format, b, label, got, Gap(got, predicted));
        RecordProperty(name + "_" + label + "_gap",
                       std::format("{:.3e}", Gap(got, predicted)));
        EXPECT_NE(got, 0.0);
        EXPECT_LT(Gap(got, predicted), 1e-8) << name << " " << label;
      }
    }
  }
}

TEST_F(TangentTest, In3DOnBurgersBodiesTheViscosityIsTheirBranches) {
  // A 3-D run on lateral moduli whose mantle is a Burgers body
  // (`model.burgers` over the viscous mantle of the ball with an elastic
  // core, below), the toy history:
  //  - `ln_viscosity` equal to one against the central difference of two
  //    solve runs with the deck's viscosity and the region's Kelvin
  //    viscosity scaled by 1 +- 1e-3 (the step and the tolerance of the 3-D
  //    gate along ln viscosity, In3DItsDirectionsAreSolvesDifferences);
  //  - a lateral `ln_viscosity` and the same field given as both branches'
  //    own (`ln_viscosity_branch_0` and `ln_viscosity_branch_1`) are one
  //    direction, to rounding;
  //  - and the change of a measurement along 0.4 of the uniform change
  //    against 0.4 of the pairing `kernel` gives its branches' kernels
  //    with it (`uniform_log_viscosity_pairing`, the sum over
  //    `K_ln_eta_0` and `K_ln_eta_1`).
  // The core is elastic below 1000 km radius, so the direction is confined
  // to a shell; the ball viscous down to its centre, whose centre element
  // is carried at the Gauss points, has its own gate
  // (In3DOnABallViscousToItsCentreTheViscosityIsTheDecks).
  constexpr auto kStep = 1e-3;
  constexpr auto kUniform = 0.4;
  const auto half = 0.5 * (std::log(1.0 + kStep) - std::log(1.0 - kStep));
  const auto cored = [](double viscosity) {
    auto out = std::string{
        "a ball with an elastic core\ngia deck for the sl3d tests\n"};
    for (const auto& [r, eta] :
         {std::pair{0.0, std::numeric_limits<double>::quiet_NaN()},
          std::pair{1000e3, std::numeric_limits<double>::quiet_NaN()},
          std::pair{1000e3, viscosity}, std::pair{6371e3, viscosity}}) {
      out += std::format("{} 5500 10000 5500 10000 5500 1 {:.17g}\n", r, eta);
    }
    return out;
  };
  for (const auto& [name, scale] : {std::pair{"ball.deck", 1.0},
                                    std::pair{"plus.deck", 1.0 + kStep},
                                    std::pair{"minus.deck", 1.0 - kStep}}) {
    static_cast<void>(Write(name, cored(1e21 * scale)));
  }
  const auto model = ModelOf(dir_ / "ball.deck");
  WriteToyHistory(dir_ / "history.nc");
  WriteLateralModuli(dir_ / "moduli.nc", model);
  const auto files = SamplingGridOfFiles();
  const auto lateral = [](double r, double theta, double phi) {
    return 0.5 + (r / 6371e3) * std::sin(theta) * std::cos(phi);
  };
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  WriteModelFields(dir_ / "d_part.nc", model,
                   {{"ln_viscosity",
                     [](double, double, double) { return kUniform; }}},
                   true);
  WriteModelFields(dir_ / "d_all.nc", model, {{"ln_viscosity", lateral}}, false,
                   files);
  WriteModelFields(dir_ / "d_each.nc", model,
                   {{"ln_viscosity_branch_0", lateral},
                    {"ln_viscosity_branch_1", lateral}},
                   false, files);
  constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25}])";
  const auto config = [&](const std::string& name, const std::string& deck,
                          double scale, const std::string& direction = "",
                          bool kernels = false) {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{"deck": "{}",
    "moduli": {{"file": "moduli.nc"}},
    "burgers": [{{"depth_km": [0, 5371], "kelvin_viscosity": {:.17g},
                  "kelvin_modulus_fraction": 0.5}}]}},
  "discretisation": {{"lmax": 8}})",
                            deck, 1e20 * scale),
                kDated,
                R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::string{kTight3D} + "}",
                Outputs(name, std::string{", "} + kMeasurements),
                kernels ? std::format(R"("kernels": {{
    "parameters": ["viscosity"], {}, "output": "out/{}_kernels"}})",
                                      kMeasurements, name)
                        : std::string{},
                direction.empty()
                    ? std::string{}
                    : std::format(R"("perturbation": {{"model": {{"viscosity":
    {{"file": "{}"}}}}}})",
                                  direction)}));
  };
  const auto solved = [&](const std::string& name, const std::string& deck,
                          double scale) {
    static_cast<void>(app::RunSolve(config(name, deck, scale)));
    return Out(name);
  };
  const auto summary =
      app::RunTangent(config("one", "ball.deck", 1.0, "d_one.nc"));
  EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), ")) << summary;
  const auto agreement = AgainstDifference(
      Out("one_tangent"), solved("plus", "plus.deck", 1.0 + kStep),
      solved("minus", "minus.deck", 1.0 - kStep), half);
  std::cout << std::format(
      "[tangent, 3-D, model.burgers, ln_viscosity] the worst {:.3e} from "
      "solve's difference ({})\n",
      agreement.worst, agreement.where);
  RecordProperty("burgers_gap", std::format("{:.3e}", agreement.worst));
  EXPECT_LT(agreement.worst, 1e-5) << agreement.where;

  static_cast<void>(app::RunTangent(config("all", "ball.deck", 1.0, "d_all.nc")));
  static_cast<void>(
      app::RunTangent(config("each", "ball.deck", 1.0, "d_each.nc")));
  const auto branches = DepartureOf(Out("all_tangent"), Out("each_tangent"));
  std::cout << std::format(
      "[tangent, 3-D, model.burgers] a lateral ln_viscosity and the same "
      "field as each branch's own: {:.3e} apart\n",
      branches);
  RecordProperty("branches_gap", std::format("{:.3e}", branches));
  EXPECT_GT(MaxAbs(Out("all_tangent").variables.at("displacement").values), 0.0);
  EXPECT_LT(branches, 1e-9);

  static_cast<void>(
      app::RunKernel(config("k", "ball.deck", 1.0, "", true)));
  static_cast<void>(
      app::RunTangent(config("part", "ball.deck", 1.0, "d_part.nc")));
  const auto got = ChangesIn(dir_ / "out" / "part_tangent_measurements.txt")
                       .change.at("site");
  const auto predicted =
      kUniform * MetaOf(dir_ / "out" / "k_kernels" / "site_viscosity_kernels.nc",
                        "uniform_log_viscosity_pairing");
  std::cout << std::format(
      "[tangent against kernels, 3-D, model.burgers] change {:.6e}; {:.3e} "
      "from the branches' kernels' uniform pairing\n",
      got, Gap(got, predicted));
  RecordProperty("pairing_gap", std::format("{:.3e}", Gap(got, predicted)));
  EXPECT_NE(got, 0.0);
  EXPECT_LT(Gap(got, predicted), 1e-8);
}

TEST_F(TangentTest, In3DOnABallViscousToItsCentreTheViscosityIsTheDecks) {
  // The gate of In3DOnBurgersBodiesTheViscosityIsTheirBranches on balls
  // viscous down to their centre, under lateral moduli: the 3-D run
  // carries the element holding the centre at its interior Gauss points,
  // so a direction in the viscosity is taken there as a deck's viscosity
  // moved moves the run. On the homogeneous Maxwell ball, and on the same
  // ball a Burgers body from its surface to its centre (`model.burgers`),
  // the toy history:
  //  - `ln_viscosity` equal to one against the central difference of two
  //    solve runs with the deck's viscosity (and the Kelvin viscosity)
  //    scaled by 1 +- 1e-3, at the 3-D gate's tolerance along ln viscosity;
  //  - and the change of a measurement along 0.4 of the uniform change
  //    against 0.4 of the pairing `kernel` gives with it
  //    (`uniform_log_viscosity_pairing`).
  constexpr auto kStep = 1e-3;
  constexpr auto kUniform = 0.4;
  const auto half = 0.5 * (std::log(1.0 + kStep) - std::log(1.0 - kStep));
  for (const auto& [name, scale] : {std::pair{"ball.deck", 1.0},
                                    std::pair{"plus.deck", 1.0 + kStep},
                                    std::pair{"minus.deck", 1.0 - kStep}}) {
    static_cast<void>(Write(name, BallDeck(0.0, 1e21 * scale)));
  }
  const auto model = ModelOf(dir_ / "ball.deck");
  WriteToyHistory(dir_ / "history.nc");
  WriteLateralModuli(dir_ / "moduli.nc", model);
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  WriteModelFields(dir_ / "d_part.nc", model,
                   {{"ln_viscosity",
                     [](double, double, double) { return kUniform; }}},
                   true);
  constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25}])";
  for (const auto burgers : {false, true}) {
    const auto body = std::string{burgers ? "burgers" : "maxwell"};
    const auto config = [&](const std::string& name, const std::string& deck,
                            double scale, const std::string& direction = "",
                            bool kernels = false) {
      return Write(
          body + "_" + name + ".json",
          Config({std::format(R"("model": {{"deck": "{}",
    "moduli": {{"file": "moduli.nc"}}{}}},
  "discretisation": {{"lmax": 8}})",
                              deck,
                              burgers ? std::format(R"(,
    "burgers": [{{"depth_km": [0, 6371], "kelvin_viscosity": {:.17g},
                  "kelvin_modulus_fraction": 0.5}}])",
                                                    1e20 * scale)
                                      : std::string{}),
                  kDated,
                  R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                  R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                  std::string{kTight3D} + "}",
                  Outputs(body + "_" + name, std::string{", "} + kMeasurements),
                  kernels ? std::format(R"("kernels": {{
    "parameters": ["viscosity"], {}, "output": "out/{}_kernels"}})",
                                        kMeasurements, body + "_" + name)
                          : std::string{},
                  direction.empty()
                      ? std::string{}
                      : std::format(R"("perturbation": {{"model": {{"viscosity":
    {{"file": "{}"}}}}}})",
                                    direction)}));
    };
    const auto solved = [&](const std::string& name, const std::string& deck,
                            double scale) {
      static_cast<void>(app::RunSolve(config(name, deck, scale)));
      return Out(body + "_" + name);
    };
    const auto summary =
        app::RunTangent(config("one", "ball.deck", 1.0, "d_one.nc"));
    EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), ")) << summary;
    const auto agreement = AgainstDifference(
        Out(body + "_one_tangent"), solved("plus", "plus.deck", 1.0 + kStep),
        solved("minus", "minus.deck", 1.0 - kStep), half);
    std::cout << std::format(
        "[tangent, 3-D, {} ball viscous to its centre, ln_viscosity] the worst "
        "{:.3e} from solve's difference ({})\n",
        body, agreement.worst, agreement.where);
    RecordProperty(body + "_gap", std::format("{:.3e}", agreement.worst));
    EXPECT_LT(agreement.worst, 1e-5) << body << " " << agreement.where;

    static_cast<void>(app::RunKernel(config("k", "ball.deck", 1.0, "", true)));
    static_cast<void>(
        app::RunTangent(config("part", "ball.deck", 1.0, "d_part.nc")));
    const auto got =
        ChangesIn(dir_ / "out" / (body + "_part_tangent_measurements.txt"))
            .change.at("site");
    const auto predicted =
        kUniform * MetaOf(dir_ / "out" / (body + "_k_kernels") /
                              "site_viscosity_kernels.nc",
                          "uniform_log_viscosity_pairing");
    std::cout << std::format(
        "[tangent against kernels, 3-D, {} ball viscous to its centre] change "
        "{:.6e}; {:.3e} from the kernels' uniform pairing\n",
        body, got, Gap(got, predicted));
    RecordProperty(body + "_pairing_gap", std::format("{:.3e}", Gap(got, predicted)));
    EXPECT_NE(got, 0.0) << body;
    EXPECT_LT(Gap(got, predicted), 1e-8) << body;
  }
}

TEST_F(TangentTest, In3DAChangeOfTheModuliAtTheCentreIsAChangeLikeAnyOther) {
  // A 3-D run on prem8 under a lateral viscosity alone carries no memory
  // nodes in its elastic inner core, and the element holding the centre is
  // an element like any other there: a change of the moduli about the
  // centre is refused at the member, naming the depth of that element's
  // first memory point (below 6330 km), and not dropped.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_centre.nc",
                   ModelOf(data + "/prem8.elastic.deck", app::ModelFormat::Elastic),
                   {{"mu", [](double r, double, double) { return r < 30e3 ? 1e9 : 0.0; }}},
                   true);
  const auto config = Write(
      "centre.json",
      Config({std::format(R"("model": {{"deck": "{}/prem8.elastic.deck",
    "format": "elastic", "viscosity": {{"file": "{}/viscosity_log.nc"}}}},
  "discretisation": {{"lmax": 8}})",
                          data, data),
              R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 2})",
              R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
              R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
              R"("rotation": "full")", Outputs("centre"),
              R"("perturbation": {"model": {"moduli": {"file": "d_centre.nc"}}})"}));
  auto message = std::string{};
  try {
    static_cast<void>(app::RunTangent(config));
    ADD_FAILURE() << "a change of the moduli at the centre was taken";
  } catch (const pm::Error& error) {
    message = error.what();
  }
  EXPECT_NE(message.find("perturbation.model.moduli.file"), std::string::npos) << message;
  EXPECT_NE(message.find("in an element where the 3-D run carries no memory nodes"),
            std::string::npos)
      << message;
  const auto at = message.find(" km depth");
  ASSERT_NE(at, std::string::npos) << message;
  const auto from = message.rfind(' ', at - 1);
  const auto depth = std::stod(message.substr(from + 1, at - from - 1));
  std::cout << std::format("[a change of the moduli at the centre] refused at {:.6g} km depth\n",
                           depth);
  EXPECT_GT(depth, 6330.0) << message;
}

TEST_F(TangentTest, TheContinuousMethodIsTheDiscreteButAlongTheViscosity) {
  // `tangent.method`, as `kernels.method`. Along the forcing and the
  // moduli the continuous run is the discrete one, under both schemes, to
  // rounding. Along the viscosity, under the exponential trapezoid, it is
  // the continuous equations' run and converges to the discrete one at the
  // scheme's second order: their gap falls by about four as the steps
  // halve (250, 125 and 62.5 yr against the ball's relaxation time of 190
  // yr). Under Sdirk2 it is refused (RefusesWhatItCannotDo). The
  // measurements are the `kernels` block's where the `tangent` block
  // lists none, and its own where it does.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity", [](double r, double, double) {
                       return 0.3 + 0.7 * std::pow(r / 6371000.0, 2);
                     }}},
                   true);
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu", [](double, double, double) { return 0.1 * kBallMu; }},
                    {"kappa",
                     [](double, double, double) { return 0.1 * kBallKappa; }}},
                   true);
  constexpr auto kLoad = R"("forcing": {"load": [{"cap": {"latitude": 20,
      "longitude": 120, "width_deg": 20, "amplitude_kg_m2": 3.0e4},
      "water_budget": true, "moves_solid_surface": false}]})";
  constexpr auto kModuli = R"("model": {"moduli": {"file": "d_moduli.nc"}})";
  constexpr auto kViscosity = R"("model": {"viscosity": {"file": "d_eta.nc"}})";
  constexpr auto kSite = R"({"type": "relative_sea_level_cap", "name": "site",
      "latitude": -30, "longitude": 10, "width_deg": 25})";
  constexpr auto kUplift = R"({"type": "displacement_point", "name": "uplift",
      "latitude": 55, "longitude": 10, "date_kyr_bp": 1})";
  struct Case {
    std::string name, perturbation, method, scheme = "sdirk2";
    int nodes = 5;
    std::string tangent = "", kernels = "";
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({kBall,
                std::format(R"("problem": "quasi_static",
  "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": {}}})",
                            c.nodes),
                R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::format(R"("rotation": "full", "moments": "hydrostatic",
  "solver": {{"rtol": 1e-12, "scheme": "{}"}})",
                            c.scheme),
                Outputs(c.name,
                        std::format(R"(, "method": "{}"{})", c.method,
                                    c.tangent)),
                c.kernels,
                R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto ran = [&](const Case& c) {
    const auto summary = app::RunTangent(config(c));
    const auto continuous = c.method == "continuous";
    EXPECT_EQ(summary.find("(the continuous tangent-linear run along ") !=
                  std::string::npos,
              continuous)
        << summary;
    const auto file = Out(c.name + "_tangent");
    EXPECT_EQ(std::get<std::string>(file.meta.at("tangent_method")), c.method);
    return file;
  };

  // -- the forcing and the moduli: one run by either method
  for (const auto* scheme : {"sdirk2", "exponential_trapezoid"}) {
    for (const auto& [kind, members] :
         {std::pair{"forcing", kLoad}, std::pair{"moduli", kModuli}}) {
      const auto tag = std::format("{}_{}", kind, scheme);
      const auto discrete = ran({.name = tag + "_d", .perturbation = members,
                                 .method = "discrete", .scheme = scheme});
      const auto continuous =
          ran({.name = tag + "_c", .perturbation = members,
               .method = "continuous", .scheme = scheme});
      const auto apart = DepartureOf(discrete, continuous);
      std::cout << std::format(
          "[tangent, continuous against discrete, {}, {}] {:.3e} apart\n",
          kind, scheme, apart);
      RecordProperty(tag + "_gap", std::format("{:.3e}", apart));
      EXPECT_GT(MaxAbs(discrete.variables.at("displacement").values), 0.0);
      EXPECT_LT(apart, 1e-12) << tag;
    }
  }

  // -- the viscosity under the trapezoid: converging with the step
  auto gaps = std::vector<double>{};
  for (const auto nodes : {9, 17, 33}) {
    const auto tag = std::format("eta_{}", nodes);
    const auto discrete =
        ran({.name = tag + "_d", .perturbation = kViscosity,
             .method = "discrete", .scheme = "exponential_trapezoid",
             .nodes = nodes});
    const auto continuous =
        ran({.name = tag + "_c", .perturbation = kViscosity,
             .method = "continuous", .scheme = "exponential_trapezoid",
             .nodes = nodes});
    gaps.push_back(DepartureOf(discrete, continuous));
    std::cout << std::format(
        "[tangent, continuous against discrete, ln viscosity, trapezoid, {} "
        "nodes] {:.3e} apart\n",
        nodes, gaps.back());
    RecordProperty(tag + "_gap", std::format("{:.3e}", gaps.back()));
  }
  EXPECT_GT(gaps.back(), 1e-6);
  for (auto k = std::size_t{1}; k < gaps.size(); ++k) {
    const auto ratio = gaps[k - 1] / gaps[k];
    RecordProperty(std::format("ratio_{}", k), std::format("{:.3f}", ratio));
    EXPECT_GT(ratio, 3.0) << k;
    EXPECT_LT(ratio, 5.5) << k;
  }

  // -- the measurements: the kernels block's list, or the block's own
  const auto kernels = std::format(
      R"("kernels": {{"parameters": ["load"], "measurements": [{}],
    "output": "out/unused_kernels"}})",
      kSite);
  static_cast<void>(ran({.name = "listed_by_kernels", .perturbation = kLoad,
                         .method = "discrete", .kernels = kernels}));
  static_cast<void>(
      ran({.name = "listed_by_both", .perturbation = kLoad,
           .method = "discrete",
           .tangent = std::format(R"(, "measurements": [{}, {}])", kSite,
                                  kUplift),
           .kernels = kernels}));
  static_cast<void>(
      ran({.name = "listed_by_tangent", .perturbation = kLoad,
           .method = "discrete",
           .tangent = std::format(R"(, "measurements": [{}])", kSite)}));
  const auto table = [&](const std::string& name) {
    return ChangesIn(dir_ / "out" / (name + "_tangent_measurements.txt"));
  };
  const auto byKernels = table("listed_by_kernels");
  const auto byBoth = table("listed_by_both");
  const auto byTangent = table("listed_by_tangent");
  EXPECT_EQ(byKernels.change.size(), 1u);
  EXPECT_EQ(byBoth.change.size(), 2u);
  EXPECT_EQ(byTangent.change.size(), 1u);
  EXPECT_NE(byKernels.change.at("site"), 0.0);
  EXPECT_EQ(byBoth.change.at("site"), byKernels.change.at("site"));
  EXPECT_EQ(byTangent.change.at("site"), byKernels.change.at("site"));
  EXPECT_FALSE(fs::exists(dir_ / "out" / "forcing_sdirk2_d_tangent_measurements.txt"));
}

TEST_F(TangentTest, In3DTheContinuousMethodIsTheDiscreteButAlongTheViscosity) {
  // `tangent.method` on a 3-D run, the toy history on the ball under a
  // lateral viscosity and lateral moduli, frozen shorelines, as the 1-D
  // run's above: along the ice and the moduli (lateral changes of each)
  // the continuous run is the discrete one under both schemes, to
  // rounding; along a lateral change of ln viscosity, under the
  // exponential trapezoid, it converges to the discrete one at the
  // scheme's second order as the steps halve (250, 125 and 62.5 yr).
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteToyField(dir_ / "change.nc", "ice_thickness_change",
                [](std::size_t k, double theta, double phi) {
                  return (10.0 + 5.0 * static_cast<double>(k)) *
                         ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
                });
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity",
                     [](double r, double theta, double phi) {
                       return 0.5 + std::pow(r / 6371000.0, 2) *
                                        std::sin(theta) * std::cos(phi);
                     }}},
                   false, SamplingGridOfFiles());
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu",
                     [](double, double theta, double phi) {
                       return 0.2 * kBallMu * std::sin(theta) * std::sin(phi);
                     }},
                    {"kappa",
                     [](double, double theta, double) {
                       return 0.1 * kBallKappa * std::cos(theta);
                     }}},
                   false, SamplingGridOfFiles());
  WriteLateralViscosity(dir_ / "eta.nc", model);
  WriteLateralModuli(dir_ / "moduli.nc", model);
  struct Case {
    std::string name, perturbation, method, scheme;
    int nodes = 5;
  };
  const auto ran = [&](const Case& c) {
    const auto summary = app::RunTangent(Write(
        c.name + ".json",
        Config({R"("model": {"deck": "ball.deck",
    "viscosity": {"file": "eta.nc"}, "moduli": {"file": "moduli.nc"}},
  "discretisation": {"lmax": 8})",
                std::format(R"("problem": "quasi_static",
  "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": {}}})",
                            c.nodes),
                R"("forcing": {"ice_change": [
    {"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                std::format(R"({}, "scheme": "{}"}})", kTight3D, c.scheme),
                Outputs(c.name, std::format(R"(, "method": "{}")", c.method)),
                R"("perturbation": {)" + c.perturbation + "}"})));
    EXPECT_TRUE(summary.starts_with("3-D (lateral viscosity, lateral moduli)"))
        << summary;
    const auto file = Out(c.name + "_tangent");
    EXPECT_EQ(std::get<std::string>(file.meta.at("tangent_method")), c.method);
    return file;
  };
  constexpr auto kIce = R"("forcing": {"ice_change": [
      {"change": {"file": "change.nc"}}]})";
  constexpr auto kModuli = R"("model": {"moduli": {"file": "d_moduli.nc"}})";
  constexpr auto kViscosity = R"("model": {"viscosity": {"file": "d_eta.nc"}})";

  // -- the ice and the moduli: one run by either method
  for (const auto* scheme : {"sdirk2", "exponential_trapezoid"}) {
    for (const auto& [kind, members] :
         {std::pair{"ice", kIce}, std::pair{"moduli", kModuli}}) {
      const auto tag = std::format("{}_{}", kind, scheme);
      const auto discrete = ran({.name = tag + "_d", .perturbation = members,
                                 .method = "discrete", .scheme = scheme});
      const auto continuous =
          ran({.name = tag + "_c", .perturbation = members,
               .method = "continuous", .scheme = scheme});
      const auto apart = DepartureOf(discrete, continuous);
      std::cout << std::format(
          "[tangent, 3-D, continuous against discrete, {}, {}] {:.3e} apart\n",
          kind, scheme, apart);
      RecordProperty(tag + "_gap", std::format("{:.3e}", apart));
      EXPECT_GT(MaxAbs(discrete.variables.at("displacement").values), 0.0);
      EXPECT_LT(apart, 1e-12) << tag;
    }
  }

  // -- the viscosity under the trapezoid: converging with the step
  auto gaps = std::vector<double>{};
  for (const auto nodes : {9, 17, 33}) {
    const auto tag = std::format("eta_{}", nodes);
    const auto discrete =
        ran({.name = tag + "_d", .perturbation = kViscosity,
             .method = "discrete", .scheme = "exponential_trapezoid",
             .nodes = nodes});
    const auto continuous =
        ran({.name = tag + "_c", .perturbation = kViscosity,
             .method = "continuous", .scheme = "exponential_trapezoid",
             .nodes = nodes});
    gaps.push_back(DepartureOf(discrete, continuous));
    std::cout << std::format(
        "[tangent, 3-D, continuous against discrete, ln viscosity, "
        "trapezoid, {} nodes] {:.3e} apart\n",
        nodes, gaps.back());
    RecordProperty(tag + "_gap", std::format("{:.3e}", gaps.back()));
  }
  EXPECT_GT(gaps.back(), 1e-6);
  for (auto k = std::size_t{1}; k < gaps.size(); ++k) {
    const auto ratio = gaps[k - 1] / gaps[k];
    RecordProperty(std::format("ratio_{}", k), std::format("{:.3f}", ratio));
    EXPECT_GT(ratio, 3.0) << k;
    EXPECT_LT(ratio, 5.5) << k;
  }
}

}  // namespace
