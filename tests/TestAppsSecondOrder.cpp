// `sl3d kernel` with `kernels.second_order`: the second-order kernels, each
// kernel's derivative along the config's `perturbation` block (the action
// of the measurement's Hessian on the direction). Each second-order file is
// the central difference of the first-order files of two `sl3d kernel` runs
// whose configs have the perturbation's members added to their inputs,
// plus and minus, per kind of direction (the forcing's lists, ln viscosity
// of every branch and of one, the moduli, the state's sea level), 1-D and
// 3-D, static and through time, with an ocean or without one, frozen or
// migrating shorelines, a calibrated run's; the Hessian is symmetric
// (along a then paired with b, against along b then paired with a); the
// files are the library's own call's, and a trajectory store's run's those
// of the run in memory; and what is not built is refused, by name.

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
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <planetmodel/deck.hpp>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/json.hpp>
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
#include "Recorded.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace {

namespace fs = std::filesystem;
namespace pm = planetmodel;
using namespace sl3dpp;
using sl3dpp::test::WriteToyHistory;

constexpr auto kLmax = 8;

class SecondOrderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("sl3dpp_test_second_order_" +
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
  /// The directory a kernel run of `name` writes into (Kernels(name)).
  fs::path KernelsOf(const std::string& name) const {
    return dir_ / "out" / (name + "_kernels");
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
/// with `radial`, stored as functions of the radius alone. A field of a
/// direction in the viscosity is written with its spec (LogViscositySpec).
using Field3 = std::function<double(double, double, double)>;
void WriteModelFields(const fs::path& path, const EarthModel& model,
                      const std::map<std::string, Field3>& fields, bool radial,
                      const pm::AngularGrid& grid = pm::GaussLegendre(8)) {
  const auto& boundaries = model.Boundaries();
  auto file = pm::ModelFile(
      pm::Skeleton(std::vector<double>(boundaries.begin(), boundaries.end())));
  file.SetName("sl3dpp second-order tests");
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

/// The shape of the toy history's polar ice cap, one at the pole.
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

double MaxAbs(std::span<const double> values) {
  auto out = 0.0;
  for (const auto v : values) out = std::max(out, std::abs(v));
  return out;
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

/// The rows of numbers of a text table, its `#` lines left out, and the
/// names of its columns (its last `#` line before the rows).
struct Table {
  std::vector<std::string> columns;
  std::vector<std::vector<double>> rows;
};
Table ReadTable(const fs::path& path) {
  auto in = std::ifstream(path);
  EXPECT_TRUE(in) << path;
  auto out = Table{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) continue;
    if (line.starts_with("#")) {
      if (!out.rows.empty()) continue;
      auto words = std::istringstream(line.substr(1));
      out.columns.clear();
      for (std::string word; words >> word;) out.columns.push_back(word);
      continue;
    }
    auto words = std::istringstream(line);
    auto row = std::vector<double>{};
    for (double v; words >> v;) row.push_back(v);
    out.rows.push_back(std::move(row));
  }
  return out;
}

/// A table's kernel columns (named `K_...` or `dJ_...`), laid end to end.
std::vector<double> KernelColumns(const Table& table) {
  auto out = std::vector<double>{};
  for (const auto& row : table.rows) {
    for (auto c = std::size_t{0}; c < table.columns.size() && c < row.size();
         ++c) {
      const auto& name = table.columns[c];
      if (name.starts_with("K_") || name.starts_with("dJ_")) {
        out.push_back(row[c]);
      }
    }
  }
  return out;
}

/// How a second-order kernel holds against the central difference of the
/// first-order kernels of two runs a step `eps` apart along its direction,
/// (plus - minus) / (2 eps): `gap`, the largest departure over the
/// difference's largest magnitude, and `size`, that magnitude over the
/// first-order kernel's. A difference at the rounding of the first-order
/// kernels (below 1e-10 of their size over eps) is a derivative that
/// vanishes, and the second-order kernel must too, to 1e-9 of that size
/// (`vanishes`).
struct Held {
  double gap = 0.0;
  double size = 0.0;
  bool vanishes = false;
};
Held HeldAgainst(std::span<const double> second, std::span<const double> plus,
                 std::span<const double> minus, double eps,
                 const std::string& what) {
  EXPECT_EQ(second.size(), plus.size()) << what;
  EXPECT_EQ(plus.size(), minus.size()) << what;
  auto out = Held{};
  if (second.size() != plus.size() || plus.size() != minus.size()) {
    out.gap = 1.0;
    return out;
  }
  auto scale = 0.0, diff = 0.0, first = 0.0;
  for (auto i = std::size_t{0}; i < second.size(); ++i) {
    const auto want = (plus[i] - minus[i]) / (2.0 * eps);
    scale = std::max(scale, std::abs(want));
    diff = std::max(diff, std::abs(second[i] - want));
    first = std::max(first, std::abs(plus[i]));
  }
  if (scale <= 1e-10 * first / eps) {
    out.vanishes = true;
    EXPECT_LE(MaxAbs(second), 1e-9 * std::max(first / eps, 1e-300)) << what;
    return out;
  }
  out.gap = diff / scale;
  out.size = first > 0.0 ? scale / first : 0.0;
  return out;
}

/// The second-order files of a kernel run against the central difference
/// of two first-order runs' files: every `<m>_<kind>_second_order.*` in
/// `second` against `<m>_<kind>.*` in `plus` and `minus`, field by field
/// (a surface file's fields, a table's kernel columns, a model file's
/// `K_` fields in each part and its `uniform_log_viscosity_pairing`).
/// Returns the worst departure and where, and the fields that did not
/// vanish.
struct Agreement {
  double worst = 0.0;
  std::string where;
  int moved = 0;
  int files = 0;
};
Agreement FilesAgainst(const fs::path& second, const fs::path& plus,
                       const fs::path& minus, double eps) {
  auto out = Agreement{};
  const auto hold = [&](const std::string& what, std::span<const double> got,
                        std::span<const double> up,
                        std::span<const double> down) {
    const auto held = HeldAgainst(got, up, down, eps, what);
    if (held.vanishes) return;
    ++out.moved;
    if (held.gap > out.worst) {
      out.worst = held.gap;
      out.where = what;
    }
  };
  for (const auto& entry : fs::directory_iterator(second)) {
    const auto name = entry.path().filename().string();
    const auto at = name.find("_second_order.");
    if (at == std::string::npos) continue;
    ++out.files;
    const auto first =
        name.substr(0, at) + name.substr(at + std::string{"_second_order"}.size());
    EXPECT_TRUE(fs::exists(plus / first)) << first;
    if (!fs::exists(plus / first)) continue;
    if (entry.path().extension() == ".txt") {
      const auto got = ReadTable(entry.path());
      auto up = ReadTable(plus / first);
      auto down = ReadTable(minus / first);
      // a moment table lists the run's source points: a run with a source
      // appended has one more, whose rows the second-order file has not
      if (const auto at = std::ranges::find(got.columns, "point");
          at != got.columns.end()) {
        const auto column = static_cast<std::size_t>(at - got.columns.begin());
        auto points = 0.0;
        for (const auto& row : got.rows) points = std::max(points, row[column] + 1);
        for (auto* table : {&up, &down}) {
          std::erase_if(table->rows, [&](const std::vector<double>& row) {
            return row[column] >= points;
          });
        }
      }
      hold(name, KernelColumns(got), KernelColumns(up), KernelColumns(down));
      continue;
    }
    if (name.find("_ice_kernels") != std::string::npos ||
        name.find("_load_kernels") != std::string::npos) {
      const auto got = pm::ReadSurfaceFile(entry.path());
      const auto up = pm::ReadSurfaceFile(plus / first);
      const auto down = pm::ReadSurfaceFile(minus / first);
      EXPECT_TRUE(got.meta.contains("second_order")) << name;
      for (const auto& [field, variable] : got.variables) {
        EXPECT_TRUE(up.variables.contains(field)) << name << " " << field;
        if (!up.variables.contains(field)) continue;
        hold(name + ":" + field, variable.values, up.variables.at(field).values,
             down.variables.at(field).values);
      }
      continue;
    }
    // a model file of field kernels
    const auto got = pm::ReadModelFile(entry.path());
    const auto up = pm::ReadModelFile(plus / first);
    const auto down = pm::ReadModelFile(minus / first);
    EXPECT_EQ(got.Parts().size(), up.Parts().size()) << name;
    for (auto p = std::size_t{0}; p < got.Parts().size(); ++p) {
      for (const auto& [field, variable] : got.Parts()[p].variables) {
        if (!field.starts_with("K_")) continue;
        hold(std::format("{}:{}[{}]", name, field, p), variable.values,
             up.Parts()[p].variables.at(field).values,
             down.Parts()[p].variables.at(field).values);
      }
    }
    if (got.Meta().contains("meta_uniform_log_viscosity_pairing") ||
        got.Meta().contains("uniform_log_viscosity_pairing")) {
      const auto key = got.Meta().contains("uniform_log_viscosity_pairing")
                           ? std::string{"uniform_log_viscosity_pairing"}
                           : std::string{"meta_uniform_log_viscosity_pairing"};
      hold(name + ":" + key,
           std::get<std::vector<double>>(got.Meta().at(key)),
           std::get<std::vector<double>>(up.Meta().at(key)),
           std::get<std::vector<double>>(down.Meta().at(key)));
    }
  }
  return out;
}

constexpr auto kDated = R"("problem": "quasi_static",
  "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5})";
constexpr auto kTight = R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12})";
constexpr auto kTight3D = R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12, "step_rtol": 1e-12, "step_max_iterations": 200)";
constexpr auto kOcean = R"("ocean": {"at": "start",
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})";
constexpr auto kMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10, "date_kyr_bp": 1}])";

/// The static problem's measurements, at its one instant.
constexpr auto kStaticMeasurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10}])";

/// A `kernels` block writing under out/<name>_kernels: its parameters, the
/// two measurements (`measurements`, kMeasurements by default),
/// `second_order` where asked and further members.
std::string KernelsBlock(const std::string& name, const std::string& parameters,
                         bool second, const std::string& more = "",
                         const std::string& measurements = kMeasurements) {
  return std::format(R"("kernels": {{"parameters": [{}], {},
    "output": "out/{}_kernels"{}{}}})",
                     parameters, measurements, name,
                     second ? R"(, "second_order": true)" : "", more);
}

/// Prints and records a kind's agreement, and holds it to `tolerance`.
void Report(const std::string& label, const Agreement& agreement,
            double tolerance) {
  std::cout << std::format(
      "[second order, {}] {} files, {} moved fields; the worst {:.3e} from "
      "the kernels' difference ({})\n",
      label, agreement.files, agreement.moved, agreement.worst,
      agreement.where);
  ::testing::Test::RecordProperty(label, std::format("{:.3e}", agreement.worst));
  EXPECT_GT(agreement.files, 0) << label;
  EXPECT_GT(agreement.moved, 0) << label;
  EXPECT_LT(agreement.worst, tolerance) << label << " " << agreement.where;
}

}  // namespace

namespace {

/// The largest departure of the numbers of two runs' second-order files
/// (every `*_second_order.*` of `a` against `b`'s of its name), each over
/// the largest magnitude of `b`'s; with `first`, of their first-order
/// kernel files instead (every other `*_kernel*` file of `b`).
double SecondOrderDeparture(const fs::path& a, const fs::path& b,
                            bool first = false) {
  auto worst = 0.0;
  auto files = 0;
  const auto against = [&](std::span<const double> x,
                           std::span<const double> y) {
    EXPECT_EQ(x.size(), y.size());
    if (x.size() != y.size()) return;
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      diff = std::max(diff, std::abs(x[i] - y[i]));
    }
    const auto scale = MaxAbs(y);
    worst = std::max(worst, scale > 0.0 ? diff / scale : diff);
  };
  for (const auto& entry : fs::directory_iterator(first ? b : a)) {
    const auto name = entry.path().filename().string();
    const auto second = name.find("_second_order.") != std::string::npos;
    if (name.find("_kernel") == std::string::npos || second == first) continue;
    ++files;
    if (entry.path().extension() == ".txt") {
      against(KernelColumns(ReadTable(a / name)),
              KernelColumns(ReadTable(b / name)));
    } else if (name.find("_ice_kernels") != std::string::npos ||
               name.find("_load_kernels") != std::string::npos) {
      const auto x = pm::ReadSurfaceFile(a / name);
      const auto y = pm::ReadSurfaceFile(b / name);
      for (const auto& [field, variable] : y.variables) {
        against(x.variables.at(field).values, variable.values);
      }
    } else {
      const auto x = pm::ReadModelFile(a / name);
      const auto y = pm::ReadModelFile(b / name);
      for (auto p = std::size_t{0}; p < y.Parts().size(); ++p) {
        for (const auto& [field, variable] : y.Parts()[p].variables) {
          if (!field.starts_with("K_")) continue;
          against(x.Parts()[p].variables.at(field).values, variable.values);
        }
      }
    }
  }
  EXPECT_GT(files, 0) << a;
  return worst;
}

}  // namespace

TEST_F(SecondOrderTest, TheForcingsDirectionsAreKernelsDifferences) {
  // A radial run through time on the ball under the toy history, frozen
  // shorelines, Sdirk2 and the discrete method (the defaults). Along each
  // member of the forcing (an ice change on a time axis, a load in the water
  // budget, a tide, a seismic source) the second-order kernels are the
  // central difference of the first-order kernels of two runs with the
  // member's term appended to their lists, plus and minus. The problem is
  // linear in its forcing, so the kernels in the ice, the loads and the tide
  // do not move, and the viscosity's and the moduli's move linearly: the
  // difference is exact, and what is left is rounding (measured 3e-12 to
  // 1.3e-11).
  static_cast<void>(Write("ball.deck", BallDeck()));
  WriteToyHistory(dir_ / "history.nc");
  for (const auto& [name, scale] :
       {std::pair{"change.nc", 1.0}, std::pair{"change_minus.nc", -1.0}}) {
    WriteToyField(dir_ / name, "ice_thickness_change",
                  [scale](std::size_t k, double theta, double phi) {
                    return scale * (10.0 + 5.0 * static_cast<double>(k)) *
                           ToyCap(theta) * (1.0 + 0.5 * std::cos(phi));
                  });
  }
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}, "water_budget": true,
             "moves_solid_surface": false}})",
        3.0e4 * s);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 3, "order": 2, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}}})",
        0.6 * s, 0.2 * s);
  };
  const auto source = [](double s) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 300,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}}})",
        1e22 * s, -2e22 * s, 1e22 * s, 3e22 * s, -1e22 * s, 2e22 * s);
  };
  constexpr auto kParameters =
      R"("ice", "load", "tide", "viscosity", "moduli")";
  struct Kind {
    std::string name, list;
    std::function<std::string(double)> term;
    double tolerance;
  };
  const auto kinds = std::vector<Kind>{{"ice", "ice_change", ice, 5e-11},
                                       {"load", "load", load, 5e-11},
                                       {"tide", "tide", tide, 5e-11},
                                       {"seismic", "seismic", source, 5e-11}};
  for (const auto& kind : kinds) {
    const auto config = [&](const std::string& name, double s, bool second) {
      auto forcing = std::string{R"("ice_change": [{"thickness": {"file": "history.nc"}})"};
      if (kind.list == "ice_change" && s != 0.0) forcing += ", " + kind.term(s);
      forcing += "]";
      if (kind.list != "ice_change" && s != 0.0) {
        forcing += std::format(R"(, "{}": [{}])", kind.list, kind.term(s));
      }
      return Write(
          name + ".json",
          Config({R"("model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8})",
                  kDated, "\"forcing\": {" + forcing + "}", kOcean, kTight,
                  KernelsBlock(name, kParameters, second),
                  second ? std::format(R"("perturbation": {{"forcing": {{"{}": [{}]}}}})",
                                       kind.list, kind.term(1.0))
                         : std::string{}}));
    };
    const auto summary =
        app::RunKernel(config("s_" + kind.name, 0.0, true));
    EXPECT_NE(summary.find("; and their second-order kernels along the "
                           "perturbation's " +
                           kind.list + ")"),
              std::string::npos)
        << summary;
    static_cast<void>(app::RunKernel(config("p_" + kind.name, 1.0, false)));
    static_cast<void>(app::RunKernel(config("m_" + kind.name, -1.0, false)));
    const auto agreement =
        FilesAgainst(KernelsOf("s_" + kind.name), KernelsOf("p_" + kind.name),
                     KernelsOf("m_" + kind.name), 1.0);
    Report("1-D, " + kind.name, agreement, kind.tolerance);
  }
}

TEST_F(SecondOrderTest, TheEarthsDirectionsAreKernelsDifferences) {
  // The radial run of the forcing test along the Earth's parameters, each
  // against the central difference of the first-order kernels of two runs
  // on the Earth moved by plus and minus a step of the direction:
  //  - ln viscosity: a radial model file of the change, a function of the
  //    radius, against runs on viscosity files of the viscosity times
  //    exp(+-1e-3 of it), the run itself on the file of the viscosity;
  //  - the moduli: a radial model file of the changes of kappa and mu, a
  //    tenth of each, against runs on decks of the moduli scaled by
  //    1 +- 1e-4 (the viscosity scaled with the shear modulus, the
  //    relaxation times held, as the direction holds them).
  // Every kernel moves. What is left is the differences' own error: their
  // remainder, second order in the step (measured 1.4e-5, 1.2e-6 and
  // 1.4e-7 along ln viscosity at steps of 1e-2, 3e-3 and 1e-3), and the
  // solves' rounding over the step, which is as large at 1e-3 (1.4e-7 at
  // 3e-4, 7.6e-7 at 1e-4). The first-order files written beside them, from
  // the second-order sweep's kernels, are the first-order sweep's to
  // rounding.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kStep = 1e-3;
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
  for (const auto& [name, scale] : {std::pair{"stiff.deck", kStep},
                                    std::pair{"soft.deck", -kStep}}) {
    static_cast<void>(Write(name, BallDeck(kTenth * scale,
                                           1e21 * (1.0 + kTenth * scale))));
  }
  constexpr auto kParameters =
      R"("ice", "load", "tide", "viscosity", "moduli")";
  const auto config = [&](const std::string& name, const std::string& model_,
                          const std::string& perturbation = "") {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{{}}}, "discretisation": {{"lmax": 8}})",
                            model_),
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}],
    "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 0.5}]})",
                kOcean, kTight,
                KernelsBlock(name, kParameters, !perturbation.empty()),
                perturbation.empty() ? std::string{}
                                     : R"("perturbation": )" + perturbation}));
  };
  const auto file = [](const std::string& viscosity) {
    return std::format(R"("deck": "ball.deck", "viscosity": {{"file": "{}"}})",
                       viscosity);
  };
  struct Kind {
    std::string name, perturbation, base, plus, minus;
    double tolerance;
  };
  const auto kinds = std::vector<Kind>{
      {"viscosity", R"({"model": {"viscosity": {"file": "d_eta.nc"}}})",
       file("eta.nc"), file("eta_plus.nc"), file("eta_minus.nc"), 1e-6},
      {"moduli", R"({"model": {"moduli": {"file": "d_moduli.nc"}}})",
       R"("deck": "ball.deck")", R"("deck": "stiff.deck")",
       R"("deck": "soft.deck")", 1e-6}};
  for (const auto& kind : kinds) {
    const auto summary =
        app::RunKernel(config("s_" + kind.name, kind.base, kind.perturbation));
    EXPECT_NE(summary.find("; and their second-order kernels along the "
                           "perturbation's " +
                           kind.name + ")"),
              std::string::npos)
        << summary;
    static_cast<void>(app::RunKernel(config("p_" + kind.name, kind.plus)));
    static_cast<void>(app::RunKernel(config("m_" + kind.name, kind.minus)));
    const auto agreement =
        FilesAgainst(KernelsOf("s_" + kind.name), KernelsOf("p_" + kind.name),
                     KernelsOf("m_" + kind.name), kStep);
    Report("1-D, " + kind.name, agreement, kind.tolerance);
    // the first-order files beside them, from the second-order sweep, are
    // the first-order sweep's
    static_cast<void>(app::RunKernel(config("f_" + kind.name, kind.base)));
    const auto first = SecondOrderDeparture(KernelsOf("s_" + kind.name),
                                            KernelsOf("f_" + kind.name), true);
    std::cout << std::format(
        "[second order, 1-D, {}] the first-order files {:.3e} from the "
        "first-order sweep's\n",
        kind.name, first);
    EXPECT_LT(first, 1e-11) << kind.name;
  }
}

namespace {

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

/// The lidded ball as a burgers deck: a steady viscosity of 1e21 Pa s and
/// a Kelvin element of 1e20 Pa s and half the shear modulus, both
/// viscosities scaled.
void WriteBurgersBall(const fs::path& path, double scale = 1.0) {
  pm::WriteDeck(path,
                LiddedBall({{"viscosity", scale * 1e21},
                            {"kelvin_viscosity", scale * 1e20},
                            {"kelvin_modulus_fraction", 0.5}},
                           pm::BurgersHeader("lidded ball")),
                pm::BURGERS());
}

/// The lidded ball as a prony deck of the branches given, each a strength
/// and a time (s).
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

}  // namespace

TEST_F(SecondOrderTest, OnABurgersOrPronyBodyTheViscosityIsItsBranches) {
  // The radial run of the forcing test on the lidded ball as a deck of two
  // branches, the second-order kernels against the central difference of
  // the first-order kernels of runs on decks a step 1e-3 in ln tau apart:
  //  - `ln_viscosity` on the burgers deck, every branch's relaxation time
  //    moved alike: decks with both viscosities scaled by 1 +- 1e-3;
  //  - `ln_viscosity_branch_1` on a prony deck (strengths 0.7 and 0.3,
  //    times 4e8 and 2e10 s), the faster branch alone: prony decks with its
  //    time scaled (the decks' eight digits hold it).
  // The kernel files give a column per branch (`K_branch_0`, `K_branch_1`),
  // each held to its difference. The differences' remainder is second order
  // in the step (6.9e-6 and 7.7e-7 at 3e-3 and 1e-3; at 3e-4 the solves'
  // rounding, 4e-7, is as large).
  constexpr auto kStep = 1e-3;
  const auto half = 0.5 * (std::log(1.0 + kStep) - std::log(1.0 - kStep));
  WriteToyHistory(dir_ / "history.nc");
  WriteBurgersBall(dir_ / "burgers.deck");
  WriteBurgersBall(dir_ / "burgers_plus.deck", 1.0 + kStep);
  WriteBurgersBall(dir_ / "burgers_minus.deck", 1.0 - kStep);
  const auto prony = [](double scale) {
    return std::vector<std::array<double, 2>>{{0.7, 4e8}, {0.3, 2e10 * scale}};
  };
  WritePronyBall(dir_ / "prony.deck", prony(1.0));
  WritePronyBall(dir_ / "prony_plus.deck", prony(1.0 + kStep));
  WritePronyBall(dir_ / "prony_minus.deck", prony(1.0 - kStep));
  const auto model = ModelOf(dir_ / "burgers.deck", app::ModelFormat::Burgers);
  const auto one = [](double, double, double) { return 1.0; };
  WriteModelFields(dir_ / "d_all.nc", model, {{"ln_viscosity", one}}, true);
  WriteModelFields(dir_ / "d_one.nc", model, {{"ln_viscosity_branch_1", one}},
                   true);
  const auto config = [&](const std::string& name, const std::string& deck,
                          const std::string& format,
                          const std::string& direction = "") {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{"deck": "{}", "format": "{}"}},
  "discretisation": {{"lmax": 8}})",
                            deck, format),
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                kOcean, kTight,
                KernelsBlock(name, R"("ice", "viscosity", "moduli")",
                             !direction.empty()),
                direction.empty()
                    ? std::string{}
                    : std::format(R"("perturbation": {{"model": {{"viscosity":
    {{"file": "{}"}}}}}})",
                                  direction)}));
  };
  for (const auto& [name, format, direction] :
       {std::array<std::string, 3>{"burgers_all", "burgers", "d_all.nc"},
        std::array<std::string, 3>{"prony_one", "prony", "d_one.nc"}}) {
    static_cast<void>(
        app::RunKernel(config("s_" + name, format + ".deck", format, direction)));
    static_cast<void>(
        app::RunKernel(config("p_" + name, format + "_plus.deck", format)));
    static_cast<void>(
        app::RunKernel(config("m_" + name, format + "_minus.deck", format)));
    const auto table =
        ReadTable(KernelsOf("s_" + name) / "site_viscosity_kernel_second_order.txt");
    EXPECT_EQ(table.columns,
              (std::vector<std::string>{"r_m", "K_branch_0", "K_branch_1", "W_m3"}));
    const auto agreement = FilesAgainst(
        KernelsOf("s_" + name), KernelsOf("p_" + name), KernelsOf("m_" + name),
        half);
    Report("1-D, " + name, agreement, 2e-6);
  }
}

TEST_F(SecondOrderTest, UnderMigratingShorelinesTheStatesSeaLevel) {
  // The radial run under migrating shorelines, its state's topography a
  // file of its own with the equator half a metre above the sea, which the
  // melt floods: along a change of that topography (the state's sea level,
  // `perturbation.ocean`), the second-order kernels against the central
  // difference of the first-order kernels of runs on the topography moved
  // by plus and minus a tenth of the change, no cell's flotation flipping
  // (the runs' ocean functions are the run's, node for node). The initial
  // sea level moves the run through the shorelines it places, the
  // viscosity's and the moduli's kernels with it; the kernels in the ice
  // and the loads, about the run's own shorelines, do not move. With the
  // shorelines held the run is affine in the initial sea level, and the
  // difference exact: what is left is rounding (8.6e-10; 7e-9 at a
  // hundredth).
  static_cast<void>(Write("ball.deck", BallDeck()));
  WriteToyHistory(dir_ / "history.nc");
  constexpr auto kStep = 1e-1;
  for (const auto& [name, scale] :
       {std::pair{"topography.nc", 0.0}, std::pair{"topography_plus.nc", kStep},
        std::pair{"topography_minus.nc", -kStep}}) {
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
  const auto config = [&](const std::string& name,
                          const std::string& topography, bool second) {
    return Write(
        name + ".json",
        Config({R"("model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8})",
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                std::format(R"("ocean": {{"at": "start", "shorelines": "migrating",
    "topography": {{"file": "{}"}}, "ice": {{"file": "history.nc"}}}})",
                            topography),
                kTight,
                std::format(R"("output": {{"surface": "out/{}.nc"}})", name),
                KernelsBlock(name, R"("ice", "load", "viscosity", "moduli")",
                             second),
                second ? R"("perturbation": {"ocean": {"topography":
                   {"file": "d_topography.nc"}}})"
                       : ""}));
  };
  const auto summary =
      app::RunKernel(config("s", "topography.nc", true));
  EXPECT_NE(summary.find("along the perturbation's topography)"),
            std::string::npos)
      << summary;
  // no cell flips between the runs
  auto oceans = std::vector<std::vector<double>>{};
  for (const auto& [name, topography] :
       {std::pair{"s", "topography.nc"}, std::pair{"p", "topography_plus.nc"},
        std::pair{"m", "topography_minus.nc"}}) {
    const auto path = config(name, topography, false);
    static_cast<void>(app::RunSolve(path));
    if (std::string{name} != "s") static_cast<void>(app::RunKernel(path));
    oceans.push_back(pm::ReadSurfaceFile(dir_ / "out" / (std::string{name} + ".nc"))
                         .variables.at("ocean_function")
                         .values);
  }
  EXPECT_EQ(oceans[1], oceans[0]) << "a cell flipped";
  EXPECT_EQ(oceans[2], oceans[0]) << "a cell flipped";
  // the shorelines move along the run: the last node's ocean is not the
  // first's
  const auto size = oceans[0].size() / 5;
  EXPECT_FALSE(std::ranges::equal(std::span(oceans[0]).first(size),
                                  std::span(oceans[0]).last(size)))
      << "no cell flooded along the run";
  const auto agreement =
      FilesAgainst(KernelsOf("s"), KernelsOf("p"), KernelsOf("m"), kStep);
  Report("1-D, migrating, the state's topography", agreement, 5e-9);
}


TEST_F(SecondOrderTest, TheStaticProblem) {
  // The static problem with an ocean (one node, the elastic Earth), under
  // an ice change, a load and a tide, its kernels in them and in the
  // moduli. Under frozen shorelines, along an ice change, a load, a tide
  // and the moduli; under migrating ones, along the state's topography.
  // Each second-order file against the central difference of the
  // first-order files of runs with the member appended, plus and minus
  // (the forcing's and the state's, the problem linear in them, at a step
  // of one; the moduli's on decks a step 1e-3 of a tenth of the moduli
  // apart). Measured: 5e-13 to 2.4e-12 along the forcing, 7e-11 along the
  // topography, 1.2e-7 along the moduli (the difference's remainder). The
  // problem has no step, and the continuous method's second-order kernels
  // are the discrete's, to rounding (3e-13).
  static_cast<void>(Write("ball.deck", BallDeck()));
  const auto model = ModelOf(dir_ / "ball.deck");
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
  constexpr auto kStep = 1e-3;
  constexpr auto kTenth = 0.1;
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu", [](double, double, double) { return kTenth * kBallMu; }},
                    {"kappa",
                     [](double, double, double) { return kTenth * kBallKappa; }}},
                   true);
  static_cast<void>(Write("stiff.deck", BallDeck(kTenth * kStep)));
  static_cast<void>(Write("soft.deck", BallDeck(-kTenth * kStep)));
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}, "water_budget": true,
             "moves_solid_surface": false}})",
        3.0e4 * s);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 3, "order": 2, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}}})",
        0.6 * s, 0.2 * s);
  };
  struct Case {
    std::string name, shorelines = "frozen", deck = "ball.deck",
                      topography = "topography.nc", list = "", term = "",
                      perturbation = "", method = "";
  };
  const auto config = [&](const Case& c) {
    auto lists = std::map<std::string, std::string>{
        {"ice_change",
         R"({"cap_fraction": {"latitude": 85, "longitude": 0,
                              "width_deg": 20, "fraction": -0.3}})"},
        {"load", load(1.0)},
        {"tide", tide(1.0)}};
    if (!c.list.empty()) lists[c.list] += ", " + c.term;
    return Write(
        c.name + ".json",
        Config({std::format(R"("problem": "static",
  "model": {{"deck": "{}"}}, "discretisation": {{"lmax": 8}})",
                            c.deck),
                std::format(R"("forcing": {{"ice_change": [{}], "load": [{}],
                   "tide": [{}]}})",
                            lists["ice_change"], lists["load"], lists["tide"]),
                std::format(R"("ocean": {{"date_kyr_bp": 2, "shorelines": "{}",
    "topography": {{"file": "{}"}}, "ice": {{"file": "history.nc"}}}})",
                            c.shorelines, c.topography),
                kTight,
                KernelsBlock(c.name, R"("ice", "load", "tide", "moduli")",
                             !c.perturbation.empty(), c.method,
                             kStaticMeasurements),
                c.perturbation.empty() ? std::string{}
                                       : R"("perturbation": )" + c.perturbation}));
  };
  struct Kind {
    std::string name;
    Case second, plus, minus;
    double step, tolerance;
  };
  const auto forcing = [&](const std::string& name, const std::string& list,
                           const std::function<std::string(double)>& term) {
    return Kind{name,
                {.name = "s_" + name,
                 .perturbation = std::format(R"({{"forcing": {{"{}": [{}]}}}})",
                                             list, term(1.0))},
                {.name = "p_" + name, .list = list, .term = term(1.0)},
                {.name = "m_" + name, .list = list, .term = term(-1.0)},
                1.0,
                5e-11};
  };
  const auto kinds = std::vector<Kind>{
      forcing("ice", "ice_change", ice), forcing("load", "load", load),
      forcing("tide", "tide", tide),
      {"moduli",
       {.name = "s_moduli",
        .perturbation = R"({"model": {"moduli": {"file": "d_moduli.nc"}}})"},
       {.name = "p_moduli", .deck = "stiff.deck"},
       {.name = "m_moduli", .deck = "soft.deck"},
       kStep,
       1e-6},
      {"topography",
       {.name = "s_topography",
        .shorelines = "migrating",
        .perturbation =
            R"({"ocean": {"topography": {"file": "d_topography.nc"}}})"},
       {.name = "p_topography",
        .shorelines = "migrating",
        .topography = "topography_plus.nc"},
       {.name = "m_topography",
        .shorelines = "migrating",
        .topography = "topography_minus.nc"},
       1.0,
       5e-10}};
  for (const auto& kind : kinds) {
    const auto summary = app::RunKernel(config(kind.second));
    EXPECT_TRUE(summary.starts_with("1-D, elastic, static, ")) << summary;
    static_cast<void>(app::RunKernel(config(kind.plus)));
    static_cast<void>(app::RunKernel(config(kind.minus)));
    const auto agreement =
        FilesAgainst(KernelsOf(kind.second.name), KernelsOf(kind.plus.name),
                     KernelsOf(kind.minus.name), kind.step);
    Report("1-D, static, " + kind.name, agreement, kind.tolerance);
  }
  // the continuous method, about the problem of one node
  auto continuous = kinds[3].second;
  continuous.name = "c_moduli";
  continuous.method = R"(, "method": "continuous")";
  static_cast<void>(app::RunKernel(config(continuous)));
  const auto methods =
      SecondOrderDeparture(KernelsOf("c_moduli"), KernelsOf("s_moduli"));
  std::cout << std::format(
      "[second order, 1-D, static] the continuous method's second-order "
      "kernels {:.3e} from the discrete's\n",
      methods);
  RecordProperty("static_methods", std::format("{:.3e}", methods));
  EXPECT_LT(methods, 1e-10);
}

namespace {

/// The 3-D runs' Earth: the ball with a viscosity that varies with
/// latitude, eta = 1e21 exp(kAnomaly cos(theta) + more), and lateral
/// moduli, mu(r) (1 + kShear cos(theta) + shear) and kappa(r)
/// (1 + kBulk sin(theta) cos(phi) + bulk).
constexpr auto kAnomaly = 0.3;
constexpr auto kShear = 0.1;
constexpr auto kBulk = 0.05;
using Angular = std::function<double(double, double)>;

/// The angular grid a 3-D run samples its lateral structure on at degree
/// `lmax`: a model file written on it is read at its own points.
SurfaceGrid SamplingGrid(int lmax = kLmax) {
  return SurfaceGrid(lmax, {.nmax = 2, .oversampling = 1.5});
}
pm::AngularGrid SamplingGridOfFiles(int lmax = kLmax) {
  const auto grid = SamplingGrid(lmax);
  return pm::GaussLegendre(static_cast<int>(grid.CoLatitudes().size()) - 1,
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

}  // namespace

TEST_F(SecondOrderTest, In3DItsDirectionsAreKernelsDifferences) {
  // A 3-D run through time: the toy history on the ball under a lateral
  // viscosity and lateral moduli, frozen shorelines, Sdirk2 and the
  // discrete method. The second-order kernels (the ice's, the loads', the
  // tide's, and the fields of the viscosity's and the moduli's) against the
  // central difference of the first-order kernels of two 3-D runs:
  //  - the ice, a `change` file, a load in the water budget and a seismic
  //    source, each term appended plus and minus;
  //  - ln viscosity, a model file of a change that varies with the radius
  //    and laterally, the runs on viscosity files of the viscosity times
  //    exp(+-1e-3 of it);
  //  - the moduli, a model file of lateral changes of mu and kappa, the
  //    runs on moduli files with the changes added, 1e-3 of them.
  // Measured: 4e-11 to 6e-11 along the ice and the load (the problem
  // linear in them), 2.2e-9 along the source (the 3-D solves' rounding
  // against the small change it makes to the bulk modulus's kernel at the
  // poles' rows), 4.5e-7 along ln viscosity and 6e-8 along the moduli (the
  // differences' remainder at the step).
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
                      perturbation = "", more = "";
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
    {{"thickness": {{"file": "history.nc"}}}}{}],
    "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 0.5}}]{}}})",
                            c.ice.empty() ? "" : ", " + c.ice, c.more),
                kOcean, std::string{kTight3D} + "}",
                KernelsBlock(c.name,
                             R"("ice", "load", "tide", "viscosity", "moduli")",
                             !c.perturbation.empty()),
                c.perturbation.empty()
                    ? ""
                    : R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}, "water_budget": true,
             "moves_solid_surface": false}})",
        3.0e4 * s);
  };
  const auto source = [](double s) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 300,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}}})",
        1e22 * s, -2e22 * s, 1e22 * s, 3e22 * s, -1e22 * s, 2e22 * s);
  };
  struct Kind {
    std::string name, perturbation;
    Case plus, minus;
    double step, tolerance;
  };
  const auto kinds = std::vector<Kind>{
      {"ice", R"("forcing": {"ice_change": [)" + ice(1.0) + "]}",
       {.name = "p_ice", .ice = ice(1.0)}, {.name = "m_ice", .ice = ice(-1.0)},
       1.0, 2e-10},
      {"load", R"("forcing": {"load": [)" + load(1.0) + "]}",
       {.name = "p_load", .more = R"(, "load": [)" + load(1.0) + "]"},
       {.name = "m_load", .more = R"(, "load": [)" + load(-1.0) + "]"}, 1.0,
       2e-10},
      {"seismic", R"("forcing": {"seismic": [)" + source(1.0) + "]}",
       {.name = "p_seismic", .more = R"(, "seismic": [)" + source(1.0) + "]"},
       {.name = "m_seismic", .more = R"(, "seismic": [)" + source(-1.0) + "]"},
       1.0, 1e-8},
      {"viscosity", R"("model": {"viscosity": {"file": "d_eta.nc"}})",
       {.name = "p_eta", .viscosity = "eta_plus.nc"},
       {.name = "m_eta", .viscosity = "eta_minus.nc"}, kStep, 2e-6},
      {"moduli", R"("model": {"moduli": {"file": "d_moduli.nc"}})",
       {.name = "p_moduli", .moduli = "moduli_plus.nc"},
       {.name = "m_moduli", .moduli = "moduli_minus.nc"}, kStep, 2e-7}};
  for (const auto& kind : kinds) {
    const auto summary = app::RunKernel(
        config({.name = "s_" + kind.name, .perturbation = kind.perturbation}));
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity, lateral moduli), viscoelastic, "))
        << summary;
    static_cast<void>(app::RunKernel(config(kind.plus)));
    static_cast<void>(app::RunKernel(config(kind.minus)));
    const auto agreement =
        FilesAgainst(KernelsOf("s_" + kind.name), KernelsOf(kind.plus.name),
                     KernelsOf(kind.minus.name), kind.step);
    Report("3-D, " + kind.name, agreement, kind.tolerance);
  }
  // the field files say what they are
  const auto field = pm::ReadModelFile(KernelsOf("s_viscosity") /
                                       "site_viscosity_kernels_second_order.nc");
  EXPECT_TRUE(field.Meta().contains("second_order"));
  EXPECT_TRUE(field.Meta().contains("perturbation_viscosity"));
}

TEST_F(SecondOrderTest, In3DOnBurgersBodiesTheViscosityIsTheirBranches) {
  // A 3-D run on lateral moduli whose mantle is a Burgers body
  // (`model.burgers` over the viscous mantle of a ball with an elastic
  // core), the toy history:
  //  - `ln_viscosity` equal to one: the second-order kernels against the
  //    central difference of the first-order kernels of runs with the
  //    deck's viscosity and the region's Kelvin viscosity scaled by
  //    1 +- 1e-3;
  //  - a lateral `ln_viscosity` and the same field given as both branches'
  //    own (`ln_viscosity_branch_0` and `ln_viscosity_branch_1`) are one
  //    direction, and their second-order kernels one, to rounding.
  // Measured: 9.1e-7 against the difference (its remainder at the step);
  // the two directions' kernels bitwise alike.
  constexpr auto kStep = 1e-3;
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
  WriteModelFields(dir_ / "d_all.nc", model, {{"ln_viscosity", lateral}}, false,
                   files);
  WriteModelFields(dir_ / "d_each.nc", model,
                   {{"ln_viscosity_branch_0", lateral},
                    {"ln_viscosity_branch_1", lateral}},
                   false, files);
  const auto config = [&](const std::string& name, const std::string& deck,
                          double scale, const std::string& direction = "") {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{"deck": "{}",
    "moduli": {{"file": "moduli.nc"}},
    "burgers": [{{"depth_km": [0, 5371], "kelvin_viscosity": {:.17g},
                  "kelvin_modulus_fraction": 0.5}}]}},
  "discretisation": {{"lmax": 8}})",
                            deck, 1e20 * scale),
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                kOcean, std::string{kTight3D} + "}",
                KernelsBlock(name, R"("ice", "viscosity", "moduli")",
                             !direction.empty()),
                direction.empty()
                    ? std::string{}
                    : std::format(R"("perturbation": {{"model": {{"viscosity":
    {{"file": "{}"}}}}}})",
                                  direction)}));
  };
  const auto summary = app::RunKernel(config("one", "ball.deck", 1.0, "d_one.nc"));
  EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), ")) << summary;
  static_cast<void>(app::RunKernel(config("plus", "plus.deck", 1.0 + kStep)));
  static_cast<void>(app::RunKernel(config("minus", "minus.deck", 1.0 - kStep)));
  const auto agreement = FilesAgainst(KernelsOf("one"), KernelsOf("plus"),
                                      KernelsOf("minus"), half);
  Report("3-D, model.burgers, ln_viscosity", agreement, 2e-6);
  // the two branches' kernels each
  const auto fields = pm::ReadModelFile(KernelsOf("one") /
                                        "site_viscosity_kernels_second_order.nc");
  EXPECT_TRUE(fields.Parts().front().variables.contains("K_ln_eta_0"));
  EXPECT_TRUE(fields.Parts().front().variables.contains("K_ln_eta_1"));

  static_cast<void>(app::RunKernel(config("all", "ball.deck", 1.0, "d_all.nc")));
  static_cast<void>(app::RunKernel(config("each", "ball.deck", 1.0, "d_each.nc")));
  const auto branches = SecondOrderDeparture(KernelsOf("all"), KernelsOf("each"));
  std::cout << std::format(
      "[second order, 3-D, model.burgers] a lateral ln_viscosity and the same "
      "field as each branch's own: {:.3e} apart\n",
      branches);
  RecordProperty("branches_gap", std::format("{:.3e}", branches));
  EXPECT_LT(branches, 1e-9);
}

TEST_F(SecondOrderTest, In3DTheStaticProblemOnLateralModuli) {
  // The static problem on lateral moduli, a 3-D run of one node, under an
  // ice change, a load and a tide: along a change of the ice, a multiple of
  // the run's own lateral moduli (the runs on moduli files of it scaled by
  // 1 +- 1e-3) and, under migrating shorelines, the state's topography, the
  // second-order kernels against the central difference of the first-order
  // kernels; and the continuous method's are the discrete's, the problem
  // having no step. Measured: 5.5e-11 along the ice, 6.2e-8 along the
  // moduli (the difference's remainder), 1.6e-8 along the topography (the
  // 3-D closure's rounding against a small change, at the poles' rows of
  // the moduli kernels); the methods 7e-12 apart.
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
  struct Case {
    std::string name, moduli = "moduli.nc", ice = "", shorelines = "frozen",
                      topography = "topography.nc", perturbation = "",
                      method = "";
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
                       "fraction": -0.3}}}}{}],
    "load": [{{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                        "amplitude_kg_m2": 3.0e4}}, "water_budget": true,
              "moves_solid_surface": false}}],
    "tide": [{{"degree": 3, "order": 2, "amplitude_m2_s2": 0.6}}]}})",
                            c.ice.empty() ? "" : ", " + c.ice),
                std::format(R"("ocean": {{"date_kyr_bp": 2, "shorelines": "{}",
    "topography": {{"file": "{}"}}, "ice": {{"file": "history.nc"}}}})",
                            c.shorelines, c.topography),
                std::string{kTight3D} + "}",
                KernelsBlock(c.name, R"("ice", "load", "tide", "moduli")",
                             !c.perturbation.empty(), c.method,
                             kStaticMeasurements),
                c.perturbation.empty()
                    ? ""
                    : R"("perturbation": {)" + c.perturbation + "}"}));
  };
  const auto ice = [](double s) {
    return std::format(R"({{"change": {{"file": "{}"}}}})",
                       s > 0 ? "change.nc" : "change_minus.nc");
  };
  struct Kind {
    std::string name;
    Case second, plus, minus;
    double step, tolerance;
  };
  const auto kinds = std::vector<Kind>{
      {"ice",
       {.name = "s_ice",
        .perturbation = R"("forcing": {"ice_change": [)" + ice(1.0) + "]}"},
       {.name = "p_ice", .ice = ice(1.0)},
       {.name = "m_ice", .ice = ice(-1.0)},
       1.0,
       2e-10},
      {"moduli",
       {.name = "s_moduli",
        .perturbation = R"("model": {"moduli": {"file": "d_moduli.nc"}})"},
       {.name = "p_moduli", .moduli = "moduli_plus.nc"},
       {.name = "m_moduli", .moduli = "moduli_minus.nc"},
       kStep,
       2e-7},
      {"topography",
       {.name = "s_topography",
        .shorelines = "migrating",
        .perturbation = R"("ocean": {"topography": {"file": "d_topography.nc"}})"},
       {.name = "p_topography",
        .shorelines = "migrating",
        .topography = "topography_plus.nc"},
       {.name = "m_topography",
        .shorelines = "migrating",
        .topography = "topography_minus.nc"},
       1.0,
       5e-8}};
  for (const auto& kind : kinds) {
    const auto summary = app::RunKernel(config(kind.second));
    EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), elastic, static, "))
        << summary;
    static_cast<void>(app::RunKernel(config(kind.plus)));
    static_cast<void>(app::RunKernel(config(kind.minus)));
    const auto agreement =
        FilesAgainst(KernelsOf(kind.second.name), KernelsOf(kind.plus.name),
                     KernelsOf(kind.minus.name), kind.step);
    Report("3-D, static, " + kind.name, agreement, kind.tolerance);
  }
  auto continuous = kinds[1].second;
  continuous.name = "c_moduli";
  continuous.method = R"(, "method": "continuous")";
  static_cast<void>(app::RunKernel(config(continuous)));
  const auto methods =
      SecondOrderDeparture(KernelsOf("c_moduli"), KernelsOf("s_moduli"));
  std::cout << std::format(
      "[second order, 3-D, static] the continuous method's second-order "
      "kernels {:.3e} from the discrete's\n",
      methods);
  RecordProperty("static_methods", std::format("{:.3e}", methods));
  EXPECT_LT(methods, 1e-10);
}

namespace {

/// The tide direction of the symmetry gates: the coefficient of degree 2
/// and order 1, its real part one m^2 s^-2 at every node.
constexpr auto kTideDirection =
    R"({"forcing": {"tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0}]}})";

/// The second-order tide kernels of a measurement paired with the tide
/// direction: the sum over the nodes of the degree-2, order-1 rows'
/// `dJ_damplitude`.
double TidePaired(const fs::path& table) {
  const auto read = ReadTable(table);
  const auto column = [&](const std::string& name) {
    const auto at = std::ranges::find(read.columns, name);
    EXPECT_NE(at, read.columns.end()) << name;
    return static_cast<std::size_t>(at - read.columns.begin());
  };
  const auto l = column("l"), m = column("m"), re = column("dJ_damplitude");
  auto out = 0.0;
  for (const auto& row : read.rows) {
    if (row[l] == 2.0 && row[m] == 1.0) out += row[re];
  }
  return out;
}

/// The radial second-order viscosity kernels of a measurement on a Maxwell
/// body of viscosity `eta` paired with a change of ln eta, `change(r)` (by
/// default one everywhere): the sum over the rows of W_m3 K_branch_0 eta
/// times the change at the row's radius.
double UniformPaired1D(const fs::path& table, double eta,
                       const std::function<double(double)>& change = {}) {
  const auto read = ReadTable(table);
  EXPECT_EQ(read.columns,
            (std::vector<std::string>{"r_m", "K_branch_0", "W_m3"}));
  auto out = 0.0;
  for (const auto& row : read.rows) {
    out += row[2] * row[1] * eta * (change ? change(row[0]) : 1.0);
  }
  return out;
}

/// A number of a model file's metadata.
double MetaOf(const fs::path& path, const std::string& key) {
  return std::get<std::vector<double>>(pm::ReadModelFile(path).Meta().at(key))
      .front();
}

}  // namespace

TEST_F(SecondOrderTest, TheHessianIsSymmetric) {
  // The second-order kernels along a direction a paired with a direction b
  // are those along b paired with a: a the tide (a coefficient at every
  // node), b ln viscosity raised by one everywhere, the tide kernels along
  // b against the viscosity kernels along a, by the discrete method under
  // Sdirk2, whose Hessian is the exact derivative of the exact gradient.
  //  - 1-D: the ball through time, frozen shorelines; the viscosity
  //    kernels' pairing the sum over the rows of W K eta;
  //  - 3-D: the ball under a lateral viscosity and lateral moduli; the
  //    pairing the file's `uniform_log_viscosity_pairing`;
  //  - 1-D about a calibrated run under migrating shorelines, the partial
  //    kernels' derivatives (`kernels.calibration.derivative` "partial"),
  //    the Hessian of the run with the calibrated initial sea level held:
  //    a calibrated run takes the ice alone, and a and b are two changes
  //    of ln viscosity (one everywhere, and 1 - r / 2a), the viscosity
  //    kernels along each paired with the other (the files' kernels in
  //    eta: the pairings each have the term -sum W K_lnEta da db, which is
  //    symmetric in a and b).
  // Measured: 1.7e-11 and 2.6e-12 (1-D), 2.4e-13 and 6.9e-14 (3-D), 1.7e-12
  // and 2.9e-13 (calibrated).
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  const auto linear = [](double r) { return 1.0 - 0.5 * r / 6371e3; };
  WriteModelFields(dir_ / "d_linear.nc", model,
                   {{"ln_viscosity",
                     [=](double r, double, double) { return linear(r); }}},
                   true);
  WriteLateralViscosity(dir_ / "eta.nc", model);
  WriteLateralModuli(dir_ / "moduli.nc", model);
  const auto equator = [](double theta) {
    return std::exp(-std::pow((theta - std::numbers::pi / 2) / 0.15, 2));
  };
  WriteToyField(dir_ / "topography.nc", "topography",
                [=](std::size_t, double theta, double) {
                  return 3000.0 * std::cos(theta) -
                         3500.0 * std::exp(-std::pow(theta / 0.3, 2)) -
                         equator(theta);
                });
  struct Case {
    std::string name, model, ocean, kernels, solver;
    bool lateral;
  };
  const auto cases = std::vector<Case>{
      {"radial", R"("model": {"deck": "ball.deck"})", kOcean, "", kTight, false},
      {"lateral", R"("model": {"deck": "ball.deck",
    "viscosity": {"file": "eta.nc"}, "moduli": {"file": "moduli.nc"}})",
       kOcean, "", std::string{kTight3D} + "}", true},
      {"calibrated", R"("model": {"deck": "ball.deck"})",
       R"("ocean": {"at": "end", "shorelines": "migrating",
    "calibration": {"rtol": 1e-12, "iterations": 60},
    "topography": {"file": "topography.nc"}, "ice": {"file": "marine.nc"}})",
       R"(, "calibration": {"derivative": "partial"})", kTight, false}};
  for (const auto& c : cases) {
    const auto config = [&](const std::string& name,
                            const std::string& direction) {
      const auto history =
          c.name == "calibrated" ? std::string{"marine.nc"} : "history.nc";
      return Write(
          name + ".json",
          Config({c.model + R"(, "discretisation": {"lmax": 8})", kDated,
                  std::format(R"("forcing": {{"ice_change": [{{"thickness":
                     {{"file": "{}"}}}}]}})",
                              history),
                  c.ocean, c.solver,
                  KernelsBlock(name,
                               c.name == "calibrated"
                                   ? R"("ice", "viscosity")"
                                   : R"("tide", "viscosity")",
                               true, c.kernels),
                  R"("perturbation": )" + direction}));
    };
    const auto calibrated = c.name == "calibrated";
    static_cast<void>(app::RunKernel(config(
        c.name + "_a",
        calibrated ? R"({"model": {"viscosity": {"file": "d_linear.nc"}}})"
                   : kTideDirection)));
    const auto summary = app::RunKernel(config(
        c.name + "_b", R"({"model": {"viscosity": {"file": "d_one.nc"}}})"));
    if (calibrated) {
      EXPECT_NE(summary.find("; partial derivatives, the calibrated initial "
                             "sea level held"),
                std::string::npos)
          << summary;
      const auto ice = pm::ReadSurfaceFile(
          KernelsOf("calibrated_b") / "site_ice_kernels_second_order.nc");
      EXPECT_EQ(std::get<std::string>(ice.meta.at("calibration_derivative")),
                "partial");
      EXPECT_GT(MaxAbs(ice.variables.at("initial_sea_level_dual").values), 0.0);
    }
    for (const auto* measurement : {"site", "uplift"}) {
      const auto m = std::string{measurement};
      const auto along = [&](const std::string& name) {
        return KernelsOf(c.name + "_" + name) /
               (m + "_viscosity_kernel_second_order.txt");
      };
      const auto ab =
          c.lateral ? MetaOf(KernelsOf(c.name + "_a") /
                                 (m + "_viscosity_kernels_second_order.nc"),
                             "uniform_log_viscosity_pairing")
                    : UniformPaired1D(along("a"), 1e21);
      const auto ba = calibrated
                          ? UniformPaired1D(along("b"), 1e21, linear)
                          : TidePaired(KernelsOf(c.name + "_b") /
                                       (m + "_tide_kernels_second_order.txt"));
      const auto gap = std::abs(ab - ba) / std::abs(ba);
      std::cout << std::format(
          "[second order, symmetry, {}, {}] along a paired with b {:.9e}, "
          "along b paired with a {:.9e}: {:.3e} apart\n",
          c.name, m, ab, ba, gap);
      RecordProperty(c.name + "_" + m, std::format("{:.3e}", gap));
      EXPECT_NE(ba, 0.0);
      EXPECT_LT(gap, 1e-10) << c.name << " " << m;
    }
  }
}

TEST_F(SecondOrderTest, OfACalibratedRunUnderFrozenShorelines) {
  // A calibrated run (the state at the window's end) under frozen
  // shorelines does not depend on its initial sea level: its total
  // kernels are its partial ones, the calibration's multiplier zero at
  // every parameter, and their second-order kernels are the derivatives
  // of both. Along ln viscosity raised by one, against the central
  // difference of the first-order kernels of two runs on decks of the
  // viscosity times exp(+-1e-3), each calibrating again; the multiplier's
  // derivative, `target_sea_level_dual`, is zero.
  static_cast<void>(Write("ball.deck", BallDeck()));
  const auto model = ModelOf(dir_ / "ball.deck");
  constexpr auto kStep = 1e-3;
  static_cast<void>(Write("up.deck", BallDeck(0.0, 1e21 * std::exp(kStep))));
  static_cast<void>(Write("down.deck", BallDeck(0.0, 1e21 * std::exp(-kStep))));
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  const auto config = [&](const std::string& name, const std::string& deck,
                          bool second) {
    return Write(
        name + ".json",
        Config({std::format(R"("model": {{"deck": "{}"}},
  "discretisation": {{"lmax": 8}})",
                            deck),
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                R"("ocean": {"at": "end", "calibration": {"rtol": 1e-12},
    "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}})",
                kTight, KernelsBlock(name, R"("ice", "viscosity", "moduli")", second),
                second ? R"("perturbation": {"model": {"viscosity":
                   {"file": "d_one.nc"}}})"
                       : ""}));
  };
  const auto summary = app::RunKernel(config("s", "ball.deck", true));
  EXPECT_NE(summary.find("; total derivatives, the initial sea level "
                         "calibrated again"),
            std::string::npos)
      << summary;
  static_cast<void>(app::RunKernel(config("p", "up.deck", false)));
  static_cast<void>(app::RunKernel(config("m", "down.deck", false)));
  const auto agreement =
      FilesAgainst(KernelsOf("s"), KernelsOf("p"), KernelsOf("m"), kStep);
  Report("1-D, calibrated, frozen, total", agreement, 1e-6);
  const auto ice =
      pm::ReadSurfaceFile(KernelsOf("s") / "site_ice_kernels_second_order.nc");
  EXPECT_EQ(std::get<std::string>(ice.meta.at("calibration_derivative")),
            "total");
  EXPECT_EQ(MaxAbs(ice.variables.at("target_sea_level_dual").values), 0.0);
}

TEST_F(SecondOrderTest, TheFilesAreTheLibrarysCall) {
  // The program's second-order kernels are the library's own call about
  // the same recorded run (the program's run, built by its pieces): the
  // direction formed by hand (ln viscosity raised by one, delta itau_b =
  // -itau_b everywhere it relaxes), v's tangent-linear run recorded, and
  // AdjointSolver::SecondOrderKernels / AdjointSolver3D::SecondOrderKernels
  // with the measurement's duals; the files' fields converted back to the
  // library's units, to rounding. By either `kernels.method` under Sdirk2,
  // the default scheme.
  //  - 1-D: the ice file's `thickness_dual` (the first node's less the
  //    baseline's, the state held), the moduli table's `K_mu`, and the
  //    viscosity table's `K_branch_0` (the derivative of -(itau^2 / mu) K
  //    along delta itau = -itau: -(itau^2 / mu) dK + 2 (itau^2 / mu) K);
  //  - 3-D: the ice file's `thickness_dual` and the viscosity file's
  //    `uniform_log_viscosity_pairing`.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  WriteLateralViscosity(dir_ / "eta.nc", model);
  WriteLateralModuli(dir_ / "moduli.nc", model);
  const auto config = [&](const std::string& name, const std::string& model_,
                          const std::string& solver, AdjointMethod method) {
    return Write(
        name + ".json",
        Config({model_ + R"(, "discretisation": {"lmax": 8})", kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                kOcean, solver,
                KernelsBlock(name, R"("ice", "viscosity", "moduli")", true,
                             method == AdjointMethod::Continuous
                                 ? R"(, "method": "continuous")"
                                 : ""),
                R"("perturbation": {"model": {"viscosity": {"file": "d_one.nc"}}})"}));
  };
  // the largest departure of a file's values from the library's, over the
  // library's largest
  const auto departure = [](std::span<const double> file,
                            std::span<const double> library) {
    EXPECT_EQ(file.size(), library.size());
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < std::min(file.size(), library.size());
         ++i) {
      diff = std::max(diff, std::abs(file[i] - library[i]));
    }
    return diff / MaxAbs(library);
  };
  // the ice file's thickness kernel against the library's, the first node's
  // less its baseline
  const auto thickness = [&](const fs::path& file, const Kernels& library,
                             const app::Built& built, double length) {
    const auto read = pm::ReadSurfaceFile(file);
    auto want = std::vector<double>{};
    const auto toKernel = built.factor / (length * length * length);
    for (auto n = std::size_t{0}; n < library.thickness.size(); ++n) {
      for (auto i = std::size_t{0}; i < library.thickness[n].size(); ++i) {
        want.push_back(toKernel * (library.thickness[n][i] -
                                   (n == 0 ? library.baseline[i] : 0.0)));
      }
    }
    return departure(read.variables.at("thickness_dual").values, want);
  };
  auto worst = 0.0;
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto suffix = method == AdjointMethod::Discrete ? "" : "_continuous";

    // -- 1-D
    {
      const auto radial = std::string{"radial"} + suffix;
      const auto path =
          config(radial, R"("model": {"deck": "ball.deck"})", kTight, method);
      static_cast<void>(app::RunKernel(path));
      const auto top = pm::ReadJsonFile(path);
      auto input = app::ReadKernelInput(path);
      const auto earth = app::BuildEarth(input.run, top);
      const auto& run = input.run;
      static_cast<void>(app::WithRecorded1D(
          run, input.trajectory, top, earth, app::Meaning(run, earth),
          [&](const app::Recorded1D& recorded) {
            const auto& evolution = *recorded.evolution;
            const auto& times = recorded.steps.times;
            const auto& rheology = *recorded.rheology;
            const auto trajectory =
                ForwardTrajectory::FromSnapshots(recorded.snapshots);
            TangentPerturbation v;
            for (auto b = std::size_t{0}; b < rheology.BranchCount(); ++b) {
              auto change = rheology.BranchAt(b).itau;
              for (auto& x : change) x = -x;
              v.itau.push_back(std::move(change));
            }
            const auto tangent =
                TangentLinearSolver(evolution, {.method = method,
                                                .rtol = run.solver.rtol,
                                                .record = true})
                    .Run(times, v, &trajectory);
            const auto& solver = *recorded.solver;
            const auto& material = earth.material;
            const auto& scales = material.ModelScales();
            const auto rSi = scales.Length();
            for (const auto& spec : input.kernels.measurements) {
              const auto built = app::BuildMeasurement<Evolution::Snapshot>(
                  spec, run, solver, recorded.snapshots, recorded.steps);
              const auto [kernels, derivatives] =
                  AdjointSolver(evolution, {.method = method,
                                            .rtol = run.solver.rtol,
                                            .material_kernels = true})
                      .SecondOrderKernels(times, built.measurements, tangent,
                                          v.itau, trajectory);
              const auto stem = KernelsOf(radial) / spec.name;
              auto views = Kernels{};
              views.thickness = derivatives.thickness;
              views.baseline = derivatives.baseline;
              worst = std::max(
                  worst, thickness(stem.string() + "_ice_kernels_second_order.nc",
                                   views, built, solver.SolverScales().Length()));
              // the moduli table: r_m W_m3 kappa_Pa mu_Pa K_kappa K_mu
              const auto moduli = ReadTable(stem.string() +
                                            "_moduli_kernel_second_order.txt");
              auto column = std::vector<double>{};
              for (const auto& row : moduli.rows) column.push_back(row[5]);
              auto want = derivatives.muKernel;
              const auto toMu = built.factor /
                                (scales.Factor(ModulusDimensions) * rSi * rSi * rSi);
              for (auto& x : want) x *= toMu;
              worst = std::max(worst, departure(column, want));
              // the viscosity table: r_m K_branch_0 W_m3
              const auto viscosity = ReadTable(
                  stem.string() + "_viscosity_kernel_second_order.txt");
              column.clear();
              for (const auto& row : viscosity.rows) column.push_back(row[1]);
              const auto& branch = rheology.BranchAt(0);
              const auto toEta = built.factor / (scales.Factor(ViscosityDimensions) *
                                                 rSi * rSi * rSi);
              want.assign(branch.itau.size(), 0.0);
              for (auto i = std::size_t{0}; i < want.size(); ++i) {
                if (!(branch.mu[i] > 0.0)) continue;
                const auto a = branch.itau[i] * branch.itau[i] / branch.mu[i];
                want[i] = toEta * (-a * derivatives.itauKernel[0][i] +
                                   2.0 * a * kernels.itauKernel[0][i]);
              }
              worst = std::max(worst, departure(column, want));
            }
            return std::string{};
          }));
    }

    // -- 3-D
    {
      const auto lateral = std::string{"lateral"} + suffix;
      const auto path = config(lateral, R"("model": {"deck": "ball.deck",
      "viscosity": {"file": "eta.nc"}, "moduli": {"file": "moduli.nc"}})",
                               std::string{kTight3D} + "}", method);
      static_cast<void>(app::RunKernel(path));
      const auto top = pm::ReadJsonFile(path);
      auto input = app::ReadKernelInput(path);
      const auto earth = app::BuildEarth(input.run, top);
      const auto& run = input.run;
      static_cast<void>(app::WithRecorded3D(
          run, input.trajectory, top, earth, app::Meaning(run, earth),
          [&](const app::Recorded3D& recorded) {
            const auto& evolution = *recorded.evolution;
            const auto& times = recorded.steps.times;
            const auto& op = *recorded.op;
            const auto& coupling = op.Coupling();
            const auto ones = std::vector<double>(coupling.GridValueCount(), 1.0);
            TangentPerturbation3D v;
            v.logViscosity = ones;
            const auto tangent =
                TangentLinearSolver3D(
                    evolution, {.method = method,
                                .rtol = run.solver.rtol,
                                .max_coupled_iterations =
                                    run.solver.step_max_iterations,
                                .coupled_rtol = run.solver.step_rtol,
                                .record = true})
                    .Run(times, v, recorded.trajectory);
            const auto itau = TangentLinearSolver3D::ItauOfLogViscosity(op, ones);
            const auto& solver = *recorded.solver;
            for (const auto& spec : input.kernels.measurements) {
              const auto built = app::BuildMeasurement<Evolution3D::Snapshot>(
                  spec, run, solver, recorded.snapshots, recorded.steps);
              const auto [kernels, derivatives] =
                  AdjointSolver3D(evolution,
                                  {.method = method,
                                   .rtol = run.solver.rtol,
                                   .max_coupled_iterations =
                                       run.solver.step_max_iterations,
                                   .coupled_rtol = run.solver.step_rtol,
                                   .material_kernels = true})
                      .SecondOrderKernels(times, built.measurements, tangent,
                                          itau, *recorded.trajectory);
              const auto stem = KernelsOf(lateral) / spec.name;
              auto views = Kernels{};
              views.thickness = derivatives.ice.thickness;
              views.baseline = derivatives.ice.baseline;
              worst = std::max(
                  worst, thickness(stem.string() + "_ice_kernels_second_order.nc",
                                   views, built, solver.SolverScales().Length()));
              auto uniform = 0.0;
              for (const auto& branch : derivatives.logViscosityKernel3D) {
                uniform += VolumePairing(coupling, branch, ones);
              }
              uniform *= built.factor;
              const auto file = MetaOf(stem.string() +
                                           "_viscosity_kernels_second_order.nc",
                                       "uniform_log_viscosity_pairing");
              worst = std::max(worst, std::abs(file - uniform) / std::abs(uniform));
            }
            return std::string{};
          }));
    }
    std::cout << std::format(
        "[second order, the library's call, {}] the files {:.3e} from it\n",
        method == AdjointMethod::Discrete ? "discrete" : "continuous", worst);
  }
  RecordProperty("library_gap", std::format("{:.3e}", worst));
  EXPECT_LT(worst, 1e-12);
}

TEST_F(SecondOrderTest, TheContinuousMethodConvergesToTheDiscreteUnderSdirk2) {
  // `kernels.method` "continuous" with `second_order` under Sdirk2, the
  // default scheme, on the ball under the toy history along a change of ln
  // viscosity, 1-D and 3-D (a lateral viscosity and lateral moduli), at a
  // step and its halving (250 and 125 yr): the second-order kernels tend to
  // the discrete method's at second order, the viscosity kernel's file and
  // the ice kernel's.
  //
  // The ice file is compared node by node, and there second order is a
  // statement of the entries' ABSOLUTE departure. A node's entry of
  // `thickness_dual` is a quadrature weight times the kernel's density in
  // time, of the order of the step itself. The density stops at a
  // measurement, so the entry at the measurement's node is one-sided: half
  // a step times the density just before it by the continuous method, the
  // scheme's own end weight by the discrete. The two differ there by an
  // amount of second order, which is first order against the entry (and
  // against the file's largest value, an entry too): PRINTED, as the entry's
  // share, and not what is held. A term missing from the sweep at a
  // measurement would be first order absolutely, the share not falling.
  // The first-order file carries the same departure in its entry less
  // `instantaneous_dual`, beside the delta's amplitude, which does not
  // shrink with the step; along a change of viscosity the delta's
  // derivative is zero and the second-order entry is the one-sided part
  // alone. The site's measurement is at the last node, the uplift's at the
  // middle one. Measured at 9 and 17 nodes, 1-D and 3-D: the viscosity
  // files fall by 3.59 to 4.18; the ice entries' departure at the
  // measurement's node by 3.52 to 3.62, its share of the entry 0.14 to 0.24
  // at 9 nodes and falling by 1.92 to 1.96; at the other nodes by 4.15 to
  // 5.08.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_eta.nc", model,
                   {{"ln_viscosity",
                     [](double r, double, double) {
                       return 0.3 + 0.7 * std::pow(r / 6371000.0, 2);
                     }}},
                   true);
  WriteLateralViscosity(dir_ / "eta.nc", model);
  WriteLateralModuli(dir_ / "moduli.nc", model);
  const auto config = [&](const std::string& name, bool lateral, int nodes,
                          const std::string& method) {
    return Write(
        name + ".json",
        Config(
            {lateral
                 ? R"("model": {"deck": "ball.deck",
    "viscosity": {"file": "eta.nc"}, "moduli": {"file": "moduli.nc"}},
  "discretisation": {"lmax": 8})"
                 : R"("model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8})",
             std::format(R"("problem": "quasi_static",
  "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": {}}})",
                         nodes),
             R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
             kOcean,
             lateral ? std::string{kTight3D} + "}" : std::string{kTight},
             KernelsBlock(name, R"("ice", "viscosity")", true,
                          std::format(R"(, "method": "{}")", method)),
             R"("perturbation": {"model": {"viscosity": {"file": "d_eta.nc"}}})"}));
  };
  // the largest departure of the continuous files' values from the
  // discrete's, over the discrete's largest
  const auto departure = [](std::span<const double> c,
                            std::span<const double> d) {
    EXPECT_EQ(c.size(), d.size());
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < std::min(c.size(), d.size()); ++i) {
      diff = std::max(diff, std::abs(c[i] - d[i]));
    }
    return diff / MaxAbs(d);
  };
  for (const bool lateral : {false, true}) {
    auto gaps = std::map<std::string, std::vector<double>>{};
    // per measurement, at each step: the ice entries' absolute departure at
    // the measurement's node and at the other nodes, and the discrete
    // entry's size at that node
    auto ice_gaps = std::map<std::string, std::vector<std::array<double, 3>>>{};
    for (const auto nodes : {9, 17}) {
      const auto tag =
          std::format("{}_{}", lateral ? "lateral" : "radial", nodes);
      for (const auto* method : {"discrete", "continuous"}) {
        static_cast<void>(
            app::RunKernel(config(tag + "_" + method, lateral, nodes, method)));
      }
      for (const auto* site : {"site", "uplift"}) {
        auto of = [&](const char* method) {
          return KernelsOf(tag + "_" + method) / site;
        };
        const auto ice = [&](const char* method) {
          return pm::ReadSurfaceFile(of(method).string() +
                                     "_ice_kernels_second_order.nc")
              .variables.at("thickness_dual")
              .values;
        };
        {
          // the entries' largest absolute departure at the measurement's
          // node and at the others, and the discrete entry's size there
          const auto c = ice("continuous");
          const auto d = ice("discrete");
          ASSERT_EQ(c.size(), d.size());
          const auto per = d.size() / static_cast<std::size_t>(nodes);
          ASSERT_EQ(per * static_cast<std::size_t>(nodes), d.size());
          const auto node = static_cast<std::size_t>(
              std::string_view{site} == "site" ? nodes - 1 : (nodes - 1) / 2);
          auto there = 0.0, elsewhere = 0.0, entry = 0.0;
          for (auto i = std::size_t{0}; i < d.size(); ++i) {
            const auto gap = std::abs(c[i] - d[i]);
            if (i / per == node) {
              there = std::max(there, gap);
              entry = std::max(entry, std::abs(d[i]));
            } else {
              elsewhere = std::max(elsewhere, gap);
            }
          }
          ice_gaps[std::format("{} ice", site)].push_back(
              {there, elsewhere, entry});
        }
        if (lateral) {
          const auto pairing = [&](const char* method) {
            return MetaOf(
                of(method).string() + "_viscosity_kernels_second_order.nc",
                "uniform_log_viscosity_pairing");
          };
          gaps[std::format("{} viscosity", site)].push_back(
              std::abs(pairing("continuous") - pairing("discrete")) /
              std::abs(pairing("discrete")));
        } else {
          const auto table = [&](const char* method) {
            return KernelColumns(ReadTable(
                of(method).string() + "_viscosity_kernel_second_order.txt"));
          };
          gaps[std::format("{} viscosity", site)].push_back(
              departure(table("continuous"), table("discrete")));
        }
      }
    }
    for (const auto& [name, g] : gaps) {
      const auto label = std::format("{}, {}", lateral ? "3-D" : "1-D", name);
      std::cout << std::format(
          "[second order, continuous against discrete, sdirk2, {}] {:.3e} "
          "{:.3e} "
          "({:.2f})\n",
          label, g[0], g[1], g[0] / g[1]);
      RecordProperty(label, std::format("{:.3e} {:.3e}", g[0], g[1]));
      EXPECT_LT(g[0], 0.05) << label;
      EXPECT_GT(g[0] / g[1], 3.0) << label;
      EXPECT_LT(g[0] / g[1], 5.0) << label;
    }
    for (const auto& [name, g] : ice_gaps) {
      const auto label = std::format("{}, {}", lateral ? "3-D" : "1-D", name);
      std::cout << std::format(
          "[second order, continuous against discrete, sdirk2, {}] at the "
          "measurement's node {:.3e} {:.3e} ({:.2f}), of the entry {:.3e} "
          "{:.3e} ({:.2f}); at the other nodes {:.3e} {:.3e} ({:.2f})\n",
          label, g[0][0], g[1][0], g[0][0] / g[1][0], g[0][0] / g[0][2],
          g[1][0] / g[1][2], (g[0][0] / g[0][2]) / (g[1][0] / g[1][2]), g[0][1],
          g[1][1], g[0][1] / g[1][1]);
      RecordProperty(label, std::format("{:.3e} {:.3e}", g[0][0], g[1][0]));
      // second order absolutely, at the measurement's node and elsewhere
      EXPECT_GT(g[0][0] / g[1][0], 3.0) << label;
      EXPECT_GT(g[0][1] / g[1][1], 3.0) << label;
      // and the entry's share at the first step, of the order of the step
      // against the relaxation time
      EXPECT_LT(g[0][0] / g[0][2], 0.5) << label;
    }
  }
}

TEST_F(SecondOrderTest, RefusesWhatIsNotBuilt) {
  // What the second-order kernels cannot be is refused by name, each
  // saying what is not built and the cure, by `kernel` and by the other
  // commands that read the `kernels` block:
  //  - `second_order` without a `perturbation` block, the direction;
  //  - the kernels as fields about a radial run (`lateral` true);
  //  - a calibrated run under migrating shorelines, its kernels the total
  //    derivatives (the partial ones' derivatives are built);
  //  - a decay time, whose fit's second derivative is not built.
  // The continuous method is taken through time under either scheme. A
  // config holding a `perturbation` block and no `second_order` writes the
  // first-order kernels alone.
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  WriteToyHistory(dir_ / "history.nc");
  WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  constexpr auto kDirection =
      R"("perturbation": {"model": {"viscosity": {"file": "d_one.nc"}}})";
  struct Case {
    std::string name, kernels = "", solver = kTight, ocean = kOcean,
                      perturbation = kDirection, measurements = kMeasurements,
                      output = "";
    bool second = true;
  };
  const auto config = [&](const Case& c) {
    return Write(
        c.name + ".json",
        Config({R"("model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8})",
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                c.ocean, c.solver, c.output,
                KernelsBlock(c.name, R"("ice", "viscosity")", c.second,
                             c.kernels, c.measurements),
                c.perturbation}));
  };
  const auto kernel = [](const fs::path& path) { return app::RunKernel(path); };
  const auto solve = [](const fs::path& path) { return app::RunSolve(path); };
  {
    const auto path = config({.name = "none",
                              .perturbation = "",
                              .output = R"("output": {"surface": "out/none.nc"})"});
    ExpectRefused(kernel, path, "kernels.second_order",
                  "the direction is the top-level `perturbation` block");
    ExpectRefused(solve, path, "kernels.second_order",
                  "the direction is the top-level `perturbation` block");
  }
  ExpectRefused(kernel,
                config({.name = "lateral", .kernels = R"(, "lateral": true)"}),
                "kernels.lateral",
                "the second-order kernels as fields about a radial run are not "
                "built");
  ExpectRefused(kernel,
                config({.name = "calibrated",
                        .ocean = R"("ocean": {"at": "end", "shorelines": "migrating",
    "topography": {"file": "marine.nc"}, "ice": {"file": "marine.nc"}})"}),
                "kernels.second_order",
                "with the derivative of the calibration's multiplier, are not "
                "built");
  ExpectRefused(kernel,
                config({.name = "decay",
                        .measurements = R"("measurements": [
      {"type": "relative_sea_level_cap", "name": "site", "latitude": -30,
       "longitude": 10, "width_deg": 25},
      {"type": "decay_time", "name": "decay", "latitude": 60,
       "longitude": 0, "width_deg": 20, "window_start_kyr_bp": 2,
       "time_uncertainty_kyr": 0.0, "value_uncertainty_m": 0.5}])"}),
                "kernels.measurements[1].type",
                "the second derivative of the fit");
  // the continuous method under the default scheme, Sdirk2, and under the
  // one-stage scheme by name, is taken
  const auto sdirk2 = config(
      {.name = "continuous", .kernels = R"(, "method": "continuous")"});
  EXPECT_NE(app::RunKernel(sdirk2).find("second-order kernels"), std::string::npos);
  EXPECT_TRUE(fs::exists(KernelsOf("continuous") /
                         "site_viscosity_kernel_second_order.txt"));
  const auto trapezoid = config(
      {.name = "trapezoid",
       .kernels = R"(, "method": "continuous")",
       .solver = R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12, "scheme": "exponential_trapezoid"})"});
  EXPECT_NE(app::RunKernel(trapezoid).find("second-order kernels"),
            std::string::npos);
  EXPECT_TRUE(fs::exists(KernelsOf("trapezoid") /
                         "site_viscosity_kernel_second_order.txt"));
  // a perturbation block without `second_order`: the first-order kernels
  static_cast<void>(app::RunKernel(config({.name = "first", .second = false})));
  EXPECT_TRUE(fs::exists(KernelsOf("first") / "site_ice_kernels.nc"));
  EXPECT_FALSE(fs::exists(KernelsOf("first") / "site_ice_kernels_second_order.nc"));
}

TEST_F(SecondOrderTest, ABodyWithoutAnOceanUnderAnEarthquakeAtANode) {
  // A radial body without an ocean on its own clock (years), under a load,
  // a tide and a fault whose earthquake falls at a node of the window,
  // which the run steps twice: its kernels in the load and the tide carry
  // the left limit's companions there, and its moment kernels the right
  // side's. Along a load with a history, a tide, a point source of the
  // earthquake's onset, ln viscosity raised by one and the moduli, the
  // second-order kernels (the companions and the moment kernels with them)
  // against the central difference of the first-order kernels of runs
  // with the members appended plus and minus, or on decks of the viscosity
  // times exp(+-1e-3) and of the moduli a step 1e-3 of a tenth apart.
  // Measured: 5e-14 to 7.5e-11 along the forcing (rounding), 5e-7 and 8e-8
  // along the viscosity and the moduli (the differences' remainder).
  const auto deck = Write("ball.deck", BallDeck());
  const auto model = ModelOf(deck);
  constexpr auto kStep = 1e-3;
  constexpr auto kTenth = 0.1;
  static_cast<void>(Write("up.deck", BallDeck(0.0, 1e21 * std::exp(kStep))));
  static_cast<void>(Write("down.deck", BallDeck(0.0, 1e21 * std::exp(-kStep))));
  static_cast<void>(Write("stiff.deck", BallDeck(kTenth * kStep,
                                                 1e21 * (1.0 + kTenth * kStep))));
  static_cast<void>(Write("soft.deck", BallDeck(-kTenth * kStep,
                                                1e21 * (1.0 - kTenth * kStep))));
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  WriteModelFields(dir_ / "d_moduli.nc", model,
                   {{"mu", [](double, double, double) { return kTenth * kBallMu; }},
                    {"kappa",
                     [](double, double, double) { return kTenth * kBallKappa; }}},
                   true);
  const auto load = [](double s) {
    return std::format(
        R"({{"cap": {{"latitude": 20, "longitude": 120, "width_deg": 20,
                    "amplitude_kg_m2": {}}}, "history": [[0, 0.2], [300, 1]]}})",
        3.0e4 * s);
  };
  const auto tide = [](double s) {
    return std::format(
        R"({{"degree": 3, "order": 2, "amplitude_m2_s2": {},
          "imaginary_m2_s2": {}}})",
        0.6 * s, 0.2 * s);
  };
  const auto source = [](double s) {
    return std::format(
        R"({{"point": {{"latitude": 30, "longitude": 40, "depth_km": 30,
            "moment_tensor_nm": {{"rr": {}, "tt": {}, "pp": {}, "rt": {},
                                 "rp": {}, "tp": {}}}}}, "onset": 100}})",
        1e21 * s, -2e21 * s, 1e21 * s, 3e21 * s, -1e21 * s, 2e21 * s);
  };
  struct Case {
    std::string name, deck = "ball.deck", list = "", term = "",
                      perturbation = "";
  };
  const auto config = [&](const Case& c) {
    auto lists = std::map<std::string, std::string>{
        {"load", load(1.0)},
        {"tide", tide(1.0)},
        {"seismic", R"({"fault": {"latitude": 35, "longitude": 45, "depth_km": 20,
            "strike_deg": 30, "dip_deg": 40, "rake_deg": 90, "length_km": 200,
            "width_km": 80, "moment_nm": 1e21}, "onset": 100})"}};
    if (!c.list.empty()) lists[c.list] += ", " + c.term;
    return Write(
        c.name + ".json",
        Config({std::format(R"("problem": "quasi_static",
  "model": {{"deck": "{}"}}, "discretisation": {{"lmax": 8}})",
                            c.deck),
                R"("times": {"unit": "yr", "at": [0, 100, 200, 400]})",
                std::format(R"("forcing": {{"load": [{}], "tide": [{}],
                   "seismic": [{}]}})",
                            lists["load"], lists["tide"], lists["seismic"]),
                R"("rotation": "full", "moments": "hydrostatic",
  "solver": {"rtol": 1e-12})",
                std::format(R"("kernels": {{"parameters": ["load", "tide",
      "moment", "viscosity", "moduli"],
    "measurements": [
      {{"type": "displacement_point", "name": "uplift", "latitude": 55,
       "longitude": 10, "time": 200}},
      {{"type": "stokes_coefficient", "name": "c20", "degree": 2,
       "order": 0}}],
    "output": "out/{}_kernels"{}}})",
                            c.name,
                            c.perturbation.empty() ? ""
                                                   : R"(, "second_order": true)"),
                c.perturbation.empty() ? std::string{}
                                       : R"("perturbation": )" + c.perturbation}));
  };
  struct Kind {
    std::string name;
    Case second, plus, minus;
    double step, tolerance;
  };
  const auto forcing = [&](const std::string& list,
                           const std::function<std::string(double)>& term) {
    return Kind{list,
                {.name = "s_" + list,
                 .perturbation = std::format(R"({{"forcing": {{"{}": [{}]}}}})",
                                             list, term(1.0))},
                {.name = "p_" + list, .list = list, .term = term(1.0)},
                {.name = "m_" + list, .list = list, .term = term(-1.0)},
                1.0,
                5e-10};
  };
  const auto kinds = std::vector<Kind>{
      forcing("load", load), forcing("tide", tide), forcing("seismic", source),
      {"viscosity",
       {.name = "s_viscosity",
        .perturbation = R"({"model": {"viscosity": {"file": "d_one.nc"}}})"},
       {.name = "p_viscosity", .deck = "up.deck"},
       {.name = "m_viscosity", .deck = "down.deck"},
       kStep,
       1e-6},
      {"moduli",
       {.name = "s_moduli",
        .perturbation = R"({"model": {"moduli": {"file": "d_moduli.nc"}}})"},
       {.name = "p_moduli", .deck = "stiff.deck"},
       {.name = "m_moduli", .deck = "soft.deck"},
       kStep,
       1e-6}};
  for (const auto& kind : kinds) {
    const auto summary = app::RunKernel(config(kind.second));
    EXPECT_NE(summary.find(", no ocean, "), std::string::npos) << summary;
    static_cast<void>(app::RunKernel(config(kind.plus)));
    static_cast<void>(app::RunKernel(config(kind.minus)));
    const auto agreement =
        FilesAgainst(KernelsOf(kind.second.name), KernelsOf(kind.plus.name),
                     KernelsOf(kind.minus.name), kind.step);
    Report("1-D, no ocean, an earthquake at a node, " + kind.name, agreement,
           kind.tolerance);
  }
  // the companions at the earthquake's node, of the kinds that jump there
  const auto load_ = pm::ReadSurfaceFile(KernelsOf("s_viscosity") /
                                         "uplift_load_kernels_second_order.nc");
  EXPECT_TRUE(load_.variables.contains("load_dual"));
  const auto moments =
      ReadTable(KernelsOf("s_viscosity") / "uplift_moment_kernels_second_order.txt");
  EXPECT_FALSE(moments.rows.empty());
}

TEST_F(SecondOrderTest, FromTheTrajectoryStore) {
  // The second-order kernels of a run read back from its trajectory store
  // (recorded every second node, the gaps rebuilt by the sweeps and the
  // tangent-linear run alike) are those of the run recorded in memory at
  // the same stride: the store holds the forward run alone, and serves any
  // direction.
  static_cast<void>(Write("ball.deck", BallDeck()));
  const auto model = ModelOf(dir_ / "ball.deck");
  WriteToyHistory(dir_ / "history.nc");
  WriteModelFields(dir_ / "d_one.nc", model,
                   {{"ln_viscosity", [](double, double, double) { return 1.0; }}},
                   true);
  const auto config = [&](const std::string& name, const std::string& store) {
    return Write(
        name + ".json",
        Config({R"("model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8})",
                kDated,
                R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})",
                kOcean, kTight,
                store.empty()
                    ? std::string{R"("trajectory": {"record_stride": 2})"}
                    : std::format(R"("trajectory": {{"directory": "{}",
    "record_stride": 2}})",
                                  store),
                KernelsBlock(name, R"("ice", "viscosity", "moduli")", true),
                R"("perturbation": {"model": {"viscosity": {"file": "d_one.nc"}}})"}));
  };
  static_cast<void>(app::RunKernel(config("memory", "")));
  const auto recorded = app::RunKernel(config("recorded", "store"));
  EXPECT_NE(recorded.find("recorded to the trajectory store"), std::string::npos)
      << recorded;
  const auto read = app::RunKernel(config("read", "store"));
  EXPECT_NE(read.find("read back from the trajectory store"), std::string::npos)
      << read;
  for (const auto* name : {"recorded", "read"}) {
    const auto departure =
        SecondOrderDeparture(KernelsOf(name), KernelsOf("memory"));
    std::cout << std::format(
        "[second order, the trajectory store, {}] {:.3e} from the run in "
        "memory\n",
        name, departure);
    EXPECT_LT(departure, 1e-12) << name;
  }
}
