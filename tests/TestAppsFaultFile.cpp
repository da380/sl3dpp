// The fault file as a seismic term of sl3d (`forcing.seismic[k].fault_file`):
// the twin of a planar fault, static and through time, with an ocean and
// without; the sign and geometry of a thrust; the quadrature's convergence
// on a curved patch; slip on a time axis against the same history given as
// a factor and against the file evaluated directly; a file served at its
// stored times alone, read at them through the run's units; the moment
// kernels at the quadrature's points; the refusals; and the run hash. The
// faults are written by planetmodel's own writer into each test's
// directory (FaultFixtures.hpp), on the homogeneous test ball. A build
// whose planetmodel cannot read fault files refuses the term.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <planetmodel/error.hpp>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "AppFixtures.hpp"
#include "Blocks.hpp"
#include "Commands.hpp"
#include "Common.hpp"
#include "FaultFixtures.hpp"
#include "Files.hpp"
#include "Forcing.hpp"
#include "RunHash.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

#include <planetmodel/faultfile.hpp>

namespace {

namespace fs = std::filesystem;
namespace pm = planetmodel;
using namespace sl3dpp;
using test::WriteToyHistory;

class FaultFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("sl3dpp_test_fault_file_" +
            std::string{::testing::UnitTest::GetInstance()
                            ->current_test_info()
                            ->name()});
    fs::remove_all(dir_);
    fs::create_directories(dir_);
    test::WriteFaultFixtures(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path Write(const std::string& name, const std::string& text) const {
    const auto path = dir_ / name;
    auto out = std::ofstream(path);
    out << text;
    return path;
  }

  /// The homogeneous test ball (GIA deck: r, rho, vpv, vsv, vph, vsh, eta,
  /// viscosity) of the given radius: mu = 5500^3 Pa, isotropic.
  fs::path Ball(const std::string& name = "ball.deck",
                double radius = 6371000.0) const {
    return Write(name, std::format("homogeneous test ball\n"
                                   "gia deck for the sl3d tests\n"
                                   "0 5500 10000 5500 10000 5500 1 1e21\n"
                                   "{:.1f} 5500 10000 5500 10000 5500 1 1e21\n",
                                   radius));
  }

  static std::string Data(const std::string& name) {
    return std::string{SL3DPP_TEST_DATA_DIR} + "/" + name;
  }

  /// A fault file of the test's own directory (FaultFixtures.hpp).
  std::string Fixture(const std::string& name) const {
    return (dir_ / name).string();
  }

  fs::path dir_;
};


/// The twin plane as a planar `fault` term, 4 by 2 patches.
constexpr auto kFault = R"("fault": {"latitude": -40, "longitude": 150,
    "depth_km": 30, "strike_deg": 30, "dip_deg": 15, "rake_deg": 90,
    "length_km": 600, "width_km": 200, "moment_nm": 1e22,
    "patches_along_strike": 4, "patches_down_dip": 2})";

/// A `fault_file` term of a fault file in the config's directory (the
/// test's), its quadrature `cells` and further members.
std::string FileTerm(const std::string& file, const std::string& cells = "4, 2",
                     const std::string& more = "") {
  const auto comma = cells.find(',');
  return std::format(R"("fault_file": {{"file": "{}",
      "cells_along_strike": {}, "cells_down_dip": {}{}}})",
                     file, cells.substr(0, comma), cells.substr(comma + 1),
                     more);
}

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

/// A field of a surface file at node `n` (static: 0).
std::vector<double> Node(const pm::SurfaceFile& file, const std::string& name,
                         std::size_t n = 0) {
  const auto& values = file.variables.at(name).values;
  const auto size = file.time ? values.size() / file.time->size() : values.size();
  return {values.begin() + static_cast<std::ptrdiff_t>(n * size),
          values.begin() + static_cast<std::ptrdiff_t>((n + 1) * size)};
}

/// A field on the grid of `lmax` at a point (colatitude and longitude,
/// radians), by its expansion.
double ValueAt(int lmax, const std::vector<double>& field, double theta,
               double phi) {
  const SurfaceGrid grid(lmax);
  const auto c = grid.Expand(field);
  const auto Y = app::HarmonicsAt(
      lmax, {90.0 - theta * 180.0 / std::numbers::pi, phi * 180.0 / std::numbers::pi});
  auto out = 0.0;
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto k = SurfaceGrid::Index(l, m);
      out += (m == 0 ? 1.0 : 2.0) * (c[k] * Y[k]).real();
    }
  }
  return out;
}

/// The point `distance` (m) along the great circle from (lat, lon) at
/// `azimuth` (degrees from north), as colatitude and longitude (radians),
/// on a sphere of radius 6371 km.
std::pair<double, double> Destination(double lat, double lon, double azimuth,
                                      double distance) {
  const auto d = distance / 6371e3;
  const auto p = lat * std::numbers::pi / 180.0, a = azimuth * std::numbers::pi / 180.0;
  const auto q = std::asin(std::sin(p) * std::cos(d) +
                           std::cos(p) * std::sin(d) * std::cos(a));
  const auto l = lon * std::numbers::pi / 180.0 +
                 std::atan2(std::sin(a) * std::sin(d) * std::cos(p),
                            std::cos(d) - std::sin(p) * std::sin(q));
  return {std::numbers::pi / 2.0 - q, l};
}

TEST_F(FaultFileTest, IsThePlanarFaultsTwin) {
  // The twin plane's file, its uniform thrust the slip of 1e22 N m on the
  // ball (mu = 5500^3 Pa), laid out on the planar fault's 4 by 2 cells with
  // one point each, is the planar `fault` term: every field the same to
  // rounding (1e-11 of its largest value; the sources' positions and
  // tensors agree to 1e-14, TestFault), in the static problem and through
  // time, without an ocean and with one. The summary counts it as a source
  // and the metadata describe it.
  static_cast<void>(Ball());
  WriteToyHistory(dir_ / "history.nc");
  const auto run = [&](const std::string& name, const std::string& setting,
                       const std::string& term, const std::string& more) {
    const auto config = Write(name + ".json", std::format(R"({{
      "model": {{"deck": "ball.deck"}}, "discretisation": {{"lmax": 8}},
      {}, "forcing": {{"seismic": [{{{}{}}}]}},
      "output": {{"surface": "out/{}.nc"}}}})",
                                                          setting, term, more, name));
    const auto summary = app::RunSolve(config);
    EXPECT_NE(summary.find("seismic (1 source)"), std::string::npos) << summary;
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto ocean = std::string{R"("ocean": {"date_kyr_bp": 2,
      "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic")"};
  const auto dated = std::string{R"("ocean": {"at": "start",
      "topography": {"file": "history.nc"}, "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic")"};
  struct Case {
    std::string name, setting, more;
  };
  for (const auto& [name, setting, more] : {
           Case{"static", R"("problem": "static", "rotation": "none")", ""},
           Case{"static_ocean", R"("problem": "static", )" + ocean, ""},
           Case{"forwards",
                R"("problem": "quasi_static", "rotation": "none",
                   "times": {"unit": "kyr", "at": [0, 0.4, 1.0]})",
                R"(, "onset": 0.4)"},
           Case{"dated_ocean",
                R"("problem": "quasi_static",
                   "times": {"at_kyr_bp": [2, 1.5, 1, 0]}, )" + dated,
                R"(, "onset_kyr_bp": 1.5, "history_kyr_bp": [[1.5, 1], [0, 2]])"}}) {
    const auto planar = run(name + "_planar", setting, kFault, more);
    const auto file = run(name + "_file", setting, FileTerm("fault_plane.nc"), more);
    const auto gap = Departure(file, planar);
    RecordProperty(name, std::format("{:.2e}", gap));
    EXPECT_LT(gap, 1e-11) << name;
    EXPECT_TRUE(planar.variables.contains("toroidal_potential")) << name;
    const auto described = std::get<std::string>(file.meta.at("seismic"));
    EXPECT_TRUE(described.starts_with("fault file " +
                                      Fixture("fault_plane.nc") +
                                      " (1 patch, static), 4 by 2 cells a "
                                      "patch, 1 Gauss point each way a cell"))
        << described;
  }
}

TEST_F(FaultFileTest, AThrustLiftsTheSurfaceOverItsUpDipPartTowardsTheTrench) {
  // The twin plane's thrust (strike 30, dipping 15 degrees to azimuth 120)
  // at lmax 48, its fields smoothed over 2 degrees: the surface rises over
  // the fault's up-dip half, more than it does 300 km down dip of the
  // centroid, and moves there towards the trench, up dip (azimuth 300).
  static_cast<void>(Ball());
  constexpr auto kLmax = 48;
  const auto config = Write("thrust.json", std::format(R"({{
      "problem": "static", "model": {{"deck": "ball.deck"}},
      "discretisation": {{"lmax": {}}}, "rotation": "none",
      "forcing": {{"seismic": [{{{}}}]}},
      "output": {{"surface": "out/thrust.nc", "fields": ["horizontal_displacement"],
                  "smoothing": {{"kind": "gaussian", "width_deg": 2.0}}}}}})",
                                                       kLmax,
                                                       FileTerm("fault_plane.nc", "8, 4")));
  static_cast<void>(app::RunSolve(config));
  const auto file = pm::ReadSurfaceFile(dir_ / "out" / "thrust.nc");
  const auto up = Node(file, "displacement_smoothed");
  const auto [ta, pa] = Destination(-40.0, 150.0, 300.0, 50e3);
  const auto [tb, pb] = Destination(-40.0, 150.0, 120.0, 300e3);
  const auto upDip = ValueAt(kLmax, up, ta, pa);
  const auto downDip = ValueAt(kLmax, up, tb, pb);
  RecordProperty("uplift_up_dip_m", std::format("{:.4f}", upDip));
  RecordProperty("uplift_down_dip_m", std::format("{:.4f}", downDip));
  EXPECT_GT(upDip, 0.0);
  EXPECT_GT(upDip, downDip);
  // the horizontal motion over the up-dip half, by the smoothed potentials'
  // gradients: east = d_phi V / sin(theta) - d_theta W, north = -d_theta V
  // - d_phi W / sin(theta)
  const auto V = Node(file, "tangential_potential_smoothed");
  const auto W = Node(file, "toroidal_potential_smoothed");
  const auto h = 1e-5;
  auto dTheta = [&, &ta = ta, &pa = pa](const std::vector<double>& f) {
    return (ValueAt(kLmax, f, ta + h, pa) - ValueAt(kLmax, f, ta - h, pa)) / (2 * h);
  };
  auto dPhi = [&, &ta = ta, &pa = pa](const std::vector<double>& f) {
    return (ValueAt(kLmax, f, ta, pa + h) - ValueAt(kLmax, f, ta, pa - h)) /
           (2 * h * std::sin(ta));
  };
  const auto east = dPhi(V) - dTheta(W);
  const auto north = -dTheta(V) - dPhi(W);
  const auto azimuth = std::atan2(east, north) * 180.0 / std::numbers::pi;
  const auto off = std::remainder(azimuth - 300.0, 360.0);
  RecordProperty("motion_azimuth_deg", std::format("{:.1f}", azimuth));
  EXPECT_LT(std::abs(off), 30.0) << azimuth;
  EXPECT_GT(std::hypot(east, north), 0.0);
}

TEST_F(FaultFileTest, ItsQuadratureConvergesOnACurvedPatch) {
  // The curved megathrust (a bicubic patch, its thrust tapered to nothing
  // at its edges) at lmax 32, its surface displacement smoothed over 2
  // degrees, on 4 by 2, 8 by 4, 16 by 8 and 32 by 16 cells of one point:
  // each refinement moves the fields by less than the one before, about a
  // quarter as much (the midpoint rule's second order), and the last by
  // less than 1e-3 of their size.
  static_cast<void>(Ball());
  auto fields = std::vector<std::vector<double>>{};
  for (const auto& cells : {"4, 2", "8, 4", "16, 8", "32, 16"}) {
    const auto name = std::format("curved_{}", fields.size());
    const auto config = Write(name + ".json", std::format(R"({{
        "problem": "static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 32}}, "rotation": "none",
        "forcing": {{"seismic": [{{{}}}]}},
        "output": {{"surface": "out/{}.nc", "fields": ["horizontal_displacement"],
                    "smoothing": {{"kind": "gaussian", "width_deg": 2.0}}}}}})",
                                                          FileTerm("fault_curved.nc", cells),
                                                          name));
    static_cast<void>(app::RunSolve(config));
    const auto file = pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
    auto joined = Node(file, "displacement_smoothed");
    for (const auto* other : {"displacement_east_smoothed", "displacement_north_smoothed"}) {
      const auto values = Node(file, other);
      joined.insert(joined.end(), values.begin(), values.end());
    }
    fields.push_back(std::move(joined));
  }
  auto size = 0.0;
  for (const auto v : fields.back()) size = std::max(size, std::abs(v));
  auto steps = std::vector<double>{};
  for (auto k = std::size_t{1}; k < fields.size(); ++k) {
    auto step = 0.0;
    for (auto i = std::size_t{0}; i < fields[k].size(); ++i) {
      step = std::max(step, std::abs(fields[k][i] - fields[k - 1][i]));
    }
    steps.push_back(step / size);
    RecordProperty(std::format("step_{}", k), std::format("{:.3e}", step / size));
  }
  for (auto k = std::size_t{1}; k < steps.size(); ++k) {
    EXPECT_LT(steps[k], 0.4 * steps[k - 1]) << k;
  }
  EXPECT_LT(steps.back(), 1e-3);
}

TEST_F(FaultFileTest, SlipOnATimeAxisIsItsHistory) {
  // The growing file (the twin's slip reached linearly from -2000 to 0 yr
  // on its own axis) is the static twin under the same history as a
  // factor, to rounding: on a run's own clock in years, its zero placed at
  // 2000 yr (`time_origin`) and held past its end (`outside_span`); and in
  // a dated run, read by date.
  static_cast<void>(Ball());
  const auto run = [&](const std::string& name, const std::string& times,
                       const std::string& term) {
    const auto config = Write(name + ".json", std::format(R"({{
        "problem": "quasi_static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 8}}, "rotation": "none", "times": {},
        "forcing": {{"seismic": [{{{}}}]}}, "output": {{"surface": "out/{}.nc"}}}})",
                                                          times, term, name));
    static_cast<void>(app::RunSolve(config));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto forwards = std::string{R"({"unit": "yr", "at": [0, 300, 1000, 1700, 2000, 2600]})"};
  const auto node = run("node", forwards,
                        FileTerm("fault_growing.nc", "4, 2",
                                 R"(, "time_origin": 2000, "outside_span": "hold")"));
  const auto factor = run("factor", forwards,
                          FileTerm("fault_plane.nc") +
                              R"(, "history": [[0, 0], [2000, 1]])");
  RecordProperty("forwards", std::format("{:.2e}", Departure(node, factor)));
  EXPECT_LT(Departure(node, factor), 1e-11);
  const auto dated = std::string{R"({"at_kyr_bp": [2.5, 2, 1.2, 0.4, 0]})"};
  const auto byDate = run("by_date", dated,
                          FileTerm("fault_growing.nc", "4, 2",
                                   R"(, "outside_span": "hold")"));
  const auto datedFactor = run("dated_factor", dated,
                               FileTerm("fault_plane.nc") +
                                   R"(, "history_kyr_bp": [[2, 0], [0, 1]])");
  RecordProperty("dated", std::format("{:.2e}", Departure(byDate, datedFactor)));
  EXPECT_LT(Departure(byDate, datedFactor), 1e-11);
  const auto described = std::get<std::string>(byDate.meta.at("seismic"));
  EXPECT_NE(described.find("(1 patch, slip at 2 times, linear)"), std::string::npos)
      << described;
  EXPECT_NE(described.find(", read by date"), std::string::npos) << described;
  // nothing moves before the slip begins: no slip, exactly zero
  auto before = 0.0;
  for (const auto v : Node(node, "displacement", 0)) before = std::max(before, std::abs(v));
  EXPECT_EQ(before, 0.0);
}

TEST_F(FaultFileTest, PreviousSlipJumpsAtItsStoredTimes) {
  // A file whose slip is read by `previous` (the twin's slip none from 0
  // yr, whole from 300 yr, doubled from 1000 yr, held after its span):
  // each stored time inside
  // the window is a jump of the forcing, the node repeated where one falls
  // there and a pair of the library's own between nodes. The run is the
  // static twin under the same factors as a history read by `previous`, to
  // rounding; it is the same at the window's nodes wherever they fall
  // against the stored times, where reading the slip at the nodes alone
  // spread each jump over the step that holds it (the factors linear
  // between the straddling nodes); the same slip read as `linear` is
  // another history.
  static_cast<void>(Ball());
  namespace f = test::fault_fixtures;
  const auto years = pm::Scales(
      {.length = 1.0, .mass = 1.0, .time = f::kYear, .temperature = 1.0});
  for (const auto& [name, rule] : {std::pair{"steps", pm::TimeRule::Previous},
                                   std::pair{"ramps", pm::TimeRule::Linear}}) {
    f::Write(dir_ / (std::string{"fault_"} + name + ".nc"),
             pm::Fault({f::Plane(f::RakeSlip(
                           {{90.0, 0.0}, {90.0, f::kSlip}, {90.0, 2.0 * f::kSlip}}))},
                       pm::TimeSupport::Node, rule,
                       std::vector<double>{0.0, 300.0, 1000.0}, years),
             "the twin plane, its slip stepping at 300 and 1000 yr");
  }
  const auto run = [&](const std::string& name, const std::string& at,
                       const std::string& term) {
    const auto config = Write(name + ".json", std::format(R"({{
        "problem": "quasi_static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 8}}, "rotation": "none",
        "times": {{"unit": "yr", "at": [{}]}},
        "forcing": {{"seismic": [{{{}}}]}}, "output": {{"surface": "out/{}.nc"}}}})",
                                                          at, term, name));
    static_cast<void>(app::RunSolve(config));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  // the largest departure of two runs' fields at pairs of nodes
  const auto departure = [](const pm::SurfaceFile& a, const pm::SurfaceFile& b,
                            const std::vector<std::pair<std::size_t, std::size_t>>& nodes) {
    auto out = 0.0;
    for (const auto& [i, j] : nodes) {
      for (const auto* name : {"displacement", "potential", "toroidal_potential"}) {
        const auto x = Node(a, name, i);
        const auto y = Node(b, name, j);
        auto diff = 0.0, scale = 0.0;
        for (auto k = std::size_t{0}; k < y.size(); ++k) {
          diff = std::max(diff, std::abs(x[k] - y[k]));
          scale = std::max(scale, std::abs(y[k]));
        }
        out = std::max(out, scale > 0.0 ? diff / scale : diff);
      }
    }
    return out;
  };
  const auto onNodes = std::string{"0, 100, 300, 600, 1000, 1500"};
  const auto straddling = std::string{"0, 100, 600, 1500"};
  const auto hold = std::string{R"(, "outside_span": "hold")"};
  const auto steps =
      run("steps", onNodes, FileTerm("fault_steps.nc", "4, 2", hold));
  const auto factor =
      run("factor", onNodes,
          FileTerm("fault_plane.nc") +
              R"(, "history": [[0, 0], [300, 1], [1000, 2]], "time_interpolation": "previous")");
  const auto straddled = run("straddled", straddling,
                             FileTerm("fault_steps.nc", "4, 2", hold));
  const auto spread =
      run("spread", straddling,
          FileTerm("fault_plane.nc") + R"(, "history": [[100, 0], [600, 1], [1500, 2]])");
  const auto ramps =
      run("ramps", onNodes, FileTerm("fault_ramps.nc", "4, 2", hold));
  ASSERT_EQ(steps.time->size(), 6u);
  ASSERT_EQ(straddled.time->size(), 4u);
  const auto every = std::vector<std::pair<std::size_t, std::size_t>>{
      {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
  const auto common = std::vector<std::pair<std::size_t, std::size_t>>{
      {1, 1}, {2, 3}, {3, 5}};
  const auto twin = departure(factor, steps, every);
  const auto after = departure(straddled, steps, common);
  const auto before = departure(spread, steps, common);
  const auto linear = departure(ramps, steps, {{2, 2}, {3, 3}, {4, 4}, {5, 5}});
  std::cout << std::format(
      "[previous slip] the twin's history {:.3e}; nodes straddling the stored "
      "times {:.3e}; spread over their steps {:.3e}; read as linear {:.3e}\n",
      twin, after, before, linear);
  // measured 0 (the twin), 0 (straddling), 0.13 spread, 1.5 linear
  RecordProperty("twin", std::format("{:.3e}", twin));
  RecordProperty("straddled", std::format("{:.3e}", after));
  RecordProperty("spread", std::format("{:.3e}", before));
  RecordProperty("linear", std::format("{:.3e}", linear));
  EXPECT_LT(twin, 1e-11);
  EXPECT_LT(after, 1e-10);
  EXPECT_GT(before, 1e-2);
  EXPECT_GT(linear, 1e-2);
  // nothing moves before the first slip
  for (const auto v : Node(steps, "displacement", 1)) EXPECT_EQ(v, 0.0);
}

TEST_F(FaultFileTest, StoredTimesAreReadThroughTheRunsUnits) {
  // A file served at its stored times alone (time rule `none`, its axis in
  // years: no slip at -1000 yr, the twin's at -999 yr) is read at them by a
  // run counting forwards in kyr with its zero at 120 kyr on the run's
  // clock, nodes 119 and 119.001: the second node reaches the file as
  // -998.99999999999511 yr, 4.9e-12 yr from its stored time, which a
  // tolerance on the 1 yr span alone would refuse, and the program's
  // tolerance on every file's time axis, 1e-9 of the larger of the span
  // and its ends' magnitudes (planetmodel's scale), reads as -999 yr. The
  // run is the static twin under the same history as a factor, to rounding.
  static_cast<void>(Ball());
  const auto run = [&](const std::string& name, const std::string& term) {
    const auto config = Write(name + ".json", std::format(R"({{
        "problem": "quasi_static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 8}}, "rotation": "none",
        "times": {{"unit": "kyr", "at": [119, 119.001]}},
        "forcing": {{"seismic": [{{{}}}]}}, "output": {{"surface": "out/{}.nc"}}}})",
                                                          term, name));
    static_cast<void>(app::RunSolve(config));
    return pm::ReadSurfaceFile(dir_ / "out" / (name + ".nc"));
  };
  const auto stored = run(
      "stored", FileTerm("fault_stored.nc", "4, 2", R"(, "time_origin": 120)"));
  const auto factor =
      run("factor", FileTerm("fault_plane.nc") +
                        R"(, "history": [[119, 0], [119.001, 1]])");
  RecordProperty("departure", std::format("{:.2e}", Departure(stored, factor)));
  EXPECT_LT(Departure(stored, factor), 1e-11);
  // the date the second node reaches the file at, against its axis
  const auto fault = pm::ReadFaultFile(Fixture("fault_stored.nc")).fault;
  const auto date = (119.001 - 120.0) * app::KYR / fault.GetScales().Time();
  EXPECT_NE(date, -999.0);
  // a tolerance on the axis's span alone, a year, refuses it
  EXPECT_THROW(static_cast<void>(fault.Rows(date, {.time_rtol = 1e-15})),
               pm::Error);
  const auto rows = fault.Rows(date, {.time_rtol = app::kAxisRtol});
  EXPECT_EQ(rows.first.first, 1u);
  EXPECT_EQ(rows.first.second, 1.0);
  EXPECT_EQ(rows.second.second, 0.0);
}

TEST_F(FaultFileTest, TurningSlipIsTheFileReadAtEachNode) {
  // The turning file (a thrust at 0 yr, a strike-slip of twice it at 10
  // yr, linear between): the term's sources at a run's nodes, its slices'
  // tensors weighted by their hat functions, are the moments of the file's
  // slip as planetmodel reads it at each node's time (Fault::Quadrature at
  // that date), with the ball's moduli: to rounding at every point, the
  // direction turning between the slices.
  const auto deck = Ball();
  const auto block = pm::ReadJson(std::format(R"({{"seismic": [{{{}}}]}})",
                                              FileTerm("fault_turning.nc", "3, 2")));
  const auto yr = app::TimeUnit{.name = "yr", .seconds = app::KYR / 1000.0};
  auto forcing = app::ReadForcing(
      block, dir_, 8,
      app::TermClock{.form = app::TermClock::Form::Forwards,
                     .start_kyr_bp = std::nullopt,
                     .unit = yr},
      false);
  const auto model =
      EarthModel::FromDeck(
          Deck::Read(deck, app::FormatOf(app::ModelFormat::Gia)).value())
          .value();
  const auto R = model.SurfaceRadius();
  const auto moduli = [&](const BodyPoint& at) { return LoveModuliAt(model, at.radius); };
  const auto times = std::vector<double>{0.0, 2.5, 5.0, 7.5, 10.0};
  app::PlaceFaultFiles(forcing.seismic, R, moduli, block, times);
  const auto fault = pm::ReadFaultFile(Fixture("fault_turning.nc")).fault;
  auto worst = 0.0;
  for (const auto t : times) {
    const auto sources = app::SeismicSourcesAt(forcing.seismic, t, R);
    const auto q = fault.Quadrature(3, 2, 1, t);
    ASSERT_EQ(sources.size(), q.Size());
    for (auto p = std::size_t{0}; p < q.Size(); ++p) {
      const auto& at = sources[p].at;
      const auto want =
          SlipMoment(moduli(at), at, q.slip[p], q.normals[p], q.weights[p]);
      const auto& got = sources[p].moment;
      const auto scale = std::max({std::abs(want.rr), std::abs(want.tt),
                                   std::abs(want.pp), std::abs(want.rt),
                                   std::abs(want.rp), std::abs(want.tp)});
      for (const auto& [a, b] : {std::pair{got.rr, want.rr}, std::pair{got.tt, want.tt},
                                 std::pair{got.pp, want.pp}, std::pair{got.rt, want.rt},
                                 std::pair{got.rp, want.rp}, std::pair{got.tp, want.tp}}) {
        worst = std::max(worst, std::abs(a - b) / scale);
      }
      EXPECT_NEAR(at.radius, std::hypot(q.points[p][0], q.points[p][1], q.points[p][2]),
                  1e-6);
    }
  }
  RecordProperty("worst", std::format("{:.2e}", worst));
  EXPECT_LT(worst, 1e-13);
  // the direction turns: at the middle of the strike (point 2, u = 0), a
  // thrust's M_rr at the start (sin 2 delta of its moment), a
  // strike-slip's none at the end
  const auto start = app::SeismicSourcesAt(forcing.seismic, 0.0, R);
  const auto end = app::SeismicSourcesAt(forcing.seismic, 10.0, R);
  EXPECT_GT(std::abs(start[2].moment.rr), 0.4 * std::abs(start[2].moment.rt));
  EXPECT_LT(std::abs(end[2].moment.rr), 1e-12 * std::abs(start[2].moment.rr));
}

TEST_F(FaultFileTest, ItsMomentKernelsAreAtItsQuadraturesPoints) {
  // `kernel` with the moment parameter: the table lists the fault file's
  // quadrature points, patch by patch, and its kernels and value are the
  // twin planar fault's, row for row, to rounding.
  static_cast<void>(Ball());
  const auto config = [&](const std::string& name, const std::string& term) {
    return Write(name + ".json", std::format(R"({{
        "problem": "static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 8}}, "rotation": "none",
        "forcing": {{"seismic": [{{{}}}]}},
        "output": {{"surface": "out/{}.nc"}},
        "kernels": {{"parameters": ["moment"],
          "measurements": [{{"type": "displacement_cap", "name": "uplift",
                             "latitude": -38, "longitude": 148, "width_deg": 10}}],
          "output": "out/{}_kernels"}}}})",
                                             term, name, name));
  };
  static_cast<void>(app::RunKernel(config("planar", kFault)));
  static_cast<void>(app::RunKernel(config("file", FileTerm("fault_plane.nc"))));
  const auto rows = [&](const std::string& name) {
    auto in = std::ifstream(dir_ / "out" / (name + "_kernels") / "uplift_moment_kernels.txt");
    auto points = std::vector<std::vector<double>>{};
    auto kernels = std::vector<std::vector<double>>{};
    for (std::string line; std::getline(in, line);) {
      const auto point = line.starts_with("# point ");
      if (line.starts_with("#") && !point) continue;
      auto words = std::istringstream(point ? line.substr(8) : line);
      auto row = std::vector<double>{};
      for (double v; words >> v;) row.push_back(v);
      (point ? points : kernels).push_back(std::move(row));
    }
    return std::pair{points, kernels};
  };
  const auto [planarPoints, planarKernels] = rows("planar");
  const auto [filePoints, fileKernels] = rows("file");
  ASSERT_EQ(filePoints.size(), 8u);
  ASSERT_EQ(planarPoints.size(), filePoints.size());
  ASSERT_EQ(planarKernels.size(), fileKernels.size());
  for (auto p = std::size_t{0}; p < filePoints.size(); ++p) {
    EXPECT_EQ(filePoints[p][2], static_cast<double>(p));  // the patch column
    for (auto c = std::size_t{3}; c < 6; ++c) {
      EXPECT_NEAR(filePoints[p][c], planarPoints[p][c],
                  1e-9 * std::max(1.0, std::abs(planarPoints[p][c])));
    }
  }
  auto scale = 0.0, gap = 0.0;
  for (auto r = std::size_t{0}; r < fileKernels.size(); ++r) {
    for (auto c = std::size_t{3}; c < fileKernels[r].size(); ++c) {
      scale = std::max(scale, std::abs(planarKernels[r][c]));
      gap = std::max(gap, std::abs(fileKernels[r][c] - planarKernels[r][c]));
    }
  }
  RecordProperty("gap", std::format("{:.2e}", gap / scale));
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(gap, 1e-10 * scale);
}

TEST_F(FaultFileTest, IsRefusedByName) {
  // Unknown members; a reference radius not the model's surface radius
  // without `place_by`, and with `"place_by": "radius"` points above the
  // surface (a ball 21 km smaller); slip on a time axis beside an onset or
  // in the static problem, read outside its span without `hold`, given a
  // time origin in a dated run; a time origin on a static file.
  static_cast<void>(Ball());
  static_cast<void>(Ball("small.deck", 6350000.0));
  const auto solve = [&](const std::string& name, const std::string& setting,
                         const std::string& term, const std::string& deck = "ball.deck") {
    return Write(name + ".json", std::format(R"({{
        "model": {{"deck": "{}"}}, "discretisation": {{"lmax": 8}},
        "rotation": "none", {}, "forcing": {{"seismic": [{{{}}}]}},
        "output": {{"surface": "out/{}.nc"}}}})",
                                             deck, setting, term, name));
  };
  const auto refused = [&](const fs::path& config, const std::string& member,
                           const std::string& what) {
    try {
      static_cast<void>(app::RunSolve(config));
      ADD_FAILURE() << config << " was not refused";
    } catch (const pm::Error& error) {
      const auto message = std::string{error.what()};
      EXPECT_NE(message.find(member), std::string::npos) << message;
      EXPECT_NE(message.find(what), std::string::npos) << message;
    }
  };
  const auto fixed = std::string{R"("problem": "static")"};
  const auto forwards =
      std::string{R"("problem": "quasi_static", "times": {"unit": "yr", "at": [0, 1000, 2500]})"};
  refused(solve("unknown", fixed, FileTerm("fault_plane.nc", "4, 2", R"(, "cells": 3)")),
          "forcing.seismic[0].fault_file.cells", "");
  refused(
      solve("missing", fixed, R"("fault_file": {"file": "fault_plane.nc"})"),
      "forcing.seismic[0].fault_file", "cells_along_strike");
  refused(solve("count", fixed, FileTerm("fault_plane.nc", "0, 2")),
          "forcing.seismic[0].fault_file.cells_along_strike", "is not a positive count");
  refused(solve("both", fixed, std::string{kFault} + ", " + FileTerm("fault_plane.nc")),
          "forcing.seismic[0]", "exactly one of \"point\", \"fault\" and \"fault_file\"");
  refused(solve("mismatch", fixed, FileTerm("fault_plane.nc"), "small.deck"),
          "forcing.seismic[0].fault_file", "\"place_by\": \"depth\"");
  refused(solve("above", fixed,
                FileTerm("fault_plane.nc", "4, 2", R"(, "place_by": "radius")"),
                "small.deck"),
          "forcing.seismic[0].fault_file", "above the model's surface");
  static_cast<void>(app::RunSolve(solve(
      "by_depth", fixed, FileTerm("fault_plane.nc", "4, 2", R"(, "place_by": "depth")"),
      "small.deck")));
  refused(solve("onset", forwards,
                FileTerm("fault_growing.nc", "4, 2", R"(, "time_origin": 2000)") +
                    R"(, "onset": 0)"),
          "forcing.seismic[0].onset", "slip is on a time axis of its own");
  refused(solve("node_static", fixed, FileTerm("fault_growing.nc")),
          "forcing.seismic[0].fault_file.file", "which the static problem does not read");
  refused(solve("outside", forwards,
                FileTerm("fault_growing.nc", "4, 2", R"(, "time_origin": 2000)")),
          "forcing.seismic[0].fault_file.file", "\"outside_span\": \"hold\"");
  refused(solve("origin_dated",
                R"("problem": "quasi_static", "times": {"at_kyr_bp": [2, 0]})",
                FileTerm("fault_growing.nc", "4, 2", R"(, "time_origin": 0)")),
          "forcing.seismic[0].fault_file.time_origin",
          "reads a fault file with a time axis by date");
  refused(solve("origin_static", forwards,
                FileTerm("fault_plane.nc", "4, 2", R"(, "time_origin": 0)")),
          "forcing.seismic[0].fault_file.time_origin", "drop it");
  refused(solve("absent", fixed,
                R"("fault_file": {"file": "absent.nc", "cells_along_strike": 1,
                    "cells_down_dip": 1})"),
          "forcing.seismic[0].fault_file.file", "absent.nc");
}

TEST_F(FaultFileTest, TakesTheLocalModuliOfALateralModel) {
  // On a moduli file that varies laterally the moduli at a point are the
  // deck's (transversely isotropic in prem8's upper mantle) plus the file's
  // relative perturbations there, delta mu = s mu and delta kappa = b kappa
  // against the deck's Voigt averages, an isotropic change; and the static
  // problem on that Earth runs under the fault file, 3-D.
  const auto deck = Data("prem8.elastic.deck");
  const auto model =
      EarthModel::FromDeck(
          Deck::Read(deck, app::FormatOf(app::ModelFormat::Elastic)).value())
          .value();
  ASSERT_TRUE(io::WriteModuli(
      dir_ / "moduli.nc", model, SurfaceGrid(8),
      [](double, double theta, double phi) {
        return 0.05 * std::sin(theta) * std::cos(phi);
      },
      [](double, double theta, double) { return 0.02 * std::cos(theta); },
      {.name = "toy moduli", .elementMetres = 500e3}));
  const auto config = Write("lateral.json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}", "format": "elastic", "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": 8}}, "rotation": "none",
      "forcing": {{"seismic": [{{{}}}]}},
      "output": {{"surface": "out/lateral.nc"}}}})",
                                                        deck,
                                                        FileTerm("fault_curved.nc", "4, 2",
                                                                 R"(, "place_by": "depth")")));
  const auto top = pm::ReadJsonFile(config);
  const auto input = app::ReadSolveInput(config);
  const auto earth = app::BuildEarth(input.run, top);
  ASSERT_TRUE(earth.LateralModuli());
  const auto at = BodyPoint{.radius = 6.25e6, .colatitude = 1.0, .longitude = 0.3};
  const auto local = app::ModuliOf(earth)(at);
  const auto radial = LoveModuliAt(earth.model, at.radius);
  const auto r = at.radius / earth.model.ModelScales().Length();
  const auto s = earth.moduli->Shear()(r, at.colatitude, at.longitude);
  const auto b = earth.moduli->Bulk()(r, at.colatitude, at.longitude);
  EXPECT_NEAR(s, 0.05 * std::sin(1.0) * std::cos(0.3), 5e-3);
  EXPECT_NE(radial.L, radial.N);
  const auto& [A, C, F, L, N] = radial;
  const auto mu = s * (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0;
  const auto kappa = b * (4.0 * A + C + 4.0 * F - 4.0 * N) / 9.0;
  EXPECT_NEAR(local.L - L, mu, 1e-12 * L);
  EXPECT_NEAR(local.N - N, mu, 1e-12 * L);
  EXPECT_NEAR(local.A - A, kappa + 4.0 * mu / 3.0, 1e-12 * A);
  EXPECT_NEAR(local.C - C, kappa + 4.0 * mu / 3.0, 1e-12 * A);
  EXPECT_NEAR(local.F - F, kappa - 2.0 * mu / 3.0, 1e-12 * A);
  const auto summary = app::RunSolve(config);
  EXPECT_TRUE(summary.starts_with("3-D (lateral moduli), elastic, static")) << summary;
  EXPECT_NE(summary.find("seismic (1 source)"), std::string::npos) << summary;
}

TEST_F(FaultFileTest, ItsRunHashesTheFilesContents) {
  // The trajectory store's hash of a run under a fault file holds the
  // file's contents, by its bytes (a copy elsewhere hashes alike, another
  // file does not), its quadrature and its placement.
  static_cast<void>(Ball());
  fs::copy_file(dir_ / "fault_plane.nc", dir_ / "copy.nc");
  const auto hash = [&](const std::string& name, const std::string& term) {
    const auto config = Write(name + ".json", std::format(R"({{
        "problem": "quasi_static", "model": {{"deck": "ball.deck"}},
        "discretisation": {{"lmax": 8}}, "rotation": "none",
        "times": {{"unit": "yr", "at": [0, 1]}},
        "forcing": {{"seismic": [{{{}}}]}}, "output": {{"surface": "out/{}.nc"}}}})",
                                                          term, name));
    return app::TrajectoryHashOf(app::ReadSolveInput(config).run, false);
  };
  const auto base = hash("base", FileTerm("fault_plane.nc"));
  EXPECT_EQ(hash("copy", R"("fault_file": {"file": "copy.nc", "cells_along_strike": 4,
                         "cells_down_dip": 2})"),
            base);
  EXPECT_NE(hash("other", FileTerm("fault_turning.nc")), base);
  EXPECT_NE(hash("cells", FileTerm("fault_plane.nc", "4, 3")), base);
  EXPECT_NE(hash("points", FileTerm("fault_plane.nc", "4, 2", R"(, "points": 2)")), base);
  EXPECT_NE(hash("placed", FileTerm("fault_plane.nc", "4, 2", R"(, "place_by": "depth")")),
            base);
  EXPECT_NE(hash("planar", kFault), base);
}


}  // namespace
