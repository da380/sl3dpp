// The io/ boundary for surface files: a file written in its own units on
// its own grid comes back on the solver grid in SI, at any time by the
// file's rules, held or refused outside its span; what cannot be read is
// an IoError.

#include "sl3dpp/io/FieldHistory.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <planetmodel/grid.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <stdexcept>
#include <vector>

#include "sl3dpp/SurfaceGrid.hpp"

namespace {

namespace pm = planetmodel;

// A smooth low-degree pattern, exactly representable at small lmax.
double Pattern(double theta, double phi) {
  return 2.0 + std::cos(theta) + 0.5 * std::sin(theta) * std::cos(phi);
}

// The test file: stored values are SI / 2 under a length scale of two,
// a time axis {0, 1} in units of 100 s, a node field growing linearly
// in time, and a dimensionless static field.
std::filesystem::path WriteFile(const std::filesystem::path& path) {
  const auto grid = pm::Equiangular(90, 180);
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.scales = pm::Scales({.length = 2.0, .time = 100.0});
  file.time = std::vector<double>{0.0, 1.0};
  file.source = "sl3dpp tests";
  auto slice = [&grid](double amplitude) {
    auto values = std::vector<double>{};
    for (const auto theta : grid.Colatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        values.push_back(amplitude * Pattern(theta, phi) / 2.0);
      }
    }
    return values;
  };
  auto thickness = slice(1.0);
  const auto later = slice(3.0);
  thickness.insert(thickness.end(), later.begin(), later.end());
  file.variables.emplace(
      "ice_thickness",
      pm::SurfaceField("ice_thickness", std::move(thickness),
                       {.support = "node", .time_interpolation = "linear",
                        .lateral_interpolation = "cubic",
                        .dimensions = pm::units::LENGTH}));
  file.variables.emplace(
      "mask", pm::SurfaceField("mask", slice(2.0),
                               {.lateral_interpolation = "cubic"}));
  pm::WriteSurfaceFile(path, file);
  return path;
}

class FieldHistoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    path_ = new std::filesystem::path(
        std::filesystem::temp_directory_path() /
        "sl3dpp_test_field_history.nc");
    WriteFile(*path_);
  }
  static void TearDownTestSuite() {
    std::filesystem::remove(*path_);
    delete path_;
    path_ = nullptr;
  }
  static const std::filesystem::path& Path() { return *path_; }

 private:
  static const std::filesystem::path* path_;
};

const std::filesystem::path* FieldHistoryTest::path_ = nullptr;

TEST_F(FieldHistoryTest, ResamplesToSiOnTheSolverGrid) {
  const auto grid = sl3dpp::SurfaceGrid(16);
  const auto read = sl3dpp::io::FieldHistory::Read(Path(), grid);
  ASSERT_TRUE(read) << read.error().message;
  const auto& history = *read;
  EXPECT_EQ(history.Names(),
            (std::vector<std::string>{"ice_thickness", "mask"}));
  EXPECT_TRUE(history.Has("mask") && !history.Has("missing"));
  EXPECT_EQ(history.Source(), "sl3dpp tests");
  ASSERT_TRUE(history.Span());
  EXPECT_EQ(*history.Span(), (std::pair{0.0, 100.0}));

  // the node field at the second node: SI (the stored half, doubled),
  // on the solver's nodes, to the cubic resampling's accuracy
  const auto at_end = history.Values("ice_thickness", 100.0);
  ASSERT_EQ(at_end.size(), grid.FieldSize());
  auto n = std::size_t{0};
  auto worst = 0.0;
  for (const auto theta : grid.CoLatitudes()) {
    for (const auto phi : grid.Longitudes()) {
      worst = std::max(worst,
                       std::abs(at_end[n++] - 3.0 * Pattern(theta, phi)));
    }
  }
  EXPECT_LT(worst, 1e-5);

  // linear in time between the nodes, exactly
  const auto at_half = history.Values("ice_thickness", 50.0);
  const auto at_start = history.Values("ice_thickness", 0.0);
  for (auto i = std::size_t{0}; i < at_half.size(); ++i) {
    EXPECT_DOUBLE_EQ(at_half[i], 0.5 * at_start[i] + 0.5 * at_end[i]);
  }

  // the static, dimensionless field takes any time and no factor of two
  const auto mask = history.Values("mask", -7.0e9);
  n = 0;
  worst = 0.0;
  for (const auto theta : grid.CoLatitudes()) {
    for (const auto phi : grid.Longitudes()) {
      worst = std::max(worst, std::abs(mask[n++] - Pattern(theta, phi)));
    }
  }
  EXPECT_LT(worst, 1e-5);

  // outside the span the ends are held by default
  EXPECT_EQ(history.Values("ice_thickness", 1.0e9), at_end);
  EXPECT_EQ(history.Values("ice_thickness", -1.0), at_start);
}

TEST_F(FieldHistoryTest, RefusesOutsideTheSpanWhenAsked) {
  const auto grid = sl3dpp::SurfaceGrid(8);
  const auto read = sl3dpp::io::FieldHistory::Read(
      Path(), grid, {.hold_outside = false});
  ASSERT_TRUE(read) << read.error().message;
  EXPECT_NO_THROW(static_cast<void>(read->Values("ice_thickness", 100.0)));
  EXPECT_THROW(static_cast<void>(read->Values("ice_thickness", 101.0)),
               std::runtime_error);
  EXPECT_THROW(static_cast<void>(read->Values("missing", 0.0)),
               std::runtime_error);
}

TEST(FieldHistory, RefusesWhatIsNotASurfaceFile) {
  const auto grid = sl3dpp::SurfaceGrid(4);
  const auto missing = sl3dpp::io::FieldHistory::Read(
      std::filesystem::temp_directory_path() / "sl3dpp_no_such_file.nc",
      grid);
  ASSERT_FALSE(missing);
  EXPECT_FALSE(missing.error().message.empty());
}

}  // namespace
