#pragma once

// Fixtures the sl3d app tests share (TestApps, and TestAppsDistributed
// under MPI).

#include <cmath>
#include <filesystem>
#include <format>
#include <optional>
#include <planetmodel/grid.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <planetmodel/vocabulary.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace sl3dpp::test {

/// A toy world for the evolution: land north, ocean south, a polar ice
/// cap shrinking linearly over the window (2 kyr BP to the present).
/// `bump_last` adds a smooth mid-latitude patch to the final date's ice
/// alone, for finite differences of the kernels.
/// With `marine`, the polar ice sits in a basin 500 m below sea level and
/// thins to a fifth by the present, so it floats off as it melts.
/// `fields` picks what is written: both, the ice history alone, or the
/// topography alone; `hole` puts a value that is not a number in the ice;
/// `names` are the variables' names in the file.
enum class ToyFields { Both, Ice, Topography };
struct ToyNames {
  std::string ice = "ice_thickness";
  std::string topography = "topography";
};
inline void WriteToyHistory(const std::filesystem::path& path,
                            double bump_last = 0.0, bool marine = false,
                            ToyFields fields = ToyFields::Both,
                            bool hole = false, const ToyNames& names = {}) {
  namespace pm = planetmodel;
  constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::vector<double>{2.0, 1.0, 0.0};  // kyr BP
  auto time = std::vector<double>{};
  for (const auto d : dates) time.push_back(-d * kyr);
  auto ice = std::vector<double>{};
  auto topo = std::vector<double>{};
  // melt from 2 to 1 kyr BP, then hold: the hold is the decay window (a
  // marine history melts on)
  const auto fractions = marine ? std::vector<double>{1.0, 0.6, 0.2}
                                : std::vector<double>{1.0, 0.7, 0.7};
  for (auto k = std::size_t{0}; k < dates.size(); ++k) {
    const auto fraction = fractions[k];
    const auto bump = k + 1 == dates.size() ? bump_last : 0.0;
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        topo.push_back(
            3000.0 * std::cos(theta) -
            (marine ? 3500.0 * std::exp(-std::pow(theta / 0.3, 2)) : 0.0));
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
  if (hole) ice[9 * 36 + 5] = std::nan("");
  if (fields != ToyFields::Topography) {
    file.variables.emplace(names.ice,
                           pm::SurfaceField(names.ice, ice, options));
  }
  if (fields != ToyFields::Ice) {
    file.variables.emplace(names.topography,
                           pm::SurfaceField(names.topography, topo, options));
  }
  pm::WriteSurfaceFile(path, file);
}

/// The toy history (WriteToyHistory's ice and topography, without a bump
/// or the marine basin) on a time axis of its own: `time`, its three
/// dates' places in stored units of `timeScale` seconds (the file's time
/// scale), and, with `land`, WriteToyLand's field beside them as a static
/// field.
inline void WriteToyHistoryOn(const std::filesystem::path& path,
                              const std::vector<double>& time,
                              double timeScale = 1.0, bool land = false) {
  namespace pm = planetmodel;
  const auto grid = pm::Equiangular(19, 36);
  const auto fractions = std::vector<double>{1.0, 0.7, 0.7};
  auto ice = std::vector<double>{};
  auto topo = std::vector<double>{};
  auto held = std::vector<double>{};
  for (const auto fraction : fractions) {
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        topo.push_back(3000.0 * std::cos(theta));
        ice.push_back(theta < 0.45 ? fraction * 1500.0 *
                                         std::exp(-std::pow(theta / 0.3, 2))
                                   : 0.0);
      }
    }
  }
  for (const auto theta : grid.Colatitudes()) {
    for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
      held.push_back(theta > 1.9 && theta < 2.5 ? 1.0 : 0.0);
    }
  }
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  file.scales = pm::Scales(pm::units::ScaleValues{.time = timeScale});
  file.time = time;
  auto options = pm::SurfaceFieldOptions{};
  options.support = "node";
  options.time_interpolation = "linear";
  options.dimensions = pm::units::LENGTH;
  file.variables.emplace("ice_thickness",
                         pm::SurfaceField("ice_thickness", ice, options));
  file.variables.emplace("topography",
                         pm::SurfaceField("topography", topo, options));
  if (land) file.variables.emplace("land", pm::SurfaceField("land", held, {}));
  pm::WriteSurfaceFile(path, file);
}

/// The toy history's ice thickness CHANGE from its first date (2 kyr BP),
/// `scale` times (WriteToyHistory's ice, without a bump or the marine
/// basin), as a file of one variable, `name`: on the history's own time
/// axis, the change at each date, or, without `timeAxis`, the change at
/// the present alone as a static field. The values are the differences of
/// the history's own, formed as it forms them.
inline void WriteToyChange(const std::filesystem::path& path, bool timeAxis,
                           double scale = 1.0,
                           const std::string& name = "ice_thickness_change") {
  namespace pm = planetmodel;
  constexpr auto kyr = 1.0e3 * 365.25 * 86400.0;
  const auto grid = pm::Equiangular(19, 36);
  const auto dates = std::vector<double>{2.0, 1.0, 0.0};  // kyr BP
  const auto fractions = std::vector<double>{1.0, 0.7, 0.7};
  auto time = std::vector<double>{};
  auto change = std::vector<double>{};
  for (auto k = std::size_t{0}; k < dates.size(); ++k) {
    if (!timeAxis && k + 1 < dates.size()) continue;
    time.push_back(-dates[k] * kyr);
    for (const auto theta : grid.Colatitudes()) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        const auto shape = std::exp(-std::pow(theta / 0.3, 2));
        const auto now = theta < 0.45 ? fractions[k] * 1500.0 * shape : 0.0;
        const auto then = theta < 0.45 ? fractions[0] * 1500.0 * shape : 0.0;
        change.push_back(scale * (now - then));
      }
    }
  }
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  auto options = pm::SurfaceFieldOptions{};
  options.dimensions = pm::units::LENGTH;
  if (timeAxis) {
    file.time = time;
    options.support = "node";
    options.time_interpolation = "linear";
  }
  file.variables.emplace(name, pm::SurfaceField(name, change, options));
  pm::WriteSurfaceFile(path, file);
}

/// A field held as land for the toy world, without a time axis: one over
/// a band of the southern ocean (colatitudes 1.9 to 2.5), zero elsewhere,
/// under the name given. `hole` puts a value that is not a number in it.
inline void WriteToyLand(const std::filesystem::path& path,
                         const std::string& name = "land", bool hole = false) {
  namespace pm = planetmodel;
  const auto grid = pm::Equiangular(19, 36);
  auto held = std::vector<double>{};
  for (const auto theta : grid.Colatitudes()) {
    for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
      held.push_back(theta > 1.9 && theta < 2.5 ? 1.0 : 0.0);
    }
  }
  if (hole) held[9 * 36 + 5] = std::nan("");
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "sl3dpp tests";
  file.variables.emplace(name, pm::SurfaceField(name, held, {}));
  pm::WriteSurfaceFile(path, file);
}

/// The spec a model file states for a field of a direction in the
/// viscosity, `ln_viscosity` or `ln_viscosity_branch_<b>`, which
/// planetmodel's vocabulary does not name: a change of the natural
/// logarithm of a viscosity, a dimensionless scalar. Nothing for another
/// name, whose spec is the vocabulary's.
inline std::optional<planetmodel::FieldSpec> LogViscositySpec(
    const std::string& name) {
  namespace pm = planetmodel;
  constexpr auto kBranch = std::string_view{"ln_viscosity_branch_"};
  if (name == "ln_viscosity") {
    return pm::FieldSpec{pm::SCALAR, pm::units::DIMENSIONLESS,
                         "change of the natural logarithm of the viscosity "
                         "of every relaxing branch alike"};
  }
  if (name.starts_with(kBranch)) {
    return pm::FieldSpec{
        pm::SCALAR, pm::units::DIMENSIONLESS,
        std::format("change of the natural logarithm of the viscosity of "
                    "relaxing branch {} alone",
                    name.substr(kBranch.size()))};
  }
  return std::nullopt;
}

}  // namespace sl3dpp::test
