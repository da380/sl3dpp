#pragma once

// Fixtures the sl3d app tests share (TestApps, and TestAppsDistributed
// under MPI).

#include <cmath>
#include <filesystem>
#include <planetmodel/grid.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <vector>

namespace sl3dpp::test {

/// A toy world for the evolution: land north, ocean south, a polar ice
/// cap shrinking linearly over the window (2 kyr BP to the present).
/// `bump_last` adds a smooth mid-latitude patch to the final date's ice
/// alone, for finite differences of the kernels.
/// With `marine`, the polar ice sits in a basin 500 m below sea level and
/// thins to a fifth by the present, so it floats off as it melts.
inline void WriteToyHistory(const std::filesystem::path& path,
                            double bump_last = 0.0, bool marine = false) {
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
  file.variables.emplace("ice_thickness",
                         pm::SurfaceField("ice_thickness", ice, options));
  file.variables.emplace("topography",
                         pm::SurfaceField("topography", topo, options));
  pm::WriteSurfaceFile(path, file);
}

}  // namespace sl3dpp::test
