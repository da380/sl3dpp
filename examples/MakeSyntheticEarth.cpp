/**
 * @file MakeSyntheticEarth.cpp
 * @brief Writes the examples' synthetic Earth: a planetmodel surface file
 * holding `ice_thickness` and `topography` over a glacial cycle, in the
 * form pyslfp's `export_ice_history` writes from ICE-5G/6G/7G, so that
 * every example and sl3d config runs on either.
 *
 * The planet is a toy: smooth continents on a 4 km deep ocean, and ice
 * sheets that grow nowhere and only melt. At 21 ka the Laurentide and
 * Fennoscandian sheets stand at their maximum and Greenland and
 * Antarctica are larger than today; each melts linearly over its own
 * window, the northern sheets entirely, the polar ones to their present
 * size. The ice-sheet profiles are parabolic (Vialov's to leading order).
 * The topography is the surface height, the ice included, as in the
 * ICE-xG files: so the sea level of any date (SeaLevelFromTopography) is
 * minus the fixed bedrock height on land.
 *
 * The build runs it, writing synthetic_earth.nc beside the examples:
 *
 *   MakeSyntheticEarth [output.nc]
 *
 * It needs no data and no network, which is what lets CI run the examples.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <planetmodel/surfacefile.hpp>
#include <string>
#include <vector>

#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Version.hpp"

namespace {

namespace pm = planetmodel;

constexpr auto kKyr = 1.0e3 * 365.25 * 86400.0;  // s

/// Great-circle distance in degrees from a point given by latitude and
/// longitude in degrees, at colatitude theta and longitude phi (radians).
double DistanceDeg(double theta, double phi, double lat, double lon) {
  const auto site = sl3dpp::GeographicPoint{lat, lon};
  const auto c = std::cos(theta) * std::cos(site.CoLatitude()) +
                 std::sin(theta) * std::sin(site.CoLatitude()) *
                     std::cos(phi - site.Longitude());
  return std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

/// A continent as a smooth blob: centre and e-folding radius in degrees.
struct Blob {
  double lat, lon, radius;
};

constexpr auto kContinents = std::array{
    Blob{50.0, 260.0, 22.0},   // North America
    Blob{25.0, 255.0, 10.0},   // Mexico
    Blob{-15.0, 300.0, 16.0},  // South America
    Blob{-38.0, 292.0, 9.0},   // Patagonia
    Blob{52.0, 20.0, 11.0},    // Europe
    Blob{65.0, 22.0, 7.0},     // Scandinavia
    Blob{55.0, 90.0, 26.0},    // northern Asia
    Blob{28.0, 85.0, 14.0},    // southern Asia
    Blob{5.0, 20.0, 22.0},     // Africa
    Blob{-25.0, 135.0, 11.0},  // Australia
    Blob{-90.0, 0.0, 20.0},    // Antarctica
    Blob{73.0, 318.0, 8.0},    // Greenland
};

/// The bedrock height, m: a 4 km deep ocean, continents 600 m high, the
/// coast a smooth step through the blobs' summed influence.
double Bedrock(double theta, double phi) {
  auto influence = 0.0;
  for (const auto& blob : kContinents) {
    const auto d = DistanceDeg(theta, phi, blob.lat, blob.lon) / blob.radius;
    influence += std::exp(-d * d);
  }
  const auto land = 0.5 * (1.0 + std::tanh((influence - 0.5) / 0.05));
  return -4000.0 + 4600.0 * land;
}

/// An ice sheet: its centre, and its radius (degrees) and central
/// thickness (m) at 21 ka and today, melting linearly between two dates.
struct Sheet {
  double lat, lon;
  double radiusMax, heightMax;
  double radiusNow, heightNow;
  double meltStartKyr, meltEndKyr;
};

constexpr auto kSheets = std::array{
    Sheet{60.0, 265.0, 22.0, 3000.0, 0.0, 0.0, 16.0, 7.0},     // Laurentide
    Sheet{64.0, 22.0, 11.0, 2500.0, 0.0, 0.0, 18.0, 10.0},     // Fennoscandian
    Sheet{72.0, 318.0, 10.0, 3000.0, 8.0, 2500.0, 15.0, 8.0},  // Greenland
    Sheet{-90.0, 0.0, 25.0, 3300.0, 22.0, 3000.0, 14.0, 6.0},  // Antarctica
};

/// The ice thickness, m, at a date in kyr before present.
double Ice(double theta, double phi, double dateKyr) {
  auto thickness = 0.0;
  for (const auto& sheet : kSheets) {
    // 1 before the melt, 0 after it, linear in between
    const auto glacial = std::clamp(
        (dateKyr - sheet.meltEndKyr) / (sheet.meltStartKyr - sheet.meltEndKyr),
        0.0, 1.0);
    const auto radius =
        sheet.radiusNow + glacial * (sheet.radiusMax - sheet.radiusNow);
    const auto height =
        sheet.heightNow + glacial * (sheet.heightMax - sheet.heightNow);
    if (radius <= 0.0 || height <= 0.0) continue;
    const auto d = DistanceDeg(theta, phi, sheet.lat, sheet.lon) / radius;
    if (d < 1.0)
      thickness = std::max(thickness, height * std::sqrt(1.0 - d * d));
  }
  return thickness;
}

}  // namespace

int main(int argc, char** argv) {
  const auto path = std::string{argc > 1 ? argv[1] : "synthetic_earth.nc"};
  const auto grid = pm::Equiangular(180, 360);  // 1 degree, cell centred

  auto dates = std::vector<double>{};  // kyr BP, oldest first
  for (auto d = 21; d >= 0; --d) dates.push_back(static_cast<double>(d));

  auto time = std::vector<double>{};
  auto ice = std::vector<double>{};
  auto topography = std::vector<double>{};
  for (const auto date : dates) {
    time.push_back(-date * kKyr);
    for (const auto theta : grid.Colatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        const auto thickness = Ice(theta, phi, date);
        const auto bedrock = Bedrock(theta, phi);
        ice.push_back(thickness);
        // the surface height, ice included, as the ICE-xG files give it
        topography.push_back(
            bedrock < 0.0 && thickness == 0.0 ? bedrock : bedrock + thickness);
      }
    }
  }

  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.time = time;
  file.source = std::string{"sl3dpp examples (MakeSyntheticEarth, sl3dpp "} +
                std::string{sl3dpp::Version()} + ")";
  file.title = "synthetic Earth: a toy glacial cycle from 21 ka";
  auto options = pm::SurfaceFieldOptions{};
  options.support = "node";
  options.time_interpolation = "linear";
  options.dimensions = pm::units::LENGTH;
  options.meaning = "ice thickness";
  file.variables.emplace("ice_thickness",
                         pm::SurfaceField("ice_thickness", ice, options));
  options.meaning = "surface height above sea level, ice included";
  file.variables.emplace("topography",
                         pm::SurfaceField("topography", topography, options));
  pm::WriteSurfaceFile(path, file);

  std::cout << "wrote " << path << " (" << dates.size()
            << " dates, 21 ka to the present, 1 degree)\n";
  return 0;
}
