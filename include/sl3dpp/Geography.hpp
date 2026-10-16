#pragma once

/**
 * @file Geography.hpp
 * @brief Conversions between geographic coordinates (latitude and
 * longitude in degrees, the data-facing convention) and the library's
 * spherical polar coordinates (colatitude theta and longitude phi in
 * radians, the SurfaceGrid convention), plus a small value type carrying
 * one position across that boundary.
 *
 * Conventions: latitude in [-90, 90] degrees north-positive; longitude
 * in degrees east-positive, any branch (returned in [0, 360) from
 * radians); theta = pi/2 - latitude in radians; phi = longitude in
 * radians.
 */

#include <cmath>
#include <numbers>

namespace sl3dpp {

[[nodiscard]] constexpr double Radians(double degrees) {
  return degrees * std::numbers::pi / 180.0;
}

[[nodiscard]] constexpr double Degrees(double radians) {
  return radians * 180.0 / std::numbers::pi;
}

/// The colatitude (radians) of a latitude in degrees.
[[nodiscard]] constexpr double CoLatitudeOf(double latitudeDegrees) {
  return std::numbers::pi / 2.0 - Radians(latitudeDegrees);
}

/// The latitude (degrees) of a colatitude in radians.
[[nodiscard]] constexpr double LatitudeOf(double colatitudeRadians) {
  return Degrees(std::numbers::pi / 2.0 - colatitudeRadians);
}

/// One position on the sphere, held geographically (degrees) and
/// convertible to and from the grid's spherical polar radians.
struct GeographicPoint {
  double latitude = 0.0;   ///< degrees, north positive
  double longitude = 0.0;  ///< degrees, east positive

  [[nodiscard]] static GeographicPoint FromSpherical(double theta,
                                                     double phi) {
    double lon = std::fmod(Degrees(phi), 360.0);
    if (lon < 0.0) lon += 360.0;
    return {LatitudeOf(theta), lon};
  }

  [[nodiscard]] constexpr double CoLatitude() const {
    return CoLatitudeOf(latitude);
  }
  [[nodiscard]] constexpr double Longitude() const {
    return Radians(longitude);
  }
};

}  // namespace sl3dpp
