#pragma once

// The fault files the fault-file tests read, written at test time by
// planetmodel's own writer (planetmodel::WriteFaultFile), on the
// homogeneous test ball (radius 6371 km, rho 5500 kg m^-3, vs 5500 m/s, so
// mu = 5500^3 Pa):
//
//   fault_plane.nc    the twin of a planar `fault` term: a plane centred at
//                     40 S, 150 E, 30 km deep, strike 30, dip 15, 600 by
//                     200 km, a uniform thrust (rake 90) of the slip that
//                     gives it a moment of 1e22 N m on the ball,
//                     M0 / (mu L W); static;
//   fault_growing.nc  that plane, its slip on a time axis in years:
//                     cumulative slip 0 at -2000 yr and the twin's at
//                     0 yr, linear between;
//   fault_turning.nc  that plane, two slices whose slip turns: a thrust of
//                     the twin's slip at 0 yr, a strike-slip of twice it at
//                     10 yr, linear between;
//   fault_stored.nc   that plane, its slip served at its stored times alone
//                     (time rule `none`): 0 at -1000 yr and the twin's at
//                     -999 yr;
//   fault_curved.nc   a curved megathrust: a bicubic patch of 7 by 5
//                     control positions swept down from a small-circle
//                     trench (6 degrees from 30 S, 160 E, azimuths 250 to
//                     290 about it, dipping towards it) along the great
//                     circles to the pole, its dip steepening from 12
//                     degrees at the trench by a degree every 4 km to 50
//                     km deep, its thrust tapered to nothing at every edge
//                     (5 m at its middle), bicubic on 9 by 7 nodes; static.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/Fault.hpp"

#ifdef SL3DPP_HAVE_FAULT_FILE

#include <planetmodel/faultfile.hpp>

namespace sl3dpp::test {

namespace fault_fixtures {

namespace pm = planetmodel;

inline constexpr auto kRadius = 6371e3;                      // the ball's
inline constexpr auto kMu = 5500.0 * 5500.0 * 5500.0;        // its mu, Pa
inline constexpr auto kYear = 3.15576e7;                     // s
inline constexpr auto kSlip = 1e22 / (kMu * 600e3 * 200e3);  // m

/// Slip of `magnitude` at `rake` degrees in the fault plane, one value a
/// slice on one cell.
inline pm::FaultSlip RakeSlip(std::vector<std::pair<double, double>> slices) {
  auto values = std::vector<double>{};
  for (const auto& [rake, magnitude] : slices) {
    const auto lambda = rake * std::numbers::pi / 180.0;
    values.insert(values.end(), {magnitude * std::cos(lambda),
                                 magnitude * std::sin(lambda), 0.0});
  }
  const auto n = slices.size();
  return pm::FaultSlip(std::move(values), n, 1, 1, "cell");
}

/// The twin plane with a slip.
inline pm::FaultPatch Plane(pm::FaultSlip slip) {
  return pm::FaultPatch::Plane(
      pm::FaultFrame::Geographic(kRadius),
      pm::PlaneParameters{.centroid = {-40.0, 150.0, kRadius - 30e3},
                          .strike = 30.0,
                          .dip = 15.0,
                          .length = 600e3,
                          .width = 200e3},
      std::move(slip));
}

/// A fault written, titled, as the tests' own.
inline void Write(const std::filesystem::path& path, pm::Fault fault,
                  std::string title) {
  pm::WriteFaultFile(path, pm::FaultFile{.fault = std::move(fault),
                                         .source = "sl3dpp tests",
                                         .title = std::move(title)});
}

/// A unit vector of latitude and longitude, degrees.
inline std::array<double, 3> Direction(double latitude, double longitude) {
  const auto p = latitude * std::numbers::pi / 180.0;
  const auto l = longitude * std::numbers::pi / 180.0;
  return {std::cos(p) * std::cos(l), std::cos(p) * std::sin(l), std::sin(p)};
}

/// The latitude and longitude, degrees, of a unit vector.
inline std::pair<double, double> LatLon(const std::array<double, 3>& x) {
  const auto degrees = 180.0 / std::numbers::pi;
  return {std::asin(std::clamp(x[2], -1.0, 1.0)) * degrees,
          std::atan2(x[1], x[0]) * degrees};
}

/// The point at `angle` (radians) along the great circle from unit vector
/// `from` towards `to` (a unit vector, or a direction tangent at `from`).
inline std::array<double, 3> Along(const std::array<double, 3>& from,
                                   const std::array<double, 3>& to,
                                   double angle) {
  const auto c = from[0] * to[0] + from[1] * to[1] + from[2] * to[2];
  auto d = std::array<double, 3>{to[0] - c * from[0], to[1] - c * from[1],
                                 to[2] - c * from[2]};
  const auto n = std::hypot(d[0], d[1], d[2]);
  for (auto& v : d) v /= n;
  return {std::cos(angle) * from[0] + std::sin(angle) * d[0],
          std::cos(angle) * from[1] + std::sin(angle) * d[1],
          std::cos(angle) * from[2] + std::sin(angle) * d[2]};
}

/// The curved megathrust (file comment).
inline pm::FaultPatch Curved() {
  constexpr auto nu = 7, nv = 5;
  const auto degree = std::numbers::pi / 180.0;
  const auto pole = Direction(-30.0, 160.0);
  // the dip a + b z at depth z: geometric down dip, so that with w the
  // length down dip z(w) and the horizontal distance x(w) are closed forms
  const auto a = 12.0 * degree, b = degree / 4e3;
  const auto bottom = 50e3;
  const auto width =
      std::log(std::tan((a + b * bottom) / 2.0) / std::tan(a / 2.0)) / b;
  auto positions = std::vector<pm::Vector3>{};
  // north and east at the pole
  const auto north = Direction(-30.0 + 90.0, 160.0);
  const auto east =
      std::array<double, 3>{north[1] * pole[2] - north[2] * pole[1],
                            north[2] * pole[0] - north[0] * pole[2],
                            north[0] * pole[1] - north[1] * pole[0]};
  for (auto i = 0; i < nu; ++i) {
    // the trench point: 6 degrees from the pole at the azimuth about it
    const auto azimuth = (250.0 + 40.0 * i / (nu - 1)) * degree;
    const auto toward = std::array<double, 3>{
        std::cos(azimuth) * north[0] + std::sin(azimuth) * east[0],
        std::cos(azimuth) * north[1] + std::sin(azimuth) * east[1],
        std::cos(azimuth) * north[2] + std::sin(azimuth) * east[2]};
    const auto trench = Along(pole, toward, 6.0 * degree);
    for (auto j = 0; j < nv; ++j) {
      const auto w = width * j / (nv - 1);
      const auto z =
          (2.0 * std::atan(std::tan(a / 2.0) * std::exp(b * w)) - a) / b;
      const auto x = std::log(std::sin(a + b * z) / std::sin(a)) / b;
      const auto [lat, lon] = LatLon(Along(trench, pole, x / kRadius));
      positions.push_back({lat, lon, kRadius - z});
    }
  }
  // the thrust tapered to nothing at every edge, on 9 by 7 nodes
  constexpr auto su = 9, sv = 7;
  auto values = std::vector<double>{};
  for (auto i = 0; i < su; ++i) {
    for (auto j = 0; j < sv; ++j) {
      const auto u = -0.5 + static_cast<double>(i) / (su - 1);
      const auto v = -0.5 + static_cast<double>(j) / (sv - 1);
      const auto m = 5.0 * std::pow(std::cos(std::numbers::pi * u), 2) *
                     std::pow(std::cos(std::numbers::pi * v), 2);
      values.insert(values.end(), {m * std::cos(90.0 * degree),
                                   m * std::sin(90.0 * degree), 0.0});
    }
  }
  return pm::FaultPatch::Structured(
      pm::FaultFrame::Geographic(kRadius),
      pm::ControlGrid{.n_u = nu,
                      .n_v = nv,
                      .positions = std::move(positions),
                      .interpolation = "cubic"},
      pm::FaultSlip(std::move(values), 1, su, sv, "cubic"));
}

}  // namespace fault_fixtures

/// Writes the fault files of the file comment into `directory`.
inline void WriteFaultFixtures(const std::filesystem::path& directory) {
  namespace f = fault_fixtures;
  namespace pm = planetmodel;
  const auto years = pm::Scales(
      {.length = 1.0, .mass = 1.0, .time = f::kYear, .temperature = 1.0});
  f::Write(directory / "fault_plane.nc",
           pm::Fault({f::Plane(f::RakeSlip({{90.0, f::kSlip}}))}),
           "the twin of a planar fault of 1e22 N m on the test ball");
  f::Write(directory / "fault_growing.nc",
           pm::Fault({f::Plane(f::RakeSlip({{90.0, 0.0}, {90.0, f::kSlip}}))},
                     pm::TimeSupport::Node, pm::TimeRule::Linear,
                     std::vector<double>{-2000.0, 0.0}, years),
           "the twin plane, its slip growing linearly from -2000 to 0 yr");
  f::Write(
      directory / "fault_turning.nc",
      pm::Fault(
          {f::Plane(f::RakeSlip({{90.0, f::kSlip}, {0.0, 2.0 * f::kSlip}}))},
          pm::TimeSupport::Node, pm::TimeRule::Linear,
          std::vector<double>{0.0, 10.0}, years),
      "the twin plane, a thrust at 0 yr turning to a strike-slip by 10 yr");
  f::Write(directory / "fault_stored.nc",
           pm::Fault({f::Plane(f::RakeSlip({{90.0, 0.0}, {90.0, f::kSlip}}))},
                     pm::TimeSupport::Node, pm::TimeRule::None,
                     std::vector<double>{-1000.0, -999.0}, years),
           "the twin plane, its slip at -1000 and -999 yr alone");
  f::Write(directory / "fault_curved.nc", pm::Fault({f::Curved()}),
           "a curved megathrust with a tapered thrust");
}

}  // namespace sl3dpp::test

#endif
