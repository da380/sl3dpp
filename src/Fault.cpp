#include "sl3dpp/Fault.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace sl3dpp {

namespace {

using Vector = std::array<double, 3>;

Vector Combine(double a, const Vector& x, double b, const Vector& y, double c,
               const Vector& z) {
  return {a * x[0] + b * y[0] + c * z[0], a * x[1] + b * y[1] + c * z[1],
          a * x[2] + b * y[2] + c * z[2]};
}

double Dot(const Vector& a, const Vector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// The spherical polar unit vectors at (theta, phi), Cartesian.
struct PolarBasis {
  Vector r, theta, phi;
  PolarBasis(double colatitude, double longitude)
      : r{std::sin(colatitude) * std::cos(longitude),
          std::sin(colatitude) * std::sin(longitude), std::cos(colatitude)},
        theta{std::cos(colatitude) * std::cos(longitude),
              std::cos(colatitude) * std::sin(longitude),
              -std::sin(colatitude)},
        phi{-std::sin(longitude), std::cos(longitude), 0.0} {}
};

/// The fault's unit vectors at a point: along strike, down dip, the
/// normal (from the foot wall to the hanging wall) and the slip of the
/// hanging wall (Aki & Richards 2002, eq. 4.88, north-east-down there).
struct FaultFrame {
  Vector strike, downDip, normal, slip;
  FaultFrame(const PolarBasis& at, double strikeDeg, double dipDeg,
             double rakeDeg) {
    const auto s = Radians(strikeDeg), d = Radians(dipDeg), l = Radians(rakeDeg);
    const Vector north{-at.theta[0], -at.theta[1], -at.theta[2]};
    const auto& east = at.phi;
    const Vector down{-at.r[0], -at.r[1], -at.r[2]};
    strike = Combine(std::cos(s), north, std::sin(s), east, 0.0, down);
    // the horizontal direction to the right of the strike, then tipped
    const auto right = Combine(-std::sin(s), north, std::cos(s), east, 0.0, down);
    downDip = Combine(std::cos(d), right, 0.0, east, std::sin(d), down);
    normal = Combine(-std::sin(d), right, 0.0, east, std::cos(d), down);
    // the normal from the foot wall up into the hanging wall
    for (auto& v : normal) v = -v;
    // the hanging wall's slip: along strike by cos(rake), up dip by sin
    slip = Combine(std::cos(l), strike, -std::sin(l), downDip, 0.0, down);
  }
};

/// M0 (n u + u n) in the polar basis at a point.
SphericalTensor TensorAt(const PolarBasis& at, const Vector& n, const Vector& u,
                         double moment) {
  auto component = [&](const Vector& a, const Vector& b) {
    return moment * (Dot(a, n) * Dot(b, u) + Dot(a, u) * Dot(b, n));
  };
  return {.rr = component(at.r, at.r),
          .tt = component(at.theta, at.theta),
          .pp = component(at.phi, at.phi),
          .rt = component(at.r, at.theta),
          .rp = component(at.r, at.phi),
          .tp = component(at.theta, at.phi)};
}

}  // namespace

SphericalTensor DoubleCouple(double strike, double dip, double rake) {
  // any point serves: the components are in the point's own basis
  const PolarBasis at(1.0, 0.5);
  const FaultFrame frame(at, strike, dip, rake);
  return TensorAt(at, frame.normal, frame.slip, 1.0);
}

std::vector<PointSource> PlanarFaultSources(const PlanarFault& fault,
                                            double surfaceRadius,
                                            int alongStrike, int downDip) {
  if (alongStrike < 1 || downDip < 1) {
    throw std::invalid_argument("a fault has at least one patch each way");
  }
  if (!(surfaceRadius > 0.0) || !(fault.length >= 0.0) ||
      !(fault.width >= 0.0) || !(fault.dip >= 0.0 && fault.dip <= 90.0)) {
    throw std::invalid_argument(
        "a fault has a positive surface radius, a length and a width that "
        "are not negative, and a dip of 0 to 90 degrees");
  }
  const PolarBasis centre(fault.centroid.CoLatitude(),
                          fault.centroid.Longitude());
  const FaultFrame frame(centre, fault.strike, fault.dip, fault.rake);
  const auto radius = surfaceRadius - fault.depth;
  const Vector origin{radius * centre.r[0], radius * centre.r[1],
                      radius * centre.r[2]};
  const auto share = fault.moment / (alongStrike * downDip);
  auto out = std::vector<PointSource>{};
  for (auto i = 0; i < alongStrike; ++i) {
    const auto a = fault.length * ((i + 0.5) / alongStrike - 0.5);
    for (auto j = 0; j < downDip; ++j) {
      const auto b = fault.width * ((j + 0.5) / downDip - 0.5);
      const auto x = Combine(1.0, origin, a, frame.strike, b, frame.downDip);
      const auto r = std::sqrt(Dot(x, x));
      if (r > surfaceRadius) continue;  // above the surface: left out
      const auto colatitude = std::acos(x[2] / r);
      const auto longitude = std::atan2(x[1], x[0]);
      const PolarBasis here(colatitude, longitude);
      out.push_back({.at = {.radius = r,
                            .colatitude = colatitude,
                            .longitude = longitude},
                     .moment = TensorAt(here, frame.normal, frame.slip, share)});
    }
  }
  return out;
}

}  // namespace sl3dpp
