#include "sl3dpp/Fault.hpp"

#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>

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

LoveModuli LoveModuliAt(const EarthModel& model, double r) {
  const auto rho = model.Density(r);
  auto isotropic = [](double kappa, double mu) {
    return LoveModuli{.A = kappa + 4.0 * mu / 3.0,
                      .C = kappa + 4.0 * mu / 3.0,
                      .F = kappa - 2.0 * mu / 3.0,
                      .L = mu,
                      .N = mu};
  };
  if (model.Has("kappa") && model.Has("mu")) {
    return isotropic(model.Evaluate("kappa", r), model.Evaluate("mu", r));
  }
  if (model.Has("vpv") && model.Has("vsv")) {
    const auto vpv = model.Evaluate("vpv", r);
    const auto vsv = model.Evaluate("vsv", r);
    const auto vph = model.Has("vph") ? model.Evaluate("vph", r) : vpv;
    const auto vsh = model.Has("vsh") ? model.Evaluate("vsh", r) : vsv;
    const auto eta = model.Has("eta") ? model.Evaluate("eta", r) : 1.0;
    auto out = LoveModuli{.A = rho * vph * vph,
                          .C = rho * vpv * vpv,
                          .F = 0.0,
                          .L = rho * vsv * vsv,
                          .N = rho * vsh * vsh};
    out.F = eta * (out.A - 2.0 * out.L);
    return out;
  }
  if (model.Has("vp") && model.Has("vs")) {
    const auto vp = model.Evaluate("vp", r);
    const auto vs = model.Evaluate("vs", r);
    const auto mu = rho * vs * vs;
    return isotropic(rho * vp * vp - 4.0 * mu / 3.0, mu);
  }
  throw std::invalid_argument(
      "the model holds no moduli: it needs kappa and mu, vpv and vsv (with "
      "optional vph, vsh, eta), or vp and vs");
}

SphericalTensor SlipMoment(const LoveModuli& moduli, const BodyPoint& at,
                           const std::array<double, 3>& slip,
                           const std::array<double, 3>& normal, double area) {
  const PolarBasis basis(at.colatitude, at.longitude);
  // the slip and the normal in the polar basis at the point
  const auto s = Vector{Dot(slip, basis.r), Dot(slip, basis.theta),
                        Dot(slip, basis.phi)};
  const auto n = Vector{Dot(normal, basis.r), Dot(normal, basis.theta),
                        Dot(normal, basis.phi)};
  // e = (s n + n s) / 2, indices 0 = r, 1 = theta, 2 = phi
  const auto e = [&](int i, int j) { return 0.5 * (s[i] * n[j] + s[j] * n[i]); };
  const auto& [A, C, F, L, N] = moduli;
  return {.rr = area * (F * (e(1, 1) + e(2, 2)) + C * e(0, 0)),
          .tt = area * (A * e(1, 1) + (A - 2.0 * N) * e(2, 2) + F * e(0, 0)),
          .pp = area * ((A - 2.0 * N) * e(1, 1) + A * e(2, 2) + F * e(0, 0)),
          .rt = area * 2.0 * L * e(0, 1),
          .rp = area * 2.0 * L * e(0, 2),
          .tp = area * 2.0 * N * e(1, 2)};
}

#ifdef SL3DPP_HAVE_FAULT_FILE

FaultSources FaultSourcesOf(const planetmodel::Fault& fault,
                            const FaultSourceOptions& options) {
  if (options.cellsAlongStrike < 1 || options.cellsDownDip < 1 ||
      options.points < 1) {
    throw std::invalid_argument(
        "a fault's quadrature has at least one cell each way and one point");
  }
  if (!(options.surfaceRadius > 0.0) || !options.moduli) {
    throw std::invalid_argument(
        "a fault's sources need the model's surface radius and its moduli");
  }
  const auto& frame = fault.Frame();
  if (frame.GetKind() != planetmodel::FaultFrame::Kind::Geographic) {
    throw std::invalid_argument(
        "the fault is in a Cartesian frame, a flat body; this body is a "
        "sphere, and takes a fault in the geographic frame");
  }
  const auto length = fault.GetScales().Length();
  const auto surface = options.surfaceRadius;
  const auto reference = frame.Reference() * length;
  const auto same = std::abs(reference - surface) <= kSameSurface * surface;
  if (!same && !options.placement) {
    throw std::invalid_argument(std::format(
        "the fault's reference radius, {:.10g} m, is not the model's surface "
        "radius, {:.10g} m: say how the fault is placed, by its points' "
        "depths below its reference surface or by their radii",
        reference, surface));
  }
  const auto byDepth = !same && *options.placement == FaultPlacement::Depth;
  const auto degrees = 180.0 / std::numbers::pi;
  // the geometry once, every slice's slip at its points
  const auto q =
      fault.SliceQuadrature(static_cast<std::size_t>(options.cellsAlongStrike),
                            static_cast<std::size_t>(options.cellsDownDip),
                            static_cast<std::size_t>(options.points));
  auto out = FaultSources{};
  out.slices.resize(q.NSlices());
  for (auto i = std::size_t{0}; i < q.Size(); ++i) {
    // the point, placed, and the moduli there
    const auto& x = q.points[i];
    const auto stored = std::sqrt(Dot(x, x)) * length;
    const auto radius = byDepth ? surface - (reference - stored) : stored;
    const auto at =
        BodyPoint{.radius = radius,
                  .colatitude = std::atan2(std::hypot(x[0], x[1]), x[2]),
                  .longitude = std::atan2(x[1], x[0])};
    if (radius > surface) {
      throw std::invalid_argument(std::format(
          "patch {}'s quadrature point at (u, v) = ({:g}, {:g}), latitude "
          "{:.6g}, longitude {:.6g}, lies {:.6g} m above the model's surface "
          "(its depth {:.6g} m): a fault lies inside the body",
          q.patch[i], q.u[i], q.v[i], 90.0 - at.colatitude * degrees,
          at.longitude * degrees, radius - surface, surface - radius));
    }
    const auto moduli = options.moduli(at);
    if (!(moduli.L > 0.0 && moduli.N > 0.0)) {
      throw std::invalid_argument(std::format(
          "patch {}'s quadrature point at depth {:.6g} m lies where the "
          "model has no shear modulus (a fluid), which holds no fault",
          q.patch[i], surface - radius));
    }
    out.points.push_back(at);
    out.depths.push_back(surface - radius);
    out.patch.push_back(q.patch[i]);
    for (auto slice = std::size_t{0}; slice < q.NSlices(); ++slice) {
      const auto& s = q.slip[slice][i];
      out.slices[slice].push_back(
          SlipMoment(moduli, at, {s[0] * length, s[1] * length, s[2] * length},
                     q.normals[i], q.weights[i] * length * length));
    }
  }
  return out;
}

std::vector<SphericalTensor> FaultSources::MomentsAt(
    const planetmodel::TimeRows& rows) const {
  const auto [first, second] = rows;
  if (first.first >= slices.size() || second.first >= slices.size()) {
    throw std::invalid_argument(
        std::format("the rows name slices {} and {} of a fault's {}",
                    first.first, second.first, slices.size()));
  }
  auto scaled = [](double w, const SphericalTensor& M) {
    return SphericalTensor{.rr = w * M.rr,
                           .tt = w * M.tt,
                           .pp = w * M.pp,
                           .rt = w * M.rt,
                           .rp = w * M.rp,
                           .tp = w * M.tp};
  };
  auto out = std::vector<SphericalTensor>{};
  out.reserve(points.size());
  for (auto p = std::size_t{0}; p < points.size(); ++p) {
    auto M = scaled(first.second, slices[first.first][p]);
    if (second.second != 0.0) {
      const auto S = scaled(second.second, slices[second.first][p]);
      M = {.rr = M.rr + S.rr,
           .tt = M.tt + S.tt,
           .pp = M.pp + S.pp,
           .rt = M.rt + S.rt,
           .rp = M.rp + S.rp,
           .tp = M.tp + S.tp};
    }
    out.push_back(M);
  }
  return out;
}

#endif

}  // namespace sl3dpp
