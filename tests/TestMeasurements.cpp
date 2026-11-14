// Measurement duals and the decay-time estimator: a cap mean integrates
// to one and reads a band-limited field's mean over the cap exactly at
// every width, a cap's displacement measurement reads the displacement
// smoothed as a vector at the centre, the point dual reads a band-limited
// field exactly, the
// coefficient dual reads the real orthonormal pair, the rotation and
// field measurements carry their duals, a clean exponential is fitted
// exactly, the fit's weights ARE the datum's derivative (held to central
// differences, times pinned and free), and windows without a decay are
// refused.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <fstream>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GaussQuad/All>

#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace {

using namespace sl3dpp;

TEST(MeasurementDuals, CapMeanIntegratesToOneAndReadsAConstant) {
  const auto grid = SurfaceGrid(32);
  const auto a = 6.371e6;
  const auto dual = CapMeanDual(grid, a, {60.0, -85.0}, 0.1);
  const auto ones = std::vector<double>(grid.FieldSize(), 1.0);
  EXPECT_NEAR(PairedValue(grid, a, dual, ones), 1.0, 1e-12);
  // a constant field's cap mean is the constant
  auto field = std::vector<double>(grid.FieldSize(), -3.25);
  EXPECT_NEAR(PairedValue(grid, a, dual, field), -3.25, 1e-11);
}

TEST(MeasurementDuals, PointDualReadsABandLimitedFieldExactly) {
  const auto grid = SurfaceGrid(16);
  const auto a = 2.0;
  auto smooth = [](double theta, double phi) {
    return 1.0 + 2.0 * std::cos(theta) +
           0.7 * std::sin(theta) * std::sin(phi);  // degree one: band-limited
  };
  const auto field = grid.FromFunction(smooth);
  const GeographicPoint site{-32.05, 115.74};
  const auto dual = PointDual(grid, a, site);
  EXPECT_NEAR(PairedValue(grid, a, dual, field),
              smooth(site.CoLatitude(), site.Longitude()), 1e-10);
}

TEST(MeasurementDuals, PointDisplacementReadsADirection) {
  // The displacement u = U r + grad_1 V - r x grad_1 W at a site along a
  // direction: the measurement's three duals paired with band-limited U,
  // V and W read direction . u exactly.
  const auto grid = SurfaceGrid(16);
  const auto a = 2.0;
  auto f = [](double theta, double phi) {
    return std::cos(theta) + 0.7 * std::sin(theta) * std::sin(phi) +
           0.4 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };
  auto dTheta = [](double theta, double phi) {
    return -std::sin(theta) + 0.7 * std::cos(theta) * std::sin(phi) +
           0.8 * std::sin(theta) * std::cos(theta) * std::cos(2.0 * phi);
  };
  auto dPhiOverSin = [](double theta, double phi) {
    return 0.7 * std::cos(phi) - 0.8 * std::sin(theta) * std::sin(2.0 * phi);
  };
  const auto field = grid.FromFunction(f);
  for (const auto site : {GeographicPoint{-32.05, 115.74},
                          GeographicPoint{71.3, 203.1}}) {
    const auto theta = site.CoLatitude();
    const auto phi = site.Longitude();
    // the gradient's components: east (1 / sin) d_phi, north -d_theta
    const auto gradEast = dPhiOverSin(theta, phi);
    const auto gradNorth = -dTheta(theta, phi);
    for (const auto d : {SurfaceDirection{.east = 1.0},
                         SurfaceDirection{.north = 1.0},
                         SurfaceDirection{.up = 1.0},
                         SurfaceDirection{.east = 0.3, .north = -0.8, .up = 0.5}}) {
      const auto m = PointDisplacementMeasurement(grid, a, site, d);
      if (d.up != 0.0) {
        EXPECT_NEAR(PairedValue(grid, a, m.displacementDual, field),
                    d.up * f(theta, phi), 1e-10);
      } else {
        EXPECT_TRUE(m.displacementDual.empty());
      }
      if (d.east == 0.0 && d.north == 0.0) {
        EXPECT_TRUE(m.tangentialPotentialDual.empty());
        EXPECT_TRUE(m.toroidalPotentialDual.empty());
        continue;
      }
      // V = f: direction . grad_1 f
      EXPECT_NEAR(PairedValue(grid, a, m.tangentialPotentialDual, field),
                  d.east * gradEast + d.north * gradNorth, 1e-10);
      // W = f: -r x grad_1 f has the east part -d_theta f = north's
      // gradient, and the north part -(1 / sin) d_phi f; of f's degree-2
      // term alone, its degree-1 terms being a rigid rotation, which the
      // measurement leaves out
      const auto east2 = -0.8 * std::sin(theta) * std::sin(2.0 * phi);
      const auto north2 =
          -0.8 * std::sin(theta) * std::cos(theta) * std::cos(2.0 * phi);
      EXPECT_NEAR(PairedValue(grid, a, m.toroidalPotentialDual, field),
                  d.east * north2 - d.north * east2, 1e-10);
    }
  }
}

TEST(MeasurementDuals, CapDisplacementIsTheSmoothedVectorAtTheCentre) {
  // The displacement smoothed over a cap as a vector: band-limited U, V
  // and W of every degree, each weighted by the cap's degree weights (W in
  // the frame of no net rotation of the surface, as sl3d's output smooths
  // it), and the displacement field of the smoothed potentials; the cap
  // measurement read at a grid point, along each axis and along a mixed
  // direction, is that field's components there, to rounding, at widths
  // the grid resolves and one it does not, next to either pole and
  // between. Its upward part is the cap mean of U.
  constexpr auto lmax = 24;
  const auto grid = SurfaceGrid(lmax, {.nmax = 1});
  const auto a = 6.371e6;
  const auto fieldOf = [&](double phase) {
    auto c = grid.ZeroCoefficients();
    for (auto l = 0; l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto k = static_cast<double>(SurfaceGrid::Index(l, m));
        c[SurfaceGrid::Index(l, m)] = {
            std::sin(1.7 * k + phase), m == 0 ? 0.0 : std::cos(2.3 * k - phase)};
      }
    }
    return grid.Synthesise(c);
  };
  const auto U = fieldOf(0.3);
  const auto V = fieldOf(1.1);
  const auto W = fieldOf(-0.7);
  const auto smoothed = [&](std::span<const double> field,
                            const std::vector<double>& weights) {
    auto c = grid.Expand(field);
    for (auto l = 0; l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) c[SurfaceGrid::Index(l, m)] *= weights[l];
    }
    return grid.Synthesise(c);
  };
  const auto nphi = grid.Longitudes().size();
  const auto rows = grid.CoLatitudes().size();
  auto worst = 0.0;
  for (const auto widthDeg : {2.0, 7.0, 25.0}) {
    const auto width = widthDeg * std::numbers::pi / 180.0;
    const auto weights = CapMeanWeights(lmax, width);
    const auto u = DisplacementField(grid, smoothed(U, weights),
                                     smoothed(V, weights),
                                     smoothed(WithoutNetRotation(grid, W), weights));
    auto scale = 0.0;
    for (const auto* c : {&u.east, &u.north, &u.up}) {
      for (const auto v : *c) scale = std::max(scale, std::abs(v));
    }
    for (const auto [it, ip] :
         {std::pair{std::size_t{0}, nphi / 3}, std::pair{rows / 4, nphi / 7},
          std::pair{rows / 2, nphi * 3 / 4}, std::pair{rows - 1, nphi / 2}}) {
      const auto at = it * nphi + ip;
      const auto site = GeographicPoint::FromSpherical(grid.CoLatitudes()[it],
                                                       grid.Longitudes()[ip]);
      for (const auto d : {SurfaceDirection{.east = 1.0},
                           SurfaceDirection{.north = 1.0},
                           SurfaceDirection{.up = 1.0},
                           SurfaceDirection{.east = 0.3, .north = -0.8, .up = 0.5}}) {
        const auto m = CapDisplacementMeasurement(grid, a, site, width, d);
        auto read = 0.0;
        for (const auto& [dual, field] :
             {std::pair{&m.displacementDual, &U},
              std::pair{&m.tangentialPotentialDual, &V},
              std::pair{&m.toroidalPotentialDual, &W}}) {
          if (!dual->empty()) read += PairedValue(grid, a, *dual, *field);
        }
        const auto want =
            d.east * u.east[at] + d.north * u.north[at] + d.up * u.up[at];
        worst = std::max(worst, std::abs(read - want) / scale);
        // to rounding through the transforms (3e-14 at worst)
        EXPECT_NEAR(read, want, 1e-11 * scale)
            << widthDeg << " deg at " << site.latitude << ", "
            << site.longitude << " along " << d.east << ", " << d.north
            << ", " << d.up;
        if (d.up != 0.0) {
          auto cap = CapMeanDual(grid, a, site, width);
          for (auto& v : cap) v *= d.up;
          // the kernel sampled point by point (no transform), scaled as the
          // measurement scales it: bitwise
          EXPECT_EQ(m.displacementDual, cap);
        } else {
          EXPECT_TRUE(m.displacementDual.empty());
        }
      }
    }
  }
  std::cout << std::format(
      "[cap displacement] the smoothed field's components within {:.2e} of "
      "the largest\n",
      worst);
  EXPECT_THROW(static_cast<void>(CapDisplacementMeasurement(
                   grid, a, GeographicPoint{10.0, 20.0}, 0.0, {.east = 1.0})),
               std::invalid_argument);
}

TEST(MeasurementDuals, AVectorWeightReadsTheDisplacement) {
  // The displacement field of analytic V and W has the analytic
  // gradient's components, and a vector weight's duals pair with U, V
  // and W as the weight does with the displacement.
  const auto grid = SurfaceGrid(16, {.nmax = 1});
  const auto a = 2.0;
  auto f = [](double theta, double phi) {
    return std::cos(theta) + 0.7 * std::sin(theta) * std::sin(phi) +
           0.4 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };
  auto dTheta = [](double theta, double phi) {
    return -std::sin(theta) + 0.7 * std::cos(theta) * std::sin(phi) +
           0.8 * std::sin(theta) * std::cos(theta) * std::cos(2.0 * phi);
  };
  auto dPhiOverSin = [](double theta, double phi) {
    return 0.7 * std::cos(phi) - 0.8 * std::sin(theta) * std::sin(2.0 * phi);
  };
  const auto field = grid.FromFunction(f);
  const auto gradEast = grid.FromFunction(dPhiOverSin);
  const auto gradNorth = grid.FromFunction(
      [&](double theta, double phi) { return -dTheta(theta, phi); });
  auto worst = [](std::span<const double> x, std::span<const double> y,
                  double sign) {
    auto out = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      out = std::max(out, std::abs(x[i] - sign * y[i]));
    }
    return out;
  };
  // V = f: grad_1 f. W = f: -r x grad_1 f, east -d_theta f, north
  // -(1 / sin) d_phi f.
  const auto spheroidal = DisplacementField(grid, {}, field, {});
  EXPECT_LT(worst(spheroidal.east, gradEast, 1.0), 1e-10);
  EXPECT_LT(worst(spheroidal.north, gradNorth, 1.0), 1e-10);
  const auto toroidal = DisplacementField(grid, {}, {}, field);
  EXPECT_LT(worst(toroidal.east, gradNorth, 1.0), 1e-10);
  EXPECT_LT(worst(toroidal.north, gradEast, -1.0), 1e-10);

  // a weight with all three parts against a displacement with all three
  auto g = [](double theta, double phi) {
    return 0.5 - std::cos(theta) * std::cos(theta) +
           0.3 * std::sin(theta) * std::cos(phi);
  };
  const auto other = grid.FromFunction(g);
  auto weight = DisplacementField(grid, other, field, other);
  for (auto* c : {&weight.east, &weight.north, &weight.up}) {
    for (auto& v : *c) v /= a * a;
  }
  const auto U = grid.FromFunction(
      [](double theta, double phi) { return std::sin(theta) * std::sin(phi); });
  // V and W each with a part along the weight's own
  auto mixedV = other, mixedW = field;
  for (auto i = std::size_t{0}; i < mixedV.size(); ++i) {
    mixedV[i] += 0.6 * field[i];
    mixedW[i] -= 0.4 * other[i];
  }
  auto mixedU = U;
  for (auto i = std::size_t{0}; i < mixedU.size(); ++i) mixedU[i] += other[i];
  // the measurement reads the displacement without its rigid rotation
  const auto u = DisplacementField(grid, mixedU, mixedV,
                                   WithoutNetRotation(grid, mixedW));
  auto dot = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < dot.size(); ++i) {
    dot[i] = weight.east[i] * u.east[i] + weight.north[i] * u.north[i] +
             weight.up[i] * u.up[i];
  }
  const auto direct = a * a * grid.Integrate(dot);
  const auto m = DisplacementMeasurement(grid, weight);
  const auto paired = PairedValue(grid, a, m.displacementDual, mixedU) +
                      PairedValue(grid, a, m.tangentialPotentialDual, mixedV) +
                      PairedValue(grid, a, m.toroidalPotentialDual, mixedW);
  EXPECT_GT(std::abs(direct), 0.1);
  EXPECT_NEAR(paired, direct, 1e-10 * std::abs(direct));

  // and the point measurement is the weight concentrated at the site
  const GeographicPoint site{-32.05, 115.74};
  const auto point = PointDisplacementMeasurement(
      grid, a, site, {.east = 0.3, .north = -0.8, .up = 0.5});
  const auto there = PairedValue(grid, a, point.displacementDual, U) +
                     PairedValue(grid, a, point.tangentialPotentialDual, other) +
                     PairedValue(grid, a, point.toroidalPotentialDual, field);
  const auto theta = site.CoLatitude();
  const auto phi = site.Longitude();
  const auto gTheta = 2.0 * std::cos(theta) * std::sin(theta) +
                      0.3 * std::cos(theta) * std::cos(phi);
  const auto gPhiOverSin = -0.3 * std::sin(phi);
  // u = U r + grad_1 g - r x grad_1 f at the site, of f's degree-2 term
  const auto uEast =
      gPhiOverSin - 0.8 * std::sin(theta) * std::cos(theta) * std::cos(2.0 * phi);
  const auto uNorth = -gTheta + 0.8 * std::sin(theta) * std::sin(2.0 * phi);
  const auto uUp = std::sin(theta) * std::sin(phi);
  EXPECT_NEAR(there, 0.3 * uEast - 0.8 * uNorth + 0.5 * uUp, 1e-10);
}

TEST(MeasurementDuals, HorizontalMeasurementsDoNotSeeARigidRotation) {
  // A rigid rotation u = c x r is the toroidal potential c . r, found
  // again by NetRotation; the displacement measurements pair with it to
  // nothing, their duals of W having no degree 1.
  const auto grid = SurfaceGrid(16, {.nmax = 1});
  const auto a = 2.0;
  const std::array<double, 3> c{0.3, -0.7, 0.5};
  const auto rigid = RigidRotationPotential(grid, c);
  const auto found = NetRotation(grid, rigid);
  for (auto i = std::size_t{0}; i < 3; ++i) EXPECT_NEAR(found[i], c[i], 1e-12);

  // its displacement field is c x r
  const auto u = DisplacementField(grid, {}, {}, rigid);
  const auto nphi = grid.Longitudes().size();
  auto worst = 0.0;
  for (auto it = std::size_t{0}; it < grid.CoLatitudes().size(); ++it) {
    for (auto ip = std::size_t{0}; ip < nphi; ++ip) {
      const auto theta = grid.CoLatitudes()[it];
      const auto phi = grid.Longitudes()[ip];
      const std::array<double, 3> r{std::sin(theta) * std::cos(phi),
                                    std::sin(theta) * std::sin(phi),
                                    std::cos(theta)};
      const std::array<double, 3> cross{c[1] * r[2] - c[2] * r[1],
                                        c[2] * r[0] - c[0] * r[2],
                                        c[0] * r[1] - c[1] * r[0]};
      const std::array<double, 3> east{-std::sin(phi), std::cos(phi), 0.0};
      const std::array<double, 3> north{-std::cos(theta) * std::cos(phi),
                                        -std::cos(theta) * std::sin(phi),
                                        std::sin(theta)};
      auto e = 0.0, n = 0.0;
      for (auto i = std::size_t{0}; i < 3; ++i) {
        e += cross[i] * east[i];
        n += cross[i] * north[i];
      }
      worst = std::max({worst, std::abs(u.east[it * nphi + ip] - e),
                        std::abs(u.north[it * nphi + ip] - n)});
    }
  }
  EXPECT_LT(worst, 1e-12);

  // a field with a rotation in it, and the same field without
  const auto field = grid.FromFunction([](double theta, double phi) {
    return std::cos(theta) + 0.7 * std::sin(theta) * std::sin(phi) +
           0.4 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  });
  const auto still = WithoutNetRotation(grid, field);
  const auto none = NetRotation(grid, still);
  for (auto v : none) EXPECT_NEAR(v, 0.0, 1e-12);
  const GeographicPoint site{-32.05, 115.74};
  const auto point = PointDisplacementMeasurement(
      grid, a, site, {.east = 0.3, .north = -0.8});
  EXPECT_NEAR(PairedValue(grid, a, point.toroidalPotentialDual, rigid), 0.0,
              1e-11);
  const auto value = PairedValue(grid, a, point.toroidalPotentialDual, field);
  EXPECT_GT(std::abs(value), 0.1);
  EXPECT_NEAR(value, PairedValue(grid, a, point.toroidalPotentialDual, still),
              1e-11);
  auto weight = DisplacementField(grid, {}, field, field);
  const auto whole = DisplacementMeasurement(grid, weight);
  EXPECT_NEAR(PairedValue(grid, a, whole.toroidalPotentialDual, rigid), 0.0,
              1e-11);
}

TEST(MeasurementDuals, BaselinesAndAnglesAreTheGeometrysChanges) {
  // The change of a length and of an angle between sites, against the
  // sites moved by a small multiple of the displacement and the geometry
  // measured again. The displacement holds a rigid rotation (the degree
  // 1 of W) and the measurements, which leave it out, agree with the
  // geometry all the same: neither quantity sees it.
  const auto grid = SurfaceGrid(16, {.nmax = 1});
  const auto a = 2.0;
  auto f = [](double theta, double phi) {
    return std::cos(theta) + 0.7 * std::sin(theta) * std::sin(phi) +
           0.4 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };
  auto fTheta = [](double theta, double phi) {
    return -std::sin(theta) + 0.7 * std::cos(theta) * std::sin(phi) +
           0.8 * std::sin(theta) * std::cos(theta) * std::cos(2.0 * phi);
  };
  auto fPhiOverSin = [](double theta, double phi) {
    return 0.7 * std::cos(phi) - 0.8 * std::sin(theta) * std::sin(2.0 * phi);
  };
  auto g = [](double theta, double phi) {
    return 0.5 - std::cos(theta) * std::cos(theta) +
           0.3 * std::sin(theta) * std::cos(phi);
  };
  auto gTheta = [](double theta, double phi) {
    return 2.0 * std::cos(theta) * std::sin(theta) +
           0.3 * std::cos(theta) * std::cos(phi);
  };
  auto gPhiOverSin = [](double, double phi) { return -0.3 * std::sin(phi); };
  // u = U r + grad_1 V - r x grad_1 W with U = g, V = g, W = f
  const auto U = grid.FromFunction(g), V = U;
  const auto W = grid.FromFunction(f);
  using Vector = std::array<double, 3>;
  auto position = [&](GeographicPoint site, double step) {
    const auto theta = site.CoLatitude();
    const auto phi = site.Longitude();
    const Vector r{std::sin(theta) * std::cos(phi),
                   std::sin(theta) * std::sin(phi), std::cos(theta)};
    const Vector east{-std::sin(phi), std::cos(phi), 0.0};
    const Vector north{-std::cos(theta) * std::cos(phi),
                       -std::cos(theta) * std::sin(phi), std::sin(theta)};
    const auto uEast = gPhiOverSin(theta, phi) - fTheta(theta, phi);
    const auto uNorth = -gTheta(theta, phi) - fPhiOverSin(theta, phi);
    const auto uUp = g(theta, phi);
    auto x = Vector{};
    for (auto i = std::size_t{0}; i < 3; ++i) {
      x[i] = a * r[i] + step * (uEast * east[i] + uNorth * north[i] + uUp * r[i]);
    }
    return x;
  };
  auto paired = [&](const Measurement& m) {
    return PairedValue(grid, a, m.displacementDual, U) +
           PairedValue(grid, a, m.tangentialPotentialDual, V) +
           PairedValue(grid, a, m.toroidalPotentialDual, W);
  };
  const GeographicPoint A{-32.05, 115.74}, B{-12.4, 130.9}, C{-41.3, 174.8};
  const auto h = 1e-5;

  auto length = [&](double step) {
    const auto x = position(A, step), y = position(B, step);
    return std::hypot(y[0] - x[0], y[1] - x[1], y[2] - x[2]);
  };
  const auto baseline = paired(BaselineMeasurement(grid, a, A, B));
  EXPECT_GT(std::abs(baseline), 0.01);
  EXPECT_NEAR(baseline, (length(h) - length(-h)) / (2.0 * h), 1e-8);
  // the other way round, the same change
  EXPECT_NEAR(paired(BaselineMeasurement(grid, a, B, A)), baseline, 1e-11);

  auto angle = [&](double step) {
    const auto x = position(A, step), y = position(B, step),
               z = position(C, step);
    auto dot = 0.0, pp = 0.0, qq = 0.0;
    for (auto i = std::size_t{0}; i < 3; ++i) {
      dot += (y[i] - x[i]) * (z[i] - x[i]);
      pp += (y[i] - x[i]) * (y[i] - x[i]);
      qq += (z[i] - x[i]) * (z[i] - x[i]);
    }
    return std::acos(dot / std::sqrt(pp * qq));
  };
  const auto turned = paired(AngleMeasurement(grid, a, A, B, C));
  EXPECT_GT(std::abs(turned), 0.01);
  EXPECT_NEAR(turned, (angle(h) - angle(-h)) / (2.0 * h), 1e-8);

  EXPECT_THROW(static_cast<void>(BaselineMeasurement(grid, a, A, A)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(AngleMeasurement(grid, a, A, B, B)),
               std::invalid_argument);
}

TEST(MeasurementDuals, CoefficientDualReadsTheRealPair) {
  const auto grid = SurfaceGrid(8);
  const auto a = 3.0;
  // a field with one cosine and one sine coefficient set
  auto coefficients = grid.ZeroCoefficients();
  coefficients[SurfaceGrid::Index(3, 2)] =
      ComplexFromRealPair({0.8, -1.4}, 2);
  const auto field = grid.Synthesise(coefficients);
  EXPECT_NEAR(PairedValue(grid, a, CoefficientDual(grid, a, 3, 2), field), 0.8,
              1e-12);
  EXPECT_NEAR(PairedValue(grid, a, CoefficientDual(grid, a, 3, -2), field),
              -1.4, 1e-12);
  EXPECT_NEAR(PairedValue(grid, a, CoefficientDual(grid, a, 3, 1), field), 0.0,
              1e-12);
  EXPECT_THROW(static_cast<void>(CoefficientDual(grid, a, 9, 0)),
               std::invalid_argument);
}

TEST(MeasurementDuals, CapMeanWeightsAreTheCapMeanAboutEveryPoint) {
  // A field of every degree weighted by CapMeanWeights and synthesised is
  // its cap mean about every point: read at grid points and between them
  // (the band-limited point dual), it is CapMeanDual's pairing there, to
  // rounding, at a width the grid resolves (its weights below rounding by
  // degree lmax + 2) and at one it does not.
  constexpr auto lmax = 24;
  const auto grid = SurfaceGrid(lmax);
  const auto width = 20.0 * std::numbers::pi / 180.0;
  const auto weights = CapMeanWeights(lmax, width);
  ASSERT_EQ(weights.size(), std::size_t{lmax + 1});
  EXPECT_EQ(weights[0], 1.0);  // normalised by itself: exactly
  for (auto l = 1; l <= lmax; ++l) {
    EXPECT_GT(weights[l], 0.0);
    EXPECT_LT(weights[l], weights[l - 1]);
  }
  EXPECT_LT(CapMeanWeights(lmax + 2, width).back(), 1e-14);
  auto coefficients = grid.ZeroCoefficients();
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto k = static_cast<double>(SurfaceGrid::Index(l, m));
      coefficients[SurfaceGrid::Index(l, m)] = {
          std::sin(1.7 * k + 0.3), m == 0 ? 0.0 : std::cos(2.3 * k - 0.1)};
    }
  }
  const auto field = grid.Synthesise(coefficients);
  auto weighted = coefficients;
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) weighted[SurfaceGrid::Index(l, m)] *= weights[l];
  }
  const auto smoothed = grid.Synthesise(weighted);
  const auto scale = *std::ranges::max_element(
      smoothed, {}, [](double v) { return std::abs(v); });
  const auto a = 6.371e6;
  const auto nphi = grid.Longitudes().size();
  auto sites = std::vector<std::pair<GeographicPoint, std::size_t>>{};
  for (const auto [it, ip] : {std::pair{0, 0}, std::pair{7, 11}, std::pair{12, 30},
                              std::pair{24, 48}}) {
    sites.emplace_back(
        GeographicPoint::FromSpherical(grid.CoLatitudes()[it],
                                       grid.Longitudes()[ip]),
        static_cast<std::size_t>(it) * nphi + ip);
  }
  for (const auto site : {GeographicPoint{-32.05, 115.74},
                          GeographicPoint{71.3, 203.1}}) {
    sites.emplace_back(site, grid.FieldSize());
  }
  for (const auto& [site, at] : sites) {
    const auto measured =
        PairedValue(grid, a, CapMeanDual(grid, a, site, width), field);
    const auto value = at < grid.FieldSize()
                           ? smoothed[at]
                           : PairedValue(grid, a, PointDual(grid, a, site),
                                         smoothed);
    // (to 5e-15 here: the synthesis against the kernel's quadrature)
    EXPECT_NEAR(value, measured, 1e-12 * std::abs(scale))
        << site.latitude << " " << site.longitude;
  }
  // a cap the grid does not resolve: the same
  const auto narrow = 2.0 * std::numbers::pi / 180.0;
  const auto narrowWeights = CapMeanWeights(lmax, narrow);
  auto blurred = coefficients;
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      blurred[SurfaceGrid::Index(l, m)] *= narrowWeights[l];
    }
  }
  const auto site = sites.back().first;
  const auto gap = std::abs(
      PairedValue(grid, a, PointDual(grid, a, site), grid.Synthesise(blurred)) -
      PairedValue(grid, a, CapMeanDual(grid, a, site, narrow), field));
  EXPECT_LT(gap, 1e-13 * std::abs(scale));
  EXPECT_THROW(static_cast<void>(CapMeanWeights(lmax, 0.0)),
               std::invalid_argument);
}

namespace {

/// The mean over the Gaussian cap of `width` about `site` of the field
/// f(r) = sum_j (n_j . r)^k_j, by quadrature about the site: Gauss-Legendre
/// in the distance d over [0, min(pi, 12 width)] (the weight there below
/// e^-72 of its peak) and the trapezoid in the azimuth, exact for the
/// field's degree. Independent of the grids and of the degree weights.
double FineCapMean(GeographicPoint site, double width,
                   const std::vector<std::pair<std::array<double, 3>, int>>& terms,
                   int degree) {
  const auto theta0 = site.CoLatitude();
  const auto phi0 = site.Longitude();
  const auto s = std::array{std::sin(theta0) * std::cos(phi0),
                            std::sin(theta0) * std::sin(phi0), std::cos(theta0)};
  const auto e1 = std::array{std::cos(theta0) * std::cos(phi0),
                             std::cos(theta0) * std::sin(phi0), -std::sin(theta0)};
  const auto e2 = std::array{-std::sin(phi0), std::cos(phi0), 0.0};
  const auto reach = std::min(std::numbers::pi, 12.0 * width);
  const auto rule = GaussQuad::GaussLegendreQuadrature1D<double>(600);
  const auto azimuths = 2 * degree + 4;
  auto weighted = 0.0, total = 0.0;
  for (auto k = std::size_t{0}; k < rule.Points().size(); ++k) {
    const auto d = 0.5 * reach * (rule.Points()[k] + 1.0);
    const auto w = 0.5 * reach * rule.Weights()[k] * std::sin(d) *
                   std::exp(-0.5 * (d / width) * (d / width));
    auto ring = 0.0;
    for (auto j = 0; j < azimuths; ++j) {
      const auto alpha = 2.0 * std::numbers::pi * j / azimuths;
      auto r = std::array<double, 3>{};
      for (auto c = 0; c < 3; ++c) {
        r[c] = std::cos(d) * s[c] +
               std::sin(d) * (std::cos(alpha) * e1[c] + std::sin(alpha) * e2[c]);
      }
      for (const auto& [n, power] : terms) {
        ring += std::pow(n[0] * r[0] + n[1] * r[1] + n[2] * r[2], power);
      }
    }
    weighted += w * ring / azimuths;
    total += w;
  }
  return weighted / total;
}

}  // namespace

TEST(MeasurementDuals, CapMeanIsExactAtEveryWidth) {
  // The cap measurement of a band-limited field is its mean over the cap,
  // exactly, at every width: against a fine quadrature of the mean about
  // the site, independent of the grid and of the degree weights, and
  // against the field smoothed by the degree weights read at the site
  // (output.smoothing's), to rounding, at 2 degrees on grids that do not
  // resolve the cap (lmax 32 and 96) as at widths they do; and at widths
  // the grid resolves the dual is the Gaussian sampled at the grid's
  // points and normalised by its quadrature, to rounding.
  const auto a = 6.371e6;
  for (const auto lmax : {32, 96}) {
    const auto grid = SurfaceGrid(lmax);
    // a field of degree lmax: powers of r along three directions
    const auto terms = std::vector<std::pair<std::array<double, 3>, int>>{
        {{0.36, -0.48, 0.8}, lmax}, {{0.6, 0.0, -0.8}, lmax / 2}, {{0.0, 1.0, 0.0}, 3}};
    const auto field = grid.FromFunction([&](double theta, double phi) {
      const auto r = std::array{std::sin(theta) * std::cos(phi),
                                std::sin(theta) * std::sin(phi), std::cos(theta)};
      auto sum = 0.0;
      for (const auto& [n, power] : terms) {
        sum += std::pow(n[0] * r[0] + n[1] * r[1] + n[2] * r[2], power);
      }
      return sum;
    });
    const auto coefficients = grid.Expand(field);
    for (const auto widthDeg : {2.0, 6.0, 20.0, 40.0}) {
      const auto width = widthDeg * std::numbers::pi / 180.0;
      const auto weights = CapMeanWeights(lmax, width);
      auto weighted = coefficients;
      for (auto l = 0; l <= lmax; ++l) {
        for (auto m = 0; m <= l; ++m) weighted[SurfaceGrid::Index(l, m)] *= weights[l];
      }
      const auto smoothed = grid.Synthesise(weighted);
      const auto scale = *std::ranges::max_element(
          smoothed, {}, [](double v) { return std::abs(v); });
      for (const auto site :
           {GeographicPoint{40.0, 30.0}, GeographicPoint{-32.05, 115.74},
            GeographicPoint{89.0, 200.0}, GeographicPoint{0.3, 300.0}}) {
        const auto measured =
            PairedValue(grid, a, CapMeanDual(grid, a, site, width), field);
        const auto fine = FineCapMean(site, width, terms, lmax);
        const auto atSite = PairedValue(grid, a, PointDual(grid, a, site), smoothed);
        // to rounding (2e-14 and 5e-14 at worst, the second through the
        // transforms)
        EXPECT_NEAR(measured, fine, 1e-11 * std::abs(scale))
            << "lmax " << lmax << ", " << widthDeg << " degrees at "
            << site.latitude << ", " << site.longitude;
        EXPECT_NEAR(measured, atSite, 1e-11 * std::abs(scale))
            << "lmax " << lmax << ", " << widthDeg << " degrees at "
            << site.latitude << ", " << site.longitude;
      }
      // where the grid resolves the cap (its weights below rounding by
      // degree lmax + 2, and its weight at the antipode too), the dual is
      // the sampled Gaussian
      const auto resolved =
          CapMeanWeights(lmax + 2, width).back() < 1e-15 && widthDeg <= 20.0;
      if (!resolved) continue;
      const auto site = GeographicPoint{40.0, 30.0};
      auto sampled = grid.FromFunction([&](double theta, double phi) {
        const auto c = std::cos(theta) * std::cos(site.CoLatitude()) +
                       std::sin(theta) * std::sin(site.CoLatitude()) *
                           std::cos(phi - site.Longitude());
        const auto d = std::acos(std::clamp(c, -1.0, 1.0));
        return std::exp(-0.5 * (d / width) * (d / width));
      });
      const auto norm = a * a * grid.Integrate(sampled);
      for (auto& v : sampled) v /= norm;
      const auto dual = CapMeanDual(grid, a, site, width);
      const auto peak = *std::ranges::max_element(dual);
      auto worst = 0.0;
      for (auto i = std::size_t{0}; i < dual.size(); ++i) {
        worst = std::max(worst, std::abs(dual[i] - sampled[i]));
      }
      std::cout << std::format(
          "[lmax {}, {} degrees] the dual against the sampled Gaussian: {:.1e} of its "
          "peak\n",
          lmax, widthDeg, worst / peak);
      EXPECT_LT(worst, 1e-11 * peak) << "lmax " << lmax << ", " << widthDeg;
    }
  }
}

TEST(Measurements, CarryTheirDuals) {
  auto dual = std::vector<double>{1.0, 2.0};
  const auto sea = FieldMeasurement(FieldChannel::SeaLevel, dual);
  EXPECT_EQ(sea.relativeSeaLevelDual, dual);
  EXPECT_TRUE(sea.displacementDual.empty() && sea.potentialDual.empty());
  const auto phi = FieldMeasurement(FieldChannel::Potential, dual);
  EXPECT_EQ(phi.potentialDual, dual);
  const auto spin = RotationMeasurement(AngularVelocityComponent::Z);
  EXPECT_EQ(spin.angularVelocityDual, (std::array{0.0, 0.0, 1.0}));
  EXPECT_TRUE(spin.relativeSeaLevelDual.empty());
}

std::vector<double> Window(std::size_t n, double span) {
  auto out = std::vector<double>{};
  for (auto i = std::size_t{0}; i < n; ++i) {
    out.push_back(span * static_cast<double>(i) / static_cast<double>(n - 1));
  }
  return out;
}

std::vector<double> Series(const std::vector<double>& t, double a, double tau,
                           double b) {
  auto out = std::vector<double>{};
  for (const auto x : t) out.push_back(a * std::exp(-x / tau) + b);
  return out;
}

TEST(DecayTime, FitsACleanExponentialExactly) {
  const auto t = Window(12, 5.0);
  const auto y = Series(t, 40.0, 1.7, -3.0);
  const auto fit =
      FitDecayTime(t, y, {.time_uncertainty = 0.0, .value_uncertainty = 0.5});
  EXPECT_NEAR(fit.decay_time, 1.7, 1e-8);
  EXPECT_NEAR(fit.amplitude, 40.0, 1e-6);
  EXPECT_NEAR(fit.offset, -3.0, 1e-7);
  EXPECT_NEAR(fit.misfit, 0.0, 1e-12);
  EXPECT_EQ(fit.model_times, t);
  // freeing the times changes nothing on clean data
  const auto freed =
      FitDecayTime(t, y, {.time_uncertainty = 0.1, .value_uncertainty = 0.5});
  EXPECT_NEAR(freed.decay_time, 1.7, 1e-7);
}

void HoldWeightsToCentralDifferences(double time_uncertainty) {
  const auto t = Window(10, 4.0);
  // reproducibly noisy data, so the weights are non-trivial
  auto y = Series(t, 25.0, 1.1, 2.0);
  auto state = 123456789ull;
  for (auto& v : y) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    v += 0.4 * (static_cast<double>(state >> 11) / 9007199254740992.0 - 0.5);
  }
  const auto options = DecayTimeOptions{.time_uncertainty = time_uncertainty,
                                        .value_uncertainty = 0.3};
  const auto fit = FitDecayTime(t, y, options);
  const auto step = 1e-6 * 25.0;
  for (auto j = std::size_t{0}; j < t.size(); j += 3) {
    auto plus = y, minus = y;
    plus[j] += step;
    minus[j] -= step;
    const auto difference = (FitDecayTime(t, plus, options).decay_time -
                             FitDecayTime(t, minus, options).decay_time) /
                            (2.0 * step);
    EXPECT_NEAR(fit.weights[j], difference,
                1e-5 * std::max(1.0, std::abs(difference)))
        << "sample " << j << ", time uncertainty " << time_uncertainty;
  }
}

TEST(DecayTime, WeightsAreTheDatumsDerivativeWithPinnedTimes) {
  HoldWeightsToCentralDifferences(0.0);
}

TEST(DecayTime, WeightsAreTheDatumsDerivativeWithFreeTimes) {
  HoldWeightsToCentralDifferences(0.05);
}

TEST(DecayTime, RefusesWindowsWithoutADecay) {
  const auto t = Window(8, 3.0);
  auto growing = std::vector<double>{};
  for (const auto x : t) growing.push_back(1.0 + 2.0 * x);  // no decay
  EXPECT_THROW(static_cast<void>(FitDecayTime(
                   t, growing,
                   {.time_uncertainty = 0.0, .value_uncertainty = 0.5})),
               std::invalid_argument);
  const auto y = Series(t, 40.0, 1.7, -3.0);
  EXPECT_THROW(static_cast<void>(FitDecayTime(
                   t, y, {.time_uncertainty = 0.0, .value_uncertainty = 0.0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(FitDecayTime(
                   std::vector<double>{0.0, 1.0, 2.0},
                   std::vector<double>{3.0, 2.0, 1.0},
                   {.time_uncertainty = 0.0, .value_uncertainty = 0.5})),
               std::invalid_argument);
}

TEST(DecayTime, RefusesUncertaintiesOutOfScale) {
  // Only the ratio of the uncertainties shapes the fit, so one in the wrong
  // unit (years for kiloyears, metres for a non-dimensional series) is
  // refused where it leaves the fit meaningless, naming the likely slip.
  const auto t = Window(8, 3.0);
  const auto y = Series(t, 40.0, 1.7, -3.0);
  const auto span = t.back() - t.front();
  auto [low, high] = std::ranges::minmax(y);
  auto refused = [&](DecayTimeOptions options, const std::string& unit) {
    try {
      static_cast<void>(FitDecayTime(t, y, options));
      ADD_FAILURE() << "accepted " << options.time_uncertainty << ", "
                    << options.value_uncertainty;
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string{error.what()}.find(unit), std::string::npos)
          << error.what();
    }
  };
  refused({.time_uncertainty = 1000.0 * span / 8.0, .value_uncertainty = 0.5},
          "times' unit");
  refused({.time_uncertainty = 0.0, .value_uncertainty = 2.0 * (high - low)},
          "values' unit");
  // just inside both bounds the fit runs
  EXPECT_NO_THROW(static_cast<void>(
      FitDecayTime(t, y,
                   {.time_uncertainty = 0.01 * span,
                    .value_uncertainty = 0.01 * (high - low)})));
}

TEST(DecayTime, MatchesPyslfpOnTheFixture) {
  // tests/data/decay_time.txt, written by scripts/make_decay_fixtures.py
  // from pyslfp's fit_decay_time: the two estimators are the same
  // construction, so the C++ is held to the Python's numbers.
  auto in = std::ifstream(std::string(SL3DPP_TEST_DATA_DIR) +
                          "/decay_time.txt");
  ASSERT_TRUE(in) << "run scripts/make_decay_fixtures.py from pyslfp";
  auto line = std::string{};
  auto t = std::vector<double>{}, y = std::vector<double>{};
  auto options = DecayTimeOptions{};
  auto fit = DecayTimeFit{};
  auto have_fit = false;
  auto checked = 0;
  auto refit = [&]() {
    fit = FitDecayTime(t, y, options);
    have_fit = true;
  };
  while (std::getline(in, line)) {
    auto words = std::istringstream(line);
    auto what = std::string{};
    words >> what;
    if (what == "case") {
      auto name = std::string{};
      words >> name >> options.time_uncertainty >> options.value_uncertainty;
      t.clear();
      y.clear();
      have_fit = false;
    } else if (what == "sample") {
      auto a = 0.0, b = 0.0;
      words >> a >> b;
      t.push_back(a);
      y.push_back(b);
    } else {
      if (!have_fit) refit();
      auto index = std::size_t{0};
      auto want = 0.0;
      if (what == "weight") words >> index;
      words >> want;
      const auto got = what == "decay_time"  ? fit.decay_time
                       : what == "amplitude" ? fit.amplitude
                       : what == "offset"    ? fit.offset
                                             : fit.weights.at(index);
      EXPECT_NEAR(got, want, 1e-6 * std::max(1.0, std::abs(want))) << line;
      ++checked;
    }
  }
  EXPECT_GE(checked, 26);
}

TEST(DecayTime, MeasurementsCarryTheWeightedDual) {
  const auto t = Window(6, 5.0);
  const auto y = Series(t, 10.0, 2.0, 0.0);
  const auto nodes = std::vector<std::size_t>{4, 6, 8, 10, 12, 14};
  const auto dual = std::vector<double>{2.0, -1.0};
  auto fit = DecayTimeFit{};
  const auto measurements = DecayTimeMeasurements(
      nodes, t, y, FieldChannel::SeaLevel, dual,
      {.time_uncertainty = 0.0, .value_uncertainty = 0.5}, &fit);
  ASSERT_EQ(measurements.size(), nodes.size());
  EXPECT_NEAR(fit.decay_time, 2.0, 1e-7);
  for (auto j = std::size_t{0}; j < nodes.size(); ++j) {
    const auto& m = measurements.at(nodes[j]);
    ASSERT_EQ(m.relativeSeaLevelDual.size(), dual.size());
    // one product with the fit's own weight: to the last bits
    EXPECT_DOUBLE_EQ(m.relativeSeaLevelDual[0], 2.0 * fit.weights[j]);
    EXPECT_DOUBLE_EQ(m.relativeSeaLevelDual[1], -1.0 * fit.weights[j]);
    EXPECT_TRUE(m.displacementDual.empty());
  }
  const auto twice = std::vector<std::size_t>{1, 1, 2, 3, 4, 5};
  EXPECT_THROW(
      static_cast<void>(DecayTimeMeasurements(
          twice, t, y, FieldChannel::SeaLevel, dual,
          {.time_uncertainty = 0.0, .value_uncertainty = 0.5})),
      std::invalid_argument);
}

}  // namespace
