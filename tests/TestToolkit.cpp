// Smoke tests: the dependency stack is wired up and behaves.

#include <gtest/gtest.h>

#include <GSHTrans/All>
#include <GaussQuad/All>
#include <cmath>
#include <numbers>

#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Version.hpp"

TEST(Toolkit, Version) { EXPECT_EQ(sl3dpp::Version(), "0.1.0"); }

// GLL quadrature integrates a low-degree polynomial exactly.
TEST(Toolkit, GaussLobattoQuadrature) {
  auto q = GaussQuad::GaussLobattoLegendreQuadrature1D<double>(5);
  auto integral = q.Integrate([](double x) { return x * x; });
  EXPECT_NEAR(integral, 2.0 / 3.0, 1e-14);
}

// A scalar spherical harmonic transform round-trips on the grid.
TEST(Toolkit, SphericalHarmonicRoundTrip) {
  using namespace GSHTrans;
  constexpr auto lMax = 16;
  auto grid = GaussLegendreGrid<double, All, Single>(lMax, 0);
  auto f =
      SpinField<0, decltype(grid), RealValued>(grid, [](auto theta, auto phi) {
        return std::cos(theta) + std::sin(theta) * std::cos(phi);
      });
  auto g = Evaluate(Expand(f, lMax));
  double err = 0;
  for (auto [v1, v2] : std::views::zip(f, g)) {
    err = std::max(err, std::abs(v1 - v2));
  }
  EXPECT_LT(err, 1e-12);
}

// Geographic <-> spherical polar conversions round-trip and hit the
// landmarks.
TEST(Toolkit, GeographicConversions) {
  using namespace sl3dpp;
  EXPECT_DOUBLE_EQ(CoLatitudeOf(90.0), 0.0);
  EXPECT_NEAR(CoLatitudeOf(0.0), std::numbers::pi / 2.0, 1e-15);
  EXPECT_NEAR(CoLatitudeOf(-90.0), std::numbers::pi, 1e-15);
  EXPECT_DOUBLE_EQ(LatitudeOf(CoLatitudeOf(37.5)), 37.5);

  const GeographicPoint cambridge{52.2, 0.12};
  EXPECT_NEAR(cambridge.CoLatitude(), Radians(90.0 - 52.2), 1e-15);
  EXPECT_NEAR(cambridge.Longitude(), Radians(0.12), 1e-15);
  const auto back = GeographicPoint::FromSpherical(cambridge.CoLatitude(),
                                                   cambridge.Longitude());
  EXPECT_NEAR(back.latitude, cambridge.latitude, 1e-12);
  EXPECT_NEAR(back.longitude, cambridge.longitude, 1e-12);

  // A west longitude lands in [0, 360).
  const auto west = GeographicPoint::FromSpherical(1.0, Radians(-70.0));
  EXPECT_NEAR(west.longitude, 290.0, 1e-12);
}
