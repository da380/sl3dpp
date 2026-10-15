// Smoke tests: the dependency stack is wired up and behaves.

#include <gtest/gtest.h>

#include <GSHTrans/All>
#include <GaussQuad/All>
#include <cmath>
#include <numbers>

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
