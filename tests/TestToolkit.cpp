// Smoke tests: the dependency stack is wired up and behaves.

#include <gtest/gtest.h>
#include <omp.h>

#include <GSHTrans/All>
#include <GaussQuad/All>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <planetmodel/version.hpp>
#include <span>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/detail/AndersonMixer.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"

TEST(Toolkit, Version) { EXPECT_EQ(sl3dpp::Version(), "0.1.0"); }

// planetmodel tracks main, so its version is whatever was fetched: what the
// library reports must be what a code linking it alongside sees.
TEST(Toolkit, PlanetModelVersion) {
  EXPECT_FALSE(sl3dpp::PlanetModelVersion().empty());
  EXPECT_EQ(sl3dpp::PlanetModelVersion(), planetmodel::Version());
}

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
  auto err = 0.0;
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

// The threaded task loop rethrows a task's exception after the loop (one
// leaving an OpenMP region would terminate), and the other tasks still run.
TEST(Toolkit, ParallelForRethrowsAfterTheLoop) {
  auto ran = std::vector<int>(16, 0);
  EXPECT_THROW(sl3dpp::detail::ParallelFor(16, true,
                                           [&](int i) {
                                             ran[i] = 1;
                                             if (i == 5) {
                                               throw std::runtime_error("5");
                                             }
                                           }),
               std::runtime_error);
  EXPECT_EQ(std::ranges::count(ran, 1), 16);
}

// The threaded mixer: at one thread bitwise the serial update; at four,
// its per-thread partial sums change only round-off, and the iteration
// still converges. A nonlinear contraction over a vector held as three
// segments, as the 3D step holds its coefficient blocks; the iterates are
// compared midway, before convergence can wash differences out.
TEST(Toolkit, AndersonMixerThreadedMatchesSerial) {
  constexpr auto n = std::size_t{20000};
  auto image = [](const std::vector<double>& x) {
    auto g = std::vector<double>(x.size());
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      const auto c = std::sin(0.37 * static_cast<double>(i));
      g[i] = 0.5 * std::sin(x[i]) + 0.3 * x[(i + 1) % x.size()] + c;
    }
    return g;
  };
  auto run = [&](bool parallel, int threads) {
    const auto saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    auto mixer = sl3dpp::detail::AndersonMixer(5, parallel);
    auto x = std::vector<double>(3 * n, 0.0);
    auto midway = std::vector<double>{};
    auto residual = 0.0;
    for (auto iteration = 0; iteration < 40; ++iteration) {
      if (iteration == 12) midway = x;
      const auto g = image(x);
      residual = 0.0;
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        residual = std::max(residual, std::abs(g[i] - x[i]));
      }
      const auto segments =
          std::array{std::span(x).subspan(0, n), std::span(x).subspan(n, n),
                     std::span(x).subspan(2 * n)};
      const auto images = std::array{std::span<const double>(g).subspan(0, n),
                                     std::span<const double>(g).subspan(n, n),
                                     std::span<const double>(g).subspan(2 * n)};
      mixer.UpdateInPlace(segments, images);
    }
    omp_set_num_threads(saved);
    return std::pair(midway, residual);
  };
  const auto [serial, serialResidual] = run(false, 4);
  const auto [one, oneResidual] = run(true, 1);
  const auto [four, fourResidual] = run(true, 4);
  EXPECT_LT(serialResidual, 1e-8);
  EXPECT_LT(fourResidual, 1e-8);
  auto worst = 0.0;
  for (auto i = std::size_t{0}; i < serial.size(); ++i) {
    ASSERT_EQ(one[i], serial[i]) << "at " << i;
    worst = std::max(worst, std::abs(four[i] - serial[i]));
  }
  EXPECT_LT(worst, 1e-10);
}
