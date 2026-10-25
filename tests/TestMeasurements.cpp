// Measurement duals and the decay-time estimator: a cap mean integrates
// to one, the point dual reads a band-limited field exactly, the
// coefficient dual reads the real orthonormal pair, the rotation and
// field measurements carry their duals, a clean exponential is fitted
// exactly, the fit's weights ARE the datum's derivative (held to central
// differences, times pinned and free), and windows without a decay are
// refused.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

TEST(Measurements, CarryTheirDuals) {
  auto dual = std::vector<double>{1.0, 2.0};
  const auto sea = FieldMeasurement(FieldChannel::SeaLevel, dual);
  EXPECT_EQ(sea.seaLevelDual, dual);
  EXPECT_TRUE(sea.displacementDual.empty() && sea.potentialDual.empty());
  const auto phi = FieldMeasurement(FieldChannel::Potential, dual);
  EXPECT_EQ(phi.potentialDual, dual);
  const auto spin = RotationMeasurement(AngularVelocityComponent::Z);
  EXPECT_EQ(spin.angularVelocityDual, (std::array{0.0, 0.0, 1.0}));
  EXPECT_TRUE(spin.seaLevelDual.empty());
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
  // construction, so the port is held to the Python's numbers.
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
    ASSERT_EQ(m.seaLevelDual.size(), dual.size());
    EXPECT_DOUBLE_EQ(m.seaLevelDual[0], 2.0 * fit.weights[j]);
    EXPECT_DOUBLE_EQ(m.seaLevelDual[1], -1.0 * fit.weights[j]);
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
