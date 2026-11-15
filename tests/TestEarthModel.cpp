// EarthModel: structure, fields, classification, the columns of
// planetmodel's viscoelastic decks joined as planetmodel joins them, and the
// reference gravity against closed forms.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <map>
#include <numbers>
#include <planetmodel/deck.hpp>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Scales.hpp"

using sl3dpp::Deck;
using sl3dpp::EarthModel;
using sl3dpp::GravitationalConstantSI;
using sl3dpp::Side;

namespace {

constexpr auto kPi = std::numbers::pi;

/// A deck of `layers` (lo, hi, rho) with `n` equally spaced knots each and
/// constant density per layer.
Deck UniformDeck(const std::vector<std::array<double, 3>>& layers,
                 std::size_t n = 5) {
  std::vector<double> radius, rho;
  for (const auto& [lo, hi, density] : layers) {
    for (auto k = std::size_t{0}; k < n; ++k) {
      radius.push_back(std::lerp(lo, hi, static_cast<double>(k) / (n - 1)));
      rho.push_back(density);
    }
  }
  return Deck::Create(std::move(radius), {{"rho", std::move(rho)}}).value();
}

}  // namespace

TEST(EarthModel, HomogeneousSphereClosedForms) {
  constexpr auto rho0 = 5000.0;
  constexpr auto b = 6.0e6;
  // Two layers of the same density, to exercise the layer accumulation.
  auto model = EarthModel::FromDeck(
                   UniformDeck({{0.0, 0.5 * b, rho0}, {0.5 * b, b, rho0}}))
                   .value();

  const auto mass = (4.0 / 3.0) * kPi * rho0 * b * b * b;
  EXPECT_NEAR(model.Mass(), mass, 1e-9 * mass);
  const auto moi = (8.0 * kPi / 15.0) * rho0 * std::pow(b, 5);
  EXPECT_NEAR(model.MomentOfInertia(), moi, 1e-9 * moi);

  for (double r : {0.2 * b, 0.5 * b, 0.9 * b, b}) {
    const auto g = (4.0 / 3.0) * kPi * GravitationalConstantSI * rho0 * r;
    EXPECT_NEAR(model.Gravity(r), g, 1e-9 * g);
    const double phi = (2.0 / 3.0) * kPi * GravitationalConstantSI * rho0 *
                       (r * r - 3.0 * b * b);
    EXPECT_NEAR(model.Potential(r), phi, 1e-9 * std::abs(phi));
    const double p = (2.0 / 3.0) * kPi * GravitationalConstantSI * rho0 * rho0 *
                     (b * b - r * r);
    EXPECT_NEAR(model.Pressure(r), p, 1e-9 * std::max(p, 1.0));
  }
  // Outside the body.
  const auto gm = GravitationalConstantSI * mass;
  EXPECT_NEAR(model.Gravity(2 * b), gm / (4 * b * b), 1e-9 * gm / (4 * b * b));
  EXPECT_NEAR(model.Potential(2 * b), -gm / (2 * b), 1e-9 * gm / (2 * b));
}

TEST(EarthModel, QuadraticDensityClosedForms) {
  // rho = rho_c (1 - (r/b)^2), one layer, enough knots for not-a-knot to
  // reproduce the quadratic exactly.
  constexpr auto rhoc = 12000.0;
  constexpr auto b = 6.371e6;
  const auto n = std::size_t{9};
  std::vector<double> radius(n), rho(n);
  for (auto k = std::size_t{0}; k < n; ++k) {
    radius[k] = b * static_cast<double>(k) / (n - 1);
    rho[k] = rhoc * (1.0 - std::pow(radius[k] / b, 2));
  }
  auto deck = Deck::Create(std::move(radius), {{"rho", std::move(rho)}});
  auto model = EarthModel::FromDeck(deck.value()).value();

  auto massInside = [&](double r) {
    return 4.0 * kPi * rhoc *
           (r * r * r / 3.0 - std::pow(r, 5) / (5.0 * b * b));
  };
  EXPECT_NEAR(model.Mass(), massInside(b), 1e-9 * massInside(b));
  for (double r : {0.3 * b, 0.7 * b}) {
    const auto g = GravitationalConstantSI * massInside(r) / (r * r);
    EXPECT_NEAR(model.Gravity(r), g, 1e-9 * g);
    const double outer = 4.0 * kPi * rhoc *
                         ((b * b - r * r) / 2.0 -
                          (std::pow(b, 4) - std::pow(r, 4)) / (4.0 * b * b));
    const auto phi = -GravitationalConstantSI * (massInside(r) / r + outer);
    EXPECT_NEAR(model.Potential(r), phi, 1e-9 * std::abs(phi));
  }
  // phi(0) = -pi G rho_c b^2, and g = d(phi)/dr by finite differences.
  EXPECT_NEAR(model.Potential(0.0),
              -kPi * GravitationalConstantSI * rhoc * b * b,
              1e-9 * kPi * GravitationalConstantSI * rhoc * b * b);
  const double r = 0.6 * b, h = 1.0;  // 1 m step
  EXPECT_NEAR((model.Potential(r + h) - model.Potential(r - h)) / (2 * h),
              model.Gravity(r), 1e-5 * model.Gravity(r));
}

TEST(EarthModel, SiInSiOutAndDimensionlessView) {
  constexpr auto rho0 = 5000.0;
  constexpr auto b = 6.0e6;
  auto model = EarthModel::FromDeck(UniformDeck({{0.0, b, rho0}})).value();

  // SI surface.
  EXPECT_DOUBLE_EQ(model.SurfaceRadius(), b);
  EXPECT_NEAR(model.Density(0.5 * b), rho0, 1e-9 * rho0);

  // The internal numbers: surface at 1, G = 1, density rho0 / rho_bar.
  auto nd = model.DimensionlessView();
  EXPECT_DOUBLE_EQ(nd.SurfaceRadius(), 1.0);
  EXPECT_NEAR(nd.G(), 1.0, 1e-14);
  EXPECT_NEAR(nd.Density(0.5), rho0 / sl3dpp::EarthMeanDensitySI,
              1e-12 * rho0 / sl3dpp::EarthMeanDensitySI);
  // The two surfaces are the same numbers under the scales.
  const auto& scales = model.ModelScales();
  EXPECT_NEAR(nd.Mass() * scales.Factor({.mass = 1}), model.Mass(),
              1e-12 * model.Mass());
  EXPECT_NEAR(nd.Gravity(0.5) * scales.Factor(sl3dpp::AccelerationDimensions),
              model.Gravity(0.5 * b), 1e-12 * model.Gravity(0.5 * b));
}

TEST(EarthModel, SidedEvaluationAtABoundary) {
  constexpr auto b = 6.0e6;
  auto model = EarthModel::FromDeck(
                   UniformDeck({{0.0, 0.5 * b, 5000.0}, {0.5 * b, b, 3000.0}}))
                   .value();
  EXPECT_NEAR(model.Density(0.5 * b, Side::Below), 5000.0, 1e-6);
  EXPECT_NEAR(model.Density(0.5 * b, Side::Above), 3000.0, 1e-6);
  EXPECT_EQ(model.LayerOf(0.5 * b, Side::Below), 0u);
  EXPECT_EQ(model.LayerOf(0.5 * b, Side::Above), 1u);
}

TEST(EarthModel, LayerClassification) {
  // Solid elastic inner core, fluid outer core, viscoelastic mantle.
  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> radius, rho, mu, viscosity;
  auto layer = [&](double lo, double hi, double density, double shear,
                   double eta) {
    for (auto k = 0; k < 3; ++k) {
      radius.push_back(std::lerp(lo, hi, k / 2.0));
      rho.push_back(density);
      mu.push_back(shear);
      viscosity.push_back(eta);
    }
  };
  layer(0.0, 1.0e6, 13000.0, 150e9, nan);
  layer(1.0e6, 3.0e6, 11000.0, 0.0, nan);
  layer(3.0e6, 6.0e6, 5000.0, 100e9, 1e21);
  auto deck =
      Deck::Create(std::move(radius), {{"rho", std::move(rho)},
                                       {"mu", std::move(mu)},
                                       {"viscosity", std::move(viscosity)}})
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  EXPECT_FALSE(model.IsFluid(0));
  EXPECT_TRUE(model.IsFluid(1));
  EXPECT_FALSE(model.IsFluid(2));
  EXPECT_FALSE(model.IsViscoelastic(0));
  EXPECT_FALSE(model.IsViscoelastic(1));
  EXPECT_TRUE(model.IsViscoelastic(2));
  EXPECT_FALSE(model.Has("viscosity", 0));
  EXPECT_TRUE(model.Has("viscosity", 2));
  EXPECT_THROW((void)model.EvaluateInLayer("viscosity", 0, 0.5e6),
               std::domain_error);
}

TEST(EarthModel, ViscoelasticColumnsAreJoinedAsPlanetmodelJoinsThem) {
  // A viscosity changing by orders of magnitude within a layer is joined as
  // planetmodel's DeckLayers joins it under its kinds for its viscoelastic
  // decks (ViscoelasticDeckKinds: in its logarithm by straight lines,
  // geometric between knots), value and derivative to rounding, while the
  // elastic columns stay the model's own splines; a viscosity constant
  // through a layer is that constant; kinds given replace the defaults
  // (none: every column the model's spline; `linear`: the values by
  // straight lines); and the refusals: a value not positive under a
  // logarithmic kind, `rho` given a kind, a kind naming a column the deck
  // lacks, the second derivative of a joined column.
  const auto radius =
      std::vector<double>{0.0, 1e6, 2e6, 3e6, 3e6, 4e6, 5e6, 6e6};
  const auto graded =
      std::vector<double>{1e21, 1e21, 1e21, 1e21, 1e19, 3e20, 2e22, 1e23};
  const auto columns = [&](const std::vector<double>& viscosity) {
    return std::vector<std::pair<std::string, std::vector<double>>>{
        {"rho",
         {6000.0, 5900.0, 5700.0, 5400.0, 4000.0, 3900.0, 3700.0, 3400.0}},
        {"mu",
         {1.2e11, 1.1e11, 1.0e11, 0.9e11, 0.8e11, 0.75e11, 0.7e11, 0.6e11}},
        {"viscosity", viscosity}};
  };
  const auto deck = Deck::Create(radius, columns(graded)).value();
  const auto model = EarthModel::FromDeck(deck).value();
  const auto table = planetmodel::Deck(radius, columns(graded));
  auto kinds = std::map<std::string, std::string, std::less<>>{};
  for (const auto& [name, kind] : sl3dpp::ViscoelasticDeckKinds()) {
    if (name == "viscosity") kinds.emplace(name, kind);
  }
  const auto joined = planetmodel::DeckLayers(table, "cubic", kinds);
  auto worst = 0.0, worstSlope = 0.0;
  for (auto r = 3.05e6; r < 6e6; r += 0.1e6) {
    const auto& f = joined.layers[1].at("viscosity");
    worst = std::max(
        worst, std::abs(model.EvaluateInLayer("viscosity", 1, r) - f.Value(r)) /
                   f.Value(r));
    worstSlope = std::max(worstSlope,
                          std::abs(model.EvaluateInLayer("viscosity", 1, r, 1) -
                                   f.Derivative(r)) /
                              std::abs(f.Derivative(r)));
  }
  RecordProperty("value", std::format("{:.2e}", worst));
  RecordProperty("derivative", std::format("{:.2e}", worstSlope));
  // (2.5e-15 each here: the two joins' rounding)
  EXPECT_LT(worst, 1e-13);
  EXPECT_LT(worstSlope, 1e-13);
  // geometric between knots
  EXPECT_NEAR(model.EvaluateInLayer("viscosity", 1, 4.5e6),
              std::sqrt(3e20 * 2e22), 1e-14 * std::sqrt(3e20 * 2e22));
  for (const auto r : {0.0, 0.7e6, 2.3e6, 3e6}) {
    // a deterministic evaluation of a constant: to the last bits
    EXPECT_DOUBLE_EQ(model.EvaluateInLayer("viscosity", 0, r), 1e21);
  }
  // the elastic columns are the model's splines, whatever the kinds
  const auto splined =
      EarthModel::FromDeck(deck,
                           {.kinds = std::map<std::string, std::string>{}})
          .value();
  for (const auto r : {3.3e6, 4.4e6, 5.5e6}) {
    // the same spline built from the same column: bitwise
    EXPECT_EQ(splined.EvaluateInLayer("mu", 1, r),
              model.EvaluateInLayer("mu", 1, r));
    EXPECT_NE(splined.EvaluateInLayer("viscosity", 1, r),
              model.EvaluateInLayer("viscosity", 1, r));
  }
  const auto linear =
      EarthModel::FromDeck(
          deck, {.kinds = std::map<std::string, std::string>{{"viscosity",
                                                              "linear"}}})
          .value();
  EXPECT_NEAR(linear.EvaluateInLayer("viscosity", 1, 4.5e6),
              0.5 * (3e20 + 2e22), 1e-14 * 2e22);
  // the refusals
  auto negative = graded;
  negative[5] = -1.0;
  const auto refused =
      EarthModel::FromDeck(Deck::Create(radius, columns(negative)).value());
  ASSERT_FALSE(refused.has_value());
  EXPECT_NE(refused.error().message.find("viscosity"), std::string::npos)
      << refused.error().message;
  const auto byRho = EarthModel::FromDeck(
      deck, {.kinds = std::map<std::string, std::string>{{"rho", "linear"}}});
  ASSERT_FALSE(byRho.has_value());
  EXPECT_NE(byRho.error().message.find("'rho'"), std::string::npos);
  const auto lacking = EarthModel::FromDeck(
      deck, {.kinds = std::map<std::string, std::string>{{"vs", "linear"}}});
  ASSERT_FALSE(lacking.has_value());
  EXPECT_NE(lacking.error().message.find("a column the deck lacks"),
            std::string::npos)
      << lacking.error().message;
  EXPECT_THROW(static_cast<void>(model.EvaluateInLayer("viscosity", 1, 4e6, 2)),
               std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(model.EvaluateInLayer("mu", 1, 4e6, 2)));
}

TEST(EarthModel, RefusesMissingRhoAndPartialNaN) {
  auto noRho = Deck::Create({0.0, 1.0, 2.0}, {{"mu", {1.0, 1.0, 1.0}}});
  EXPECT_FALSE(EarthModel::FromDeck(noRho.value()).has_value());

  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
  auto partial = Deck::Create(
      {0.0, 1.0, 2.0}, {{"rho", {1.0, 1.0, 1.0}}, {"mu", {1.0, nan, 1.0}}});
  EXPECT_FALSE(EarthModel::FromDeck(partial.value()).has_value());
}

TEST(EarthModel, LegacyPremLooksLikeTheEarth) {
  auto deck = Deck::ReadLegacySl3d(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                                   "prem.simp.vm1")
                  .value();
  // The Fortran splined naturally; match it.
  auto model =
      EarthModel::FromDeck(deck, {.spline = sl3dpp::SplineKind::Natural})
          .value();
  EXPECT_NEAR(model.Mass(), 5.972e24, 0.01e24);
  EXPECT_NEAR(model.Gravity(model.SurfaceRadius()), 9.82, 0.03);
  // The normalised moment of inertia, a sharp test of the mass profile.
  EXPECT_NEAR(model.MomentOfInertia() /
                  (model.Mass() * std::pow(model.SurfaceRadius(), 2)),
              0.3308, 0.002);
  // Fluid outer core found by mu = 0; viscous mantle present.
  auto anyFluid = false, anyViscous = false;
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    anyFluid = anyFluid || model.IsFluid(i);
    anyViscous = anyViscous || model.IsViscoelastic(i);
  }
  EXPECT_TRUE(anyFluid);
  EXPECT_TRUE(anyViscous);
  // Central pressure within half a percent of PREM's 364 GPa.
  EXPECT_NEAR(model.Pressure(0.0), 3.64e11, 2e9);
}
