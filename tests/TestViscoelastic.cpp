// The viscoelastic stepper: weight algebra, rheology construction, the
// elastic and long-term limits, measured order of accuracy, the exact
// inverse step, and the effective-operator limits. The pyslfp comparison
// gate lives in TestFixtures.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

/// The PREM-like elastic model from the stage-1 deck, with a 1e21 Pa s
/// Maxwell mantle: solid layers fully inside [3480, 6151] km. The same
/// rule selects the layers in the pyslfp fixture script.
struct TestBody {
  EarthModel model;
  Material material;
  Rheology rheology;
  double viscositySI = 1e21;

  static TestBody Make(int lmax) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::CreateGraded(model, lmax).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    const double eta = 1e21 / scales.Factor(ViscosityDimensions);
    std::map<std::size_t, RadialValue> viscosity;
    const auto nd = model.DimensionlessView();
    const auto b = nd.Boundaries();
    const double lo = 3.4799e6 / scales.Length();
    const double hi = 6.1511e6 / scales.Length();
    for (std::size_t i = 0; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= lo && b[i + 1] <= hi) {
        viscosity.emplace(i, eta);
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return TestBody{std::move(model), std::move(material), std::move(rheology)};
  }
};

/// A constant unit (non-dimensional) load on every degree.
LoadCallable UnitLoad() {
  return [](int, double) { return std::vector<double>{1.0}; };
}

double SurfaceU(const ViscoelasticOperator& op,
                const std::map<int, std::vector<double>>& solutions, int l) {
  return op.Surface(solutions, Component::U).at(l)[0];
}

}  // namespace

TEST(StepWeights, AlgebraAndSeries) {
  // alpha + beta = 1 - e: a constant strain is reproduced exactly.
  for (double h : {1e-8, 1e-4, 9.9e-4, 1.1e-3, 0.5, 5.0}) {
    const auto w = ExponentialTrapezoidWeights(h);
    EXPECT_NEAR(w.alpha + w.beta, 1.0 - w.e, 1e-15);
    EXPECT_NEAR(w.e, std::exp(-h), 1e-15);
    // Continuity across the series switch and agreement with the
    // quotient where both are accurate.
    const double phi1 = 1.0 - w.beta;
    if (h > 1e-6) EXPECT_NEAR(phi1, (1.0 - std::exp(-h)) / h, 1e-12);
  }
  // h = 0 (an elastic node): the variable stays put, effective weight 1.
  const auto w0 = ExponentialTrapezoidWeights(0.0);
  EXPECT_DOUBLE_EQ(w0.e, 1.0);
  EXPECT_DOUBLE_EQ(w0.alpha, 0.0);
  EXPECT_DOUBLE_EQ(w0.beta, 0.0);
  EXPECT_DOUBLE_EQ(BackwardEulerWeight(0.0), 1.0);

  // The inverse step: composing the update at h with the update at -h
  // (strains swapped) is the identity, for any strains.
  const double h = 0.7, m0 = 0.3, d0 = -1.1, d1 = 0.8;
  const auto f = ExponentialTrapezoidWeights(h);
  const auto r = ExponentialTrapezoidWeights(-h);
  const double m1 = f.e * m0 + f.alpha * d0 + f.beta * d1;
  EXPECT_NEAR(r.e * m1 + r.alpha * d1 + r.beta * d0, m0, 1e-14);

  // Derivatives by central differences.
  const double eps = 1e-6;
  const auto dw = ExponentialTrapezoidWeightDerivatives(h);
  const auto plus = ExponentialTrapezoidWeights(h + eps);
  const auto minus = ExponentialTrapezoidWeights(h - eps);
  EXPECT_NEAR(dw.e, (plus.e - minus.e) / (2 * eps), 1e-8);
  EXPECT_NEAR(dw.alpha, (plus.alpha - minus.alpha) / (2 * eps), 1e-8);
  EXPECT_NEAR(dw.beta, (plus.beta - minus.beta) / (2 * eps), 1e-8);
  EXPECT_NEAR(
      BackwardEulerWeightDerivative(h),
      (BackwardEulerWeight(h + eps) - BackwardEulerWeight(h - eps)) / (2 * eps),
      1e-8);
}

TEST(Rheology, MaxwellConstructionAndLimits) {
  auto body = TestBody::Make(4);
  const auto& rheology = body.rheology;
  EXPECT_EQ(rheology.BranchCount(), 1u);

  // tau = eta / mu: the smallest relaxation time belongs to the largest
  // shear modulus of the viscous region.
  const auto& scales = body.model.ModelScales();
  const double tauSI = rheology.MinRelaxationTime() * scales.Time();
  EXPECT_GT(tauSI, 1e9);   // ~hundreds of years at least
  EXPECT_LT(tauSI, 1e13);  // well under a million years

  // The long-term material has no shear where the branch relaxes (a
  // Maxwell body), and the unrelaxed one everywhere else.
  auto longTerm = rheology.LongTermMaterial();
  const auto [kappaL, muL] = longTerm.Moduli().KappaMu();
  const auto [kappaU, muU] = body.material.Moduli().KappaMu();
  for (std::size_t i = 0; i < muL.size(); ++i) {
    if (rheology.IsActive(i)) {
      EXPECT_NEAR(muL[i], 0.0, 1e-12 * muU[i]);
    } else {
      EXPECT_DOUBLE_EQ(muL[i], muU[i]);
    }
    EXPECT_NEAR(kappaL[i], kappaU[i], 1e-12 * kappaU[i]);
  }

  // A fluid layer refuses to relax.
  std::map<std::size_t, RadialValue> onFluid;
  for (std::size_t i = 0; i < body.model.LayerCount(); ++i) {
    if (body.model.IsFluid(i)) onFluid.emplace(i, 1e21);
  }
  EXPECT_FALSE(Rheology::Maxwell(body.material, onFluid).has_value());
}

TEST(Viscoelastic, ElasticAndLongTermLimits) {
  auto body = TestBody::Make(4);
  ViscoelasticOperator op(body.rheology, 4);
  const auto& scales = body.model.ModelScales();
  const auto elastic = op.ElasticLoveNumbers().ConvertedTo(scales);

  // At t = 0 the response to a unit load is the unrelaxed (elastic) one.
  auto state = op.ZeroState();
  auto solutions = op.SolveElastic(state, UnitLoad());
  const double h0 = SurfaceU(op, solutions, 2);
  EXPECT_NEAR(h0, elastic.h_u[2] + elastic.h_phi[2], 1e-10 * std::abs(h0));

  // Stepping a held load relaxes towards the long-term solid's response.
  const auto longTermTable =
      ComputeLoveNumbers(body.rheology.LongTermMaterial(), 4)
          .ConvertedTo(scales);
  const double hInf = longTermTable.h_u[2] + longTermTable.h_phi[2];
  const double tau = body.rheology.MinRelaxationTime();
  // The slowest decay modes (the elastic lithosphere's) run to many
  // Maxwell times, so the approach is pushed hard and judged at 1%.
  double dt = tau;
  for (int i = 0; i < 1500; ++i) {
    auto [next, sols] =
        op.Step(state, dt, Scheme::ExponentialTrapezoid, UnitLoad());
    state = std::move(next);
    solutions = std::move(sols);
    dt = std::min(dt * 1.1, 1000 * tau);  // stride out as modes die
  }
  const double hT = SurfaceU(op, solutions, 2);
  EXPECT_NEAR(hT, hInf, 1e-2 * std::abs(hInf));
  // ... and has moved far from the elastic response on the way.
  EXPECT_GT(std::abs(hT - h0), 5.0 * std::abs(hT - hInf));
}

TEST(Viscoelastic, MeasuredOrderOfAccuracy) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const double tau = body.rheology.MinRelaxationTime();
  const double T = 4.0 * tau;

  auto advance = [&](Scheme scheme, int steps) {
    auto state = op.ZeroState();
    std::map<int, std::vector<double>> solutions;
    const double dt = T / steps;
    for (int i = 0; i < steps; ++i) {
      auto [next, sols] = op.Step(state, dt, scheme, UnitLoad());
      state = std::move(next);
      solutions = std::move(sols);
    }
    if (scheme == Scheme::ExponentialEuler) {
      solutions = op.SolveElastic(state, UnitLoad());
    }
    return SurfaceU(op, solutions, 2);
  };

  // Richardson: (x_h - x_h/2) / (x_h/2 - x_h/4) ~ 2^order.
  for (auto [scheme, expected] : {std::pair{Scheme::ExponentialTrapezoid, 4.0},
                                  std::pair{Scheme::BackwardEuler, 2.0},
                                  std::pair{Scheme::ExponentialEuler, 2.0}}) {
    const double x1 = advance(scheme, 16);
    const double x2 = advance(scheme, 32);
    const double x4 = advance(scheme, 64);
    const double ratio = (x1 - x2) / (x2 - x4);
    EXPECT_NEAR(ratio, expected, 0.35 * expected)
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Viscoelastic, ExactInverseStep) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const double tau = body.rheology.MinRelaxationTime();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto state = op.ZeroState();
    op.PrimeStrain(state, UnitLoad());
    // March a few steps so the state is non-trivial, then invert them.
    std::vector<ViscoelasticState> history{state};
    const double dt = 0.8 * tau;
    for (int i = 0; i < 3; ++i) {
      auto [next, sols] = op.Step(history.back(), dt, scheme, UnitLoad());
      history.push_back(std::move(next));
    }
    auto current = history.back();
    for (int i = 2; i >= 0; --i) {
      auto [previous, sols] = op.StepBack(current, dt, scheme, UnitLoad());
      double scale = 0.0, diff = 0.0;
      for (const auto& [l, m] : history.back().m) {
        for (double v : m) scale = std::max(scale, std::abs(v));
      }
      for (const auto& [l, m] : history[i].m) {
        const auto& mBack = previous.m.at(l);
        for (std::size_t j = 0; j < m.size(); ++j) {
          diff = std::max(diff, std::abs(m[j] - mBack[j]));
        }
      }
      EXPECT_LT(diff, 1e-9 * std::max(scale, 1e-30))
          << "node " << i << ", scheme " << static_cast<int>(scheme);
      EXPECT_NEAR(previous.t, history[i].t, 1e-12 * tau);
      current = std::move(previous);
    }
  }
}

TEST(Viscoelastic, EffectiveLoveNumberLimits) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const double tau = body.rheology.MinRelaxationTime();
  const auto elastic = op.ElasticLoveNumbers();
  const auto longTerm = ComputeLoveNumbers(body.rheology.LongTermMaterial(), 2);

  // dt -> 0: the effective solid tends to the unrelaxed one.
  const auto quick =
      op.EffectiveLoveNumbers(Scheme::ExponentialTrapezoid, 1e-6 * tau);
  EXPECT_NEAR(quick.h_u[2], elastic.h_u[2], 1e-5 * std::abs(elastic.h_u[2]));
  // dt -> infinity: towards the long-term solid. The approach is
  // sqrt(phi1)-slow (a residual-shear boundary layer), so at
  // phi1 = 1e-8 the limit is met at the per-mille level only.
  const auto slow =
      op.EffectiveLoveNumbers(Scheme::ExponentialTrapezoid, 1e8 * tau);
  EXPECT_NEAR(slow.h_u[2], longTerm.h_u[2], 2e-3 * std::abs(longTerm.h_u[2]));

  // The adaptive controller accepts a sane step and shrinks a huge one.
  auto state = op.ZeroState();
  // From a cold start under a switched-on load the ETD1 estimate
  // scales like h / rtol, so acceptance wants a small first step.
  auto fine = op.AdaptiveStep(state, 1e-5 * tau, 1e-3, UnitLoad());
  EXPECT_TRUE(fine.state.has_value());
  auto coarse = op.AdaptiveStep(state, 50.0 * tau, 1e-10, UnitLoad());
  EXPECT_LT(coarse.dtNext, 50.0 * tau);
}
