// The viscoelastic stepper: weight algebra, rheology construction (the
// Burgers body's Prony terms against its modulus and creep function, and
// the bodies a model's fields state, planetmodel's prony and burgers decks
// among them), the elastic and long-term limits, measured order of accuracy,
// and the effective-operator limits; the step
// recipes (StepRecipe.hpp) and Sdirk2 through them: its tableau and
// amplification on a scalar equation, its order, its zero-length step and
// what it refuses; a step's flavour held through the cache's evictions.
// The pyslfp comparison lives in TestFixtures.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/Viscoelastic.hpp"

#include <planetmodel/deck.hpp>

using namespace sl3dpp;

namespace {

/// The PREM-like elastic model from prem8.elastic.deck, with a 1e21 Pa s
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
    const auto eta = 1e21 / scales.Factor(ViscosityDimensions);
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto nd = model.DimensionlessView();
    const auto b = nd.Boundaries();
    const auto lo = 3.4799e6 / scales.Length();
    const auto hi = 6.1511e6 / scales.Length();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
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
    const auto phi1 = 1.0 - w.beta;
    if (h > 1e-6) EXPECT_NEAR(phi1, (1.0 - std::exp(-h)) / h, 1e-12);
  }
  // h = 0 (an elastic node): the variable stays put, effective weight 1.
  const auto w0 = ExponentialTrapezoidWeights(0.0);
  EXPECT_DOUBLE_EQ(w0.e, 1.0);
  EXPECT_DOUBLE_EQ(w0.alpha, 0.0);
  EXPECT_DOUBLE_EQ(w0.beta, 0.0);
  EXPECT_DOUBLE_EQ(BackwardEulerWeight(0.0), 1.0);

  // Every step runs forwards: a negative h is refused.
  EXPECT_THROW((void)ExponentialTrapezoidWeights(-0.7), std::invalid_argument);

  // Derivatives by central differences.
  const auto h = 0.7;
  const auto eps = 1e-6;
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
  const auto tauSI = rheology.MinRelaxationTime() * scales.Time();
  EXPECT_GT(tauSI, 1e9);   // ~hundreds of years at least
  EXPECT_LT(tauSI, 1e13);  // well under a million years

  // The long-term material has no shear where the branch relaxes (a
  // Maxwell body), and the unrelaxed one everywhere else.
  auto longTerm = rheology.LongTermMaterial();
  const auto [kappaL, muL] = longTerm.Moduli().KappaMu();
  const auto [kappaU, muU] = body.material.Moduli().KappaMu();
  for (auto i = std::size_t{0}; i < muL.size(); ++i) {
    if (rheology.IsActive(i)) {
      EXPECT_NEAR(muL[i], 0.0, 1e-12 * muU[i]);
    } else {
      EXPECT_DOUBLE_EQ(muL[i], muU[i]);  // nothing relaxes it: as it was
    }
    EXPECT_NEAR(kappaL[i], kappaU[i], 1e-12 * kappaU[i]);
  }

  // A fluid layer refuses to relax.
  auto onFluid = std::map<std::size_t, RadialValue>{};
  for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
    if (body.model.IsFluid(i)) onFluid.emplace(i, 1e21);
  }
  EXPECT_FALSE(Rheology::Maxwell(body.material, onFluid).has_value());
}

TEST(Rheology, BurgersPronyTermsAreTheBurgersBody) {
  // The two Prony terms of a Burgers body against its own definitions:
  // the modulus in the Laplace variable, the unrelaxed modulus and the
  // steady viscosity, and the creep function. Under a unit stress held
  // from t = 0 the Prony body's strain e, from 1 = sum_k mu_k (e - m_k)
  // with dm_k/dt = itau_k (e - m_k), integrated by RK4, is the Burgers
  // creep function J(t) = 1/mu + t/eta + (1 - exp(-t mu_K/eta_K)) / mu_K.
  struct Body {
    double mu, eta, muK, etaK;
  };
  const auto bodies = std::vector<Body>{
      {1.0, 1.0, 1.0, 0.1},  // the post-seismic examples' asthenosphere
      {1.0, 1.0, 0.5, 2.0},  // a Kelvin element slower than the Maxwell one
      {6.6e10, 1.0e19, 6.6e10, 1.0e18},  // SI
      {2.0, 3.0, 1.0e3, 1.0e-3},         // a stiff, quick Kelvin element
      {1.0, 1.0, 1.0e-4, 1.0e-4},        // a soft one
      {1.0, 1.0e6, 1.0, 1.0}};           // a near-rigid dashpot
  auto laplace = 0.0;
  for (const auto& [mu, eta, muK, etaK] : bodies) {
    const auto terms = BurgersPronyTerms(mu, eta, muK, etaK);
    const auto label = std::format("({}, {}, {}, {})", mu, eta, muK, etaK);
    EXPECT_GT(terms[0].mu, 0.0) << label;
    EXPECT_GT(terms[1].mu, 0.0) << label;
    EXPECT_LT(terms[0].itau, terms[1].itau) << label;
    EXPECT_NEAR(terms[0].mu + terms[1].mu, mu, 1e-14 * mu) << label;
    EXPECT_NEAR(terms[0].mu / terms[0].itau + terms[1].mu / terms[1].itau, eta,
                1e-13 * eta)
        << label;
    // 1 / M(s) = 1/mu + 1/(eta s) + 1/(mu_K + eta_K s) on the scale of
    // the rates, real and complex
    const auto scale = terms[1].itau;
    for (const auto s :
         {std::complex<double>(1e-3, 0.0), std::complex<double>(0.7, 0.0),
          std::complex<double>(30.0, 0.0), std::complex<double>(0.2, 1.5),
          std::complex<double>(-0.01, 4.0)}) {
      const auto z = s * scale;
      const auto prony = terms[0].mu * z / (z + terms[0].itau) +
                         terms[1].mu * z / (z + terms[1].itau);
      const auto burgers =
          1.0 / (1.0 / mu + 1.0 / (eta * z) + 1.0 / (muK + etaK * z));
      EXPECT_LT(std::abs(prony - burgers), 1e-13 * std::abs(burgers))
          << label << " at s = " << z;
      laplace =
          std::max(laplace, std::abs(prony - burgers) / std::abs(burgers));
    }
  }
  RecordProperty("laplace_relative", std::format("{:.3e}", laplace));
  // the creep function, on the bodies whose rates a few thousand RK4
  // steps resolve
  auto creepGap = 0.0;
  for (const auto& [mu, eta, muK, etaK] :
       std::vector<Body>{bodies[0], bodies[1], bodies[2]}) {
    const auto terms = BurgersPronyTerms(mu, eta, muK, etaK);
    const auto h = 0.01 / terms[1].itau;
    const auto end = 5.0 * std::max(eta / mu, etaK / muK);
    auto strain = [&](const std::array<double, 2>& m) {
      return (1.0 + terms[0].mu * m[0] + terms[1].mu * m[1]) /
             (terms[0].mu + terms[1].mu);
    };
    auto rate = [&](const std::array<double, 2>& m) {
      const auto e = strain(m);
      return std::array<double, 2>{terms[0].itau * (e - m[0]),
                                   terms[1].itau * (e - m[1])};
    };
    auto creep = [&](double t) {
      return 1.0 / mu + t / eta + (1.0 - std::exp(-t * muK / etaK)) / muK;
    };
    auto m = std::array<double, 2>{};
    auto worst = std::abs(strain(m) - creep(0.0));
    const auto steps = static_cast<int>(std::ceil(end / h));
    for (auto n = 0; n < steps; ++n) {
      auto shifted = [&](const std::array<double, 2>& k, double f) {
        return std::array<double, 2>{m[0] + f * h * k[0], m[1] + f * h * k[1]};
      };
      const auto k1 = rate(m);
      const auto k2 = rate(shifted(k1, 0.5));
      const auto k3 = rate(shifted(k2, 0.5));
      const auto k4 = rate(shifted(k3, 1.0));
      for (auto k = 0; k < 2; ++k) {
        m[k] += h / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
      }
      worst = std::max(worst, std::abs(strain(m) - creep((n + 1) * h)));
    }
    EXPECT_LT(worst, 1e-9 * creep(steps * h))
        << "creep of (" << mu << ", " << eta << ", " << muK << ", " << etaK
        << ")";
    creepGap = std::max(creepGap, worst / creep(steps * h));
  }
  RecordProperty("creep_relative", std::format("{:.3e}", creepGap));
  EXPECT_THROW(static_cast<void>(BurgersPronyTerms(1.0, 0.0, 1.0, 1.0)),
               std::invalid_argument);
}

TEST(Rheology, BurgersLayersAreTwoBranches) {
  // Rheology::Burgers on the PREM-like body: a Maxwell lower mantle and a
  // Burgers upper mantle (its Kelvin element of the unrelaxed modulus and
  // a tenth of the steady viscosity), two branches whose moduli sum to
  // the unrelaxed one and whose summed mu_k tau_k is the steady
  // viscosity at every Burgers node, the Maxwell nodes as Maxwell's.
  auto body = TestBody::Make(4);
  const auto& scales = body.model.ModelScales();
  const auto eta = 1e21 / scales.Factor(ViscosityDimensions);
  const auto& mesh = body.material.Mesh();
  const auto b = body.model.DimensionlessView().Boundaries();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  auto kelvin = std::map<std::size_t, Rheology::Kelvin>{};
  const auto d670 = 5.7011e6 / scales.Length();
  for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
    if (body.model.IsFluid(i) || b[i] < 3.4799e6 / scales.Length() ||
        b[i + 1] > 6.1511e6 / scales.Length()) {
      continue;
    }
    viscosity[i] = eta;
    if (b[i] >= d670 - 1e-6) {
      kelvin[i] =
          Rheology::Kelvin{.modulus_fraction = 1.0, .viscosity = 0.1 * eta};
    }
  }
  ASSERT_FALSE(kelvin.empty());
  const auto burgers =
      Rheology::Burgers(body.material, viscosity, kelvin).value();
  ASSERT_EQ(burgers.BranchCount(), 2u);
  const auto& maxwell = body.rheology;
  const auto mu = burgers.MuUnrelaxed();
  auto burgersNodes = 0;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto isBurgers =
        kelvin.contains(static_cast<std::size_t>(mesh.LayerOf(e)));
    for (auto i = 0; i < mesh.NGll(); ++i) {
      const auto node = static_cast<std::size_t>(e) * mesh.NGll() + i;
      const auto& slow = burgers.BranchAt(0);
      const auto& fast = burgers.BranchAt(1);
      if (!isBurgers) {
        // the Maxwell node's own arithmetic, no transform: bitwise
        EXPECT_EQ(slow.mu[node], maxwell.BranchAt(0).mu[node]);
        EXPECT_EQ(slow.itau[node], maxwell.BranchAt(0).itau[node]);
        EXPECT_EQ(fast.mu[node], 0.0);
        continue;
      }
      ++burgersNodes;
      EXPECT_NEAR(slow.mu[node] + fast.mu[node], mu[node], 1e-14 * mu[node]);
      EXPECT_NEAR(
          slow.mu[node] / slow.itau[node] + fast.mu[node] / fast.itau[node],
          eta, 1e-13 * eta);
      // the Maxwell time tau_M = eta / mu: the post-seismic examples'
      // ratios, 0.475 mu at 2.05 tau_M and 0.525 mu at 0.0488 tau_M
      const auto tauM = eta / mu[node];
      EXPECT_NEAR(slow.mu[node] / mu[node], 0.4750, 5e-4);
      EXPECT_NEAR(1.0 / (slow.itau[node] * tauM), 2.0510, 5e-4);
      EXPECT_NEAR(1.0 / (fast.itau[node] * tauM), 0.04875, 5e-5);
      EXPECT_NEAR(burgers.MuLongTerm()[node], 0.0, 1e-14 * mu[node]);
    }
  }
  EXPECT_GT(burgersNodes, 0);
  // a Kelvin element without a steady viscosity is refused
  auto orphan = kelvin;
  orphan[0] = Rheology::Kelvin{.modulus_fraction = 1.0, .viscosity = eta};
  EXPECT_FALSE(Rheology::Burgers(body.material, viscosity, orphan).has_value());
}

namespace {

/// A homogeneous two-layer body (rho 5500, vp 10000, vs 5500: mu = 5500^3
/// Pa), a relaxing interior to 6271 km under an elastic lid, two knots a
/// layer: its columns, `interior` each the interior's value and NaN in the
/// lid.
std::pair<std::vector<double>,
          std::vector<std::pair<std::string, std::vector<double>>>>
LiddedColumns(const std::vector<std::pair<std::string, double>>& interior) {
  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{
      {"rho", std::vector<double>(4, 5500.0)},
      {"vpv", std::vector<double>(4, 10000.0)},
      {"vsv", std::vector<double>(4, 5500.0)},
      {"vph", std::vector<double>(4, 10000.0)},
      {"vsh", std::vector<double>(4, 5500.0)},
      {"eta", std::vector<double>(4, 1.0)}};
  for (const auto& [name, value] : interior) {
    columns.emplace_back(name, std::vector<double>{value, value, nan, nan});
  }
  return {{0.0, 6271e3, 6271e3, 6371e3}, std::move(columns)};
}

/// The material of a deck at lmax 4.
Material MaterialOf(const EarthModel& model) {
  return Material::Create(model, RadialMesh::Create(model, {.lmax = 4}).value())
      .value();
}

}  // namespace

TEST(Rheology, FromModelReadsTheBodiesItsFieldsState) {
  // A model's own fields as planetmodel's viscoelastic decks state them: a
  // prony layer's branches as written (mu_k = w_k mu, itau_k = 1 / tau_k),
  // three of them summing to 0.9 keeping a relaxed modulus of 0.1 mu; a
  // burgers layer's Burgers body (Rheology::Burgers of its columns, to
  // rounding); strengths summing past one by a deck's rounding taken to sum to
  // one, past it by more refused; and the bodies a layer cannot be.
  const auto material = [](const auto& interior) {
    auto [radius, columns] = LiddedColumns(interior);
    auto model =
        EarthModel::FromDeck(
            Deck::Create(std::move(radius), std::move(columns)).value())
            .value();
    auto solid = MaterialOf(model);
    return std::pair{std::move(model), std::move(solid)};
  };
  const auto strengths = std::array{0.5, 0.3, 0.1};
  const auto times = std::array{1e8, 1e9, 1e10};
  auto prony = std::vector<std::pair<std::string, double>>{};
  for (auto k = std::size_t{0}; k < 3; ++k) {
    prony.emplace_back(std::format("relaxation_strength_{}", k + 1),
                       strengths[k]);
    prony.emplace_back(std::format("relaxation_time_{}", k + 1), times[k]);
  }
  {
    const auto [model, solid] = material(prony);
    const auto rheology = Rheology::FromModel(solid, model).value();
    ASSERT_EQ(rheology.BranchCount(), 3u);
    const auto mu = rheology.MuUnrelaxed();
    const auto timeScale = model.ModelScales().Time();
    auto active = 0;
    for (auto i = std::size_t{0}; i < mu.size(); ++i) {
      if (!rheology.IsActive(i)) continue;
      ++active;
      for (auto k = std::size_t{0}; k < 3; ++k) {
        EXPECT_NEAR(rheology.BranchAt(k).mu[i], strengths[k] * mu[i],
                    1e-15 * mu[i]);
        EXPECT_NEAR(rheology.BranchAt(k).itau[i] * times[k] / timeScale, 1.0,
                    1e-14);
      }
      EXPECT_NEAR(rheology.MuLongTerm()[i], 0.1 * mu[i], 1e-14 * mu[i]);
    }
    EXPECT_GT(active, 0);
    // a Kelvin element is no prony model's
    EXPECT_FALSE(
        Rheology::FromModel(
            solid, model,
            {{0, Rheology::Kelvin{.modulus_fraction = 1.0, .viscosity = 1.0}}})
            .has_value());
  }
  {
    // the burgers layer: Rheology::Burgers of the columns
    const auto [model, solid] =
        material(std::vector<std::pair<std::string, double>>{
            {"viscosity", 1e21},
            {"kelvin_viscosity", 1e20},
            {"kelvin_modulus_fraction", 0.5}});
    const auto rheology = Rheology::FromModel(solid, model).value();
    const auto factor = model.ModelScales().Factor(ViscosityDimensions);
    const auto twin =
        Rheology::Burgers(solid, {{0, 1e21 / factor}},
                          {{0, Rheology::Kelvin{.modulus_fraction = 0.5,
                                                .viscosity = 1e20 / factor}}})
            .value();
    ASSERT_EQ(rheology.BranchCount(), 2u);
    for (auto k = std::size_t{0}; k < 2; ++k) {
      const auto& got = rheology.BranchAt(k);
      const auto& want = twin.BranchAt(k);
      ASSERT_EQ(got.mu.size(), want.mu.size());
      for (auto i = std::size_t{0}; i < got.mu.size(); ++i) {
        EXPECT_NEAR(got.mu[i], want.mu[i], 1e-14 * std::abs(want.mu[i])) << i;
        EXPECT_NEAR(got.itau[i], want.itau[i], 1e-14 * std::abs(want.itau[i]))
            << i;
      }
    }
  }
  {
    // past one by a deck's rounding: taken to sum to one
    const auto [model, solid] =
        material(std::vector<std::pair<std::string, double>>{
            {"relaxation_strength_1", 0.600000005},
            {"relaxation_time_1", 1e8},
            {"relaxation_strength_2", 0.400000005},
            {"relaxation_time_2", 1e9}});
    const auto rheology = Rheology::FromModel(solid, model).value();
    const auto mu = rheology.MuUnrelaxed();
    for (auto i = std::size_t{0}; i < mu.size(); ++i) {
      if (!rheology.IsActive(i)) continue;
      EXPECT_NEAR(rheology.BranchAt(0).mu[i] + rheology.BranchAt(1).mu[i],
                  mu[i], 1e-15 * mu[i]);
      EXPECT_NEAR(rheology.BranchAt(0).mu[i] / mu[i], 0.600000005 / 1.00000001,
                  1e-15);
    }
  }
  const auto refused =
      [&](const std::vector<std::pair<std::string, double>>& columns,
          const std::string& what) {
        const auto [model, solid] = material(columns);
        const auto rheology = Rheology::FromModel(solid, model);
        ASSERT_FALSE(rheology.has_value()) << what;
        EXPECT_NE(rheology.error().message.find(what), std::string::npos)
            << rheology.error().message;
      };
  refused({{"relaxation_strength_1", 0.7},
           {"relaxation_time_1", 1e8},
           {"relaxation_strength_2", 0.4},
           {"relaxation_time_2", 1e9}},
          "sum to 1.1");
  refused({{"viscosity", 1e21},
           {"relaxation_strength_1", 1.0},
           {"relaxation_time_1", 1e8}},
          "beside a viscosity");
  refused({{"viscosity", 1e21}, {"kelvin_viscosity", 1e20}},
          "without `kelvin_modulus_fraction`");
  refused({{"kelvin_viscosity", 1e20}, {"kelvin_modulus_fraction", 1.0}},
          "a Kelvin element and no viscosity");
  refused({{"relaxation_strength_1", 1.0}},
          "strength or time without the other");
}


TEST(Rheology, PlanetmodelsDecksAreReadWithTheirChecks) {
  // planetmodel's prony and burgers decks, written by its own writer, read
  // by its readers (PronyFormat, BurgersFormat) and their rheologies by
  // FromModel: a prony deck's three branches as written; a burgers deck's
  // Burgers layer; and a deck whose header is not its format's refused,
  // naming the line.
  const auto dir =
      std::filesystem::temp_directory_path() / "sl3dpp_test_rheology_decks";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  auto [radius, columns] = LiddedColumns({{"relaxation_strength_1", 0.5},
                                          {"relaxation_time_1", 1e8},
                                          {"relaxation_strength_2", 0.3},
                                          {"relaxation_time_2", 1e9},
                                          {"relaxation_strength_3", 0.1},
                                          {"relaxation_time_3", 1e10}});
  planetmodel::WriteDeck(
      dir / "three.prony.deck",
      planetmodel::Deck(radius, columns,
                        planetmodel::PronyHeader("three branches", 3)),
      planetmodel::PRONY());
  const auto deck = Deck::Read(dir / "three.prony.deck", PronyFormat()).value();
  EXPECT_TRUE(deck.Has("relaxation_time_3"));
  const auto model = EarthModel::FromDeck(deck).value();
  const auto rheology = Rheology::FromModel(MaterialOf(model), model).value();
  EXPECT_EQ(rheology.BranchCount(), 3u);
  auto [bradius, bcolumns] = LiddedColumns({{"viscosity", 1e21},
                                            {"kelvin_viscosity", 1e20},
                                            {"kelvin_modulus_fraction", 0.5}});
  planetmodel::WriteDeck(
      dir / "lidded.burgers.deck",
      planetmodel::Deck(bradius, bcolumns,
                        planetmodel::BurgersHeader("lidded")),
      planetmodel::BURGERS());
  const auto burgersDeck =
      Deck::Read(dir / "lidded.burgers.deck", BurgersFormat()).value();
  const auto burgersModel = EarthModel::FromDeck(burgersDeck).value();
  EXPECT_EQ(Rheology::FromModel(MaterialOf(burgersModel), burgersModel)
                ->BranchCount(),
            2u);
  // a Maxwell deck is no burgers deck: its column line refused
  {
    auto out = std::ofstream(dir / "maxwell.deck");
    out << "a maxwell deck\nr rho vpv vsv vph vsh eta viscosity\n"
           "0 5500 10000 5500 10000 5500 1 1e21\n"
           "6371000 5500 10000 5500 10000 5500 1 1e21\n";
  }
  const auto wrong = Deck::Read(dir / "maxwell.deck", BurgersFormat());
  ASSERT_FALSE(wrong.has_value());
  EXPECT_NE(wrong.error().message.find(
                "a burgers deck's second line names its columns"),
            std::string::npos)
      << wrong.error().message;
  std::filesystem::remove_all(dir);
}


TEST(Viscoelastic, ElasticAndLongTermLimits) {
  auto body = TestBody::Make(4);
  ViscoelasticOperator op(body.rheology, 4);
  const auto& scales = body.model.ModelScales();
  const auto elastic = op.ElasticLoveNumbers().ConvertedTo(scales);

  // At t = 0 the response to a unit load is the unrelaxed (elastic) one.
  auto state = op.ZeroState();
  auto solutions = op.SolveElastic(state, UnitLoad());
  const auto h0 = SurfaceU(op, solutions, 2);
  EXPECT_NEAR(h0, elastic.h_u[2] + elastic.h_phi[2], 1e-10 * std::abs(h0));

  // Stepping a held load relaxes towards the long-term solid's response.
  const auto longTermTable =
      ComputeLoveNumbers(body.rheology.LongTermMaterial(), 4)
          .ConvertedTo(scales);
  const auto hInf = longTermTable.h_u[2] + longTermTable.h_phi[2];
  const auto tau = body.rheology.MinRelaxationTime();
  // The slowest decay modes (the elastic lithosphere's) run to many
  // Maxwell times, so the approach is pushed hard and judged at 1%.
  auto dt = tau;
  for (auto i = 0; i < 1500; ++i) {
    auto [next, sols] =
        op.Step(state, dt, Scheme::ExponentialTrapezoid, UnitLoad());
    state = std::move(next);
    solutions = std::move(sols);
    dt = std::min(dt * 1.1, 1000 * tau);  // stride out as modes die
  }
  const auto hT = SurfaceU(op, solutions, 2);
  EXPECT_NEAR(hT, hInf, 1e-2 * std::abs(hInf));
  // ... and has moved far from the elastic response on the way.
  EXPECT_GT(std::abs(hT - h0), 5.0 * std::abs(hT - hInf));
}

TEST(Viscoelastic, ShortestRelaxationIsWhereTheRheologyRelaxesFastest) {
  // The shortest relaxation time an operator steps (RelaxationSite). At
  // the GLL points it is the rheology's MinRelaxationTime, to the bit (the
  // same largest inverse time, inverted), eta / mu at the base of the
  // PREM-like body's uniform Maxwell mantle, where mu is largest; at the
  // interior Gauss points the largest of the interpolated inverse times,
  // a little longer, in the same element. A Burgers upper mantle's faster
  // branch is shorter still, and is the one named; nothing relaxing, it
  // is infinite. StepRatio is a grid's longest step over it.
  auto body = TestBody::Make(4);
  const auto& mesh = body.material.Mesh();
  const auto& scales = body.model.ModelScales();
  const auto eta = 1e21 / scales.Factor(ViscosityDimensions);
  const auto cmb = 3.48e6 / scales.Length();
  const ViscoelasticOperator gll(body.rheology, 4,
                                 {.memory_points = MemoryPointSet::Gll});
  const auto atNodes = gll.ShortestRelaxation();
  EXPECT_EQ(atNodes.time, body.rheology.MinRelaxationTime());
  auto muMax = 0.0;
  for (auto i = std::size_t{0}; i < body.rheology.MuUnrelaxed().size(); ++i) {
    if (body.rheology.IsActive(i)) {
      muMax = std::max(muMax, body.rheology.MuUnrelaxed()[i]);
    }
  }
  EXPECT_NEAR(atNodes.time, eta / muMax, 1e-14 * atNodes.time);
  EXPECT_EQ(atNodes.branch, 0u);
  EXPECT_EQ(atNodes.layer, mesh.LayerOf(atNodes.element));
  EXPECT_NEAR(atNodes.radius, cmb, 1e-12);
  EXPECT_FALSE(atNodes.lateral);

  const ViscoelasticOperator gauss(body.rheology, 4);
  const auto atPoints = gauss.ShortestRelaxation();
  auto largest = 0.0;
  for (auto i = std::size_t{0}; i < gauss.PointItau(0).size(); ++i) {
    if (gauss.PointMu(0)[i] > 0.0) {
      largest = std::max(largest, gauss.PointItau(0)[i]);
    }
  }
  EXPECT_EQ(atPoints.time, 1.0 / largest);
  EXPECT_EQ(atPoints.element, atNodes.element);
  EXPECT_EQ(atPoints.layer, atNodes.layer);
  EXPECT_GT(atPoints.radius, cmb);
  std::cout << std::format(
      "[shortest relaxation] Gauss points {:.6f} times the nodes', {:.3g} km "
      "above the core\n",
      atPoints.time / atNodes.time,
      (atPoints.radius - cmb) * scales.Length() / 1e3);
  // measured: 1.00045 times the nodes' (the first Gauss point 5.2 km above
  // the core, mu falling upwards there)
  EXPECT_GT(atPoints.time, atNodes.time);
  EXPECT_LT(atPoints.time, 1.01 * atNodes.time);

  // a Burgers upper mantle, its Kelvin element of the unrelaxed modulus and
  // a tenth of the steady viscosity: the faster branch relaxes in 0.049 of
  // the upper mantle's Maxwell time, and is the shortest (measured: 0.118
  // times the Maxwell mantle's)
  const auto b = body.model.DimensionlessView().Boundaries();
  const auto d670 = 5.7011e6 / scales.Length();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  auto kelvin = std::map<std::size_t, Rheology::Kelvin>{};
  for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
    if (body.model.IsFluid(i) || b[i] < 3.4799e6 / scales.Length() ||
        b[i + 1] > 6.1511e6 / scales.Length()) {
      continue;
    }
    viscosity[i] = eta;
    if (b[i] >= d670 - 1e-6) {
      kelvin[i] =
          Rheology::Kelvin{.modulus_fraction = 1.0, .viscosity = 0.1 * eta};
    }
  }
  const auto burgers =
      Rheology::Burgers(body.material, viscosity, kelvin).value();
  const auto fast = ViscoelasticOperator(burgers, 4, {.memory_points =
                                                          MemoryPointSet::Gll})
                        .ShortestRelaxation();
  EXPECT_EQ(fast.time, burgers.MinRelaxationTime());
  EXPECT_EQ(fast.branch, 1u);
  EXPECT_TRUE(kelvin.contains(static_cast<std::size_t>(fast.layer)));
  EXPECT_LT(fast.time, 0.2 * atNodes.time);

  // nothing relaxing: an elastic body
  const auto elastic = Rheology::Create(body.material, {}).value();
  const auto none = ViscoelasticOperator(elastic, 4).ShortestRelaxation();
  EXPECT_EQ(none.time, std::numeric_limits<double>::infinity());
  EXPECT_EQ(none.StepRatio(std::vector<double>{0.0, 1.0}), 0.0);

  // the longest step, a repeated node a step of zero length, in the
  // operator's unit: exact in binary
  const auto half = RelaxationSite{.time = 0.5};
  const auto times = std::vector<double>{0.0, 1.0, 1.0, 3.5, 4.0};
  EXPECT_EQ(half.StepRatio(times), 5.0);
  EXPECT_EQ(half.StepRatio(times, 2.0), 10.0);
  EXPECT_EQ(half.StepRatio(std::vector<double>{3.0}), 0.0);
  EXPECT_EQ(atNodes.StepRatio(std::vector<double>{0.0, atNodes.time}), 1.0);
}

TEST(Viscoelastic, MeasuredOrderOfAccuracy) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const auto tau = body.rheology.MinRelaxationTime();
  const auto T = 4.0 * tau;

  auto advance = [&](Scheme scheme, int steps) {
    auto state = op.ZeroState();
    auto solutions = std::map<int, std::vector<double>>{};
    const auto dt = T / steps;
    for (auto i = 0; i < steps; ++i) {
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
    const auto x1 = advance(scheme, 16);
    const auto x2 = advance(scheme, 32);
    const auto x4 = advance(scheme, 64);
    const auto ratio = (x1 - x2) / (x2 - x4);
    EXPECT_NEAR(ratio, expected, 0.35 * expected)
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Viscoelastic, EffectiveLoveNumberLimits) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const auto tau = body.rheology.MinRelaxationTime();
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

TEST(Viscoelastic, AZeroLengthStepIsTheElasticSolveWithTheMemoryHeld) {
  // A step of zero length (h = 0: e = 1, alpha = beta = 0, every effective
  // weight one) from a state with memory, under a load other than the
  // start's: its memory is the start's, bitwise, its solutions those of
  // the unrelaxed systems under the memory and the new load, and its Love
  // numbers the unrelaxed ones; by the two-phase step of each implicit
  // scheme and by Step under all three. The memory points at the Gauss
  // points and at the GLL nodes. The toroidal partner likewise under a
  // force.
  auto body = TestBody::Make(4);
  const auto tau = body.rheology.MinRelaxationTime();
  const LoadCallable doubled = [](int, double) { return std::vector<double>{2.0}; };
  for (const auto points : {MemoryPointSet::Gauss, MemoryPointSet::Gll}) {
    ViscoelasticOperator op(body.rheology, 4, {.memory_points = points});
    for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler,
                        Scheme::ExponentialEuler}) {
      const auto label = std::format("scheme {}, {}", static_cast<int>(scheme),
                                     points == MemoryPointSet::Gll ? "GLL" : "Gauss");
      auto state = op.ZeroState();
      op.PrimeStrain(state, UnitLoad());
      for (auto i = 0; i < 3; ++i) {
        state = op.Step(state, 0.7 * tau, Scheme::ExponentialTrapezoid, UnitLoad())
                    .first;
      }
      const auto elastic = op.SolveElastic(state, doubled);
      auto worst = 0.0, scale = 0.0;
      auto compare = [&](const std::map<int, std::vector<double>>& got) {
        for (const auto& [l, x] : elastic) {
          for (auto i = std::size_t{0}; i < x.size(); ++i) {
            worst = std::max(worst, std::abs(got.at(l)[i] - x[i]));
            scale = std::max(scale, std::abs(x[i]));
          }
        }
      };
      // (the start strain is the state's cache, its load the earlier one)
      auto [next, solutions] = op.Step(state, 0.0, scheme, doubled);
      EXPECT_EQ(next.m, state.m) << label;
      EXPECT_EQ(next.t, state.t) << label;
      if (scheme == Scheme::ExponentialEuler) {
        // the exponential Euler step hands back its start's solutions
        continue;
      }
      compare(solutions);
      auto step = op.StartStep(state, 0.0, scheme, UnitLoad());
      EXPECT_EQ(step.EffectiveLove().h_u, op.ElasticLoveNumbers().h_u) << label;
      auto [opened, opens] = step.Finish(doubled);
      EXPECT_EQ(opened.m, state.m) << label;
      compare(opens);
      // measured: bitwise (the zero-length step's systems are the
      // unrelaxed ones, assembled alike)
      EXPECT_LT(worst, 1e-13 * scale) << label;
    }
    // the toroidal partner
    const ToroidalViscoelastic toroidal(op);
    auto tstate = toroidal.ZeroState();
    auto force = ToroidalViscoelastic::Fields{};
    for (auto l = 1; l <= 4; ++l) {
      const auto ndof = static_cast<std::size_t>(toroidal.Unrelaxed(l).DofCount());
      auto f = std::vector<double>(ndof * static_cast<std::size_t>(2 * (l + 1)), 0.0);
      for (auto i = std::size_t{0}; i < f.size(); ++i) {
        f[i] = std::sin(0.37 * static_cast<double>(i + l));
      }
      force.emplace(l, std::move(f));
    }
    toroidal.SetStrain(tstate, toroidal.SolveElastic(tstate, force));
    for (auto i = 0; i < 3; ++i) {
      tstate = toroidal.Step(tstate, 0.7 * tau, Scheme::ExponentialTrapezoid, force)
                   .first;
    }
    auto twice = force;
    for (auto& [l, f] : twice) {
      for (auto& v : f) v *= 2.0;
    }
    for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
      auto [next, w] = toroidal.Step(tstate, 0.0, scheme, twice);
      EXPECT_EQ(next.m, tstate.m);
      const auto want = toroidal.SolveElastic(tstate, twice);
      auto worst = 0.0, scale = 0.0;
      for (const auto& [l, x] : want) {
        for (auto i = std::size_t{0}; i < x.size(); ++i) {
          worst = std::max(worst, std::abs(w.at(l)[i] - x[i]));
          scale = std::max(scale, std::abs(x[i]));
        }
      }
      // measured: bitwise
      EXPECT_LT(worst, 1e-13 * scale);
    }
  }
}

TEST(StepRecipe, TheRecipesAndTheEngineOnAScalarEquation) {
  // The one-stage schemes are one stage of themselves; any scheme's step
  // of zero length is one stage of zero length; Sdirk2 is Alexander's
  // tableau, two backward Euler stages of length gamma h (gamma^2 - 2 gamma
  // + 1/2 = 0, L-stability), the second from (1 - r) m_n + r Y_1. On
  // y' = k (g(t) - y), each stage a backward Euler step under g at its end
  // time, the engine gives Sdirk2's amplification (1 + (1 - 2 gamma) z) /
  // (1 - gamma z)^2 to rounding, vanishing as z -> -infinity, and its
  // second order under a forcing.
  constexpr auto g = kSdirk2Gamma;
  EXPECT_NEAR(g, 1.0 - 1.0 / std::sqrt(2.0), 1e-16);
  EXPECT_LT(std::abs(g * g - 2.0 * g + 0.5), 1e-16);
  for (const auto scheme :
       {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto one = RecipeOf(scheme, 0.7);
    ASSERT_EQ(one.stages.size(), 1u);
    EXPECT_EQ(one.stages[0].family, scheme);
    EXPECT_EQ(one.stages[0].c, 1.0);
    EXPECT_EQ(StageLength(one.stages[0], 0.7), 0.7);
  }
  const auto still = RecipeOf(Scheme::Sdirk2, 0.0);
  ASSERT_EQ(still.stages.size(), 1u);
  EXPECT_EQ(still.stages[0].family, Scheme::BackwardEuler);
  const auto two = RecipeOf(Scheme::Sdirk2, 0.7);
  ASSERT_EQ(two.stages.size(), 2u);
  for (const auto& stage : two.stages) {
    EXPECT_EQ(stage.family, Scheme::BackwardEuler);
    EXPECT_EQ(stage.scale, g);
  }
  EXPECT_EQ(StageLength(two.stages[0], 0.7), StageLength(two.stages[1], 0.7));
  EXPECT_EQ(two.stages[0].c, g);
  EXPECT_EQ(two.stages[1].c, 1.0);
  ASSERT_EQ(two.stages[1].start.size(), 2u);
  const auto r = (1.0 - g) / g;
  EXPECT_NEAR(two.stages[1].start[0].second, 1.0 - r, 1e-15);
  EXPECT_NEAR(two.stages[1].start[1].second, r, 1e-15);
  EXPECT_THROW((void)RecipeOf(Scheme::ExponentialEuler, 0.7),
               std::invalid_argument);
  EXPECT_THROW((void)RecipeOf(Scheme::Sdirk2, -0.7), std::invalid_argument);

  struct Scalar {
    double t = 0.0;
    double y = 0.0;
  };
  const auto step = [](double k, const std::function<double(double)>& forcing,
                       const Scalar& from, double h) {
    return StepByRecipe(
        RecipeOf(Scheme::Sdirk2, h), h, from,
        [&](const Scalar& x, const RecipeStage&, double length, std::size_t) {
          const auto t = x.t + length;
          return Scalar{t,
                        (x.y + length * k * forcing(t)) / (1.0 + length * k)};
        },
        [](std::span<const std::pair<double, const Scalar*>> terms,
           const RecipeStage&) {
          auto out = Scalar{terms.front().second->t, 0.0};
          for (const auto& [a, x] : terms) out.y += a * x->y;
          return out;
        });
  };
  const auto none = [](double) { return 0.0; };
  for (const auto z : {0.1, 1.0, 10.0, 1e3, 1e8}) {
    const auto next = step(z, none, Scalar{0.25, 1.0}, 1.0);
    const auto want =
        (1.0 - (1.0 - 2.0 * g) * z) / ((1.0 + g * z) * (1.0 + g * z));
    EXPECT_NEAR(next.y, want, 1e-15) << "z " << z;
    EXPECT_EQ(next.t, 1.25);
  }
  EXPECT_LT(std::abs(step(1e8, none, Scalar{0.0, 1.0}, 1.0).y), 1e-7);
  // y' = k (sin t - y) to t = 2 against its exact solution
  const auto k = 3.0;
  const auto exact = [&](double t) {
    const auto c = 1.0 / (1.0 + k * k);
    return k * c * (k * std::sin(t) - std::cos(t)) +
           (1.0 + k * c) * std::exp(-k * t);
  };
  auto errors = std::vector<double>{};
  for (const auto steps : {10, 20, 40}) {
    auto x = Scalar{0.0, 1.0};
    for (auto n = 0; n < steps; ++n) {
      x = step(k, [](double t) { return std::sin(t); }, x, 2.0 / steps);
    }
    errors.push_back(std::abs(x.y - exact(2.0)));
  }
  for (auto i = std::size_t{1}; i < errors.size(); ++i) {
    EXPECT_NEAR(std::log2(errors[i - 1] / errors[i]), 2.0, 0.15) << i;
  }
}

TEST(Viscoelastic, Sdirk2StepsAtSecondOrder) {
  // The operator's own step under Sdirk2 (two backward Euler stages, each
  // closed under the load channels at its end time) is the composition of
  // those stages by hand, bitwise, and converges at second order under a
  // held load, where backward Euler is first order (MeasuredOrderOfAccuracy).
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const auto tau = body.rheology.MinRelaxationTime();
  const LoadCallable rising = [](int, double t) {
    return std::vector<double>{1.0 + 0.3 * t};
  };

  auto state = op.ZeroState();
  state =
      op.Step(state, 0.6 * tau, Scheme::ExponentialTrapezoid, UnitLoad()).first;
  const auto h = 0.9 * tau;
  const auto recipe = RecipeOf(Scheme::Sdirk2, h);
  const auto length = StageLength(recipe.stages[0], h);
  auto first = op.Step(state, length, Scheme::BackwardEuler, rising).first;
  using Term = std::pair<double, const ViscoelasticState*>;
  const auto terms = std::array{Term{recipe.stages[1].start[0].second, &state},
                                Term{recipe.stages[1].start[1].second, &first}};
  auto combined = CombinedState(terms);
  combined.t = state.t + (1.0 - kSdirk2Gamma) * h;
  const auto [byHand, handSolutions] =
      op.Step(combined, length, Scheme::BackwardEuler, rising);
  const auto [stepped, solutions] = op.Step(state, h, Scheme::Sdirk2, rising);
  EXPECT_EQ(stepped.m, byHand.m);
  EXPECT_EQ(solutions, handSolutions);
  EXPECT_EQ(stepped.t, state.t + h);

  const auto T = 4.0 * tau;
  auto advance = [&](int steps) {
    auto x = op.ZeroState();
    auto at = std::map<int, std::vector<double>>{};
    for (auto i = 0; i < steps; ++i) {
      auto [next, sols] = op.Step(x, T / steps, Scheme::Sdirk2, UnitLoad());
      x = std::move(next);
      at = std::move(sols);
    }
    return SurfaceU(op, at, 2);
  };
  const auto x1 = advance(16), x2 = advance(32), x4 = advance(64);
  const auto ratio = (x1 - x2) / (x2 - x4);
  std::cout << std::format("[sdirk2] Richardson ratio {:.4f}\n", ratio);
  // measured: 4.02
  EXPECT_NEAR(ratio, 4.0, 0.2);
}

TEST(Viscoelastic, Sdirk2ZeroLengthStepAndWhatItRefuses) {
  // A step of zero length under Sdirk2 is backward Euler's, the elastic
  // solve with the memory held
  // (AZeroLengthStepIsTheElasticSolveWithTheMemoryHeld): by Step and by
  // StartStep, the toroidal partner alike. At a step of nonzero length what
  // takes one stage or one set of weights refuses it: StartStep, the
  // effective solid and its Love numbers, the toroidal step.
  auto body = TestBody::Make(4);
  const auto tau = body.rheology.MinRelaxationTime();
  const LoadCallable doubled = [](int, double) {
    return std::vector<double>{2.0};
  };
  for (const auto points : {MemoryPointSet::Gauss, MemoryPointSet::Gll}) {
    ViscoelasticOperator op(body.rheology, 4, {.memory_points = points});
    auto state = op.ZeroState();
    op.PrimeStrain(state, UnitLoad());
    for (auto i = 0; i < 3; ++i) {
      state = op.Step(state, 0.7 * tau, Scheme::Sdirk2, UnitLoad()).first;
    }
    const auto elastic = op.SolveElastic(state, doubled);
    const auto [next, solutions] = op.Step(state, 0.0, Scheme::Sdirk2, doubled);
    EXPECT_EQ(next.m, state.m);
    auto worst = 0.0, scale = 0.0;
    for (const auto& [l, x] : elastic) {
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        worst = std::max(worst, std::abs(solutions.at(l)[i] - x[i]));
        scale = std::max(scale, std::abs(x[i]));
      }
    }
    // measured: bitwise
    EXPECT_LT(worst, 1e-13 * scale);
    auto opened = op.StartStep(state, 0.0, Scheme::Sdirk2, UnitLoad());
    EXPECT_EQ(opened.EffectiveLove().h_u, op.ElasticLoveNumbers().h_u);
    const auto finished = opened.Finish(doubled).first;
    EXPECT_EQ(finished.m, state.m);
    EXPECT_THROW((void)op.StartStep(state, tau, Scheme::Sdirk2, UnitLoad()),
                 std::invalid_argument);
    EXPECT_THROW((void)op.EffectiveLoveNumbers(Scheme::Sdirk2, tau),
                 std::invalid_argument);
    if (points == MemoryPointSet::Gll) {
      EXPECT_THROW((void)op.EffectiveMaterial(Scheme::Sdirk2, tau),
                   std::invalid_argument);
    }
    const ToroidalViscoelastic toroidal(op);
    auto tstate = toroidal.ZeroState();
    toroidal.SetStrain(tstate, toroidal.SolveElastic(tstate, {}));
    EXPECT_EQ(toroidal.Step(tstate, 0.0, Scheme::Sdirk2, {}).first.m, tstate.m);
    EXPECT_THROW((void)toroidal.Step(tstate, tau, Scheme::Sdirk2, {}),
                 std::invalid_argument);
  }
}

TEST(Viscoelastic, AStepHoldsItsFlavourThroughTheCachesEvictions) {
  // A step holds its (scheme, dt) flavour, of which the operator's cache
  // keeps the four most recently used: steps of five other sizes taken
  // while it is open (between opening and finishing, and inside the load
  // channel it is finished under, between its reads of the flavour's unit
  // responses) evict the flavour from the cache and leave the step's
  // memory solutions, end state and end solutions those of a step taken
  // with nothing between, to rounding. By the opened step of each
  // one-stage implicit scheme, and by Step under those and Sdirk2 (whose
  // stages hold their own flavour); at the Gauss points and the GLL nodes.
  auto body = TestBody::Make(4);
  const auto tau = body.rheology.MinRelaxationTime();
  using Fields = std::map<int, std::vector<double>>;
  auto worst = [](const Fields& got, const Fields& want) {
    auto error = 0.0, scale = 0.0;
    for (const auto& [l, x] : want) {
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        error = std::max(error, std::abs(got.at(l)[i] - x[i]));
        scale = std::max(scale, std::abs(x[i]));
      }
    }
    return scale > 0.0 ? error / scale : error;
  };
  for (const auto points : {MemoryPointSet::Gauss, MemoryPointSet::Gll}) {
    ViscoelasticOperator op(body.rheology, 4, {.memory_points = points});
    for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler,
                        Scheme::Sdirk2}) {
      const auto label = std::format("scheme {}, {}", static_cast<int>(scheme),
                                     points == MemoryPointSet::Gll ? "GLL" : "Gauss");
      auto state = op.ZeroState();
      op.PrimeStrain(state, UnitLoad());
      state = op.Step(state, 0.7 * tau, scheme, UnitLoad()).first;
      const auto dt = 0.3 * tau;
      auto others = [&] {
        for (auto k = 1; k <= 5; ++k) {
          (void)op.Step(state, (0.3 + 0.11 * k) * tau, scheme, UnitLoad());
        }
      };
      // a unit load whose first reading steps at the five other sizes
      auto armed = true;
      const LoadCallable evicting = [&](int l, double t) {
        if (armed) {
          armed = false;
          others();
        }
        return UnitLoad()(l, t);
      };

      const auto [want, wantSolutions] = op.Step(state, dt, scheme, UnitLoad());
      const auto [got, gotSolutions] = op.Step(state, dt, scheme, evicting);
      EXPECT_FALSE(armed) << label;
      EXPECT_LT(worst(got.m, want.m), 1e-12) << label;
      EXPECT_LT(worst(gotSolutions, wantSolutions), 1e-12) << label;
      if (scheme == Scheme::Sdirk2) continue;

      const auto freshMemory =
          op.StartStep(state, dt, scheme, UnitLoad()).MemorySolutions();
      const auto held = op.StartStep(state, dt, scheme, UnitLoad());
      others();
      EXPECT_LT(worst(held.MemorySolutions(), freshMemory), 1e-12) << label;
      others();
      armed = true;
      const auto [next, solutions] = held.Finish(evicting);
      EXPECT_FALSE(armed) << label;
      EXPECT_LT(worst(next.m, want.m), 1e-12) << label;
      EXPECT_LT(worst(solutions, wantSolutions), 1e-12) << label;
    }
  }
}
