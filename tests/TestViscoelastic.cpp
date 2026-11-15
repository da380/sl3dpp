// The viscoelastic stepper: weight algebra, rheology construction (the
// Burgers body's Prony terms against its modulus and creep function, and
// the bodies a model's fields state, planetmodel's prony and burgers decks
// among them), the elastic and long-term limits, measured order of accuracy,
// the exact inverse step, and the effective-operator limits. The pyslfp
// comparison lives in TestFixtures.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
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
#include "sl3dpp/StepWeights.hpp"
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

  // The inverse step: composing the update at h with the update at -h
  // (strains swapped) is the identity, for any strains.
  const auto h = 0.7, m0 = 0.3, d0 = -1.1, d1 = 0.8;
  const auto f = ExponentialTrapezoidWeights(h);
  const auto r = ExponentialTrapezoidWeights(-h);
  const auto m1 = f.e * m0 + f.alpha * d0 + f.beta * d1;
  EXPECT_NEAR(r.e * m1 + r.alpha * d1 + r.beta * d0, m0, 1e-14);

  // Derivatives by central differences.
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

TEST(Viscoelastic, ExactInverseStep) {
  auto body = TestBody::Make(2);
  ViscoelasticOperator op(body.rheology, 2);
  const auto tau = body.rheology.MinRelaxationTime();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto state = op.ZeroState();
    op.PrimeStrain(state, UnitLoad());
    // March a few steps so the state is non-trivial, then invert them.
    auto history = std::vector<ViscoelasticState>{state};
    const auto dt = 0.8 * tau;
    for (auto i = 0; i < 3; ++i) {
      auto [next, sols] = op.Step(history.back(), dt, scheme, UnitLoad());
      history.push_back(std::move(next));
    }
    auto current = history.back();
    for (auto i = 2; i >= 0; --i) {
      auto [previous, sols] = op.StepBack(current, dt, scheme, UnitLoad());
      auto scale = 0.0, diff = 0.0;
      for (const auto& [l, m] : history.back().m) {
        for (double v : m) scale = std::max(scale, std::abs(v));
      }
      for (const auto& [l, m] : history[i].m) {
        const auto& mBack = previous.m.at(l);
        for (auto j = std::size_t{0}; j < m.size(); ++j) {
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
