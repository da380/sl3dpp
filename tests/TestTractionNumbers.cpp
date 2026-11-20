/**
 * @file TestTractionNumbers.cpp
 * @brief The non-gravitating degree operator and the traction numbers:
 * the E1 comparison with pyslfp's traction numbers of two solid decks,
 * without and with gravity (identical discrete problem, agreement near
 * machine precision), the degree-0 closed form, reciprocity, and the
 * refusals. Fixtures from scripts/make_e1_fixtures.py.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace {

constexpr auto kLmax = 8;

sl3dpp::DeckFormat ElasticFormat() {
  return sl3dpp::DeckFormat("elastic",
                            {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0);
}

sl3dpp::Material MaterialFromDeck(const std::string& stem) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = sl3dpp::Deck::Read(data / stem, ElasticFormat()).value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = kLmax}).value();
  return sl3dpp::Material::Create(model, mesh).value();
}

struct FixtureTable {
  double radius = 0;
  // rows[l] = {h_u, l_u, h_v, l_v}, SI.
  std::vector<std::array<double, 4>> rows;
};

/// The fixture, keyed by "<model>:<gravitating>".
std::map<std::string, FixtureTable> ReadFixture() {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  std::ifstream file(data / "e1_traction_numbers.txt");
  EXPECT_TRUE(file.is_open());
  std::map<std::string, FixtureTable> out;
  FixtureTable* current = nullptr;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    auto head = std::string{};
    row >> head;
    if (head == "model") {
      std::string name, word;
      int gravitating = 0;
      double radius = 0;
      row >> name >> word >> gravitating >> word >> radius;
      current = &out[name + ":" + std::to_string(gravitating)];
      current->radius = radius;
      continue;
    }
    if (head == "table") continue;
    if (current == nullptr) {
      ADD_FAILURE() << "table row outside a model block";
      continue;
    }
    // std::stod rather than operator>>, which refuses "nan".
    auto& entries = current->rows.emplace_back();
    std::istringstream full(line);
    std::string token;
    full >> token;  // the degree
    for (auto& v : entries) {
      full >> token;
      v = std::stod(token);
    }
  }
  return out;
}

void ExpectMatches(const sl3dpp::TractionNumbers& ours,
                   const FixtureTable& theirs) {
  // each code's solves' rounding: 6e-12 of a degree's scale at worst
  constexpr auto tol = 1e-9;
  EXPECT_NEAR(ours.radius, theirs.radius, tol * theirs.radius);
  ASSERT_EQ(ours.degrees.size(), theirs.rows.size());
  for (auto l = 0; l <= kLmax; ++l) {
    const double mine[4] = {ours.h_u[l], ours.l_u[l], ours.h_v[l],
                            ours.l_v[l]};
    auto scale = 0.0;
    for (auto c = 0; c < 4; ++c) {
      if (!std::isnan(theirs.rows[l][c])) {
        scale = std::max(scale, std::abs(theirs.rows[l][c]));
      }
    }
    for (auto c = 0; c < 4; ++c) {
      if (std::isnan(theirs.rows[l][c])) {
        EXPECT_TRUE(std::isnan(mine[c])) << "l = " << l << ", column " << c;
      } else {
        EXPECT_NEAR(mine[c], theirs.rows[l][c], tol * scale)
            << "l = " << l << ", column " << c;
      }
    }
  }
}

}  // namespace

TEST(E1Gate, TractionNumbersMatchPyslfp) {
  const auto fixture = ReadFixture();
  ASSERT_EQ(fixture.size(), 4u);
  for (const auto& name : {"homogeneous", "two_solid"}) {
    const auto material = MaterialFromDeck(std::string("e1_") + name + ".deck");
    for (const auto gravitating : {false, true}) {
      const auto ours =
          sl3dpp::ComputeTractionNumbers(material, kLmax, gravitating);
      ExpectMatches(ours,
                    fixture.at(std::string(name) + ":" +
                               std::to_string(static_cast<int>(gravitating))));
    }
  }
}

TEST(TractionNumbers, DegreeZeroIsUniformCompression) {
  // A unit pressure compresses the homogeneous sphere uniformly:
  // U(a) = -a / (3 kappa), with kappa = rho (vp^2 - 4 vs^2 / 3), to the
  // solve's rounding (3e-13 here).
  const auto material = MaterialFromDeck("e1_homogeneous.deck");
  const auto tn = sl3dpp::ComputeTractionNumbers(material, 0, false);
  constexpr auto rho = 5500.0, vp = 10000.0, vs = 5500.0, a = 6371.0e3;
  const auto kappa = rho * (vp * vp - 4.0 * vs * vs / 3.0);
  EXPECT_NEAR(tn.h_u[0], -a / (3.0 * kappa), 1e-10 * a / (3.0 * kappa));
}

TEST(TractionNumbers, Reciprocity) {
  const auto material = MaterialFromDeck("e1_two_solid.deck");
  for (const auto gravitating : {false, true}) {
    const auto tn =
        sl3dpp::ComputeTractionNumbers(material, kLmax, gravitating);
    EXPECT_LT(tn.ReciprocityResidual(), 1e-10);
  }
}

TEST(TractionNumbers, TieToTheGravitatingLoveNumbers) {
  // On the full problem the traction channels are the load-force channels
  // divided by g, by construction of the forcings.
  const auto material = MaterialFromDeck("e1_two_solid.deck");
  const auto tn = sl3dpp::ComputeTractionNumbers(material, kLmax, true);
  const auto love = sl3dpp::ComputeLoveNumbers(material, kLmax);
  const auto g = love.surface_gravity;
  for (auto l = 0; l <= kLmax; ++l) {
    EXPECT_NEAR(tn.h_u[l] * g, love.h_u[l], 1e-13 * std::abs(love.h_u[l]));
    EXPECT_NEAR(tn.l_v[l] * g, love.l_v[l],
                1e-13 * std::max(std::abs(love.l_v[l]), 1e-30));
  }
}

TEST(TractionNumbers, Refusals) {
  // Without gravity degree 1 is kept, in the frame of the centre of mass
  // (PointSource.AHomogeneousSphereWithoutGravityInClosedForm gates it); a
  // surface traction there has a net force, so its numbers stay NaN.
  const auto material = MaterialFromDeck("e1_homogeneous.deck");
  const auto one = sl3dpp::DegreeOperator(material, 1, {.gravitating = false});
  EXPECT_EQ(one.PinnedDofs().size(), 1u);
  EXPECT_EQ(one.RigidTranslation().size(),
            static_cast<std::size_t>(one.DofCount()));
  EXPECT_TRUE(std::isnan(
      sl3dpp::ComputeTractionNumbers(material, 2, false).h_u[1]));
  EXPECT_THROW(sl3dpp::DegreeOperator(material, 1,
                                      {.gravitating = false,
                                       .pin_fraction = 1.0}),
               std::invalid_argument);
  const auto two = sl3dpp::DegreeOperator(material, 2, {.gravitating = false});
  EXPECT_TRUE(two.PinnedDofs().empty());
  EXPECT_THROW((void)two.RigidTranslation(), std::invalid_argument);
  EXPECT_THROW((void)two.CentreOfMassRow(), std::invalid_argument);

  // A fluid core: above degree zero the system stops at its top, the
  // mantle's base traction-free; at degree zero it keeps the whole body.
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck =
      sl3dpp::Deck::Read(data / "prem8.elastic.deck", ElasticFormat()).value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = 4}).value();
  auto prem = sl3dpp::Material::Create(model, mesh).value();
  auto base = 0;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (prem.IsFluid(e)) base = e + 1;
  }
  ASSERT_GT(base, 0);
  for (const auto l : {1, 2, 3}) {
    const auto op = sl3dpp::DegreeOperator(prem, l, {.gravitating = false});
    EXPECT_EQ(op.FirstElement(), base) << l;
  }
  EXPECT_EQ(sl3dpp::DegreeOperator(prem, 0, {.gravitating = false})
                .FirstElement(),
            0);

  // More than one fluid region is refused.
  auto layered = std::vector<double>{};
  auto rho = std::vector<double>{}, vp = rho, vs = rho;
  const std::array<std::array<double, 3>, 5> layers{
      {{1.0e6, 9.0e3, 4.0e3},
       {2.0e6, 9.0e3, 0.0},
       {3.0e6, 9.0e3, 4.0e3},
       {3.5e6, 9.0e3, 0.0},
       {6.371e6, 9.0e3, 4.0e3}}};
  auto below = 0.0;
  for (const auto& [top, p, s] : layers) {
    for (const auto r : {below, top}) {
      layered.push_back(r);
      rho.push_back(5000.0);
      vp.push_back(p);
      vs.push_back(s);
    }
    below = top;
  }
  auto twoFluid = sl3dpp::Deck::Create(
                      std::move(layered),
                      {{"rho", rho}, {"vpv", vp}, {"vsv", vs}, {"vph", vp},
                       {"vsh", vs}, {"eta", std::vector<double>(rho.size(), 1.0)}})
                      .value();
  auto twoModel = sl3dpp::EarthModel::FromDeck(twoFluid).value();
  auto twoMesh = sl3dpp::RadialMesh::Create(twoModel, {.lmax = 4}).value();
  auto twoMaterial = sl3dpp::Material::Create(twoModel, twoMesh).value();
  EXPECT_THROW(
      sl3dpp::DegreeOperator(twoMaterial, 2, {.gravitating = false}),
      std::invalid_argument);
  EXPECT_NO_THROW(sl3dpp::DegreeOperator(twoMaterial, 2));
}
