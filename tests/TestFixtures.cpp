// The stage-0 verification gate (plan §6): reference gravity against
// planetmodel on the identical deck file, read by both codes with the same
// (not-a-knot) interpolation, so agreement is expected near machine
// precision.

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace {

struct GravityFixture {
  double mass = 0;
  double moment = 0;
  std::vector<std::pair<double, double>> gravity;  // (r, g), SI
};

GravityFixture ReadFixture(const std::filesystem::path& path) {
  std::ifstream file(path);
  EXPECT_TRUE(file.is_open()) << path;
  GravityFixture fixture;
  bool inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    std::string head;
    if (!inTable) {
      row >> head;
      if (head == "mass") {
        row >> fixture.mass;
      } else if (head == "moment") {
        row >> fixture.moment;
      } else if (head == "gravity") {
        inTable = true;
      }
      continue;
    }
    double r = 0, g = 0;
    row >> r >> g;
    fixture.gravity.emplace_back(r, g);
  }
  return fixture;
}

}  // namespace

TEST(Stage1Gate, LoveNumbersMatchPyslfp) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = sl3dpp::Deck::Read(
                  data / "prem8.elastic.deck",
                  sl3dpp::DeckFormat(
                      "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
                  .value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  constexpr int lmax = 32;
  auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = sl3dpp::Material::Create(model, mesh).value();
  auto ln = sl3dpp::ComputeLoveNumbers(material, lmax);

  // Parse the pyslfp reference.
  std::ifstream file(data / "stage1_love_numbers.txt");
  ASSERT_TRUE(file.is_open());
  double radius = 0, surfaceGravity = 0, fluidTidal = 0;
  double axial[5] = {};
  std::vector<std::array<double, 13>> table;
  bool inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!inTable) {
      std::string head;
      row >> head;
      if (head == "radius") row >> radius;
      if (head == "surface_gravity") row >> surfaceGravity;
      if (head == "axial") {
        for (auto& v : axial) row >> v;
      }
      if (head == "fluid_tidal_2") row >> fluidTidal;
      if (head == "table") inTable = true;
      continue;
    }
    auto& entries = table.emplace_back();
    for (auto& v : entries) row >> v;
  }
  ASSERT_EQ(table.size(), static_cast<std::size_t>(lmax + 1));

  constexpr double tol = 1e-10;
  EXPECT_NEAR(ln.radius, radius, tol * radius);
  EXPECT_NEAR(ln.surface_gravity, surfaceGravity, tol * surfaceGravity);
  const std::vector<double>* columns[12] = {
      &ln.h_u, &ln.l_u, &ln.k_u, &ln.h_phi, &ln.l_phi, &ln.k_phi,
      &ln.h_v, &ln.l_v, &ln.k_v, &ln.h_t,   &ln.l_t,   &ln.k_t};
  for (const auto& entries : table) {
    const int l = static_cast<int>(entries[0]);
    // One scale per degree: tiny channels (a removed dof's zeros) are
    // compared absolutely against the degree's magnitude.
    double scale = 0.0;
    for (int c = 0; c < 12; ++c)
      scale = std::max(scale, std::abs(entries[c + 1]));
    for (int c = 0; c < 12; ++c) {
      EXPECT_NEAR((*columns[c])[l], entries[c + 1], tol * scale)
          << "l = " << l << ", column " << c;
    }
  }
  // k_c and m_phi vanish by the shell theorem, so each code computes pure
  // round-off there (~1e-11 of m_u, uncorrelated): the floor is absolute,
  // on the natural scale m_u.
  const double values[5] = {ln.h_c, ln.k_c, ln.m_u, ln.m_phi, ln.m_c};
  for (int i = 0; i < 5; ++i) {
    const double scale =
        std::max({std::abs(axial[i]), std::abs(values[i]), 1e-30});
    EXPECT_NEAR(values[i], axial[i],
                std::max(tol * scale, 1e-10 * std::abs(axial[2])))
        << "axial " << i;
  }
  EXPECT_NEAR(sl3dpp::FluidTidalNumber(material, 2), fluidTidal,
              tol * fluidTidal);
}

TEST(Stage0Gate, GravityMatchesPlanetmodel) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = sl3dpp::Deck::Read(data / "prem8.rho.deck",
                                 sl3dpp::DeckFormat("rho-only", {"rho"}, 0))
                  .value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  const auto fixture = ReadFixture(data / "stage0_gravity.txt");
  ASSERT_GT(fixture.gravity.size(), 10u);

  EXPECT_NEAR(model.Mass(), fixture.mass, 1e-13 * fixture.mass);
  EXPECT_NEAR(model.MomentOfInertia(), fixture.moment, 1e-13 * fixture.moment);
  for (const auto& [r, g] : fixture.gravity) {
    EXPECT_NEAR(model.Gravity(r), g, 1e-13 * std::max(g, 1.0))
        << "at r = " << r;
  }
}
