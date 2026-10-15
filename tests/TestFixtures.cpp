// The stage-0 verification gate (plan §6): reference gravity against
// planetmodel on the identical deck file, read by both codes with the same
// (not-a-knot) interpolation, so agreement is expected near machine
// precision.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"

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
