// Deck construction, validation, file reading, and the legacy sl3d reader.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "sl3dpp/Deck.hpp"

using sl3dpp::Deck;
using sl3dpp::DeckFormat;

namespace {

std::filesystem::path WriteFile(const std::string& name,
                                const std::string& text) {
  auto path = std::filesystem::path(testing::TempDir()) / name;
  std::ofstream(path) << text;
  return path;
}

}  // namespace

TEST(Deck, CreateAndLayering) {
  // Two layers split by the repeated radius at 1.
  auto deck = Deck::Create({0.0, 0.5, 1.0, 1.0, 1.5, 2.0},
                           {{"rho", {5.0, 5.0, 5.0, 3.0, 3.0, 3.0}}})
                  .value();
  EXPECT_EQ(deck.KnotCount(), 6u);
  EXPECT_EQ(deck.LayerCount(), 2u);
  ASSERT_EQ(deck.Boundaries().size(), 3u);
  EXPECT_DOUBLE_EQ(deck.Boundaries()[1], 1.0);
  EXPECT_EQ(deck.Layers()[0].first, 0u);
  EXPECT_EQ(deck.Layers()[0].second, 3u);
  EXPECT_EQ(deck.Layers()[1].first, 3u);
  EXPECT_EQ(deck.Layers()[1].second, 6u);
  EXPECT_TRUE(deck.Has("rho"));
  EXPECT_FALSE(deck.Has("mu"));
}

TEST(Deck, RefusesBadStructure) {
  // Decreasing radius.
  EXPECT_FALSE(Deck::Create({0.0, 1.0, 0.5}, {{"rho", {1, 1, 1}}}).has_value());
  // A radius repeated three times.
  EXPECT_FALSE(
      Deck::Create({0.0, 1.0, 1.0, 1.0, 2.0}, {{"rho", {1, 1, 1, 1, 1}}})
          .has_value());
  // A one-knot layer.
  EXPECT_FALSE(Deck::Create({0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0},
                            {{"rho", {1, 1, 1, 1, 1, 1, 1}}})
                   .has_value());
  // Column length mismatch.
  EXPECT_FALSE(Deck::Create({0.0, 1.0}, {{"rho", {1, 1, 1}}}).has_value());
}

TEST(Deck, ReadsASimpleTable) {
  auto path = WriteFile("simple.deck",
                        "# header line\n"
                        "0.0  5000.0  1.0\n"
                        "\n"
                        "1000.0  4000.0  nan\n"
                        "2000.0  3000.0  nan\n");
  auto deck = Deck::Read(path, DeckFormat("test", {"rho", "extra"}, 1)).value();
  EXPECT_EQ(deck.KnotCount(), 3u);
  EXPECT_EQ(deck.Header().size(), 1u);
  EXPECT_DOUBLE_EQ(deck.Column("rho")[1], 4000.0);
  EXPECT_TRUE(std::isnan(deck.Column("extra")[2]));
}

TEST(Deck, RefusesWrongWidth) {
  auto path = WriteFile("wrong.deck",
                        "0.0  5000.0\n"
                        "1000.0  4000.0  7.0\n");
  auto result = Deck::Read(path, DeckFormat("test", {"rho"}, 0));
  ASSERT_FALSE(result.has_value());
  // planetmodel's reader words the width mismatch in values per row.
  EXPECT_NE(result.error().message.find("values"), std::string::npos);
}

TEST(Deck, MineosFormatAdmitsBothWidths) {
  auto format = sl3dpp::MineosFormat();
  auto path = WriteFile("iso.deck",
                        "title\n1 1.0 1\n4 0 0\n"
                        "0.0      13000 11000 3500 100 100\n"
                        "1000000  12000 11000 3500 100 100\n"
                        "2000000  11000 10000 3000 100 100\n"
                        "6371000  3000   6000 3000 100 100\n");
  auto deck = Deck::Read(path, format).value();
  EXPECT_TRUE(deck.Has("vp"));
  EXPECT_FALSE(deck.Has("vpv"));
  EXPECT_EQ(deck.Header().size(), 3u);
}

TEST(Deck, ReadsTheLegacySl3dModel) {
  auto deck = Deck::ReadLegacySl3d(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                                   "prem.simp.vm1")
                  .value();
  EXPECT_EQ(deck.KnotCount(), 222u);
  EXPECT_DOUBLE_EQ(deck.Radius().back(), 6.371e6);
  // kappa converted from GPa, viscosity from log10 (0 meaning absent).
  EXPECT_NEAR(deck.Column("kappa")[0], 1425.3394787252205e9, 1e2);
  EXPECT_TRUE(std::isnan(deck.Column("viscosity")[0]));
  const auto viscosity = deck.Column("viscosity");
  auto anyViscous = false;
  for (auto v : viscosity) anyViscous = anyViscous || v > 1e20;
  EXPECT_TRUE(anyViscous);  // the mantle's 10^21.7
}
