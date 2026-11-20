/**
 * @file TestModelField.cpp
 * @brief The io/ boundary for a model file's field taken as it stands
 * (io::ModelField): the fixtures of TestModelViscosity read as plain
 * fields. The fully 3D viscosity at the Python's off-grid probe points,
 * where both ends interpolate the same stored numbers by the same stated
 * rules; a radial file, read within a layer at its boundary; a field the
 * file does not hold; and a file of another body.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/io/ModelField.hpp"

namespace {

using namespace sl3dpp;
using io::ModelField;

std::filesystem::path DataPath(const std::string& name) {
  return std::filesystem::path(SL3DPP_TEST_DATA_DIR) / name;
}

EarthModel Prem8() {
  auto deck =
      Deck::Read(DataPath("prem8.elastic.deck"),
                 DeckFormat("elastic",
                            {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  return EarthModel::FromDeck(deck).value();
}

/// The model's layers of the fixtures' viscous mantle.
std::vector<std::size_t> ViscousLayers(const EarthModel& model) {
  const auto length = model.ModelScales().Length();
  const auto b = model.DimensionlessView().Boundaries();
  auto out = std::vector<std::size_t>{};
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && b[i] >= 3.4799e6 / length &&
        b[i + 1] <= 6.1511e6 / length) {
      out.push_back(i);
    }
  }
  return out;
}

struct Probe {
  double r = 0.0;  // SI
  double theta = 0.0;
  double phi = 0.0;
  double value = 0.0;  // SI, the Python's reading
};

std::vector<Probe> ReadProbes(const std::filesystem::path& path) {
  auto in = std::ifstream(path);
  EXPECT_TRUE(in.is_open()) << path;
  auto probes = std::vector<Probe>{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.front() == '#') continue;
    auto probe = Probe{};
    std::istringstream(line) >> probe.r >> probe.theta >> probe.phi >>
        probe.value;
    probes.push_back(probe);
  }
  EXPECT_FALSE(probes.empty()) << path;
  return probes;
}

}  // namespace

// The field as it stands is the file's own reading: the fully 3D
// viscosity at off-grid points, to rounding, in SI under its dimensions
// and as stored (the fixture's scales are SI, so the same numbers).
TEST(ModelField, AFieldAsItStandsMatchesThePythonReading) {
  const auto model = Prem8();
  const auto length = model.ModelScales().Length();
  for (const auto dimensioned : {true, false}) {
    const auto read = ModelField::Read(
        DataPath("viscosity_full3d.nc"), model, "viscosity",
        dimensioned ? std::optional{ViscosityDimensions} : std::nullopt);
    ASSERT_TRUE(read) << read.error().message;
    EXPECT_FALSE(read->IsRadial());
    const auto value = read->Value();
    for (const auto& probe :
         ReadProbes(DataPath("viscosity_probes_full3d.txt"))) {
      const auto r = probe.r / length;
      EXPECT_NEAR(value(r, probe.theta, probe.phi), probe.value,
                  1e-11 * probe.value);
      // within the probe's own layer, the same
      EXPECT_EQ(read->ValueIn(model.LayerOf(probe.r), r, probe.theta,
                              probe.phi),
                value(r, probe.theta, probe.phi));
    }
  }
}

// A radial file is radial, holds its field on the viscous layers alone,
// and is read within a layer at the layer's own boundary; elsewhere the
// field is zero.
TEST(ModelField, ARadialFieldIsReadWithinItsLayer) {
  const auto model = Prem8();
  const auto read =
      ModelField::Read(DataPath("viscosity_radial.nc"), model, "viscosity");
  ASSERT_TRUE(read) << read.error().message;
  EXPECT_TRUE(read->IsRadial());
  const auto viscous = ViscousLayers(model);
  ASSERT_FALSE(viscous.empty());
  const auto b = model.DimensionlessView().Boundaries();
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    const auto held = std::ranges::find(viscous, i) != viscous.end();
    EXPECT_EQ(read->Holds(i), held) << "layer " << i;
    for (const auto r : {b[i], 0.5 * (b[i] + b[i + 1]), b[i + 1]}) {
      if (held) {
        EXPECT_NEAR(read->ValueIn(i, r), 1e21, 1e9) << "layer " << i;
      } else {
        EXPECT_EQ(read->ValueIn(i, r), 0.0) << "layer " << i;
      }
    }
  }
  // by the radius alone, the layer above a boundary owns it
  const auto value = read->Value();
  EXPECT_NEAR(value(0.5 * (b[viscous.front()] + b[viscous.front() + 1]), 0.3,
                    1.0),
              1e21, 1e9);
  EXPECT_EQ(value(0.5 * b[1], 0.3, 1.0), 0.0);
}

// A field the file does not hold is nothing where it may be left out and
// an error where it is asked for; a file of another body is refused.
TEST(ModelField, AMissingFieldAndAnotherBodyAreSaid) {
  const auto model = Prem8();
  const auto absent =
      ModelField::ReadIfHeld(DataPath("viscosity_radial.nc"), model, "mu");
  ASSERT_TRUE(absent) << absent.error().message;
  EXPECT_FALSE(absent->has_value());
  const auto asked =
      ModelField::Read(DataPath("viscosity_radial.nc"), model, "mu");
  ASSERT_FALSE(asked);
  EXPECT_NE(asked.error().message.find("holds no mu field"), std::string::npos)
      << asked.error().message;
  const auto missing =
      ModelField::Read(DataPath("no_such_file.nc"), model, "viscosity");
  EXPECT_FALSE(missing);
  // a body of another radius
  const auto deck = Deck::Read(
      DataPath("e1_homogeneous.deck"),
      DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0));
  ASSERT_TRUE(deck) << deck.error().message;
  const auto other = EarthModel::FromDeck(*deck);
  ASSERT_TRUE(other) << other.error().message;
  ASSERT_GT(std::abs(other->SurfaceRadius() - model.SurfaceRadius()), 1.0);
  const auto refused =
      ModelField::Read(DataPath("viscosity_radial.nc"), *other, "viscosity");
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().message.find("is not the model's"),
            std::string::npos)
      << refused.error().message;
}
