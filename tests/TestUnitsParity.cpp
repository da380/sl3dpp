/**
 * @file TestUnitsParity.cpp
 * @brief Scales.hpp mirrors planetmodel::units exactly: the library keeps
 * its own copy of the constants and factors rather than aliasing
 * planetmodel's, and this parity gate holds the copy to them, constant by
 * constant and factor by factor.
 */

#include <gtest/gtest.h>

#include <planetmodel/units.hpp>

#include "sl3dpp/Scales.hpp"

namespace {

using namespace sl3dpp;

}  // namespace

TEST(UnitsParity, TheConstantsAreOne) {
  EXPECT_EQ(GravitationalConstantSI, planetmodel::units::G_SI);
  EXPECT_EQ(EarthMeanDensitySI, planetmodel::units::EARTH_MEAN_DENSITY);
}

TEST(UnitsParity, GeophysicalScalesAgreeFactorByFactor) {
  const auto ours = Scales::Geophysical(EarthRadiusSI);
  const auto theirs = planetmodel::Scales::Geophysical(EarthRadiusSI);
  EXPECT_EQ(ours.Mass(), theirs.Mass());
  EXPECT_EQ(ours.Length(), theirs.Length());
  EXPECT_EQ(ours.Time(), theirs.Time());

  const auto pairs = std::vector<
      std::pair<Dimensions, planetmodel::units::Dimensions>>{
      {DensityDimensions, planetmodel::units::DENSITY},
      {VelocityDimensions, planetmodel::units::VELOCITY},
      {AccelerationDimensions, planetmodel::units::GRAVITY},
      {ModulusDimensions, planetmodel::units::MODULUS},
      {ViscosityDimensions, planetmodel::units::VISCOSITY},
      {PotentialDimensions,
       planetmodel::units::Dimensions{.length = 2, .time = -2}},
      {GravitationalConstantDimensions,
       planetmodel::units::GRAVITATIONAL_CONSTANT},
  };
  for (const auto& [mine, other] : pairs) {
    EXPECT_DOUBLE_EQ(ours.Factor(mine), theirs.Factor(other));
  }
  EXPECT_DOUBLE_EQ(ours.GravitationalConstant(), 1.0);
  EXPECT_DOUBLE_EQ(theirs.Factor(planetmodel::units::GRAVITATIONAL_CONSTANT),
                   ours.Factor(GravitationalConstantDimensions));
}
