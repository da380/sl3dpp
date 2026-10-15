#pragma once

/**
 * @file Scales.hpp
 * @brief Non-dimensionalisation: what one unit of each base dimension is,
 * in SI. Mirrors planetmodel's `Scales` exactly, constants included, so
 * that fixture comparisons are free of unit skew.
 */

#include <cmath>

namespace sl3dpp {

/// Newton's constant in SI, planetmodel's value (`units.G_SI`).
inline constexpr double GravitationalConstantSI = 6.6743e-11;

/// The Earth's mean density in SI, planetmodel's value
/// (`units.EARTH_MEAN_DENSITY`).
inline constexpr double EarthMeanDensitySI = 5515.0;

/// The Earth's radius in SI, the old Fortran's `r_norm`.
inline constexpr double EarthRadiusSI = 6.371e6;

/**
 * @brief Integer exponents of the base dimensions of a quantity.
 *
 * `Dimensions{.mass = 1, .length = -3}` is a density. Only mass, length
 * and time: everything here is mechanics.
 */
struct Dimensions {
  int mass = 0;
  int length = 0;
  int time = 0;
};

// Dimensions of the common quantities.
inline constexpr Dimensions DensityDimensions{.mass = 1, .length = -3};
inline constexpr Dimensions VelocityDimensions{.length = 1, .time = -1};
inline constexpr Dimensions AccelerationDimensions{.length = 1, .time = -2};
inline constexpr Dimensions ModulusDimensions{
    .mass = 1, .length = -1, .time = -2};
inline constexpr Dimensions ViscosityDimensions{
    .mass = 1, .length = -1, .time = -1};
inline constexpr Dimensions PotentialDimensions{.length = 2, .time = -2};
inline constexpr Dimensions GravitationalConstantDimensions{
    .mass = -1, .length = 3, .time = -2};

/**
 * @brief What one unit of mass, length and time is, in SI.
 *
 * A quantity of dimensions d stored under these scales relates to its SI
 * value by stored = SI / Factor(d). The default is SI itself (all scales
 * one); Geophysical() gives the scales under which G = 1, planetmodel's
 * `Scales.geophysical`.
 */
class Scales {
 public:
  /// SI: every scale is one.
  constexpr Scales() = default;

  /// Explicit scales; each must be positive.
  constexpr Scales(double mass, double length, double time)
      : mass_(mass), length_(length), time_(time) {}

  /**
   * @brief Scales for self-gravitating problems: G becomes one.
   *
   * @param length The body's radius in metres.
   * @param density A reference density in kg m^-3, conventionally the
   * Earth's mean. The mass scale is density length^3 and the time scale
   * is fixed by G density T^2 = 1.
   */
  static Scales Geophysical(double length = EarthRadiusSI,
                            double density = EarthMeanDensitySI) {
    return Scales(density * length * length * length, length,
                  1.0 / std::sqrt(GravitationalConstantSI * density));
  }

  [[nodiscard]] constexpr double Mass() const { return mass_; }
  [[nodiscard]] constexpr double Length() const { return length_; }
  [[nodiscard]] constexpr double Time() const { return time_; }

  /// The SI value of one stored unit of a quantity of dimensions d.
  [[nodiscard]] constexpr double Factor(Dimensions d) const {
    return Power(mass_, d.mass) * Power(length_, d.length) *
           Power(time_, d.time);
  }

  /// Newton's constant under these scales (one under Geophysical()).
  [[nodiscard]] constexpr double GravitationalConstant() const {
    return GravitationalConstantSI / Factor(GravitationalConstantDimensions);
  }

 private:
  double mass_ = 1.0;
  double length_ = 1.0;
  double time_ = 1.0;

  static constexpr double Power(double x, int n) {
    if (n < 0) return 1.0 / Power(x, -n);
    double result = 1.0;
    for (; n > 0; --n) result *= x;
    return result;
  }
};

}  // namespace sl3dpp
