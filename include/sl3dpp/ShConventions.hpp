#pragma once

/**
 * @file ShConventions.hpp
 * @brief Conversions between GSHTrans's native spherical-harmonic basis and
 * the real orthonormal basis of pyshtools (planetmodel/pyslfp), used only
 * at the IO and benchmark boundary (plan §2.5, §5.5).
 *
 * GSHTrans: complex, orthonormal, Condon-Shortley phase,
 * Y_lm = (-1)^m K_lm P_lm e^{i m phi}, with the real-field symmetry
 * f_{l,-m} = (-1)^m conj(f_lm) and m >= 0 storage.
 *
 * pyshtools "ortho", csphase = 1: real orthonormal, no CS phase,
 * f = sum_l [ a_l0 Y_l0 + sum_{m>0} a_lm sqrt(2) K P cos(m phi)
 *                               + b_lm sqrt(2) K P sin(m phi) ].
 *
 * Matching the two expansions term by term gives, for m > 0,
 *
 *   f_lm = (-1)^m (a_lm - i b_lm) / sqrt(2),
 *   a_lm = (-1)^m sqrt(2) Re f_lm,   b_lm = -(-1)^m sqrt(2) Im f_lm,
 *
 * and f_l0 = a_l0 (real). These identities are verified against pyshtools
 * numerically by the stage-2 fixtures.
 */

#include <complex>
#include <numbers>

namespace sl3dpp {

/// The cosine/sine pair of a real orthonormal (pyshtools) expansion.
struct RealPair {
  double cos = 0.0;
  double sin = 0.0;
};

/// The GSHTrans coefficient from the real pair at order m >= 0.
[[nodiscard]] inline std::complex<double> ComplexFromRealPair(RealPair pair,
                                                              int m) {
  if (m == 0) return pair.cos;
  const auto sign = m % 2 == 0 ? 1.0 : -1.0;
  return sign * std::complex<double>(pair.cos, -pair.sin) / std::numbers::sqrt2;
}

/// The real pair at order m >= 0 from the GSHTrans coefficient.
[[nodiscard]] inline RealPair RealPairFromComplex(std::complex<double> f,
                                                  int m) {
  if (m == 0) return {f.real(), 0.0};
  const auto sign = m % 2 == 0 ? 1.0 : -1.0;
  return {sign * std::numbers::sqrt2 * f.real(),
          -sign * std::numbers::sqrt2 * f.imag()};
}

}  // namespace sl3dpp
