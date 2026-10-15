#pragma once

/**
 * @file LoveNumbers.hpp
 * @brief Generalised Love numbers by degree, pyslfp's conventions
 * (`love_numbers.table` there states them in full; Al-Attar et al. 2024,
 * eq. 62).
 *
 * Per degree, the surface response (U, V, phi) to four forcings: the
 * traction piece of a unit surface load (channel u), its attraction piece
 * (channel phi), a unit tangential traction (channel v), and the unit
 * external potential psi = (r/a)^l (channel t). The full load numbers are
 * the channel sums h = h_u + h_phi and so on. Degree 1 is in the
 * centre-of-mass frame, where the potential channel vanishes; at degree 0
 * the tidal numbers vanish while the load numbers do not
 * (k_0 = -4 pi G a by mass conservation). Degree 0 also carries the five
 * axial numbers of the rotational feedback: h_c and k_c, the surface
 * response to the centrifugal potential (r/a)^2, and the inertia moments
 * m_u, m_phi, m_c of the responses to the two load channels and to the
 * centrifugal potential.
 *
 * SI in, SI out: phi is the physical potential perturbation, negative near
 * added mass; h and l are metres per unit surface density (kg m^-2), k is
 * potential per unit surface density, the tidal numbers are per unit
 * potential. The bilinear form is symmetric, so g h_phi = k_u,
 * h_v = l(l+1) l_u, k_v = g l(l+1) l_phi and
 * m_u = sqrt(4 pi) g a^4 h_c / 2, m_phi = sqrt(4 pi) a^4 k_c / 2;
 * `ReciprocityResidual` reports the worst relative violation.
 */

#include <vector>

#include "sl3dpp/Material.hpp"

namespace sl3dpp {

struct LoveNumbers {
  std::vector<int> degrees;

  // One value per degree, SI. Channels: u = load traction piece,
  // phi = load attraction piece, v = tangential traction, t = tide.
  std::vector<double> h_u, l_u, k_u;
  std::vector<double> h_phi, l_phi, k_phi;
  std::vector<double> h_v, l_v, k_v;
  std::vector<double> h_t, l_t, k_t;

  // The axial numbers of degree 0, SI.
  double h_c = 0, k_c = 0, m_u = 0, m_phi = 0, m_c = 0;

  double radius = 0;           ///< the surface radius, m
  double surface_gravity = 0;  ///< g there, m s^-2

  /// The worst relative violation of the reciprocity identities across
  /// the degrees (and the axial pair), a structural check of the
  /// assembly's symmetry.
  [[nodiscard]] double ReciprocityResidual() const;
};

struct LoveNumberOptions {
  double eps = 1e-8;
  TruncationRule truncation = TruncationRule::Graded;
};

/// The Love numbers of a material for every degree from 0 to lmax: each
/// degree assembled and factorised once, solved for the four forcings
/// (degree 0 for the centrifugal potential besides), degrees in parallel
/// under OpenMP.
[[nodiscard]] LoveNumbers ComputeLoveNumbers(const Material& material, int lmax,
                                             LoveNumberOptions options = {});

/// The degree-l tidal Love number of the material's hydrostatic fluid
/// limit (every region a mu = 0 solid, solved by the LU path): the k_hyd
/// of the rotational-stability literature. A homogeneous body gives
/// Kelvin's 3 / (2 (l - 1)). SI (dimensionless). Degree >= 2.
[[nodiscard]] double FluidTidalNumber(const Material& material, int l = 2,
                                      LoveNumberOptions options = {});

}  // namespace sl3dpp
