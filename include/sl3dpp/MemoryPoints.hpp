#pragma once

/**
 * @file MemoryPoints.hpp
 * @brief Where the internal variables live within each radial element, and
 * the shear term an operator integrates there so that the discrete energy
 * stays a sum of squares and the implicit step's elimination of the new
 * internal variables exact (plan.md §5.5).
 *
 * The internal variables live at the element's N = ngll - 1 interior
 * Gauss-Legendre points by default: the same 2N - 1 exactness for the
 * memory-force integral, no doubling at shared element boundaries, ~20%
 * fewer points. The displacement stays at the GLL nodes either way. The
 * alternative is the GLL nodes themselves (pyslfp's and the Fortran's
 * choice), kept for like-for-like comparisons with those codes.
 *
 * The shear energy splits as int mu_inf |d|^2 + sum_k int mu_k |d - m_k|^2,
 * the long-term part at the GLL nodes (the stiffness assembly's own
 * quadrature) and each branch's at the memory points, where m_k lives.
 * The step eliminates m+ = m~ + beta d(u) into equilibrium, leaving the
 * effective operator K_inf + sum_k D^T W (mu_k w_k) D, with D the strain at
 * the memory points, W their quadrature and w_k = 1 - beta_k the branch's
 * unrelaxed fraction: positive semi-definite term by term for any step.
 * At the GLL nodes that is the operator assembled with relaxed moduli;
 * at Gauss points it is not, so the operators accept the point term
 * explicitly (PointShear), and with GLL points that reproduces the
 * relaxed-moduli assembly to round-off.
 *
 * The split matters. Integrating the unrelaxed shear at the nodes and
 * subtracting sum_k D^T W (mu_k beta_k) D at the points leaves, as a
 * Maxwell branch relaxes fully (beta -> 1, h = dt / tau large), the
 * difference of two quadratures of the same energy: of no definite sign,
 * and the step goes unstable (h of a few hundred, PREM Maxwell mantle).
 */

#include <span>
#include <vector>

#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

/// The point set the internal variables live on within each element.
enum class MemoryPointSet { Gll, Gauss };

/// The reference geometry of a memory point set on the mesh's element: the
/// points on [-1, 1], their reference quadrature weights, and the values
/// and xi-derivatives there of the element's GLL Lagrange basis (what
/// interpolates nodal fields to the points and evaluates strains there).
/// For the GLL set the values are the identity and the derivatives the
/// mesh's own differentiation matrix, exactly.
struct MemoryPoints {
  MemoryPointSet set = MemoryPointSet::Gll;
  int count = 0;
  int ngll = 0;
  std::vector<double> xi;
  std::vector<double> weight;
  std::vector<double> value;       ///< (count, ngll): l_i(xi_p)
  std::vector<double> derivative;  ///< (count, ngll): l_i'(xi_p)

  [[nodiscard]] static MemoryPoints Of(MemoryPointSet set,
                                       const RadialMesh& mesh);

  [[nodiscard]] double Value(int p, int i) const {
    return value[static_cast<std::size_t>(p) * ngll + i];
  }
  [[nodiscard]] double Derivative(int p, int i) const {
    return derivative[static_cast<std::size_t>(p) * ngll + i];
  }

  /// The radius of point p of element e: the node's own at the GLL points,
  /// else the element's (affine) radius interpolated there.
  [[nodiscard]] double RadiusAt(const RadialMesh& mesh, int e, int p) const;

  /// Nodal (element, gll) values interpolated to the points, (element,
  /// point), through each element's Lagrange basis.
  [[nodiscard]] std::vector<double> Interpolate(
      std::span<const double> nodal) const;

  /// The transpose of Interpolate: point values (element, point) to nodal
  /// (element, gll), sum_p l_i(xi_p) v_p. What maps a sensitivity to the
  /// point values onto the nodal values they interpolate.
  [[nodiscard]] std::vector<double> InterpolateTranspose(
      std::span<const double> atPoints) const;
};

/// A shear modulus that lives at the memory points, which an operator
/// integrates with their quadrature and adds to its material's assembly:
/// per element and point the modulus s (the branches' sum_k mu_k w_k), and
/// the term is sum_p jac w_p s_p times the deviatoric form at the point.
/// With GLL points it is exactly the material's shear modulus raised by s.
/// Read during the operator's construction only.
struct PointShear {
  const MemoryPoints* points = nullptr;
  std::span<const double> modulus;  ///< (ElementCount, points->count)
};

}  // namespace sl3dpp
