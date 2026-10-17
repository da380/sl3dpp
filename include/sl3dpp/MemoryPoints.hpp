#pragma once

/**
 * @file MemoryPoints.hpp
 * @brief Where the internal variables live within each radial element, and
 * the relaxation term an operator subtracts so that the implicit step's
 * elimination of the new internal variables stays exact (plan.md §5.5).
 *
 * The internal variables live at the element's GLL nodes by default (the
 * displacement's own nodes; pyslfp's and the Fortran's choice), or at its
 * N = ngll - 1 interior Gauss-Legendre points: the same 2N - 1 exactness
 * for the memory-force integral, no doubling at shared element boundaries,
 * ~20% fewer points.
 *
 * The step eliminates m+ = m~ + beta d(u) into equilibrium, leaving the
 * effective operator K_U - sum_k D^T W (mu_k beta_k) D, with D the strain
 * at the memory points and W their quadrature. At the GLL nodes that is
 * the operator assembled with relaxed moduli (the stiffness assembly's own
 * quadrature points); at Gauss points it is not, so the operators accept
 * the relaxation term explicitly (Relaxation), and with GLL points that
 * reproduces the relaxed-moduli assembly to round-off.
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

/// A relaxation term an operator subtracts from its assembly: per element
/// and memory point, the coefficient c = sum_k mu_k beta_k (the relaxed
/// part of the shear modulus), and the term is sum_p jac w_p c_p times the
/// deviatoric form at the point. Read during the operator's construction
/// only.
struct Relaxation {
  const MemoryPoints* points = nullptr;
  std::span<const double> coefficient;  ///< (ElementCount, points->count)
};

}  // namespace sl3dpp
