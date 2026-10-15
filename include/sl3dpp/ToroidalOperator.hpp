#pragma once

/**
 * @file ToroidalOperator.hpp
 * @brief The degree-l toroidal system: the W block of the loading problem,
 * needed once lateral viscosity couples spheroidal and toroidal motion
 * (docs/stage4.md §4). Non-dimensional, like every internal component.
 *
 * For one degree l >= 2, with k^2 = l(l + 1), the unknown is the toroidal
 * radial scalar W on solid nodes; the bilinear form, in its transversely
 * isotropic shape (Myhill & Al-Attar; the DSpecM1D assembly expands to
 * exactly this, and L = N = mu recovers the Fortran's `amat_tor`):
 *
 *   A(W, W') = int [ k^2 L (r d_r W - W)(r d_r W' - W')
 *                   + k^2 (k^2 - 2) N W W' ] dr.
 *
 * No gravity, no fluid terms, no exterior condition: fluid regions carry
 * no toroidal dof, so disconnected solid sections appear as independent
 * SPD blocks of one banded factorisation. Degrees 0 and 1 are refused —
 * l = 0 has no toroidal motion, and at l = 1 the form is singular (rigid
 * rotation W proportional to r), to be handled with the rotation
 * bookkeeping when the memory coupling needs it (stage4.md §8).
 *
 * Truncation follows the spheroidal operator: the element subset of the
 * one global mesh from the mesh's truncation rule.
 */

#include <span>
#include <vector>

#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

class ToroidalOperator {
 public:
  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
  };

  /// Assemble and factorise (Cholesky; the blocks are SPD for l >= 2).
  /// The material is borrowed and must outlive the operator. Throws
  /// std::invalid_argument for l < 2.
  ToroidalOperator(const Material& material, int l, Options options);
  ToroidalOperator(const Material& material, int l)
      : ToroidalOperator(material, l, Options{}) {}

  [[nodiscard]] int Degree() const { return l_; }
  [[nodiscard]] int FirstElement() const { return firstElement_; }
  [[nodiscard]] int DofCount() const { return ndof_; }

  /// The W dof of a global node, -1 on fluid-only nodes and below the
  /// element subset.
  [[nodiscard]] int Dof(int node) const { return dof_[node]; }

  /// The dof at the surface node (the surface is solid by Material's
  /// construction).
  [[nodiscard]] int SurfaceDof() const;

  /// x with A x = b for nrhs column-major right-hand sides.
  [[nodiscard]] std::vector<double> Solve(std::span<const double> b,
                                          int nrhs = 1) const;

  /// Nodal W over the mesh from a solution vector, zero where there is
  /// no dof (fluid regions, the truncated depths).
  [[nodiscard]] std::vector<double> Expand(std::span<const double> x) const;

 private:
  void BuildDofMap();
  void Assemble();
  void Factorise();

  [[nodiscard]] double& Band(int i, int j) {
    return band_[static_cast<std::size_t>(j) * (kd_ + 1) + (kd_ + i - j)];
  }

  const Material* material_;
  int l_;
  int firstElement_ = 0;
  int kd_ = 0;
  int ndof_ = 0;
  std::vector<int> dof_;      // per global node
  std::vector<double> band_;  // upper band, column-major
  std::vector<double> scale_;
  std::vector<double> factor_;
};

}  // namespace sl3dpp
