#pragma once

/**
 * @file ToroidalOperator.hpp
 * @brief The degree-l toroidal system: the W block of the loading problem,
 * needed once lateral viscosity couples spheroidal and toroidal motion
 * (docs/dev/lateral-coupling.md §4). Non-dimensional, like every internal
 * component.
 *
 * For one degree l >= 1, with k^2 = l(l + 1), the unknown is the toroidal
 * radial scalar W on solid nodes; the bilinear form, in its transversely
 * isotropic shape (Myhill & Al-Attar; the DSpecM1D assembly expands to
 * exactly this, and L = N = mu recovers the Fortran's `amat_tor`):
 *
 *   A(W, W') = int [ k^2 L (r d_r W - W)(r d_r W' - W')
 *                   + k^2 (k^2 - 2) N W W' ] dr.
 *
 * No gravity, no fluid terms, no exterior condition: fluid regions carry
 * no toroidal dof, so disconnected solid regions (the mantle, the inner
 * core) appear as independent blocks of one banded factorisation, SPD for
 * l >= 2. Degree 0 is refused: it has no toroidal motion.
 *
 * At l = 1 the form is singular: rigid rotation (W proportional to r)
 * carries no strain, one null vector per solid region. Lateral viscosity
 * drives degree-1 toroidal motion, so the degree is included
 * (docs/design.md §8), with the gauge fixed in the most convenient way: one dof
 * per region is pinned to zero in the factorisation (the region's top node by
 * default; not the centre, where the rigid mode vanishes). Which dof is
 * pinned does not matter: the memory force is torque-free, so the pinned
 * solve solves the full system, and the pin changes the solution only by
 * rigid rotations, which nothing well defined sees (a horizontal
 * displacement at a point does; docs/design.md §8). Apply is the full, unpinned
 * operator.
 *
 * Truncation follows the spheroidal operator: the element subset of the
 * one global mesh from the mesh's truncation rule, and so do radial cuts:
 * the same system solved by substructuring over segments of the mesh
 * (DegreeOperator.hpp).
 */

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "sl3dpp/Collectives.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

namespace detail {
class BandSubstructure;
}

class ToroidalOperator {
 public:
  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
    /// At l = 1, the gauge: where each solid region's pinned dof sits, as
    /// a fraction of the way down from the region's top node (0 = the
    /// top). Below 1, so never a region's bottom node (the centre, for the
    /// inner core, where the rigid mode vanishes).
    double pin_fraction = 0.0;
    /// A shear modulus at the memory points to add (MemoryPoints.hpp):
    /// with the long-term material, the elastic or a step's effective
    /// operator with internal variables at those points. Read during
    /// construction only.
    const PointShear* point_shear = nullptr;
    /// Radial cuts, as DegreeOperator::Options::cuts: the system solved by
    /// substructuring over the segments between them (a cut node in a
    /// fluid region carries no toroidal dof and separates nothing).
    std::vector<int> cuts{};
    /// Distributed, as DegreeOperator::Options::segments and combine.
    std::vector<int> segments{};
    std::function<void(std::span<double>)> combine{};
  };

  /// Assemble and factorise (Cholesky; at l = 1 with one dof per solid
  /// region pinned). The material is borrowed and must outlive the
  /// operator. Throws std::invalid_argument for l < 1, or for a pin
  /// fraction outside [0, 1).
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

  /// x with A x = b for nrhs column-major right-hand sides; at l = 1 in
  /// the pinned gauge (x zero at the pinned dofs), which needs b
  /// torque-free (orthogonal to every RigidRotation) to solve A x = b.
  /// Distributed, as DegreeOperator::Solve.
  [[nodiscard]] std::vector<double> Solve(
      std::span<const double> b, int nrhs = 1,
      RightHandSide rhs = RightHandSide::Whole) const;

  /// Solve split at its one collective, as DegreeOperator::BeginSolve.
  [[nodiscard]] PendingSolve BeginSolve(
      std::span<const double> b, int nrhs,
      RightHandSide rhs = RightHandSide::Whole) const;
  [[nodiscard]] std::vector<double> FinishSolve(PendingSolve&& pending) const;

  /// As DegreeOperator::Holds and MaskToShare.
  [[nodiscard]] bool Holds(int dof) const;
  void MaskToShare(std::span<double> x, int nrhs = 1) const;

  /// A x with the full (unpinned) operator, nrhs column-major columns.
  [[nodiscard]] std::vector<double> Apply(std::span<const double> x,
                                          int nrhs = 1) const;

  /// The solid regions (blocks separated by fluid) within the subset.
  [[nodiscard]] int RegionCount() const {
    return static_cast<int>(regions_.size());
  }

  /// The rigid rotation of one solid region: W = r on its dofs, zero
  /// elsewhere. At l = 1 these span the null space.
  [[nodiscard]] std::vector<double> RigidRotation(int region) const;

  /// The pinned dofs (one per solid region at l = 1; none above).
  [[nodiscard]] std::span<const int> PinnedDofs() const { return pinned_; }

  /// The radial segments the solve runs over: 1 for the direct solve.
  [[nodiscard]] int SegmentCount() const;

  /// Nodal W over the mesh from a solution vector, zero where there is
  /// no dof (fluid regions, the truncated depths).
  [[nodiscard]] std::vector<double> Expand(std::span<const double> x) const;

 private:
  void BuildDofMap();
  void Assemble(const PointShear* pointShear);
  /// The dofs of the cut nodes above the first element (an empty range at
  /// a fluid node), and the count of cuts at or below it.
  [[nodiscard]] std::pair<std::vector<std::pair<int, int>>, int> Separators(
      std::span<const int> cuts) const;
  void Factorise(const Options& options);

  [[nodiscard]] double& Band(int i, int j) {
    return band_[static_cast<std::size_t>(j) * (kd_ + 1) + (kd_ + i - j)];
  }

  const Material* material_;
  int l_;
  double pinFraction_ = 0.0;
  int firstElement_ = 0;
  int kd_ = 0;
  int ndof_ = 0;
  std::vector<int> dof_;                      // per global node
  std::vector<std::pair<int, int>> regions_;  // dof ranges [first, end)
  std::vector<int> pinned_;
  std::vector<double> band_;  // upper band, column-major
  std::vector<double> scale_;
  std::vector<double> factor_;
  std::shared_ptr<const detail::BandSubstructure> substructure_;
};

}  // namespace sl3dpp
