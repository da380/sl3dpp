#pragma once

/**
 * @file DegreeOperator.hpp
 * @brief The degree-l system of the loading problem: dofs, band assembly,
 * factorisation, right-hand sides. A port of pyslfp's
 * `love_numbers.assembly.DegreeSystem`, non-dimensional throughout.
 *
 * For one spherical-harmonic degree l, with k^2 = l(l + 1), the unknowns
 * are the radial and tangential displacement scalars U, V and the
 * potential perturbation phi on solid nodes, and phi alone on fluid
 * nodes, where the deformation is characterised by the potential with the
 * stratification term g^-1 (d rho / dr) phi phi' r^2 and explicit
 * fluid-solid interface couplings. The bilinear form is eq. (D76) of
 * Al-Attar & Tromp (2014) with its elastic block written for a
 * transversely isotropic medium: with X = r d_r U, Y = 2U - k^2 V and
 * S = U + r d_r V - V,
 *
 *   C X X' + (A - N) Y Y' + F (X Y' + Y X') + k^2 L S S'
 *     + k^2 (k^2 - 2) N V V',
 *
 * the isotropic form being A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3,
 * L = N = mu. The exterior is closed by the Dirichlet-to-Neumann term
 * (l + 1) a phi phi' / 4 pi G at the surface and, where the domain starts
 * above the centre, the interior continuation l r phi phi' / 4 pi G at the
 * bottom. For l = 1 the surface potential dof is removed (the
 * centre-of-mass frame); for l = 0 there is no tangential dof, fluid
 * regions are mu = 0 elastic media, and U(0) = 0 is imposed at a centre.
 * Degrees above zero act on the element subset of the one global mesh
 * where the solution is not negligible (docs/design.md §3).
 *
 * The system is symmetric, stored as an upper band, scaled symmetrically
 * by its diagonal and factorised on construction: banded Cholesky, falling
 * back to banded LU (the fluid limit's semi-definite elastic block, say).
 * Solves take one or several right-hand sides. With radial cuts the same
 * system is solved by substructuring (detail/BandSubstructure.hpp): each
 * segment of the mesh between cuts eliminates its interior on its own, and
 * the cut nodes' dofs solve together, the algebra of a solve distributed
 * over radial segments (docs/dev/radial-substructuring.md).
 *
 * With `gravitating = false` the same discretisation solves the
 * non-gravitating problem of the cross-code benchmark hierarchy: the
 * gravity and potential terms are dropped, the unknowns are U and V alone
 * (U alone at l = 0), and the natural forcings are the unit tractions. It
 * takes a solid model -- a fluid region has no restoring force without
 * gravity -- and excludes degree 1, where a surface traction has a net
 * force against the rigid translations. This mirrors pyslfp's
 * `DegreeSystem(..., gravitating=False)` exactly.
 */

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

namespace detail {
class BandSubstructure;
}

/// The three solution components at a node.
enum class Component { U = 0, V = 1, Phi = 2 };

/// Which piece of the surface-load force a right-hand side carries.
enum class LoadPart { Force, Potential, Both };

class DegreeOperator {
 public:
  struct Options {
    /// The relative level below which a degree's solution is truncated
    /// with depth.
    double eps = 1e-8;
    /// The truncation-radius rule (docs/design.md §3).
    TruncationRule truncation = TruncationRule::Graded;
    /// A shear modulus at the memory points to add (MemoryPoints.hpp):
    /// with the long-term material, the elastic or a step's effective
    /// operator with internal variables at those points. Read during
    /// construction only.
    const PointShear* point_shear = nullptr;
    /// false drops the gravity and potential terms and the phi dofs: the
    /// non-gravitating problem, on a solid model, l != 1 (see above).
    bool gravitating = true;
    /// Radial cuts: elements (increasing, each inside the mesh) at whose
    /// bottom node the mesh is cut into segments, the system then solved
    /// by substructuring with the cut nodes' dofs as the separators. The
    /// solution is the direct solve's to rounding. Cuts at or below the
    /// degree's first element are ignored; empty, the direct solve.
    std::vector<int> cuts{};
    /// Distributed (with `combine`): the segments of the cut mesh this
    /// process solves, numbered from the bottom of the whole mesh (segment
    /// p lies between cuts p - 1 and p); every segment must belong to
    /// exactly one process. `combine` sums a vector over the processes in
    /// place (an MPI all-reduce: the library calls no MPI itself), which
    /// makes construction and Solve collective. Empty, one process solves
    /// every segment.
    std::vector<int> segments{};
    std::function<void(std::span<double>)> combine{};
  };

  /// Assemble and factorise the degree-l system. The material is
  /// borrowed and must outlive the operator. Throws std::invalid_argument
  /// for a negative degree and std::runtime_error when the factorisation
  /// fails outright (Cholesky and LU both).
  DegreeOperator(const Material& material, int l, Options options);
  DegreeOperator(const Material& material, int l)
      : DegreeOperator(material, l, Options{}) {}

  [[nodiscard]] int Degree() const { return l_; }
  [[nodiscard]] int FirstElement() const { return firstElement_; }
  [[nodiscard]] int DofCount() const { return ndof_; }

  /// The global dof of a component at a global node, -1 where absent.
  [[nodiscard]] int Dof(int node, Component component) const {
    return dof_[static_cast<std::size_t>(node) * 3 +
                static_cast<int>(component)];
  }

  /// The dof of U, V or phi at the surface node, -1 when absent.
  [[nodiscard]] int SurfaceDof(Component component) const;

  // -- right-hand sides -------------------------------------------------------

  /// The surface-load force of a unit surface density: the traction piece
  /// -g a^2 on U'(a), the attraction piece -a^2 on phi'(a), or both. At
  /// l = 1 the potential piece vanishes with the removed dof.
  [[nodiscard]] std::vector<double> Load(LoadPart part = LoadPart::Both) const;

  /// The force of a unit tangential traction at the surface, -g grad_1
  /// Y_lm per unit amplitude in the units of a surface density: the V dof
  /// takes -g a^2 l(l + 1). Zero at l = 0, where there is no tangential
  /// dof.
  [[nodiscard]] std::vector<double> Tangential() const;

  /// The force of a unit normal traction pressing on the surface, -Y_lm
  /// r_hat per unit amplitude: -a^2 on U'(a). It is the traction piece of
  /// `Load` per unit of traction rather than of surface density --
  /// `Load(LoadPart::Force)` is g times this -- and the natural forcing of
  /// the non-gravitating problem, where no g sets the scale.
  [[nodiscard]] std::vector<double> Traction() const;

  /// The force of a unit tangential traction, -grad_1 Y_lm per unit
  /// amplitude: -a^2 l(l + 1) on V'(a); `Tangential` is g times this. Zero
  /// at l = 0, where there is no tangential dof.
  [[nodiscard]] std::vector<double> TractionTangential() const;

  /// The force of a unit external potential psi = (r/a)^power, the power
  /// defaulting to l (the harmonic tide): the known psi moved to the
  /// right-hand side term by term, the fluid density perturbation
  /// responding to the total potential. Power 2 at l = 0 is the spherical
  /// mean of a centrifugal potential change. A constant potential is a
  /// gauge and gives zero.
  [[nodiscard]] std::vector<double> Tide(int power = -1) const;

  // -- solving
  // ------------------------------------------------------------------

  /// x with A x = b for `nrhs` right-hand sides stored column-major in
  /// `b` (DofCount() rows each); the result has the same layout.
  /// Distributed (Options::segments), only this process's segments and the
  /// cut nodes are solved; the rest of x is unspecified.
  [[nodiscard]] std::vector<double> Solve(std::span<const double> b,
                                          int nrhs = 1) const;

  /// A x for `nrhs` column-major columns: the assembled (unscaled)
  /// operator, for residuals.
  [[nodiscard]] std::vector<double> Apply(std::span<const double> x,
                                          int nrhs = 1) const;

  /// Whether the (scaled) Cholesky factorisation held; false means the LU
  /// fallback is in use (in some segment, when substructured).
  [[nodiscard]] bool UsedCholesky() const { return !choleskyFailed_; }

  /// The radial segments the solve runs over: 1 for the direct solve.
  [[nodiscard]] int SegmentCount() const;

  // -- views
  // --------------------------------------------------------------------

  struct NodalSolution {
    std::vector<double> U;    ///< (element, node) row-major over the mesh
    std::vector<double> V;    ///< NaN through fluid elements above l = 0
    std::vector<double> phi;  ///< always live
  };

  /// Nodal arrays from a solution vector: absent components and elements
  /// below the solved subset are zero; U and V are NaN through fluid
  /// elements above degree zero, where the quasi-static problem does not
  /// determine the displacement.
  [[nodiscard]] NodalSolution Expand(std::span<const double> x) const;

 private:
  void BuildDofMap();
  void Assemble(const PointShear* pointShear);
  /// The dof ranges of the cut nodes above the first element, and the
  /// count of cuts at or below it.
  [[nodiscard]] std::pair<std::vector<std::pair<int, int>>, int> Separators(
      std::span<const int> cuts) const;
  void Factorise(const Options& options);

  [[nodiscard]] double& Band(int i, int j) {
    return band_[static_cast<std::size_t>(j) * (kd_ + 1) + (kd_ + i - j)];
  }
  void Point(int i, int j, double value);

  struct Interface {
    int node;     ///< global node at the fluid-solid boundary
    double r;     ///< its radius
    double g;     ///< gravity there
    double rhoF;  ///< the fluid-side density
    double sign;  ///< + with the fluid below, - with it above
  };
  [[nodiscard]] std::vector<Interface> Interfaces() const;

  const Material* material_;
  int l_;
  bool gravitating_ = true;
  int firstElement_ = 0;
  int kd_ = 0;
  int ndof_ = 0;
  std::vector<int> dof_;      // (node, component) row-major, -1 absent
  std::vector<double> band_;  // upper band, column-major (kd + 1) x ndof

  // The factorisation: the symmetric diagonal scaling s, then the Cholesky
  // band or the LU band with its pivots, or with cuts the substructured
  // factors (immutable, so copies share them).
  std::vector<double> scale_;
  std::vector<double> factor_;
  std::vector<int> pivots_;
  bool choleskyFailed_ = false;
  std::shared_ptr<const detail::BandSubstructure> substructure_;
};

}  // namespace sl3dpp
