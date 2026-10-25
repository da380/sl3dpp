#pragma once

/**
 * @file RadialMesh.hpp
 * @brief The single global GLL mesh every degree and every grid field
 * shares (docs/design.md §3), following planetmodel's `Mesh1D`/`RadialMesh`.
 *
 * The mesh is pure geometry: breakpoints pinned at every model boundary
 * inside the range, each span subdivided uniformly into elements no wider
 * than `drmax`, `ngll` Gauss-Lobatto-Legendre nodes per element, a global
 * numbering of the shared nodes, and the reference-element data a weak form
 * assembles with. Nodal arrays are stored per element, shape (element,
 * node), so a shared node carries one value from each side.
 *
 * Units: the mesh is an internal component and lives in the model's
 * non-dimensional numbers — its radii, nodal arrays and gravity are those
 * of `EarthModel::Dimensionless`. Only the construction options cross the
 * user boundary and are SI (metres), converted once on Create. The
 * breakpoints overload takes raw numbers and converts nothing.
 *
 * Degree-dependent truncation is a per-degree element subset of this one
 * mesh — never a separate sub-mesh, since the pseudospectral coupling needs
 * a single radial node set. Two truncation-radius rules are offered
 * (docs/design.md §3): the graded rule r = rmax eps^(1/(l+1)) of
 * pyslfp, and the old Fortran's r = rmax (1 - 20/(l+0.5)).
 */

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/EarthModel.hpp"

namespace sl3dpp {

/// Why a RadialMesh could not be built.
struct MeshError {
  std::string message;
};

/// The truncation-radius heuristic for a degree-l solve.
enum class TruncationRule {
  /// r_l = rmax eps^(1/(l+1)), from the interior decay (r/rmax)^(l+1)
  /// (pyslfp's grading).
  Graded,
  /// r_l = rmax (1 - 20/(l+0.5)), the old Fortran's rule.
  Legacy,
};

class RadialMesh {
 public:
  struct Options {
    int ngll = 5;
    /// Maximum element width, metres (raw for the breakpoints overload).
    /// Exactly one of drmax and lmax.
    std::optional<double> drmax{};
    /// Sizes the mesh for degrees up to lmax: drmax = 0.1 rmax / (lmax + 1),
    /// planetmodel's rule.
    std::optional<int> lmax{};
    /// Mesh range in metres, defaulting to the whole model; must lie
    /// within it. Only for the model overload.
    std::optional<double> rmin{};
    std::optional<double> rmax{};
  };

  /// A mesh over (part of) a model: every model boundary strictly inside
  /// the range is an element boundary, and each element knows its layer.
  [[nodiscard]] static std::expected<RadialMesh, MeshError> Create(
      const EarthModel& model, Options options);
  [[nodiscard]] static std::expected<RadialMesh, MeshError> Create(
      const EarthModel& model) {
    return Create(model, Options{});
  }

  /// A mesh over explicit breakpoints (strictly increasing; every one an
  /// element boundary before subdivision), with element i's "layer" being
  /// the span it came from. For tests and meshes with no model behind them.
  [[nodiscard]] static std::expected<RadialMesh, MeshError> Create(
      std::span<const double> breakpoints, Options options);

  /**
   * @brief A mesh for every degree up to lmax, fine at the surface and
   * widening with depth (pyslfp's `graded_mesh`).
   *
   * The degree-l solution is negligible below eps of its surface value
   * deeper than a(1 - eps^(1/(l+1))), and the uniform lmax rule resolves
   * it with elements no wider than 0.1 a / (l + 1); at depth d the highest
   * degree reaching it has (l + 1) = ln(eps) / ln(1 - d/a), so an element
   * there need be no narrower than 0.1 a ln(1 - d/a) / ln(eps) — about
   * d / 184 for eps = 1e-8 — floored at the uniform rule's width for lmax.
   * Every degree then sees at least the uniform rule's resolution
   * throughout its own element subset, while the whole mesh has a few
   * hundred elements rather than the ten (lmax + 1) of the uniform rule.
   * Every model boundary is an element boundary, the last elements above
   * one shortened or split so none is wider than the rule at its top.
   */
  [[nodiscard]] static std::expected<RadialMesh, MeshError> CreateGraded(
      const EarthModel& model, int lmax, int ngll = 5, double eps = 1e-8);

  // -- geometry ---------------------------------------------------------------

  [[nodiscard]] int ElementCount() const {
    return static_cast<int>(left_.size());
  }
  [[nodiscard]] int NGll() const { return ngll_; }

  /// The number of distinct (shared) nodes,
  /// ElementCount() (ngll - 1) + 1.
  [[nodiscard]] int NodeCount() const {
    return ElementCount() * (ngll_ - 1) + 1;
  }

  [[nodiscard]] double Left(int element) const { return left_[element]; }
  [[nodiscard]] double Right(int element) const { return right_[element]; }

  /// Half-width of the element: dr/dxi of the affine map from [-1, 1].
  [[nodiscard]] double Jacobian(int element) const { return jac_[element]; }

  /// The radius of node i of an element.
  [[nodiscard]] double Radius(int element, int node) const {
    return radius_[static_cast<std::size_t>(element) * ngll_ + node];
  }

  /// All node radii, per element, row-major (element, node): shared nodes
  /// appear once from each side, with exactly equal values.
  [[nodiscard]] std::span<const double> Radii() const { return radius_; }

  /// The global index of node i of an element; interior shared nodes have
  /// one index from each side.
  [[nodiscard]] int GlobalIndex(int element, int node) const {
    return element * (ngll_ - 1) + node;
  }

  /// The element whose interval contains r: a shared boundary resolves to
  /// the element above, and r outside the mesh clips to the first or last.
  [[nodiscard]] int ElementOf(double r) const;

  /// The model layer (or breakpoint span) each element lies in.
  [[nodiscard]] int LayerOf(int element) const { return layer_[element]; }

  // -- reference element ------------------------------------------------------

  /// GLL points on [-1, 1].
  [[nodiscard]] std::span<const double> Points() const { return xi_; }

  /// GLL weights.
  [[nodiscard]] std::span<const double> Weights() const { return weights_; }

  /// The Lagrange derivative matrix D(k, i) = l_i'(xi_k), row-major, the
  /// 'hp(knode, inode)' of standard SEM codes: quadrature sums read
  /// sum_k w_k f(xi_k) D(k, i) D(k, j).
  [[nodiscard]] double Derivative(int k, int i) const {
    return deriv_[static_cast<std::size_t>(k) * ngll_ + i];
  }

  // -- degree truncation ------------------------------------------------------

  /// The radius below which a degree-l solution is negligible, by the rule;
  /// clipped to the mesh.
  [[nodiscard]] double TruncationRadius(
      int l, TruncationRule rule = TruncationRule::Graded,
      double eps = 1e-8) const;

  /// The first element of the DOF subset for a degree-l solve: the element
  /// containing the truncation radius.
  [[nodiscard]] int StartElement(int l,
                                 TruncationRule rule = TruncationRule::Graded,
                                 double eps = 1e-8) const;

  /// Radial cuts (DegreeOperator::Options::cuts) splitting the mesh into
  /// `parts` segments of about equal weight: the element boundaries nearest
  /// the equal shares of the summed element weights (one per element when
  /// empty), only where both sides lie in one layer, so never at a model
  /// boundary (solid-fluid ones included). Increasing; fewer than
  /// parts - 1 when the layers leave too few boundaries.
  [[nodiscard]] std::vector<int> BalancedCuts(
      int parts, std::span<const double> elementWeights = {}) const;

  // -- nodal sampling ---------------------------------------------------------

  /// A model field at the nodes, per element, row-major (element, node):
  /// each element takes the field of its own layer, so a node on a model
  /// boundary carries both one-sided values; `nu` asks for the nu-th
  /// radial derivative. An element whose layer lacks the field is refused
  /// (std::domain_error), as the model's Require is.
  [[nodiscard]] std::vector<double> Nodal(const EarthModel& model,
                                          const std::string& name,
                                          int nu = 0) const;

  /// The model's reference gravity at the nodes, per element.
  [[nodiscard]] std::vector<double> NodalGravity(const EarthModel& model) const;

 private:
  RadialMesh() = default;

  [[nodiscard]] static std::expected<RadialMesh, MeshError> Build(
      std::span<const double> breakpoints, std::span<const int> spanLayer,
      const Options& options);

  int ngll_ = 0;
  std::vector<double> left_;
  std::vector<double> right_;
  std::vector<double> jac_;
  std::vector<double> radius_;  // (element, node) row-major
  std::vector<int> layer_;
  std::vector<double> xi_;
  std::vector<double> weights_;
  std::vector<double> deriv_;  // (ngll, ngll) row-major
};

}  // namespace sl3dpp
