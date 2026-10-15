#pragma once

/**
 * @file GridCoupling.hpp
 * @brief The grid side of the 3D memory coupling (docs/stage4.md §2–3,
 * §7a): tensor fields over (viscous radial node × angular grid), the
 * transforms between them and their per-degree spectral components, and
 * the assembly of the memory force onto the spheroidal and toroidal
 * degree systems.
 *
 * The radial memory nodes are the GLL nodes of the elements where any
 * rheology branch is active, kept per (element, node) so a shared mesh
 * node carries its two one-sided values, exactly as the nodal fields do.
 * A TensorField holds the five canonical components (SpinComponents.hpp)
 * as complex grid fields per radial node; a SpectralTensor holds their
 * spin-N coefficients. Degrees enter a node's spectral data only within
 * their own element subset (the same truncation rule as the solvers),
 * matching the Fortran's depth loop.
 *
 * Pointwise weight fields (the scheme weights of lateral viscosity, the
 * s(x) of the rate form) multiply TensorFields on the grid; the radial
 * branch moduli mu_k(r) — the elastic structure stays 1D — scale the
 * analysed components per node, and the force assembly applies only the
 * quadrature factors. With angle-uniform weights the whole pipeline
 * reduces exactly to the 1D degree-basis memory force: the zero-lateral
 * gate of stage 4.
 *
 * First version: all orders m in -l..l are carried as complex columns
 * (the Fortran's storage). The Hermitian m >= 0 reduction — halving
 * storage and transforms — is the documented optimisation to take once
 * the lateral iteration is gated.
 */

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SpinComponents.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

namespace sl3dpp {

class GridCoupling {
 public:
  using Complex = std::complex<double>;

  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
    /// Headroom for the pointwise products (the 3/2 rule by default).
    double oversampling = 1.5;
    bool parallel = false;
    /// GSHTrans's BLAS matrix kernel for the batched transforms.
    bool matrix_kernel = false;
  };

  /// The rheology is borrowed and must outlive the coupling.
  GridCoupling(const Rheology& rheology, int lmax, Options options);
  GridCoupling(const Rheology& rheology, int lmax)
      : GridCoupling(rheology, lmax, Options{}) {}

  [[nodiscard]] const SurfaceGrid& Grid() const { return grid_; }
  [[nodiscard]] int LMax() const { return lmax_; }

  // -- the radial memory nodes
  // -------------------------------------------------

  [[nodiscard]] int NodeCount() const {
    return static_cast<int>(nodes_.size());
  }
  [[nodiscard]] int ElementOfNode(int node) const {
    return nodes_[node].element;
  }
  [[nodiscard]] double RadiusOfNode(int node) const {
    return nodes_[node].radius;
  }

  // -- tensors, grid and spectral
  // ----------------------------------------------

  /// The five component fields, [c][node * Grid().FieldSize() + x].
  struct TensorField {
    std::array<std::vector<Complex>, 5> c;
  };
  [[nodiscard]] TensorField ZeroTensor() const;

  /// The spin-N coefficients of the five components per radial node,
  /// [c][node * Grid().SpinCoefficientSize(spin of c) + k].
  struct SpectralTensor {
    std::array<std::vector<Complex>, 5> c;
  };
  [[nodiscard]] SpectralTensor ZeroSpectral() const;

  /// The spins of the component slots.
  static constexpr std::array<int, 5> kSpins{-2, -1, 0, 1, 2};

  /// Synthesis of every component at every node.
  [[nodiscard]] TensorField Synthesise(const SpectralTensor& spectral) const;

  /// Analysis back to spectral components.
  [[nodiscard]] SpectralTensor Analyse(const TensorField& field) const;

  /// Pointwise multiplication by a real weight field over
  /// (node * FieldSize + x), all components alike.
  void Multiply(TensorField& field, std::span<const double> weight) const;

  // -- degree-basis entry and exit
  // -----------------------------------------------

  /// Adds a degree-basis internal-variable state to a spectral tensor:
  /// per degree l, the scalars (M, R, S) as complex (l, m) columns at
  /// every radial node, laid out (node, column m + l), zero below the
  /// degree's element subset. The state enters through
  /// ComponentsFromScalars at each node.
  void AddScalarState(SpectralTensor& spectral, int l,
                      std::span<const Complex> M, std::span<const Complex> R,
                      std::span<const Complex> S) const;

  /// Adds the deviatoric strain of degree solutions to a spectral tensor:
  /// per degree, complex (l, m) columns over the spheroidal system's dofs
  /// (column-major, 2l + 1 columns) and, optionally, the toroidal
  /// system's. The nodal values and their r d_r derivatives are gathered
  /// through the operators' dof maps and the mesh's differentiation
  /// matrix, then mapped by DeviatoricComponents.
  void AddStrainOfSolution(SpectralTensor& spectral, int l,
                           const DegreeOperator& spheroidal,
                           std::span<const Complex> x,
                           const ToroidalOperator* toroidal = nullptr,
                           std::span<const Complex> w = {}) const;

  /// The memory force of an analysed tensor whose components have been
  /// scaled by sum_k mu_k(r) per node: per degree, the complex column
  /// right-hand sides over the spheroidal system's dofs (column m + l of
  /// 2l + 1, column-major) and, where requested, the toroidal system's.
  [[nodiscard]] std::map<int, std::vector<Complex>> SpheroidalForce(
      const SpectralTensor& weighted,
      const std::function<const DegreeOperator&(int)>& system) const;
  [[nodiscard]] std::map<int, std::vector<Complex>> ToroidalForce(
      const SpectralTensor& weighted,
      const std::function<const ToroidalOperator&(int)>& system) const;

  /// Scales every component of an analysed tensor by a per-node factor
  /// (a radial branch modulus, say).
  void ScaleByNode(SpectralTensor& spectral,
                   std::span<const double> factor) const;

 private:
  struct Node {
    int element;
    int nodeInElement;
    double radius;
  };

  /// The first radial memory node whose element is within degree l's
  /// subset (the nodes are element-ordered).
  [[nodiscard]] int FirstNodeFor(int firstElement) const;

  const Rheology* rheology_;
  int lmax_;
  Options options_;
  SurfaceGrid grid_;
  std::vector<Node> nodes_;
  std::map<int, int> firstElementOf_;  // per degree, the truncation
};

}  // namespace sl3dpp
