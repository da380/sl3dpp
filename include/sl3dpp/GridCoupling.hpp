#pragma once

/**
 * @file GridCoupling.hpp
 * @brief The grid side of the 3D memory coupling (docs/dev/lateral-coupling.md
 * §2–3, §7a): tensor fields over (viscous radial node × angular grid), the
 * transforms between them and their per-degree spectral components, and
 * the assembly of the memory force onto the spheroidal and toroidal
 * degree systems.
 *
 * The radial memory nodes are the memory points (MemoryPoints.hpp) of the
 * elements where any rheology branch is active: by default their
 * interior Gauss points, where the strain is interpolated from the nodes
 * and the force rows spread back through the same Lagrange values; or
 * their GLL nodes, ~25% more, kept per (element, node) so a shared mesh
 * node carries its two one-sided values, exactly as the nodal fields do.
 * A TensorField holds the five canonical components (SpinComponents.hpp)
 * as complex grid fields per radial node; a SpectralTensor holds their
 * spin-N coefficients. Degrees enter a node's spectral data only within
 * their own element subset (the same truncation rule as the solvers),
 * matching the Fortran's depth loop.
 *
 * Node blocks (Options::transform_degrees): a degree's solution has no
 * dofs below its truncation radius, so a deep node's strain carries no
 * degree above the largest whose subset reaches its element, and its
 * force rows above that degree do not exist. Each run of nodes sharing a
 * transform degree is a block with its own grid (and Wigner table), laid
 * end to end in every field: a node's grid values start at FieldOffset
 * and number FieldSizeOf, its coefficients are its block's degree's
 * (the (l, m) orders are prefixes of lmax's, so carrying them between
 * blocks is zero padding). Nothing the degree systems keep is dropped; a
 * block's grid resolves the pointwise weights in the products to about
 * its own degree under the 3/2 rule rather than lmax's, which is exact
 * for weights band-limited at that degree. With one block (the default)
 * everything is as on the single lmax grid, bit for bit.
 *
 * Pointwise weight fields (the scheme weights of lateral viscosity, the
 * s(x) of the rate form) multiply TensorFields on the grid; the radial
 * branch moduli mu_k(r) — the elastic structure stays 1D — scale the
 * analysed components per node, and the force assembly applies only the
 * quadrature factors. With angle-uniform weights the whole pipeline
 * reduces exactly to the 1D degree-basis memory force: the zero-lateral
 * gate (LateralGate in the tests).
 *
 * Reality: the pipeline carries REAL tensors, so only the spin -2 and -1
 * components are stored as complex fields and the spin 0 component as a
 * real field; the +1 and +2 components are their pointwise images under
 * A(+N) = (-1)^N conj(A(-N)) and are reconstructed in coefficient space
 * by A^{(N)}_{lm} = (-1)^m conj(A^{(-N)}_{l,-m}) at the force exit — the
 * same orbit relation GSHTrans's tensor layer encodes (eq:reality in
 * Tensor/TensorField.hpp). This halves the storage and the transforms.
 * The contract it rests on: every set of degree columns entering the
 * pipeline is Hermitian, f_{l,-m} = (-1)^m conj(f_{lm}), i.e. represents
 * a real field — the m < 0 spin-0 information is derived, not stored.
 *
 * The degree-basis boundary carries m >= 0 columns only (l + 1 per
 * degree, index m): the fields are real, so the m < 0 columns are the
 * Hermitian images and never exist as data. Scatter writes each
 * column's -m twin into the full-m spin blocks from the derived
 * components; the forces come back as m >= 0 columns likewise. (The
 * Fortran carried all 2l + 1; this halves the banded solves and the
 * boundary loops.)
 */

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/MemoryPoints.hpp"
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
    /// Threaded transforms (GSHTrans) and per-degree force loops (OpenMP),
    /// on OpenMP's thread count (OMP_NUM_THREADS); false forces serial.
    bool parallel = true;
    /// GSHTrans's BLAS matrix kernel for the batched transforms.
    bool matrix_kernel = false;
    /// Where the radial memory nodes sit within each element (the
    /// default matches ViscoelasticOperator's).
    MemoryPointSet memory_points = MemoryPointSet::Gauss;
    /// The degrees the transforms may be cut to with depth (the node
    /// blocks, file comment): each viscous element's memory nodes are
    /// transformed at the least of these, or lmax, that covers every
    /// degree whose dofs reach the element under the truncation rule, on
    /// a grid of that degree with the same oversampling. Empty: every
    /// node at lmax on one grid. Each must lie in [2, lmax].
    std::vector<int> transform_degrees{};
    /// The elements whose memory nodes this coupling holds, [first, end),
    /// an end of -1 meaning the mesh's: a process's own under a radial
    /// distribution (Viscoelastic3D::Options::segments), its forces then
    /// its own elements' rows. The whole mesh by default. A range without
    /// viscous elements holds no nodes.
    std::pair<int, int> elements{0, -1};
  };

  /// The rheology is borrowed and must outlive the coupling.
  GridCoupling(const Rheology& rheology, int lmax, Options options);
  GridCoupling(const Rheology& rheology, int lmax)
      : GridCoupling(rheology, lmax, Options{}) {}
  /// Shares an existing grid (a cheap handle to its Wigner table) rather
  /// than building one: it must be at this lmax with spin coverage >= 2.
  GridCoupling(const Rheology& rheology, int lmax, const SurfaceGrid& grid,
               Options options);

  /// The transform degree of every element of the rheology's mesh under
  /// these options, 0 for an element without memory nodes: what the
  /// blocks follow, found without building a grid (to place radial cuts
  /// before the coupling exists, say). Refuses a transform degree outside
  /// [2, lmax].
  [[nodiscard]] static std::vector<int> TransformDegrees(
      const Rheology& rheology, int lmax, const Options& options);
  /// A model of each element's share of the transform work, for balancing
  /// radial cuts (RadialMesh::BalancedCuts): its memory points times
  /// (L + 1)^3 at its transform degree L, in units of one point at lmax;
  /// 0 without memory nodes. A formula, not a timing, so that every
  /// process places the same cuts.
  [[nodiscard]] static std::vector<double> TransformWeights(
      const Rheology& rheology, int lmax, const Options& options);

  /// The grid at lmax (shared or built): every node's with one block, the
  /// lmax block's otherwise (GridOf for a node's own).
  [[nodiscard]] const SurfaceGrid& Grid() const { return grid_; }
  [[nodiscard]] int LMax() const { return lmax_; }

  // -- the node blocks
  // -------------------------------------------------

  /// The runs of memory nodes transformed at one degree, bottom up; one
  /// unless Options::transform_degrees cuts some.
  [[nodiscard]] int BlockCount() const {
    return static_cast<int>(blocks_.size());
  }
  [[nodiscard]] int FirstNodeOfBlock(int block) const {
    return blocks_[block].firstNode;
  }
  [[nodiscard]] int NodeCountOfBlock(int block) const {
    return blocks_[block].nodeCount;
  }
  [[nodiscard]] const SurfaceGrid& GridOfBlock(int block) const {
    return blocks_[block].grid;
  }
  [[nodiscard]] int BlockOfNode(int node) const { return blockOf_[node]; }
  /// The grid a node's values sit on, at its block's degree.
  [[nodiscard]] const SurfaceGrid& GridOf(int node) const {
    return blocks_[blockOf_[node]].grid;
  }
  /// Where a node's grid values start in a field over the nodes (a
  /// TensorField component, a weight, a kernel), and how many there are:
  /// node * Grid().FieldSize() and Grid().FieldSize() with one block.
  [[nodiscard]] std::size_t FieldOffset(int node) const {
    return fieldOffset_[node];
  }
  [[nodiscard]] std::size_t FieldSizeOf(int node) const {
    return fieldOffset_[node + 1] - fieldOffset_[node];
  }
  /// The grid values over every node: the size of a field over them.
  [[nodiscard]] std::size_t GridValueCount() const {
    return fieldOffset_.back();
  }
  /// A real field over the nodes, in this layout, on the lmax grid at
  /// every node ([node * Grid().FieldSize() + x]): a lower block's node
  /// is expanded on its own grid and synthesised on Grid() (exact for a
  /// field band-limited at its block's degree; a projection onto those
  /// degrees otherwise), an lmax node copied. For writing and plotting.
  [[nodiscard]] std::vector<double> OnSurfaceGrid(
      std::span<const double> values) const;

  // -- the radial memory nodes
  // -------------------------------------------------

  [[nodiscard]] const MemoryPoints& Points() const { return points_; }
  [[nodiscard]] int NodeCount() const {
    return static_cast<int>(nodes_.size());
  }
  [[nodiscard]] int ElementOfNode(int node) const {
    return nodes_[node].element;
  }
  [[nodiscard]] double RadiusOfNode(int node) const {
    return nodes_[node].radius;
  }
  /// The node's point index within its element (its GLL node index at the
  /// GLL points).
  [[nodiscard]] int NodeInElementOf(int node) const {
    return nodes_[node].nodeInElement;
  }
  /// The node's radial quadrature weight, w_p times the element Jacobian.
  [[nodiscard]] double WeightOfNode(int node) const;

  // -- tensors, grid and spectral
  // ----------------------------------------------

  /// The stored component fields of a real tensor: the spin -2 and -1
  /// components, [c][FieldOffset(node) + x] on GridOf(node), and the real
  /// spin 0 component likewise; spins +1 and +2 are derived (file
  /// comment).
  struct TensorField {
    std::array<std::vector<Complex>, 2> c;
    std::vector<double> r;
  };
  [[nodiscard]] TensorField ZeroTensor() const;

  /// The coefficients of the stored components per radial node, to its
  /// block's degree: spin -2 and -1 over all orders, each node's
  /// GridOf(node).SpinCoefficientSize(spin) values end to end (node *
  /// Grid().SpinCoefficientSize(spin) + k with one block), and the spin 0
  /// component in the real m >= 0 layout likewise (Index(l, m) within a
  /// node).
  struct SpectralTensor {
    std::array<std::vector<Complex>, 2> c;
    std::vector<Complex> r;
  };
  [[nodiscard]] SpectralTensor ZeroSpectral() const;

  /// The spins of the stored complex component slots.
  static constexpr std::array<int, 2> kSpins{-2, -1};

  /// Synthesis of every component at every node.
  [[nodiscard]] TensorField Synthesise(const SpectralTensor& spectral) const;
  /// The same into a caller's tensor, every value overwritten: reusing
  /// one saves allocating (and the system zeroing) a fresh one per call.
  /// Shaped as ZeroTensor's; an empty one is shaped first.
  void Synthesise(const SpectralTensor& spectral, TensorField& field) const;

  /// Analysis back to spectral components.
  [[nodiscard]] SpectralTensor Analyse(const TensorField& field) const;
  /// The same into a caller's spectral tensor, every value overwritten;
  /// shaped as ZeroSpectral's, an empty one shaped first.
  void Analyse(const TensorField& field, SpectralTensor& spectral) const;

  /// Pointwise multiplication by a real weight field over the nodes
  /// (FieldOffset(node) + x), all components alike.
  void Multiply(TensorField& field, std::span<const double> weight) const;

  // -- degree-basis entry and exit
  // -----------------------------------------------

  /// Adds a degree-basis internal-variable state to a spectral tensor:
  /// per degree l, the scalars (M, R, S) as complex m >= 0 columns at
  /// every radial node, laid out (node, column m), zero below the
  /// degree's element subset. The state enters through
  /// ComponentsFromScalars at each node.
  void AddScalarState(SpectralTensor& spectral, int l,
                      std::span<const Complex> M, std::span<const Complex> R,
                      std::span<const Complex> S) const;

  /// Adds the deviatoric strain of degree solutions to a spectral tensor:
  /// per degree, complex m >= 0 columns over the spheroidal system's dofs
  /// (column-major, l + 1 columns) and, optionally, the toroidal
  /// system's. The values and their r d_r derivatives at the memory nodes
  /// are gathered through the operators' dof maps and the element's
  /// Lagrange basis, then mapped by DeviatoricComponents.
  void AddStrainOfSolution(SpectralTensor& spectral, int l,
                           const DegreeOperator& spheroidal,
                           std::span<const Complex> x,
                           const ToroidalOperator* toroidal = nullptr,
                           std::span<const Complex> w = {}) const;

  /// The memory force of an analysed tensor whose components have been
  /// scaled by sum_k mu_k(r) per node: per degree, the complex column
  /// right-hand sides over the spheroidal system's dofs (column m of
  /// l + 1, column-major) and, where requested, the toroidal system's.
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

  /// A run of nodes at one transform degree, and its grid at that degree.
  struct Block {
    int firstNode = 0;
    int nodeCount = 0;
    SurfaceGrid grid;
  };

  /// The first radial memory node of degree l's work: its element within
  /// the subset starting at `firstElement` (the nodes are
  /// element-ordered) and its block's degree at least l.
  [[nodiscard]] int FirstNodeFor(int firstElement, int l) const;

  /// Adds one (l, m) column's five components to the stored blocks.
  void Scatter(SpectralTensor& spectral, int node, int l, int m,
               std::span<const Complex, 5> A) const;
  /// The five components of one (l, m) column from the stored blocks,
  /// the derived ones reconstructed by the reality relation.
  [[nodiscard]] std::array<Complex, 5> Gather(const SpectralTensor& spectral,
                                              int node, int l, int m) const;

  const Rheology* rheology_;
  int lmax_;
  Options options_;
  SurfaceGrid grid_;
  MemoryPoints points_;
  std::vector<Node> nodes_;
  std::map<int, int> firstElementOf_;  // per degree, the truncation
  std::vector<Block> blocks_;
  std::vector<int> blockOf_;  // per node
  // Per node and one past the last, where each node's values start: its
  // grid values, spin -2 and -1 coefficients and spin 0 coefficients.
  std::vector<std::size_t> fieldOffset_;
  std::array<std::vector<std::size_t>, 2> spinOffset_;
  std::vector<std::size_t> coefficientOffset_;
  std::vector<int> firstNodeOfDegree_;  // per degree, its first block's
};

}  // namespace sl3dpp
