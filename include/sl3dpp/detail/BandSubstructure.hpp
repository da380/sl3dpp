#pragma once

/**
 * @file detail/BandSubstructure.hpp
 * @brief The solve of a symmetric band system by substructuring, the
 * algebra of a distributed radial solve done in one process. Internal.
 *
 * The dofs are cut into contiguous segments by separators, contiguous dof
 * ranges (a cut node's dofs) through which alone the segments couple. Each
 * segment p eliminates its interior I on its own,
 *
 *   S = K_SS - sum_p K_SI^p (K_II^p)^-1 K_IS^p,
 *   g = b_S  - sum_p K_SI^p (K_II^p)^-1 b_I^p,
 *
 * the separators' dofs solve S x_S = g together, and each segment
 * back-substitutes x_I = (K_II^p)^-1 b_I - ((K_II^p)^-1 K_IS^p) x_S, both
 * terms kept from the reduction, so a right-hand side costs one interior
 * solve per segment. Construction factorises each interior block (banded
 * Cholesky, falling back to banded LU as DegreeOperator does), forms the
 * coupling columns (K_II^p)^-1 K_IS^p (one per adjacent separator dof) and
 * factorises S (the same way: it is positive definite whenever the matrix
 * is, and no worse conditioned). docs/dev/radial-substructuring.md §4.
 *
 * Distributed, each process owns some of the segments and factorises and
 * solves those alone; a `combine` callback sums a vector over the
 * processes (an MPI all-reduce, which the library does not call itself):
 * once at construction for S, once per solve for g. Every process then
 * holds S and x_S, and its own segments' x_I. One process owning every
 * segment, with no callback, is the serial solve.
 */

#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace sl3dpp::detail {

class BandSubstructure {
 public:
  /// A vector summed over the processes sharing a solve, in place.
  using Combine = std::function<void(std::span<double>)>;

  /// `ab` is the upper band of a symmetric n x n matrix with kd
  /// superdiagonals, in BandCholeskyFactor's storage, read during
  /// construction only. `separators` are sorted, disjoint dof ranges
  /// [first, end) within [0, n), possibly empty (two segments that do not
  /// couple at all); the segments are the ranges between them (possibly
  /// empty). With `combine`, which sums over the processes, `owned` lists
  /// the segments this process solves (possibly none), every segment
  /// owned by exactly one process; without, one process solves them all
  /// and `owned` is ignored. Throws
  /// std::invalid_argument when a band entry couples two segments
  /// directly, or a segment to a separator other than its two neighbours,
  /// and std::runtime_error when a factorisation fails (Cholesky and LU
  /// both). Distributed, construction is collective (one `combine`).
  BandSubstructure(std::span<const double> ab, int n, int kd,
                   std::vector<std::pair<int, int>> separators,
                   std::span<const int> owned = {}, Combine combine = {});

  /// x <- A^-1 x for nrhs column-major columns of n values each. Every
  /// process passes the whole right-hand side; only its own segments'
  /// rows and the separators' are read (the separators' from the owner of
  /// segment 0), and only those are solved: its own segments' interiors
  /// and every separator. Distributed, collective (one `combine`).
  void Solve(std::span<double> x, int nrhs) const;

  /// Whether this process solves segment p.
  [[nodiscard]] bool Owns(int segment) const {
    return owned_[static_cast<std::size_t>(segment)] != 0;
  }

  [[nodiscard]] int SegmentCount() const {
    return static_cast<int>(segments_.size());
  }
  /// The size of the separators' system.
  [[nodiscard]] int SeparatorDofCount() const { return schur_.n; }
  /// The interior dofs of each segment, [first, first + count).
  [[nodiscard]] std::pair<int, int> SegmentRange(int segment) const {
    const auto& s = segments_[static_cast<std::size_t>(segment)];
    return {s.first, s.count};
  }
  /// Whether every factorisation this process holds (its segments' and the
  /// separators' system) is a Cholesky one.
  [[nodiscard]] bool UsedCholesky() const;

 private:
  /// A band factorisation of a symmetric matrix: Cholesky, or banded LU
  /// when Cholesky fails.
  struct Factor {
    int n = 0;
    int kd = 0;
    bool cholesky = true;
    std::vector<double> values;
    std::vector<int> pivots;

    /// `get(i, j)` gives A(i, j) for i <= j <= i + kd.
    template <typename Get>
    void Build(int size, int bandwidth, Get&& get);
    void Solve(std::span<double> x, int nrhs) const;
  };

  struct Segment {
    int first = 0;  ///< interior dofs [first, first + count)
    int count = 0;
    /// The separators' system indices this segment couples to (its lower
    /// separator's dofs, then its upper's), and K_IS and K_II^-1 K_IS on
    /// them, count x columns, column-major.
    std::vector<int> columns{};
    std::vector<double> coupling{};
    std::vector<double> reduced{};
    Factor interior{};
  };

  int n_ = 0;
  std::vector<std::pair<int, int>> separators_;
  std::vector<Segment> segments_;
  std::vector<char> owned_;
  Combine combine_;
  Factor schur_;
};

}  // namespace sl3dpp::detail
