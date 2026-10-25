#pragma once

/**
 * @file Collectives.hpp
 * @brief What a radially distributed computation needs from the processes
 * sharing it, as callbacks, and the pieces of a degree solve split around
 * its one collective (docs/dev/radial-substructuring.md §9-11).
 *
 * The library calls no MPI itself: Distributed.hpp (built with
 * SL3DPP_WITH_MPI) fills Collectives from a communicator, and a test may
 * fill them otherwise (threads in one process, say). Empty callbacks mean
 * one process.
 */

#include <functional>
#include <span>
#include <vector>

namespace sl3dpp {

/// The collective operations over the processes sharing a distributed
/// computation, each in place on a span of the same size on every
/// process. Every process calls them alike and in the same order, from
/// outside any parallel region. Empty: one process (Distributed() false).
struct Collectives {
  /// Each value summed over the processes.
  std::function<void(std::span<double>)> sum{};
  /// Each value's largest over the processes.
  std::function<void(std::span<double>)> max{};
  /// The first process's values, on every process.
  std::function<void(std::span<double>)> share{};

  [[nodiscard]] bool Distributed() const { return static_cast<bool>(sum); }
};

/// How a distributed solve's right-hand side is held: in Whole on every
/// process (a load, a tide: the separators' rows are read from one), or
/// in Shares, each process holding its own segments' rows and its part
/// of the separators' rows, which the solve sums (a force assembled
/// element by element on each process's own elements).
enum class RightHandSide { Whole, Shares };

/// A degree solve begun but not finished (DegreeOperator::BeginSolve,
/// ToroidalOperator::BeginSolve): a distributed caller sums `separators`
/// over the processes, for many solves in one collective, and then
/// finishes each. Without radial cuts the solve is already done and
/// `separators` is empty.
struct PendingSolve {
  /// The separators' reduced right-hand sides, column-major.
  std::vector<double> separators;
  /// The working columns: the scaled right-hand side, then the solution.
  std::vector<double> values;
  /// Each owned segment's interior solve, kept for the back-substitution.
  std::vector<std::vector<double>> interiors;
  int nrhs = 0;
};

}  // namespace sl3dpp
