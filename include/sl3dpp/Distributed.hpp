#pragma once

/**
 * @file Distributed.hpp
 * @brief The radial distribution over MPI ranks: one contiguous segment of
 * the radial mesh per rank, the per-degree solves substructured over the
 * segments (docs/dev/radial-substructuring.md). Compiled only with the CMake
 * option SL3DPP_WITH_MPI, which defines SL3DPP_HAVE_MPI; without it this
 * header declares nothing.
 *
 * The solvers themselves call no MPI: DegreeOperator and ToroidalOperator
 * take the cuts, the segments a process owns and a `combine` callback that
 * sums a vector over the processes. This header supplies those from a
 * communicator, so that a rank builds its degree systems as
 *
 *     const auto share = sl3dpp::ShareMesh(mesh, MPI_COMM_WORLD);
 *     DegreeOperator system(material, l, {.cuts = share.cuts,
 *                                         .segments = share.segments,
 *                                         .combine = share.combine});
 *
 * and every construction and Solve is then collective over the
 * communicator.
 */

#ifdef SL3DPP_HAVE_MPI

#include <mpi.h>

#include <functional>
#include <span>
#include <vector>

#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

/// The `combine` callback of the distributed solves: an in-place sum over
/// the ranks of `comm` (MPI_Allreduce). The communicator is borrowed and
/// must outlive the callback.
[[nodiscard]] std::function<void(std::span<double>)> SumOver(MPI_Comm comm);

/// One rank's share of the radial mesh: the cuts (the same on every rank),
/// the segments it owns, and the sum over the communicator.
struct RadialShare {
  std::vector<int> cuts;
  std::vector<int> segments;
  std::function<void(std::span<double>)> combine;
};

/// Cuts the mesh into one segment per rank of `comm`
/// (RadialMesh::BalancedCuts, balancing the element weights; one per
/// element when empty), rank r owning segment r from the bottom. Throws
/// std::invalid_argument when the layers leave too few element boundaries
/// for the ranks. Collective only in that every rank must call it alike.
[[nodiscard]] RadialShare ShareMesh(const RadialMesh& mesh, MPI_Comm comm,
                                    std::span<const double> weights = {});

}  // namespace sl3dpp

#endif  // SL3DPP_HAVE_MPI
