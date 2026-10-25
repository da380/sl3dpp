#pragma once

/**
 * @file Ranks.hpp
 * @brief The processes a run of sl3d spans: one, or, built with
 * SL3DPP_WITH_MPI and started under mpiexec, the ranks of MPI_COMM_WORLD.
 *
 * On several ranks the 3D commands (evolve, kernel) distribute their run
 * over radius, one radial segment per rank balanced on the transform
 * work (docs/dev/radial-substructuring.md §11-12): every rank reads the
 * config and the inputs, steps its own memory nodes, and solves the sea
 * level equation alike; rank 0 alone writes the outputs and prints. Each
 * rank checkpoints its own state into its own subdirectory of the
 * checkpoint directory. The other commands refuse several ranks. The MPI
 * calls live in Ranks.cpp, under SL3DPP_HAVE_MPI.
 */

#include <filesystem>
#include <span>
#include <vector>

#include "Commands.hpp"
#include "sl3dpp/Collectives.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::app {

struct Ranks {
  int rank = 0;
  int size = 1;
  /// Over the ranks; empty with one.
  Collectives collectives{};

  [[nodiscard]] bool Distributed() const { return size > 1; }
  /// The rank that writes the outputs and prints.
  [[nodiscard]] bool First() const { return rank == 0; }
};

/// The ranks of this run: one process until StartRanks.
[[nodiscard]] const Ranks& TheRanks();

/// The ranks of MPI_COMM_WORLD (MPI initialised); one without MPI.
void StartRanks();

/// Runs as if on `ranks` until the guard goes (a test's one-rank
/// reference inside an MPI job).
class RanksOverride {
 public:
  explicit RanksOverride(Ranks ranks);
  ~RanksOverride();
  RanksOverride(const RanksOverride&) = delete;
  RanksOverride& operator=(const RanksOverride&) = delete;

 private:
  Ranks saved_;
};

/// The 3D operator's options for a run: the configured transform degrees
/// and, on several ranks, this rank's radial segment (ShareMesh over
/// MPI_COMM_WORLD, balanced on GridCoupling::TransformWeights) and the
/// collectives.
[[nodiscard]] Viscoelastic3D::Options OperatorOptions(const EvolveInput& input,
                                                      const Rheology& rheology);

/// This rank's own subdirectory of a directory of per-run scratch (the
/// checkpoints, the spilled trajectory): rank_RRRR on several ranks, the
/// directory itself on one.
[[nodiscard]] std::filesystem::path RankDirectory(
    const std::filesystem::path& directory);

/// A field over the memory nodes, on rank 0 the ranks' parts laid end to
/// end in rank order (the whole mesh's layout), empty on the others; the
/// field itself on one rank.
[[nodiscard]] std::vector<double> GatherToFirst(std::span<const double> part);

}  // namespace sl3dpp::app
