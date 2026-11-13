#pragma once

/**
 * @file Trajectory.hpp
 * @brief The top-level `trajectory` block `solve` and `kernel` share, and
 * their use of the library's trajectory store (io::TrajectoryStore,
 * docs/trajectory-store.md): a run's description for the store (its hash,
 * TrajectoryHashOf, its nodes, its stride, its kind and its processes'
 * parts), and the store surveyed, recorded into and opened with every
 * refusal named at `trajectory.directory`.
 */

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "Common.hpp"
#include "Input.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"

namespace sl3dpp::app {

/// The `trajectory` block: where a run's forward trajectory is recorded
/// on disk, and every how many nodes a state is kept. Without a
/// directory, `kernel` records in memory at the stride and `solve`
/// records nothing.
struct TrajectoryBlock {
  std::filesystem::path directory;  ///< empty: no store
  int record_stride = 1;
};

/// Reads the top-level `trajectory` block of a run where given, relative
/// paths against `directory`: `directory` (a path) and `record_stride` (a
/// positive integer, default 1), the stride refused in the static problem,
/// whose one node is recorded whole, and other than 1 under seismic
/// sources, whose states are not rebuilt without the sources' forces. The
/// block is refused without gravity, which has no kernels.
[[nodiscard]] TrajectoryBlock ReadTrajectoryBlock(
    const Json& top, const Run& run, const std::filesystem::path& directory);

/// A run described for the store: its hash (TrajectoryHashOf), the
/// dimension, the nodes' solver times (one at zero for the static
/// problem), the stride, whether it migrates, is under seismic sources or
/// under a tide, and the solver grid's field size. In 1-D the run is one
/// process's, the whole mesh; in 3-D the operator gives this process's part
/// and the radial cuts, and the ranks their collectives.
[[nodiscard]] io::TrajectoryStore::Description TrajectoryOf(
    const Run& run, int stride, std::span<const double> times,
    const GRDSolver& solver);
[[nodiscard]] io::TrajectoryStore::Description TrajectoryOf(
    const Run& run, int stride, std::span<const double> times,
    const GRDSolver& solver, const Viscoelastic3D& op);

/// How `trajectory.directory` stands for the run (io::TrajectoryStore::
/// Survey), the store's refusals named at the member.
[[nodiscard]] io::TrajectoryStore::Standing SurveyTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run);

/// The store at `trajectory.directory` opened for recording the run, or
/// for reading its complete recording, the refusals named at the member.
[[nodiscard]] io::TrajectoryStore RecordTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run);
[[nodiscard]] io::TrajectoryStore OpenTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run);

/// A size on disk for a summary: bytes, KiB, MiB or GiB.
[[nodiscard]] std::string SizeOf(std::uint64_t bytes);

}  // namespace sl3dpp::app
