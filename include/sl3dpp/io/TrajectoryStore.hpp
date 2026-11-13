#pragma once

/**
 * @file TrajectoryStore.hpp
 * @brief A forward run's recording on disk: everything the adjoint sweeps
 * take from the run (the states at the kept nodes, every node's loads,
 * tides and surface fields, a migrating run's ocean functions, the
 * sources' forces and toroidal states of a run under interior sources,
 * and the surface state the run started from), written node by node as
 * the evolution produces it and read back by a later process, so that
 * the sweeps of any number of measurements run from one recording
 * without running the forward problem again. The on-disk form is
 * docs/trajectory-store.md's, a specification a second implementation
 * can follow.
 *
 * A store is a directory: a manifest stating what the recording is of
 * (the caller's run hash, the dimension, the node times, the record
 * stride and the nodes kept, the run's kind, the processes and each one's
 * part of the mesh) and, once the recording finished, every record's size
 * and a flag saying it is complete; the surface state the run started
 * from; one surface record per node; and under `rank_RRRR`, one state
 * record per kept node of the process of that rank. Every file is written
 * to a temporary name and renamed into place, so that a killed run leaves
 * no half record, and a recording that did not finish is known by its
 * manifest.
 *
 * The run hash is the caller's: opaque bytes that must change with
 * whatever changes the forward run's numbers (the model, the degree, the
 * times, the forcing, the ocean, the rotation, the solves' tolerances and
 * the time step), so that a store is only ever read by the run that
 * recorded it. The record stride and the processes' parts are checked
 * beside it.
 *
 * Distributed over radial segments (Viscoelastic3D::Options::segments),
 * every process records its own segment's states in its own
 * subdirectory, and the first process alone writes the manifest, the
 * start and the surface records; the directory must be one every process
 * sees. Survey, Record, Open and Close are then collective over the
 * Description's collectives, and every process reads the surface records
 * and its own states back.
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Collectives.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::io {

/// Why a trajectory store cannot be recorded into or read: each fault its
/// own kind, its message naming the directory or the file.
class TrajectoryStoreError : public std::runtime_error {
 public:
  enum class Kind {
    /// The directory holds files no store writes, or store records with
    /// no manifest, or is not a directory.
    Foreign,
    /// The manifest does not begin with the store's tag.
    Tag,
    /// The manifest is of a format version this program does not read.
    Version,
    /// The manifest was written in another byte order.
    ByteOrder,
    /// The recording did not finish.
    Incomplete,
    /// The store is another run's: its hash is not the caller's.
    Hash,
    /// The store is this run's recorded at another stride.
    Stride,
    /// The store is this run's over other processes, or another partition
    /// of the mesh.
    Layout,
    /// The store's dimension, times or field size are not the caller's
    /// (a hash that does not cover them).
    Mismatch,
    /// A record the manifest lists is absent.
    Missing,
    /// A record is not of the size the manifest gives, or not whole.
    Truncated,
    /// A file could not be written or read.
    Io
  };
  TrajectoryStoreError(Kind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}
  [[nodiscard]] Kind GetKind() const { return kind_; }

 private:
  Kind kind_;
};

class TrajectoryStore {
 public:
  /// The format version this program writes and reads.
  static constexpr std::uint64_t kVersion = 1;

  /// One process's part of a run: the radial segments it owns, contiguous
  /// (Viscoelastic3D::Options::segments; segment 0 of one for a run on one
  /// process), and the grid values of a 3D state's fields on them
  /// (GridCoupling::GridValueCount; 0 in 1D).
  struct Part {
    std::uint64_t first_segment = 0;
    std::uint64_t segments = 1;
    std::uint64_t grid_values = 0;
    [[nodiscard]] bool operator==(const Part&) const = default;
  };

  /// What a recording is of, as its recorder and its readers state it.
  /// Every process describes its own part; the rest is every process's
  /// alike.
  struct Description {
    /// The caller's run hash (opaque; docs/trajectory-store.md).
    std::string hash;
    /// 1 (Evolution) or 3 (Evolution3D).
    int dimension = 1;
    /// The run's node times, the evolution's own (solver units,
    /// increasing; one node for the static problem).
    std::vector<double> times;
    /// The evolution's Options::record_stride (>= 1): every stride-th
    /// node and the last are kept.
    int record_stride = 1;
    /// Whether the shorelines migrate (every surface record carries its
    /// ocean function), the run is under interior sources (every node is
    /// kept, with its forces, its strain and in 1D its toroidal state),
    /// and it is under a prescribed tide.
    bool migrating = false;
    bool sources = false;
    bool tide = false;
    /// The points of a surface field (SurfaceGrid::FieldSize).
    std::uint64_t field_size = 0;
    /// The radial cuts of the run's mesh (Viscoelastic3D::Options::
    /// radial_cuts; empty on one process), every process's alike.
    std::vector<int> cuts;
    /// This process's part, its rank and the count of processes, and the
    /// collectives over them (empty: one process).
    Part part;
    int rank = 0;
    int ranks = 1;
    Collectives collectives{};

    /// Whether node n is kept: every stride-th and the last.
    [[nodiscard]] bool Kept(std::size_t node) const;
  };

  /// What a store's manifest states (docs/trajectory-store.md).
  struct Manifest {
    std::uint64_t version = kVersion;
    bool complete = false;
    std::string hash;
    /// The program and version that wrote it, for provenance.
    std::string writer;
    int dimension = 1;
    std::vector<double> times;
    int record_stride = 1;
    std::vector<bool> kept;
    bool migrating = false;
    bool sources = false;
    bool tide = false;
    std::uint64_t field_size = 0;
    std::vector<int> cuts;
    std::vector<Part> parts;  ///< one per rank
    /// The records' sizes in bytes, header included: the start, each
    /// node's surface record, and each rank's state record of each node
    /// (zero where the node is not kept); all zero until complete.
    std::uint64_t start_bytes = 0;
    std::vector<std::uint64_t> surface_bytes;
    std::vector<std::vector<std::uint64_t>> state_bytes;
  };

  /// How a directory stands for the run a description states: absent or
  /// empty, a complete store of this run recorded as described, or an
  /// incomplete store of this run (one whose recording did not finish).
  enum class Standing { Empty, Complete, Incomplete };

  /// How `directory` stands for `run`. Anything else is refused
  /// (TrajectoryStoreError): files no store writes, a manifest of another
  /// format, another run's store, complete or not, or this run's complete
  /// store at another stride, over other processes or of another shape.
  /// Reads the manifest alone. Collective.
  [[nodiscard]] static Standing Survey(const std::filesystem::path& directory,
                                       const Description& run);

  /// Starts recording `run` into `directory`, created if absent: it must
  /// stand Empty or Incomplete (Survey), and an incomplete store's own
  /// files (the names the store writes: nothing else is there) are
  /// removed first. Writes the manifest, not complete. Collective.
  [[nodiscard]] static TrajectoryStore Record(
      const std::filesystem::path& directory, const Description& run);

  /// Continues recording `run` into the incomplete store of this run in
  /// `directory` from `node`, the node a resumed run's first call starts
  /// from (its checkpoint's): the store must stand Incomplete (Survey) and
  /// be recorded as described (the dimension, times, field size and kind,
  /// the stride and the processes' layout, as a complete store is held to
  /// them), and hold whole every record the run wrote to that node, the
  /// start, the surface record of every node to `node` and every process's
  /// state record of every kept node to it (each record's header the one
  /// its name gives, its length its file's). Those records are kept as the
  /// recording's own; the records of later nodes and every partial file
  /// are removed. A store short of a record is refused
  /// (TrajectoryStoreError::Incomplete) and one recorded otherwise as
  /// Open refuses it, nothing removed. The recording then goes on from the
  /// node after `node` (Attach), and Close completes it as Record's. The
  /// manifest is written by Close alone. Collective.
  [[nodiscard]] static TrajectoryStore Continue(
      const std::filesystem::path& directory, const Description& run,
      std::size_t node);

  /// Opens the complete store of `run` in `directory` for reading: it
  /// must stand Complete (Survey), and every record the manifest lists
  /// (the start and the surface records, and this process's states) must
  /// be there at its size. Collective.
  [[nodiscard]] static TrajectoryStore Open(
      const std::filesystem::path& directory, const Description& run);

  /// The manifest of the store in `directory` as it stands, complete or
  /// not, held to its tag, byte order and version alone: what a tool
  /// inspecting a store reads.
  [[nodiscard]] static Manifest ManifestOf(
      const std::filesystem::path& directory);

  /// The name of a process's subdirectory, `rank_RRRR`.
  [[nodiscard]] static std::string RankDirectoryName(int rank);

  /// A 3D run's part on this process, and its radial cuts (the operator's
  /// options).
  [[nodiscard]] static Part PartOf(const Viscoelastic3D& op);

  // -- recording ---------------------------------------------------------------

  /// The surface state the run starts from, its background (a calibrated
  /// initial sea level's): the first process's start record.
  void PutStart(const GRDSolver::State& background) const;

  /// A node's snapshot as the evolution hands it (Options::on_snapshot):
  /// its surface record, on the first process; in 1D its state record
  /// where the node is kept (the internal state, the toroidal state and
  /// the sources' forces it carries); in 3D the sources' forces it
  /// carries, held for the node's state record (PutState).
  void Put(std::size_t node, const Evolution::Snapshot& snapshot) const;
  void Put(std::size_t node, const Evolution3D::Snapshot& snapshot) const;

  /// A 3D kept node's state as the evolution hands it (Options::
  /// on_state): this process's state record of the node.
  void PutState(std::size_t node, Viscoelastic3D::State&& state) const;

  /// Sets an evolution's options to record into the store: the record
  /// stride, on_snapshot (after any the caller set, which is kept) and, in
  /// 3D, on_state, so that a kept 3D state goes to disk rather than onto
  /// its snapshot. A store continued from node k (Continue) records a
  /// resumed call, whose nodes count from k: the record offset is set to
  /// k, and each node the evolution hands out is put at its node of the
  /// whole run, k after its own. The options' copies hold the store.
  void Attach(Evolution::Options& options) const;
  void Attach(Evolution3D::Options& options) const;

  /// Completes the recording: refuses one that lacks a node's record, and
  /// writes the manifest with every record's size and the complete flag.
  /// Collective.
  void Close() const;

  // -- reading -----------------------------------------------------------------

  [[nodiscard]] const Manifest& GetManifest() const;
  [[nodiscard]] const std::filesystem::path& Directory() const;

  /// The surface state the run started from.
  [[nodiscard]] GRDSolver::State Start() const;

  /// A 1D run's snapshots, every node's surface record with its state
  /// record where kept: what the run's own snapshots held, for
  /// ForwardTrajectory::FromSnapshots and the measurements alike.
  [[nodiscard]] std::vector<Evolution::Snapshot> Snapshots1D() const;

  /// A 3D run's snapshots, without their states (read through
  /// Trajectory3D) and with the sources' forces of a run under sources.
  [[nodiscard]] std::vector<Evolution3D::Snapshot> Snapshots3D() const;

  /// A 3D kept node's state, this process's, read from its record.
  [[nodiscard]] Viscoelastic3D::State State3D(std::size_t node) const;

  /// A 3D run's trajectory over its snapshots (Snapshots3D, or those the
  /// recording run returned): ForwardTrajectory3D::FromSnapshots with the
  /// kept nodes' states fetched from this process's records one at a
  /// time. The trajectory holds the store.
  [[nodiscard]] ForwardTrajectory3D Trajectory3D(
      std::span<const Evolution3D::Snapshot> snapshots) const;

  /// The bytes of every process's records and the manifest, as the
  /// manifest gives them (a complete store's).
  [[nodiscard]] std::uint64_t Bytes() const;

  /// The states this object has read back (State3D, Snapshots1D), and the
  /// records it has written.
  [[nodiscard]] std::size_t StatesRead() const;
  [[nodiscard]] std::size_t RecordsWritten() const;

 private:
  struct Impl;
  explicit TrajectoryStore(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace sl3dpp::io
