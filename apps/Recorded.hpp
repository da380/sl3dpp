#pragma once

/**
 * @file Recorded.hpp
 * @brief The recorded forward run the derivative commands linearise about
 * (`kernel`'s sweeps, `tangent`'s run), and the measurements read off it.
 *
 * The forward run is `solve`'s run of the config's blocks, built by the
 * same pieces (Common.hpp): the static problem as a run of one node, or
 * the quasi-static problem through time, radial or, as the `model` block
 * makes it, 3-D, its initial sea level calibrated first where the state
 * is at the window's end. Its snapshots are recorded in memory at the
 * `trajectory` block's stride, or, with a directory, into the trajectory
 * store there (Trajectory.hpp), or read back from the complete store of
 * the run there without running it. Each runner builds the run and hands
 * it, with what it ran on, to the command's own work (`then`), whose
 * summary it returns with the store's account after it.
 */

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Forcing.hpp"
#include "Input.hpp"
#include "Trajectory.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::app {

/// What a recorded run ran on, whatever its dimension: the run, its Earth,
/// its sea level solver, the library's grid it stepped (StepGridOf; the
/// static problem's one node), its world (`world` through time, `fixed` in
/// the static problem, the other null), its surface loads (null without
/// any) and what the config was taken to mean with the inputs as read.
struct RecordedRun {
  const Run* run = nullptr;
  const BuiltEarth* earth = nullptr;
  const GRDSolver* solver = nullptr;
  StepGrid steps;
  const Loaded* world = nullptr;
  const StaticWorld* fixed = nullptr;
  const SurfaceLoads* loads = nullptr;
  std::string read;

  /// The state's inputs as read (empty without an ocean).
  [[nodiscard]] std::string_view IceRead() const {
    return world ? world->iceRead : fixed->iceRead;
  }
  [[nodiscard]] std::string_view SeaLevelRead() const {
    return world ? world->seaLevelRead : fixed->seaLevelRead;
  }
  /// The background state the run started from (a calibrated run's, the
  /// calibrated one).
  [[nodiscard]] const GRDSolver::State& Background() const {
    return world ? world->background : fixed->background;
  }
  /// The run's ice (null without an ocean).
  [[nodiscard]] const RunIce* Ice() const {
    return world ? world->ice.get() : fixed->ice.get();
  }
};

/// A recorded radial run: the rheology and operator it stepped, the
/// evolution about its background state (marked where the state was
/// calibrated under migrating shorelines) and its snapshots at the
/// library's nodes.
struct Recorded1D : RecordedRun {
  const Rheology* rheology = nullptr;
  const ViscoelasticOperator* op = nullptr;
  const Evolution* evolution = nullptr;
  std::span<const Evolution::Snapshot> snapshots;
};

/// A recorded 3-D run, as the radial one, with its trajectory (the
/// store's, fetched back as a sweep reaches each state, or the snapshots'
/// own).
struct Recorded3D : RecordedRun {
  const Rheology* rheology = nullptr;
  const Viscoelastic3D* op = nullptr;
  const Evolution3D* evolution = nullptr;
  std::span<const Evolution3D::Snapshot> snapshots;
  const ForwardTrajectory3D* trajectory = nullptr;
};

/// The static problem on a radial Earth: the one-node run of the radial
/// evolution on a branchless rheology, recorded, or read from the
/// trajectory store of the run.
[[nodiscard]] std::string WithRecordedStatic1D(
    const Run& run, const TrajectoryBlock& trajectory, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded1D&)>& then);

/// The static problem on lateral moduli: the one-node run of the 3-D
/// evolution on the stepper of a branchless rheology carrying the moduli,
/// distributed over radial segments on several ranks, recorded, or read
/// from the trajectory store of the run.
[[nodiscard]] std::string WithRecordedStatic3D(
    const Run& run, const TrajectoryBlock& trajectory, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded3D&)>& then);

/// A radial run through time: the radial rheology of the deck or of a
/// radial viscosity file, its initial sea level calibrated first where
/// asked, then the recorded run; or, from the complete trajectory store of
/// the run, its calibrated start and its recording read back.
[[nodiscard]] std::string WithRecorded1D(
    const Run& run, const TrajectoryBlock& trajectory, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded1D&)>& then);

/// A 3-D run through time (a lateral viscosity, lateral moduli or both),
/// distributed over radial segments on several ranks: recorded in memory,
/// or into the trajectory store of `trajectory.directory` (each rank's
/// states in its own subdirectory) and fetched back one at a time, or read
/// from the complete store of the run there without the forward run.
[[nodiscard]] std::string WithRecorded3D(
    const Run& run, const TrajectoryBlock& trajectory, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded3D&)>& then);

/// What one measurement becomes against a run: the node-keyed duals, the
/// forward value, the unit of J (J_SI = factor * J_nd) with its SI
/// dimensions and its unit's name, and, for a decay time, the fit.
struct Built {
  std::map<std::size_t, Measurement> measurements;
  double value_nd = 0.0;
  double factor = 1.0;
  Dimensions objective = {};
  std::string unit;
  std::optional<DecayTimeFit> fit;
  /// The window's nodes the duals sit on; `measurements` is keyed by their
  /// library nodes (StepGrid), the right limit at a repeat.
  std::vector<std::size_t> nodes;
};

/// A measurement of a `kernels` block against a run's snapshots at the
/// library's nodes (`steps`): a measurement dated at a repeated node reads
/// its right limit, the co-seismic field.
template <class Snapshot>
[[nodiscard]] Built BuildMeasurement(const MeasurementSpec& spec,
                                     const Run& run, const GRDSolver& solver,
                                     std::span<const Snapshot> snapshots,
                                     const StepGrid& steps);

/// The duals of one node paired with a snapshot's fields there: the
/// measurement's part of J, non-dimensional, a dual of a field the
/// snapshot does not carry (a toroidal potential a run does not have)
/// pairing with nothing. Linear in the snapshot: paired with a
/// tangent-linear run's, the first-order change of the measurement.
template <class Snapshot>
[[nodiscard]] double PairedWith(const GRDSolver& solver,
                                const Measurement& measurement,
                                const Snapshot& snapshot);

/// An adjoint method's word, as `kernels.method` names it.
[[nodiscard]] std::string_view MethodWord(AdjointMethod method);

}  // namespace sl3dpp::app
