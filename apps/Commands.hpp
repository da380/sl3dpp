#pragma once

/**
 * @file Commands.hpp
 * @brief The sl3d subcommands: each reads a JSON config (Input.hpp), runs
 * one of the library's entry points, writes its outputs with provenance,
 * and returns a one-line summary for the terminal.
 *
 * `love-numbers` solves the per-degree systems of a radial model and
 * writes the SI Love-number table as plain text under a provenance
 * header. `solve` solves the sea level equation, the ocean coupled to the
 * solid Earth, on a state (the sea level and the ice at one instant) read
 * from file inputs (pyslfp's `export_ice_history` writes such files),
 * under any mix of an ice change, surface loads, a tide and seismic
 * sources, with frozen or migrating shorelines: in the static problem the
 * solve at the state's date on the elastic Earth, in the quasi-static
 * problem the run through time from a state at an end of the window,
 * restartable from disk checkpoints; without a state, the same problems
 * on a body without an ocean, the response of the solid body to the
 * loads, the tide and the sources with the rotational feedback as asked.
 * `kernel` gives the adjoint sensitivity kernels of measurements of the
 * same problems, read from the same config, its `kernels` block naming
 * the measurements and the kernels written, and, with `second_order`, the
 * kernels' derivatives along the config's `perturbation` block beside
 * them. `tangent` gives the
 * tangent-linear run of the same problems along a direction in their
 * inputs, the config's `perturbation` block (Perturbation.hpp), written as
 * `solve`'s files, their fields first-order changes, where its `tangent`
 * block says. All three read a config of blocks (Blocks.hpp, Forcing.hpp,
 * Files.hpp) under a top-level `problem`, 1-D or 3-D as its `model` block
 * makes it, and one file serves them all: each checks the others' blocks
 * and does not use them (CheckOtherBlocks), `kernel` and `tangent` the
 * `output`, `checkpoint` and `restart` members of `solve` too. The
 * `trajectory` block is every command's (Trajectory.hpp): `solve` records
 * its forward run into the trajectory store it names, and `kernel` sweeps
 * from that store, and `tangent` runs about it, without running the
 * forward problem, or record into it when it holds no complete recording
 * of the run (Recorded.hpp).
 *
 * The readers are separate from the runners so the configs can be
 * checked without running anything; both throw `planetmodel::Error`
 * naming the config, the member and the fault.
 */

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Blocks.hpp"
#include "Common.hpp"
#include "Forcing.hpp"
#include "Perturbation.hpp"
#include "Trajectory.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp::app {

/// The config of `sl3d love-numbers`.
struct LoveNumbersInput {
  std::filesystem::path model;
  ModelFormat format = ModelFormat::Gia;
  int lmax = 0;
  std::filesystem::path output;
  /// Where to write the conventional dimensionless numbers too
  /// (LoveNumbers::Conventional); empty for none.
  std::filesystem::path conventional_output{};
};

/// Reads a love-numbers config, relative paths against its directory.
[[nodiscard]] LoveNumbersInput ReadLoveNumbersInput(
    const std::filesystem::path& config);

/// Runs a love-numbers config and returns the summary line.
[[nodiscard]] std::string RunLoveNumbers(const std::filesystem::path& config);

/// The config of `sl3d solve`: the sea level equation, the ocean coupled
/// to the solid Earth, under any mix of an ice change, surface loads, a
/// tide and seismic sources, with frozen or migrating shorelines. The
/// static problem is the solve on the state (the sea level and the ice) at
/// its date on the elastic Earth, radial or, under lateral moduli, 3-D;
/// the quasi-static problem is the run through time from a state at an
/// end of the window, its nodes dates before present or times on the
/// run's own clock in a unit (Window), of the radial model, or, under a
/// lateral viscosity or lateral moduli, the 3-D one, restartable from disk
/// checkpoints. Without an `ocean` block the state is absent and the body
/// has no ocean.
struct SolveInput {
  Run run;
  CheckpointBlock checkpoint;  ///< no directory: no checkpoints
  /// The trajectory store the run is recorded into (no directory: none).
  /// A run that resumes from a checkpoint continues the incomplete store
  /// its interrupted run left from the checkpoint's node, and is refused
  /// one that does not hold the run to that node.
  TrajectoryBlock trajectory;
  /// Whether a run finding a checkpoint in the directory resumes from it.
  bool restart = true;
  /// The surface file, the exchange tables of surface coefficients and,
  /// in the static problem, the volume file (Outputs.hpp).
  OutputBlock output;
};

/// Reads the members of a solve config beside its run, relative paths
/// against `directory`, held to the run: `checkpoint` and `restart`
/// (refused in the static problem, and a checkpoint under seismic
/// sources) and `output` (its members held to the problem, the volume
/// file refused under seismic sources and without gravity; required
/// where `outputRequired`, and otherwise read only where given).
[[nodiscard]] SolveInput ReadSolveMembers(
    const Json& top, Run run, const std::filesystem::path& directory,
    bool outputRequired);

/// Reads a solve config, relative paths against its directory, with its
/// `trajectory` block (ReadTrajectoryBlock); a `kernels` block is checked
/// (ReadKernelsBlock) and left unused.
[[nodiscard]] SolveInput ReadSolveInput(const std::filesystem::path& config);

/// Runs (or resumes) a solve config and returns the summary line.
[[nodiscard]] std::string RunSolve(const std::filesystem::path& config);

/// The kernels a kernel config writes, each a set of fields per
/// measurement (`kernels.parameters`): in the ice history, in a surface
/// load at each node (in the water budget, outside it, and a layer on the
/// solid surface), in the tidal potential's coefficients at each node, in
/// the point moment tensors of the run's seismic sources at each node (a
/// fault's patches each one), in the viscosity, and in the elastic moduli
/// kappa and mu.
enum class KernelParameter { Ice, Load, Tide, Moment, Viscosity, Moduli };

/// The word a config names a parameter by.
[[nodiscard]] std::string_view NameOf(KernelParameter parameter);

/// One measurement of a kernel config: a named linear functional of one
/// node's outputs, or the decay time of a site's series over a window of
/// nodes. Its instants are located on the run's nodes when the config is
/// read: a date before present (`date_kyr_bp`) in a dated run, a time on
/// the run's own clock in its unit (`time`) in one counting forwards, and
/// the one node of the static problem, which takes neither.
struct MeasurementSpec {
  enum class Type {
    RelativeSeaLevelCap,
    RelativeSeaLevelPoint,
    DisplacementCap,
    DisplacementPoint,
    DisplacementVectorCap,
    DisplacementVectorPoint,
    PotentialCap,
    PotentialPoint,
    StokesCoefficient,
    Rotation,
    DecayTime
  };
  Type type = Type::RelativeSeaLevelCap;
  std::string word;  ///< the type as the config names it
  std::string name;
  GeographicPoint site{};  ///< caps, points and the decay time's site
  double width = 0.0;      ///< a cap's Gaussian width, radians
  int degree = 0;          ///< a coefficient's degree and order (m < 0
  int order = 0;           ///< the sine term)
  AngularVelocityComponent component = AngularVelocityComponent::Z;
  /// A vector displacement's direction, by its components to the east, to
  /// the north and upwards at the site (a cap's centre): the smoothed
  /// displacement there along it (CapDisplacementMeasurement), or the
  /// displacement at the point.
  SurfaceDirection direction{};
  /// The node a single-instant measurement reads (0 in the static
  /// problem).
  std::size_t node = 0;
  /// The decay time's window of nodes, the first the earlier, and the
  /// estimator's uncertainties (Measurements.hpp): the timing's in the
  /// run's unit (kyr in a dated run), the series' in metres. The fit runs
  /// over every node of the window.
  std::size_t first = 0;
  std::size_t last = 0;
  double time_uncertainty = 0.0;
  double value_uncertainty_m = 0.0;
};

/// The `kernels` block: the adjoint method, the kernels written, the
/// measurements and the directory the kernel files go into. One adjoint
/// sweep, and one set of kernel files, per measurement. How the forward
/// run is recorded is the top-level `trajectory` block's.
struct KernelsBlock {
  /// The discrete adjoint by default: the exact gradient of the stepped
  /// run, its error the forward run's own at any step. The continuous
  /// adjoint converges to the same kernels, accurate once the steps are
  /// short against the shortest relaxation time (docs/adjoints.md §1).
  AdjointMethod method = AdjointMethod::Discrete;
  /// The kernels written, each named once, in the order of
  /// KernelParameter.
  std::vector<KernelParameter> parameters;
  /// `lateral`: the kernels in the viscosity and the moduli as fields over
  /// the memory nodes' grids whatever the background (true), or the radial
  /// profiles of a radial run (false). Absent: fields on a model with
  /// lateral structure, profiles on a radial one; false on a model with
  /// lateral structure is refused when the run is built.
  std::optional<bool> lateral;
  std::vector<MeasurementSpec> measurements;
  std::filesystem::path output;  ///< the directory the files land in
  /// `calibration`, a calibrated run's alone (`"at": "end"`): which
  /// derivatives its kernels are. `derivative` "total" (the default): the
  /// initial sea level re-calibrated as the parameters change, the
  /// calibration's multiplier found by sweeps beside each measurement's
  /// (InitialSeaLevel.hpp), at most `iterations` of them to `rtol`; or
  /// "partial": the calibrated initial sea level held, one sweep. Under
  /// frozen shorelines the two are one and no multiplier's sweep runs.
  bool total = true;
  int calibration_iterations = 20;
  double calibration_rtol = 1e-8;
  /// `second_order`: beside each kernel, its second-order kernel, the
  /// kernel's derivative along the direction of the config's top-level
  /// `perturbation` block (the action of the measurement's Hessian on the
  /// direction), by `method`; false by default, the first-order kernels
  /// alone whatever else the config holds.
  bool second_order = false;

  [[nodiscard]] bool Writes(KernelParameter parameter) const;
};

/// Which derivatives of a calibrated run a derivative command forms
/// (`kernels.calibration`, `tangent.calibration`): the total ones, the
/// initial sea level re-calibrated as the parameters change, by an
/// iteration of at most `iterations` passes to `rtol` (InitialSeaLevel.hpp:
/// the multiplier's sweeps, the linearised calibration's runs), or the
/// partial ones, the calibrated initial sea level held.
struct CalibratedDerivative {
  bool total = true;
  int iterations = 20;
  double rtol = 1e-8;
};

/// Reads a `calibration` member of a derivative command's block: an object
/// of `derivative` ("total", the default, or "partial"), `iterations` (at
/// least 1, default 20) and `rtol` (positive, default 1e-8), refused on a
/// run that does not calibrate, and its iteration's members beside
/// "partial". `pass` names a pass of the iteration ("sweep", "run"), and
/// `partial` and `iteration` the partial derivatives and the iteration as
/// a refusal states them.
[[nodiscard]] CalibratedDerivative ReadCalibratedDerivative(
    const Json& calibration, const Run& run, std::string_view pass,
    std::string_view partial, std::string_view iteration);

/// Reads a list of measurements (`kernels.measurements`,
/// `tangent.measurements`) of a run: at least one, each read against the
/// run, their names plain file names no two share.
[[nodiscard]] std::vector<MeasurementSpec> ReadMeasurements(
    const Json& measurements, const Run& run);

/// Reads a `kernels` block of a run, relative paths against `directory`,
/// and holds it to the run: `method` (a word, default "discrete"),
/// `parameters` (required, distinct words, each one the run gives; its
/// absence refused, listing them), `lateral` (true or false; true refused
/// unless the parameters name the viscosity or the moduli), `measurements`
/// (required, ReadMeasurements: at least one, each located on the run's
/// nodes and held to its band and its ocean), `output` (required, the
/// directory) and `calibration` (an object of `derivative`, `iterations`
/// and `rtol`, refused on a run that does not calibrate, and its
/// iteration's members beside "partial"). The recording's members,
/// `record_stride` and `trajectory`, are refused, naming the top-level
/// `trajectory` block's. The block is refused without gravity, which has
/// no kernels. `second_order` (true or false) is held to the run: the
/// second-order kernels are refused, each by name, of a calibrated
/// run under migrating shorelines as total derivatives (the partial ones,
/// `calibration.derivative` "partial", are built), and of a `decay_time`
/// measurement (the fit's second derivative is not built).
[[nodiscard]] KernelsBlock ReadKernelsBlock(
    const Json& block, const Run& run, const std::filesystem::path& directory);

/// Refuses, at `kernels.second_order`, the second-order kernels of a config
/// with no top-level `perturbation` block, the direction they are along.
/// Every command reading the `kernels` block calls it.
void CheckSecondOrderDirection(const Json& top, const KernelsBlock& block);

/// The config of `sl3d kernel`: a run, read as `solve` reads it (the
/// static problem or the quasi-static one, with an ocean or without,
/// under any forcing, 1-D or 3-D as the model block makes it), the
/// `kernels` block, and the `trajectory` block: the forward run recorded in
/// memory at its stride, or, with a directory, swept from the complete
/// trajectory store of this run there, or recorded into it.
struct KernelInput {
  Run run;
  KernelsBlock kernels;
  TrajectoryBlock trajectory;
  /// The direction of the second-order kernels (`kernels.second_order`),
  /// the top-level `perturbation` block; absent without them.
  std::optional<PerturbationBlock> perturbation;
};

/// Reads a kernel config, relative paths against its directory: the run,
/// the `kernels` block and the `trajectory` block, with `output`,
/// `checkpoint` and `restart` checked where given (ReadSolveMembers) and
/// left unused. The members of the `kernels` block given at the top level
/// are refused, naming it, and so is a run without gravity (`gravity`:
/// "none"), which has no kernels. With `kernels.second_order` the
/// `perturbation` block is read (ReadPerturbationBlock) as the direction
/// of the second-order kernels; without, it is checked and left unused.
[[nodiscard]] KernelInput ReadKernelInput(const std::filesystem::path& config);

/// Runs a kernel config and returns the summary line.
[[nodiscard]] std::string RunKernel(const std::filesystem::path& config);

/// The `tangent` block: what `tangent` does with a config's `perturbation`
/// block. `output` is where its files go, a block of `solve`'s `output`
/// block's members that a tangent-linear run writes (`surface`, required,
/// `fields` and `smoothing`): the surface file `solve` writes, its fields
/// their first-order changes along the direction. `calibration` is a
/// calibrated run's alone (CalibratedDerivative): the total derivative,
/// the linearised calibration's change of the initial sea level in it, or
/// the partial one. `method` is `kernels.method`'s, with its words and
/// its default: "discrete", the derivative of the stepped run, the
/// transpose of the discrete adjoint, under every scheme; or "continuous",
/// the run of the linearised equations stepped by the scheme, whose drive
/// along a change of the relaxation times is the continuous equations'
/// (TangentLinearSolver::Options::method). The two differ along a
/// direction in the viscosity under the exponential trapezoid alone;
/// along the forcing, the state's sea level and the moduli, and along
/// every direction under Sdirk2, they are one run. `measurements` is a list as
/// `kernels.measurements`, the measurements whose first-order changes the
/// run tabulates: the config's `kernels.measurements` where the block
/// gives none (TangentInput::measurements).
struct TangentBlock {
  AdjointMethod method = AdjointMethod::Discrete;
  OutputBlock output;
  CalibratedDerivative calibration;
  std::optional<std::vector<MeasurementSpec>> measurements;
};

/// Reads a `tangent` block of a run, relative paths against `directory`,
/// and holds it to the run: `output` (required; its `exchange`,
/// `exchange_nodes` and `volume` refused, a tangent-linear run's tables
/// and response inside the body not being built) and `calibration`
/// (ReadCalibratedDerivative), `method` and `measurements`
/// (ReadMeasurements).
[[nodiscard]] TangentBlock ReadTangentBlock(
    const Json& block, const Run& run, const std::filesystem::path& directory);

/// The config of `sl3d tangent`: a run, read as `solve` reads it, the
/// `perturbation` block (the direction, Perturbation.hpp), the `tangent`
/// block, the `trajectory` block (the forward run recorded or read back as
/// `kernel`'s is) and the measurements whose first-order changes are
/// formed: `tangent.measurements`, or where the block has none the
/// `kernels` block's (`measuredBy` says which, "tangent" or "kernels");
/// empty where the config gives neither, and the run writes its file
/// alone.
struct TangentInput {
  Run run;
  PerturbationBlock perturbation;
  TangentBlock tangent;
  TrajectoryBlock trajectory;
  std::vector<MeasurementSpec> measurements;
  std::string measuredBy;
};

/// Reads a tangent config, relative paths against its directory: the run,
/// the `perturbation`, `tangent` and `trajectory` blocks, a `kernels`
/// block where given (ReadKernelsBlock, the whole block held to the run as
/// `kernel` holds it), and `output`, `checkpoint` and `restart` checked
/// where given (ReadSolveMembers) and left unused. A run without gravity
/// is refused.
[[nodiscard]] TangentInput ReadTangentInput(
    const std::filesystem::path& config);

/// Runs a tangent config and returns the summary line.
[[nodiscard]] std::string RunTangent(const std::filesystem::path& config);

/// Checks the blocks of the derivative commands a config holds beside
/// those `command` reads itself (`kernels`, `perturbation`, `tangent`), so
/// that one config serves every command and a fault in a block is found
/// whichever runs: each is read by its own reader against the run and
/// left unused.
void CheckOtherBlocks(const Json& top, const Run& run,
                      const std::filesystem::path& directory,
                      std::string_view command);

}  // namespace sl3dpp::app
