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
 * the measurements and the kernels written. Both read a config of blocks
 * (Blocks.hpp, Forcing.hpp, Files.hpp) under a top-level `problem`, 1-D
 * or 3-D as its `model` block makes it, and one file serves both: `solve`
 * checks a `kernels` block and does not use it, and `kernel` checks the
 * `output`, `checkpoint` and `restart` members of `solve` and does not
 * use them. The `trajectory` block is both commands' (Trajectory.hpp):
 * `solve` records its forward run into the trajectory store it names, and
 * `kernel` sweeps from that store without running the forward problem, or
 * records into it when it holds no complete recording of the run.
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
  /// the north and upwards in the local frame at each point.
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
  AdjointMethod method = AdjointMethod::Continuous;
  /// The kernels written, each named once, in the order of
  /// KernelParameter.
  std::vector<KernelParameter> parameters;
  std::vector<MeasurementSpec> measurements;
  std::filesystem::path output;  ///< the directory the files land in

  [[nodiscard]] bool Writes(KernelParameter parameter) const;
};

/// Reads a `kernels` block of a run, relative paths against `directory`,
/// and holds it to the run: `method` (a word, default "continuous"),
/// `parameters` (required, distinct words, each one the run gives; its
/// absence refused, listing them), `measurements` (required, at least
/// one, each located on the run's nodes and held to its band and its
/// ocean) and `output` (required, the directory). The
/// recording's members, `record_stride` and `trajectory`, are refused,
/// naming the top-level `trajectory` block's. The block is refused without
/// gravity, which has no kernels.
[[nodiscard]] KernelsBlock ReadKernelsBlock(
    const Json& block, const Run& run, const std::filesystem::path& directory);

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
};

/// Reads a kernel config, relative paths against its directory: the run,
/// the `kernels` block and the `trajectory` block, with `output`,
/// `checkpoint` and `restart` checked where given (ReadSolveMembers) and
/// left unused. The members of the `kernels` block given at the top level
/// are refused, naming it, and so is a run without gravity (`gravity`:
/// "none"), which has no kernels.
[[nodiscard]] KernelInput ReadKernelInput(const std::filesystem::path& config);

/// Runs a kernel config and returns the summary line.
[[nodiscard]] std::string RunKernel(const std::filesystem::path& config);

}  // namespace sl3dpp::app
