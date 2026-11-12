#pragma once

/**
 * @file Commands.hpp
 * @brief The sl3d subcommands: each reads a JSON config (Input.hpp), runs
 * one of the library's entry points, writes its outputs with provenance,
 * and returns a one-line summary for the terminal.
 *
 * `love-numbers` solves the per-degree systems of a radial model and
 * writes the SI Love-number table as plain text under a provenance
 * header. `response` solves for the planet's response to a prescribed
 * load and tide, with no ocean: in the static problem the elastic
 * response, in the quasi-static problem the viscoelastic one through
 * time. `sea-level` solves the sea level equation, the ocean coupled to
 * the solid Earth, under an ice load, its ice and sea level read from
 * file inputs (pyslfp's `export_ice_history` writes such files): in the
 * static problem the elastic fingerprint of an ice change, in the
 * quasi-static problem GIA through time under an ice history, restartable
 * from disk checkpoints. `kernel1d` and `kernel` give the adjoint
 * sensitivity kernels of measurements of a run through time. Each of
 * these four reads a config of blocks (Blocks.hpp, Forcing.hpp,
 * Files.hpp) under a top-level `problem`, 1-D or 3-D as its `model` block
 * makes it.
 *
 * The readers are separate from the runners so the configs can be
 * checked without running anything; both throw `planetmodel::Error`
 * naming the config, the member and the fault.
 */

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "Blocks.hpp"
#include "Common.hpp"
#include "Forcing.hpp"
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

/// The config of `sl3d sea-level`: the sea level equation, the ocean
/// coupled to the solid Earth. The static problem is the elastic
/// fingerprint of an ice change on the ice and sea level at a date, on the
/// radial Earth or, under lateral moduli, the 3-D one; the quasi-static
/// problem is GIA through time under an ice history, of the radial model,
/// or, under a
/// lateral viscosity or lateral moduli, the 3-D one, restartable from disk
/// checkpoints.
struct SeaLevelInput {
  SeaLevelRun run;
  CheckpointBlock checkpoint;  ///< no directory: no checkpoints
  /// Whether a run finding a checkpoint in the directory resumes from it.
  bool restart = true;
  OutputBlock output;  ///< `surface` alone
};

/// Reads a sea-level config, relative paths against its directory.
[[nodiscard]] SeaLevelInput ReadSeaLevelInput(
    const std::filesystem::path& config);

/// Runs (or resumes) a sea-level config and returns the summary line.
[[nodiscard]] std::string RunSeaLevel(const std::filesystem::path& config);

/// One measurement of a kernel config: a named linear functional of one
/// time node's outputs, or the decay time of a site's series over a
/// window of nodes. Every date must be a node of the config's times.
struct MeasurementSpec {
  enum class Type {
    RelativeSeaLevelCap,
    RelativeSeaLevelPoint,
    DisplacementCap,
    DisplacementPoint,
    PotentialCap,
    PotentialPoint,
    StokesCoefficient,
    Rotation,
    DecayTime
  };
  Type type = Type::RelativeSeaLevelCap;
  std::string name;
  GeographicPoint site{};  ///< caps, points and the decay time's site
  double width = 0.0;      ///< a cap's Gaussian width, radians
  int degree = 0;          ///< a coefficient's degree and order (m < 0
  int order = 0;           ///< the sine term)
  AngularVelocityComponent component = AngularVelocityComponent::Z;
  double date_kyr_bp = 0.0;  ///< a single-date measurement's node
  /// The decay time's window of nodes and the estimator's uncertainties
  /// (Measurements.hpp): the fit runs over every node of the window.
  double window_start_kyr_bp = 0.0;
  double window_end_kyr_bp = 0.0;
  double time_uncertainty_kyr = 0.0;
  double value_uncertainty_m = 0.0;
};

/// The config of `sl3d kernel1d` and `sl3d kernel`: the members of
/// `sea-level`'s run through time (the quasi-static problem alone), a
/// recording stride, and the
/// measurements — one adjoint sweep, and one set of kernel files, per
/// measurement. `kernel1d` runs a radial model, `kernel` a lateral
/// viscosity.
struct KernelInput {
  SeaLevelRun run;
  int record_stride = 1;
  AdjointMethod method = AdjointMethod::Continuous;
  /// `sl3d kernel` alone: with a directory given, the recorded states
  /// are spilled there during the forward run and read back one at a
  /// time by the sweep, so the trajectory's N / stride states live on
  /// disk instead of in memory — the lever for large lmax.
  std::filesystem::path trajectory_directory;
  std::vector<MeasurementSpec> measurements;
  std::filesystem::path output;  ///< the directory the files land in
};

/// Reads a kernel config, relative paths against its directory.
[[nodiscard]] KernelInput ReadKernelInput(const std::filesystem::path& config);

/// Runs a 1D kernel config and returns the summary line.
[[nodiscard]] std::string RunKernel1D(const std::filesystem::path& config);

/// Runs a 3-D (lateral-viscosity) kernel config.
[[nodiscard]] std::string RunKernel3D(const std::filesystem::path& config);

/// The config of `sl3d response`: the response of the planet to a
/// prescribed surface load and external tidal potential, with no ocean.
/// The static problem is the instantaneous elastic response: of the radial
/// model, or, under lateral moduli, the 3D static problem of
/// docs/adjoints.md §13j with the radial model's response beside it. The
/// quasi-static problem is the viscoelastic response stepped through the
/// nodes of `times`: of the radial model, or, under a lateral viscosity or
/// lateral moduli, the 3D stepper's, with the radial reference model's
/// beside it. A config is made of blocks (Blocks.hpp, Forcing.hpp).
struct ResponseInput {
  Problem problem = Problem::Static;
  ModelBlock model;
  int lmax = 0;
  std::optional<TimesBlock> times;  ///< the quasi-static problem's
  Forcing forcing;
  SolverBlock solver = SolverDefaults(SolverOf::Response);
  OutputBlock output;
};

/// Reads a response config, relative paths against its directory.
[[nodiscard]] ResponseInput ReadResponseInput(
    const std::filesystem::path& config);

/// Runs a response config and returns the summary line.
[[nodiscard]] std::string RunResponse(const std::filesystem::path& config);

}  // namespace sl3dpp::app
