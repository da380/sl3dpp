#pragma once

/**
 * @file Commands.hpp
 * @brief The sl3d subcommands: each reads a JSON config (Input.hpp), runs
 * one of the library's entry points, writes its outputs with provenance,
 * and returns a one-line summary for the terminal.
 *
 * `love-numbers` solves the per-degree systems of a radial model and
 * writes the SI Love-number table as plain text under a provenance
 * header. `fingerprint` solves the elastic sea level equation for an
 * ice change on a present-day state read from a planetmodel surface
 * file (pyslfp's `export_ice_history` writes one), and writes the
 * fingerprint — sea-level change, displacement, potential, the ice
 * change and the ocean function, with the angular-velocity perturbation
 * in the metadata — as a planetmodel surface file.
 *
 * The readers are separate from the runners so the configs can be
 * checked without running anything; both throw `planetmodel::Error`
 * naming the config, the member and the fault.
 */

#include <filesystem>
#include <string>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp::app {

/// The deck formats a config may name: planetmodel's GIA and MINEOS
/// tables, and the library's six-column elastic table (rho, vpv, vsv,
/// vph, vsh, eta; no header), the test and example decks' format.
enum class ModelFormat { Gia, Mineos, Elastic };

/// The format a `ModelFormat` stands for.
[[nodiscard]] DeckFormat FormatOf(ModelFormat format);

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

/// How a fingerprint config changes the ice.
struct IceChange {
  enum class Kind { CapFraction, CapThickness } kind = Kind::CapFraction;
  GeographicPoint site;
  double width = 0.0;   ///< the Gaussian cap's width, radians
  double amount = 0.0;  ///< the fraction, or the thickness in metres
};

/// The config of `sl3d fingerprint`.
struct FingerprintInput {
  std::filesystem::path model;
  ModelFormat format = ModelFormat::Gia;
  int lmax = 0;
  std::filesystem::path state;  ///< a planetmodel surface file
  double date_kyr_bp = 0.0;     ///< the state's date, kyr before present
  IceChange change;
  RotationalFeedback rotation = RotationalFeedback::Full;
  std::filesystem::path output;  ///< the surface file to write
};

/// Reads a fingerprint config, relative paths against its directory.
[[nodiscard]] FingerprintInput ReadFingerprintInput(
    const std::filesystem::path& config);

/// Runs a fingerprint config and returns the summary line.
[[nodiscard]] std::string RunFingerprint(const std::filesystem::path& config);

/// The config of `sl3d evolve1d`: a 1D viscoelastic GIA run under an
/// ice history, restartable from disk checkpoints.
struct EvolveInput {
  std::filesystem::path model;
  ModelFormat format = ModelFormat::Gia;
  int lmax = 0;
  /// `sl3d evolve` alone: a planetmodel model file whose viscosity,
  /// eta(x) = eta_bar(r) exp(delta(x)), supplies both the radial Maxwell
  /// body and the lateral perturbation (io/ModelViscosity); empty for
  /// the 1D command, whose viscosity is the deck's own column.
  std::filesystem::path viscosity;
  bool viscosity_file_radial = false;  ///< the file's own radial reference
  std::filesystem::path state;        ///< a planetmodel surface file
  std::filesystem::path ice_history;  ///< likewise; often the same file
  double start_kyr_bp = 0.0;          ///< the window, kyr before present
  double end_kyr_bp = 0.0;
  int nodes = 0;                      ///< time nodes, the ends included
  Scheme scheme = Scheme::ExponentialTrapezoid;
  RotationalFeedback rotation = RotationalFeedback::Full;
  /// The shorelines: frozen at the start's (the linear equation), or
  /// following flotation.
  Shorelines shorelines = Shorelines::Frozen;
  /// With `calibrate`, the initial sea level is calibrated so that the run
  /// reaches the state file's sea level at the window's end, by the method
  /// and within the iterations and tolerance given (InitialSeaLevel.hpp),
  /// instead of being read from the state file at the window's start.
  bool calibrate = false;
  CalibrationMethod calibration_method = CalibrationMethod::FixedPoint;
  int calibration_iterations = 10;
  double calibration_rtol = 1e-4;
  /// The 3D command's combined (strain, ocean, rotation) fixed point.
  double coupled_rtol = 1e-9;
  int max_coupled_iterations = 60;
  /// The 3D commands' transforms cut with depth: the degrees a memory
  /// node's grid may take below lmax (Viscoelastic3D::Options::
  /// transform_degrees); empty, every node at lmax.
  std::vector<int> transform_degrees;
  /// With "hydrostatic", the moments of inertia difference is set from
  /// the model's own hydrostatic fluid limit (WithHydrostaticMoments),
  /// which the full transverse feedback needs when the input moments
  /// sustain less oblateness than the model's tidal response.
  bool hydrostatic_moments = false;
  double rtol = 1e-9;
  /// Checkpointing: every `checkpoint_every`-th node's recurrence and
  /// the history so far go to `checkpoint_directory` (none when the
  /// directory is empty); a run finding a checkpoint there resumes from
  /// it unless `restart` is false; a positive `max_walltime_min` makes
  /// the run checkpoint and stop cleanly once the wall clock passes it.
  std::filesystem::path checkpoint_directory;
  int checkpoint_every = 0;
  double max_walltime_min = 0.0;
  bool restart = true;
  std::filesystem::path output;  ///< the surface file of SI histories
};

/// Reads an evolve config, relative paths against its directory.
[[nodiscard]] EvolveInput ReadEvolveInput(const std::filesystem::path& config);

/// Runs (or resumes) a 1D evolve config and returns the summary line.
[[nodiscard]] std::string RunEvolve(const std::filesystem::path& config);

/// Runs (or resumes) a 3D (lateral-viscosity) evolve config.
[[nodiscard]] std::string RunEvolve3D(const std::filesystem::path& config);

/// One measurement of a kernel config: a named linear functional of one
/// time node's outputs, or the decay time of a site's series over a
/// window of nodes. Every date must be a node of the config's times.
struct MeasurementSpec {
  enum class Type {
    SeaLevelCap,
    SeaLevelPoint,
    DisplacementCap,
    DisplacementPoint,
    PotentialCap,
    PotentialPoint,
    StokesCoefficient,
    Rotation,
    DecayTime
  };
  Type type = Type::SeaLevelCap;
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

/// The config of `sl3d kernel1d` and `sl3d kernel`: the evolve commands'
/// run members, a recording stride, and the measurements — one adjoint
/// sweep, and one set of kernel files, per measurement.
struct KernelInput {
  EvolveInput run;
  int record_stride = 1;
  AdjointMethod method = AdjointMethod::Continuous;  ///< kernel1d alone
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

/// Runs a 3D (lateral-viscosity) kernel config.
[[nodiscard]] std::string RunKernel3D(const std::filesystem::path& config);

}  // namespace sl3dpp::app
