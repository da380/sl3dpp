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

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Geography.hpp"
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
  /// The 3D command's combined (strain, ocean, rotation) fixed point.
  double coupled_rtol = 1e-9;
  int max_coupled_iterations = 60;
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

}  // namespace sl3dpp::app
