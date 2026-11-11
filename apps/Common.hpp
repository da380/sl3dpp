#pragma once

/**
 * @file Common.hpp
 * @brief The run scaffold the commands with an ocean share (`sea-level`,
 * `kernel1d`, `kernel`): their shared config members and the one reader of
 * them, the Earth built from the `model` block (1-D or 3-D as its files
 * make it), the background state and ice history read from surface files
 * onto the solver grid, the time grid of the window, and the initial sea
 * level's calibration.
 */

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Blocks.hpp"
#include "Forcing.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/ModelModuli.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace sl3dpp::app {

/// The members `sea-level`, `kernel1d` and `kernel` share: the Earth, the
/// degree, the window, the forcing, the ocean, the rotation and the solves.
/// Without `times` the run is the elastic fingerprint (`sea-level` alone).
struct SeaLevelRun {
  ModelBlock model;
  int lmax = 0;
  /// The window, dates before present; empty for the elastic fingerprint.
  std::optional<TimesBlock> times;
  SeaLevelForcing forcing;
  OceanBlock ocean;
  RotationalFeedback rotation = RotationalFeedback::Full;
  /// With "hydrostatic", the moments of inertia difference is set from
  /// the model's own hydrostatic fluid limit (WithHydrostaticMoments),
  /// which the full transverse feedback needs when the input moments
  /// sustain less oblateness than the model's tidal response.
  bool hydrostatic_moments = false;
  SolverBlock solver = SolverDefaults(SolverOf::SeaLevel);
};

/// The top-level members `ReadSeaLevelRun` reads, for a command's list of
/// its members.
[[nodiscard]] std::vector<std::string_view> SeaLevelRunMemberNames();

/// Refuses every top-level member not among `allowed`: one a block of the
/// layout holds instead (`state`, `coupled`, `shorelines`, ...) is named
/// with the block it belongs in.
void AllowTopLevel(const Json& top, std::span<const std::string_view> allowed);

/// Reads the shared members, relative paths against `directory`. The
/// members that need a `times` block (a viscosity file, the ocean's and
/// the time step's) are refused without one, naming the member, and so
/// are the 3-D step's unless a moduli file is given (the fingerprint on
/// lateral moduli takes them; on a radial file the runner refuses them).
[[nodiscard]] SeaLevelRun ReadSeaLevelRun(
    const Json& top, const std::filesystem::path& directory);

/// Refuses the `solver` members that set the 3-D step (`step`,
/// `step_rtol`, `step_max_iterations`, `transform_degrees`, `cache_mb`)
/// for a run on a radial model, `what` naming the run.
void RefuseThreeDSolver(const Json& top, std::string_view what);

/// The Earth of a run: the deck's model and material at lmax, and the
/// viscosity and moduli files' contents where given.
struct BuiltEarth {
  EarthModel model;
  Material material;
  std::optional<io::ModelViscosity> viscosity;
  std::optional<io::ModelModuli> moduli;

  /// Whether the viscosity file's viscosity, or the moduli file's moduli,
  /// vary laterally: either makes a run through time 3-D.
  [[nodiscard]] bool LateralViscosity() const {
    return viscosity && !viscosity->IsRadial();
  }
  [[nodiscard]] bool LateralModuli() const {
    return moduli && !moduli->IsRadial();
  }
  [[nodiscard]] bool Lateral() const {
    return LateralViscosity() || LateralModuli();
  }
  /// What the model was taken to be, for a summary: `3-D (lateral
  /// viscosity)`, `3-D (lateral moduli)`, `3-D (lateral viscosity, lateral
  /// moduli)`, `1-D`, or `1-D` followed by `(the viscosity file is
  /// radial)`, `(the moduli file is radial)` or `(the viscosity and moduli
  /// files are radial)`.
  [[nodiscard]] std::string Dimension() const;
};

/// Builds the run's Earth: the deck's model and material at lmax, and the
/// model files of `model.viscosity` and `model.moduli` read about their
/// chosen references.
[[nodiscard]] BuiltEarth BuildEarth(const SeaLevelRun& run, const Json& top);

/// The Maxwell rheology of a run through time: the viscosity file's
/// radial reference where given, the deck's viscosity column otherwise.
[[nodiscard]] Rheology RheologyOf(const BuiltEarth& earth, const Json& top);

/// The options of a 3-D run's operator: the transforms, cache and rank
/// cuts (OperatorOptions), the step solver, and the lateral moduli where
/// the model has them.
[[nodiscard]] Viscoelastic3D::Options OperatorOptionsOf(
    const SeaLevelRun& run, const BuiltEarth& earth, const Rheology& rheology);

/// The lateral perturbation delta of a 3-D run's viscosity: the viscosity
/// file's, or zero under the deck's column.
[[nodiscard]] std::function<double(double, double, double)> DeltaOf(
    const BuiltEarth& earth);

/// The SLE parameters, the moments set hydrostatically where asked.
[[nodiscard]] SeaLevelParameters ParametersOf(const SeaLevelRun& run,
                                              const Material& material);

/// The world a run moves: the state and ice histories on the solver
/// grid, the window-start ice (SI), and the background state built from
/// the start's thickness and topography by the flotation construction.
struct Loaded {
  io::FieldHistory state;
  io::FieldHistory ice;
  std::vector<double> ice0;  ///< the start's ice thickness, SI
  GRDSolver::State background;
};
[[nodiscard]] Loaded LoadWorld(const SeaLevelRun& run, const Json& top,
                               const GRDSolver& solver);

/// The sea level the state file gives at the window's end, by the
/// flotation construction, solver units: the calibration's target. Refused
/// when the window's end lies outside the file's time span.
[[nodiscard]] std::vector<double> FinalSeaLevelOf(const SeaLevelRun& run,
                                                  const Json& top,
                                                  const GRDSolver& solver,
                                                  const Loaded& world);

/// The window's uniform time grid, SLE units, zero at its start.
/// The date (kyr before present) of a solver time `t`, which counts
/// forwards from the window's start in the solver's own unit.
[[nodiscard]] double DateKyrBp(const SeaLevelRun& run, const Scales& scales,
                               double t);

[[nodiscard]] std::vector<double> MakeTimes(const SeaLevelRun& run,
                                            const Scales& scales);

/// The grounded ice load relative to the window's start, as the drivers
/// take it: a callable of SLE time over the solver grid. The references
/// must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceLoadOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world);

/// The ice thickness the migrating runs take, solver units: the
/// background's ice plus the history's change from the start's ice
/// (world.ice0, as IceLoadOf measures it). The references must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceThicknessOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world);

/// The run's ice history under its shorelines: IceLoadOf when frozen,
/// IceThicknessOf when migrating.
[[nodiscard]] IceHistory HistoryOf(const SeaLevelRun& run,
                                   const GRDSolver& solver,
                                   const Loaded& world);

/// Rebuilds the world's start from the ice history: its ice at the
/// window's start (world.ice0 too) under the given sea level, the ocean
/// function by flotation.
void StartFromIceHistory(const SeaLevelRun& run, const GRDSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess);

/// Calibrates the world's initial sea level against the state file's sea
/// level at the window's end, as the config asks, by an evolution the
/// caller built to stream nothing, and adopts it as the world's
/// background. The start is pyslfp's: the ice history's ice at the window's
/// start, the target sea level as the guess (StartFromIceHistory), so the
/// state file is read at the end alone. Refused when the calibration runs
/// out of iterations. Returns the misfits before and after (SI metres) and
/// the runs taken.
struct CalibrationSummary {
  double before_m = 0.0;
  double after_m = 0.0;
  std::size_t runs = 0;
};
CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution& evolution,
                                  std::span<const double> times);
CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution3D& evolution,
                                  std::span<const double> times);

}  // namespace sl3dpp::app
