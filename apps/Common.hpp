#pragma once

/**
 * @file Common.hpp
 * @brief The run scaffold the commands with an ocean share (`sea-level`,
 * `kernel1d`, `kernel`): their shared config members and the one reader of
 * them, the Earth built from the `model` block (1-D or 3-D as its files
 * make it), the ice and the sea level read from their file inputs onto
 * the solver grid, the background state built from them, the time grid of
 * the window, and the initial sea level's calibration.
 */

#include <memory>
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

/// The members `sea-level`, `kernel1d` and `kernel` share: the problem,
/// the Earth, the degree, the window, the forcing, the ocean, the rotation
/// and the solves. The static problem is the elastic fingerprint
/// (`sea-level` alone).
struct SeaLevelRun {
  Problem problem = Problem::Static;
  ModelBlock model;
  int lmax = 0;
  /// The window, dates before present: the quasi-static problem's.
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
/// members only the quasi-static problem takes (a viscosity file, the
/// ocean's shorelines and calibration, the time step) are refused in the
/// static one, naming the member, and so are the 3-D step's unless a
/// moduli file is given (the fingerprint on lateral moduli takes them; on
/// a radial file the runner refuses them). In the static problem an
/// `ocean.topography` dated otherwise than the ice is refused.
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

/// The world a run moves: the sea level's file and the ice history on
/// the solver grid (one object when they are one file), the ice
/// history's ice at the window's start (SI), the background built
/// from that ice and the sea level, and the two inputs as read.
struct Loaded {
  std::shared_ptr<const io::FieldHistory> seaLevel;
  std::shared_ptr<const io::FieldHistory> ice;
  std::vector<double> ice0;  ///< the start's ice thickness, SI
  GRDSolver::State background;
  std::string iceRead;       ///< the ice as read, for summaries and files
  std::string seaLevelRead;  ///< the sea level as read, likewise
};

/// The `ocean` member holding the run's sea level input,
/// `ocean.sea_level` or `ocean.topography`, and `forcing.ice`.
[[nodiscard]] const Json& SeaLevelMember(const SeaLevelRun& run,
                                         const Json& top);
[[nodiscard]] const Json& IceMember(const Json& top);

/// The ice and the sea level read from the same file, where the two
/// inputs name one, or each from its own: their files opened onto `grid`
/// and their variables checked.
struct OpenedInputs {
  std::shared_ptr<const io::FieldHistory> ice;
  std::shared_ptr<const io::FieldHistory> seaLevel;
};
[[nodiscard]] OpenedInputs OpenInputs(const SeaLevelRun& run, const Json& top,
                                      const SurfaceGrid& grid);

/// The sea level the run's sea level input gives at a time (SI seconds of
/// the file's axis), SI: its variable as it stands, or, for
/// `ocean.topography`, the sea level by flotation from it with the given
/// ice thickness (SI). Values that are not numbers are refused at the
/// input's member.
[[nodiscard]] std::vector<double> SeaLevelFrom(
    const SeaLevelRun& run, const Json& top, const io::FieldHistory& file,
    std::span<const double> ice, double time,
    const SeaLevelParameters& parameters);

/// The ice thickness of the ice's variable at a time (SI seconds), SI;
/// values that are not numbers are refused.
[[nodiscard]] std::vector<double> IceThicknessAt(const io::FieldHistory& ice,
                                                 std::string_view variable,
                                                 double time);

/// Reads the run through time's world: the ice history's ice at the
/// window's start and the sea level at the end of the window its input
/// names (`at`), the background built from the two.
[[nodiscard]] Loaded LoadWorld(const SeaLevelRun& run, const Json& top,
                               const GRDSolver& solver);

/// The sea level input's sea level at the window's end (with the
/// ice history's ice there, where it holds topography), solver units:
/// the calibration's target. Refused when the window's end lies outside
/// the file's time span.
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
/// background's ice plus the history's change from its own start
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

/// Calibrates the world's initial sea level against the sea level input's
/// sea level at the window's end (its `"at": "end"`), by an evolution the
/// caller built to stream nothing, and adopts it as the world's
/// background. The start is pyslfp's: the ice history's ice at the window's
/// start, the target sea level as the guess (StartFromIceHistory), so the
/// sea level's file is read at the end alone. Refused when the calibration runs
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
