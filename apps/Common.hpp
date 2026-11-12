#pragma once

/**
 * @file Common.hpp
 * @brief The run scaffold the commands with an ocean share (`sea-level`,
 * `kernel1d`, `kernel`): their shared config members and the one reader of
 * them, the Earth built from the `model` block (1-D or 3-D as its files
 * make it), the state (the sea level and the ice at one instant) and the
 * ice change read from their file inputs onto the solver grid, the ice at
 * any time (RunIce, the one place it is formed), the background state
 * built from them, the time grid of the window, and the initial sea
 * level's calibration.
 */

#include <memory>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
/// a radial file the runner refuses them). In the static problem a
/// `thickness` term of the ice change without the state's date is
/// refused, and with the state at the window's end a `change` term.
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

/// The files a run's inputs name, read onto the solver grid: each file
/// once, however many inputs name it (the state's sea level and ice, the
/// land and the ice change's terms are often one file), the variable of
/// every input that names it checked.
class InputFiles {
 public:
  /// The grid must outlive the files.
  explicit InputFiles(const SurfaceGrid& grid) : grid_(&grid) {}

  /// The input's file on the grid, read the first time an input names it
  /// (a file that cannot be read is refused at `object`'s `file`), and
  /// the input's variable checked (refused at `object`, CheckVariable).
  [[nodiscard]] std::shared_ptr<const io::FieldHistory> Open(
      const FileInput& input, const Json& object);

  [[nodiscard]] const SurfaceGrid& Grid() const { return *grid_; }

 private:
  const SurfaceGrid* grid_;
  std::vector<std::pair<FileInput, std::shared_ptr<const io::FieldHistory>>>
      read_;
};

/// The ice of a run on the solver grid, SI: the state's ice (`ocean.ice`
/// read at the state's instant) plus the ice change, the sum of the terms
/// of `forcing.ice_change` (IceChangeTerm). A `thickness` term's change
/// is its file's value less its own value at the state's instant, a
/// `change` term's its file's value, a cap's its field (of a fraction of
/// the state's ice for `cap_fraction`). In the static problem each term
/// is read at its own date and the change is one field, the same whatever
/// time is asked; in a run through time each file is read at the time
/// asked. Every use of the ice goes through it: the state a run starts
/// from, the frozen load, the migrating thickness, the calibration's
/// start and the fingerprint's change.
class RunIce {
 public:
  /// Reads the state's ice at `stateTime` (SI seconds of the files' axes)
  /// and the terms' files through `files`, a term's file held to its kind
  /// (a `thickness` file without a time axis is refused, and in the
  /// static problem a `change` file is held to its date, CheckDate).
  /// Values that are not numbers are refused at the member reading them.
  /// `top`, the config, must outlive the ice.
  RunIce(const SeaLevelRun& run, const Json& top, InputFiles& files,
         double stateTime);

  /// The state's ice thickness, SI.
  [[nodiscard]] const std::vector<double>& State() const { return state_; }
  /// Whether there is a change at all (any term).
  [[nodiscard]] bool Changes() const { return !terms_.empty(); }
  /// The ice change at SI time `time`: the terms' sum, the first taken as
  /// it stands (zero without terms).
  [[nodiscard]] std::vector<double> ChangeAt(double time) const;
  /// The change from SI time `since` to `time`: each file term's value at
  /// `time` less its value at `since` (a cap, the same at every time,
  /// adds nothing). It is ChangeAt(time) - ChangeAt(since), formed term by
  /// term, without passing through the state's instant.
  [[nodiscard]] std::vector<double> ChangeSince(double time,
                                                double since) const;
  /// The ice at SI time `time`: the state's plus ChangeAt(time).
  [[nodiscard]] std::vector<double> At(double time) const;
  /// The ice a change may take beyond what the state holds: 1e-9 times
  /// the state's largest thickness, or 1e-9 m without ice.
  [[nodiscard]] double Tolerance() const;
  /// Refuses, at `forcing.ice_change`, a change that removes more ice than
  /// the state holds at any of the SI times given (beyond Tolerance), with
  /// the worst deficit, its place and, `dated`, its date. Where the state's
  /// own thickness is negative (a resampling's undershoot) it holds none.
  void CheckHeld(std::span<const double> times, bool dated) const;

 private:
  struct Term {
    IceChangeTerm term;
    const Json* member = nullptr;  ///< `thickness`, `change` or the cap
    std::shared_ptr<const io::FieldHistory> file;
    /// A `thickness` term's value at the state's instant.
    std::vector<double> reference;
    /// The static problem's change, and a cap's, read or built once.
    std::vector<double> fixed;
  };
  /// A file term's values at a time, those that are not numbers refused.
  [[nodiscard]] std::vector<double> ValuesOf(const Term& term,
                                             double time) const;
  /// A term's change at a time.
  [[nodiscard]] std::vector<double> TermAt(const Term& term,
                                           double time) const;

  const SurfaceGrid* grid_;
  const Json* block_ = nullptr;  ///< `forcing.ice_change`, where given
  std::vector<double> state_;
  std::vector<Term> terms_;
};

/// The ice change's terms as described (Describe), joined by
/// `separator`; empty without terms.
[[nodiscard]] std::string DescribeIceChange(const SeaLevelRun& run,
                                            std::string_view separator);

/// The world a run moves: the sea level's file on the solver grid, the
/// run's ice, the ice the run starts from (SI), the background built
/// from that ice and the state's sea level, and the state's inputs as
/// read.
struct Loaded {
  std::shared_ptr<const io::FieldHistory> seaLevel;
  std::shared_ptr<const RunIce> ice;
  /// The ice at the window's start, SI: the state's, or, with the state
  /// at the window's end, the state's plus the change back to the start.
  std::vector<double> ice0;
  GRDSolver::State background;
  std::string iceRead;       ///< the state's ice as read, for summaries and files
  std::string seaLevelRead;  ///< the state's sea level as read, likewise
};

/// The surface held as land whatever the flotation, `ocean.land`: its
/// field read through `files` at `time` (SI; a field without a time axis
/// is the same at every date) and cut at one half, with the input as
/// read. Both empty without the member: flotation alone.
struct LandMask {
  std::vector<char> held;
  std::string read;
};
[[nodiscard]] LandMask ReadLand(const SeaLevelRun& run, const Json& top,
                                InputFiles& files, double time);

/// The `ocean` member holding the state's sea level input,
/// `ocean.sea_level` or `ocean.topography`, the one holding its ice,
/// `ocean.ice`, and the ice change, `forcing.ice_change`.
[[nodiscard]] const Json& SeaLevelMember(const SeaLevelRun& run,
                                         const Json& top);
[[nodiscard]] const Json& IceMember(const Json& top);
[[nodiscard]] const Json& IceChangeMember(const Json& top);

/// The static problem's rule of the state's date, once the state's files
/// are read through `files`: `ocean.date_kyr_bp` is required when any of
/// them (the sea level, the ice, the land) has a time axis, and refused
/// when none has one.
void CheckStateDate(const SeaLevelRun& run, const Json& top,
                    InputFiles& files);

/// The sea level the state's sea level input gives at a time (SI seconds
/// of the file's axis), SI: its variable as it stands, or, for
/// `ocean.topography`, the sea level by flotation from it with the given
/// ice thickness (SI), the state's. Values that are not numbers are
/// refused at the input's member.
[[nodiscard]] std::vector<double> SeaLevelFrom(
    const SeaLevelRun& run, const Json& top, const io::FieldHistory& file,
    std::span<const double> ice, double time,
    const SeaLevelParameters& parameters);

/// An ice field's variable at a time (SI seconds), SI: a thickness or a
/// change of it; values that are not numbers are refused.
[[nodiscard]] std::vector<double> IceThicknessAt(const io::FieldHistory& ice,
                                                 std::string_view variable,
                                                 double time);

/// Reads the run through time's world: the state at the end of the window
/// `ocean.at` names (a state's file with a time axis that ends before the
/// window's end is refused when the state is there), the run's ice, held
/// at every node to what the state holds (RunIce::CheckHeld) and, under
/// migrating shorelines with the state at the start, refused where a
/// change is present at the start, and the background built from the
/// ice at the window's start and the state's sea level.
[[nodiscard]] Loaded LoadWorld(const SeaLevelRun& run, const Json& top,
                               const GRDSolver& solver);

/// The state's sea level at the window's end (a topography converted
/// with the state's ice), solver units: the calibration's target.
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

/// The grounded ice load of the change from the ice the run starts from
/// (world.ice0), as the drivers take it: a callable of SLE time over the
/// solver grid. With the state at the window's start the change is the
/// ice change itself, so a change present at the start loads the first
/// node; with the state at its end it is the change since the start. The
/// references must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceLoadOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world);

/// The ice thickness the migrating runs take, solver units: the
/// background's ice plus the same change as IceLoadOf's. The references
/// must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceThicknessOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world);

/// The run's ice history under its shorelines: IceLoadOf when frozen,
/// IceThicknessOf when migrating.
[[nodiscard]] IceHistory HistoryOf(const SeaLevelRun& run,
                                   const GRDSolver& solver,
                                   const Loaded& world);

/// Rebuilds the world's start from the run's ice: its ice at the
/// window's start (world.ice0 too) under the given sea level, the ocean
/// function by flotation.
void StartFromIceHistory(const SeaLevelRun& run, const GRDSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess);

/// Calibrates the world's initial sea level against the state's sea level
/// at the window's end (`"at": "end"`), by an evolution the caller built
/// to stream nothing, and adopts it as the world's background. The start
/// is pyslfp's: the run's ice at the window's start, the target sea level
/// as the guess (StartFromIceHistory), so the sea level's file is read at
/// the end alone. Refused when the calibration runs out of iterations.
/// Returns the misfits before and after (SI metres) and the runs taken.
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
