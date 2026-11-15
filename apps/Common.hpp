#pragma once

/**
 * @file Common.hpp
 * @brief The run scaffold the commands of the sea level equation share
 * (`solve` and `kernel`): their shared config members and the one reader
 * of them, the Earth built from the `model` block (1-D or 3-D as its
 * files make it), the state (the sea level and the ice at one instant,
 * where the run has an ocean) and the ice change read from their file
 * inputs onto the solver grid, the ice at any time (RunIce, the one place
 * it is formed), the background state built from them, the time grid of
 * the window, the initial sea level's calibration, the forcing in the
 * evolutions' form and the drive of a run through time, the static
 * problem's world and its one-node solve, and the summary's account of
 * what a config was taken to mean.
 */

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
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

/// What jumps at a time of a run's forcing: the surface terms by the
/// kernels they have (the ice; the loads in the water budget, outside it
/// and the layers on the solid surface; the tide) and the seismic terms by
/// their index in `forcing.seismic`.
struct Jumping {
  bool ice = false;
  bool inside = false;
  bool outside = false;
  bool layer = false;
  bool tide = false;
  std::vector<std::size_t> seismic{};

  /// Adds what another jump at the same time jumps.
  void Join(const Jumping& other);
};

/// A jump of a run's forcing: its time on the terms' clock, the tolerance
/// a time is read on it with there (a time that close is that time), what
/// jumps, and the member of the config's `forcing` block it comes from
/// (`forcing.<list>[term].<member>`), which a refusal names.
struct ForcingJump {
  double at = 0.0;
  double tolerance = 0.0;
  Jumping what;
  std::string_view list;
  std::size_t term = 0;
  std::string member;
};

/// The members `solve` and `kernel` share, a run: the problem, the Earth,
/// the degree, the window, the forcing, the ocean, the rotation and the
/// solves.
struct Run {
  Problem problem = Problem::Static;
  ModelBlock model;
  int lmax = 0;
  /// The quasi-static problem's nodes: dates before present (a dated
  /// run), or times on the run's own clock (a run counting forwards).
  std::optional<Window> times;
  Forcing forcing;
  /// The state, its instant, the shorelines and the calibration, the
  /// `ocean` block's; absent, the body has no ocean (GRDSolver::
  /// NoOceanState), and no state, shorelines or calibration.
  std::optional<OceanBlock> ocean;
  RotationalFeedback rotation = RotationalFeedback::Full;
  /// Whether the run carries gravity, `gravity` ("full", the default, or
  /// "none"): without it the body has no ocean, no tide and no rotational
  /// feedback, and is forced on a radial model by surface loads, by their
  /// weight, and seismic sources (ViscoelasticOperator::Options::
  /// gravitating), the loads' degree-1 part removed (a net force).
  bool gravitating = true;
  /// With "hydrostatic", the moments of inertia difference is set from
  /// the model's own hydrostatic fluid limit (WithHydrostaticMoments),
  /// which the full transverse feedback needs when the input moments
  /// sustain less oblateness than the model's tidal response.
  bool hydrostatic_moments = false;
  SolverBlock solver;
  /// The jumps of the terms read from files on a time axis (a field's or
  /// a fault file's rule `previous`, an `interval` field), found once the
  /// files are read (FindFileJumps); the onsets' and the `previous`
  /// histories' are the forcing's own (JumpsOf).
  std::vector<ForcingJump> fileJumps;

  /// Whether the shorelines migrate (never without an ocean).
  [[nodiscard]] bool Migrating() const {
    return ocean && ocean->shorelines == Shorelines::Migrating;
  }
  /// Whether the initial sea level is calibrated (never without an ocean).
  [[nodiscard]] bool Calibrating() const { return ocean && ocean->calibrate; }
};

/// The top-level members `ReadRun` reads, for a command's list of its
/// members.
[[nodiscard]] std::vector<std::string_view> RunMemberNames();

/// Refuses every top-level member not among `allowed`: one a block of the
/// layout holds instead (`state`, `coupled`, `shorelines`, ...) is named
/// with the block it belongs in.
void AllowTopLevel(const Json& top, std::span<const std::string_view> allowed);

/// Reads the shared members, relative paths against `directory`. The
/// members only the quasi-static problem takes (a viscosity file, the
/// initial sea level's calibration, the time step, a term's history or
/// onset) are refused in the static one, naming the member, and so are the
/// 3-D step's unless a moduli file is given (the static problem on lateral
/// moduli takes them; on a radial file the runner refuses them). The
/// `times` block's form (ReadWindow) sets the form of the terms' times
/// (ReadForcing) and of the state's instant (ReadOceanBlock). In
/// the static problem a `thickness` term of the ice change without the
/// state's date is refused, and with the state at the window's end a
/// `change` term or a cap. The `ocean` block is optional: without it the
/// body has no ocean, and an ice change is refused. A forcing beside the
/// ice (a load, a tide, a seismic source) is refused with a calibrated
/// initial sea level, whose calibration runs under the ice alone. Without
/// gravity (`gravity`:
/// "none") the `ocean` block, a tide, a rotational feedback other than
/// "none" (the default then) and hydrostatic moments are refused, each
/// naming its member; the surface loads' degree-1 part, a net force, is
/// removed by the evolution, and said when the loads are built (DriveOf,
/// StaticLoadsOf).
[[nodiscard]] Run ReadRun(const Json& top,
                          const std::filesystem::path& directory);

/// Refuses the `solver` members that set the 3-D step (`step`,
/// `step_rtol`, `step_max_iterations`, `transform_degrees`, `cache_mb`)
/// for a run on a radial model, `what` naming the run; with `transforms`
/// the grids' members (`transform_degrees`, `cache_mb`) pass, for the
/// lateral kernels of a radial run, which expand on the grids.
void RefuseThreeDSolver(const Json& top, std::string_view what,
                        bool transforms = false);

/// Whether a config asks for lateral kernels (`kernels.lateral` true):
/// the grids' members of the `solver` block (`transform_degrees`,
/// `cache_mb`) then pass on a radial model, for the kernel sweep's grids,
/// and `solve` reads them and leaves them unused, as it does the
/// `kernels` block.
[[nodiscard]] bool LateralKernelsAsked(const Json& top);

/// The Earth of a run: the deck's model and material at lmax, and the
/// viscosity and moduli files' contents where given.
struct BuiltEarth {
  EarthModel model;
  Material material;
  std::optional<io::ModelViscosity> viscosity;
  std::optional<io::ModelModuli> moduli;
  /// The Kelvin elements of `model.burgers` by layer (KelvinOf); empty
  /// for the deck's own rheology.
  std::map<std::size_t, Rheology::Kelvin> kelvin;

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

/// The Kelvin elements of a run's `model.burgers` on its deck's model, by
/// layer: each region's layers, the Kelvin viscosity in the model's units.
/// A region is refused at its `depth_km` where an end does not lie on a
/// boundary of the deck's layers (to 1e-9 of the surface radius), naming
/// the layer it cuts and its depths, or lies beyond the deck, and where it
/// covers a layer that is fluid or holds no viscosity, naming it.
[[nodiscard]] std::map<std::size_t, Rheology::Kelvin> KelvinOf(
    const Run& run, const EarthModel& model, const Json& top);

/// Builds the run's Earth: the deck's model and material at lmax, the
/// Kelvin elements of `model.burgers` (KelvinOf), and the model files of
/// `model.viscosity` and `model.moduli` read about their chosen
/// references.
[[nodiscard]] BuiltEarth BuildEarth(const Run& run, const Json& top);

/// The elastic moduli of a run's Earth at a point (radius in metres), SI:
/// the deck's (LoveModuliAt) and, where the moduli file varies laterally,
/// its relative perturbations there of the shear and bulk moduli, s and b,
/// taken against the deck's Voigt-averaged mu and kappa as the 3-D
/// operator takes them (delta mu = s mu, delta kappa = b kappa, an
/// isotropic change of the deck's tensor).
[[nodiscard]] std::function<LoveModuli(const BodyPoint&)> ModuliOf(
    const BuiltEarth& earth);

/// Places the run's fault files (PlaceFaultFiles) in its Earth: the deck's
/// surface radius, the moduli at each point (ModuliOf), and the run's
/// nodes on the terms' clock. Nothing to do without one.
void PlaceSeismicSources(Run& run, const BuiltEarth& earth, const Json& top);

/// Finds the jumps of a run through time's terms read from files on a
/// time axis (Run::fileJumps): a `thickness` or `change` term of the ice
/// change read by date and a `field` load term whose field is read by
/// `previous` or is of `interval` support (io::FieldHistory::JumpsIn, the
/// files read without resampling), and a placed fault file read by
/// `previous` (FaultFileJumps; after PlaceSeismicSources). A file that
/// cannot be read, or lacks its variable, has none here: the run's own
/// reading of it refuses it, saying what it holds. Nothing in the static
/// problem.
void FindFileJumps(Run& run);

/// The rheology of a run through time: the viscosity file's radial
/// reference where given (a Maxwell body); otherwise the deck's own
/// (Rheology::FromModel: a Maxwell deck's viscosity, a burgers deck's
/// Maxwell and Burgers layers, a prony deck's branches), with the
/// Burgers layers of `model.burgers` (BuiltEarth::kelvin).
[[nodiscard]] Rheology RheologyOf(const BuiltEarth& earth, const Json& top);

/// The options of a 3-D run's operator: the transforms, cache and rank
/// cuts (OperatorOptions), the step solver, and the lateral moduli where
/// the model has them.
[[nodiscard]] Viscoelastic3D::Options OperatorOptionsOf(
    const Run& run, const BuiltEarth& earth, const Rheology& rheology);

/// The lateral perturbation delta of a 3-D run's viscosity: the viscosity
/// file's, or zero under the deck's column.
[[nodiscard]] std::function<double(double, double, double)> DeltaOf(
    const BuiltEarth& earth);

/// The SLE parameters, the moments set hydrostatically where asked.
[[nodiscard]] SeaLevelParameters ParametersOf(const Run& run,
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
/// asked (a held file, a run counting forwards' `change`, is one field),
/// and a cap is held through the run or scaled by its history's factor at
/// the time asked. Times asked are SI seconds on the run's axis
/// (Window::Axis: the calendar's in a dated run and the static problem,
/// the run's own clock in one counting forwards); every file is read
/// within its field's time span unless its input holds the span's ends
/// (IceThicknessAt). Every use of the ice goes through it: the state a run
/// starts from, the frozen load, the migrating thickness, the
/// calibration's start and the static problem's change.
class RunIce {
 public:
  /// Reads the state's ice at `stateTime` (SI seconds of the state's
  /// files' axes) and the terms' files through `files`, a term's file held
  /// to its kind (a `thickness` file without a time axis is refused, in
  /// the static problem a `change` file is held to its date, CheckDate,
  /// and in a run counting forwards a `change` file with a time axis is
  /// refused). Values that are not numbers are refused at the member
  /// reading them. `top`, the config, must outlive the ice.
  RunIce(const Run& run, const Json& top, InputFiles& files,
         double stateTime);

  /// The state's ice thickness, SI.
  [[nodiscard]] const std::vector<double>& State() const { return state_; }
  /// Whether there is a change at all (any term).
  [[nodiscard]] bool Changes() const { return !terms_.empty(); }
  /// The ice change at SI time `time`: the terms' sum, the first taken as
  /// it stands (zero without terms). With `before`, its left limit there:
  /// each file term's (ReadValues) and each cap's history's
  /// (AmplitudeHistory::Before), the right limit being the plain read.
  [[nodiscard]] std::vector<double> ChangeAt(double time,
                                             bool before = false) const;
  /// The change from SI time `since` to `time`: each file term's value at
  /// `time` less its value at `since`, and a cap's change at the one less
  /// at the other (nothing without a history). It is ChangeAt(time) -
  /// ChangeAt(since), formed term by term, without passing through the
  /// state's instant; `before` reads `time`'s left limit.
  [[nodiscard]] std::vector<double> ChangeSince(double time, double since,
                                                bool before = false) const;
  /// The ice at SI time `time`: the state's plus ChangeAt(time).
  [[nodiscard]] std::vector<double> At(double time) const;
  /// The ice a change may take beyond what the state holds: 1e-9 times
  /// the state's largest thickness, or 1e-9 m without ice.
  [[nodiscard]] double Tolerance() const;
  /// Refuses, at `forcing.ice_change`, a change that removes more ice than
  /// the state holds at any of the SI times given (beyond Tolerance), and
  /// at the left limits of `leftLimits`, with the worst deficit, its place
  /// and, in a run through time (`window`), its time as the window labels
  /// it. Where the state's own thickness is negative (a resampling's
  /// undershoot) it holds none.
  void CheckHeld(std::span<const double> times, const Window* window,
                 std::span<const double> leftLimits = {}) const;

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
  /// A file term's values at a time, or their left limit there, read
  /// within its span (`where` the read, IceThicknessAt), those that are
  /// not numbers refused.
  [[nodiscard]] std::vector<double> ValuesOf(const Term& term, double time,
                                             std::string_view where,
                                             bool before = false) const;
  /// A term's change at a time, or its left limit there.
  [[nodiscard]] std::vector<double> TermAt(const Term& term, double time,
                                           bool before = false) const;

  const SurfaceGrid* grid_;
  const Json* block_ = nullptr;  ///< `forcing.ice_change`, where given
  std::vector<double> state_;
  std::vector<Term> terms_;
};

/// The ice change's terms as described (Describe), joined by
/// `separator`; empty without terms.
[[nodiscard]] std::string DescribeIceChange(const Run& run,
                                            std::string_view separator);

/// The world a run moves: the sea level's file on the solver grid, the
/// run's ice, the ice the run starts from (SI), the background built
/// from that ice and the state's sea level, and the state's inputs as
/// read. Without an ocean the background is GRDSolver::NoOceanState and
/// the rest is empty.
struct Loaded {
  std::shared_ptr<const io::FieldHistory> seaLevel;
  std::shared_ptr<const RunIce> ice;
  /// The ice at the window's start, SI: the state's, or, with the state
  /// at the window's end, the state's plus the change back to the start.
  std::vector<double> ice0;
  GRDSolver::State background;
  std::string iceRead;       ///< the state's ice as read, for summaries and files
  std::string seaLevelRead;  ///< the state's sea level as read, likewise
  /// With the state at the window's start, the largest change of the ice
  /// there from the state's own, m (zero without one, and with the state
  /// at the window's end, whose run starts from the ice at the start).
  double changeAtStart = 0.0;
  /// Under migrating shorelines in a dated run with a land field on a time
  /// axis (`ocean.land`), the land held at each node's solver time: the
  /// field read at the node's date and cut at one half, every node's read
  /// with the world (the background holds the first's). Empty otherwise,
  /// the background's land held at every node.
  std::function<std::vector<char>(double)> landHeld;

  /// Whether a change is present at the window's start: one larger than
  /// the ice's tolerance (RunIce::Tolerance); never without an ocean.
  [[nodiscard]] bool ChangedAtStart() const {
    return ice && changeAtStart > ice->Tolerance();
  }
};

/// The surface held as land whatever the flotation, `ocean.land`: its
/// field read through `files` at `time` (SI, held to the field's time
/// span, `where` the read, ReadValues; a static field is the same at every
/// date) and cut at one half, with the input as read, `slice` saying when
/// for a field on a time axis (`held at every date` for a static one).
/// Both empty without the member: flotation alone.
struct LandMask {
  std::vector<char> held;
  std::string read;
};
[[nodiscard]] LandMask ReadLand(const Run& run, const Json& top,
                                InputFiles& files, double time,
                                std::string_view where,
                                std::string_view slice);

/// Sets an evolution's land held at each node (Options::land_held) to the
/// world's, where the land held changes in time (Loaded::landHeld); leaves
/// it absent otherwise.
template <class EvolutionOptions>
void HoldLand(EvolutionOptions& options, const Loaded& world) {
  if (world.landHeld) options.land_held = world.landHeld;
}

/// The `ocean` member holding the state's sea level input,
/// `ocean.sea_level` or `ocean.topography`, the one holding its ice,
/// `ocean.ice`, and the ice change, `forcing.ice_change`.
[[nodiscard]] const Json& SeaLevelMember(const Run& run,
                                         const Json& top);
[[nodiscard]] const Json& IceMember(const Json& top);
[[nodiscard]] const Json& IceChangeMember(const Json& top);

/// The rule of the state's date in the static problem and in a run
/// counting forwards, once the state's files are read through `files`:
/// `ocean.date_kyr_bp` is required when any of them (the sea level, the
/// ice, the land) has a time axis, and refused when none has one.
void CheckStateDate(const Run& run, const Json& top,
                    InputFiles& files);

/// The sea level the state's sea level input gives at a time (SI seconds
/// of the file's axis, the state's instant, StateReadOf), SI, read within
/// its field's time span unless the input holds the span's ends
/// (ReadValues): its variable as it stands, or, for `ocean.topography`,
/// the sea level by flotation from it with the given ice thickness (SI),
/// the state's. Values that are not numbers are refused at the input's
/// member.
[[nodiscard]] std::vector<double> SeaLevelFrom(
    const Run& run, const Json& top, const io::FieldHistory& file,
    std::span<const double> ice, double time,
    const SeaLevelParameters& parameters);

/// An ice field's input at a time (SI seconds of its file's axis), SI,
/// read within its field's time span (ReadValues, `where` the read; with
/// `before`, its left limit there): a thickness or a change of it. Values
/// that are not numbers are refused, at the input's `file` as a read
/// outside the span is.
[[nodiscard]] std::vector<double> IceThicknessAt(const FileInput& input,
                                                 const io::FieldHistory& file,
                                                 const Json& member,
                                                 double time,
                                                 std::string_view where,
                                                 bool before = false);

/// Where the state's members are read, as a refusal of a read outside a
/// file's span states it: the state's date in the static problem and a
/// run counting forwards, the window's end `ocean.at` names in a dated
/// run.
[[nodiscard]] std::string StateReadOf(const Run& run);

/// Reads the run through time's world: the state at the end of the window
/// `ocean.at` names in a dated run, or at its date, `ocean.date_kyr_bp`,
/// in a run counting forwards (CheckStateDate), the state then at the
/// window's start, each file read within its field's time span unless its
/// input holds the span's ends (the sea level's first); the run's ice,
/// held at every node to what the state holds (RunIce::CheckHeld), with
/// the change present at the window's start; the land held (`ocean.land`):
/// a field on a time axis under migrating shorelines in a dated run at
/// every node's date (Loaded::landHeld, the background the first's), under
/// frozen ones at the state's instant, in a run counting forwards at the
/// state's date, and a static field once; and the background built from
/// the ice at the window's start, the state's sea level and that land.
/// Without an ocean, the world of a body without one (Loaded), nothing
/// read.
[[nodiscard]] Loaded LoadWorld(const Run& run, const Json& top,
                               const GRDSolver& solver);

/// The state's sea level at the window's end (a topography converted
/// with the state's ice), solver units: the calibration's target.
[[nodiscard]] std::vector<double> FinalSeaLevelOf(const Run& run,
                                                  const Json& top,
                                                  const GRDSolver& solver,
                                                  const Loaded& world);

/// A solver time `t` (SLE units, counting forwards from the window's
/// start) as a summary or an error states it: its date, `<d> kyr BP`, in
/// a dated run, its time on the run's clock, `<t> <unit>`, in one counting
/// forwards (Window::Label).
[[nodiscard]] std::string WhenOf(const Run& run, const Scales& scales,
                                 double t);

/// The window's time nodes, SLE units, zero at its start (Window::Times).
[[nodiscard]] std::vector<double> MakeTimes(const Run& run,
                                            const Scales& scales);

/// The window's nodes on the terms' clock (SolverClock): kyr from the
/// window's start in a dated run, the nodes as given in one counting
/// forwards. Where a jump of the forcing falls against the nodes is
/// decided on them (JumpsBetweenNodes, StepGridOf).
[[nodiscard]] std::vector<double> NodesOnTheClock(const Window& window);

/// The jumps of a run's forcing after its window's start and not after
/// its end, in time order: each seismic onset (read with SeismicTerm::
/// OnsetAt's tolerance), each time after the first of a history read by
/// `previous` (AmplitudeHistory::Tolerance), on any term that takes a
/// history, and the jumps of its files (Run::fileJumps). A jump at the
/// window's start (to its tolerance) or before it is none: the first node
/// is the right limit there. Empty in the static problem.
[[nodiscard]] std::vector<ForcingJump> JumpsOf(const Run& run);

/// A jump of a run's forcing strictly between two nodes of its window:
/// the jump (the first at its time) and the window's node before it.
struct JumpBetween {
  ForcingJump jump;
  std::size_t node = 0;
};

/// The jumps of a run's forcing (JumpsOf) strictly between two nodes of
/// its window, in time order, each time once: a jump within its tolerance
/// of a node is that node's, and jumps within the tolerance of an earlier
/// one share its time.
[[nodiscard]] std::vector<JumpBetween> JumpsBetweenNodes(const Run& run);

/// The time grid the library steps a run over (Evolution::Run): the
/// window's nodes, with each jump of the forcing (JumpsOf: a seismic onset,
/// a time of a `previous` history, a jump of a file read by `previous` or
/// of `interval` support) held as a time repeated once, its left limit then
/// its right. The left limit reads each term's left limit there
/// (Evolution::Forcing::sourcesBefore, directLoadBefore and their kin;
/// ForcingOf), the right its value, and the zero-length step between is
/// the static elastic solve with the internal variables held. A jump at a
/// node repeats that node; one between two (JumpsBetweenNodes) is a pair
/// of the library's own, inserted between them at the jump, where every
/// other term is read as at any time (by its history, its file's time
/// rule); jumps at one time share one pair. The run's outputs keep one
/// record per node of the window, the right limit at a repeat, and
/// nothing of an inserted pair; the library's trajectory keeps every node.
struct StepGrid {
  /// The window node of an inserted pair's nodes, which is none.
  static constexpr auto kInserted = static_cast<std::size_t>(-1);
  /// The library's nodes, SLE units, zero at the window's start.
  std::vector<double> times;
  /// Per library node, the window's node it is (both of a repeat's), or
  /// kInserted.
  std::vector<std::size_t> window;
  /// Per window node, its library node: the right limit at a repeat.
  std::vector<std::size_t> library;
  /// Whether library node k is the left limit of a pair, a node of the
  /// window's repeated or an inserted one.
  [[nodiscard]] bool Left(std::size_t k) const {
    return k + 1 < times.size() && times[k + 1] == times[k];
  }
  /// Whether library node k is of an inserted pair, no node of the
  /// window's.
  [[nodiscard]] bool Inserted(std::size_t k) const {
    return window[k] == kInserted;
  }
  /// What jumps at each pair, by the library node of its left limit.
  std::map<std::size_t, Jumping> jumps{};
  /// Whether the library steps any node beside the window's one each.
  [[nodiscard]] bool Repeats() const { return times.size() > library.size(); }
  /// What jumps at the pair whose right limit is library node k; nothing
  /// where k is not one.
  [[nodiscard]] Jumping JumpingAt(std::size_t k) const {
    const auto found = k > 0 ? jumps.find(k - 1) : jumps.end();
    return found == jumps.end() ? Jumping{} : found->second;
  }
};
[[nodiscard]] StepGrid StepGridOf(const Run& run, const Scales& scales);

/// The terms' clock (TermClock) against the solver's time: the time on it
/// at solver time t (SLE units, zero at the window's start) is
/// `zero + t * rate`, kyr from the window's start in a dated run and in
/// the static problem (whose one node is at zero), the run's own clock in
/// its unit in one counting forwards.
struct SolverClock {
  double zero = 0.0;
  double rate = 0.0;
  [[nodiscard]] double At(double t) const { return zero + t * rate; }
};
[[nodiscard]] SolverClock SolverClockOf(const Run& run,
                                        const Scales& scales);

/// The grounded ice load of the change from the ice the run starts from
/// (world.ice0), as the drivers take it: a callable of SLE time over the
/// solver grid. With the state at the window's start the change is the
/// ice change itself, so a change present at the start loads the first
/// node; with the state at its end it is the change since the start. With
/// `before`, its left limit at each time (RunIce::ChangeAt), the left
/// limits of a jump of the ice. The references must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceLoadOf(
    const Run& run, const GRDSolver& solver, const Loaded& world,
    bool before = false);

/// The ice thickness the migrating runs take, solver units: the
/// background's ice plus the same change as IceLoadOf's, so that a change
/// present at the window's start is ice the start's solve takes; with
/// `before`, its left limits. The references must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceThicknessOf(
    const Run& run, const GRDSolver& solver, const Loaded& world,
    bool before = false);

/// The run's ice history under its shorelines: IceLoadOf when frozen,
/// IceThicknessOf when migrating, with its left limits where the run's
/// forcing jumps (StepGridOf repeats a node).
[[nodiscard]] IceHistory HistoryOf(const Run& run,
                                   const GRDSolver& solver,
                                   const Loaded& world);

/// Rebuilds the world's start from the run's ice: its ice at the
/// window's start (world.ice0 too) under the given sea level, the ocean
/// function by flotation.
void StartFromIceHistory(const Run& run, const GRDSolver& solver,
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
CalibrationSummary CalibrateWorld(const Run& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution& evolution,
                                  std::span<const double> times);
CalibrationSummary CalibrateWorld(const Run& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution3D& evolution,
                                  std::span<const double> times);

/// The forcing of a run in the form the library's evolutions take
/// (Evolution::Forcing or Evolution3D::Forcing), in the solver's units,
/// each channel a function of the solver's time read on the terms' clock
/// (SolverClockOf: kyr from the window's start in a dated run, the run's
/// own clock in its unit in one counting forwards; the static problem's
/// one node is at zero, where its terms are what they are): the ice
/// change's grounded load, `iceLoad` (given under frozen shorelines alone:
/// under migrating ones the ice is given as its thickness, and a body
/// without an ocean has none), joined by the surface loads in the water
/// budget; the loads outside the water budget, the layer on the solid
/// surface, the tide and the seismic sources. A channel no term gives is
/// absent. Where the run's forcing jumps (StepGridOf repeats a node) every
/// channel given has its left limit beside it (`iceLoadBefore` the ice
/// load's; the others' from their terms, TimedLoads::At, TideAt,
/// SeismicSourcesAt), and under migrating shorelines the ice thickness's
/// (`thicknessBefore`). `loads` are the run's surface loads (null without
/// any), held by the forcing.
template <class EvolutionForcing>
[[nodiscard]] EvolutionForcing ForcingOf(
    const Run& run, const GRDSolver& solver,
    const std::function<std::vector<double>(double)>& iceLoad,
    std::shared_ptr<const SurfaceLoads> loads,
    const std::function<std::vector<double>(double)>& iceLoadBefore = {},
    const std::function<std::vector<double>(double)>& thicknessBefore = {});

/// What drives a run through time on its world: the ice's grounded load
/// under frozen shorelines (IceLoadOf; none without an ocean), its
/// thickness under migrating ones (IceThicknessOf), whether the run is
/// forced beside them, and that forcing (ForcingOf) with the surface loads
/// it holds (null without any). A run is forced by its loads, tide and
/// seismic sources, and, under migrating shorelines, by an ice change
/// present at the window's start, taken through the forcing's entries
/// (with nothing beside the ice, an empty forcing) so that the start is
/// solved; and a run whose forcing jumps (StepGridOf repeats a node) is
/// forced, its left limits in the forcing (the ice's among them, which the
/// overloads taking the ice alone do not take). The world must outlive
/// the drive.
template <class EvolutionForcing>
struct Drive {
  std::function<std::vector<double>(double)> directLoad;
  std::function<std::vector<double>(double)> thickness;
  bool forced = false;
  EvolutionForcing forcing;
  std::shared_ptr<const SurfaceLoads> loads;
};

/// The drive of a run through time on its world (Drive), the surface
/// loads read onto the solver's grid against the config's `forcing`.
/// Without gravity, the loads' degree-1 part, a net force, which nothing
/// balances and the evolution removes, is said in one line on standard
/// error with the node where it is largest, where it is above rounding
/// (DegreeOneShare above 1e-12).
template <class EvolutionForcing>
[[nodiscard]] Drive<EvolutionForcing> DriveOf(const Run& run, const Json& top,
                                              const GRDSolver& solver,
                                              const Loaded& world);

/// The run of an evolution through the times under a drive: the frozen
/// run under the ice's load or the forcing, or the migrating run under
/// the ice's thickness and, where forced, the forcing; and the same from a
/// checkpoint, `remaining` the window's times from the checkpoint's node.
template <class EvolutionT>
[[nodiscard]] std::vector<typename EvolutionT::Snapshot> RunDriven(
    EvolutionT& evolution, std::span<const double> times,
    const Drive<typename EvolutionT::Forcing>& drive, bool migrating);
template <class EvolutionT>
[[nodiscard]] std::vector<typename EvolutionT::Snapshot> ResumeDriven(
    EvolutionT& evolution, const typename EvolutionT::Checkpoint& from,
    std::span<const double> remaining,
    const Drive<typename EvolutionT::Forcing>& drive, bool migrating);

/// A 1-D run's refusal of an unconverged node, `when` the node as the
/// window labels it: the sea level solve's passes, its last relative
/// change and the tolerance asked of it (`rtol`, solver.rtol).
void Check1D(const Evolution::Snapshot& snapshot, std::string_view when,
             double rtol);
/// A 3-D run's refusal of an unconverged node: the step's combined fixed
/// point's last residual, its passes and the tolerance asked of it
/// (`stepRtol`, solver.step_rtol), or its sea level solve's failure.
void Check3D(const Evolution3D::Snapshot& snapshot, std::string_view when,
             double stepRtol);

/// The static problem's world on a run's grid: the state at its date
/// (`ocean.ice` and `ocean.sea_level` or `ocean.topography`, each file
/// with a time axis read at `ocean.date_kyr_bp` within its field's time
/// span unless its input holds the span's ends, and `ocean.land`), the
/// background state built from it in the solver's units, its ocean
/// function flotation's, the ice change at the date (RunIce, held to what
/// the state holds), SI, and the state's inputs as read. Without an ocean
/// the background is GRDSolver::NoOceanState and the rest is empty.
struct StaticWorld {
  GRDSolver::State background;
  std::vector<double> change;
  std::string iceRead;
  std::string seaLevelRead;
};
[[nodiscard]] StaticWorld LoadStaticWorld(const Run& run, const Json& top,
                                          const GRDSolver& solver);

/// The static problem's surface loads on a grid (SurfaceLoads, each term
/// read once, a `field` file at its date), or null without any. Without
/// gravity, the loads' degree-1 part is said as DriveOf says it.
[[nodiscard]] std::shared_ptr<const SurfaceLoads> StaticLoadsOf(
    const Run& run, const Json& top, const SurfaceGrid& grid);

/// The static problem solved as a run of one node of either evolution,
/// built by the caller on the world's background: the solve of the start
/// alone on the unrelaxed body, its one snapshot. Under frozen shorelines
/// the linear sea level equation under the forcing, the ice change
/// entering as its grounded load on the state's ocean function, rho_ice (1
/// - C) dI, in the water budget; under migrating shorelines the nonlinear
/// one, the ice the state's plus the change, whose flotation, with the sea
/// level's, places the shorelines (the tide's potential among what moves
/// the sea level). The surface loads, the tide and the seismic sources
/// join either (ForcingOf). Without an ocean, the linear
/// equation on a body without one under the forcing beside the ice alone:
/// the solid body's response.
template <class EvolutionT>
[[nodiscard]] std::vector<typename EvolutionT::Snapshot> SolveStatic(
    const Run& run, const GRDSolver& solver, const StaticWorld& world,
    std::shared_ptr<const SurfaceLoads> loads, EvolutionT& evolution);

/// The elastic rheology of the static problem: the deck's elastic
/// structure with no relaxing branch.
[[nodiscard]] Rheology ElasticRheologyOf(const BuiltEarth& earth,
                                         const Json& top);

/// What the config was taken to mean, for a summary: the model's
/// dimension, elastic and static (the static problem) or viscoelastic
/// with the window (its dates, or its times on the run's clock in its
/// unit), the shorelines and, through time, the calibration (or, in
/// their place, that the body has no ocean), that the run has no gravity
/// where it has none, and the rotation and moments. The forcing follows
/// in the summary (WithInputs).
[[nodiscard]] std::string Meaning(const Run& run, const BuiltEarth& earth);

/// What the config was taken to mean, followed by the state's inputs as
/// read (none without an ocean), the ice change's terms, the forcing
/// beside the ice, where there are any, and each `field` load's input as
/// read (`inputs`, DescribeField).
[[nodiscard]] std::string WithInputs(std::string_view meaning, const Run& run,
                                     std::string_view iceRead,
                                     std::string_view seaLevelRead,
                                     std::span<const std::string> inputs);

/// Each `field` load's input as read, of a run's loads (none: empty).
[[nodiscard]] std::vector<std::string> InputsOf(const SurfaceLoads* loads);

}  // namespace sl3dpp::app
