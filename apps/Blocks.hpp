#pragma once

/**
 * @file Blocks.hpp
 * @brief The config blocks of the commands made of blocks (`solve`,
 * `kernel`), each read by one reader here:
 * `problem` (static, or quasi-static through time), `discretisation`
 * (the truncation degree, which `love-numbers` reads too), `model` (the deck,
 * its format, the lateral moduli and the 3D viscosity), `times` (the time
 * nodes of a run stepped through time), `ocean` (the state, the sea level
 * and the ice on the body at one instant, the shorelines and the initial
 * sea level's calibration), `solver` (the
 * solves' tolerances, budget and transforms, and the time step),
 * `checkpoint` (a run's checkpoints) and `output` (the files written),
 * and the words of `rotation` and `moments`. The `forcing` blocks have
 * their own file, Forcing.hpp, and the file inputs theirs, Files.hpp.
 *
 * Each reader refuses members it does not know, as every config reader
 * does (Input.hpp), and resolves relative paths against the config's
 * directory.
 */

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Files.hpp"
#include "Input.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::app {

/// The problem a run solves (the top-level `problem`): the static one, at
/// one instant, or the quasi-static one, stepped through the nodes of a
/// `times` block. The rheology is the `model` block's business.
enum class Problem { Static, QuasiStatic };

/// Reads the required top-level `problem`, "static" or "quasi_static", and
/// holds the `times` block to it: the static problem refuses one, the
/// quasi-static problem requires one.
[[nodiscard]] Problem ReadProblem(const Json& top);

/// The `discretisation` block: the degree the run's spherical harmonic
/// expansions are truncated at, which sets its grid and its radial mesh.
struct DiscretisationBlock {
  int lmax = 0;
};

/// Reads the required top-level `discretisation` block, of `lmax` alone
/// (required, an integer of at least `least`). A top-level `lmax` is
/// refused, naming the block it belongs in.
[[nodiscard]] DiscretisationBlock ReadDiscretisationBlock(const Json& top,
                                                          int least);

/// The deck formats a config may name: planetmodel's GIA and MINEOS
/// tables, and the library's six-column elastic table (rho, vpv, vsv,
/// vph, vsh, eta; no header), the test and example decks' format.
/// Gia is planetmodel's Maxwell deck (`"maxwell"` in a config; `"gia"`, the
/// format's other name, is accepted too).
enum class ModelFormat { Gia, Mineos, Elastic };

/// The format a `ModelFormat` stands for.
[[nodiscard]] DeckFormat FormatOf(ModelFormat format);

/// The `model` block: the deck, its format and, where given, a model
/// file of lateral moduli and a model file of 3D viscosity.
struct ModelBlock {
  std::filesystem::path deck;
  ModelFormat format = ModelFormat::Gia;
  /// A planetmodel model file whose `mu` and `kappa` fields supply the
  /// lateral perturbations of the moduli (io/ModelModuli); empty for the
  /// radial model alone.
  std::filesystem::path moduli;
  bool moduli_file_radial = false;  ///< the file's own radial reference
  /// A planetmodel model file whose `viscosity` field supplies the
  /// viscosity eta(x) = eta_bar(r) exp(delta(x)), the radial Maxwell body
  /// and its lateral perturbation alike (io/ModelViscosity); empty for the
  /// deck's own viscosity column.
  std::filesystem::path viscosity;
  bool viscosity_file_radial = false;  ///< the file's own radial reference
};

/// Reads a `model` block, relative paths against `directory`: `deck`
/// (required), `format` (a word, default "maxwell"), `moduli` (an object
/// of `file` and `reference`) and `viscosity` (an object of `file` and
/// `reference`).
[[nodiscard]] ModelBlock ReadModelBlock(const Json& block,
                                        const std::filesystem::path& directory);

/// The two forms of the `times` block, chosen by the members present.
enum class Calendar {
  /// Dates in kyr before present, on the calendar the files with a time
  /// axis share: `start_kyr_bp`, `end_kyr_bp` and `nodes`, or `at_kyr_bp`.
  Dated,
  /// Times counted forwards on the run's own clock in the block's `unit`:
  /// `start`, `end` and `nodes`, or `at`. A run with no calendar.
  Forwards
};

/// The `times` block: the nodes of a run through time, the window from the
/// first to the last.
struct Window {
  Calendar calendar = Calendar::Dated;
  /// The unit the nodes are given in, and the terms' times beside them:
  /// kyr for dates; the block's `unit` on a forwards clock.
  TimeUnit unit;
  /// Whether the nodes were listed (`at_kyr_bp`, `at`) rather than evenly
  /// spaced.
  bool listed = false;
  /// The nodes as the config gives them, in `unit`: dates before present,
  /// strictly decreasing (dated), or times on the run's clock, strictly
  /// increasing (forwards). Evenly spaced, the start less (dated) or plus
  /// (forwards) n times the one spacing, the last the end itself.
  std::vector<double> nodes;

  [[nodiscard]] bool Dated() const { return calendar == Calendar::Dated; }
  [[nodiscard]] std::size_t Size() const { return nodes.size(); }
  /// The first node and the last, in `unit` (dated, kyr BP).
  [[nodiscard]] double Start() const { return nodes.front(); }
  [[nodiscard]] double End() const { return nodes.back(); }
  /// Node `n`'s place on the run's axis, SI seconds: dated, minus its date
  /// (the calendar's axis, the present at zero and the past negative, on
  /// which files with a time axis are read); forwards, its time on the
  /// run's clock.
  [[nodiscard]] double Axis(std::size_t n) const;
  /// Each node's time since the first, in units of `scale` seconds (the
  /// solver's time unit). Dated: the window's span in seconds times n over
  /// (nodes - 1) for evenly spaced nodes, the difference of their places
  /// on the axis for listed ones, each over `scale`. Forwards: the node
  /// less the first, in the run's unit, times the unit over `scale`. A
  /// step's response moves with the last bit of its length (by up to about
  /// 1e-12 relative under a one-ulp change of a node), so runs whose times
  /// are formed otherwise (another unit, dates for times) agree to that.
  [[nodiscard]] std::vector<double> Times(double scale) const;
  /// A time of the run's axis (SI seconds) as a summary or an error states
  /// it: `<d> kyr BP` (dated) or `<t> <unit>` (forwards), to six figures.
  [[nodiscard]] std::string Label(double axis) const;
};

/// Reads the `times` block. Dated: `start_kyr_bp` (required),
/// `end_kyr_bp` (default 0, below the start) and `nodes` (at least two,
/// evenly spaced, both ends included), or `at_kyr_bp`, the dates
/// themselves, at least two, strictly decreasing. Forwards: `unit`
/// (required: "kyr", "yr" or "day"), `start` (default 0) with `end`
/// (later than the start) and `nodes`, or `at`, the times themselves, at
/// least two, strictly increasing. Exactly one way of giving the nodes in
/// either form; members of the two forms together are refused, naming
/// both, and so are times in kyr without a unit (`start_kyr`, `end_kyr`,
/// `at_kyr`), naming the forwards form's members with its `unit`.
[[nodiscard]] Window ReadWindow(const Json& block);

/// The end of a run's window an input belongs to.
enum class WindowEnd { Start, End };

/// The sea level of the sea level equation: a file input that holds the
/// sea level itself (`ocean.sea_level`, used as it stands) or the
/// topography (`ocean.topography`, the ICE-NG convention, converted with
/// the state's own ice, `ocean.ice`, by SeaLevelFromTopography).
struct SeaLevelSource {
  enum class Kind { SeaLevel, Topography };
  Kind kind = Kind::Topography;
  /// The file and its variable (by default `sea_level` or `topography`).
  FileInput input;

  /// The member's name, `sea_level` or `topography`.
  [[nodiscard]] std::string_view Member() const;
  /// What the input holds, for a summary: `sea level` or `topography`.
  [[nodiscard]] std::string_view Quantity() const;
};

/// The `ocean` block of a run with an ocean: the state, what is on the
/// body at one instant (the sea level and the ice, each a file input, and
/// the land held), the instant itself, the shorelines and, in the
/// quasi-static problem, the initial sea level's calibration. What happens
/// to the ice is the forcing's (`forcing.ice_change`, Forcing.hpp).
struct OceanBlock {
  SeaLevelSource seaLevel;
  /// `ocean.ice`: the ice thickness on the body at the state's instant
  /// (its variable by default `ice_thickness`).
  FileInput ice;
  /// `ocean.land`, where given: a field that is one where the surface is
  /// held as land whatever the flotation and zero elsewhere (landlocked
  /// depressions), static or on a time axis (a land that changes in time,
  /// read at every node's date under migrating shorelines in a dated run,
  /// LoadWorld). Absent, flotation alone.
  std::optional<FileInput> land;
  /// The state's date, kyr before present (`ocean.date_kyr_bp`), at which
  /// every member's file with a time axis is read, in the static problem
  /// and in a run counting forwards (which has no calendar of its own);
  /// absent when none has one. A dated run reads the state at the end of
  /// its window `at` names instead.
  std::optional<double> date_kyr_bp;
  /// The quasi-static problem: the end of the window the state belongs to
  /// (`ocean.at`), the start's being the initial state and the end's the
  /// state the run reaches, the target of the calibration (a dated run's
  /// alone).
  WindowEnd at = WindowEnd::Start;
  /// The shorelines: frozen at the state's (the linear equation), or
  /// following flotation (the nonlinear one), in either problem.
  Shorelines shorelines = Shorelines::Frozen;
  /// With the state at the window's end, the initial sea level is
  /// calibrated so that the run reaches the state's sea level there, by
  /// the method and within the iterations and tolerance given
  /// (InitialSeaLevel.hpp); with the state at the start, its sea level is
  /// the initial sea level itself.
  bool calibrate = false;
  CalibrationMethod calibration_method = CalibrationMethod::FixedPoint;
  int calibration_iterations = 10;
  double calibration_rtol = 1e-4;
};

/// Reads an `ocean` block of a run of `problem` (through time, on the
/// `calendar` of its times), relative paths against `directory`. It holds
/// exactly one of `sea_level` and `topography` and, required, `ice`, each
/// a file input (Files.hpp) of `file`, `format`, `variable` and
/// `outside_span` alone, and optionally `land`, another. The state's instant is the block's: in the static
/// problem `date_kyr_bp` (whether it is required is the files' business,
/// checked once they are read, CheckStateDate); in the quasi-static
/// problem `at` ("start" or "end", required), and in a run counting
/// forwards `date_kyr_bp` too, the date of the state's files, as the
/// static problem's. `"at": "end"` is refused in a run counting forwards:
/// it reads the ice back along dated ice histories, which such a run does
/// not have. Both problems take `shorelines` (a word, default "frozen");
/// the quasi-static problem also takes `calibration` (an object of
/// `method`, `iterations` and `rtol`, each optional), which `"at": "end"`
/// alone takes and turns on, and the static problem refuses. The static
/// problem refuses `at`, and a dated run `date_kyr_bp`. A date or an `at`
/// on a member is refused, naming the block's.
[[nodiscard]] OceanBlock ReadOceanBlock(const Json& block,
                                        const std::filesystem::path& directory,
                                        Problem problem,
                                        Calendar calendar = Calendar::Dated);

/// The `solver` block: the sea level solves' tolerance, the 3D closure's
/// tolerance, budget and transforms, and the time step.
struct SolverBlock {
  /// The relative tolerance of the sea level solves (and of the kernels'
  /// adjoint sea level solves).
  double rtol = 1e-9;
  /// The relative tolerance and the iteration budget of the 3D closure:
  /// each step's combined fixed point over the strain, the ocean load and
  /// the rotation (Evolution3D::Options::coupled_rtol,
  /// max_coupled_iterations; under lateral moduli the start's zero-length
  /// step too, and the static problem's one zero-length step).
  double step_rtol = 1e-9;
  int step_max_iterations = 60;
  /// The degrees the 3D coupling's transforms may be cut to with depth
  /// (Viscoelastic3D::Options::transform_degrees); empty, every node at
  /// lmax.
  std::vector<int> transform_degrees;
  /// The cache every transform grid of the run blocks its work for, in
  /// MiB; 0, the library's default.
  std::size_t cache_mb = 0;
  /// A viscoelastic run's time step, and how a 3D step's linear system is
  /// closed (Viscoelastic3D::Options::StepSolver).
  Scheme scheme = Scheme::ExponentialTrapezoid;
  Viscoelastic3D::Options::StepSolver step =
      Viscoelastic3D::Options::StepSolver::FixedPoint;
};

/// Reads a `solver` block of a run at `lmax`, each member optional:
/// `rtol`, `step_rtol`, `step_max_iterations`, `transform_degrees`,
/// `cache_mb`, `scheme` and `step`.
[[nodiscard]] SolverBlock ReadSolverBlock(const Json& block, int lmax);

/// The `checkpoint` block of a run through time: every `every`-th node's
/// recurrence and the history so far go to `directory`; a positive
/// `max_walltime_min` makes the run checkpoint and stop cleanly once the
/// wall clock passes it.
struct CheckpointBlock {
  std::filesystem::path directory;  ///< empty: no checkpoints
  int every = 0;
  double max_walltime_min = 0.0;
};

/// Reads a `checkpoint` block, relative paths against `directory`:
/// `directory` and `every` (required) and `max_walltime_min` (default 0).
[[nodiscard]] CheckpointBlock ReadCheckpointBlock(
    const Json& block, const std::filesystem::path& directory);

/// A field of the surface file written only where `output.fields` names
/// it, beside those always written.
enum class OptionalField {
  /// `horizontal_displacement`: the tangential displacement's components
  /// to the east and to the north, `displacement_east` and
  /// `displacement_north`.
  HorizontalDisplacement
};

/// The words of `output.fields`, as a refusal lists them.
inline constexpr auto kOptionalFields =
    std::string_view{"\"horizontal_displacement\""};

/// The smoothing of `output.smoothing`: its kind (`gaussian`, the one
/// built) and its width in degrees as given (`width_deg`, the Gaussian
/// cap's standard deviation in great-circle distance, as a cap
/// measurement's).
struct Smoothing {
  enum class Kind { Gaussian };
  Kind kind = Kind::Gaussian;
  double width_deg = 0.0;

  /// The width in radians, converted as a cap measurement's is.
  [[nodiscard]] double Width() const {
    return width_deg * std::numbers::pi / 180.0;
  }
};

/// The `output` block: the files a run writes, and what its surface file
/// holds beside the fields always written.
struct OutputBlock {
  std::filesystem::path surface;   ///< the surface file, required
  std::filesystem::path exchange;  ///< the coefficient table; empty for none
  std::filesystem::path volume;    ///< the model file; empty for none
  /// The node indices whose exchange tables a run through time writes;
  /// empty for every node.
  std::vector<int> exchange_nodes;
  /// The optional fields added to the surface file, in the config's order.
  std::vector<OptionalField> fields;
  /// The smoothing whose fields are added beside the raw ones; none
  /// without the member.
  std::optional<Smoothing> smoothing;

  /// Whether `fields` names the field.
  [[nodiscard]] bool Asks(OptionalField field) const {
    return std::ranges::find(fields, field) != fields.end();
  }
};

/// Reads an `output` block, relative paths against `directory`:
/// `surface` (required), `exchange`, `volume`, `exchange_nodes` (a list
/// of non-negative integers), `fields` (a list of the words of
/// kOptionalFields, none twice) and `smoothing` (an object of `kind`,
/// "gaussian", and `width_deg`, a positive angle, both required).
[[nodiscard]] OutputBlock ReadOutputBlock(
    const Json& block, const std::filesystem::path& directory);

/// The `rotation` word: "none", "axial" or "full".
[[nodiscard]] RotationalFeedback RotationOf(const Json& word);

/// The `gravity` word: "full" (true) or "none" (false), whether the run
/// carries gravity (ViscoelasticOperator::Options::gravitating).
[[nodiscard]] bool GravityOf(const Json& word);

/// The `moments` word: "input" (false) or "hydrostatic" (true), whether
/// the moments of inertia difference is set from the model's own
/// hydrostatic fluid limit (WithHydrostaticMoments).
[[nodiscard]] bool HydrostaticMomentsOf(const Json& word);

/// The words of a rotation and of the shorelines, as configs give them.
[[nodiscard]] std::string_view NameOf(RotationalFeedback rotation);
[[nodiscard]] std::string_view NameOf(Shorelines shorelines);

}  // namespace sl3dpp::app
