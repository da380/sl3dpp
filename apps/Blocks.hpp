#pragma once

/**
 * @file Blocks.hpp
 * @brief The config blocks of the commands made of blocks (`response`,
 * `sea-level`, `kernel1d`, `kernel`), each read by one reader here:
 * `problem` (static, or quasi-static through time), `model` (the deck,
 * its format, the lateral moduli and the 3D viscosity), `times` (the time
 * nodes of a run stepped through time), `ocean` (the sea level, the
 * shorelines and the initial sea level's calibration), `solver` (the
 * solves' tolerances, budget and transforms, and the time step),
 * `checkpoint` (a run's checkpoints) and `output` (the files written),
 * and the words of `rotation` and `moments`. The `forcing` blocks have
 * their own file, Forcing.hpp, and the file inputs theirs, Files.hpp.
 *
 * Each reader refuses members it does not know, as every config reader
 * does (Input.hpp), and resolves relative paths against the config's
 * directory.
 */

#include <cstddef>
#include <filesystem>
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

/// The two forms of a `times` block; each command takes one.
enum class TimesForm {
  /// kyr on the run's own clock, counted forwards: `start_kyr` (default
  /// 0) with `end_kyr` and `nodes`, or `at_kyr` (`response`).
  Forwards,
  /// Dates in kyr before present: `start_kyr_bp`, `end_kyr_bp` (default
  /// 0, later than the start, so smaller) and `nodes`, evenly spaced, both
  /// ends included (`sea-level`, `kernel1d`, `kernel`).
  BeforePresent
};

/// The `times` block: the time nodes of a run stepped through time.
struct TimesBlock {
  /// The forwards form: the nodes in kyr on the run's own clock, strictly
  /// increasing, the first the start.
  std::vector<double> nodes_kyr;
  /// The forwards form: the step from each node to the next, one fewer
  /// than the nodes: for evenly spaced nodes the one spacing
  /// (end - start) / (nodes - 1) every time, so that equal steps are equal
  /// to the bit; the differences of the nodes given otherwise.
  std::vector<double> steps_kyr;
  /// The before-present form: the window's start and end, kyr before
  /// present (the start the larger), and the number of nodes evenly
  /// spaced over it, both ends included.
  double start_kyr_bp = 0.0;
  double end_kyr_bp = 0.0;
  int nodes = 0;
};

/// Reads a `times` block of the given form. Forwards: `start_kyr`
/// (default 0) with `end_kyr` (later than the start) and `nodes` (at least
/// two, evenly spaced, both ends included); or instead an explicit list
/// `at_kyr`, strictly increasing, of at least two entries, its first the
/// start; exactly one of the two. Before present: `start_kyr_bp`
/// (required), `end_kyr_bp` (default 0, below the start) and `nodes` (at
/// least two). The members of the other form are refused, naming the form
/// this command takes.
[[nodiscard]] TimesBlock ReadTimesBlock(const Json& block, TimesForm form);

/// The end of a run's window an input belongs to.
enum class WindowEnd { Start, End };

/// The sea level of the sea level equation: a file input that holds the
/// sea level itself (`ocean.sea_level`, used as it stands) or the
/// topography (`ocean.topography`, the ICE-NG convention, converted with
/// the run's own ice at the same instant by SeaLevelFromTopography).
struct SeaLevelSource {
  enum class Kind { SeaLevel, Topography };
  Kind kind = Kind::Topography;
  /// The file, its variable (by default `sea_level` or `topography`) and,
  /// in the static problem, its date (`date_kyr_bp`, kyr before present),
  /// required of a file with a time axis.
  FileInput input;
  /// The quasi-static problem: the end of the window the sea level is
  /// read at (where its file has a time axis), the start's being the
  /// initial sea level and the end's the target of the calibration.
  WindowEnd at = WindowEnd::Start;

  /// The member's name, `sea_level` or `topography`.
  [[nodiscard]] std::string_view Member() const;
  /// What the input holds, for a summary: `sea level` or `topography`.
  [[nodiscard]] std::string_view Quantity() const;
};

/// The `ocean` block of a run with an ocean: the sea level and, in the
/// quasi-static problem, the shorelines and the initial sea level's
/// calibration.
struct OceanBlock {
  SeaLevelSource seaLevel;
  /// The shorelines: frozen at the start's (the linear equation), or
  /// following flotation.
  Shorelines shorelines = Shorelines::Frozen;
  /// With the sea level at the window's end, the initial sea level is
  /// calibrated so that the run reaches it there, by the method and within
  /// the iterations and tolerance given (InitialSeaLevel.hpp); with the
  /// sea level at the start, it is the initial sea level itself.
  bool calibrate = false;
  CalibrationMethod calibration_method = CalibrationMethod::FixedPoint;
  int calibration_iterations = 10;
  double calibration_rtol = 1e-4;
};

/// Reads an `ocean` block of a run of `problem`, relative paths against
/// `directory`. It holds exactly one of `sea_level` and `topography`, a
/// file input (Files.hpp): in the static problem with its `date_kyr_bp`
/// beside the file where the file has a time axis; in the quasi-static
/// problem with `at` ("start" or "end", required) and no date. The
/// quasi-static problem takes `shorelines` (a word, default "frozen") and
/// `calibration` (an object of `method`, `iterations` and `rtol`, each
/// optional), which the sea level's `"at": "end"` alone takes and turns
/// on; the static problem refuses both.
[[nodiscard]] OceanBlock ReadOceanBlock(const Json& block,
                                        const std::filesystem::path& directory,
                                        Problem problem);

/// Whose `solver` block: the members and the defaults differ.
enum class SolverOf {
  /// `response`'s: no sea level equation (no `rtol`); the 3D closure to
  /// 1e-10 within 200 iterations by default.
  Response,
  /// `sea-level`'s and the kernel commands': `rtol` for the sea level
  /// solves; the 3D closure to 1e-9 within 60 iterations by default.
  SeaLevel
};

/// The `solver` block: the sea level solves' tolerance, the 3D closure's
/// tolerance, budget and transforms, and the time step.
struct SolverBlock {
  /// The relative tolerance of the sea level solves (and of the kernels'
  /// adjoint sea level solves): the commands with an ocean alone.
  double rtol = 1e-9;
  /// The relative tolerance and the iteration budget of the 3D closure.
  /// `sea-level` and the kernel commands: each step's combined fixed
  /// point over the strain, the ocean load and the rotation
  /// (Evolution3D::Options::coupled_rtol, max_coupled_iterations;
  /// under lateral moduli the start's zero-length step too, and the
  /// fingerprint's one zero-length step). `response`:
  /// each step's closure (the fixed point or the conjugate gradients) and
  /// the unrelaxed solve under lateral moduli (an elastic response's one
  /// solve, a viscoelastic run's first node).
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

/// A command's `solver` block with the block absent: its defaults.
[[nodiscard]] SolverBlock SolverDefaults(SolverOf command);

/// Reads a `solver` block of a run at `lmax`, each member optional:
/// `step_rtol`, `step_max_iterations`, `transform_degrees`, `cache_mb`,
/// `scheme` and `step`, and, for a command with an ocean, `rtol`.
[[nodiscard]] SolverBlock ReadSolverBlock(const Json& block, int lmax,
                                          SolverOf command);

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

/// The `output` block: the files a run writes.
struct OutputBlock {
  std::filesystem::path surface;   ///< the surface file, required
  std::filesystem::path exchange;  ///< the coefficient table; empty for none
  std::filesystem::path volume;    ///< the model file; empty for none
  /// The node indices whose exchange tables a run through time writes;
  /// empty for every node.
  std::vector<int> exchange_nodes;
};

/// Reads an `output` block, relative paths against `directory`:
/// `surface` (required), `exchange`, `volume` and `exchange_nodes` (a list
/// of non-negative integers).
[[nodiscard]] OutputBlock ReadOutputBlock(
    const Json& block, const std::filesystem::path& directory);

/// The `rotation` word: "none", "axial" or "full".
[[nodiscard]] RotationalFeedback RotationOf(const Json& word);

/// The `moments` word: "input" (false) or "hydrostatic" (true), whether
/// the moments of inertia difference is set from the model's own
/// hydrostatic fluid limit (WithHydrostaticMoments).
[[nodiscard]] bool HydrostaticMomentsOf(const Json& word);

/// The words of a rotation and of the shorelines, as configs give them.
[[nodiscard]] std::string_view NameOf(RotationalFeedback rotation);
[[nodiscard]] std::string_view NameOf(Shorelines shorelines);

}  // namespace sl3dpp::app
