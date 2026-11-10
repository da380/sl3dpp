#pragma once

/**
 * @file Blocks.hpp
 * @brief The config blocks a command made of blocks reads, each by one
 * reader here: `model` (the deck, its format, the lateral moduli and the
 * 3D viscosity), `times` (the time nodes of a run stepped through time),
 * `solver` (the solves' tolerance, budget and transforms, and the time
 * step) and `output` (the files written). The `forcing` block has its own
 * file, Forcing.hpp.
 *
 * Each reader refuses members it does not know, as every config reader
 * does (Input.hpp), and resolves relative paths against the config's
 * directory.
 */

#include <cstddef>
#include <filesystem>
#include <vector>

#include "Input.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::app {

/// The deck formats a config may name: planetmodel's GIA and MINEOS
/// tables, and the library's six-column elastic table (rho, vpv, vsv,
/// vph, vsh, eta; no header), the test and example decks' format.
/// Gia is planetmodel's Maxwell deck (`"maxwell"` in a config; `"gia"`, the
/// format's old name, is kept as an alias).
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

/// The `times` block: the time nodes of a run stepped through time, in
/// kyr on the run's own clock, counted forwards.
struct TimesBlock {
  /// The nodes, strictly increasing; the first is the start.
  std::vector<double> nodes_kyr;
  /// The step from each node to the next, one fewer than the nodes: for
  /// evenly spaced nodes the one spacing (end - start) / (nodes - 1)
  /// every time, so that equal steps are equal to the bit; the
  /// differences of the nodes given otherwise.
  std::vector<double> steps_kyr;
};

/// Reads a `times` block: `start_kyr` (default 0) with `end_kyr` (later
/// than the start) and `nodes` (at least two, evenly spaced, both ends
/// included); or instead an explicit list `at_kyr`, strictly increasing,
/// of at least two entries, its first the start. Exactly one of the two
/// forms.
[[nodiscard]] TimesBlock ReadTimesBlock(const Json& block);

/// The `solver` block: the 3D solves' tolerance, budget and transforms,
/// and the time step of a run stepped through time.
struct SolverBlock {
  /// The relative tolerance and the iteration budget of the 3D solves:
  /// the conjugate gradients of an elastic run under lateral moduli, or
  /// each step's closure (the fixed point or the conjugate gradients) of
  /// a viscoelastic run.
  double rtol = 1e-10;
  int max_iterations = 200;
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

/// Reads a `solver` block of a run at `lmax`: `rtol`, `max_iterations`,
/// `transform_degrees`, `cache_mb`, `scheme` and `step`, each optional.
[[nodiscard]] SolverBlock ReadSolverBlock(const Json& block, int lmax);

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

}  // namespace sl3dpp::app
