#pragma once

/**
 * @file Blocks.hpp
 * @brief The config blocks a command made of blocks reads, each by one
 * reader here: `model` (the deck, its format and the lateral moduli),
 * `solver` (the 3D solve's tolerance, budget and transforms) and `output`
 * (the files written). The `forcing` block has its own file, Forcing.hpp.
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
/// file of lateral moduli.
struct ModelBlock {
  std::filesystem::path deck;
  ModelFormat format = ModelFormat::Gia;
  /// A planetmodel model file whose `mu` and `kappa` fields supply the
  /// lateral perturbations of the moduli (io/ModelModuli); empty for the
  /// radial model alone.
  std::filesystem::path moduli;
  bool moduli_file_radial = false;  ///< the file's own radial reference
};

/// Reads a `model` block, relative paths against `directory`: `deck`
/// (required), `format` (a word, default "maxwell") and `moduli` (an
/// object of `file` and `reference`).
[[nodiscard]] ModelBlock ReadModelBlock(const Json& block,
                                        const std::filesystem::path& directory);

/// The `solver` block: the conjugate-gradient solve under lateral moduli
/// and the 3D operator's transforms.
struct SolverBlock {
  double rtol = 1e-10;
  int max_iterations = 200;
  /// The degrees the 3D coupling's transforms may be cut to with depth
  /// (Viscoelastic3D::Options::transform_degrees); empty, every node at
  /// lmax.
  std::vector<int> transform_degrees;
  /// The cache every transform grid of the run blocks its work for, in
  /// MiB; 0, the library's default.
  std::size_t cache_mb = 0;
};

/// Reads a `solver` block of a run at `lmax`: `rtol`, `max_iterations`,
/// `transform_degrees` and `cache_mb`, each optional.
[[nodiscard]] SolverBlock ReadSolverBlock(const Json& block, int lmax);

/// The `output` block: the files a run writes.
struct OutputBlock {
  std::filesystem::path surface;   ///< the surface file, required
  std::filesystem::path exchange;  ///< the coefficient table; empty for none
  std::filesystem::path volume;    ///< the model file; empty for none
};

/// Reads an `output` block, relative paths against `directory`:
/// `surface` (required), `exchange` and `volume`.
[[nodiscard]] OutputBlock ReadOutputBlock(
    const Json& block, const std::filesystem::path& directory);

}  // namespace sl3dpp::app
