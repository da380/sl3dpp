#pragma once

/**
 * @file RunHash.hpp
 * @brief What a run's files on disk belong to: the hash of the members
 * that decide its numbers and of its input files' contents, so that
 * `solve`'s checkpoints and the trajectory store are read back by the
 * run that wrote them alone.
 */

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "Common.hpp"

namespace sl3dpp::app {

/// FNV-1a, folded over whatever is added to it.
class Fnv1a {
 public:
  void Add(std::span<const std::byte> bytes);
  template <class T>
  void AddValue(const T& value) {
    Add(std::as_bytes(std::span(&value, 1)));
  }
  /// A text's characters.
  void AddText(std::string_view text);
  /// A file's contents, read in blocks.
  void AddFile(const std::filesystem::path& path);
  [[nodiscard]] std::uint64_t Value() const { return hash_; }

 private:
  std::uint64_t hash_ = 14695981039346656037ull;
};

/// Adds the members of a run that decide its forward computation, and the
/// contents of its input files, to a hash: what determines the forward
/// run's trajectory, nothing of what only a reverse sweep reads. The
/// deck, its format and lmax, the Burgers regions of `model.burgers`
/// (added only where given),
/// the viscosity and moduli files with their references, the state's
/// files and variables (or the body's having no ocean), the land, the ice
/// change term by term (a file's contents, a cap's numbers whole beside
/// its description, which rounds them), the nodes (the window and its
/// count, or the listed nodes in their form and unit with the state's
/// date), the time step, the rotation and moments, the tolerances and
/// budget of the solves, the 3-D step's solver and, on a `lateral` run
/// (one through the 3-D operator: a lateral viscosity or lateral moduli),
/// its transforms (`transform_degrees`; on a radial run they set the
/// reverse sweep's lateral kernels alone, and are left out), the
/// shorelines and the state's end of the window, the loads and the tide
/// whole, the calibration and the run's having no gravity (added only
/// then, so that a run with gravity keeps its hash). The paths themselves
/// are left out, so that a run directory may move. In the static problem,
/// whose run has no window, the nodes' place holds the problem and the
/// state's date instead.
/// `solve`'s checkpoints hash these members alone (with, on several
/// ranks, the rank count).
void AddRunMembers(Fnv1a& hash, const Run& run, bool lateral);

/// The trajectory store's run hash (io::TrajectoryStore::Description::
/// hash): AddRunMembers with what checkpoints leave out because a
/// checkpointed run cannot have it or need not mind it: the seismic
/// sources whole (a fault file's contents, by its bytes, with its
/// quadrature, placement and time origin), the static problem's dates of
/// its terms, the file
/// inputs' formats, and the memory points the operators carry their
/// internal variables at. Not the rank count (the store checks the
/// processes' parts itself), and nothing of the `kernels`, `output`,
/// `checkpoint` or `trajectory` blocks: a recorded run serves every
/// sweep of it. `lateral` as AddRunMembers's.
[[nodiscard]] std::string TrajectoryHashOf(const Run& run, bool lateral);

}  // namespace sl3dpp::app
