#pragma once

/**
 * @file Files.hpp
 * @brief The file inputs of the configs: a member naming a variable of a
 * file, `{"file", "format", "variable", ...}`, its reader, and the checks
 * every such input shares.
 *
 * `format` names the file's type and selects its reader: ReadFile is the
 * one place the types are told apart, so another type is one more case
 * there. `planetmodel_surface`, a planetmodel surface file read onto the
 * run's grid by io::FieldHistory, is the default and the only type.
 * `variable` names the field read, by default the quantity the member
 * names (`ice_thickness`, `sea_level`, `topography`, a `field` term's
 * `load`). When an input is read is its member's business: a date beside
 * the file in a static problem, the window's nodes or one of its ends in
 * a quasi-static one.
 */

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "Input.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// The file types a file input may name (`format`).
enum class FileFormat { PlanetmodelSurface };

/// A file input: the file, its type, the variable read from it and,
/// where its member gives one, the date it is read at (kyr before present
/// for `sea-level`'s inputs, `date_kyr_bp`; kyr on the run's clock for
/// `response`'s `field` term, `date_kyr`).
struct FileInput {
  std::filesystem::path file;
  FileFormat format = FileFormat::PlanetmodelSurface;
  std::string variable;
  std::optional<double> date;
};

/// Reads a file input's `file` (required, a relative path resolved against
/// `directory`), `format` (a word, default "planetmodel_surface") and
/// `variable` (not empty; default `variable`). The caller allows the
/// object's members and reads its date.
[[nodiscard]] FileInput ReadFileInput(const Json& object,
                                      const std::filesystem::path& directory,
                                      std::string_view variable);

/// A finite number that may be absent: a file input's date.
[[nodiscard]] std::optional<double> ReadDate(const Json& object,
                                             std::string_view key);

/// Reads a file input's file onto `grid` by its format's reader; a file
/// that cannot be read is refused at `object`'s `file`.
[[nodiscard]] io::FieldHistory ReadFile(const FileInput& input,
                                        const SurfaceGrid& grid,
                                        const Json& object);

/// Refuses an input whose variable its file does not hold, listing the
/// variables the file holds, at `object`'s `variable` (its `file` when
/// the variable is the default).
void CheckVariable(const FileInput& input, const io::FieldHistory& file,
                   const Json& object);

/// ReadFile, then CheckVariable.
[[nodiscard]] io::FieldHistory OpenFileInput(const FileInput& input,
                                             const SurfaceGrid& grid,
                                             const Json& object);

/// Whether two inputs name one file of one type, which is then read once.
[[nodiscard]] bool SameFile(const FileInput& a, const FileInput& b);

/// The rule of an input read at one instant: a file with a time axis
/// needs the date, `key` of `object` (refused at `object` without it),
/// and a file without one refuses it (at the date).
void CheckDate(const FileInput& input, const io::FieldHistory& file,
               const Json& object, std::string_view key);

/// An input as read, for a summary line or a file's metadata:
/// `<file>:<variable> (<slice>)`, the slice saying when it was read.
[[nodiscard]] std::string DescribeInput(const FileInput& input,
                                        std::string_view slice);

}  // namespace sl3dpp::app
