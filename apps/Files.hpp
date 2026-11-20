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
 * names (`ice_thickness` for `ocean.ice` and a `thickness` term,
 * `ice_thickness_change` for a `change` term, `sea_level`, `topography`,
 * `land`, a `field` term's `load`). When an input is read is its
 * member's business: the state's members at the `ocean` block's instant,
 * a term's file at a date beside it in a static problem and at the
 * window's nodes in a quasi-static one (by date in a dated run, on the
 * run's own clock in one counting forwards). How a read outside its
 * field's time span goes is the input's: refused, or, with
 * `"outside_span": "hold"`, the field's first or last stored values held
 * (ReadValues).
 */

#include <array>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Input.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// The file types a file input may name (`format`).
enum class FileFormat { PlanetmodelSurface };

/// What a read of a field on a time axis does at a time outside the
/// field's span (`outside_span`): refused, or the field's first or last
/// stored values held.
enum class OutsideSpan { Refuse, Hold };

/// A file input: the file, its type, the variable read from it, where its
/// member gives one the date it is read at (kyr before present for a
/// term's `date_kyr_bp`; a time in the term's unit on the file's own axis
/// for a static `field` term's `date`), and what a read outside its
/// field's time span does.
struct FileInput {
  std::filesystem::path file;
  FileFormat format = FileFormat::PlanetmodelSurface;
  std::string variable;
  std::optional<double> date;
  OutsideSpan outside = OutsideSpan::Refuse;
  /// Whether `outside_span` was given, which a static field refuses.
  bool outsideGiven = false;
};

/// The members every file input takes, beside those of its member (a
/// date), for the caller's Allow.
inline constexpr auto kFileInputMembers = std::array<std::string_view, 4>{
    "file", "format", "variable", "outside_span"};

/// Reads a file input's `file` (required, a relative path resolved against
/// `directory`), `format` (a word, default "planetmodel_surface"),
/// `variable` (not empty; default `variable`) and `outside_span` (a word,
/// "refuse", the default, or "hold"). The caller allows the object's
/// members (kFileInputMembers and its own) and reads its date.
[[nodiscard]] FileInput ReadFileInput(const Json& object,
                                      const std::filesystem::path& directory,
                                      std::string_view variable);

/// Allows an input's members: kFileInputMembers and `extra`.
void AllowFileInput(const Json& object,
                    std::initializer_list<std::string_view> extra = {});

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

/// Whether an input's field has a time span: its file has a time axis
/// and the field is not static (io::FieldHistory::Static).
[[nodiscard]] bool OnTimeAxis(const FileInput& input,
                              const io::FieldHistory& file);

/// Refuses `outside_span` (at the member) on an input whose field has no
/// time span (OnTimeAxis), where no read is outside a span and the member
/// would do nothing.
void CheckOutsideSpan(const FileInput& input, const io::FieldHistory& file,
                      const Json& object);

/// ReadFile, then CheckVariable and CheckOutsideSpan.
[[nodiscard]] io::FieldHistory OpenFileInput(const FileInput& input,
                                             const SurfaceGrid& grid,
                                             const Json& object);

/// The tolerance the program reads every file's time axis with, relative
/// to the larger of the axis's span and its end times' magnitudes: the
/// scale planetmodel's readings of a time axis take their `time_rtol` on
/// (planetmodel::TimeRowsAt, planetmodel::Fault::Rows, io::FieldHistory),
/// which is handed this value: a time that close to a stored time is that
/// time, under every rule.
inline constexpr double kAxisRtol = 1e-9;

/// The tolerance at the ends of a time span (SI seconds): 1e-9 of the
/// larger of its length and its ends' magnitudes, the relative tolerance a
/// measurement's date is located on the window's nodes with. A time within
/// it of an end is taken as that end, so that rounding (a node's time
/// formed from the window, a file's axis converted from its unit) raises
/// no needless refusal.
[[nodiscard]] double SpanTolerance(std::pair<double, double> span);


/// A time of a file's axis (SI seconds) as a refusal states it: a date,
/// `<d> kyr BP` (minus the time in kyr), without `unit`; with it, `<t>
/// <unit> on the file's own axis`.
[[nodiscard]] std::string AxisLabel(double time,
                                    const std::optional<TimeUnit>& unit = {});

/// The time a read of an input's field at SI time `time` of its file's
/// axis takes. On a field with a time span (OnTimeAxis) a time within
/// SpanTolerance of an end is that end; a time beyond it is refused at
/// `object`'s `file`, naming the time and the span (AxisLabel, `unit`) and
/// `where` the read is, unless the input holds its field's ends
/// (`"outside_span": "hold"`), the file then holding its first or last
/// stored values (io::FieldHistory). A static field is read at any time.
[[nodiscard]] double CoveredTime(const FileInput& input,
                                 const io::FieldHistory& file,
                                 const Json& object, double time,
                                 std::string_view where,
                                 const std::optional<TimeUnit>& unit = {});

/// The input's variable at SI time `time` of its file's axis, SI, read at
/// CoveredTime; a refusal of the file's own (a field held at its stored
/// times alone, at another) is made at `object`'s `file`. With `before`,
/// its left limit there (io::FieldHistory::ValuesBefore): where the field
/// jumps at the time (a `previous` rule, an `interval` field), the slice
/// before; the right limit is the plain read.
[[nodiscard]] std::vector<double> ReadValues(
    const FileInput& input, const io::FieldHistory& file, const Json& object,
    double time, std::string_view where,
    const std::optional<TimeUnit>& unit = {}, bool before = false);

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
