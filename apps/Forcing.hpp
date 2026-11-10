#pragma once

/**
 * @file Forcing.hpp
 * @brief The `forcing` block of a config: a surface load, the sum of a
 * list of terms of surface mass density, and an external tidal potential,
 * the sum of a list of coefficients; the block's reader, and its builder
 * onto a run's grid.
 *
 * A load term is one of
 *
 * - `cap`: a Gaussian cap, A exp(-d^2 / (2 w^2)) in great-circle
 *   distance d from its centre, with no cut-off; sampled on the grid and
 *   expanded by the grid's quadrature;
 * - `disk`: a uniform density A within great-circle distance alpha of its
 *   centre and zero outside, expanded exactly to the grid's degree
 *   (DiskCoefficients) rather than sampled across its edge;
 * - `field`: a field of a planetmodel surface file, resampled onto the
 *   grid by the field's own lateral rule and converted to SI by its
 *   declared dimensions (io::FieldHistory), then expanded;
 * - `coefficients`: a text table of the density's coefficients
 *   (ReadCoefficientTable).
 *
 * The load is the sum of its terms, and the tide the sum of its
 * coefficients. In a run stepped through time (a `times` block) a term may
 * carry an amplitude history (`history_kyr`), the piecewise-linear factor
 * its pattern is scaled by at each time, and a `field` term may come from
 * a file with a time axis, read at each time by the file's own time rule;
 * BuildForcing builds the fixed forcing of an elastic run and TimedForcing
 * the forcing of a run through time. Coefficients throughout are the
 * grid's (SurfaceGrid.hpp): complex, orthonormal, with the Condon-Shortley
 * phase, Y_lm = sqrt((2l+1)/4pi) d^l_0m(theta) e^{i m phi}, the m >= 0 ones
 * of a real field (f_(l,-m) = (-1)^m conj(f_lm)), in SI.
 */

#include <complex>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "Input.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// A term's amplitude history: factors at strictly increasing times, kyr
/// on the run's clock. The factor at a time is their piecewise-linear
/// interpolant, held at the first factor before the first time and at the
/// last after the last; without a history it is 1 throughout.
struct AmplitudeHistory {
  std::vector<double> times_kyr;
  std::vector<double> factors;

  [[nodiscard]] bool Empty() const { return times_kyr.empty(); }
  /// The factor at time t, kyr.
  [[nodiscard]] double At(double t) const;
};

/// One term of a surface load: a surface mass density, kg m^-2.
struct LoadTerm {
  enum class Kind { Cap, Disk, Field, Coefficients };
  Kind kind = Kind::Cap;
  GeographicPoint site{};  ///< a cap's or a disk's centre
  double width = 0.0;      ///< a cap's Gaussian width w, radians
  double radius = 0.0;     ///< a disk's angular radius alpha, radians
  /// kg m^-2: a cap's at its centre, a disk's within it.
  double amplitude = 0.0;
  /// A field's surface file, or a coefficient table.
  std::filesystem::path file;
  std::string name = "load";  ///< a field's name in its file
  /// The run's time, kyr, at which a field file's time zero falls: the
  /// file is read at (t - time_origin_kyr) kyr, in seconds, at the run's
  /// time t.
  double time_origin_kyr = 0.0;
  AmplitudeHistory history;  ///< the term's factor through time
};

/// One coefficient of an external tidal potential, m^2 s^-2, at an order
/// m >= 0 (the potential is real).
struct TideTerm {
  int degree = 2;
  int order = 0;
  double amplitude = 0.0;    ///< the real part
  double imaginary = 0.0;    ///< the imaginary part, zero at order 0
  AmplitudeHistory history;  ///< the coefficient's factor through time
};

/// The `forcing` block: the load terms and the tide coefficients, each
/// list summed.
struct Forcing {
  std::vector<LoadTerm> load;
  std::vector<TideTerm> tide;
};

/// Reads a `forcing` block for a run at `lmax`, relative paths against
/// `directory`. Each element of `load` is an object holding exactly one of
/// `cap`, `disk`, `field` and `coefficients`, and, in a run through time
/// (`timed`), optionally `history_kyr`; each element of `tide` is a
/// coefficient, likewise with an optional `history_kyr`. A history is a
/// list of `[time_kyr, factor]` pairs, its times strictly increasing; a
/// `field` term of a timed run may set `time_origin_kyr`. At least one
/// load term or tide coefficient is needed. Without `timed`, a history or
/// a time origin is refused: it needs a `times` block.
[[nodiscard]] Forcing ReadForcing(const Json& block,
                                  const std::filesystem::path& directory,
                                  int lmax, bool timed = false);

/// A forcing on a run's grid, SI.
struct BuiltForcing {
  /// Each load term's coefficients, kg m^-2, in the order given.
  std::vector<std::vector<std::complex<double>>> terms;
  /// The load's coefficients, kg m^-2: the terms' sum (zero without
  /// terms).
  std::vector<std::complex<double>> loadCoefficients;
  /// The load at the grid's points, kg m^-2: the sum of each term's field
  /// whose expansion is that term's coefficients — a cap's or a field's
  /// values as sampled, a disk's or a table's synthesised from its
  /// coefficients.
  std::vector<double> load;
  /// The tidal potential's coefficients, m^2 s^-2: the coefficients' sum.
  std::vector<std::complex<double>> tide;
};

/// Builds a fixed forcing on `grid`, reading the files its terms name.
/// `block` is the config's `forcing` block, where a fault in a term's file
/// is reported: a field file that cannot be read, has a time axis, lacks
/// the field or does not cover the grid, or a coefficient table that is
/// malformed or exceeds the grid's degree.
[[nodiscard]] BuiltForcing BuildForcing(const Forcing& forcing,
                                        const SurfaceGrid& grid,
                                        const Json& block);

/// A forcing on a run's grid at any time of the run's clock: each load
/// term's pattern times its amplitude history, summed in the order given,
/// and each tide coefficient times its own. The patterns are built once,
/// as BuildForcing builds them, but for a `field` term whose file has a
/// time axis: that file is read at each time asked, at file time
/// (t - time_origin_kyr) kyr in seconds, by the file's own time rule (a
/// `node` field interpolated between its nodes, its end values held
/// outside its span), and expanded there.
class TimedForcing {
 public:
  /// Reads the terms' files onto `grid`. `block` is the config's `forcing`
  /// block, where a fault in a term's file is reported, then and at any
  /// time asked; it and `grid` must outlive the forcing.
  TimedForcing(const Forcing& forcing, const SurfaceGrid& grid,
               const Json& block);

  /// The forcing at time t, kyr, SI: each load term's coefficients (its
  /// factor applied), their sum, the load's field at the grid's points
  /// (the terms' fields, as BuiltForcing::load, each times its factor),
  /// and the tide.
  [[nodiscard]] BuiltForcing At(double t) const;

  /// Whether any term varies in time: a history, or a field file with a
  /// time axis.
  [[nodiscard]] bool Varies() const { return varies_; }

 private:
  /// A load term's pattern: fixed, or read from its file's time axis.
  struct Pattern {
    std::vector<double> field;
    std::vector<std::complex<double>> coefficients;
    std::optional<io::FieldHistory> history;
  };
  Forcing forcing_;
  const SurfaceGrid* grid_;
  const Json* block_;
  std::vector<Pattern> patterns_;
  bool varies_ = false;
};

/// A Gaussian cap at the grid's points: amplitude exp(-d^2 / (2 width^2))
/// in great-circle distance d (radians) from `centre`.
[[nodiscard]] std::vector<double> CapField(const SurfaceGrid& grid,
                                           GeographicPoint centre, double width,
                                           double amplitude);

/// The coefficients, to the grid's degree, of a uniform surface density
/// `amplitude` within great-circle distance `radius` (radians, in
/// (0, pi]) of `centre` and zero outside. By the Funk-Hecke theorem,
///
///   f_lm = 2 pi amplitude (int_{cos radius}^1 P_l(x) dx) conj(Y_lm(centre)),
///
/// the integral being 1 - cos(radius) at l = 0 and
/// sin(radius) P_l^1(cos(radius)) / (l (l + 1)) above, with P_l^1 the
/// associated Legendre function without the Condon-Shortley phase, by its
/// upward recurrence: the truncated exact expansion.
[[nodiscard]] std::vector<std::complex<double>> DiskCoefficients(
    const SurfaceGrid& grid, GeographicPoint centre, double radius,
    double amplitude);

/// The grid's harmonics Y_lm at one point, degrees 0 to `lmax`, m >= 0,
/// in the grid's coefficient layout (SurfaceGrid::Index), from GSHTrans's
/// Wigner functions: the values the grid's synthesis sums.
[[nodiscard]] std::vector<std::complex<double>> HarmonicsAt(
    int lmax, GeographicPoint point);

/// Reads a table of a real field's coefficients to degree `lmax`, in the
/// grid's convention: `#` comment lines and blank lines, and rows
/// `l m re im` with 0 <= m <= l. Degrees absent are zero. A row above
/// `lmax`, a row given twice, a nonzero imaginary part at m = 0 or a
/// malformed row is refused, reported at `member` with its line.
[[nodiscard]] std::vector<std::complex<double>> ReadCoefficientTable(
    const std::filesystem::path& path, int lmax, const Json& member);

/// One line describing a load term or a tide coefficient, for provenance:
/// its pattern and, where set, its time origin and amplitude history.
[[nodiscard]] std::string Describe(const LoadTerm& term);
[[nodiscard]] std::string Describe(const TideTerm& term);

/// What a forcing holds, as `load (2 terms) + tide (1 coefficient)`.
[[nodiscard]] std::string Summary(const Forcing& forcing);

}  // namespace sl3dpp::app
