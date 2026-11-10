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
 * Every term is a fixed pattern; the load is their sum, and the tide the
 * sum of its coefficients. Coefficients throughout are the grid's
 * (SurfaceGrid.hpp): complex, orthonormal, with the Condon-Shortley phase,
 * Y_lm = sqrt((2l+1)/4pi) d^l_0m(theta) e^{i m phi}, the m >= 0 ones of a
 * real field (f_(l,-m) = (-1)^m conj(f_lm)), in SI.
 */

#include <complex>
#include <filesystem>
#include <string>
#include <vector>

#include "Input.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp::app {

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
};

/// One coefficient of an external tidal potential, m^2 s^-2, at an order
/// m >= 0 (the potential is real).
struct TideTerm {
  int degree = 2;
  int order = 0;
  double amplitude = 0.0;  ///< the real part
  double imaginary = 0.0;  ///< the imaginary part, zero at order 0
};

/// The `forcing` block: the load terms and the tide coefficients, each
/// list summed.
struct Forcing {
  std::vector<LoadTerm> load;
  std::vector<TideTerm> tide;
};

/// Reads a `forcing` block for a run at `lmax`, relative paths against
/// `directory`. Each element of `load` is an object holding exactly one of
/// `cap`, `disk`, `field` and `coefficients`; each element of `tide` is a
/// coefficient. At least one load term or tide coefficient is needed. An
/// amplitude history (`history_kyr`) is refused: it needs a `times` block.
[[nodiscard]] Forcing ReadForcing(const Json& block,
                                  const std::filesystem::path& directory,
                                  int lmax);

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

/// Builds a forcing on `grid`, reading the files its terms name. `block`
/// is the config's `forcing` block, where a fault in a term's file is
/// reported: a field file that cannot be read, has a time axis, lacks the
/// field or does not cover the grid, or a coefficient table that is
/// malformed or exceeds the grid's degree.
[[nodiscard]] BuiltForcing BuildForcing(const Forcing& forcing,
                                        const SurfaceGrid& grid,
                                        const Json& block);

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

/// One line describing a load term or a tide coefficient, for provenance.
[[nodiscard]] std::string Describe(const LoadTerm& term);
[[nodiscard]] std::string Describe(const TideTerm& term);

/// What a forcing holds, as `load (2 terms) + tide (1 coefficient)`.
[[nodiscard]] std::string Summary(const Forcing& forcing);

}  // namespace sl3dpp::app
