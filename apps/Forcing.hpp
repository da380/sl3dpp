#pragma once

/**
 * @file Forcing.hpp
 * @brief The `forcing` blocks of the configs. `response`'s: a surface
 * load, the sum of a list of terms of surface mass density, and an
 * external tidal potential, the sum of a list of coefficients; the block's
 * reader, and its builder onto a run's grid; and, in a run through time,
 * seismic sources, point moment tensors and planar faults inside the
 * body (SeismicTerm). `sea-level`'s (and the kernel commands'): the ice
 * and, in a run through time, surface loads that each state their place
 * in the water budget and whether they move the solid surface, and
 * seismic sources (SeaLevelForcing, at the end of this file); their sea
 * level is the `ocean` block's (Blocks.hpp).
 *
 * A load term is one of
 *
 * - `cap`: a Gaussian cap, A exp(-d^2 / (2 w^2)) in great-circle
 *   distance d from its centre, with no cut-off; sampled on the grid and
 *   expanded by the grid's quadrature;
 * - `disk`: a uniform density A within great-circle distance alpha of its
 *   centre and zero outside, expanded exactly to the grid's degree
 *   (DiskCoefficients) rather than sampled across its edge;
 * - `field`: a variable of a file input (Files.hpp), resampled onto the
 *   grid by the field's own lateral rule and converted to SI by its
 *   declared dimensions (io::FieldHistory), then expanded;
 * - `coefficients`: a text table of the density's coefficients
 *   (ReadCoefficientTable).
 *
 * The load is the sum of its terms, and the tide the sum of its
 * coefficients. In the static problem a `field` term's file with a time
 * axis is read at the term's `date_kyr`. In the quasi-static problem (a
 * run stepped through time) a term may carry an amplitude history
 * (`history_kyr`), the piecewise-linear factor its pattern is scaled by at
 * each time, and a `field` term may come from a file with a time axis,
 * read at each time by the file's own time rule; BuildForcing builds the
 * fixed forcing of a static run and TimedForcing the forcing of a run
 * through time. Coefficients throughout are the
 * grid's (SurfaceGrid.hpp): complex, orthonormal, with the Condon-Shortley
 * phase, Y_lm = sqrt((2l+1)/4pi) d^l_0m(theta) e^{i m phi}, the m >= 0 ones
 * of a real field (f_(l,-m) = (-1)^m conj(f_lm)), in SI.
 */

#include <complex>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "Files.hpp"
#include "Input.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/PointSource.hpp"
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
  /// A field's file input (its variable by default `load`) and, in the
  /// static problem, the time its file is read at, kyr (`date_kyr`),
  /// required of a file with a time axis.
  FileInput field;
  std::filesystem::path file;  ///< a coefficient table
  /// The quasi-static problem: the run's time, kyr, at which a field
  /// file's time zero falls: the file is read at (t - time_origin_kyr)
  /// kyr, in seconds, at the run's time t.
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

/// One seismic source of a run through time: a point moment tensor, or a
/// planar fault of uniform slip laid out as point sources (Fault.hpp).
/// The moment is the stress glut itself: it acts from `onset_kyr` (the
/// run's start when absent) and is held, times its amplitude history
/// where one is given. SI throughout: metres and N m.
struct SeismicTerm {
  enum class Kind { Point, Fault };
  Kind kind = Kind::Point;
  GeographicPoint site{};  ///< the point, or the fault's centroid
  double depth = 0.0;      ///< below the surface, m
  /// A point's moment tensor in the spherical polar basis at the point
  /// (r up, theta south, phi east), N m.
  SphericalTensor moment{};
  /// A fault's plane and slip (Aki & Richards), degrees, its size (m),
  /// its scalar moment (N m) and its patches along strike and down dip.
  double strike = 0.0, dip = 90.0, rake = 0.0;
  double length = 0.0, width = 0.0;
  double scalarMoment = 0.0;
  int alongStrike = 1, downDip = 1;
  std::optional<double> onset_kyr;  ///< absent: from the run's start
  AmplitudeHistory history;

  /// The factor of the moment at time t, kyr: zero before the onset.
  [[nodiscard]] double FactorAt(double t) const;
};

/// The `forcing` block: the load terms and the tide coefficients, each
/// list summed, and the seismic sources.
struct Forcing {
  std::vector<LoadTerm> load;
  std::vector<TideTerm> tide;
  std::vector<SeismicTerm> seismic;
};

/// The point sources of a forcing's seismic terms at time t, kyr, SI
/// (radius in metres from the centre, moments in N m), for a surface of
/// radius `surfaceRadius` metres. Every term gives its sources at every
/// time, with zero moment before its onset.
[[nodiscard]] std::vector<PointSource> SeismicSourcesAt(const Forcing& forcing,
                                                        double t,
                                                        double surfaceRadius);

/// Reads a `forcing` block for a run at `lmax`, relative paths against
/// `directory`. Each element of `load` is an object holding exactly one of
/// `cap`, `disk`, `field` and `coefficients`, and, in the quasi-static
/// problem (`timed`), optionally `history_kyr`; each element of `tide` is
/// a coefficient, likewise with an optional `history_kyr`. A history is a
/// list of `[time_kyr, factor]` pairs, its times strictly increasing. A
/// `field` term is a file input (`file`, `format`, `variable`, the
/// variable by default `load`) with, in the static problem, `date_kyr`
/// (the time its file is read at, kyr on the file's own axis) and, in the
/// quasi-static problem, `time_origin_kyr`. At least one load term, tide
/// coefficient or seismic source is needed. Each element
/// of `seismic` is an object holding exactly one of `point` (`latitude`,
/// `longitude`, `depth_km` and either `moment_tensor_nm` with members
/// `rr`, `tt`, `pp`, `rt`, `rp`, `tp`, or `strike_deg`, `dip_deg`,
/// `rake_deg` and `moment_nm`) and `fault` (`latitude`, `longitude`,
/// `depth_km` of its centroid, `strike_deg`, `dip_deg`, `rake_deg`,
/// `length_km`, `width_km`, `moment_nm`, and optionally
/// `patches_along_strike` and `patches_down_dip`), with optionally
/// `onset_kyr` and `history_kyr`. Without `timed`, a history, a time
/// origin or a seismic source is refused: it needs the quasi-static
/// problem; with it, a field's `date_kyr` is refused.
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
  /// BuildForcing: each `field` term's input as read (DescribeField).
  std::vector<std::string> inputs;
};

/// Builds a fixed forcing on `grid`, reading the files its terms name.
/// `block` is the config's `forcing` block, where a fault in a term's file
/// is reported: a field file that cannot be read, lacks the variable or
/// does not cover the grid, that has a time axis but no `date_kyr` or a
/// `date_kyr` but no time axis, or a coefficient table that is malformed
/// or exceeds the grid's degree.
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

  /// Each `field` term's input as read (DescribeField).
  [[nodiscard]] const std::vector<std::string>& Inputs() const {
    return inputs_;
  }

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
  std::vector<std::string> inputs_;
  bool varies_ = false;
};

/// A `field` term's input as read, for a summary line and the outputs'
/// metadata: `field <file>:<variable> (<slice>)`, the slice `static` for a
/// file without a time axis, `<t> kyr` for the static problem's date, and
/// in a run through time `the run's clock` or `time origin <t> kyr`.
[[nodiscard]] std::string DescribeField(const LoadTerm& term, bool timeAxis,
                                        bool timed);

/// A site's `latitude` and `longitude`, degrees; a latitude outside
/// [-90, 90] is refused.
[[nodiscard]] GeographicPoint ReadSite(const Json& object);

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
[[nodiscard]] std::string Describe(const SeismicTerm& term);

/// What a forcing holds, as `load (2 terms) + tide (1 coefficient)`.
[[nodiscard]] std::string Summary(const Forcing& forcing);

// -- the forcing of a run with an ocean ---------------------------------------

/// The elastic fingerprint's ice change: a Gaussian cap, of a fraction of
/// the ice thickness at the date or of a thickness.
struct IceChange {
  enum class Kind { CapFraction, CapThickness } kind = Kind::CapFraction;
  GeographicPoint site;
  double width = 0.0;   ///< the Gaussian cap's width, radians
  double amount = 0.0;  ///< the fraction, or the thickness in metres
};

/// The ice: a file input of the ice thickness (its variable by default
/// `ice_thickness`). In the quasi-static problem it is an ice history,
/// the thickness through the window (a static file is a history that does
/// not change); in the static problem it is read at its date
/// (`date_kyr_bp`, kyr before present, required of a file with a time
/// axis) and carries the fingerprint's change.
struct IceForcing {
  FileInput file;
  std::optional<IceChange> change;  ///< the static problem's
};

/// A surface load of a run with an ocean: a load term (a surface mass
/// density, as `response`'s) with the two properties the sea level
/// equation needs of it. A load in the WATER BUDGET takes its mass from
/// the ocean (a hydrological load); one outside it moves no water
/// (sediment, any other mass). A load that MOVES THE SOLID SURFACE is a
/// layer laid on it, of thickness its mass over its density: the sea
/// level, measured from the solid surface, falls by the thickness, and
/// under migrating shorelines the shorelines move with it.
struct SurfaceLoadTerm {
  LoadTerm term;
  bool water_budget = false;
  bool moves_solid_surface = false;
  double density = 0.0;  ///< kg m^-3, of a load that moves the surface
};

/// The `forcing` block of `sea-level` and the kernel commands: the ice, a
/// direct load. The ice the run starts from is the history's own, so the
/// ice and the sea level cannot disagree; the load is the history's change
/// from its start. The sea level is the `ocean` block's (Blocks.hpp). A
/// run through time takes surface loads and seismic sources beside the
/// ice, changes from the run's start on the run's clock (kyr from the
/// window's start): a term present at the first node acts there.
struct SeaLevelForcing {
  IceForcing ice;
  std::vector<SurfaceLoadTerm> load;
  std::vector<SeismicTerm> seismic;

  /// Whether the run is forced by more than its ice.
  [[nodiscard]] bool BesideTheIce() const {
    return !load.empty() || !seismic.empty();
  }
};

/// Reads a `forcing` block of a run with an ocean, relative paths against
/// `directory`: `ice`, required, a file input. In the quasi-static problem
/// (`timed`) it is the ice history alone; in the static problem it takes
/// `date_kyr_bp` beside its file and the change, `type` ("cap_fraction"
/// or "cap_thickness"), `latitude`, `longitude`, `width_deg` and the
/// amount, `fraction` or `thickness_m`. The members of the other problem
/// are refused, naming what this one takes, and so are `sea_level` and
/// `state`, naming the `ocean` block's members. The quasi-static problem
/// also takes `load`, a list of load terms as `response`'s (ReadForcing),
/// each with `water_budget` and `moves_solid_surface` (both required,
/// true or false) and, where it moves the solid surface, `density_kg_m3`
/// (positive; refused otherwise), and `seismic`, a list of sources as
/// `response`'s; the static problem refuses both.
[[nodiscard]] SeaLevelForcing ReadSeaLevelForcing(
    const Json& block, const std::filesystem::path& directory, bool timed);

/// One line describing a surface load, for provenance: its term, its
/// place in the water budget and, of a layer, its density.
[[nodiscard]] std::string Describe(const SurfaceLoadTerm& term);

/// The surface loads of a run with an ocean on its grid, at any time of
/// the run's clock, by their properties.
class SurfaceLoads {
 public:
  /// Reads the terms' files onto `grid`; `block` is the config's `forcing`
  /// block (TimedForcing). It and `grid` must outlive the loads.
  SurfaceLoads(const SeaLevelForcing& forcing, const SurfaceGrid& grid,
               const Json& block);

  /// The loads at one time, SI, at the grid's points and band-limited to
  /// its degree; a kind no term has is empty.
  struct Node {
    std::vector<double> inside;   ///< in the water budget, kg m^-2
    std::vector<double> outside;  ///< outside it, kg m^-2
    std::vector<double> layer;    ///< the layer's thickness, m
  };
  [[nodiscard]] Node At(double t_kyr) const;

  /// Each `field` term's input as read (DescribeField).
  [[nodiscard]] const std::vector<std::string>& Inputs() const {
    return timed_.Inputs();
  }

 private:
  std::vector<SurfaceLoadTerm> terms_;
  const SurfaceGrid* grid_;
  TimedForcing timed_;
};

}  // namespace sl3dpp::app
