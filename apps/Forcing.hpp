#pragma once

/**
 * @file Forcing.hpp
 * @brief The `forcing` blocks of the configs. `response`'s: a surface
 * load, the sum of a list of terms of surface mass density, and an
 * external tidal potential, the sum of a list of coefficients; the block's
 * reader, and its builder onto a run's grid; and, in a run through time,
 * seismic sources, point moment tensors and planar faults inside the
 * body (SeismicTerm). `sea-level`'s (and the kernel commands'): the ice
 * change, a sum of terms, surface loads that each state their place in
 * the water budget and whether they move the solid surface, a tide and
 * seismic sources (SeaLevelForcing, at the end of this file); the state
 * they change, the sea level and the ice on the body at one instant, is
 * the `ocean` block's (Blocks.hpp), and without one the body has no
 * ocean and no ice to change.
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
 * axis is read at the term's `date_kyr` (`sea-level`'s `date_kyr_bp`, kyr
 * before present, or `date` with its `unit` on the file's own axis). In
 * the quasi-static problem (a run stepped through time) a term may carry
 * an amplitude history (`history_kyr`; `sea-level`'s `history_kyr_bp` or
 * `history`, TermClock), the piecewise-linear factor its pattern is
 * scaled by at each time, and a `field` term may come from a file with a
 * time axis, read at each time by the file's own time rule; BuildForcing
 * builds the fixed forcing of a static run and TimedForcing the forcing of
 * a run through time. Coefficients throughout are the
 * grid's (SurfaceGrid.hpp): complex, orthonormal, with the Condon-Shortley
 * phase, Y_lm = sqrt((2l+1)/4pi) d^l_0m(theta) e^{i m phi}, the m >= 0 ones
 * of a real field (f_(l,-m) = (-1)^m conj(f_lm)), in SI.
 */

#include <complex>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "Files.hpp"
#include "Input.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// How a block's terms give their times, and the clock a run through time
/// reads them on (the terms' clock, which the times below are on):
///
/// - Response: `response`'s, kyr on the run's own clock (`history_kyr`,
///   `onset_kyr`, `time_origin_kyr`; a static field's `date_kyr` on its
///   file's own axis).
/// - Dated: `sea-level`'s dated form, dates in kyr before present
///   (`history_kyr_bp`, `onset_kyr_bp`, a static field's `date_kyr_bp`; a
///   field file read by date through time), turned to kyr from the
///   window's start, the date of the clock's zero, which a run through time
///   has (`start_kyr_bp`). The static problem's terms are of this form,
///   without a start; its `field` terms may instead give `date` with its
///   `unit` on the file's own axis.
/// - Forwards: `sea-level`'s forwards form, times on the run's own clock
///   in its `unit` (`history`, `onset`, a field's `time_origin`), taken as
///   they stand.
struct TermClock {
  enum class Form { Response, Dated, Forwards };
  Form form = Form::Response;
  /// Dated, in a run through time: the window's start, kyr BP.
  std::optional<double> start_kyr_bp;
  /// Forwards: the run's unit.
  TimeUnit unit;
};

/// A term's amplitude history: factors at strictly increasing times on the
/// terms' clock (TermClock), in its unit. The factor at a time is their
/// piecewise-linear interpolant, held at the first factor before the first
/// time and at the last after the last; without a history it is 1
/// throughout.
struct AmplitudeHistory {
  std::vector<double> times;
  std::vector<double> factors;
  /// Where the history was given as dates before present
  /// (`history_kyr_bp`, `sea-level`'s dated form), the date of the clock's
  /// zero, kyr BP: the times above are then this less the dates, in kyr.
  std::optional<double> start_kyr_bp;
  /// Where it was given on a run's own clock in that run's unit
  /// (`history`, `sea-level`'s forwards form), the unit; absent for kyr
  /// (`history_kyr`, `response`'s form) and for dates.
  std::optional<TimeUnit> unit;

  [[nodiscard]] bool Empty() const { return times.empty(); }
  /// The factor at time t of the terms' clock.
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
  /// static problem, the date its file is read at (`date_kyr`, kyr on the
  /// file's own axis; `date_kyr_bp`, kyr before present; or `date`, in
  /// `unit` on the file's own axis), required of a file with a time axis.
  FileInput field;
  std::filesystem::path file;  ///< a coefficient table
  /// The quasi-static problem: the time of the terms' clock at which a
  /// field file's time zero falls, in the clock's unit: the file is read at
  /// (t - time_origin) units, in seconds, at the clock's time t.
  double time_origin = 0.0;
  /// The unit of the terms' clock (kyr but in `sea-level`'s forwards form,
  /// the run's unit) and of a static `date` on the file's own axis.
  TimeUnit unit;
  /// Whether the term's dates are before present (`sea-level`'s form): a
  /// static problem's date is then `date_kyr_bp`, the file read at minus
  /// it on its axis.
  bool before_present = false;
  /// The member the static problem's date is given by (`date_kyr`,
  /// `date_kyr_bp` or `date`), which a refusal names.
  std::string_view date_member = "date_kyr";
  /// Whether the file is read by date through time (`sea-level`'s form in
  /// a run through time): its time axis is the calendar's, read at each
  /// node's date, and the origin above is the window's start.
  bool dated = false;
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
/// The moment is the stress glut itself: it acts from its onset (the
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
  /// The onset on the terms' clock (TermClock), in its unit (`onset_kyr`,
  /// `onset_kyr_bp` turned to it, or `onset`); absent: from the run's
  /// start.
  std::optional<double> onset;
  /// Where the onset was given as a date (`onset_kyr_bp`), the date of the
  /// clock's zero, kyr BP.
  std::optional<double> start_kyr_bp;
  /// The unit of the terms' clock, which a description names.
  TimeUnit unit;
  AmplitudeHistory history;

  /// The factor of the moment at time t of the terms' clock: zero before
  /// the onset.
  [[nodiscard]] double FactorAt(double t) const;
};

/// The `forcing` block: the load terms and the tide coefficients, each
/// list summed, and the seismic sources.
struct Forcing {
  std::vector<LoadTerm> load;
  std::vector<TideTerm> tide;
  std::vector<SeismicTerm> seismic;
};

/// The point sources of a forcing's seismic terms at time t of the terms'
/// clock (kyr in `response`), SI
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

/// A forcing on a run's grid at any time of the terms' clock (TermClock):
/// each load term's pattern times its amplitude history, summed in the
/// order given, and each tide coefficient times its own. The patterns are
/// built once, as BuildForcing builds them, but for a `field` term whose
/// file has a time axis: that file is read at each time asked, at file
/// time (t - time_origin) in the term's unit, in seconds, by the file's
/// own time rule (a `node` field interpolated between its nodes, its end
/// values held outside its span), and expanded there.
class TimedForcing {
 public:
  /// Reads the terms' files onto `grid`. `block` is the config's `forcing`
  /// block, where a fault in a term's file is reported, then and at any
  /// time asked; it and `grid` must outlive the forcing. Without `timed`
  /// (a static problem's terms) each term is read once, as BuildForcing
  /// reads it, a field file at its date, and the forcing is the same at
  /// every time.
  TimedForcing(const Forcing& forcing, const SurfaceGrid& grid,
               const Json& block, bool timed = true);

  /// The forcing at time t of the terms' clock, SI: each load term's
  /// coefficients (its
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
/// file without a time axis, `<t> kyr`, `<d> kyr BP` or `<t> <unit>` for
/// the static problem's date, and in a run through time `each node's
/// date`, `the run's clock` or `time origin <t> <unit>`.
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

// -- the forcing of `sea-level` and the kernel commands -----------------------

/// One term of an ice change (`forcing.ice_change`): a change of the ice
/// thickness, metres, the change being the sum of the terms. The ice at
/// a time is the state's (`ocean.ice`) plus the change there.
///
/// - `thickness`: a file of ice THICKNESS (its variable by default
///   `ice_thickness`), whose change is its value less its own value at
///   the state's instant. The static problem reads it at its date
///   (`date_kyr_bp`, required), a dated run through time at each node's
///   date; its file has a time axis, a change of a file without one being
///   identically zero. It is a dated ice history, which a run counting
///   forwards on its own clock refuses.
/// - `change`: a file holding a thickness CHANGE (its variable by default
///   `ice_thickness_change`), taken as it stands. The static problem reads
///   a file with a time axis at its date (`date_kyr_bp`) and one without
///   as it is; a dated run through time reads one with a time axis at each
///   node's date and holds one without through the run; a run counting
///   forwards holds one without a time axis through the run and refuses
///   one with (a dated history).
/// - `cap_thickness`, `cap_fraction`: a Gaussian cap, weighted
///   exp(-d^2 / (2 w^2)) in great-circle distance d from its centre with
///   no cut-off, of a thickness at its centre or of a fraction of the
///   state's ice. A cap of negative thickness takes at each point no more
///   than the ice the state holds there. In a run through time a cap is
///   held through the run, or scaled by its amplitude history
///   (`history_kyr_bp` or `history`, as a surface load's) at each time.
struct IceChangeTerm {
  enum class Kind { Thickness, Change, CapThickness, CapFraction };
  Kind kind = Kind::Thickness;
  /// `thickness` and `change`: the file input and, in the static problem,
  /// the date it is read at, kyr before present.
  FileInput file;
  /// Whether the file is read at each node's date (a dated run through
  /// time).
  bool dated = false;
  /// Whether the file is one field held through the run (a `change` term
  /// of a run counting forwards, whose file has no time axis).
  bool held = false;
  GeographicPoint site{};  ///< a cap's centre
  double width = 0.0;      ///< a cap's Gaussian width, radians
  /// A cap's amount: its thickness at the centre in metres, or the
  /// fraction of the state's ice.
  double amount = 0.0;
  /// A cap's factor through time, in a run through time; empty, 1.
  AmplitudeHistory history;

  /// The term's member: `thickness`, `change`, `cap_thickness` or
  /// `cap_fraction`.
  [[nodiscard]] std::string_view Member() const;
};

/// A surface load of `sea-level`: a load term (a surface mass density, as
/// `response`'s) with the two properties the sea level equation needs of
/// it. A load in the WATER BUDGET takes its mass from the ocean (a
/// hydrological load); one outside it moves no water (sediment, any other
/// mass). A load that MOVES THE SOLID SURFACE is a layer laid on it, of
/// thickness its mass over its density: the sea level, measured from the
/// solid surface, falls by the thickness, and under migrating shorelines
/// the shorelines move with it. On a body without an ocean the water
/// budget has no ocean to change, and a layer is its thickness off the
/// relative sea level change alone.
struct SurfaceLoadTerm {
  LoadTerm term;
  bool water_budget = false;
  bool moves_solid_surface = false;
  double density = 0.0;  ///< kg m^-3, of a load that moves the surface
};

/// The `forcing` block of `sea-level` and the kernel commands: what
/// happens to the state (the `ocean` block's, Blocks.hpp). The ice change
/// is the sum of its terms; the ice at a time is the state's plus the
/// change there, a direct load where it is grounded. Surface loads, the
/// tide (a sum of coefficients of an external potential, as `response`'s)
/// and seismic sources act beside the ice, changes from the state: in the
/// static problem at its one instant, in a run through time on the terms'
/// clock (TermClock: kyr from the window's start in a dated run, the run's
/// own clock in its unit in one counting forwards), a term present at the
/// first node acting there. A run without an ocean has no ice change.
struct SeaLevelForcing {
  std::vector<IceChangeTerm> ice_change;
  std::vector<SurfaceLoadTerm> load;
  std::vector<TideTerm> tide;
  std::vector<SeismicTerm> seismic;

  /// Whether the run is forced by more than its ice.
  [[nodiscard]] bool BesideTheIce() const {
    return !load.empty() || !tide.empty() || !seismic.empty();
  }
};

/// Reads a `forcing` block of `sea-level` or a kernel command at `lmax`,
/// relative paths against `directory`, of a run with an ocean (`ocean`,
/// the config having an `ocean` block) or without one. `ice_change` is a
/// list of terms (IceChangeTerm), each an object holding exactly one of
/// `thickness` and `change` (file inputs, in the static problem with
/// `date_kyr_bp`, required of `thickness`), `cap_thickness` (`latitude`,
/// `longitude`, `width_deg`, `thickness_m`) and `cap_fraction`
/// (`latitude`, `longitude`, `width_deg`, `fraction`), a cap in a run
/// through time with an optional `history_kyr_bp` beside it; without an
/// ocean the member is refused, its water having nowhere to go. `load` is
/// a list of load terms as `response`'s (ReadForcing), each with
/// `water_budget` and `moves_solid_surface` (true or false; required with
/// an ocean, false by default without one) and, where it moves the solid
/// surface, `density_kg_m3` (positive; refused otherwise); `tide` a list
/// of tide coefficients and `seismic` a list of sources, as `response`'s.
/// At least one term of the lists is required. Their times are given as
/// the run's `clock` says (TermClock; absent, the static problem's, which
/// has none). In the static problem and a DATED run they are dates, kyr
/// before present, as the window's are: a history is `history_kyr_bp`,
/// pairs `[date_kyr_bp, factor]` from the oldest (the dates strictly
/// decreasing), an onset `onset_kyr_bp`; a `field` file with a time axis
/// is read at the term's `date_kyr_bp` in the static problem (or at
/// `date` in its `unit`, on the file's own axis) and at each node's date
/// in a run through time, as the ice change's files are. In a run
/// COUNTING FORWARDS they are times on the run's own clock in its unit: a
/// history is `history`, pairs `[time, factor]` (the times strictly
/// increasing), an onset `onset`, and a `field` file with a time axis is
/// read at t - `time_origin` (default 0) at the clock's time t. The names
/// of the other forms (`response`'s `history_kyr`, `onset_kyr`,
/// `date_kyr`, `time_origin_kyr` among them) are refused, naming this
/// form's, and so are the members of the other problem: a history or an
/// onset in the static problem, a term's date in a run through time. A
/// `thickness` term of the ice change, a dated ice history, is refused in
/// a run counting forwards. So are `ice`, naming `ocean.ice` and
/// `ice_change`, and `sea_level` and `state`, naming the `ocean` block's
/// members.
[[nodiscard]] SeaLevelForcing ReadSeaLevelForcing(
    const Json& block, const std::filesystem::path& directory, int lmax,
    const std::optional<TermClock>& clock, bool ocean);

/// The tide's coefficients on `grid` at time `t` of the terms' clock,
/// m^2 s^-2: each coefficient times its amplitude history's factor there,
/// summed.
[[nodiscard]] std::vector<std::complex<double>> TideAt(
    std::span<const TideTerm> tide, const SurfaceGrid& grid, double t);

/// One line describing a term of an ice change, for provenance: its kind,
/// its file and variable and when they are read, or its cap.
[[nodiscard]] std::string Describe(const IceChangeTerm& term);

/// One line describing a surface load, for provenance: its term, its
/// place in the water budget and, of a layer, its density.
[[nodiscard]] std::string Describe(const SurfaceLoadTerm& term);

/// The surface loads of a `sea-level` run on its grid, at any time of the
/// terms' clock (TermClock), by their properties.
class SurfaceLoads {
 public:
  /// Reads the terms' files onto `grid`; `block` is the config's `forcing`
  /// block (TimedForcing). It and `grid` must outlive the loads. Without
  /// `timed` the loads are the static problem's, the same at every time.
  SurfaceLoads(const SeaLevelForcing& forcing, const SurfaceGrid& grid,
               const Json& block, bool timed = true);

  /// The loads at one time, SI, at the grid's points and band-limited to
  /// its degree; a kind no term has is empty.
  struct Node {
    std::vector<double> inside;   ///< in the water budget, kg m^-2
    std::vector<double> outside;  ///< outside it, kg m^-2
    std::vector<double> layer;    ///< the layer's thickness, m
  };
  [[nodiscard]] Node At(double t) const;

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
