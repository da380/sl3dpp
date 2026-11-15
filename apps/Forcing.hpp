#pragma once

/**
 * @file Forcing.hpp
 * @brief The `forcing` block of the configs: what happens to the state
 * (the sea level and the ice on the body at one instant, the `ocean`
 * block's, Blocks.hpp). The ice change is a sum of terms (IceChangeTerm);
 * surface loads (SurfaceLoadTerm) are load terms of surface mass density
 * that each state their place in the water budget and whether they move
 * the solid surface; a tide is a sum of coefficients of an external
 * potential (TideTerm); and seismic sources are point moment tensors,
 * planar faults and the slip of fault files inside the body
 * (SeismicTerm). Without an `ocean` block
 * the body has no ocean and no ice to change.
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
 * The terms' times take the form of the run's (TermClock): dates, kyr
 * before present, in the static problem and a dated run, and times on the
 * run's own clock in its unit in a run counting forwards. In the static
 * problem a `field` term's file with a time axis is read at the term's
 * date (`date_kyr_bp`, or `date` with its `unit` on the file's own axis).
 * In the quasi-static problem (a run stepped through time) a term may
 * carry an amplitude history (`history_kyr_bp` or `history`), the
 * piecewise-linear factor its pattern is scaled by at each time, and a
 * `field` term may come from a file with a time axis, read at each time by
 * the file's own time rule (TimedLoads). Coefficients throughout are the
 * grid's (SurfaceGrid.hpp): complex, orthonormal, with the Condon-Shortley
 * phase, Y_lm = sqrt((2l+1)/4pi) d^l_0m(theta) e^{i m phi}, the m >= 0 ones
 * of a real field (f_(l,-m) = (-1)^m conj(f_lm)), in SI.
 */

#include <complex>
#include <filesystem>
#include <functional>
#include <memory>
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
/// - Dated: dates in kyr before present (`history_kyr_bp`,
///   `onset_kyr_bp`, a static field's `date_kyr_bp`; a field file read by
///   date through time), turned to kyr from the window's start, the date
///   of the clock's zero, which a run through time has (`start_kyr_bp`).
///   The static problem's terms are of this form, without a start; its
///   `field` terms may instead give `date` with its `unit` on the file's
///   own axis.
/// - Forwards: times on the run's own clock in its `unit` (`history`,
///   `onset`, a field's `time_origin`), taken as they stand.
///
/// Times in kyr on a run's own clock without a unit (`history_kyr`,
/// `onset_kyr`, `time_origin_kyr`, a field's `date_kyr`) are no form's,
/// and are refused naming the member to give.
struct TermClock {
  enum class Form { Dated, Forwards };
  Form form = Form::Dated;
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
  /// (`history_kyr_bp`, a dated run), the date of the clock's zero, kyr BP:
  /// the times above are then this less the dates, in kyr.
  std::optional<double> start_kyr_bp;
  /// Where it was given on a run's own clock in that run's unit
  /// (`history`, a run counting forwards), the unit; absent for dates.
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
  /// static problem, the date its file is read at (`date_kyr_bp`, kyr
  /// before present, or `date`, in `unit` on the file's own axis),
  /// required of a file with a time axis.
  FileInput field;
  std::filesystem::path file;  ///< a coefficient table
  /// The quasi-static problem: the time of the terms' clock at which a
  /// field file's time zero falls, in the clock's unit: the file is read at
  /// (t - time_origin) units, in seconds, at the clock's time t.
  double time_origin = 0.0;
  /// The unit of the terms' clock (kyr in the static problem and a dated
  /// run, the run's unit in one counting forwards) and of a static `date`
  /// on the file's own axis.
  TimeUnit unit;
  /// Whether the static problem's date is before present (`date_kyr_bp`):
  /// the file is then read at minus it on its axis.
  bool before_present = false;
  /// The member the static problem's date is given by (`date_kyr_bp` or
  /// `date`), which a refusal names.
  std::string_view date_member = "date_kyr_bp";
  /// Whether the file is read by date through time (a dated run): its
  /// time axis is the calendar's, read at each node's date, and the origin
  /// above is the window's start.
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

/// A fault file's fault as read and, once the Earth is built, its point
/// sources (PlaceFaultFiles).
struct FaultFileState;

/// One seismic source: a point moment tensor, a planar fault of uniform
/// slip laid out as point sources, or a fault file's slip laid out as point
/// sources at a quadrature's points (Fault.hpp). The moment is the stress
/// glut itself: in a run through time it acts from its onset (the run's
/// start when absent) and is held, times its amplitude history where one
/// is given; in the static problem it acts whole. A fault file whose slip
/// is on a time axis (`node`) is instead the slip its own rule reads at
/// each time. SI throughout: metres and N m.
struct SeismicTerm {
  enum class Kind { Point, Fault, FaultFile };
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
  /// A fault file (`fault_file`): the file; the quadrature, cells along
  /// strike and down dip on every patch and Gauss-Legendre points each way
  /// in a cell; how it is placed where its reference radius is not the
  /// model's surface radius; and, for slip on a time axis, the time of the
  /// terms' clock at which the file's time zero falls, in the clock's unit
  /// (the window's start in a dated run, which reads the file by date), and
  /// what a read outside the file's span does. The file is read, and its
  /// sources formed, when the term is placed in the model (PlaceFaultFiles):
  /// `fault`'s, null before.
  std::filesystem::path file;
  int cellsAlongStrike = 0, cellsDownDip = 0, points = 1;
  std::optional<FaultPlacement> placement;
  double time_origin = 0.0;
  bool dated = false;
  OutsideSpan outside = OutsideSpan::Refuse;
  /// The terms' clock of a run through time; absent in the static problem.
  std::optional<TermClock> clock;
  std::shared_ptr<FaultFileState> fault;
  /// The onset on the terms' clock (TermClock), in its unit
  /// (`onset_kyr_bp` turned to it, or `onset`); absent: from the run's
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
  /// Its left limit at t: zero up to the onset, the onset included (the
  /// term absent just before it), FactorAt's after. The two differ at the
  /// onset alone, where the term jumps on.
  [[nodiscard]] double FactorBefore(double t) const;
  /// Whether the term's onset falls at time t of the terms' clock, to the
  /// tolerance FactorAt reads it with.
  [[nodiscard]] bool OnsetAt(double t) const;
};

/// The point sources of seismic terms at time t of the terms' clock, SI
/// (radius in metres from the centre, moments in N m), for a surface of
/// radius `surfaceRadius` metres. Every term gives its sources at every
/// time, at its points (SeismicPointsOf), with zero moment before its
/// onset. A fault file's are its placed sources (PlaceFaultFiles): a
/// static file's slip times the term's factor; a file's slip on a time
/// axis, at each point the moments of its slices weighted as the file's
/// own rule reads them at the file's time (t - time_origin) in the term's
/// unit: the sum over the slices of each slice's tensors times its hat
/// function (`linear`: piecewise linear between the file's times, one at
/// its own and zero at the others; `previous`: one from its time to the
/// next), which is the moment of the slip the file reads at every time.
/// With `before`, their left limits at t: a term absent at its onset
/// (SeismicTerm::FactorBefore), every other as at t, its time dependence
/// continuous there.
[[nodiscard]] std::vector<PointSource> SeismicSourcesAt(
    std::span<const SeismicTerm> seismic, double t, double surfaceRadius,
    bool before = false);

/// A seismic term's points, SI (radius in metres): a point source's one, a
/// planar fault's patches' centres within the body (PlanarFaultSources), a
/// fault file's quadrature points as placed, patch by patch.
[[nodiscard]] std::vector<BodyPoint> SeismicPointsOf(const SeismicTerm& term,
                                                     double surfaceRadius);

/// Places the fault files of seismic terms in a model of surface radius
/// `surfaceRadius` (m), its elastic moduli at a point `moduli` (SI): reads
/// each file whole (planetmodel::ReadFaultFile), lays its quadrature and
/// forms its sources (FaultSourcesOf), and holds a file's slip on a time
/// axis to its span at every time of the terms' clock in `times` (the
/// run's nodes there). `block` is the config's `forcing` block, where a
/// fault is reported (at `seismic[k]`): a term the build cannot read (its
/// planetmodel has no fault file), a file that cannot be read or is in a
/// Cartesian frame, a static file given a time origin or `outside_span`, a
/// file on a time axis in the static problem or beside an onset or a
/// history, a reference radius not the model's surface radius without
/// `place_by`, a point above the surface or in a fluid, and a node's time
/// outside the span without `"outside_span": "hold"` are refused there.
void PlaceFaultFiles(std::span<SeismicTerm> seismic, double surfaceRadius,
                     const std::function<LoveModuli(const BodyPoint&)>& moduli,
                     const Json& block, std::span<const double> times);

/// The load terms on a run's grid at one time, SI.
struct BuiltLoads {
  /// Each load term's coefficients, kg m^-2, in the order given.
  std::vector<std::vector<std::complex<double>>> terms;
  /// The load at the grid's points, kg m^-2: the sum of each term's field
  /// whose expansion is that term's coefficients — a cap's or a field's
  /// values as sampled, a disk's or a table's synthesised from its
  /// coefficients.
  std::vector<double> load;
};

/// Load terms on a run's grid at any time of the terms' clock (TermClock):
/// each term's pattern times its amplitude history. The patterns are
/// built once, but for a `field` term whose file has a time axis in a run
/// through time: that file is read at each time asked, at file time
/// (t - time_origin) in the term's unit, in seconds, by the file's own time
/// rule (a `node` field interpolated between its nodes), and expanded
/// there. A time outside the field's span is refused, or, where the term
/// says `"outside_span": "hold"`, its end values are held (ReadValues).
class TimedLoads {
 public:
  /// Reads the terms' files onto `grid`. `block` is the config's `forcing`
  /// block, where a fault in a term's file is reported (at
  /// `load[k]`), then and at any time asked; it and `grid` must outlive
  /// the loads. Without `timed` (a static problem's terms) each term is
  /// read once, a field file at its date (refused without one, or with one
  /// but no time axis), and the loads are the same at every time. A field
  /// file that cannot be read, lacks the variable or does not cover the
  /// grid, or a coefficient table that is malformed or exceeds the grid's
  /// degree, is refused.
  TimedLoads(std::vector<LoadTerm> terms, const SurfaceGrid& grid,
             const Json& block, bool timed = true);

  /// The terms at time t of the terms' clock, SI, each times its factor.
  [[nodiscard]] BuiltLoads At(double t) const;

  /// Refuses, as At would, a time of the terms' clock outside the span of
  /// a `field` term's file read through time, before the run reads it.
  void CheckCovers(std::span<const double> times) const;

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
  std::vector<LoadTerm> terms_;
  const SurfaceGrid* grid_;
  const Json* block_;
  std::vector<Pattern> patterns_;
  std::vector<std::string> inputs_;
};

/// A `field` term's input as read, for a summary line and the outputs'
/// metadata: `field <file>:<variable> (<slice>)`, the slice `static` for a
/// file without a time axis, `<d> kyr BP` or `<t> <unit>` for the static
/// problem's date, and in a run through time `each node's date`, `the
/// run's clock` or `time origin <t> <unit>`.
[[nodiscard]] std::string DescribeField(const LoadTerm& term, bool timeAxis,
                                        bool timed);

/// The time on its file's own axis, seconds, at which the static problem
/// reads a `field` term's file: minus `date_kyr_bp` in kyr, or `date` in
/// its `unit`, zero without either (a file without a time axis), the
/// present one zero of either sign. The one resolved date the read and
/// the run's hash both take, so that a date before present and the same
/// date on the file's axis in kyr (its negation, exact) read and hash
/// alike; in another unit the product may differ in the last bit, and the
/// two are then two runs.
[[nodiscard]] double StaticReadTimeOf(const LoadTerm& term);

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

/// One line describing a load term, a tide coefficient or a seismic
/// source, for provenance: its pattern (a fault file's file, its patches
/// and time axis, its quadrature and placement) and, where set, its time
/// origin, onset and amplitude history.
[[nodiscard]] std::string Describe(const LoadTerm& term);
[[nodiscard]] std::string Describe(const TideTerm& term);
[[nodiscard]] std::string Describe(const SeismicTerm& term);

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

/// A surface load: a load term (a surface mass density) with the two
/// properties the sea level equation needs of it. A load in the WATER
/// BUDGET takes its mass from the ocean (a hydrological load); one outside
/// it moves no water (sediment, any other mass). A load that MOVES THE
/// SOLID SURFACE is a layer laid on it, of thickness its mass over its
/// density: the sea level, measured from the solid surface, falls by the
/// thickness, and under migrating shorelines the shorelines move with it.
/// On a body without an ocean the water budget has no ocean to change, and
/// a layer is its thickness off the relative sea level change alone.
struct SurfaceLoadTerm {
  LoadTerm term;
  bool water_budget = false;
  bool moves_solid_surface = false;
  double density = 0.0;  ///< kg m^-3, of a load that moves the surface
};

/// The `forcing` block of `solve` and `kernel`: what happens to
/// the state (the `ocean` block's, Blocks.hpp). The ice change is the sum
/// of its terms; the ice at a time is the state's plus the change there, a
/// direct load where it is grounded. Surface loads, the tide (a sum of
/// coefficients of an external potential) and seismic sources act beside
/// the ice, changes from the state: in the static problem at its one
/// instant, in a run through time on the terms' clock (TermClock: kyr from
/// the window's start in a dated run, the run's own clock in its unit in
/// one counting forwards), a term present at the first node acting there.
/// A run without an ocean has no ice change.
struct Forcing {
  std::vector<IceChangeTerm> ice_change;
  std::vector<SurfaceLoadTerm> load;
  std::vector<TideTerm> tide;
  std::vector<SeismicTerm> seismic;

  /// Whether the run is forced by more than its ice.
  [[nodiscard]] bool BesideTheIce() const {
    return !load.empty() || !tide.empty() || !seismic.empty();
  }
};

/// Reads a `forcing` block at `lmax`, relative paths against `directory`,
/// of a run with an ocean (`ocean`, the config having an `ocean` block) or
/// without one. `ice_change` is a list of terms (IceChangeTerm), each an
/// object holding exactly one of `thickness` and `change` (file inputs, in
/// the static problem with `date_kyr_bp`, required of `thickness`),
/// `cap_thickness` (`latitude`, `longitude`, `width_deg`, `thickness_m`)
/// and `cap_fraction` (`latitude`, `longitude`, `width_deg`, `fraction`),
/// a cap in a run through time with an optional history beside it;
/// without an ocean the member is refused, its water having nowhere to
/// go. `load` is a list of load terms, each an object holding exactly one
/// of `cap` (`latitude`, `longitude`, `width_deg`, `amplitude_kg_m2`),
/// `disk` (`latitude`, `longitude`, `radius_deg`, `amplitude_kg_m2`),
/// `field` (a file input, its variable by default `load`) and
/// `coefficients` (`file`), with `water_budget` and `moves_solid_surface`
/// (true or false; required with an ocean, false by default without one)
/// and, where it moves the solid surface, `density_kg_m3` (positive;
/// refused otherwise). `tide` is a list of coefficients (`degree` in
/// [2, lmax], `order` in [0, degree], `amplitude_m2_s2`, `imaginary_m2_s2`,
/// zero at order 0), and `seismic` a list of sources, each an object
/// holding exactly one of `point` (`latitude`, `longitude`, `depth_km` and
/// either `moment_tensor_nm` with members `rr`, `tt`, `pp`, `rt`, `rp`,
/// `tp`, or `strike_deg`, `dip_deg`, `rake_deg` and `moment_nm`) and
/// `fault` (`latitude`, `longitude`, `depth_km` of its centroid,
/// `strike_deg`, `dip_deg`, `rake_deg`, `length_km`, `width_km`,
/// `moment_nm`, and optionally `patches_along_strike` and
/// `patches_down_dip`) and `fault_file` (`file`, a planetmodel fault file
/// read when it is placed, PlaceFaultFiles; `cells_along_strike`,
/// `cells_down_dip`, optionally `points` and `place_by` (`depth` or
/// `radius`), and for slip on a time axis `time_origin`, refused in a
/// dated run, and `outside_span`). At least one term of the lists is
/// required. Their
/// times are given as the run's `clock` says (TermClock; absent, the
/// static problem's, which has none). In the static problem and a DATED
/// run they are dates, kyr before present, as the window's are: a history
/// is `history_kyr_bp`, pairs `[date_kyr_bp, factor]` from the oldest (the
/// dates strictly decreasing), an onset `onset_kyr_bp`; a `field` file
/// with a time axis is read at the term's `date_kyr_bp` in the static
/// problem (or at `date` in its `unit`, on the file's own axis) and at
/// each node's date in a run through time, as the ice change's files are.
/// In a run COUNTING FORWARDS they are times on the run's own clock in its
/// unit: a history is `history`, pairs `[time, factor]` (the times strictly
/// increasing), an onset `onset`, and a `field` file with a time axis is
/// read at t - `time_origin` (default 0) at the clock's time t. The names
/// of the other form are refused, naming this form's, and so are times in
/// kyr without a unit (`history_kyr`, `onset_kyr`, `time_origin_kyr`, a
/// field's `date_kyr`), and the members of the other problem: a history or
/// an onset in the static problem, a term's date in a run through time. A
/// `thickness` term of the ice change, a dated ice history, is refused in
/// a run counting forwards. So are `ice`, naming `ocean.ice` and
/// `ice_change`, and `sea_level` and `state`, naming the `ocean` block's
/// members.
[[nodiscard]] Forcing ReadForcing(const Json& block,
                                  const std::filesystem::path& directory,
                                  int lmax,
                                  const std::optional<TermClock>& clock,
                                  bool ocean);

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

/// What forces a run beside its ice, as `load (2 terms) + tide (1
/// coefficient) + seismic (1 source)`; empty without any.
[[nodiscard]] std::string Summary(const Forcing& forcing);

/// The surface loads of a run on its grid, at any time of the terms' clock
/// (TermClock), by their properties.
class SurfaceLoads {
 public:
  /// Reads the terms' files onto `grid`; `block` is the config's `forcing`
  /// block (TimedLoads). It and `grid` must outlive the loads. Without
  /// `timed` the loads are the static problem's, the same at every time.
  SurfaceLoads(const Forcing& forcing, const SurfaceGrid& grid,
               const Json& block, bool timed = true);

  /// The loads at one time, SI, at the grid's points: by their
  /// properties, band-limited to its degree (a kind no term has is empty),
  /// and every term summed as given (BuiltLoads::load: a cap's or a
  /// field's values as sampled, a disk's or a table's synthesised from its
  /// coefficients; empty without terms).
  struct Node {
    std::vector<double> inside;   ///< in the water budget, kg m^-2
    std::vector<double> outside;  ///< outside it, kg m^-2
    std::vector<double> layer;    ///< the layer's thickness, m
    std::vector<double> given;    ///< every term summed as given, kg m^-2
  };
  [[nodiscard]] Node At(double t) const;

  /// Refuses a time outside the span of a `field` term's file read
  /// through time (TimedLoads::CheckCovers).
  void CheckCovers(std::span<const double> times) const;

  /// Each `field` term's input as read (DescribeField).
  [[nodiscard]] const std::vector<std::string>& Inputs() const {
    return timed_.Inputs();
  }

 private:
  std::vector<SurfaceLoadTerm> terms_;
  const SurfaceGrid* grid_;
  TimedLoads timed_;
};

}  // namespace sl3dpp::app
