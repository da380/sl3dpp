#include "Forcing.hpp"

#include <GSHTrans/All>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "sl3dpp/io/FieldHistory.hpp"

#include <planetmodel/version.hpp>
#include <planetmodel/faultfile.hpp>

namespace sl3dpp::app {

struct FaultFileState {
  planetmodel::Fault fault;
  /// The sources, once placed (PlaceFaultFiles).
  FaultSources sources{};
  bool placed = false;
};

namespace {

using Complex = std::complex<double>;

constexpr auto kTermKinds =
    std::array<std::string_view, 4>{"cap", "disk", "field", "coefficients"};

using Form = TermClock::Form;

/// The names a form of the terms' clock (TermClock) gives a history and an
/// onset.
struct ClockNames {
  std::string_view history;
  std::string_view onset;
};

ClockNames NamesOf(Form form) {
  return form == Form::Dated ? ClockNames{"history_kyr_bp", "onset_kyr_bp"}
                             : ClockNames{"history", "onset"};
}

/// The names of a history and an onset in kyr on a run's own clock
/// without a unit, which no form takes.
constexpr auto kKyrNames = ClockNames{"history_kyr", "onset_kyr"};

/// Why a name of the other form is refused in a term of the form `mine`
/// (in a run through time, `timed`), naming this form's `wanted`.
std::string OtherClock(Form mine, std::string_view wanted, bool timed) {
  if (mine == Form::Dated) {
    return timed ? std::format("is a time on a run's own clock, the form of a "
                               "`times` block counting forwards with a "
                               "`unit`; this run's times are dates, kyr "
                               "before present: `{}`",
                               wanted)
                 : std::format("is a time on a run's own clock, which {}, "
                               "its `times` counting forwards with a `unit`",
                               kNeedsQuasiStatic);
  }
  return std::format(
      "is a date before present, the dated form of `times`; this run counts "
      "forwards on its own clock, in its `unit` (`times.unit`): `{}`",
      wanted);
}

/// Why a time in kyr on a run's own clock without a unit is refused in a
/// term of the form `mine`, naming this form's `wanted`.
std::string KyrWithoutUnit(Form mine, std::string_view wanted) {
  return mine == Form::Dated
             ? std::format("is a time in kyr on a run's own clock, without "
                           "its unit; here times are dates, kyr before "
                           "present: `{}`",
                           wanted)
             : std::format("is a time in kyr without its unit; this run "
                           "counts forwards on its own clock, in its `unit` "
                           "(`times.unit`): `{}`",
                           wanted);
}

/// Refuses the other form's names of a history (and, with `onsets`, of an
/// onset) in a term, and the names of times in kyr without a unit, naming
/// this form's.
void RefuseOtherClock(const Json& item, const TermClock& clock, bool timed,
                      bool onsets = true) {
  const auto mine = NamesOf(clock.form);
  const auto other =
      NamesOf(clock.form == Form::Dated ? Form::Forwards : Form::Dated);
  if (Has(item, kKyrNames.history)) {
    Fail(item.At(kKyrNames.history), KyrWithoutUnit(clock.form, mine.history));
  }
  if (Has(item, other.history)) {
    Fail(item.At(other.history), OtherClock(clock.form, mine.history, timed));
  }
  if (!onsets) return;
  if (Has(item, kKyrNames.onset)) {
    Fail(item.At(kKyrNames.onset), KyrWithoutUnit(clock.form, mine.onset));
  }
  if (Has(item, other.onset)) {
    Fail(item.At(other.onset), OtherClock(clock.form, mine.onset, timed));
  }
}

/// A term's amplitude history under its form's name (`history_kyr_bp` or
/// `history`): refused without a `times` block (`timed`), which a forcing
/// fixed in time cannot take.
AmplitudeHistory ReadHistory(const Json& item, bool timed,
                             const TermClock& clock = {}) {
  auto out = AmplitudeHistory{};
  RefuseOtherClock(item, clock, timed);
  const auto name = NamesOf(clock.form).history;
  if (!Has(item, name)) return out;
  const auto& history = item.At(name);
  if (!timed) {
    Fail(history,
         std::format("is an amplitude history, which {}", kNeedsQuasiStatic));
  }
  if (clock.form == Form::Dated) {
    const auto start = clock.start_kyr_bp.value();
    out.start_kyr_bp = start;
    for (const auto& pair : history.AsArray()) {
      if (!pair.IsArray() || pair.AsArray().size() != 2) {
        Fail(pair, "is not a pair [date_kyr_bp, factor]");
      }
      const auto date = pair.At(std::size_t{0}).AsNumber();
      const auto factor = pair.At(std::size_t{1}).AsNumber();
      if (!std::isfinite(date) || !std::isfinite(factor)) {
        Fail(pair, "is not a pair of finite numbers [date_kyr_bp, factor]");
      }
      const auto time = start - date;
      if (!out.times.empty() && !(time > out.times.back())) {
        Fail(pair.At(std::size_t{0}),
             "is not later than the date before it: a history's dates run "
             "from the oldest, strictly decreasing in kyr BP");
      }
      out.times.push_back(time);
      out.factors.push_back(factor);
    }
    if (out.times.empty()) {
      Fail(history, "holds no [date_kyr_bp, factor] pair");
    }
    return out;
  }
  // on the run's own clock, in its unit
  constexpr auto pairName = std::string_view{"[time, factor]"};
  out.unit = clock.unit;
  for (const auto& pair : history.AsArray()) {
    if (!pair.IsArray() || pair.AsArray().size() != 2) {
      Fail(pair, std::format("is not a pair {}", pairName));
    }
    const auto time = pair.At(std::size_t{0}).AsNumber();
    const auto factor = pair.At(std::size_t{1}).AsNumber();
    if (!std::isfinite(time) || !std::isfinite(factor)) {
      Fail(pair, std::format("is not a pair of finite numbers {}", pairName));
    }
    if (!out.times.empty() && !(time > out.times.back())) {
      Fail(pair.At(std::size_t{0}),
           "is not later than the time before it: a history's times are "
           "strictly increasing");
    }
    out.times.push_back(time);
    out.factors.push_back(factor);
  }
  if (out.times.empty()) {
    Fail(history, std::format("holds no {} pair", pairName));
  }
  return out;
}

/// A centre's latitude and longitude, degrees.
GeographicPoint SiteOf(const Json& object) { return ReadSite(object); }

/// `extra`: further members the item may hold, which the caller reads.
LoadTerm ReadLoadTerm(const Json& item, const std::filesystem::path& directory,
                      bool timed, std::span<const std::string_view> extra = {},
                      const TermClock& clock = {}) {
  auto history = ReadHistory(item, timed, clock);
  auto members =
      std::vector<std::string_view>(kTermKinds.begin(), kTermKinds.end());
  members.push_back(NamesOf(clock.form).history);
  members.insert(members.end(), extra.begin(), extra.end());
  Allow(item, std::span(members));
  const auto kinds = std::ranges::count_if(
      kTermKinds, [&item](std::string_view kind) { return Has(item, kind); });
  if (kinds != 1) {
    Fail(item,
         std::format("{}: a load term is exactly one of \"cap\", "
                     "\"disk\", \"field\" and \"coefficients\"",
                     kinds == 0 ? "names no term" : "names several terms"));
  }
  auto term = LoadTerm{};
  if (Has(item, "cap")) {
    const auto& cap = item.At("cap");
    Allow(cap, {"latitude", "longitude", "width_deg", "amplitude_kg_m2"});
    term.kind = LoadTerm::Kind::Cap;
    term.site = SiteOf(cap);
    const auto width = Number(cap, "width_deg");
    if (!(width > 0.0)) Fail(cap.At("width_deg"), "is not a positive angle");
    term.width = width * std::numbers::pi / 180.0;
    term.amplitude = Number(cap, "amplitude_kg_m2");
  } else if (Has(item, "disk")) {
    const auto& disk = item.At("disk");
    Allow(disk, {"latitude", "longitude", "radius_deg", "amplitude_kg_m2"});
    term.kind = LoadTerm::Kind::Disk;
    term.site = SiteOf(disk);
    const auto radius = Number(disk, "radius_deg");
    if (!(radius > 0.0 && radius <= 180.0)) {
      Fail(disk.At("radius_deg"), "is not an angle in (0, 180]");
    }
    term.radius = radius * std::numbers::pi / 180.0;
    term.amplitude = Number(disk, "amplitude_kg_m2");
  } else if (Has(item, "field")) {
    const auto& field = item.At("field");
    if (Has(field, "name")) {
      Fail(field.At("name"),
           "is not a member: the field read from the file is its `variable`");
    }
    term.kind = LoadTerm::Kind::Field;
    term.unit = clock.unit;
    if (clock.form == Form::Dated) {
      // the dates are before present: the run's clock's names refused,
      // and the static problem's date `date_kyr_bp`, or `date` with its
      // `unit` on the file's own axis
      if (Has(field, "time_origin_kyr")) {
        Fail(field.At("time_origin_kyr"),
             timed ? "is a time origin on a run's own clock; here a file with "
                     "a time axis is read by date, at each node's, as the ice "
                     "change's files are"
                   : "is a time origin on a run's own clock; here a file "
                     "with a time axis is read at its date, `date_kyr_bp`");
      }
      if (Has(field, "time_origin")) {
        Fail(field.At("time_origin"),
             timed ? std::string{"is a time origin on a run's own clock, the "
                                 "form of a `times` block counting forwards "
                                 "with a `unit`; here a file with a time axis "
                                 "is read by date, at each node's, as the ice "
                                 "change's files are"}
                   : std::format("is a time origin on a run's own clock, "
                                 "which {}, its `times` counting forwards "
                                 "with a `unit`; the static problem reads a "
                                 "file with a time axis at its date, "
                                 "`date_kyr_bp`, or at `date` in its `unit` "
                                 "on the file's own axis",
                                 kNeedsQuasiStatic));
      }
      if (timed) {
        for (const auto* name : {"date_kyr_bp", "date", "date_kyr"}) {
          if (Has(field, name)) {
            Fail(field.At(name),
                 "dates the file, as the static problem reads it; a run "
                 "through time reads it at each node's date");
          }
        }
        if (Has(field, "unit")) {
          Fail(field.At("unit"),
               "is the unit of the static problem's `date`; a run through "
               "time reads the file at each node's date");
        }
        AllowFileInput(field);
        term.field = ReadFileInput(field, directory, "load");
        term.before_present = true;
        // the file's time zero is the present: the window's start on the
        // clock
        term.dated = true;
        term.time_origin = clock.start_kyr_bp.value();
      } else {
        if (Has(field, "date_kyr")) {
          Fail(field.At("date_kyr"),
               "is a time in kyr on the file's own axis, without its unit: "
               "give `date` with its `unit` (\"unit\": \"kyr\"), or the "
               "date before present, `date_kyr_bp`");
        }
        AllowFileInput(field, {"date_kyr_bp", "date", "unit"});
        term.field = ReadFileInput(field, directory, "load");
        if (Has(field, "date")) {
          // the instant on the file's own axis
          if (Has(field, "date_kyr_bp")) {
            Fail(field.At("date"),
                 "dates the file twice, beside `date_kyr_bp`: give the date "
                 "before present, `date_kyr_bp`, or the time on the file's "
                 "own axis, `date` with its `unit`");
          }
          if (!Has(field, "unit")) {
            Fail(field.At("date"),
                 std::format("is a time on the file's own axis, which needs "
                             "its `unit` beside it: {}",
                             kTimeUnits));
          }
          term.unit = ReadTimeUnit(field.At("unit"));
          term.date_member = "date";
          term.field.date = ReadDate(field, "date");
        } else {
          if (Has(field, "unit")) {
            Fail(field.At("unit"),
                 "is the unit of a `date` on the file's own axis, which the "
                 "field does not give");
          }
          term.before_present = true;
          term.date_member = "date_kyr_bp";
          term.field.date = ReadDate(field, "date_kyr_bp");
        }
      }
    } else {
      // times on the run's own clock in its unit: the file read at
      // t - `time_origin`
      for (const auto* name : {"date_kyr_bp", "date_kyr", "date"}) {
        if (Has(field, name)) {
          Fail(field.At(name),
               "dates the file, as the static problem reads it; a run "
               "through time reads it at each time t of its clock, at t - "
               "`time_origin`");
        }
      }
      if (Has(field, "unit")) {
        Fail(field.At("unit"),
             "is the unit of the static problem's `date`; this run's times "
             "are in its own unit, `times.unit`");
      }
      if (Has(field, "time_origin_kyr")) {
        Fail(field.At("time_origin_kyr"),
             "is a time origin in kyr without its unit; this run counts "
             "forwards on its own clock, in its `unit` (`times.unit`): "
             "`time_origin`");
      }
      AllowFileInput(field, {"time_origin"});
      term.field = ReadFileInput(field, directory, "load");
      if (Has(field, "time_origin")) {
        const auto& origin = field.At("time_origin");
        term.time_origin = origin.AsNumber();
        if (!std::isfinite(term.time_origin)) {
          Fail(origin, "is not a finite number");
        }
      }
    }
  } else {
    const auto& coefficients = item.At("coefficients");
    Allow(coefficients, {"file"});
    term.kind = LoadTerm::Kind::Coefficients;
    term.file = PathOf(coefficients, "file", directory);
  }
  term.history = std::move(history);
  return term;
}

TideTerm ReadTideTerm(const Json& item, int lmax, bool timed,
                      const TermClock& clock = {}) {
  auto history = ReadHistory(item, timed, clock);
  Allow(item, {"degree", "order", "amplitude_m2_s2", "imaginary_m2_s2",
               NamesOf(clock.form).history});
  auto term = TideTerm{};
  term.history = std::move(history);
  term.degree = Integer(item, "degree");
  if (term.degree < 2 || term.degree > lmax) {
    Fail(item.At("degree"),
         "is not a tidal degree within [2, lmax] (a tide is taken relative "
         "to the centre of mass: no degree 0 or 1)");
  }
  term.order = Integer(item, "order", 0);
  if (term.order < 0 || term.order > term.degree) {
    Fail(item.At("order"), "is not an order within [0, degree]");
  }
  term.amplitude = Number(item, "amplitude_m2_s2");
  term.imaginary = Number(item, "imaginary_m2_s2", 0.0);
  if (term.order == 0 && term.imaginary != 0.0) {
    Fail(item.At("imaginary_m2_s2"), "is not zero at order 0 (a real field)");
  }
  return term;
}

/// A seismic term's `fault_file`: the file's path, the quadrature, the
/// placement and, for slip on a time axis, the time origin (a run counting
/// forwards; refused in a dated run, which reads such a file by date) and
/// what a read outside the span does. The file itself is read, and held to
/// the rest of the config, when it is placed (PlaceFaultFiles); `timed`, a
/// run through time's, on the terms' `clock`.
void ReadFaultFileTerm(SeismicTerm& term, const Json& item,
                       const std::filesystem::path& directory, bool timed,
                       const TermClock& clock) {
  const auto& member = item.At("fault_file");
  if (!member.IsObject()) {
    Fail(member,
         "is not a fault file input: an object of `file`, "
         "`cells_along_strike` and `cells_down_dip`");
  }
  Allow(member, {"file", "cells_along_strike", "cells_down_dip", "points",
                 "place_by", "time_origin", "outside_span"});
  term.kind = SeismicTerm::Kind::FaultFile;
  term.file = PathOf(member, "file", directory);
  term.cellsAlongStrike = Integer(member, "cells_along_strike");
  term.cellsDownDip = Integer(member, "cells_down_dip");
  term.points = Integer(member, "points", 1);
  for (const auto& [name, count] :
       {std::pair{"cells_along_strike", term.cellsAlongStrike},
        std::pair{"cells_down_dip", term.cellsDownDip},
        std::pair{"points", term.points}}) {
    if (count < 1) Fail(member.At(name), "is not a positive count");
  }
  if (Has(member, "place_by")) {
    term.placement = Choice<FaultPlacement>(
        member.At("place_by"),
        {{"depth", FaultPlacement::Depth}, {"radius", FaultPlacement::Radius}});
  }
  if (Has(member, "outside_span")) {
    term.outside = Choice<OutsideSpan>(
        member.At("outside_span"),
        {{"refuse", OutsideSpan::Refuse}, {"hold", OutsideSpan::Hold}});
  }
  if (timed) term.clock = clock;
  if (Has(member, "time_origin")) {
    if (timed && clock.form == Form::Dated) {
      Fail(member.At("time_origin"),
           "is a time origin on a run's own clock, the form of a `times` "
           "block counting forwards with a `unit`; a dated run reads a fault "
           "file with a time axis by date, at each node's, its axis the "
           "calendar's (the present at zero, the past negative), as it reads "
           "a `field` load's file");
    }
    term.time_origin = member.At("time_origin").AsNumber();
    if (!std::isfinite(term.time_origin)) {
      Fail(member.At("time_origin"), "is not a finite number");
    }
  }
  if (timed && clock.form == Form::Dated) {
    // a file on a time axis is read by date: its time zero is the present,
    // the window's start on the clock, in kyr
    term.dated = true;
    term.time_origin = clock.start_kyr_bp.value();
  }
}


/// A fault file term's file, read whole and checked as planetmodel reads
/// it, and held to the term's members and the run (`item`, the term in
/// the config): a Cartesian frame is refused; a static file refuses a
/// time origin and `outside_span`; a file on a time axis refuses the
/// static problem, an onset and a history.
std::shared_ptr<FaultFileState> ReadFault(const SeismicTerm& term,
                                          const Json& item) {
  const auto& member = item.At("fault_file");
  auto read = [&] {
    try {
      return planetmodel::ReadFaultFile(term.file);
    } catch (const std::exception& error) {
      Fail(member.At("file"), error.what());
    }
  };
  auto file = read();
  const auto& fault = file.fault;
  if (fault.Frame().GetKind() != planetmodel::FaultFrame::Kind::Geographic) {
    Fail(member.At("file"),
         "holds a fault in a Cartesian frame, a flat body: this body is a "
         "sphere, and takes a fault in the geographic frame");
  }
  if (fault.Support() != planetmodel::TimeSupport::Node) {
    for (const auto* name : {"time_origin", "outside_span"}) {
      if (!Has(member, name)) continue;
      Fail(member.At(name),
           std::format("{} of a fault file's time axis, and {} is static, its "
                       "slip the same at every time: drop it",
                       std::string_view{name} == "time_origin"
                           ? "is the time origin"
                           : "says what a read outside the span does",
                       term.file.string()));
    }
  } else {
    const auto times = fault.NSlices();
    if (!term.clock) {
      Fail(member.At("file"),
           std::format("holds slip on a time axis ({} times), which the "
                       "static problem does not read: its source is one slip, "
                       "a static fault file; a run through time reads a file "
                       "on a time axis",
                       times));
    }
    const auto names = NamesOf(term.clock->form);
    for (const auto name : {names.onset, names.history}) {
      if (!Has(item, name)) continue;
      Fail(item.At(name),
           std::format("is {}, beside a fault file whose slip is on a time "
                       "axis of its own ({} times), which is its time "
                       "dependence: drop it",
                       name == names.onset ? "an onset" : "an amplitude history",
                       times));
    }
  }
  return std::make_shared<FaultFileState>(
      FaultFileState{.fault = std::move(file.fault)});
}


/// One seismic source: a point moment tensor, a planar fault or a fault
/// file (`directory` resolving its path); `timed`, a run through time's,
/// which alone takes an onset and a history.
SeismicTerm ReadSeismicTerm(const Json& item,
                            const std::filesystem::path& directory,
                            bool timed = true, const TermClock& clock = {}) {
  RefuseOtherClock(item, clock, timed);
  const auto names = NamesOf(clock.form);
  Allow(item, {"point", "fault", "fault_file", names.onset, names.history});
  const auto kinds = static_cast<int>(Has(item, "point")) +
                     static_cast<int>(Has(item, "fault")) +
                     static_cast<int>(Has(item, "fault_file"));
  if (kinds != 1) {
    Fail(item, "is exactly one of \"point\", \"fault\" and \"fault_file\"");
  }
  auto term = SeismicTerm{};
  if (!timed && Has(item, names.onset)) {
    Fail(item.At(names.onset),
         std::format("is an onset, which {}: the static problem is the "
                     "co-seismic response",
                     kNeedsQuasiStatic));
  }
  term.history = ReadHistory(item, timed, clock);
  term.unit = clock.unit;
  if (Has(item, names.onset)) {
    const auto onset = Number(item, names.onset);
    if (clock.form == Form::Dated) {
      // a date, turned to the clock's kyr from the window's start
      term.start_kyr_bp = clock.start_kyr_bp;
      term.onset = *clock.start_kyr_bp - onset;
    } else {
      term.onset = onset;
    }
  }
  auto planeOf = [&term](const Json& object) {
    term.strike = Number(object, "strike_deg");
    term.dip = Number(object, "dip_deg");
    term.rake = Number(object, "rake_deg");
    if (!(term.dip >= 0.0 && term.dip <= 90.0)) {
      Fail(object.At("dip_deg"), "is not a dip in [0, 90]");
    }
  };
  auto placeOf = [&term](const Json& object) {
    term.site = ReadSite(object);
    term.depth = 1e3 * Number(object, "depth_km");
    if (!(term.depth > 0.0)) {
      Fail(object.At("depth_km"), "is not a depth below the surface");
    }
  };
  if (Has(item, "point")) {
    const auto& point = item.At("point");
    Allow(point, {"latitude", "longitude", "depth_km", "moment_tensor_nm",
                  "strike_deg", "dip_deg", "rake_deg", "moment_nm"});
    term.kind = SeismicTerm::Kind::Point;
    placeOf(point);
    if (Has(point, "moment_tensor_nm")) {
      for (const auto* name : {"strike_deg", "dip_deg", "rake_deg", "moment_nm"}) {
        if (Has(point, name)) {
          Fail(point.At(name),
               "describes a double couple, beside a moment tensor: give one");
        }
      }
      const auto& tensor = point.At("moment_tensor_nm");
      Allow(tensor, {"rr", "tt", "pp", "rt", "rp", "tp"});
      term.moment = {.rr = Number(tensor, "rr"), .tt = Number(tensor, "tt"),
                     .pp = Number(tensor, "pp"), .rt = Number(tensor, "rt"),
                     .rp = Number(tensor, "rp"), .tp = Number(tensor, "tp")};
    } else {
      planeOf(point);
      term.moment = DoubleCouple(term.strike, term.dip, term.rake);
      const auto moment = Number(point, "moment_nm");
      for (auto* c : {&term.moment.rr, &term.moment.tt, &term.moment.pp,
                      &term.moment.rt, &term.moment.rp, &term.moment.tp}) {
        *c *= moment;
      }
    }
  } else if (Has(item, "fault_file")) {
    ReadFaultFileTerm(term, item, directory, timed, clock);
  } else {
    const auto& fault = item.At("fault");
    Allow(fault, {"latitude", "longitude", "depth_km", "strike_deg", "dip_deg",
                  "rake_deg", "length_km", "width_km", "moment_nm",
                  "patches_along_strike", "patches_down_dip"});
    term.kind = SeismicTerm::Kind::Fault;
    placeOf(fault);
    planeOf(fault);
    term.length = 1e3 * Number(fault, "length_km");
    term.width = 1e3 * Number(fault, "width_km");
    if (!(term.length > 0.0)) Fail(fault.At("length_km"), "is not a positive length");
    if (!(term.width > 0.0)) Fail(fault.At("width_km"), "is not a positive width");
    term.scalarMoment = Number(fault, "moment_nm");
    term.alongStrike = Integer(fault, "patches_along_strike", 1);
    term.downDip = Integer(fault, "patches_down_dip", 1);
    if (term.alongStrike < 1) {
      Fail(fault.At("patches_along_strike"), "is not a positive count");
    }
    if (term.downDip < 1) {
      Fail(fault.At("patches_down_dip"), "is not a positive count");
    }
  }
  return term;
}

/// A field term's file on the grid, holding its variable. Unless `timed`
/// the file is read at one time: a time axis needs the term's date
/// (`date_kyr_bp` before present, or `date` on the file's own axis), and
/// a file without one refuses it.
io::FieldHistory OpenFieldTerm(const LoadTerm& term, const SurfaceGrid& grid,
                               const Json& field, bool timed) {
  auto history = OpenFileInput(term.field, grid, field);
  if (!timed) CheckDate(term.field, history, field, term.date_member);
  return history;
}

/// How a refusal states a time of a field term's file: as a date before
/// present where the term reads its file by date, on the file's own axis
/// in the term's unit otherwise (AxisLabel).
std::optional<TimeUnit> AxisUnitOf(const LoadTerm& term) {
  return term.before_present ? std::nullopt : std::optional{term.unit};
}

/// Where a run through time reads a field term's file with a time axis, as
/// a refusal states it.
std::string_view NodeReadOf(const LoadTerm& term) {
  return term.dated ? "a node's date, where the run reads it"
                    : "a node's time less `time_origin`, where the run "
                      "reads it";
}

/// A field term's values on the grid, SI, at an SI time of its file, read
/// within its time span unless the term holds its ends (ReadValues,
/// `where` the read); `field` is the term's `field` member, and `when`
/// names the run's time in a fault, where it matters.
std::vector<double> FieldValues(const io::FieldHistory& history,
                                const LoadTerm& term, const Json& field,
                                double time, std::string_view where,
                                std::string_view when = {}) {
  auto values = ReadValues(term.field, history, field, time, where,
                           AxisUnitOf(term));
  if (!std::ranges::all_of(values, [](double v) { return std::isfinite(v); })) {
    Fail(field.At("file"),
         std::format("holds a field \"{}\" that is not finite everywhere on "
                     "the run's grid{}",
                     term.field.variable, when));
  }
  return values;
}

/// A load term's pattern on the grid: its field at the grid's points and
/// its coefficients, the one made from the other as the term's kind says.
/// In a run through time (`timed`) a field term whose file has a time axis
/// has no fixed pattern: its file comes back instead, both arrays empty.
/// A field term says whether its file has a time axis.
struct TermPattern {
  std::vector<double> field;
  std::vector<Complex> coefficients;
  std::optional<io::FieldHistory> history;
  bool timeAxis = false;
};

TermPattern PatternOf(const LoadTerm& term, const SurfaceGrid& grid,
                      const Json& item, bool timed) {
  auto out = TermPattern{};
  switch (term.kind) {
    case LoadTerm::Kind::Cap:
      out.field = CapField(grid, term.site, term.width, term.amplitude);
      out.coefficients = grid.Expand(out.field);
      break;
    case LoadTerm::Kind::Disk:
      out.coefficients =
          DiskCoefficients(grid, term.site, term.radius, term.amplitude);
      out.field = grid.Synthesise(out.coefficients);
      break;
    case LoadTerm::Kind::Field: {
      const auto& field = item.At("field");
      auto history = OpenFieldTerm(term, grid, field, timed);
      out.timeAxis = history.Span().has_value();
      if (timed && out.timeAxis) {
        out.history.emplace(std::move(history));
        break;
      }
      // a static run reads a file with a time axis at its date; a static
      // file is the same at every time
      const auto time = timed ? 0.0 : StaticReadTimeOf(term);
      out.field = FieldValues(
          history, term, field, time,
          std::format("its date, `{}`, where it is read", term.date_member));
      out.coefficients = grid.Expand(out.field);
      break;
    }
    case LoadTerm::Kind::Coefficients:
      out.coefficients = ReadCoefficientTable(
          term.file, grid.LMax(), item.At("coefficients").At("file"));
      out.field = grid.Synthesise(out.coefficients);
      break;
  }
  return out;
}

/// "[[t, f], ...]", for provenance.
std::string Pairs(const AmplitudeHistory& history) {
  auto out = std::string{"["};
  for (auto i = std::size_t{0}; i < history.times.size(); ++i) {
    out += std::format("{}[{:g}, {:g}]", i == 0 ? "" : ", ", history.times[i],
                       history.factors[i]);
  }
  return out + "]";
}

/// A history as a term's description ends with it: its pairs as dates
/// where it was given as dates, and on a run's own clock with the clock's
/// unit where it was given on one.
std::string History(const AmplitudeHistory& history) {
  if (history.unit) {
    return std::format(", history in {} {}", history.unit->name,
                       Pairs(history));
  }
  auto dated = history;
  for (auto& t : dated.times) t = history.start_kyr_bp.value_or(0.0) - t;
  return ", history_kyr_bp " + Pairs(dated);
}

/// The words of a line, split at white space.
std::vector<std::string_view> WordsOf(std::string_view line) {
  auto out = std::vector<std::string_view>{};
  auto start = std::string_view::size_type{0};
  while (true) {
    start = line.find_first_not_of(" \t\r", start);
    if (start == std::string_view::npos) break;
    const auto end = line.find_first_of(" \t\r", start);
    out.push_back(line.substr(start, end - start));
    if (end == std::string_view::npos) break;
    start = end;
  }
  return out;
}

/// A whole word read as a number of type T, a leading + allowed; empty
/// when it is not one.
template <class T>
std::optional<T> Parse(std::string_view word) {
  if (word.size() > 1 && word.front() == '+' && word[1] != '-') {
    word.remove_prefix(1);
  }
  auto value = T{};
  const auto* end = word.data() + word.size();
  const auto [at, error] = std::from_chars(word.data(), end, value);
  if (error != std::errc{} || at != end) return std::nullopt;
  return value;
}

}  // namespace

double AmplitudeHistory::At(double t) const {
  if (times.empty()) return 1.0;
  if (t <= times.front()) return factors.front();
  if (t >= times.back()) return factors.back();
  // times[i] <= t < times[i + 1]
  const auto i = static_cast<std::size_t>(
      std::ranges::upper_bound(times, t) - times.begin() - 1);
  const auto w = (t - times[i]) / (times[i + 1] - times[i]);
  return factors[i] + w * (factors[i + 1] - factors[i]);
}

namespace {
/// The tolerance a seismic term's onset is read with on the terms' clock.
double OnsetTolerance(double onset) { return 1e-12 * (1.0 + std::abs(onset)); }
}  // namespace

double SeismicTerm::FactorAt(double t) const {
  if (onset && t < *onset - OnsetTolerance(*onset)) return 0.0;
  return history.At(t);
}

double SeismicTerm::FactorBefore(double t) const {
  if (onset && t <= *onset + OnsetTolerance(*onset)) return 0.0;
  return history.At(t);
}

bool SeismicTerm::OnsetAt(double t) const {
  return onset && std::abs(t - *onset) <= OnsetTolerance(*onset);
}

namespace {

/// A planar fault term's sources.
std::vector<PointSource> PlanarSourcesOf(const SeismicTerm& term,
                                         double surfaceRadius) {
  return PlanarFaultSources(PlanarFault{.centroid = term.site,
                                        .depth = term.depth,
                                        .strike = term.strike,
                                        .dip = term.dip,
                                        .rake = term.rake,
                                        .length = term.length,
                                        .width = term.width,
                                        .moment = term.scalarMoment},
                            surfaceRadius, term.alongStrike, term.downDip);
}


/// A fault file term's state once placed.
const FaultFileState& PlacedOf(const SeismicTerm& term) {
  if (!term.fault || !term.fault->placed) {
    throw std::logic_error(
        "a fault file's sources are formed once the Earth is built "
        "(PlaceFaultFiles)");
  }
  return *term.fault;
}

/// The time on a fault file's own axis, in the file's time unit, that time
/// t of the terms' clock reads: t - `time_origin` in the term's unit.
double FileDateOf(const SeismicTerm& term, const planetmodel::Fault& fault,
                  double t) {
  return (t - term.time_origin) * term.unit.seconds / fault.GetScales().Time();
}

/// The slices a fault file term reads at time t of the terms' clock, with
/// their weights (planetmodel::Fault::Rows): a static file's one slice; a
/// file on a time axis's by its own rule at its time (FileDateOf), a time
/// within the tolerance the program reads every file's time axis with
/// (SpanTolerance, planetmodel's `time_rtol` being kAxisRtol) of a stored
/// time read as that time, beyond the span held or refused as the term
/// says. Throws planetmodel::Error where the file refuses the time.
planetmodel::TimeRows RowsOf(const SeismicTerm& term, double t) {
  const auto& fault = PlacedOf(term).fault;
  if (fault.Support() != planetmodel::TimeSupport::Node) {
    return fault.Rows(std::nullopt);
  }
  return fault.Rows(FileDateOf(term, fault, t),
                    {.extrapolation = term.outside == OutsideSpan::Hold
                                          ? planetmodel::Extrapolation::Hold
                                          : planetmodel::Extrapolation::Refuse,
                     .time_rtol = kAxisRtol});
}

/// A fault file term's sources at time t of the terms' clock: at each
/// point the slices' tensors under the weights the file's reading at that
/// time gives them (FaultSources::MomentsAt).
std::vector<PointSource> FileSourcesOf(const SeismicTerm& term, double t) {
  const auto& sources = PlacedOf(term).sources;
  const auto moments = [&] {
    try {
      return sources.MomentsAt(RowsOf(term, t));
    } catch (const planetmodel::Error& error) {
      throw std::runtime_error(
          std::format("{}: the run reads it at {:g} {} on the terms' clock: {}",
                      term.file.string(), t, term.unit.name, error.what()));
    }
  }();
  auto out = std::vector<PointSource>{};
  out.reserve(moments.size());
  for (auto p = std::size_t{0}; p < moments.size(); ++p) {
    out.push_back({.at = sources.points[p], .moment = moments[p]});
  }
  return out;
}


}  // namespace

std::vector<BodyPoint> SeismicPointsOf(const SeismicTerm& term,
                                       double surfaceRadius) {
  auto out = std::vector<BodyPoint>{};
  switch (term.kind) {
    case SeismicTerm::Kind::Point:
      out.push_back({.radius = surfaceRadius - term.depth,
                     .colatitude = term.site.CoLatitude(),
                     .longitude = term.site.Longitude()});
      break;
    case SeismicTerm::Kind::Fault:
      for (const auto& source : PlanarSourcesOf(term, surfaceRadius)) {
        out.push_back(source.at);
      }
      break;
    case SeismicTerm::Kind::FaultFile:
      out = PlacedOf(term).sources.points;
      break;
  }
  return out;
}

void PlaceFaultFiles(std::span<SeismicTerm> seismic, double surfaceRadius,
                     const std::function<LoveModuli(const BodyPoint&)>& moduli,
                     const Json& block, std::span<const double> times) {
  for (auto k = std::size_t{0}; k < seismic.size(); ++k) {
    auto& term = seismic[k];
    if (term.kind != SeismicTerm::Kind::FaultFile) continue;
    const auto& item = block.At("seismic").At(k);
    const auto& member = item.At("fault_file");
    term.fault = ReadFault(term, item);
    auto& state = *term.fault;
    const auto& fault = state.fault;
    const auto reference =
        fault.Frame().Reference() * fault.GetScales().Length();
    if (std::abs(reference - surfaceRadius) > kSameSurface * surfaceRadius &&
        !term.placement) {
      Fail(member,
           std::format("holds a fault whose reference radius, {:.10g} m, is "
                       "not the model's surface radius, {:.10g} m: say how it "
                       "is placed, `\"place_by\": \"depth\"` (each point at its "
                       "depth below the file's reference surface, that depth "
                       "below the model's) or `\"place_by\": \"radius\"` (each "
                       "point at its radius as stored)",
                       reference, surfaceRadius));
    }
    try {
      state.sources = FaultSourcesOf(
          fault, {.cellsAlongStrike = term.cellsAlongStrike,
                  .cellsDownDip = term.cellsDownDip,
                  .points = term.points,
                  .surfaceRadius = surfaceRadius,
                  .placement = term.placement,
                  .moduli = moduli});
    } catch (const std::invalid_argument& error) {
      Fail(member, error.what());
    }
    state.placed = true;
    if (fault.Support() != planetmodel::TimeSupport::Node) continue;
    // every node's time within the file's span, and a time its rule reads
    const auto scale = fault.GetScales().Time();
    const auto& axis = *fault.Time();
    const auto where =
        term.dated ? std::string{"a node's date, where the run reads it"}
                   : std::string{"a node's time less `time_origin`, where the "
                                 "run reads it"};
    const auto label = [&](double seconds) {
      return term.dated ? std::format("{:g} kyr BP", 0.0 - seconds / KYR)
                        : std::format("{:g} {} on the file's own axis",
                                      seconds / term.unit.seconds,
                                      term.unit.name);
    };
    for (const auto t : times) {
      try {
        static_cast<void>(RowsOf(term, t));
      } catch (const planetmodel::Error& error) {
        const auto date = FileDateOf(term, fault, t);
        if (date < axis.front() || date > axis.back()) {
          Fail(member.At("file"),
               std::format("spans {} to {}, which leaves out {}, {}: give a "
                           "file that covers it, or \"outside_span\": "
                           "\"hold\" to hold the file's first or last slip "
                           "outside its span",
                           label(axis.front() * scale),
                           label(axis.back() * scale), label(date * scale),
                           where));
        }
        Fail(member.At("file"),
             std::format("is read at {}, {}: {}", label(date * scale), where,
                         error.what()));
      }
    }
  }
}

std::vector<PointSource> SeismicSourcesAt(std::span<const SeismicTerm> seismic,
                                          double t, double surfaceRadius,
                                          bool before) {
  auto out = std::vector<PointSource>{};
  for (const auto& term : seismic) {
    const auto factor = before ? term.FactorBefore(t) : term.FactorAt(t);
    auto sources = std::vector<PointSource>{};
    if (term.kind == SeismicTerm::Kind::Point) {
      sources.push_back({.at = SeismicPointsOf(term, surfaceRadius).front(),
                         .moment = term.moment});
    } else if (term.kind == SeismicTerm::Kind::Fault) {
      sources = PlanarSourcesOf(term, surfaceRadius);
    } else {
      sources = FileSourcesOf(term, t);
    }
    for (auto& source : sources) {
      for (auto* c : {&source.moment.rr, &source.moment.tt, &source.moment.pp,
                      &source.moment.rt, &source.moment.rp, &source.moment.tp}) {
        *c *= factor;
      }
      out.push_back(source);
    }
  }
  return out;
}

std::vector<double> CapField(const SurfaceGrid& grid, GeographicPoint centre,
                             double width, double amplitude) {
  const auto colat0 = centre.CoLatitude();
  const auto lon0 = centre.Longitude();
  return grid.FromFunction([&](double theta, double phi) {
    const auto c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
    const auto d = std::acos(std::clamp(c, -1.0, 1.0));
    return amplitude * std::exp(-0.5 * (d / width) * (d / width));
  });
}

std::vector<Complex> HarmonicsAt(int lmax, GeographicPoint point) {
  const auto theta = std::clamp(point.CoLatitude(), 0.0, std::numbers::pi);
  const auto phi = point.Longitude();
  const auto wigner =
      GSHTrans::Wigner<double, GSHTrans::NonNegative, GSHTrans::Single,
                       GSHTrans::Single>(lmax, lmax, 0, theta);
  auto out = std::vector<Complex>(SurfaceGrid::Index(lmax, lmax) + 1);
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] =
          wigner[l][m] * std::polar(1.0, static_cast<double>(m) * phi);
    }
  }
  return out;
}

std::vector<Complex> DiskCoefficients(const SurfaceGrid& grid,
                                      GeographicPoint centre, double radius,
                                      double amplitude) {
  const auto lmax = grid.LMax();
  const auto x = std::cos(radius);
  const auto s = std::sin(radius);
  const auto half = std::sin(0.5 * radius);
  // g_l = 2 pi amplitude int_x^1 P_l, from P_l^1(x) by its upward
  // recurrence (l - 1) P_l^1 = (2l - 1) x P_(l-1)^1 - l P_(l-2)^1
  auto g = std::vector<double>(lmax + 1);
  const auto scale = 2.0 * std::numbers::pi * amplitude;
  g[0] = scale * 2.0 * half * half;
  auto before = 0.0, current = s;  // P_0^1 and P_1^1
  for (auto l = 1; l <= lmax; ++l) {
    if (l >= 2) {
      const auto next =
          ((2.0 * l - 1.0) * x * current - l * before) / (l - 1.0);
      before = std::exchange(current, next);
    }
    g[l] = scale * s * current / (static_cast<double>(l) * (l + 1.0));
  }
  const auto harmonics = HarmonicsAt(lmax, centre);
  auto out = grid.ZeroCoefficients();
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto k = SurfaceGrid::Index(l, m);
      out[k] = g[l] * std::conj(harmonics[k]);
    }
  }
  return out;
}

std::vector<Complex> ReadCoefficientTable(const std::filesystem::path& path,
                                          int lmax, const Json& member) {
  auto in = std::ifstream(path);
  if (!in) Fail(member, std::format("cannot be read: {}", path.string()));
  const auto size = SurfaceGrid::Index(lmax, lmax) + 1;
  auto out = std::vector<Complex>(size);
  auto seen = std::vector<char>(size, 0);
  auto number = 0;
  for (std::string line; std::getline(in, line);) {
    ++number;
    const auto words = WordsOf(line);
    if (words.empty() || words.front().starts_with('#')) continue;
    const auto l = words.size() == 4 ? Parse<int>(words[0]) : std::nullopt;
    const auto m = words.size() == 4 ? Parse<int>(words[1]) : std::nullopt;
    const auto re = words.size() == 4 ? Parse<double>(words[2]) : std::nullopt;
    const auto im = words.size() == 4 ? Parse<double>(words[3]) : std::nullopt;
    if (!l || !m || !re || !im || !std::isfinite(*re) || !std::isfinite(*im)) {
      Fail(member, std::format("line {}: is not a row `l m re im` of two "
                               "integers and two finite numbers",
                               number));
    }
    if (*l < 0 || *m < 0 || *m > *l) {
      Fail(member, std::format("line {}: ({}, {}) is not a degree and order "
                               "with 0 <= m <= l",
                               number, *l, *m));
    }
    if (*l > lmax) {
      Fail(member, std::format("line {}: degree {} exceeds lmax = {}", number,
                               *l, lmax));
    }
    if (*m == 0 && *im != 0.0) {
      Fail(member, std::format("line {}: the imaginary part at order 0 is not "
                               "zero (the field is real)",
                               number));
    }
    const auto k = SurfaceGrid::Index(*l, *m);
    if (seen[k]) {
      Fail(member,
           std::format("line {}: ({}, {}) is given twice", number, *l, *m));
    }
    seen[k] = 1;
    out[k] = {*re, *im};
  }
  return out;
}

TimedLoads::TimedLoads(std::vector<LoadTerm> terms, const SurfaceGrid& grid,
                       const Json& block, bool timed)
    : terms_(std::move(terms)), grid_(&grid), block_(&block) {
  for (auto k = std::size_t{0}; k < terms_.size(); ++k) {
    const auto& term = terms_[k];
    auto pattern = PatternOf(term, grid, block.At("load").At(k), timed);
    if (term.kind == LoadTerm::Kind::Field) {
      inputs_.push_back(DescribeField(term, pattern.timeAxis, timed));
    }
    patterns_.push_back({.field = std::move(pattern.field),
                         .coefficients = std::move(pattern.coefficients),
                         .history = std::move(pattern.history)});
  }
}

void TimedLoads::CheckCovers(std::span<const double> times) const {
  for (auto k = std::size_t{0}; k < terms_.size(); ++k) {
    const auto& term = terms_[k];
    const auto& pattern = patterns_[k];
    if (!pattern.history) continue;
    const auto& field = block_->At("load").At(k).At("field");
    for (const auto t : times) {
      static_cast<void>(CoveredTime(term.field, *pattern.history, field,
                                    (t - term.time_origin) * term.unit.seconds,
                                    NodeReadOf(term), AxisUnitOf(term)));
    }
  }
}

BuiltLoads TimedLoads::At(double t) const {
  const auto& grid = *grid_;
  auto out = BuiltLoads{.terms = {}, .load = grid.ZeroField()};
  for (auto k = std::size_t{0}; k < terms_.size(); ++k) {
    const auto& term = terms_[k];
    const auto& pattern = patterns_[k];
    auto field = std::vector<double>{};
    auto coefficients = std::vector<Complex>{};
    if (pattern.history) {
      field = FieldValues(*pattern.history, term,
                          block_->At("load").At(k).At("field"),
                          (t - term.time_origin) * term.unit.seconds,
                          NodeReadOf(term),
                          std::format(" at {:g} {}", t, term.unit.name));
      coefficients = grid.Expand(field);
    } else {
      field = pattern.field;
      coefficients = pattern.coefficients;
    }
    if (!term.history.Empty()) {
      const auto factor = term.history.At(t);
      for (auto& v : field) v *= factor;
      for (auto& c : coefficients) c *= factor;
    }
    // the first term is taken as it stands, the others added to it
    if (out.terms.empty()) {
      out.load = field;
    } else {
      for (auto i = std::size_t{0}; i < field.size(); ++i) {
        out.load[i] += field[i];
      }
    }
    out.terms.push_back(std::move(coefficients));
  }
  return out;
}

std::string Describe(const LoadTerm& term) {
  auto out = std::string{};
  switch (term.kind) {
    case LoadTerm::Kind::Cap:
      out = std::format(
          "cap at latitude {:g}, longitude {:g}, width {:g} deg, {:g} kg m^-2",
          term.site.latitude, term.site.longitude,
          term.width * 180.0 / std::numbers::pi, term.amplitude);
      break;
    case LoadTerm::Kind::Disk:
      out = std::format(
          "disk at latitude {:g}, longitude {:g}, radius {:g} deg, {:g} kg "
          "m^-2",
          term.site.latitude, term.site.longitude,
          term.radius * 180.0 / std::numbers::pi, term.amplitude);
      break;
    case LoadTerm::Kind::Field:
      out = std::format("field {}:{}", term.field.file.string(),
                        term.field.variable);
      if (term.field.date) {
        out += term.before_present
                   ? std::format(", at {:g} kyr BP", *term.field.date)
                   : std::format(", at {:g} {}", *term.field.date,
                                 term.unit.name);
      }
      if (term.dated) {
        out += ", read by date";
      } else if (term.time_origin != 0.0) {
        out += std::format(", time origin {:g} {}", term.time_origin,
                           term.unit.name);
      }
      break;
    case LoadTerm::Kind::Coefficients:
      out = std::format("coefficients of {}", term.file.string());
      break;
  }
  if (!term.history.Empty()) out += History(term.history);
  return out;
}

double StaticReadTimeOf(const LoadTerm& term) {
  const auto date = term.field.date.value_or(0.0);
  const auto time =
      term.before_present ? -date * KYR : date * term.unit.seconds;
  // the present as one zero, whichever sign the spelling gives it
  return time == 0.0 ? 0.0 : time;
}

std::string DescribeField(const LoadTerm& term, bool timeAxis, bool timed) {
  auto slice = std::string{"static"};
  if (timeAxis && !timed) {
    const auto date = term.field.date.value_or(0.0);
    slice = term.before_present ? std::format("{:g} kyr BP", date)
                                : std::format("{:g} {}", date, term.unit.name);
  } else if (timeAxis) {
    slice = term.dated ? std::string{"each node's date"}
            : term.time_origin == 0.0
                ? std::string{"the run's clock"}
                : std::format("time origin {:g} {}", term.time_origin,
                              term.unit.name);
  }
  return "field " + DescribeInput(term.field, slice);
}

std::string Describe(const TideTerm& term) {
  auto out =
      std::format("tide degree {} order {}, {:g} + {:g} i m^2 s^-2",
                  term.degree, term.order, term.amplitude, term.imaginary);
  if (!term.history.Empty()) out += History(term.history);
  return out;
}

std::string Describe(const SeismicTerm& term) {
  auto out = std::string{};
  if (term.kind == SeismicTerm::Kind::Point) {
    out = std::format(
        "point source at latitude {:g}, longitude {:g}, depth {:g} km, moment "
        "tensor (rr tt pp rt rp tp) {:g} {:g} {:g} {:g} {:g} {:g} N m",
        term.site.latitude, term.site.longitude, term.depth / 1e3,
        term.moment.rr, term.moment.tt, term.moment.pp, term.moment.rt,
        term.moment.rp, term.moment.tp);
  } else if (term.kind == SeismicTerm::Kind::Fault) {
    out = std::format(
        "fault centred at latitude {:g}, longitude {:g}, depth {:g} km, strike "
        "{:g}, dip {:g}, rake {:g} deg, {:g} by {:g} km, moment {:g} N m, {} "
        "by {} patches",
        term.site.latitude, term.site.longitude, term.depth / 1e3, term.strike,
        term.dip, term.rake, term.length / 1e3, term.width / 1e3,
        term.scalarMoment, term.alongStrike, term.downDip);
  } else {
    out = std::format("fault file {}", term.file.string());
    if (term.fault) {
      const auto& fault = term.fault->fault;
      const auto patches = fault.Patches().size();
      out += std::format(" ({} patch{}, ", patches, patches == 1 ? "" : "es");
      out += fault.Support() == planetmodel::TimeSupport::Node
                 ? std::format("slip at {} times, {})", fault.NSlices(),
                               fault.TimeInterpolation())
                 : std::string{"static)"};
    }
    out += std::format(", {} by {} cells a patch, {} Gauss point{} each way a "
                       "cell",
                       term.cellsAlongStrike, term.cellsDownDip, term.points,
                       term.points == 1 ? "" : "s");
    if (term.placement) {
      out += term.placement == FaultPlacement::Depth ? ", placed by depth"
                                                     : ", placed by radius";
    }
    auto onAxis = false;
    onAxis = term.fault && term.fault->fault.Support() == planetmodel::TimeSupport::Node;
    if (onAxis && term.dated) {
      out += ", read by date";
    } else if (onAxis && term.time_origin != 0.0) {
      out += std::format(", time origin {:g} {}", term.time_origin,
                         term.unit.name);
    }
  }
  if (term.onset && term.start_kyr_bp) {
    out += std::format(", from {:g} kyr BP", *term.start_kyr_bp - *term.onset);
  } else if (term.onset) {
    out += std::format(", from {:g} {}", *term.onset, term.unit.name);
  }
  if (!term.history.Empty()) out += History(term.history);
  return out;
}

std::string Summary(const Forcing& forcing) {
  auto parts = std::vector<std::string>{};
  if (const auto n = forcing.load.size(); n > 0) {
    parts.push_back(std::format("load ({} term{})", n, n == 1 ? "" : "s"));
  }
  if (const auto n = forcing.tide.size(); n > 0) {
    parts.push_back(
        std::format("tide ({} coefficient{})", n, n == 1 ? "" : "s"));
  }
  if (const auto n = forcing.seismic.size(); n > 0) {
    parts.push_back(std::format("seismic ({} source{})", n, n == 1 ? "" : "s"));
  }
  auto out = std::string{};
  for (const auto& part : parts) out += (out.empty() ? "" : " + ") + part;
  return out;
}

namespace {

constexpr auto kIceChangeKinds = std::array<std::string_view, 4>{
    "thickness", "change", "cap_thickness", "cap_fraction"};

/// One term of an ice change of a run of the given `clock` (a run through
/// time's, `timed`, or the static problem's).
IceChangeTerm ReadIceChangeTerm(const Json& item,
                                const std::filesystem::path& directory,
                                bool timed, const TermClock& clock) {
  if (!item.IsObject()) {
    Fail(item,
         "is not a term: a term of an ice change is an object holding one of "
         "\"thickness\", \"change\", \"cap_thickness\" and \"cap_fraction\"");
  }
  RefuseOtherClock(item, clock, timed, false);
  const auto history = NamesOf(clock.form).history;
  const auto forwards = clock.form == Form::Forwards;
  auto members = std::vector<std::string_view>(kIceChangeKinds.begin(),
                                               kIceChangeKinds.end());
  members.push_back(history);
  Allow(item, std::span(members));
  const auto kinds = std::ranges::count_if(
      kIceChangeKinds,
      [&item](std::string_view kind) { return Has(item, kind); });
  if (kinds != 1) {
    Fail(item,
         std::format("{}: a term of an ice change is exactly one of "
                     "\"thickness\", \"change\", \"cap_thickness\" and "
                     "\"cap_fraction\"",
                     kinds == 0 ? "names no term" : "names several terms"));
  }
  auto term = IceChangeTerm{};
  if (Has(item, "thickness") || Has(item, "change")) {
    const auto thickness = Has(item, "thickness");
    term.kind = thickness ? IceChangeTerm::Kind::Thickness
                          : IceChangeTerm::Kind::Change;
    if (Has(item, history)) {
      Fail(item.At(history),
           "is an amplitude history, which a cap of ice change alone takes: "
           "a `thickness` or `change` term's change is its file's");
    }
    const auto& file = item.At(term.Member());
    if (thickness && forwards) {
      Fail(file,
           "is an ice history, a file of thickness whose change is measured "
           "along its time axis from the state's instant: a dated history, "
           "which makes the problem a dated one. Give the dated form of "
           "`times` (`start_kyr_bp`, `end_kyr_bp` and `nodes`, or "
           "`at_kyr_bp`); on a run's own clock the ice changes by a `change` "
           "file without a time axis, held through the run, or by a cap "
           "with its `history`");
    }
    term.file = ReadFileInput(
        file, directory, thickness ? "ice_thickness" : "ice_thickness_change");
    if (timed) {
      if (Has(file, "date_kyr_bp")) {
        Fail(file.At("date_kyr_bp"),
             forwards ? "dates the file, as the static problem reads it; a "
                        "run counting forwards on its own clock holds the "
                        "file, one field without a time axis, through the run"
                      : "dates the file, as the static problem reads it; a "
                        "run through time reads it at each node's date");
      }
      AllowFileInput(file);
      // a dated run reads the file at each node's date; one counting
      // forwards holds its one field
      term.dated = !forwards;
      term.held = forwards;
    } else {
      AllowFileInput(file, {"date_kyr_bp"});
      term.file.date = ReadDate(file, "date_kyr_bp");
      if (thickness && !term.file.date) {
        Fail(file,
             "has no `date_kyr_bp`: a thickness term's change is its file's "
             "value at that date less its value at the state's date");
      }
    }
    return term;
  }
  const auto fraction = Has(item, "cap_fraction");
  term.kind = fraction ? IceChangeTerm::Kind::CapFraction
                       : IceChangeTerm::Kind::CapThickness;
  const auto& cap = item.At(term.Member());
  Allow(cap, {"latitude", "longitude", "width_deg",
              fraction ? "fraction" : "thickness_m"});
  term.site = ReadSite(cap);
  const auto width = Number(cap, "width_deg");
  if (!(width > 0.0)) Fail(cap.At("width_deg"), "is not a positive angle");
  term.width = width * std::numbers::pi / 180.0;
  term.amount = Number(cap, fraction ? "fraction" : "thickness_m");
  term.history = ReadHistory(item, timed, clock);
  return term;
}

}  // namespace

std::string_view IceChangeTerm::Member() const {
  switch (kind) {
    case Kind::Thickness:
      return "thickness";
    case Kind::Change:
      return "change";
    case Kind::CapThickness:
      return "cap_thickness";
    case Kind::CapFraction:
      break;
  }
  return "cap_fraction";
}

GeographicPoint ReadSite(const Json& object) {
  const auto latitude = Number(object, "latitude");
  if (!(latitude >= -90.0 && latitude <= 90.0)) {
    Fail(object.At("latitude"), "is not a latitude in [-90, 90]");
  }
  return {latitude, Number(object, "longitude")};
}

Forcing ReadForcing(const Json& block, const std::filesystem::path& directory,
                    int lmax, const std::optional<TermClock>& given,
                    bool ocean) {
  // the static problem's terms are dated, and have no window
  const auto timed = given.has_value();
  const auto clock = given.value_or(
      TermClock{.form = Form::Dated, .start_kyr_bp = std::nullopt, .unit = {}});
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `forcing` is an object of the ice change, "
         "`ice_change`, the surface loads, `load`, the tide, `tide`, and the "
         "seismic sources, `seismic`");
  }
  for (const auto* name : {"state", "sea_level"}) {
    if (Has(block, name)) {
      Fail(block.At(name),
           "is not a member of `forcing`, which holds what happens to the "
           "state: the sea level is the state's, the `ocean` block's, "
           "`ocean.sea_level` (a sea level, used as it stands) or "
           "`ocean.topography` (converted with the state's ice, "
           "`ocean.ice`)");
    }
  }
  if (Has(block, "ice")) {
    Fail(block.At("ice"),
         "is not a member of `forcing`: the ice on the body is the state's, "
         "`ocean.ice`, and a change of it is a term of `forcing.ice_change` "
         "(`thickness`, `change`, `cap_thickness` or `cap_fraction`)");
  }
  Allow(block, {"ice_change", "load", "tide", "seismic"});
  if (!ocean && Has(block, "ice_change")) {
    Fail(block.At("ice_change"),
         "is an ice change, a change of the state's ice, and the config has "
         "no `ocean` block: without an ocean the ice's water has nowhere to "
         "go. Give the ice's load as a `load` term, or give the state, the "
         "`ocean` block");
  }
  auto out = Forcing{};
  if (Has(block, "load")) {
    constexpr auto kProperties = std::array<std::string_view, 3>{
        "water_budget", "moves_solid_surface", "density_kg_m3"};
    for (const auto& item : block.At("load").AsArray()) {
      auto term = SurfaceLoadTerm{};
      term.term = ReadLoadTerm(item, directory, timed, kProperties, clock);
      // a run with an ocean is told both properties; without one they
      // default to false, the water budget then having no ocean to change
      for (const auto* name : {"water_budget", "moves_solid_surface"}) {
        if (ocean && !Has(item, name)) {
          Fail(item,
               std::format("does not say `{}` (true or false): a surface "
                           "load of a run with an ocean states whether its "
                           "mass is exchanged with the ocean "
                           "(`water_budget`) and whether it is a layer on "
                           "the solid surface (`moves_solid_surface`)",
                           name));
        }
      }
      if (Has(item, "water_budget")) {
        term.water_budget = item.At("water_budget").AsBool();
      }
      if (Has(item, "moves_solid_surface")) {
        term.moves_solid_surface = item.At("moves_solid_surface").AsBool();
      }
      if (term.moves_solid_surface) {
        if (!Has(item, "density_kg_m3")) {
          Fail(item,
               "moves the solid surface and gives no `density_kg_m3`: the "
               "layer's thickness is its mass over its density");
        }
        term.density = Number(item, "density_kg_m3");
        if (!(term.density > 0.0) || !std::isfinite(term.density)) {
          Fail(item.At("density_kg_m3"), "is not a positive density");
        }
      } else if (Has(item, "density_kg_m3")) {
        Fail(item.At("density_kg_m3"),
             "is the density of a layer on the solid surface, and this load "
             "is none (`moves_solid_surface` is false)");
      }
      out.load.push_back(std::move(term));
    }
  }
  if (Has(block, "tide")) {
    for (const auto& item : block.At("tide").AsArray()) {
      out.tide.push_back(ReadTideTerm(item, lmax, timed, clock));
    }
  }
  if (Has(block, "seismic")) {
    for (const auto& item : block.At("seismic").AsArray()) {
      out.seismic.push_back(ReadSeismicTerm(item, directory, timed, clock));
    }
  }
  if (Has(block, "ice_change")) {
    for (const auto& item : block.At("ice_change").AsArray()) {
      out.ice_change.push_back(
          ReadIceChangeTerm(item, directory, timed, clock));
    }
  }
  if (out.ice_change.empty() && !out.BesideTheIce()) {
    const auto what = std::format(
        "at least one {}surface load (`load`), tide coefficient (`tide`) or "
        "seismic source (`seismic`)",
        ocean ? "term of the ice change (`ice_change`: \"thickness\", "
                "\"change\", \"cap_thickness\" or \"cap_fraction\"), "
              : "");
    if (Has(block, "ice_change")) {
      Fail(block.At("ice_change"),
           std::format("holds no term, and nothing else forces the run: "
                       "give {}",
                       what));
    }
    Fail(block, std::format("holds no forcing: give {}", what));
  }
  return out;
}

std::string Describe(const IceChangeTerm& term) {
  const auto degrees = 180.0 / std::numbers::pi;
  switch (term.kind) {
    case IceChangeTerm::Kind::Thickness:
      return term.dated
                 ? std::format("thickness {}:{}, from the state's instant to "
                               "each node's date",
                               term.file.file.string(), term.file.variable)
                 : std::format("thickness {}:{}, from the state's date to "
                               "{:g} kyr BP",
                               term.file.file.string(), term.file.variable,
                               term.file.date.value_or(0.0));
    case IceChangeTerm::Kind::Change: {
      auto out = std::format("change {}:{}", term.file.file.string(),
                             term.file.variable);
      if (term.dated) {
        out += ", read by date";
      } else if (term.held) {
        out += ", held through the run";
      } else if (term.file.date) {
        out += std::format(", at {:g} kyr BP", *term.file.date);
      }
      return out;
    }
    case IceChangeTerm::Kind::CapThickness:
    case IceChangeTerm::Kind::CapFraction:
      break;
  }
  auto out = std::format("cap at latitude {:g}, longitude {:g}, width {:g} deg",
                         term.site.latitude, term.site.longitude,
                         term.width * degrees);
  out += term.kind == IceChangeTerm::Kind::CapThickness
             ? std::format(", {:g} m of ice", term.amount)
             : std::format(", fraction {:g} of the state's ice", term.amount);
  if (!term.history.Empty()) out += History(term.history);
  return out;
}

std::string Describe(const SurfaceLoadTerm& term) {
  auto out = Describe(term.term);
  out += term.water_budget ? ", in the water budget"
                           : ", outside the water budget";
  if (term.moves_solid_surface) {
    out += std::format(", a layer on the solid surface of {:g} kg m^-3",
                       term.density);
  }
  return out;
}

std::vector<std::complex<double>> TideAt(std::span<const TideTerm> tide,
                                         const SurfaceGrid& grid, double t) {
  auto out = grid.ZeroCoefficients();
  for (const auto& term : tide) {
    out[SurfaceGrid::Index(term.degree, term.order)] +=
        term.history.At(t) * Complex{term.amplitude, term.imaginary};
  }
  return out;
}

namespace {

/// The load terms of a forcing's surface loads, in the order given.
std::vector<LoadTerm> TermsOf(const Forcing& forcing) {
  auto out = std::vector<LoadTerm>{};
  for (const auto& term : forcing.load) out.push_back(term.term);
  return out;
}

}  // namespace

SurfaceLoads::SurfaceLoads(const Forcing& forcing, const SurfaceGrid& grid,
                           const Json& block, bool timed)
    : terms_(forcing.load),
      grid_(&grid),
      timed_(TermsOf(forcing), grid, block, timed) {}

void SurfaceLoads::CheckCovers(std::span<const double> times) const {
  timed_.CheckCovers(times);
}

SurfaceLoads::Node SurfaceLoads::At(double t) const {
  const auto& grid = *grid_;
  auto built = timed_.At(t);
  auto inside = std::vector<Complex>{}, outside = std::vector<Complex>{};
  auto layer = std::vector<Complex>{};
  auto add = [&](std::vector<Complex>& to, const std::vector<Complex>& term,
                 double factor) {
    if (to.empty()) to = grid.ZeroCoefficients();
    for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += factor * term[i];
  };
  for (auto k = std::size_t{0}; k < terms_.size(); ++k) {
    add(terms_[k].water_budget ? inside : outside, built.terms[k], 1.0);
    if (terms_[k].moves_solid_surface) {
      add(layer, built.terms[k], 1.0 / terms_[k].density);
    }
  }
  auto out = Node{};
  if (!inside.empty()) out.inside = grid.Synthesise(inside);
  if (!outside.empty()) out.outside = grid.Synthesise(outside);
  if (!layer.empty()) out.layer = grid.Synthesise(layer);
  if (!terms_.empty()) out.given = std::move(built.load);
  return out;
}

}  // namespace sl3dpp::app
