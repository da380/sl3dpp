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

namespace sl3dpp::app {

namespace {

using Complex = std::complex<double>;

constexpr auto kTermKinds =
    std::array<std::string_view, 4>{"cap", "disk", "field", "coefficients"};

/// A term's amplitude history, `history_kyr`: refused without a `times`
/// block (`timed`), which a forcing fixed in time cannot take.
AmplitudeHistory ReadHistory(const Json& item, bool timed) {
  auto out = AmplitudeHistory{};
  if (!Has(item, "history_kyr")) return out;
  const auto& history = item.At("history_kyr");
  if (!timed) {
    Fail(history, "is an amplitude history, which needs a `times` block");
  }
  for (const auto& pair : history.AsArray()) {
    if (!pair.IsArray() || pair.AsArray().size() != 2) {
      Fail(pair, "is not a pair [time_kyr, factor]");
    }
    const auto time = pair.At(std::size_t{0}).AsNumber();
    const auto factor = pair.At(std::size_t{1}).AsNumber();
    if (!std::isfinite(time) || !std::isfinite(factor)) {
      Fail(pair, "is not a pair of finite numbers [time_kyr, factor]");
    }
    if (!out.times_kyr.empty() && !(time > out.times_kyr.back())) {
      Fail(pair.At(std::size_t{0}),
           "is not later than the time before it: a history's times are "
           "strictly increasing");
    }
    out.times_kyr.push_back(time);
    out.factors.push_back(factor);
  }
  if (out.times_kyr.empty()) {
    Fail(history, "holds no [time_kyr, factor] pair");
  }
  return out;
}

/// A centre's latitude and longitude, degrees.
GeographicPoint SiteOf(const Json& object) { return ReadSite(object); }

LoadTerm ReadLoadTerm(const Json& item, const std::filesystem::path& directory,
                      bool timed) {
  auto history = ReadHistory(item, timed);
  auto members =
      std::vector<std::string_view>(kTermKinds.begin(), kTermKinds.end());
  members.push_back("history_kyr");
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
    Allow(field, {"file", "name", "time_origin_kyr"});
    term.kind = LoadTerm::Kind::Field;
    term.file = PathOf(field, "file", directory);
    term.name = String(field, "name", "load");
    if (term.name.empty()) Fail(field.At("name"), "is empty");
    if (Has(field, "time_origin_kyr")) {
      const auto& origin = field.At("time_origin_kyr");
      if (!timed) Fail(origin, "is a time origin, which needs a `times` block");
      term.time_origin_kyr = origin.AsNumber();
      if (!std::isfinite(term.time_origin_kyr)) {
        Fail(origin, "is not a finite number");
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

TideTerm ReadTideTerm(const Json& item, int lmax, bool timed) {
  auto history = ReadHistory(item, timed);
  Allow(item, {"degree", "order", "amplitude_m2_s2", "imaginary_m2_s2",
               "history_kyr"});
  auto term = TideTerm{};
  term.history = std::move(history);
  term.degree = Integer(item, "degree");
  if (term.degree < 2 || term.degree > lmax) {
    Fail(item.At("degree"), "is not a tidal degree within [2, lmax]");
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

/// One seismic source: a point moment tensor or a planar fault.
SeismicTerm ReadSeismicTerm(const Json& item) {
  Allow(item, {"point", "fault", "onset_kyr", "history_kyr"});
  if (Has(item, "point") == Has(item, "fault")) {
    Fail(item, "is exactly one of \"point\" and \"fault\"");
  }
  auto term = SeismicTerm{};
  term.history = ReadHistory(item, true);
  if (Has(item, "onset_kyr")) term.onset_kyr = Number(item, "onset_kyr");
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

/// A field term's file on the grid, holding its field. A file with a time
/// axis is refused unless `timed`.
io::FieldHistory OpenFieldTerm(const LoadTerm& term, const SurfaceGrid& grid,
                               const Json& file, bool timed) {
  auto history = io::FieldHistory::Read(term.file, grid);
  if (!history) Fail(file, history.error().message);
  if (!timed && history->Span()) {
    Fail(file,
         "has a time axis: a load that varies in time needs a `times` "
         "block");
  }
  if (!history->Has(term.name)) {
    Fail(file, std::format("holds no field \"{}\"", term.name));
  }
  return std::move(*history);
}

/// A field term's values on the grid, SI, at an SI time of its file;
/// `when` names the run's time in a fault, where it matters.
std::vector<double> FieldValues(const io::FieldHistory& history,
                                const LoadTerm& term, const Json& file,
                                double time, std::string_view when = {}) {
  auto values = std::vector<double>{};
  try {
    values = history.Values(term.name, time);
  } catch (const std::runtime_error& error) {
    Fail(file, error.what());
  }
  if (!std::ranges::all_of(values, [](double v) { return std::isfinite(v); })) {
    Fail(file, std::format("holds a field \"{}\" that is not finite "
                           "everywhere on the run's grid{}",
                           term.name, when));
  }
  return values;
}

/// A load term's pattern on the grid: its field at the grid's points and
/// its coefficients, the one made from the other as the term's kind says.
/// In a run through time (`timed`) a field term whose file has a time axis
/// has no fixed pattern: its file comes back instead, both arrays empty.
struct TermPattern {
  std::vector<double> field;
  std::vector<Complex> coefficients;
  std::optional<io::FieldHistory> history;
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
      const auto& file = item.At("field").At("file");
      auto history = OpenFieldTerm(term, grid, file, timed);
      if (history.Span()) {
        out.history.emplace(std::move(history));
        break;
      }
      out.field = FieldValues(history, term, file, 0.0);
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
  for (auto i = std::size_t{0}; i < history.times_kyr.size(); ++i) {
    out += std::format("{}[{:g}, {:g}]", i == 0 ? "" : ", ",
                       history.times_kyr[i], history.factors[i]);
  }
  return out + "]";
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
  if (times_kyr.empty()) return 1.0;
  if (t <= times_kyr.front()) return factors.front();
  if (t >= times_kyr.back()) return factors.back();
  // times_kyr[i] <= t < times_kyr[i + 1]
  const auto i = static_cast<std::size_t>(
      std::ranges::upper_bound(times_kyr, t) - times_kyr.begin() - 1);
  const auto w = (t - times_kyr[i]) / (times_kyr[i + 1] - times_kyr[i]);
  return factors[i] + w * (factors[i + 1] - factors[i]);
}

Forcing ReadForcing(const Json& block, const std::filesystem::path& directory,
                    int lmax, bool timed) {
  Allow(block, {"load", "tide", "seismic"});
  auto out = Forcing{};
  if (Has(block, "load")) {
    for (const auto& item : block.At("load").AsArray()) {
      out.load.push_back(ReadLoadTerm(item, directory, timed));
    }
  }
  if (Has(block, "tide")) {
    for (const auto& item : block.At("tide").AsArray()) {
      out.tide.push_back(ReadTideTerm(item, lmax, timed));
    }
  }
  if (Has(block, "seismic")) {
    if (!timed) {
      Fail(block.At("seismic"),
           "gives seismic sources, which need a `times` block: the first "
           "node of a run through time is the co-seismic response");
    }
    for (const auto& item : block.At("seismic").AsArray()) {
      out.seismic.push_back(ReadSeismicTerm(item));
    }
  }
  if (out.load.empty() && out.tide.empty() && out.seismic.empty()) {
    Fail(block,
         "needs at least one load term, tide coefficient or seismic source");
  }
  return out;
}

double SeismicTerm::FactorAt(double t) const {
  if (onset_kyr && t < *onset_kyr - 1e-12 * (1.0 + std::abs(*onset_kyr))) {
    return 0.0;
  }
  return history.At(t);
}

std::vector<PointSource> SeismicSourcesAt(const Forcing& forcing, double t,
                                          double surfaceRadius) {
  auto out = std::vector<PointSource>{};
  for (const auto& term : forcing.seismic) {
    const auto factor = term.FactorAt(t);
    auto sources = std::vector<PointSource>{};
    if (term.kind == SeismicTerm::Kind::Point) {
      sources.push_back({.at = {.radius = surfaceRadius - term.depth,
                                .colatitude = term.site.CoLatitude(),
                                .longitude = term.site.Longitude()},
                         .moment = term.moment});
    } else {
      sources = PlanarFaultSources(
          PlanarFault{.centroid = term.site,
                      .depth = term.depth,
                      .strike = term.strike,
                      .dip = term.dip,
                      .rake = term.rake,
                      .length = term.length,
                      .width = term.width,
                      .moment = term.scalarMoment},
          surfaceRadius, term.alongStrike, term.downDip);
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

BuiltForcing BuildForcing(const Forcing& forcing, const SurfaceGrid& grid,
                          const Json& block) {
  auto out = BuiltForcing{.terms = {},
                          .loadCoefficients = grid.ZeroCoefficients(),
                          .load = grid.ZeroField(),
                          .tide = grid.ZeroCoefficients()};
  for (auto k = std::size_t{0}; k < forcing.load.size(); ++k) {
    auto [field, coefficients, history] =
        PatternOf(forcing.load[k], grid, block.At("load").At(k), false);
    // the first term is taken as it stands, the others added to it
    if (out.terms.empty()) {
      out.load = field;
      out.loadCoefficients = coefficients;
    } else {
      for (auto i = std::size_t{0}; i < field.size(); ++i) {
        out.load[i] += field[i];
      }
      for (auto i = std::size_t{0}; i < coefficients.size(); ++i) {
        out.loadCoefficients[i] += coefficients[i];
      }
    }
    out.terms.push_back(std::move(coefficients));
  }
  for (const auto& term : forcing.tide) {
    out.tide[SurfaceGrid::Index(term.degree, term.order)] +=
        Complex{term.amplitude, term.imaginary};
  }
  return out;
}

TimedForcing::TimedForcing(const Forcing& forcing, const SurfaceGrid& grid,
                           const Json& block)
    : forcing_(forcing), grid_(&grid), block_(&block) {
  for (auto k = std::size_t{0}; k < forcing_.load.size(); ++k) {
    const auto& term = forcing_.load[k];
    auto pattern = PatternOf(term, grid, block.At("load").At(k), true);
    varies_ = varies_ || pattern.history.has_value() || !term.history.Empty();
    patterns_.push_back({.field = std::move(pattern.field),
                         .coefficients = std::move(pattern.coefficients),
                         .history = std::move(pattern.history)});
  }
  for (const auto& term : forcing_.tide) {
    varies_ = varies_ || !term.history.Empty();
  }
}

BuiltForcing TimedForcing::At(double t) const {
  const auto& grid = *grid_;
  auto out = BuiltForcing{.terms = {},
                          .loadCoefficients = grid.ZeroCoefficients(),
                          .load = grid.ZeroField(),
                          .tide = grid.ZeroCoefficients()};
  for (auto k = std::size_t{0}; k < forcing_.load.size(); ++k) {
    const auto& term = forcing_.load[k];
    const auto& pattern = patterns_[k];
    auto field = std::vector<double>{};
    auto coefficients = std::vector<Complex>{};
    if (pattern.history) {
      field = FieldValues(*pattern.history, term,
                          block_->At("load").At(k).At("field").At("file"),
                          (t - term.time_origin_kyr) * KYR,
                          std::format(" at {:g} kyr", t));
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
      out.loadCoefficients = coefficients;
    } else {
      for (auto i = std::size_t{0}; i < field.size(); ++i) {
        out.load[i] += field[i];
      }
      for (auto i = std::size_t{0}; i < coefficients.size(); ++i) {
        out.loadCoefficients[i] += coefficients[i];
      }
    }
    out.terms.push_back(std::move(coefficients));
  }
  for (const auto& term : forcing_.tide) {
    out.tide[SurfaceGrid::Index(term.degree, term.order)] +=
        term.history.At(t) * Complex{term.amplitude, term.imaginary};
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
      out = std::format("field \"{}\" of {}", term.name, term.file.string());
      if (term.time_origin_kyr != 0.0) {
        out += std::format(", time origin {:g} kyr", term.time_origin_kyr);
      }
      break;
    case LoadTerm::Kind::Coefficients:
      out = std::format("coefficients of {}", term.file.string());
      break;
  }
  if (!term.history.Empty()) {
    out += ", history_kyr " + Pairs(term.history);
  }
  return out;
}

std::string Describe(const TideTerm& term) {
  auto out =
      std::format("tide degree {} order {}, {:g} + {:g} i m^2 s^-2",
                  term.degree, term.order, term.amplitude, term.imaginary);
  if (!term.history.Empty()) {
    out += ", history_kyr " + Pairs(term.history);
  }
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
  } else {
    out = std::format(
        "fault centred at latitude {:g}, longitude {:g}, depth {:g} km, strike "
        "{:g}, dip {:g}, rake {:g} deg, {:g} by {:g} km, moment {:g} N m, {} "
        "by {} patches",
        term.site.latitude, term.site.longitude, term.depth / 1e3, term.strike,
        term.dip, term.rake, term.length / 1e3, term.width / 1e3,
        term.scalarMoment, term.alongStrike, term.downDip);
  }
  if (term.onset_kyr) out += std::format(", from {:g} kyr", *term.onset_kyr);
  if (!term.history.Empty()) out += ", history_kyr " + Pairs(term.history);
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

/// The fingerprint's ice change.
IceChange ReadIceChange(const Json& change) {
  Allow(change, {"file", "type", "latitude", "longitude", "width_deg",
                 "fraction", "thickness_m"});
  auto out = IceChange{};
  out.kind = Choice<IceChange::Kind>(
      change.At("type"), {{"cap_fraction", IceChange::Kind::CapFraction},
                          {"cap_thickness", IceChange::Kind::CapThickness}});
  out.site = ReadSite(change);
  const auto width = Number(change, "width_deg");
  if (!(width > 0.0)) Fail(change.At("width_deg"), "is not a positive angle");
  out.width = width * std::numbers::pi / 180.0;
  const auto fraction = out.kind == IceChange::Kind::CapFraction;
  if (Has(change, fraction ? "thickness_m" : "fraction")) {
    Fail(change, "gives the amount of the other kind of change");
  }
  out.amount = Number(change, fraction ? "fraction" : "thickness_m");
  return out;
}

}  // namespace

GeographicPoint ReadSite(const Json& object) {
  const auto latitude = Number(object, "latitude");
  if (!(latitude >= -90.0 && latitude <= 90.0)) {
    Fail(object.At("latitude"), "is not a latitude in [-90, 90]");
  }
  return {latitude, Number(object, "longitude")};
}

SeaLevelForcing ReadSeaLevelForcing(const Json& block,
                                    const std::filesystem::path& directory,
                                    bool timed) {
  if (Has(block, "state")) {
    Fail(block.At("state"),
         "is not a member: give the initial sea level as `sea_level` (a "
         "`file` holding `sea_level` or `topography`) beside the ice history "
         "`ice`; the ice the run starts from is the history's own");
  }
  Allow(block, {"sea_level", "ice"});
  auto out = SeaLevelForcing{};
  const auto& sea = block.At("sea_level");
  if (timed && Has(sea, "date_kyr_bp")) {
    Fail(sea.At("date_kyr_bp"),
         "dates the sea level, as the elastic fingerprint (no `times` "
         "block) reads it; a run through time reads the sea level at its "
         "window's start, or calibrating at its end");
  }
  Allow(sea, {"file", "date_kyr_bp"});
  out.seaLevel.file = PathOf(sea, "file", directory);
  out.seaLevel.date_kyr_bp = Number(sea, "date_kyr_bp", 0.0);
  const auto& ice = block.At("ice");
  out.ice.history = PathOf(ice, "file", directory);
  if (timed) {
    for (const auto* name : {"type", "latitude", "longitude", "width_deg",
                             "fraction", "thickness_m"}) {
      if (Has(ice, name)) {
        Fail(ice.At(name),
             "describes an ice change, which the elastic fingerprint (no "
             "`times` block) takes; a run through time takes the ice "
             "history alone, `file`");
      }
    }
    Allow(ice, {"file"});
  } else {
    if (!Has(ice, "type")) {
      Fail(ice,
           "gives the ice history alone, which needs a `times` block; the "
           "elastic fingerprint takes an ice change on it too: `type`, "
           "`latitude`, `longitude`, `width_deg` and `fraction` or "
           "`thickness_m`");
    }
    out.ice.change = ReadIceChange(ice);
  }
  return out;
}

}  // namespace sl3dpp::app
