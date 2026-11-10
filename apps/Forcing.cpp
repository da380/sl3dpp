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
#include <stdexcept>
#include <string_view>
#include <utility>

#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

namespace {

using Complex = std::complex<double>;

constexpr auto kTermKinds =
    std::array<std::string_view, 4>{"cap", "disk", "field", "coefficients"};

/// Refuses an amplitude history, which a forcing fixed in time cannot take.
void RefuseHistory(const Json& item) {
  if (Has(item, "history_kyr")) {
    Fail(item.At("history_kyr"),
         "is an amplitude history, which needs a `times` block; a forcing "
         "that varies in time is not built yet");
  }
}

/// A centre's latitude and longitude, degrees.
GeographicPoint SiteOf(const Json& object) {
  const auto latitude = Number(object, "latitude");
  if (!(latitude >= -90.0 && latitude <= 90.0)) {
    Fail(object.At("latitude"), "is not a latitude in [-90, 90]");
  }
  return {latitude, Number(object, "longitude")};
}

LoadTerm ReadLoadTerm(const Json& item,
                      const std::filesystem::path& directory) {
  RefuseHistory(item);
  Allow(item, kTermKinds);
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
    Allow(field, {"file", "name"});
    term.kind = LoadTerm::Kind::Field;
    term.file = PathOf(field, "file", directory);
    term.name = String(field, "name", "load");
    if (term.name.empty()) Fail(field.At("name"), "is empty");
  } else {
    const auto& coefficients = item.At("coefficients");
    Allow(coefficients, {"file"});
    term.kind = LoadTerm::Kind::Coefficients;
    term.file = PathOf(coefficients, "file", directory);
  }
  return term;
}

TideTerm ReadTideTerm(const Json& item, int lmax) {
  RefuseHistory(item);
  Allow(item, {"degree", "order", "amplitude_m2_s2", "imaginary_m2_s2"});
  auto term = TideTerm{};
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

/// A field term's values on the grid, SI.
std::vector<double> ReadFieldTerm(const LoadTerm& term, const SurfaceGrid& grid,
                                  const Json& field) {
  const auto& file = field.At("file");
  const auto history = io::FieldHistory::Read(term.file, grid);
  if (!history) Fail(file, history.error().message);
  if (history->Span()) {
    Fail(file,
         "has a time axis: a load that varies in time needs a `times` "
         "block, which is not built yet");
  }
  if (!history->Has(term.name)) {
    Fail(file, std::format("holds no field \"{}\"", term.name));
  }
  auto values = std::vector<double>{};
  try {
    values = history->Values(term.name, 0.0);
  } catch (const std::runtime_error& error) {
    Fail(file, error.what());
  }
  if (!std::ranges::all_of(values, [](double v) { return std::isfinite(v); })) {
    Fail(file, std::format("holds a field \"{}\" that is not finite "
                           "everywhere on the run's grid",
                           term.name));
  }
  return values;
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

Forcing ReadForcing(const Json& block, const std::filesystem::path& directory,
                    int lmax) {
  Allow(block, {"load", "tide"});
  auto out = Forcing{};
  if (Has(block, "load")) {
    for (const auto& item : block.At("load").AsArray()) {
      out.load.push_back(ReadLoadTerm(item, directory));
    }
  }
  if (Has(block, "tide")) {
    for (const auto& item : block.At("tide").AsArray()) {
      out.tide.push_back(ReadTideTerm(item, lmax));
    }
  }
  if (out.load.empty() && out.tide.empty()) {
    Fail(block, "needs at least one load term or tide coefficient");
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
    const auto& term = forcing.load[k];
    const auto& item = block.At("load").At(k);
    auto field = std::vector<double>{};
    auto coefficients = std::vector<Complex>{};
    switch (term.kind) {
      case LoadTerm::Kind::Cap:
        field = CapField(grid, term.site, term.width, term.amplitude);
        coefficients = grid.Expand(field);
        break;
      case LoadTerm::Kind::Disk:
        coefficients =
            DiskCoefficients(grid, term.site, term.radius, term.amplitude);
        field = grid.Synthesise(coefficients);
        break;
      case LoadTerm::Kind::Field:
        field = ReadFieldTerm(term, grid, item.At("field"));
        coefficients = grid.Expand(field);
        break;
      case LoadTerm::Kind::Coefficients:
        coefficients = ReadCoefficientTable(term.file, grid.LMax(),
                                            item.At("coefficients").At("file"));
        field = grid.Synthesise(coefficients);
        break;
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
  for (const auto& term : forcing.tide) {
    out.tide[SurfaceGrid::Index(term.degree, term.order)] +=
        Complex{term.amplitude, term.imaginary};
  }
  return out;
}

std::string Describe(const LoadTerm& term) {
  switch (term.kind) {
    case LoadTerm::Kind::Cap:
      return std::format(
          "cap at latitude {:g}, longitude {:g}, width {:g} deg, {:g} kg m^-2",
          term.site.latitude, term.site.longitude,
          term.width * 180.0 / std::numbers::pi, term.amplitude);
    case LoadTerm::Kind::Disk:
      return std::format(
          "disk at latitude {:g}, longitude {:g}, radius {:g} deg, {:g} kg "
          "m^-2",
          term.site.latitude, term.site.longitude,
          term.radius * 180.0 / std::numbers::pi, term.amplitude);
    case LoadTerm::Kind::Field:
      return std::format("field \"{}\" of {}", term.name, term.file.string());
    case LoadTerm::Kind::Coefficients:
      break;
  }
  return std::format("coefficients of {}", term.file.string());
}

std::string Describe(const TideTerm& term) {
  return std::format("tide degree {} order {}, {:g} + {:g} i m^2 s^-2",
                     term.degree, term.order, term.amplitude, term.imaginary);
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
  auto out = std::string{};
  for (const auto& part : parts) out += (out.empty() ? "" : " + ") + part;
  return out;
}

}  // namespace sl3dpp::app
