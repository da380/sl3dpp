#include "Files.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace sl3dpp::app {

FileInput ReadFileInput(const Json& object,
                        const std::filesystem::path& directory,
                        std::string_view variable) {
  if (!object.IsObject()) {
    Fail(object,
         "is not a file input: an object of `file`, and optionally `format` "
         "and `variable`");
  }
  auto out = FileInput{};
  out.file = PathOf(object, "file", directory);
  if (Has(object, "format")) {
    out.format = Choice<FileFormat>(
        object.At("format"),
        {{"planetmodel_surface", FileFormat::PlanetmodelSurface}});
  }
  out.variable = String(object, "variable", variable);
  if (out.variable.empty()) Fail(object.At("variable"), "is empty");
  if (Has(object, "outside_span")) {
    out.outside = Choice<OutsideSpan>(
        object.At("outside_span"),
        {{"refuse", OutsideSpan::Refuse}, {"hold", OutsideSpan::Hold}});
    out.outsideGiven = true;
  }
  return out;
}

void AllowFileInput(const Json& object,
                    std::initializer_list<std::string_view> extra) {
  auto names = std::vector<std::string_view>(kFileInputMembers.begin(),
                                             kFileInputMembers.end());
  names.insert(names.end(), extra.begin(), extra.end());
  Allow(object, std::span(names));
}

std::optional<double> ReadDate(const Json& object, std::string_view key) {
  if (!Has(object, key)) return std::nullopt;
  const auto value = object.At(key).AsNumber();
  if (!std::isfinite(value)) Fail(object.At(key), "is not a finite number");
  return value;
}

io::FieldHistory ReadFile(const FileInput& input, const SurfaceGrid& grid,
                          const Json& object) {
  auto out = std::optional<io::FieldHistory>{};
  switch (input.format) {
    case FileFormat::PlanetmodelSurface: {
      auto read =
          io::FieldHistory::Read(input.file, grid, {.time_rtol = kAxisRtol});
      if (!read) Fail(object.At("file"), read.error().message);
      out.emplace(std::move(*read));
      break;
    }
  }
  return std::move(*out);
}

void CheckVariable(const FileInput& input, const io::FieldHistory& file,
                   const Json& object) {
  if (file.Has(input.variable)) return;
  auto held = std::string{};
  for (const auto& name : file.Names()) {
    held += std::format("{}\"{}\"", held.empty() ? "" : ", ", name);
  }
  Fail(Has(object, "variable") ? object.At("variable") : object.At("file"),
       std::format("names a variable \"{}\" that {} does not hold; it holds "
                   "{}",
                   input.variable, input.file.string(),
                   held.empty() ? "none" : held));
}

bool OnTimeAxis(const FileInput& input, const io::FieldHistory& file) {
  return file.Span().has_value() && !file.Static(input.variable);
}

void CheckOutsideSpan(const FileInput& input, const io::FieldHistory& file,
                      const Json& object) {
  if (!input.outsideGiven || OnTimeAxis(input, file)) return;
  Fail(object.At("outside_span"),
       std::format("says what a read outside its field's time span does, and "
                   "{}: the same at every date, no read of it is outside a "
                   "span. Drop it",
                   file.Span() ? std::format("\"{}\" of {} is static",
                                             input.variable,
                                             input.file.string())
                               : std::format("{} has no time axis",
                                             input.file.string())));
}

io::FieldHistory OpenFileInput(const FileInput& input, const SurfaceGrid& grid,
                               const Json& object) {
  auto file = ReadFile(input, grid, object);
  CheckVariable(input, file, object);
  CheckOutsideSpan(input, file, object);
  return file;
}

double SpanTolerance(std::pair<double, double> span) {
  const auto [first, last] = span;
  return kAxisRtol * std::max({last - first, std::abs(first), std::abs(last)});
}

std::string AxisLabel(double time, const std::optional<TimeUnit>& unit) {
  return unit ? std::format("{:g} {} on the file's own axis",
                            time / unit->seconds, unit->name)
              : std::format("{:g} kyr BP", 0.0 - time / KYR);
}

double CoveredTime(const FileInput& input, const io::FieldHistory& file,
                   const Json& object, double time, std::string_view where,
                   const std::optional<TimeUnit>& unit) {
  if (!OnTimeAxis(input, file)) return time;
  const auto span = *file.Span();
  const auto tolerance = SpanTolerance(span);
  if (time >= span.first - tolerance && time <= span.second + tolerance) {
    return std::clamp(time, span.first, span.second);
  }
  if (input.outside == OutsideSpan::Hold) return time;
  // the span from its earlier end: the oldest date first, or the file's
  // axis forwards
  Fail(object.At("file"),
       std::format("spans {} to {}, which leaves out {}, {}: give a file "
                   "that covers it, or \"outside_span\": \"hold\" on the "
                   "input to hold the file's first or last stored values "
                   "outside its span",
                   unit ? std::format("{:g}", span.first / unit->seconds)
                        : std::format("{:g}", 0.0 - span.first / KYR),
                   AxisLabel(span.second, unit), AxisLabel(time, unit),
                   where));
}

std::vector<double> ReadValues(const FileInput& input,
                               const io::FieldHistory& file, const Json& object,
                               double time, std::string_view where,
                               const std::optional<TimeUnit>& unit,
                               bool before) {
  const auto at = CoveredTime(input, file, object, time, where, unit);
  try {
    return before ? file.ValuesBefore(input.variable, at)
                  : file.Values(input.variable, at);
  } catch (const std::runtime_error& error) {
    Fail(object.At("file"), error.what());
  }
}

bool SameFile(const FileInput& a, const FileInput& b) {
  auto fault = std::error_code{};
  return a.format == b.format &&
         std::filesystem::equivalent(a.file, b.file, fault) && !fault;
}

void CheckDate(const FileInput& input, const io::FieldHistory& file,
               const Json& object, std::string_view key) {
  const auto timed = file.Span().has_value();
  if (timed && !input.date) {
    Fail(object, std::format("names a file with a time axis ({}): give the "
                             "date it is read at, `{}`",
                             input.file.string(), key));
  }
  if (!timed && input.date) {
    Fail(object.At(key),
         std::format("dates a file without a time axis ({}), the same at "
                     "every date: drop it",
                     input.file.string()));
  }
}

std::string DescribeInput(const FileInput& input, std::string_view slice) {
  return std::format("{}:{} ({})", input.file.string(), input.variable, slice);
}

}  // namespace sl3dpp::app
