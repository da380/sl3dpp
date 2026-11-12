#include "Files.hpp"

#include <cmath>
#include <format>
#include <optional>
#include <system_error>
#include <utility>

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
  return out;
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
      auto read = io::FieldHistory::Read(input.file, grid);
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

io::FieldHistory OpenFileInput(const FileInput& input, const SurfaceGrid& grid,
                               const Json& object) {
  auto file = ReadFile(input, grid, object);
  CheckVariable(input, file, object);
  return file;
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
