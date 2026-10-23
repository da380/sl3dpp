#pragma once

/**
 * @file FieldHistory.hpp
 * @brief The io/ boundary for a planetmodel surface file: its fields
 * resampled once onto a SurfaceGrid and served at any time, SI in and
 * out.
 *
 * A surface file (schema `planetmodel.surface/1`) holds named scalar
 * fields on one angular grid, static or on a time axis — an ice history
 * as pyslfp's `export_ice_history` writes it (`ice_thickness` and
 * `topography`), a loading history, the surface kernels of a
 * measurement. `Read` resamples every stored slice onto the given grid
 * by the field's own stated lateral rule (`harmonic` is read by the
 * cubic one), using the file's polar values where it carries them — a
 * missing pole on a grid lacking it is taken as the mean of the
 * nearest colatitude row, a local estimate whose reach is the first
 * grid spacing — and converts to SI: values by the field's declared dimensions under the
 * file's scales (a field declaring none is taken dimensionless), times
 * to seconds. `Values(name, time)` then serves a field at any time by
 * the file's own time semantics (planetmodel's surface history): a
 * `node` field interpolated between its nodes or read from the latest
 * one, an `interval` field from the interval holding the time, an
 * `instant` field at its stored instants alone, a `static` field at any
 * time. Outside a field's span its end values are held by default (the
 * behaviour of pyslfp's `IceNG`); `hold_outside = false` refuses such
 * times instead.
 *
 * Fields come back colatitude-major on the grid given at `Read`, the
 * library's storage order.
 *
 * Errors: planetmodel::Error is caught in `Read` and returned as
 * IoError, keeping sl3dpp's reader style. `Values` throws
 * std::runtime_error naming the field and the reason (a time outside an
 * unheld span, a field the file lacks).
 */

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/Error.hpp"

namespace sl3dpp::io {

struct FieldHistoryOptions {
  /// Outside a field's span its end values are held; false refuses the
  /// time instead. Fields served at stored times alone (an `instant`
  /// field, a `node` field stating no time rule) always refuse.
  bool hold_outside = true;
};

class FieldHistory {
 public:
  /// Reads a surface file and resamples its fields onto `grid`; refused
  /// when the file cannot be read or is not a surface file, a field
  /// states an unknown rule, or the grid cannot carry the resampling.
  [[nodiscard]] static std::expected<FieldHistory, IoError> Read(
      const std::filesystem::path& path, const SurfaceGrid& grid,
      const FieldHistoryOptions& options = {});

  /// The fields, by name, polar values not counted among them.
  [[nodiscard]] const std::vector<std::string>& Names() const;

  /// Whether the file holds a field.
  [[nodiscard]] bool Has(std::string_view name) const;

  /// The first and last time node in SI seconds; empty where the file
  /// has no time axis.
  [[nodiscard]] std::optional<std::pair<double, double>> Span() const;

  /// A field's SI values at an SI time, colatitude-major on the grid
  /// given at Read; a static field ignores the time.
  [[nodiscard]] std::vector<double> Values(std::string_view name,
                                           double time) const;

  /// The file's `source` attribute: the code that wrote it.
  [[nodiscard]] const std::string& Source() const;

 private:
  struct Impl;
  explicit FieldHistory(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

}  // namespace sl3dpp::io
