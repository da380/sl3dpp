#pragma once

/**
 * @file ModelField.hpp
 * @brief The io/ boundary for a named scalar field of a planetmodel 3D
 * model file taken as it stands: its values as a function of the radius
 * and the direction, with no reference split from it. What a perturbation
 * of a model's parameters is read as (a change of ln viscosity, of a
 * modulus), where ModelViscosity and ModelModuli read the parameters
 * themselves about a radial reference.
 *
 * Units: radii are non-dimensional under the EarthModel's ModelScales(),
 * as the other readers'. The values are SI under the dimensions the file
 * declares for the field, or under the reader's `dimensions` where it
 * declares none; a reader given no dimensions takes the stored numbers as
 * they are, a pure number. The file's body must be the model's, and every
 * model layer the file gives the field on must lie within one file layer.
 *
 * Errors and the functor's validity are as ModelViscosity's:
 * planetmodel::Error is returned as IoError on reading; the functor
 * evaluates inside the body read at load and throws off it.
 */

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/io/Error.hpp"

namespace sl3dpp::io {

class ModelField {
 public:
  /// Reads the file's field `name` against the model's geometry; refused
  /// when the file cannot be read, holds no such field on any layer, or
  /// its body or layering disagrees with the model's. With `dimensions`
  /// the values are SI, under the field's declared dimensions or, where
  /// the file declares none, under these; without, the stored numbers.
  [[nodiscard]] static std::expected<ModelField, IoError> Read(
      const std::filesystem::path& path, const EarthModel& model,
      const std::string& name, std::optional<Dimensions> dimensions = {});

  /// Read, but for a file that holds no such field on any layer, which is
  /// nothing rather than an error: a field a file may leave out.
  [[nodiscard]] static std::expected<std::optional<ModelField>, IoError>
  ReadIfHeld(const std::filesystem::path& path, const EarthModel& model,
             const std::string& name,
             std::optional<Dimensions> dimensions = {});

  /// The names of the fields the file holds on any layer, each once, in
  /// order: what a reader that takes a file's fields by name holds the
  /// file's own against. Refused when the file cannot be read.
  [[nodiscard]] static std::expected<std::vector<std::string>, IoError>
  FieldNames(const std::filesystem::path& path);

  /// The field at a non-dimensional radius and a direction, in the file
  /// layer that owns the radius (the one above at a boundary); zero where
  /// that layer does not hold the field. Valid beyond this object's
  /// lifetime.
  [[nodiscard]] std::function<double(double, double, double)> Value() const;

  /// The field at a non-dimensional radius within the model's layer
  /// `layer`, a radius on the layer's boundary read from inside it; zero
  /// where the layer does not hold the field.
  [[nodiscard]] double ValueIn(std::size_t layer, double r, double theta = 0.0,
                               double phi = 0.0) const;

  /// Whether the model's layer holds the field.
  [[nodiscard]] bool Holds(std::size_t layer) const;

  /// Whether the field is a function of the radius alone on every layer.
  [[nodiscard]] bool IsRadial() const;

 private:
  struct Impl;
  explicit ModelField(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

}  // namespace sl3dpp::io
