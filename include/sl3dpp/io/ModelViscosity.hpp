#pragma once

/**
 * @file ModelViscosity.hpp
 * @brief The io/ boundary for a planetmodel 3D model file's viscosity
 * (stage 11 WP2): eta(x) = eta_bar(r) exp(delta(x)), with the radial
 * reference as Rheology::Maxwell's per-layer map and delta as the
 * functor Viscoelastic3D takes.
 *
 * Units: the file's numbers are in its own scales (file.GetScales());
 * everything returned here is non-dimensional under the EarthModel's
 * ModelScales(), radii in and viscosities out alike (delta, a log
 * ratio, needs no conversion). The file's body must be the model's:
 * its ends are checked on reading, and every model layer the file
 * gives a viscosity must lie within one file layer.
 *
 * The reference eta_bar(r), per viscous file layer:
 *  - GeometricMean (the default): exp of the lateral mean of ln eta
 *    under the file grid's weights, taken at the layer's radial nodes
 *    and read between them by the field's own radial rule, so delta
 *    has zero lateral mean at the nodes — the within-step iteration's
 *    preference (plan.md §3). Works for any viscosity, perturbed or
 *    fully 3D; needs the file's grid to carry weights.
 *  - FileRadial: the file's own radial reference, for a viscosity
 *    stored as a perturbation of one (what planetmodel's conversion
 *    writes); refused when the stored reference varies laterally.
 *  On a log-kind perturbation of zero lateral mean the two coincide
 *  to rounding. A wholly radial viscosity takes itself as reference
 *  and delta is identically zero on its layer.
 *
 * Errors: planetmodel::Error is caught here and returned as IoError,
 * keeping sl3dpp's reader style. The returned callables evaluate
 * inside the body read at load; off it beyond the file's tolerance
 * they throw planetmodel::Error (a functor cannot return expected).
 * At a layer boundary they read the layer above (planetmodel's
 * right-continuous rule); the supported memory points are the Gauss
 * set, which is interior — a layer-aware delta overload is the
 * stage-11 follow-up if GLL memory nodes on boundaries are wanted.
 */

#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/io/Error.hpp"

namespace sl3dpp::io {

/// The radial reference a 3D viscosity is split against.
enum class ViscosityReference { GeometricMean, FileRadial };

struct ModelViscosityOptions {
  ViscosityReference reference = ViscosityReference::GeometricMean;
};

class ModelViscosity {
 public:
  /// Reads the file's viscosity against the model's geometry; refused
  /// when the file cannot be read, holds no viscosity, its body or
  /// layering disagrees with the model's, or the chosen reference
  /// cannot be built.
  [[nodiscard]] static std::expected<ModelViscosity, IoError> Read(
      const std::filesystem::path& path, const EarthModel& model,
      const ModelViscosityOptions& options = {});

  /// eta_bar per model layer the file holds a viscosity on, each a
  /// function of non-dimensional radius: Rheology::Maxwell's map.
  [[nodiscard]] const std::map<std::size_t, RadialValue>& Viscosity() const;

  /// delta(r, theta, phi) at a non-dimensional radius: the functor
  /// Viscoelastic3D samples. Zero outside the viscous layers and
  /// wherever the file's viscosity is radial. Valid beyond this
  /// object's lifetime (the file's fields are shared, immutable and
  /// thread-safe).
  [[nodiscard]] std::function<double(double, double, double)> Delta() const;

  /// Whether the file's viscosity is radial everywhere, so that Delta()
  /// is identically zero and the 1D machinery suffices.
  [[nodiscard]] bool IsRadial() const;

 private:
  struct Impl;
  explicit ModelViscosity(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

}  // namespace sl3dpp::io
