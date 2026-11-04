#pragma once

/**
 * @file ModelModuli.hpp
 * @brief The io/ boundary for a planetmodel 3D model file's elastic
 * moduli: the lateral perturbations s(x) and b(x) of
 * Viscoelastic3D::Options::moduli, mu(x) = mu_bar(r) (1 + s(x)) and
 * kappa(x) = kappa_bar(r) (1 + b(x)), from the file's `mu` and `kappa`
 * fields (planetmodel's isotropic moduli vocabulary; either may be
 * absent, its perturbation then empty).
 *
 * The reference each modulus is split against, per file layer:
 *  - LateralMean (the default): the lateral mean of the modulus under
 *    the file grid's weights at the layer's radial nodes, read between
 *    them by the field's own radial rule, so the perturbation has zero
 *    lateral mean at the nodes. The 1D structure the operator carries
 *    is the DECK's, so a file whose lateral mean departs from the deck
 *    is a different Earth: the reader checks nothing about that, and a
 *    model built from the deck's own radial model (planetmodel's
 *    perturbation of a radial model) is what it expects.
 *  - FileRadial: the file's own radial reference, for a modulus stored
 *    as a perturbation of one (what planetmodel's conversion writes);
 *    refused when the stored reference varies laterally.
 *  A wholly radial modulus takes itself as reference and its
 *  perturbation is identically zero on its layer.
 *
 * Units: the file's numbers are in its own scales; a relative
 * perturbation needs no conversion, and only the radii are mapped. The
 * file's body must be the model's, and every model layer the file
 * gives a modulus on must lie within one file layer. Errors and the
 * functors' validity are as ModelViscosity's: planetmodel::Error is
 * returned as IoError on reading; the functors evaluate inside the body
 * read at load and throw off it.
 */

#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/Error.hpp"

namespace sl3dpp::io {

/// The radial reference a 3D modulus is split against.
enum class ModulusReference { LateralMean, FileRadial };

struct ModelModuliOptions {
  ModulusReference reference = ModulusReference::LateralMean;
};

class ModelModuli {
 public:
  /// Reads the file's moduli against the model's geometry; refused when
  /// the file cannot be read, holds neither `mu` nor `kappa`, its body
  /// or layering disagrees with the model's, or the chosen reference
  /// cannot be built.
  [[nodiscard]] static std::expected<ModelModuli, IoError> Read(
      const std::filesystem::path& path, const EarthModel& model,
      const ModelModuliOptions& options = {});

  /// s(r, theta, phi) at a non-dimensional radius, the relative
  /// perturbation of the shear modulus: Viscoelastic3D::Options::
  /// moduli.shear. Empty when the file holds no `mu`; zero outside the
  /// layers that hold one and wherever it is radial.
  [[nodiscard]] std::function<double(double, double, double)> Shear() const;
  /// b(r, theta, phi) likewise for the bulk modulus (`kappa`).
  [[nodiscard]] std::function<double(double, double, double)> Bulk() const;
  /// Whether every modulus the file holds is radial, so that both
  /// perturbations are identically zero.
  [[nodiscard]] bool IsRadial() const;

  struct Impl;  ///< the reading's state, shared by the functors

 private:
  explicit ModelModuli(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
};

/// The reader's inverse, for building a 3D elastic model from a radial
/// one: writes a model file holding the model's density, its bulk
/// modulus kappa(r) (1 + bulk(x)) on every layer and its shear modulus
/// mu(r) (1 + shear(x)) on the solid layers, the radial moduli the
/// Voigt-averaged isotropic ones of the model's elastic fields, on
/// `grid`'s colatitudes and longitudes laterally and elements of about
/// `elementMetres` with `ngll` nodes radially (the lateral rule linear,
/// as a sampled field). The functors take SI metres and the grid's
/// radians, and may be empty (no perturbation). A file this writes
/// reads back through ModelModuli with the LateralMean reference as
/// the perturbations given, less their lateral means.
struct WriteModuliOptions {
  std::string name = {};
  double elementMetres = 100e3;
  std::size_t ngll = 5;
};
[[nodiscard]] std::expected<void, IoError> WriteModuli(
    const std::filesystem::path& path, const EarthModel& model,
    const SurfaceGrid& grid,
    const std::function<double(double, double, double)>& shear,
    const std::function<double(double, double, double)>& bulk,
    const WriteModuliOptions& options = {});

}  // namespace sl3dpp::io
