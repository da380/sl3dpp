#pragma once

/**
 * @file EarthModel.hpp
 * @brief A layered radial earth model built from a deck, with its reference
 * gravity.
 *
 * Units: SI in, SI out — and non-dimensional everything inside. The deck's
 * SI values are non-dimensionalised once, on construction, under the
 * model's Scales (by default the geophysical scales of the deck's own
 * surface radius, making G = 1 and the surface radius 1). Every public
 * method here takes and returns SI, converting at the call boundary, so a
 * user never sees the scales. The library's own components work in the
 * non-dimensional numbers through Dimensionless(), and anything written
 * out is re-dimensionalised at the point of writing.
 *
 * Each deck column becomes one interpolant per layer through that layer's
 * knots (a column NaN throughout a layer is absent from it; NaN on part of
 * one is refused), evaluated right-continuously at layer boundaries with
 * both one-sided values available.
 *
 * The reference gravity follows planetmodel's `mesh1d.gravity`: for a
 * spherically symmetric density, g(r) = G M(r) / r^2 with
 * M(r) = 4 pi int_0^r rho s^2 ds, the integrals taken segment-exactly on
 * the density interpolant (Gauss-Legendre of sufficient order on each
 * spline segment). The potential is
 * phi(r) = -G [ M(r)/r + 4 pi int_r^b rho s ds ] inside the body and
 * -G M / r outside, so g = d(phi)/dr > 0 matches the pyslfp sign
 * convention. Pressure is hydrostatic, p(r) = int_r^b rho g ds.
 */

#include <Interpolation/CubicSpline.hpp>
#include <cstddef>
#include <expected>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Scales.hpp"

namespace sl3dpp {

/// Why an EarthModel could not be built from a deck.
struct ModelError {
  std::string message;
};

/// How a layer's knots are joined. Cubic (not-a-knot) matches scipy's and
/// hence planetmodel's default; Natural matches the old Fortran codes. A
/// two-knot layer is a straight line under either.
enum class SplineKind { Cubic, Natural };

/// Which one-sided value answers at a layer boundary. Above (the layer
/// starting there) is the default everywhere, matching planetmodel's
/// right-continuous evaluation.
enum class Side { Below, Above };

class EarthModel {
 public:
  struct Options {
    SplineKind spline = SplineKind::Cubic;
    /// The internal scales; by default the geophysical scales of the
    /// deck's surface radius (G = 1, surface at 1).
    std::optional<Scales> scales = {};
    /// Column dimensions beyond the built-in vocabulary (rho, the
    /// velocities, kappa/mu, viscosity, and the dimensionless qkappa,
    /// qmu, eta); an unknown column is taken dimensionless.
    std::vector<std::pair<std::string, Dimensions>> dimensions = {};
  };

  /// A model from a deck (SI); refused when the deck lacks `rho`, or a
  /// column is NaN on part of a layer.
  [[nodiscard]] static std::expected<EarthModel, ModelError> FromDeck(
      const Deck& deck, Options options);
  [[nodiscard]] static std::expected<EarthModel, ModelError> FromDeck(
      const Deck& deck) {
    return FromDeck(deck, Options{});
  }

  /// The scales everything inside is stored under.
  [[nodiscard]] const Scales& ModelScales() const { return scales_; }

  // -- structure (SI) ---------------------------------------------------------

  [[nodiscard]] std::size_t LayerCount() const { return layers_.size(); }

  /// The boundary radii in metres, LayerCount() + 1 of them, centre
  /// outward.
  [[nodiscard]] std::span<const double> Boundaries() const {
    return boundariesSI_;
  }

  /// The surface radius in metres.
  [[nodiscard]] double SurfaceRadius() const { return boundariesSI_.back(); }

  /// The layer owning r (metres): the one whose half-open interval
  /// [lo, hi) contains it (side Above), or (lo, hi] (side Below); clipped
  /// to the model.
  [[nodiscard]] std::size_t LayerOf(double r, Side side = Side::Above) const {
    return LayerOfND(r / scales_.Length(), side);
  }

  /// Whether a layer is fluid: a shear indicator (`mu`, `vs` or `vsv`) is
  /// present and identically zero on it. A model with no such column has no
  /// fluid layers.
  [[nodiscard]] bool IsFluid(std::size_t layer) const {
    return layers_[layer].fluid;
  }

  /// Whether a layer is viscoelastic: `viscosity` is present and positive
  /// throughout it.
  [[nodiscard]] bool IsViscoelastic(std::size_t layer) const {
    return layers_[layer].viscoelastic;
  }

  // -- fields (SI) ------------------------------------------------------------

  [[nodiscard]] std::span<const std::string> FieldNames() const {
    return names_;
  }

  [[nodiscard]] bool Has(const std::string& name) const;

  /// Whether a layer carries the field.
  [[nodiscard]] bool Has(const std::string& name, std::size_t layer) const;

  /// The field's SI value at r (metres), from the layer `side` names at a
  /// boundary; throws std::out_of_range for an unknown name and
  /// std::domain_error when that layer lacks the field.
  [[nodiscard]] double Evaluate(const std::string& name, double r,
                                Side side = Side::Above) const;

  /// The field's SI value at r (metres) from a named layer's interpolant
  /// (which continues smoothly slightly outside its interval); `nu` asks
  /// for the nu-th radial derivative (0, 1 or 2).
  [[nodiscard]] double EvaluateInLayer(const std::string& name,
                                       std::size_t layer, double r,
                                       int nu = 0) const;

  /// The density at r, kg m^-3; every model has one.
  [[nodiscard]] double Density(double r, Side side = Side::Above) const {
    return Evaluate("rho", r, side);
  }

  // -- reference gravity (SI)
  // --------------------------------------------------

  /// The mass inside r metres (the whole body by default), kg.
  [[nodiscard]] double Mass() const;
  [[nodiscard]] double Mass(double r) const;

  /// The moment of inertia about any axis, (8 pi / 3) int rho r^4 dr,
  /// kg m^2.
  [[nodiscard]] double MomentOfInertia() const;

  /// g(r) = G M(r) / r^2 in m s^-2; zero at the centre, G M / r^2 outside.
  [[nodiscard]] double Gravity(double r) const;

  /// The reference gravitational potential in m^2 s^-2: negative,
  /// continuous, phi -> -G M / r outside the body, g = d(phi)/dr.
  [[nodiscard]] double Potential(double r) const;

  /// The hydrostatic pressure int_r^b rho g ds in Pa; zero at the surface.
  [[nodiscard]] double Pressure(double r) const;

  // -- the non-dimensional view
  // ------------------------------------------------

  /**
   * @brief The model in its internal non-dimensional numbers: the currency
   * of the library's own components (meshes, operators, steppers), which
   * never convert. Radii in, values out, all under ModelScales(); the
   * methods mirror the SI surface above.
   */
  class Dimensionless {
   public:
    [[nodiscard]] double G() const { return model_->G_; }
    [[nodiscard]] std::span<const double> Boundaries() const {
      return model_->boundaries_;
    }
    [[nodiscard]] double SurfaceRadius() const {
      return model_->boundaries_.back();
    }
    [[nodiscard]] std::size_t LayerOf(double r, Side side = Side::Above) const {
      return model_->LayerOfND(r, side);
    }
    [[nodiscard]] double Evaluate(const std::string& name, double r,
                                  Side side = Side::Above) const {
      return model_->EvaluateND(name, r, side);
    }
    [[nodiscard]] double EvaluateInLayer(const std::string& name,
                                         std::size_t layer, double r,
                                         int nu = 0) const {
      return model_->EvaluateInLayerND(name, layer, r, nu);
    }
    [[nodiscard]] double Density(double r, Side side = Side::Above) const {
      return model_->EvaluateND("rho", r, side);
    }
    [[nodiscard]] double Mass() const { return model_->massBelow_.back(); }
    [[nodiscard]] double Mass(double r) const { return model_->MassND(r); }
    [[nodiscard]] double MomentOfInertia() const {
      return model_->MomentOfInertiaND();
    }
    [[nodiscard]] double Gravity(double r) const {
      return model_->GravityND(r);
    }
    [[nodiscard]] double Potential(double r) const {
      return model_->PotentialND(r);
    }
    [[nodiscard]] double Pressure(double r) const {
      return model_->PressureND(r);
    }

   private:
    friend class EarthModel;
    explicit Dimensionless(const EarthModel* model) : model_(model) {}
    const EarthModel* model_;
  };

  /// Valid while the model lives.
  [[nodiscard]] Dimensionless DimensionlessView() const {
    return Dimensionless(this);
  }

 private:
  using View = std::ranges::owning_view<std::vector<double>>;
  using Spline = Interpolation::CubicSpline<View, View>;

  struct Layer {
    double lo = 0;
    double hi = 0;
    bool fluid = false;
    bool viscoelastic = false;
  };

  EarthModel() = default;

  [[nodiscard]] std::size_t LayerOfND(double r, Side side) const;
  [[nodiscard]] double EvaluateND(const std::string& name, double r,
                                  Side side) const;
  [[nodiscard]] double EvaluateInLayerND(const std::string& name,
                                         std::size_t layer, double r,
                                         int nu = 0) const;
  [[nodiscard]] double MassND(double r) const;
  [[nodiscard]] double MomentOfInertiaND() const;
  [[nodiscard]] double GravityND(double r) const;
  [[nodiscard]] double PotentialND(double r) const;
  [[nodiscard]] double PressureND(double r) const;

  [[nodiscard]] std::size_t IndexOf(const std::string& name) const;
  [[nodiscard]] const Spline* Find(const std::string& name,
                                   std::size_t layer) const;
  [[nodiscard]] const Spline& Require(const std::string& name,
                                      std::size_t layer) const;

  /// 4 pi int_a^b rho(s) s^p ds within one layer, segment-exact,
  /// non-dimensional.
  [[nodiscard]] double ShellMoment(std::size_t layer, double a, double b,
                                   int p) const;

  Scales scales_;
  double G_ = 1.0;
  std::vector<double> boundaries_;    // non-dimensional
  std::vector<double> boundariesSI_;  // the deck's, verbatim
  std::vector<Layer> layers_;         // non-dimensional intervals
  std::vector<std::string> names_;
  std::vector<Dimensions> dimensions_;  // parallel to names_
  // fields_[field][layer], absent where the layer lacks the field;
  // non-dimensional abscissae and ordinates.
  std::vector<std::vector<std::optional<Spline>>> fields_;
  // The non-dimensional mass inside each boundary radius.
  std::vector<double> massBelow_;
};

}  // namespace sl3dpp
