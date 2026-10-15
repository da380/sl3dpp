#include "sl3dpp/EarthModel.hpp"

#include <GaussQuad/All>
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

namespace sl3dpp {

namespace {

/// The dimensions a column name implies: planetmodel's vocabulary, with the
/// caller's extensions taking precedence.
Dimensions VocabularyDimensions(
    const std::string& name,
    std::span<const std::pair<std::string, Dimensions>> extra) {
  for (const auto& [known, dimensions] : extra) {
    if (known == name) return dimensions;
  }
  if (name == "rho") return DensityDimensions;
  if (name == "vp" || name == "vs" || name == "vpv" || name == "vsv" ||
      name == "vph" || name == "vsh") {
    return VelocityDimensions;
  }
  if (name == "kappa" || name == "mu") return ModulusDimensions;
  if (name == "viscosity") return ViscosityDimensions;
  return {};  // qkappa, qmu, eta, and anything unknown: dimensionless
}

/// int_a^b f ds by an n-point Gauss-Legendre rule, cached per n.
template <int n>
double IntegrateSegment(const auto& f, double a, double b) {
  static const auto rule = GaussQuad::GaussLegendreQuadrature1D<double>(n);
  const double mid = 0.5 * (a + b);
  const double jac = 0.5 * (b - a);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += rule.Weights()[i] * f(mid + jac * rule.Points()[i]);
  }
  return jac * sum;
}

constexpr Dimensions MassDimensions{.mass = 1};
constexpr Dimensions InertiaDimensions{.mass = 1, .length = 2};

}  // namespace

std::expected<EarthModel, ModelError> EarthModel::FromDeck(const Deck& deck,
                                                           Options options) {
  EarthModel model;
  const auto boundaries = deck.Boundaries();
  model.boundariesSI_.assign(boundaries.begin(), boundaries.end());
  model.scales_ =
      options.scales.value_or(Scales::Geophysical(model.boundariesSI_.back()));
  model.G_ = model.scales_.GravitationalConstant();
  const double length = model.scales_.Length();

  model.boundaries_ = model.boundariesSI_;
  for (auto& b : model.boundaries_) b /= length;

  const auto layers = deck.Layers();
  model.layers_.resize(layers.size());
  for (std::size_t i = 0; i < layers.size(); ++i) {
    model.layers_[i].lo = model.boundaries_[i];
    model.layers_[i].hi = model.boundaries_[i + 1];
  }

  if (!deck.Has("rho")) {
    std::string have;
    for (const auto& name : deck.Names()) {
      have += have.empty() ? name : ", " + name;
    }
    return std::unexpected(ModelError{std::format(
        "the deck has no density column 'rho'; it has [{}]", have)});
  }

  // One interpolant per (field, layer) through the layer's knots,
  // non-dimensionalised by the column's dimensions — the once-only
  // conversion at the read boundary.
  const auto radii = deck.Radius();
  for (const auto& name : deck.Names()) {
    const auto dimensions = VocabularyDimensions(name, options.dimensions);
    const double factor = model.scales_.Factor(dimensions);
    const auto values = deck.Column(name);
    std::vector<std::optional<Spline>> perLayer(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const auto [first, last] = layers[i];
      const auto n = last - first;
      std::size_t nans = 0;
      for (std::size_t k = first; k < last; ++k) {
        nans += std::isnan(values[k]) ? 1 : 0;
      }
      if (nans == n) continue;  // absent from this layer
      if (nans > 0) {
        return std::unexpected(ModelError{std::format(
            "column '{}' is NaN on part of layer {}; a column is absent "
            "from a layer only when NaN throughout it",
            name, i)});
      }
      std::vector<double> x(n), y(n);
      for (std::size_t k = 0; k < n; ++k) {
        x[k] = radii[first + k] / length;
        y[k] = values[first + k] / factor;
      }
      // Two knots give a straight line under every kind, and not-a-knot
      // needs interior knots to constrain, so short layers fall back to
      // Natural (which reproduces the line exactly; a three-knot layer
      // then differs from scipy's parabola — accepted, documented here).
      const auto condition = (options.spline == SplineKind::Cubic && n >= 4)
                                 ? Interpolation::BoundaryCondition::NotAKnot
                                 : Interpolation::BoundaryCondition::Natural;
      try {
        perLayer[i].emplace(View(std::move(x)), View(std::move(y)), condition,
                            0.0, 0.0);
      } catch (const std::exception& e) {
        return std::unexpected(ModelError{
            std::format("column '{}', layer {}: {}", name, i, e.what())});
      }
    }
    model.names_.push_back(name);
    model.dimensions_.push_back(dimensions);
    model.fields_.push_back(std::move(perLayer));
  }

  // Layer classification. Fluid: a shear indicator present and identically
  // zero; viscoelastic: viscosity present and positive throughout.
  const char* shear = nullptr;
  for (const char* candidate : {"mu", "vs", "vsv"}) {
    if (deck.Has(candidate)) {
      shear = candidate;
      break;
    }
  }
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto [first, last] = layers[i];
    if (shear != nullptr) {
      const auto values = deck.Column(shear);
      model.layers_[i].fluid =
          std::all_of(values.begin() + first, values.begin() + last,
                      [](double v) { return v == 0.0; });
    }
    if (deck.Has("viscosity")) {
      const auto values = deck.Column("viscosity");
      model.layers_[i].viscoelastic =
          std::all_of(values.begin() + first, values.begin() + last,
                      [](double v) { return v > 0.0; });
    }
  }

  // The mass inside every boundary, accumulated layer by layer.
  model.massBelow_.resize(model.boundaries_.size(), 0.0);
  for (std::size_t i = 0; i < model.layers_.size(); ++i) {
    model.massBelow_[i + 1] =
        model.massBelow_[i] +
        model.ShellMoment(i, model.layers_[i].lo, model.layers_[i].hi, 2);
  }

  return model;
}

// -- lookup ---------------------------------------------------------------

std::size_t EarthModel::IndexOf(const std::string& name) const {
  auto found = std::ranges::find(names_, name);
  if (found == names_.end()) {
    throw std::out_of_range(std::format("the model has no field '{}'", name));
  }
  return static_cast<std::size_t>(found - names_.begin());
}

bool EarthModel::Has(const std::string& name) const {
  return std::ranges::find(names_, name) != names_.end();
}

bool EarthModel::Has(const std::string& name, std::size_t layer) const {
  return Find(name, layer) != nullptr;
}

const EarthModel::Spline* EarthModel::Find(const std::string& name,
                                           std::size_t layer) const {
  const auto& spline = fields_[IndexOf(name)].at(layer);
  return spline ? &*spline : nullptr;
}

const EarthModel::Spline& EarthModel::Require(const std::string& name,
                                              std::size_t layer) const {
  const auto* spline = Find(name, layer);
  if (spline == nullptr) {
    throw std::domain_error(
        std::format("layer {} holds no field '{}'", layer, name));
  }
  return *spline;
}

std::size_t EarthModel::LayerOfND(double r, Side side) const {
  const auto& b = boundaries_;
  auto it = side == Side::Above ? std::ranges::upper_bound(b, r)
                                : std::ranges::lower_bound(b, r);
  const auto i = static_cast<std::ptrdiff_t>(it - b.begin()) - 1;
  return static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(
      i, 0, static_cast<std::ptrdiff_t>(layers_.size()) - 1));
}

// -- the non-dimensional core ----------------------------------------------

double EarthModel::EvaluateND(const std::string& name, double r,
                              Side side) const {
  return Require(name, LayerOfND(r, side)).Evaluate(r);
}

double EarthModel::EvaluateInLayerND(const std::string& name, std::size_t layer,
                                     double r, int nu) const {
  const auto& spline = Require(name, layer);
  switch (nu) {
    case 0:
      return spline.Evaluate(r);
    case 1:
      return spline.Evaluate<1>(r);
    case 2:
      return spline.Evaluate<2>(r);
    default:
      throw std::invalid_argument(
          std::format("nu must be 0, 1 or 2, got {}", nu));
  }
}

double EarthModel::ShellMoment(std::size_t layer, double a, double b,
                               int p) const {
  const auto& rho = Require("rho", layer);
  a = std::max(a, layers_[layer].lo);
  b = std::min(b, layers_[layer].hi);
  if (b <= a) return 0.0;
  // Segment by segment on the spline, so the rule's exactness (degree 9
  // with five points) covers the cubic-times-s^p integrand.
  double sum = 0.0;
  for (std::size_t i = 0; i + 1 < rho.Size(); ++i) {
    const double lo = std::max(a, rho.Node(i));
    const double hi = std::min(b, rho.Node(i + 1));
    if (hi <= lo) continue;
    sum += IntegrateSegment<5>(
        [&](double s) { return rho.Evaluate(s) * std::pow(s, p); }, lo, hi);
  }
  return 4.0 * std::numbers::pi * sum;
}

double EarthModel::MassND(double r) const {
  if (r >= boundaries_.back()) return massBelow_.back();
  if (r <= boundaries_.front()) return 0.0;
  const auto layer = LayerOfND(r, Side::Above);
  return massBelow_[layer] + ShellMoment(layer, layers_[layer].lo, r, 2);
}

double EarthModel::MomentOfInertiaND() const {
  double sum = 0.0;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    sum += ShellMoment(i, layers_[i].lo, layers_[i].hi, 4);
  }
  return (2.0 / 3.0) * sum;  // (8 pi / 3) int rho s^4: ShellMoment has 4 pi
}

double EarthModel::GravityND(double r) const {
  if (r <= 0.0) return 0.0;
  return G_ * MassND(r) / (r * r);
}

double EarthModel::PotentialND(double r) const {
  const double b = boundaries_.back();
  if (r >= b) return -G_ * massBelow_.back() / r;
  // -G [ M(r)/r + 4 pi int_r^b rho s ds ], the M/r term vanishing with r.
  double outer = 0.0;
  const auto layer = LayerOfND(std::max(r, boundaries_.front()), Side::Above);
  outer += ShellMoment(layer, r, layers_[layer].hi, 1);
  for (std::size_t j = layer + 1; j < layers_.size(); ++j) {
    outer += ShellMoment(j, layers_[j].lo, layers_[j].hi, 1);
  }
  const double inner = r > 0.0 ? MassND(r) / r : 0.0;
  return -G_ * (inner + outer);
}

double EarthModel::PressureND(double r) const {
  const double b = boundaries_.back();
  if (r >= b) return 0.0;
  r = std::max(r, boundaries_.front());
  // int_r^b rho g ds, per spline segment; the integrand is smooth within a
  // segment but not polynomial, so the eight-point rule is merely accurate.
  double sum = 0.0;
  for (std::size_t i = LayerOfND(r, Side::Above); i < layers_.size(); ++i) {
    const auto& rho = Require("rho", i);
    const double a = std::max(r, layers_[i].lo);
    for (std::size_t k = 0; k + 1 < rho.Size(); ++k) {
      const double lo = std::max(a, rho.Node(k));
      const double hi = rho.Node(k + 1);
      if (hi <= lo) continue;
      sum += IntegrateSegment<8>(
          [&](double s) { return rho.Evaluate(s) * GravityND(s); }, lo, hi);
    }
  }
  return sum;
}

// -- the SI surface ---------------------------------------------------------

double EarthModel::Evaluate(const std::string& name, double r,
                            Side side) const {
  const auto field = IndexOf(name);
  return EvaluateND(name, r / scales_.Length(), side) *
         scales_.Factor(dimensions_[field]);
}

double EarthModel::EvaluateInLayer(const std::string& name, std::size_t layer,
                                   double r, int nu) const {
  const auto field = IndexOf(name);
  // Each radial derivative divides the quantity's dimensions by a length.
  double factor = scales_.Factor(dimensions_[field]);
  for (int k = 0; k < nu; ++k) factor /= scales_.Length();
  return EvaluateInLayerND(name, layer, r / scales_.Length(), nu) * factor;
}

double EarthModel::Mass() const {
  return massBelow_.back() * scales_.Factor(MassDimensions);
}

double EarthModel::Mass(double r) const {
  return MassND(r / scales_.Length()) * scales_.Factor(MassDimensions);
}

double EarthModel::MomentOfInertia() const {
  return MomentOfInertiaND() * scales_.Factor(InertiaDimensions);
}

double EarthModel::Gravity(double r) const {
  return GravityND(r / scales_.Length()) *
         scales_.Factor(AccelerationDimensions);
}

double EarthModel::Potential(double r) const {
  return PotentialND(r / scales_.Length()) *
         scales_.Factor(PotentialDimensions);
}

double EarthModel::Pressure(double r) const {
  return PressureND(r / scales_.Length()) * scales_.Factor(ModulusDimensions);
}

}  // namespace sl3dpp
