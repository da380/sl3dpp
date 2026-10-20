#include "sl3dpp/io/ModelViscosity.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <planetmodel/modelfile.hpp>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

/// One viscous file layer: the combined field, its radial reference and
/// the conversion into the model's non-dimensional numbers, all radii
/// and values in the FILE's units except toDimensionless.
struct Layer {
  std::pair<double, double> interval;
  planetmodel::ScalarField eta;
  std::function<double(double)> reference;
  double toDimensionless = 1.0;
  bool radial = false;
};

/// The file layer owning r: the one above at a shared boundary
/// (planetmodel's right-continuous rule), clipped to the body.
int FileLayerOf(const std::vector<double>& boundaries, double r) {
  const auto i =
      std::upper_bound(boundaries.begin(), boundaries.end(), r) -
      boundaries.begin() - 1;
  return static_cast<int>(std::clamp<std::ptrdiff_t>(
      i, 0, static_cast<std::ptrdiff_t>(boundaries.size()) - 2));
}

/// The layer's slice of the part's node radii, as the library's own
/// per-layer pieces take it.
std::vector<double> LayerRadii(const planetmodel::FilePart& part, int layer) {
  const auto [e0, e1] = part.Elements(layer);
  return {part.radius.begin() + static_cast<std::ptrdiff_t>(e0 * part.ngll),
          part.radius.begin() + static_cast<std::ptrdiff_t>(e1 * part.ngll)};
}

/// The radial rule the variable states.
std::expected<planetmodel::RadialRule, IoError> RuleOf(
    const planetmodel::ModelFile& file, const planetmodel::FileVariable& var) {
  const auto text = var.Text("radial_interpolation");
  if (!text) {
    return std::unexpected(IoError{std::format(
        "{}: {} states no radial_interpolation", file.Source(), var.name)});
  }
  return planetmodel::RadialRuleNamed(*text, file.Source(), var.name);
}

/// The file's own radial reference on one layer: the viscosity
/// variable's nodal values read by its stated rule. Refused when that
/// stored reference varies laterally on the layer.
std::expected<std::function<double(double)>, IoError> FileRadialReference(
    const planetmodel::ModelFile& file, int layer, double tolerance) {
  const auto& part = file.PartOf(layer);
  const auto& var = part.variables.at("viscosity");
  auto radialLayers = std::vector<int>{};
  if (const auto listed = var.Numbers("radial_layers")) {
    for (const auto x : *listed) radialLayers.push_back(static_cast<int>(x));
  }
  if (var.IsLateral() && !std::ranges::contains(radialLayers, layer)) {
    return std::unexpected(IoError{std::format(
        "{}: the viscosity on layer {} is not a perturbation of a radial "
        "reference; split it with ViscosityReference::GeometricMean",
        file.Source(), layer)});
  }
  const auto rule = RuleOf(file, var);
  if (!rule) return std::unexpected(rule.error());
  // the stored components at every node of the layer, a lateral
  // variable's from its first column, all of whose columns are the same
  const auto [e0, e1] = part.Elements(layer);
  const auto stride =
      var.IsLateral() ? part.grid->NTheta() * part.grid->NPhi() : std::size_t{1};
  auto values = std::vector<double>{};
  values.reserve((e1 - e0) * part.ngll);
  for (auto node = e0 * part.ngll; node < e1 * part.ngll; ++node) {
    const auto v = var.values[node * stride];
    if (!(v > 0.0) || !std::isfinite(v)) {
      return std::unexpected(IoError{std::format(
          "{}: the radial viscosity reference on layer {} is {} at "
          "r = {}, not a positive viscosity",
          file.Source(), layer, v, part.radius[node])});
    }
    values.push_back(v);
  }
  auto interpolant = planetmodel::RadialInterpolant(
      planetmodel::RadialLayout(LayerRadii(part, layer), part.ngll, tolerance),
      std::move(values), *rule);
  return [interpolant = std::move(interpolant)](double r) {
    return interpolant.Value(r);
  };
}

/// The lateral geometric mean per radius: exp of the weighted mean of
/// ln eta over the file's grid at each of the layer's radial nodes,
/// read between them by the field's own radial rule.
std::expected<std::function<double(double)>, IoError> GeometricMeanReference(
    const planetmodel::ModelFile& file, int layer,
    const planetmodel::ScalarField& eta, double tolerance) {
  const auto& part = file.PartOf(layer);
  if (!part.grid || !part.grid->Weights()) {
    return std::unexpected(IoError{std::format(
        "{}: part_{}'s grid carries no quadrature weights, so the lateral "
        "geometric mean cannot be taken; use ViscosityReference::FileRadial",
        file.Source(), part.index)});
  }
  const auto& grid = *part.grid;
  const auto& weights = *grid.Weights();
  auto weightSum = 0.0;
  for (const auto w : weights) weightSum += w;
  const auto rule = RuleOf(file, part.variables.at("viscosity"));
  if (!rule) return std::unexpected(rule.error());
  const auto radii = LayerRadii(part, layer);
  auto values = std::vector<double>{};
  values.reserve(radii.size());
  for (const auto r : radii) {
    auto mean = 0.0;
    for (auto i = std::size_t{0}; i < grid.NTheta(); ++i) {
      auto row = 0.0;
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        const auto v = eta.Value(r, grid.Colatitudes()[i], grid.Longitudes()[j]);
        if (!(v > 0.0) || !std::isfinite(v)) {
          return std::unexpected(IoError{std::format(
              "{}: the viscosity on layer {} is {} at r = {}, not a "
              "positive viscosity",
              file.Source(), layer, v, r)});
        }
        row += std::log(v);
      }
      mean += weights[i] * row;
    }
    values.push_back(
        std::exp(mean / (weightSum * static_cast<double>(grid.NPhi()))));
  }
  auto interpolant = planetmodel::RadialInterpolant(
      planetmodel::RadialLayout(radii, part.ngll, tolerance),
      std::move(values), *rule);
  return [interpolant = std::move(interpolant)](double r) {
    return interpolant.Value(r);
  };
}

}  // namespace

struct ModelViscosity::Impl {
  /// r_file = r_nd * toFileRadius.
  double toFileRadius = 1.0;
  std::vector<double> boundaries;  ///< the file's, in its units
  std::map<int, std::shared_ptr<const Layer>> layers;  ///< by file layer
  std::map<std::size_t, RadialValue> viscosity;        ///< by model layer
  bool radial = true;
};

std::expected<ModelViscosity, IoError> ModelViscosity::Read(
    const std::filesystem::path& path, const EarthModel& model,
    const ModelViscosityOptions& options) {
  try {
    const auto file = planetmodel::ReadModelFile(path);
    const auto& fileScales = file.GetScales();
    const auto& modelScales = model.ModelScales();
    const auto& skeleton = file.GetSkeleton();
    const auto tolerance = file.Rtol() * skeleton.Span();
    auto impl = std::make_shared<Impl>();
    impl->toFileRadius =
        modelScales.Length() / fileScales.Factor(planetmodel::units::LENGTH);
    impl->boundaries = skeleton.Boundaries();

    // The file's body must be the model's.
    const auto dimensionless = model.DimensionlessView();
    const auto inner = dimensionless.Boundaries().front() * impl->toFileRadius;
    const auto outer = dimensionless.SurfaceRadius() * impl->toFileRadius;
    if (std::abs(outer - impl->boundaries.back()) > tolerance ||
        std::abs(inner - impl->boundaries.front()) > tolerance) {
      return std::unexpected(IoError{std::format(
          "{}: its body [{}, {}] is not the model's [{}, {}] (file units)",
          file.Source(), impl->boundaries.front(), impl->boundaries.back(),
          inner, outer)});
    }

    // Each file layer holding a viscosity.
    for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers());
         ++layer) {
      if (!file.HasField(layer, "viscosity")) continue;
      const auto field = file.FieldOn(layer, "viscosity");
      const auto dims =
          field.GetDimensions().value_or(planetmodel::units::VISCOSITY);
      if (dims.temperature != 0) {
        return std::unexpected(IoError{std::format(
            "{}: the viscosity on layer {} states a temperature exponent "
            "{}, outside sl3dpp's mechanical scales",
            file.Source(), layer, dims.temperature)});
      }
      const auto toDimensionless =
          fileScales.Factor(dims) /
          modelScales.Factor(Dimensions{
              .mass = dims.mass, .length = dims.length, .time = dims.time});
      const auto radial = field.IsRadial();
      auto reference = std::function<double(double)>{};
      if (radial) {
        reference = [scalar = field.AsScalar()](double r) {
          return scalar.Value(r, 0.0, 0.0);
        };
      } else if (options.reference == ViscosityReference::FileRadial) {
        auto made = FileRadialReference(file, layer, tolerance);
        if (!made) return std::unexpected(made.error());
        reference = std::move(*made);
      } else {
        auto made =
            GeometricMeanReference(file, layer, field.AsScalar(), tolerance);
        if (!made) return std::unexpected(made.error());
        reference = std::move(*made);
      }
      impl->radial = impl->radial && radial;
      impl->layers.emplace(
          layer, std::make_shared<const Layer>(
                     Layer{field.Interval(), field.AsScalar(),
                           std::move(reference), toDimensionless, radial}));
    }
    if (impl->layers.empty()) {
      return std::unexpected(
          IoError{file.Source() + ": holds no viscosity field"});
    }

    // The model layers inside the viscous file layers, each wholly
    // within one (a crossing means the layerings disagree).
    const auto b = dimensionless.Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      const auto lo = b[i] * impl->toFileRadius;
      const auto hi = b[i + 1] * impl->toFileRadius;
      const auto k = FileLayerOf(impl->boundaries, 0.5 * (lo + hi));
      const auto found = impl->layers.find(k);
      if (found == impl->layers.end()) continue;
      const auto& interval = found->second->interval;
      if (lo < interval.first - tolerance || hi > interval.second + tolerance) {
        return std::unexpected(IoError{std::format(
            "{}: model layer {} on [{}, {}] (file units) crosses the ends "
            "of file layer {} on [{}, {}]: the layerings disagree",
            file.Source(), i, lo, hi, k, interval.first, interval.second)});
      }
      impl->viscosity.emplace(
          i, [layer = found->second, toFile = impl->toFileRadius](double r) {
            const auto rf = std::clamp(r * toFile, layer->interval.first,
                                       layer->interval.second);
            return layer->reference(rf) * layer->toDimensionless;
          });
    }
    return ModelViscosity(std::move(impl));
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

const std::map<std::size_t, RadialValue>& ModelViscosity::Viscosity() const {
  return impl_->viscosity;
}

std::function<double(double, double, double)> ModelViscosity::Delta() const {
  return [impl = impl_](double r, double theta, double phi) {
    if (impl->radial) return 0.0;
    const auto rf = r * impl->toFileRadius;
    const auto found = impl->layers.find(FileLayerOf(impl->boundaries, rf));
    if (found == impl->layers.end() || found->second->radial) return 0.0;
    const auto& layer = *found->second;
    const auto rc =
        std::clamp(rf, layer.interval.first, layer.interval.second);
    return std::log(layer.eta.Value(rc, theta, phi) / layer.reference(rc));
  };
}

bool ModelViscosity::IsRadial() const { return impl_->radial; }

}  // namespace sl3dpp::io
