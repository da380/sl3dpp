#include "sl3dpp/io/ModelModuli.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <planetmodel/modelfile.hpp>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

/// One file layer holding a modulus: the field, its radial reference,
/// all radii in the FILE's units.
struct Layer {
  std::pair<double, double> interval;
  planetmodel::ScalarField modulus;
  std::function<double(double)> reference;
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

std::vector<double> LayerRadii(const planetmodel::FilePart& part, int layer) {
  const auto [e0, e1] = part.Elements(layer);
  return {part.radius.begin() + static_cast<std::ptrdiff_t>(e0 * part.ngll),
          part.radius.begin() + static_cast<std::ptrdiff_t>(e1 * part.ngll)};
}

std::expected<planetmodel::RadialRule, IoError> RuleOf(
    const planetmodel::ModelFile& file, const planetmodel::FileVariable& var) {
  const auto text = var.Text("radial_interpolation");
  if (!text) {
    return std::unexpected(IoError{std::format(
        "{}: {} states no radial_interpolation", file.Source(), var.name)});
  }
  return planetmodel::RadialRuleNamed(*text, file.Source(), var.name);
}

/// The file's own radial reference of a modulus on one layer.
std::expected<std::function<double(double)>, IoError> FileRadialReference(
    const planetmodel::ModelFile& file, int layer, const std::string& name,
    double tolerance) {
  const auto& part = file.PartOf(layer);
  const auto& var = part.variables.at(name);
  auto radialLayers = std::vector<int>{};
  if (const auto listed = var.Numbers("radial_layers")) {
    for (const auto x : *listed) radialLayers.push_back(static_cast<int>(x));
  }
  if (var.IsLateral() && !std::ranges::contains(radialLayers, layer)) {
    return std::unexpected(IoError{std::format(
        "{}: {} on layer {} is not a perturbation of a radial reference; "
        "split it with ModulusReference::LateralMean",
        file.Source(), name, layer)});
  }
  const auto rule = RuleOf(file, var);
  if (!rule) return std::unexpected(rule.error());
  const auto [e0, e1] = part.Elements(layer);
  const auto stride =
      var.IsLateral() ? part.grid->NTheta() * part.grid->NPhi() : std::size_t{1};
  auto values = std::vector<double>{};
  values.reserve((e1 - e0) * part.ngll);
  for (auto node = e0 * part.ngll; node < e1 * part.ngll; ++node) {
    const auto v = var.values[node * stride];
    if (!(v > 0.0) || !std::isfinite(v)) {
      return std::unexpected(IoError{std::format(
          "{}: the radial reference of {} on layer {} is {} at r = {}, not "
          "a positive modulus",
          file.Source(), name, layer, v, part.radius[node])});
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

/// The lateral mean per radius under the file grid's weights at each of
/// the layer's radial nodes, read between them by the field's rule.
std::expected<std::function<double(double)>, IoError> LateralMeanReference(
    const planetmodel::ModelFile& file, int layer, const std::string& name,
    const planetmodel::ScalarField& modulus, double tolerance) {
  const auto& part = file.PartOf(layer);
  if (!part.grid || !part.grid->Weights()) {
    return std::unexpected(IoError{std::format(
        "{}: part_{}'s grid carries no quadrature weights, so the lateral "
        "mean cannot be taken; use ModulusReference::FileRadial",
        file.Source(), part.index)});
  }
  const auto& grid = *part.grid;
  const auto& weights = *grid.Weights();
  auto weightSum = 0.0;
  for (const auto w : weights) weightSum += w;
  const auto rule = RuleOf(file, part.variables.at(name));
  if (!rule) return std::unexpected(rule.error());
  const auto radii = LayerRadii(part, layer);
  auto values = std::vector<double>{};
  values.reserve(radii.size());
  for (const auto r : radii) {
    auto mean = 0.0;
    for (auto i = std::size_t{0}; i < grid.NTheta(); ++i) {
      auto row = 0.0;
      for (auto j = std::size_t{0}; j < grid.NPhi(); ++j) {
        const auto v =
            modulus.Value(r, grid.Colatitudes()[i], grid.Longitudes()[j]);
        if (!(v > 0.0) || !std::isfinite(v)) {
          return std::unexpected(IoError{std::format(
              "{}: {} on layer {} is {} at r = {}, not a positive modulus",
              file.Source(), name, layer, v, r)});
        }
        row += v;
      }
      mean += weights[i] * row;
    }
    values.push_back(mean / (weightSum * static_cast<double>(grid.NPhi())));
  }
  auto interpolant = planetmodel::RadialInterpolant(
      planetmodel::RadialLayout(radii, part.ngll, tolerance),
      std::move(values), *rule);
  return [interpolant = std::move(interpolant)](double r) {
    return interpolant.Value(r);
  };
}

/// Every file layer holding the named modulus, read against its
/// reference.
std::expected<std::map<int, std::shared_ptr<const Layer>>, IoError> LayersOf(
    const planetmodel::ModelFile& file, const std::string& name,
    const ModelModuliOptions& options, double tolerance) {
  auto layers = std::map<int, std::shared_ptr<const Layer>>{};
  const auto& skeleton = file.GetSkeleton();
  for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers()); ++layer) {
    if (!file.HasField(layer, name)) continue;
    const auto field = file.FieldOn(layer, name);
    const auto radial = field.IsRadial();
    auto reference = std::function<double(double)>{};
    if (radial) {
      reference = [scalar = field.AsScalar()](double r) {
        return scalar.Value(r, 0.0, 0.0);
      };
    } else if (options.reference == ModulusReference::FileRadial) {
      auto made = FileRadialReference(file, layer, name, tolerance);
      if (!made) return std::unexpected(made.error());
      reference = std::move(*made);
    } else {
      auto made =
          LateralMeanReference(file, layer, name, field.AsScalar(), tolerance);
      if (!made) return std::unexpected(made.error());
      reference = std::move(*made);
    }
    layers.emplace(layer, std::make_shared<const Layer>(Layer{
                              field.Interval(), field.AsScalar(),
                              std::move(reference), radial}));
  }
  return layers;
}

}  // namespace

struct ModelModuli::Impl {
  double toFileRadius = 1.0;       ///< r_file = r_nd * toFileRadius
  std::vector<double> boundaries;  ///< the file's, in its units
  std::map<int, std::shared_ptr<const Layer>> shear, bulk;  ///< by file layer
  bool radial = true;
};

std::expected<ModelModuli, IoError> ModelModuli::Read(
    const std::filesystem::path& path, const EarthModel& model,
    const ModelModuliOptions& options) {
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

    auto shear = LayersOf(file, "mu", options, tolerance);
    if (!shear) return std::unexpected(shear.error());
    auto bulk = LayersOf(file, "kappa", options, tolerance);
    if (!bulk) return std::unexpected(bulk.error());
    if (shear->empty() && bulk->empty()) {
      return std::unexpected(
          IoError{file.Source() + ": holds neither mu nor kappa"});
    }
    for (const auto* layers : {&*shear, &*bulk}) {
      for (const auto& [layer, held] : *layers) {
        impl->radial = impl->radial && held->radial;
      }
    }

    // The model layers inside the file layers holding a modulus, each
    // wholly within one (a crossing means the layerings disagree).
    const auto b = dimensionless.Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      const auto lo = b[i] * impl->toFileRadius;
      const auto hi = b[i + 1] * impl->toFileRadius;
      const auto k = FileLayerOf(impl->boundaries, 0.5 * (lo + hi));
      for (const auto& [name, layers] :
           {std::pair{"mu", &*shear}, std::pair{"kappa", &*bulk}}) {
        const auto found = layers->find(k);
        if (found == layers->end()) continue;
        const auto& interval = found->second->interval;
        if (lo < interval.first - tolerance ||
            hi > interval.second + tolerance) {
          return std::unexpected(IoError{std::format(
              "{}: model layer {} on [{}, {}] (file units) crosses the ends "
              "of file layer {} on [{}, {}], which holds {}: the layerings "
              "disagree",
              file.Source(), i, lo, hi, k, interval.first, interval.second,
              name)});
        }
      }
    }
    impl->shear = std::move(*shear);
    impl->bulk = std::move(*bulk);
    return ModelModuli(std::move(impl));
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

namespace {

std::function<double(double, double, double)> PerturbationOf(
    std::shared_ptr<const ModelModuli::Impl> impl,
    const std::map<int, std::shared_ptr<const Layer>>& layers) {
  if (layers.empty()) return {};
  auto radial = true;
  for (const auto& [layer, held] : layers) radial = radial && held->radial;
  if (radial) return [](double, double, double) { return 0.0; };
  return [impl, &layers](double r, double theta, double phi) {
    const auto rf = r * impl->toFileRadius;
    const auto found = layers.find(FileLayerOf(impl->boundaries, rf));
    if (found == layers.end() || found->second->radial) return 0.0;
    const auto& layer = *found->second;
    const auto rc = std::clamp(rf, layer.interval.first, layer.interval.second);
    return layer.modulus.Value(rc, theta, phi) / layer.reference(rc) - 1.0;
  };
}

}  // namespace

std::function<double(double, double, double)> ModelModuli::Shear() const {
  return PerturbationOf(impl_, impl_->shear);
}

std::function<double(double, double, double)> ModelModuli::Bulk() const {
  return PerturbationOf(impl_, impl_->bulk);
}

bool ModelModuli::IsRadial() const { return impl_->radial; }

std::expected<void, IoError> WriteModuli(
    const std::filesystem::path& path, const EarthModel& model,
    const SurfaceGrid& grid,
    const std::function<double(double, double, double)>& shear,
    const std::function<double(double, double, double)>& bulk,
    const WriteModuliOptions& options) {
  if (options.ngll < 2) {
    return std::unexpected(IoError{"a file element needs two or more nodes"});
  }
  if (!(options.elementMetres > 0.0)) {
    return std::unexpected(IoError{"the element size is not positive"});
  }
  try {
    namespace pm = planetmodel;
    const auto& boundaries = model.Boundaries();
    auto file = pm::ModelFile(
        pm::Skeleton(std::vector<double>(boundaries.begin(), boundaries.end())));
    if (!options.name.empty()) file.SetName(options.name);
    file.SetMeta("code", "sl3dpp");
    const auto& skeleton = file.GetSkeleton();
    const auto fileGrid = pm::AngularGrid(
        {grid.CoLatitudes().begin(), grid.CoLatitudes().end()},
        {grid.Longitudes().begin(), grid.Longitudes().end()}, "gauss_legendre",
        static_cast<int>(grid.CoLatitudes().size()) - 1,
        std::vector<double>(grid.CoLatitudeWeights().begin(),
                            grid.CoLatitudeWeights().end()));
    const auto nlayers = static_cast<int>(skeleton.NLayers());
    file.AddPart(0, nlayers - 1,
                 pm::ElementEdges(skeleton, 0, nlayers - 1, options.elementMetres),
                 options.ngll, fileGrid);

    // The Voigt-averaged isotropic moduli of the model's elastic fields
    // at a radius of a layer (the Material's own average).
    auto moduli = [&model](std::size_t layer, double r) {
      const auto rho = model.EvaluateInLayer("rho", layer, r);
      const auto A = rho * std::pow(model.EvaluateInLayer("vph", layer, r), 2);
      const auto C = rho * std::pow(model.EvaluateInLayer("vpv", layer, r), 2);
      const auto L = rho * std::pow(model.EvaluateInLayer("vsv", layer, r), 2);
      const auto N = rho * std::pow(model.EvaluateInLayer("vsh", layer, r), 2);
      const auto F = model.EvaluateInLayer("eta", layer, r) * (A - 2.0 * L);
      return std::pair{(4.0 * A + C + 4.0 * F - 4.0 * N) / 9.0,
                       (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0};
    };
    auto rho = std::map<int, pm::LayerSource>{};
    auto kappa = std::map<int, pm::LayerSource>{};
    auto mu = std::map<int, pm::LayerSource>{};
    for (auto layer = 0; layer < nlayers; ++layer) {
      const auto index = static_cast<std::size_t>(layer);
      rho.emplace(layer, pm::LayerSource::Radial([&model, index](double r) {
        return model.EvaluateInLayer("rho", index, r);
      }));
      if (bulk) {
        kappa.emplace(layer, pm::LayerSource::Lateral(
                                 [&, index](double r, double theta, double phi) {
                                   return moduli(index, r).first *
                                          (1.0 + bulk(r, theta, phi));
                                 }));
      } else {
        kappa.emplace(layer, pm::LayerSource::Radial([&, index](double r) {
          return moduli(index, r).first;
        }));
      }
      if (model.IsFluid(index)) continue;
      if (shear) {
        mu.emplace(layer, pm::LayerSource::Lateral(
                              [&, index](double r, double theta, double phi) {
                                return moduli(index, r).second *
                                       (1.0 + shear(r, theta, phi));
                              }));
      } else {
        mu.emplace(layer, pm::LayerSource::Radial([&, index](double r) {
          return moduli(index, r).second;
        }));
      }
    }
    file.AddField("rho", rho);
    file.AddField("kappa", kappa, {.lateral = pm::LateralRule::Linear});
    if (!mu.empty()) file.AddField("mu", mu, {.lateral = pm::LateralRule::Linear});
    pm::WriteModelFile(path, file);
    return {};
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

}  // namespace sl3dpp::io
