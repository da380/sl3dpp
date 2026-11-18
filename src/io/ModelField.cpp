#include "sl3dpp/io/ModelField.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <planetmodel/modelfile.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

/// One file layer holding the field: the field, its layer's radii in the
/// file's units and the factor that takes its stored numbers to the values
/// returned.
struct Layer {
  std::pair<double, double> interval;
  planetmodel::ScalarField field;
  double factor = 1.0;
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

double ValueOn(const Layer& layer, double rf, double theta, double phi) {
  const auto rc = std::clamp(rf, layer.interval.first, layer.interval.second);
  return layer.factor * layer.field.Value(rc, theta, phi);
}

}  // namespace

struct ModelField::Impl {
  double toFileRadius = 1.0;       ///< r_file = r_nd * toFileRadius
  std::vector<double> boundaries;  ///< the file's, in its units
  std::map<int, std::shared_ptr<const Layer>> layers;  ///< by file layer
  std::map<std::size_t, std::shared_ptr<const Layer>> byModelLayer;
  bool radial = true;
};

std::expected<ModelField, IoError> ModelField::Read(
    const std::filesystem::path& path, const EarthModel& model,
    const std::string& name, std::optional<Dimensions> dimensions) {
  auto read = ReadIfHeld(path, model, name, dimensions);
  if (!read) return std::unexpected(read.error());
  if (!*read) {
    return std::unexpected(
        IoError{std::format("{}: holds no {} field", path.string(), name)});
  }
  return std::move(**read);
}

std::expected<std::vector<std::string>, IoError> ModelField::FieldNames(
    const std::filesystem::path& path) {
  try {
    const auto file = planetmodel::ReadModelFile(path);
    auto out = std::set<std::string>{};
    const auto layers = static_cast<int>(file.GetSkeleton().NLayers());
    for (auto layer = 0; layer < layers; ++layer) {
      for (auto& name : file.Names(layer)) out.insert(std::move(name));
    }
    return std::vector<std::string>(out.begin(), out.end());
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

std::expected<std::optional<ModelField>, IoError> ModelField::ReadIfHeld(
    const std::filesystem::path& path, const EarthModel& model,
    const std::string& name, std::optional<Dimensions> dimensions) {
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

    for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers());
         ++layer) {
      if (!file.HasField(layer, name)) continue;
      const auto field = file.FieldOn(layer, name);
      auto factor = 1.0;
      if (dimensions) {
        const auto declared = field.GetDimensions();
        factor = declared ? fileScales.Factor(*declared)
                          : fileScales.Factor(planetmodel::Dimensions{
                                .mass = dimensions->mass,
                                .length = dimensions->length,
                                .time = dimensions->time});
      }
      const auto radial = field.IsRadial();
      impl->radial = impl->radial && radial;
      impl->layers.emplace(
          layer, std::make_shared<const Layer>(Layer{
                     field.Interval(), field.AsScalar(), factor, radial}));
    }
    if (impl->layers.empty()) return std::optional<ModelField>{};

    // The model layers inside the file layers holding the field, each
    // wholly within one (a crossing means the layerings disagree).
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
            "of file layer {} on [{}, {}], which holds {}: the layerings "
            "disagree",
            file.Source(), i, lo, hi, k, interval.first, interval.second,
            name)});
      }
      impl->byModelLayer.emplace(i, found->second);
    }
    return std::optional<ModelField>{ModelField(std::move(impl))};
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

std::function<double(double, double, double)> ModelField::Value() const {
  return [impl = impl_](double r, double theta, double phi) {
    const auto rf = r * impl->toFileRadius;
    const auto found = impl->layers.find(FileLayerOf(impl->boundaries, rf));
    if (found == impl->layers.end()) return 0.0;
    return ValueOn(*found->second, rf, theta, phi);
  };
}

double ModelField::ValueIn(std::size_t layer, double r, double theta,
                           double phi) const {
  const auto found = impl_->byModelLayer.find(layer);
  if (found == impl_->byModelLayer.end()) return 0.0;
  return ValueOn(*found->second, r * impl_->toFileRadius, theta, phi);
}

bool ModelField::Holds(std::size_t layer) const {
  return impl_->byModelLayer.contains(layer);
}

bool ModelField::IsRadial() const { return impl_->radial; }

}  // namespace sl3dpp::io
