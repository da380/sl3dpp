#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

namespace {

/// Seconds in a kiloyear of Julian years, as pyslfp counts them.
constexpr auto KYR = 1.0e3 * 365.25 * 86400.0;

IceChange ReadChange(const Json& change) {
  Allow(change, {"type", "latitude", "longitude", "width_deg", "fraction",
                 "thickness_m"});
  auto out = IceChange{};
  out.kind = Choice<IceChange::Kind>(
      change.At("type"), {{"cap_fraction", IceChange::Kind::CapFraction},
                          {"cap_thickness", IceChange::Kind::CapThickness}});
  out.site = {Number(change, "latitude"), Number(change, "longitude")};
  const auto width = Number(change, "width_deg");
  if (!(width > 0.0)) Fail(change.At("width_deg"), "is not a positive angle");
  out.width = width * std::numbers::pi / 180.0;
  const auto fraction = out.kind == IceChange::Kind::CapFraction;
  if (Has(change, fraction ? "thickness_m" : "fraction")) {
    Fail(change, "gives the amount of the other kind of change");
  }
  out.amount = Number(change, fraction ? "fraction" : "thickness_m");
  return out;
}

/// The present-day state the solver takes, from a surface file's ice
/// thickness and topography: the sea level is minus the topography,
/// raised by the ice over land and by the floating fraction of the
/// shelves, with the solver's own densities (pyslfp's `IceNG`
/// construction).
std::pair<std::vector<double>, std::vector<double>> StateOf(
    const io::FieldHistory& history, double time,
    const SeaLevelParameters& parameters) {
  const auto ice = history.Values("ice_thickness", time);
  const auto topography = history.Values("topography", time);
  const auto flotation = parameters.ice_density / parameters.water_density;
  auto sea_level = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i) {
    sea_level[i] =
        topography[i] < 0.0 ? -topography[i] : -topography[i] + ice[i];
    if (topography[i] < 0.0 && ice[i] > 0.0) {
      sea_level[i] += flotation * ice[i];
    }
  }
  return {ice, std::move(sea_level)};
}

}  // namespace

FingerprintInput ReadFingerprintInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  Allow(top, {"model", "model_format", "lmax", "state", "change", "rotation",
              "output"});
  const auto directory = config.parent_path();
  auto out = FingerprintInput{};
  out.model = PathOf(top, "model", directory);
  if (Has(top, "model_format")) {
    out.format = Choice<ModelFormat>(top.At("model_format"),
                                     {{"gia", ModelFormat::Gia},
                                      {"mineos", ModelFormat::Mineos},
                                      {"elastic", ModelFormat::Elastic}});
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  const auto& state = top.At("state");
  Allow(state, {"file", "date_kyr_bp"});
  out.state = PathOf(state, "file", directory);
  out.date_kyr_bp = Number(state, "date_kyr_bp", 0.0);
  out.change = ReadChange(top.At("change"));
  if (Has(top, "rotation")) {
    out.rotation = Choice<RotationalFeedback>(
        top.At("rotation"), {{"none", RotationalFeedback::None},
                             {"axial", RotationalFeedback::Axial},
                             {"full", RotationalFeedback::Full}});
  }
  out.output = PathOf(top, "output", directory);
  return out;
}

std::string RunFingerprint(const std::filesystem::path& config) {
  namespace pm = planetmodel;
  const auto top = pm::ReadJsonFile(config);
  const auto input = ReadFingerprintInput(config);

  const auto deck = Deck::Read(input.model, FormatOf(input.format));
  if (!deck) Fail(top.At("model"), deck.error().message);
  const auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(top.At("model"), model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = input.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(top.At("model"), mesh.error().message);
  const auto material = Material::Create(*model, *mesh);
  if (!material) Fail(top.At("model"), material.error().message);
  const auto love = ComputeLoveNumbers(*material, input.lmax);

  const auto options = Fingerprint::Options{};
  Fingerprint fingerprint(input.lmax, love, options);
  const auto& grid = fingerprint.Grid();

  const auto history = io::FieldHistory::Read(input.state, grid);
  if (!history) Fail(top.At("state").At("file"), history.error().message);
  for (const auto name : {"ice_thickness", "topography"}) {
    if (!history->Has(name)) {
      Fail(top.At("state").At("file"),
           std::format("holds no field {}", name));
    }
  }
  const auto [ice, sea_level] =
      StateOf(*history, -input.date_kyr_bp * KYR, options.parameters);
  fingerprint.SetState(ice, sea_level);

  const auto colat0 = input.change.site.CoLatitude();
  const auto lon0 = input.change.site.Longitude();
  const auto mask = grid.FromFunction([&](double theta, double phi) {
    const auto c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
    const auto d = std::acos(std::clamp(c, -1.0, 1.0));
    return std::exp(-0.5 * (d / input.change.width) *
                    (d / input.change.width));
  });
  auto change = std::vector<double>(ice.size());
  const auto fraction = input.change.kind == IceChange::Kind::CapFraction;
  for (auto i = std::size_t{0}; i < ice.size(); ++i) {
    change[i] = input.change.amount * mask[i] * (fraction ? ice[i] : 1.0);
  }

  const auto result = fingerprint.SolveForIceChange(change, input.rotation);
  if (!result.converged) {
    throw std::runtime_error("the sea level equation did not converge");
  }

  // the fingerprint as a surface file: static SI fields on the solver grid
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", input.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d fingerprint (sl3dpp {})", Version());
  file.title = "elastic sea-level fingerprint";
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", input.model.string());
  file.meta.emplace("state", input.state.string());
  file.meta.emplace("date_kyr_bp", std::vector<double>{input.date_kyr_bp});
  file.meta.emplace("rotation",
                    std::string{input.rotation == RotationalFeedback::None
                                    ? "none"
                                    : input.rotation ==
                                              RotationalFeedback::Axial
                                          ? "axial"
                                          : "full"});
  file.meta.emplace("omega_x_rad_s",
                    std::vector<double>{result.angularVelocity[0]});
  file.meta.emplace("omega_y_rad_s",
                    std::vector<double>{result.angularVelocity[1]});
  file.meta.emplace("omega_z_rad_s",
                    std::vector<double>{result.angularVelocity[2]});
  file.meta.emplace("sle_iterations",
                    std::vector<long long>{result.iterations});
  const auto length = pm::units::LENGTH;
  const auto potential = pm::Dimensions{.length = 2, .time = -2};
  auto put = [&file](std::string name, const std::vector<double>& values,
                     std::optional<pm::Dimensions> dims,
                     std::string meaning) {
    file.variables.emplace(
        name, pm::SurfaceField(name, values,
                               {.dimensions = dims, .meaning = meaning}));
  };
  put("ice_change", change, length, "the ice-thickness change applied");
  put("sea_level_change", result.seaLevelChange, length,
      "sea-level change of the fingerprint");
  put("displacement", result.displacement, length,
      "radial surface displacement");
  put("potential", result.potential, potential,
      "gravitational potential perturbation at the surface");
  const auto ocean = fingerprint.OceanFunction();
  put("ocean_function", std::vector<double>(ocean.begin(), ocean.end()),
      std::nullopt, "the ocean function of the state");
  if (const auto parent = input.output.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(input.output, file);
  return std::format("wrote {} ({} SLE iterations)", input.output.string(),
                     result.iterations);
}

}  // namespace sl3dpp::app
