#include "Common.hpp"

#include <format>
#include <planetmodel/json.hpp>
#include <utility>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp::app {

std::vector<std::string_view> RunMemberNames() {
  return {"model",  "model_format", "lmax",    "viscosity", "state",
          "ice_history", "times",   "scheme",  "rotation",  "moments",
          "rtol",   "coupled"};
}

void ReadRunMembers(const Json& top, const std::filesystem::path& directory,
                    EvolveInput& out) {
  out.model = PathOf(top, "model", directory);
  if (Has(top, "model_format")) {
    out.format = Choice<ModelFormat>(top.At("model_format"),
                                     {{"gia", ModelFormat::Gia},
                                      {"mineos", ModelFormat::Mineos},
                                      {"elastic", ModelFormat::Elastic}});
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  if (Has(top, "viscosity")) {
    const auto& viscosity = top.At("viscosity");
    Allow(viscosity, {"file", "reference"});
    out.viscosity = PathOf(viscosity, "file", directory);
    if (Has(viscosity, "reference")) {
      out.viscosity_file_radial =
          Choice<bool>(viscosity.At("reference"),
                       {{"geometric_mean", false}, {"file_radial", true}});
    }
  }
  const auto& state = top.At("state");
  Allow(state, {"file"});
  out.state = PathOf(state, "file", directory);
  const auto& history = top.At("ice_history");
  Allow(history, {"file"});
  out.ice_history = PathOf(history, "file", directory);
  const auto& times = top.At("times");
  Allow(times, {"start_kyr_bp", "end_kyr_bp", "nodes"});
  out.start_kyr_bp = Number(times, "start_kyr_bp");
  out.end_kyr_bp = Number(times, "end_kyr_bp", 0.0);
  out.nodes = Integer(times, "nodes");
  if (!(out.start_kyr_bp > out.end_kyr_bp)) {
    Fail(times, "runs forward in time: start_kyr_bp exceeds end_kyr_bp");
  }
  if (out.nodes < 2) Fail(times.At("nodes"), "is fewer than two nodes");
  if (Has(top, "scheme")) {
    out.scheme = Choice<Scheme>(
        top.At("scheme"), {{"exponential_trapezoid", Scheme::ExponentialTrapezoid},
                           {"backward_euler", Scheme::BackwardEuler}});
  }
  if (Has(top, "rotation")) {
    out.rotation = Choice<RotationalFeedback>(
        top.At("rotation"), {{"none", RotationalFeedback::None},
                             {"axial", RotationalFeedback::Axial},
                             {"full", RotationalFeedback::Full}});
  }
  if (Has(top, "moments")) {
    out.hydrostatic_moments =
        Choice<bool>(top.At("moments"), {{"input", false},
                                         {"hydrostatic", true}});
  }
  out.rtol = Number(top, "rtol", 1e-9);
  if (Has(top, "coupled")) {
    const auto& coupled = top.At("coupled");
    Allow(coupled, {"rtol", "max_iterations"});
    out.coupled_rtol = Number(coupled, "rtol", 1e-9);
    out.max_coupled_iterations = Integer(coupled, "max_iterations", 60);
  }
}

BuiltEarth BuildEarth(const EvolveInput& input, const Json& top) {
  const auto deck = Deck::Read(input.model, FormatOf(input.format));
  if (!deck) Fail(top.At("model"), deck.error().message);
  auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(top.At("model"), model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = input.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(top.At("model"), mesh.error().message);
  auto material = Material::Create(*model, *mesh);
  if (!material) Fail(top.At("model"), material.error().message);
  return {std::move(*model), std::move(*material)};
}

SeaLevelParameters ParametersOf(const EvolveInput& input,
                                const Material& material) {
  auto parameters = SeaLevelParameters{};
  if (input.hydrostatic_moments) {
    parameters =
        WithHydrostaticMoments(parameters, FluidTidalNumber(material, 2));
  }
  return parameters;
}

Loaded LoadWorld(const EvolveInput& input, const Json& top,
                 const SeaLevelSolver& solver) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto length = solver.SolverScales().Length();
  auto state_history = io::FieldHistory::Read(input.state, grid);
  if (!state_history) {
    Fail(top.At("state").At("file"), state_history.error().message);
  }
  auto ice_history = io::FieldHistory::Read(input.ice_history, grid);
  if (!ice_history) {
    Fail(top.At("ice_history").At("file"), ice_history.error().message);
  }
  for (const auto name : {"ice_thickness", "topography"}) {
    if (!state_history->Has(name)) {
      Fail(top.At("state").At("file"), std::format("holds no field {}", name));
    }
  }
  if (!ice_history->Has("ice_thickness")) {
    Fail(top.At("ice_history").At("file"), "holds no field ice_thickness");
  }
  const auto start_si = -input.start_kyr_bp * KYR;
  auto ice0 = state_history->Values("ice_thickness", start_si);
  const auto topo0 = state_history->Values("topography", start_si);
  const auto parameters = solver.Parameters();
  const auto flotation = parameters.ice_density / parameters.water_density;
  auto ice_nd = std::vector<double>(fieldSize);
  auto sl_nd = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    auto sl = topo0[i] < 0.0 ? -topo0[i] : -topo0[i] + ice0[i];
    if (topo0[i] < 0.0 && ice0[i] > 0.0) sl += flotation * ice0[i];
    ice_nd[i] = ice0[i] / length;
    sl_nd[i] = sl / length;
  }
  auto background = solver.MakeState(std::move(ice_nd), std::move(sl_nd));
  return {std::move(*state_history), std::move(*ice_history), std::move(ice0),
          std::move(background)};
}

std::vector<double> MakeTimes(const EvolveInput& input, const Scales& scales) {
  auto times = std::vector<double>{};
  const auto span_s = (input.start_kyr_bp - input.end_kyr_bp) * KYR;
  for (auto n = 0; n < input.nodes; ++n) {
    times.push_back(span_s * n / (input.nodes - 1) / scales.Time());
  }
  return times;
}

std::function<std::vector<double>(double)> IceLoadOf(
    const EvolveInput& input, const SeaLevelSolver& solver,
    const Loaded& world) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -input.start_kyr_bp * KYR;
  const auto rho_ice = solver.IceDensity();
  return [&world, fieldSize, length, time, start_si, rho_ice](double t) {
    const auto thickness =
        world.ice.Values("ice_thickness", start_si + t * time);
    const auto& C = world.background.oceanFunction;
    auto load = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      load[i] =
          rho_ice * (1.0 - C[i]) * (thickness[i] - world.ice0[i]) / length;
    }
    return load;
  };
}

}  // namespace sl3dpp::app
