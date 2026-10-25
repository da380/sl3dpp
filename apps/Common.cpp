#include "Common.hpp"

#include <format>
#include <planetmodel/json.hpp>
#include <utility>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp::app {

std::vector<std::string_view> RunMemberNames() {
  return {"model",      "model_format",
          "lmax",       "viscosity",
          "state",      "ice_history",
          "times",      "scheme",
          "rotation",   "moments",
          "rtol",       "coupled",
          "shorelines", "calibrate_initial_sea_level"};
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
  if (Has(top, "shorelines")) {
    out.shorelines = Choice<Shorelines>(
        top.At("shorelines"),
        {{"frozen", Shorelines::Frozen}, {"migrating", Shorelines::Migrating}});
  }
  if (Has(top, "calibrate_initial_sea_level")) {
    const auto& calibration = top.At("calibrate_initial_sea_level");
    Allow(calibration, {"method", "iterations", "rtol"});
    out.calibrate = true;
    if (Has(calibration, "method")) {
      out.calibration_method = Choice<CalibrationMethod>(
          calibration.At("method"),
          {{"fixed_point", CalibrationMethod::FixedPoint},
           {"gradient", CalibrationMethod::Gradient}});
    }
    out.calibration_iterations = Integer(calibration, "iterations", 10);
    if (out.calibration_iterations < 1) {
      Fail(calibration.At("iterations"), "is fewer than one run");
    }
    out.calibration_rtol = Number(calibration, "rtol", 1e-4);
    if (!(out.calibration_rtol > 0.0)) {
      Fail(calibration.At("rtol"), "is not a positive tolerance");
    }
  }
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
  auto sl_nd = SeaLevelFromTopography(topo0, ice0, solver.Parameters());
  auto ice_nd = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    ice_nd[i] = ice0[i] / length;
    sl_nd[i] /= length;
  }
  auto background = solver.MakeState(std::move(ice_nd), std::move(sl_nd));
  return {std::move(*state_history), std::move(*ice_history), std::move(ice0),
          std::move(background)};
}

std::vector<double> FinalSeaLevelOf(const EvolveInput& input, const Json& top,
                                    const SeaLevelSolver& solver,
                                    const Loaded& world) {
  const auto end_si = -input.end_kyr_bp * KYR;
  // The file's end values are held beyond its span, which would make a
  // file ending before the window's end a target silently: refused.
  if (const auto span = world.state.Span(); span.has_value()) {
    const auto slack = 1e-9 * KYR;
    if (end_si < span->first - slack || end_si > span->second + slack) {
      Fail(top.At("state").At("file"),
           std::format("spans {:g} to {:g} kyr BP, which leaves out the "
                       "window's end ({:g} kyr BP), the calibration's target",
                       -span->first / KYR, -span->second / KYR,
                       input.end_kyr_bp));
    }
  }
  auto ice = std::vector<double>{};
  auto topography = std::vector<double>{};
  try {
    ice = world.state.Values("ice_thickness", end_si);
    topography = world.state.Values("topography", end_si);
  } catch (const std::runtime_error& error) {
    Fail(top.At("state").At("file"),
         std::format("gives no sea level at the window's end, which the "
                     "calibration needs: {}",
                     error.what()));
  }
  auto sea = SeaLevelFromTopography(topography, ice, solver.Parameters());
  const auto length = solver.SolverScales().Length();
  for (auto& v : sea) v /= length;
  return sea;
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

std::function<std::vector<double>(double)> IceThicknessOf(
    const EvolveInput& input, const SeaLevelSolver& solver,
    const Loaded& world) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -input.start_kyr_bp * KYR;
  // the background's ice plus the history's change from world.ice0, the
  // start's ice the frozen load is measured from (IceLoadOf), so a history
  // that disagrees with the state at the start is refused by the migrating
  // run rather than measured from another baseline
  return [&world, length, time, start_si](double t) {
    auto thickness = world.ice.Values("ice_thickness", start_si + t * time);
    const auto& background = world.background.iceThickness;
    for (auto i = std::size_t{0}; i < thickness.size(); ++i) {
      thickness[i] = background[i] + (thickness[i] - world.ice0[i]) / length;
    }
    return thickness;
  };
}

IceHistory HistoryOf(const EvolveInput& input, const SeaLevelSolver& solver,
                     const Loaded& world) {
  return input.shorelines == Shorelines::Frozen
             ? IceHistory::Frozen(IceLoadOf(input, solver, world))
             : IceHistory::Migrating(IceThicknessOf(input, solver, world));
}

namespace {

CalibrationOptions CalibrationOptionsOf(const EvolveInput& input) {
  auto options = CalibrationOptions{};
  options.method = input.calibration_method;
  options.iterations = input.calibration_iterations;
  options.rtol = input.calibration_rtol;
  options.adjoint.rtol = input.rtol;
  options.adjoint3D.rtol = input.rtol;
  options.adjoint3D.coupled_rtol = input.coupled_rtol;
  options.adjoint3D.max_coupled_iterations = input.max_coupled_iterations;
  return options;
}

template <class Calibration>
CalibrationSummary Adopt(const EvolveInput& input, const Json& top,
                         const SeaLevelSolver& solver, Loaded& world,
                         Calibration calibration) {
  for (const auto& snapshot : calibration.snapshots) {
    if (!snapshot.converged) {
      throw std::runtime_error(std::format(
          "the calibration's last run did not converge at t = {} (its sea "
          "level solve, or in 3D its coupled fixed point); loosen rtol or "
          "coupled.rtol",
          snapshot.t));
    }
  }
  if (!calibration.converged) {
    Fail(top.At("calibrate_initial_sea_level"),
         std::format("did not converge in {} runs (relative update {:.3g}, "
                     "tolerance {:.3g}); raise its iterations",
                     calibration.updates.size(), calibration.updates.back(),
                     input.calibration_rtol));
  }
  world.background = std::move(calibration.state);
  const auto length = solver.SolverScales().Length();
  // one run under frozen shorelines, one per update when migrating
  return {calibration.misfits.front() * length,
          calibration.misfits.back() * length, calibration.updates.size()};
}

}  // namespace

void StartFromIceHistory(const EvolveInput& input, const SeaLevelSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess) {
  const auto length = solver.SolverScales().Length();
  world.ice0 = world.ice.Values("ice_thickness", -input.start_kyr_bp * KYR);
  auto ice = world.ice0;
  for (auto& v : ice) v /= length;
  world.background = solver.MakeState(
      std::move(ice),
      std::vector<double>(seaLevelGuess.begin(), seaLevelGuess.end()));
}

namespace {

/// The calibration's start, pyslfp's construction: the ice history's ice
/// at the window's start under the target sea level, as the guess.
std::vector<double> PrepareCalibration(const EvolveInput& input,
                                       const Json& top,
                                       const SeaLevelSolver& solver,
                                       Loaded& world) {
  auto target = FinalSeaLevelOf(input, top, solver, world);
  StartFromIceHistory(input, solver, world, target);
  return target;
}

}  // namespace

CalibrationSummary CalibrateWorld(const EvolveInput& input, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(input, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(input, top, solver, world,
               CalibrateInitialSeaLevel(evolution, times,
                                        HistoryOf(input, solver, world), target,
                                        CalibrationOptionsOf(input)));
}

CalibrationSummary CalibrateWorld(const EvolveInput& input, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution3D& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(input, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(input, top, solver, world,
               CalibrateInitialSeaLevel(evolution, times,
                                        HistoryOf(input, solver, world), target,
                                        CalibrationOptionsOf(input)));
}

}  // namespace sl3dpp::app
