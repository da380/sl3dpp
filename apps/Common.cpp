#include "Common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <planetmodel/json.hpp>
#include <utility>

#include "Ranks.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp::app {

std::vector<std::string_view> SeaLevelRunMemberNames() {
  return {"model", "lmax",     "times",   "forcing",
          "ocean", "rotation", "moments", "solver"};
}

void AllowTopLevel(const Json& top, std::span<const std::string_view> allowed) {
  // the members a block holds, by the block they belong in
  constexpr auto kInBlocks = std::array{
      std::pair{std::string_view{"model_format"}, "`model.format`"},
      std::pair{std::string_view{"viscosity"}, "`model.viscosity`"},
      std::pair{std::string_view{"state"}, "`forcing.state`"},
      std::pair{std::string_view{"ice_history"}, "`forcing.ice`"},
      std::pair{std::string_view{"change"}, "`forcing.ice`"},
      std::pair{std::string_view{"scheme"}, "`solver.scheme`"},
      std::pair{std::string_view{"rtol"}, "`solver.rtol`"},
      std::pair{std::string_view{"coupled"},
                "`solver`, as `step_rtol`, `step_max_iterations`, "
                "`transform_degrees` and `cache_mb`"},
      std::pair{std::string_view{"shorelines"}, "`ocean.shorelines`"},
      std::pair{std::string_view{"calibrate_initial_sea_level"},
                "`ocean.calibrate_initial_sea_level`"}};
  for (const auto& [key, value] : top.AsObject()) {
    if (std::ranges::find(allowed, key) != allowed.end()) continue;
    for (const auto& [name, block] : kInBlocks) {
      if (key == name) {
        Fail(value,
             std::format("is not a top-level member: it belongs in {}", block));
      }
    }
  }
  Allow(top, allowed);
}

SeaLevelRun ReadSeaLevelRun(const Json& top,
                            const std::filesystem::path& directory) {
  auto out = SeaLevelRun{};
  const auto timed = Has(top, "times");
  const auto& model = top.At("model");
  if (!timed && model.IsObject() && Has(model, "viscosity")) {
    Fail(model.At("viscosity"),
         "is a viscosity, which needs a `times` block (a run through time); "
         "the elastic fingerprint has no use for it");
  }
  out.model = ReadModelBlock(model, directory);
  if (timed && out.model.viscosity.empty() &&
      out.model.format != ModelFormat::Gia) {
    Fail(model.At("format"),
         "names a deck format with no viscosity column, which a run through "
         "time (a `times` block) needs: give a \"maxwell\" deck with a "
         "viscosity, or a `model.viscosity` file");
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  if (timed) {
    out.times = ReadTimesBlock(top.At("times"), TimesForm::BeforePresent);
  }
  out.forcing = ReadSeaLevelForcing(top.At("forcing"), directory, timed);
  if (Has(top, "ocean")) {
    const auto& ocean = top.At("ocean");
    if (!timed) {
      Allow(ocean, {"shorelines", "calibrate_initial_sea_level"});
      for (const auto& [name, value] : ocean.AsObject()) {
        Fail(value,
             "needs a `times` block (a run through time): the elastic "
             "fingerprint solves the linear equation on the state's own "
             "shorelines, with no initial sea level to calibrate");
      }
    }
    out.ocean = ReadOceanBlock(ocean);
  }
  if (Has(top, "rotation")) out.rotation = RotationOf(top.At("rotation"));
  if (Has(top, "moments")) {
    out.hydrostatic_moments = HydrostaticMomentsOf(top.At("moments"));
  }
  if (Has(top, "solver")) {
    const auto& solver = top.At("solver");
    if (!timed) {
      for (const auto* name : {"scheme", "step"}) {
        if (Has(solver, name)) {
          Fail(solver.At(name),
               "sets the time step, which needs a `times` block (a run "
               "through time)");
        }
      }
      for (const auto* name : {"step_rtol", "step_max_iterations",
                               "transform_degrees", "cache_mb"}) {
        if (Has(solver, name)) {
          Fail(solver.At(name),
               "sets the 3-D step, which needs a `times` block (a run "
               "through time): the elastic fingerprint is one solve on a "
               "radial Earth");
        }
      }
    }
    out.solver = ReadSolverBlock(solver, out.lmax, SolverOf::SeaLevel);
  }
  return out;
}

void RefuseThreeDSolver(const Json& top, std::string_view what) {
  if (!Has(top, "solver")) return;
  const auto& solver = top.At("solver");
  for (const auto* name : {"step", "step_rtol", "step_max_iterations",
                           "transform_degrees", "cache_mb"}) {
    if (Has(solver, name)) {
      Fail(solver.At(name),
           std::format("sets the 3-D step, which {} does not have", what));
    }
  }
}

std::string BuiltEarth::Dimension() const {
  if (Lateral()) {
    return std::format(
        "3-D ({})",
        LateralViscosity() && LateralModuli()
            ? "lateral viscosity, lateral moduli"
            : (LateralViscosity() ? "lateral viscosity" : "lateral moduli"));
  }
  if (viscosity && moduli) {
    return "1-D (the viscosity and moduli files are radial)";
  }
  if (viscosity) return "1-D (the viscosity file is radial)";
  if (moduli) return "1-D (the moduli file is radial)";
  return "1-D";
}

BuiltEarth BuildEarth(const SeaLevelRun& run, const Json& top) {
  const auto& modelMember = top.At("model");
  const auto& deckMember = modelMember.At("deck");
  const auto deck = Deck::Read(run.model.deck, FormatOf(run.model.format));
  if (!deck) Fail(deckMember, deck.error().message);
  auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(deckMember, model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = run.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(deckMember, mesh.error().message);
  auto material = Material::Create(*model, *mesh);
  if (!material) Fail(deckMember, material.error().message);
  auto out = BuiltEarth{std::move(*model), std::move(*material), {}, {}};
  if (!run.model.viscosity.empty()) {
    auto read = io::ModelViscosity::Read(
        run.model.viscosity, out.model,
        {.reference = run.model.viscosity_file_radial
                          ? io::ViscosityReference::FileRadial
                          : io::ViscosityReference::GeometricMean});
    if (!read) {
      Fail(modelMember.At("viscosity").At("file"), read.error().message);
    }
    out.viscosity.emplace(std::move(*read));
  }
  if (!run.model.moduli.empty()) {
    auto read = io::ModelModuli::Read(
        run.model.moduli, out.model,
        {.reference = run.model.moduli_file_radial
                          ? io::ModulusReference::FileRadial
                          : io::ModulusReference::LateralMean});
    if (!read) Fail(modelMember.At("moduli").At("file"), read.error().message);
    out.moduli.emplace(std::move(*read));
  }
  return out;
}

Rheology RheologyOf(const BuiltEarth& earth, const Json& top) {
  const auto& modelMember = top.At("model");
  auto rheology =
      earth.viscosity
          ? Rheology::Maxwell(earth.material, earth.viscosity->Viscosity())
          : Rheology::FromModel(earth.material, earth.model);
  if (!rheology) {
    if (earth.viscosity) {
      Fail(modelMember.At("viscosity").At("file"), rheology.error().message);
    }
    Fail(modelMember.At("deck"),
         std::format("gives no viscosity, which a run through time (a "
                     "`times` block) needs: {}",
                     rheology.error().message));
  }
  return std::move(*rheology);
}

Viscoelastic3D::Options OperatorOptionsOf(const SeaLevelRun& run,
                                          const BuiltEarth& earth,
                                          const Rheology& rheology) {
  const auto& solver = run.solver;
  auto lateral = Viscoelastic3D::Options::LateralModuli{};
  if (earth.LateralModuli()) {
    lateral = {.shear = earth.moduli->Shear(), .bulk = earth.moduli->Bulk()};
  }
  // On several ranks under lateral moduli, the rank cuts balance the
  // transform work of the elements the coupling carries: the viscous ones
  // and those the moduli perturb.
  auto active = std::vector<char>{};
  if (earth.LateralModuli() && TheRanks().Distributed()) {
    auto probe = Viscoelastic3D::Options{};
    probe.transform_degrees = solver.transform_degrees;
    probe.cache_bytes = solver.cache_mb << 20;
    probe.moduli = lateral;
    const SurfaceGrid grid(run.lmax, {.nmax = 2,
                                      .oversampling = probe.oversampling,
                                      .parallel = probe.parallel,
                                      .matrix_kernel = probe.matrix_kernel,
                                      .cache_bytes = probe.cache_bytes});
    active = Viscoelastic3D::ActiveElementsFor(rheology, run.lmax, grid, probe);
  }
  auto options = OperatorOptions(run.lmax, solver.transform_degrees,
                                 solver.cache_mb, rheology, active);
  options.solver = solver.step;
  options.moduli = lateral;
  return options;
}

std::function<double(double, double, double)> DeltaOf(const BuiltEarth& earth) {
  if (earth.viscosity) return earth.viscosity->Delta();
  return [](double, double, double) { return 0.0; };
}

SeaLevelParameters ParametersOf(const SeaLevelRun& run,
                                const Material& material) {
  auto parameters = SeaLevelParameters{};
  if (run.hydrostatic_moments) {
    parameters =
        WithHydrostaticMoments(parameters, FluidTidalNumber(material, 2));
  }
  return parameters;
}

namespace {

/// The members naming the state file and the ice history.
const Json& StateFileMember(const Json& top) {
  return top.At("forcing").At("state").At("file");
}
const Json& IceFileMember(const Json& top) {
  return top.At("forcing").At("ice").At("file");
}

}  // namespace

Loaded LoadWorld(const SeaLevelRun& run, const Json& top,
                 const SeaLevelSolver& solver) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto length = solver.SolverScales().Length();
  auto state_history = io::FieldHistory::Read(run.forcing.state.file, grid);
  if (!state_history) {
    Fail(StateFileMember(top), state_history.error().message);
  }
  auto ice_history = io::FieldHistory::Read(run.forcing.ice.history, grid);
  if (!ice_history) Fail(IceFileMember(top), ice_history.error().message);
  for (const auto name : {"ice_thickness", "topography"}) {
    if (!state_history->Has(name)) {
      Fail(StateFileMember(top), std::format("holds no field {}", name));
    }
  }
  if (!ice_history->Has("ice_thickness")) {
    Fail(IceFileMember(top), "holds no field ice_thickness");
  }
  const auto start_si = -run.times->start_kyr_bp * KYR;
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

std::vector<double> FinalSeaLevelOf(const SeaLevelRun& run, const Json& top,
                                    const SeaLevelSolver& solver,
                                    const Loaded& world) {
  const auto end_kyr_bp = run.times->end_kyr_bp;
  const auto end_si = -end_kyr_bp * KYR;
  // The file's end values are held beyond its span, which would make a
  // file ending before the window's end a target silently: refused.
  if (const auto span = world.state.Span(); span.has_value()) {
    const auto slack = 1e-9 * KYR;
    if (end_si < span->first - slack || end_si > span->second + slack) {
      Fail(StateFileMember(top),
           std::format("spans {:g} to {:g} kyr BP, which leaves out the "
                       "window's end ({:g} kyr BP), the calibration's target",
                       -span->first / KYR, -span->second / KYR, end_kyr_bp));
    }
  }
  auto ice = std::vector<double>{};
  auto topography = std::vector<double>{};
  try {
    ice = world.state.Values("ice_thickness", end_si);
    topography = world.state.Values("topography", end_si);
  } catch (const std::runtime_error& error) {
    Fail(StateFileMember(top),
         std::format("gives no sea level at the window's end, which the "
                     "calibration needs: {}",
                     error.what()));
  }
  auto sea = SeaLevelFromTopography(topography, ice, solver.Parameters());
  const auto length = solver.SolverScales().Length();
  for (auto& v : sea) v /= length;
  return sea;
}

std::vector<double> MakeTimes(const SeaLevelRun& run, const Scales& scales) {
  const auto& window = *run.times;
  auto times = std::vector<double>{};
  const auto span_s = (window.start_kyr_bp - window.end_kyr_bp) * KYR;
  for (auto n = 0; n < window.nodes; ++n) {
    times.push_back(span_s * n / (window.nodes - 1) / scales.Time());
  }
  return times;
}

std::function<std::vector<double>(double)> IceLoadOf(
    const SeaLevelRun& run, const SeaLevelSolver& solver, const Loaded& world) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -run.times->start_kyr_bp * KYR;
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
    const SeaLevelRun& run, const SeaLevelSolver& solver, const Loaded& world) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -run.times->start_kyr_bp * KYR;
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

IceHistory HistoryOf(const SeaLevelRun& run, const SeaLevelSolver& solver,
                     const Loaded& world) {
  return run.ocean.shorelines == Shorelines::Frozen
             ? IceHistory::Frozen(IceLoadOf(run, solver, world))
             : IceHistory::Migrating(IceThicknessOf(run, solver, world));
}

namespace {

CalibrationOptions CalibrationOptionsOf(const SeaLevelRun& run) {
  auto options = CalibrationOptions{};
  options.method = run.ocean.calibration_method;
  options.iterations = run.ocean.calibration_iterations;
  options.rtol = run.ocean.calibration_rtol;
  options.adjoint.rtol = run.solver.rtol;
  options.adjoint3D.rtol = run.solver.rtol;
  options.adjoint3D.coupled_rtol = run.solver.step_rtol;
  options.adjoint3D.max_coupled_iterations = run.solver.step_max_iterations;
  return options;
}

template <class Calibration>
CalibrationSummary Adopt(const SeaLevelRun& run, const Json& top,
                         const SeaLevelSolver& solver, Loaded& world,
                         Calibration calibration) {
  for (const auto& snapshot : calibration.snapshots) {
    if (!snapshot.converged) {
      throw std::runtime_error(std::format(
          "the calibration's last run did not converge at t = {} (its sea "
          "level solve, or in 3-D its coupled fixed point); loosen "
          "solver.rtol or solver.step_rtol",
          snapshot.t));
    }
  }
  if (!calibration.converged) {
    Fail(top.At("ocean").At("calibrate_initial_sea_level"),
         std::format("did not converge in {} runs (relative update {:.3g}, "
                     "tolerance {:.3g}); raise its iterations",
                     calibration.updates.size(), calibration.updates.back(),
                     run.ocean.calibration_rtol));
  }
  world.background = std::move(calibration.state);
  const auto length = solver.SolverScales().Length();
  // one run under frozen shorelines, one per update when migrating
  return {calibration.misfits.front() * length,
          calibration.misfits.back() * length, calibration.updates.size()};
}

}  // namespace

void StartFromIceHistory(const SeaLevelRun& run, const SeaLevelSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess) {
  const auto length = solver.SolverScales().Length();
  world.ice0 =
      world.ice.Values("ice_thickness", -run.times->start_kyr_bp * KYR);
  auto ice = world.ice0;
  for (auto& v : ice) v /= length;
  world.background = solver.MakeState(
      std::move(ice),
      std::vector<double>(seaLevelGuess.begin(), seaLevelGuess.end()));
}

namespace {

/// The calibration's start, pyslfp's construction: the ice history's ice
/// at the window's start under the target sea level, as the guess.
std::vector<double> PrepareCalibration(const SeaLevelRun& run, const Json& top,
                                       const SeaLevelSolver& solver,
                                       Loaded& world) {
  auto target = FinalSeaLevelOf(run, top, solver, world);
  StartFromIceHistory(run, solver, world, target);
  return target;
}

}  // namespace

CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(run, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(
      run, top, solver, world,
      CalibrateInitialSeaLevel(evolution, times, HistoryOf(run, solver, world),
                               target, CalibrationOptionsOf(run)));
}

CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution3D& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(run, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(
      run, top, solver, world,
      CalibrateInitialSeaLevel(evolution, times, HistoryOf(run, solver, world),
                               target, CalibrationOptionsOf(run)));
}

}  // namespace sl3dpp::app
