/**
 * @file TangentCommand.cpp
 * @brief `sl3d tangent`: the tangent-linear run of the problem `solve`
 * solves, along the direction a config's `perturbation` block gives
 * (Perturbation.hpp). The forward run it linearises about is `solve`'s run
 * of the same blocks, recorded in memory or read from the trajectory store
 * as `kernel`'s is (Recorded.hpp): the static problem as a run of one
 * node, or the quasi-static problem through time, with an ocean or without
 * one, radial or 3-D as the `model` block makes it. The run is the
 * library's (TangentLinearSolver, TangentLinearSolver3D): by default the
 * derivative of the stepped run under every scheme, whose transpose is the
 * discrete adjoint, or with `tangent.method` "continuous" the linearised
 * equations' run, which differs from it along a direction in the viscosity
 * alone; about a calibrated run it is by default the total derivative,
 * the change of the calibrated initial sea level found by the linearised
 * calibration (ComputeCalibratedTangent). It writes `solve`'s surface file
 * (SolveFiles.hpp), each field its first-order change along the
 * direction, and the first-order change of each measurement the config
 * lists (`tangent.measurements`, or the `kernels` block's): what
 * `kernel`'s kernels of that measurement give when paired with the
 * direction.
 */
#include <format>
#include <fstream>
#include <optional>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Input.hpp"
#include "Outputs.hpp"
#include "Perturbation.hpp"
#include "Ranks.hpp"
#include "Recorded.hpp"
#include "SolveFiles.hpp"
#include "Trajectory.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Version.hpp"

namespace sl3dpp::app {

TangentBlock ReadTangentBlock(const Json& block, const Run& run,
                              const std::filesystem::path& directory) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `tangent` is an object of `output`, `method`, "
         "`measurements` and `calibration`");
  }
  Allow(block, {"output", "method", "measurements", "calibration"});
  if (!run.gravitating) {
    Fail(block,
         "asks for a tangent-linear run, which is not built for a run "
         "without gravity (`gravity`: \"none\")");
  }
  auto out = TangentBlock{};
  if (Has(block, "method")) {
    out.method = Choice<AdjointMethod>(
        block.At("method"), {{"continuous", AdjointMethod::Continuous},
                             {"discrete", AdjointMethod::Discrete}});
  }
  if (Has(block, "measurements")) {
    out.measurements = ReadMeasurements(block.At("measurements"), run);
  }
  if (!Has(block, "output")) {
    Fail(block,
         "has no `output`, the block of the files the tangent-linear run "
         "writes: `surface`, the surface file (solve's, its fields their "
         "first-order changes), and optionally `fields` and `smoothing`");
  }
  const auto& output = block.At("output");
  if (output.IsObject()) {
    for (const auto& [name, what] :
         {std::pair{"exchange", "exchange tables"},
          std::pair{"exchange_nodes", "exchange tables"},
          std::pair{"volume", "response inside the body"}}) {
      if (Has(output, name)) {
        Fail(output.At(name),
             std::format("is solve's: the {} of a tangent-linear run are not "
                         "built; drop the member",
                         what));
      }
    }
  }
  out.output = ReadOutputBlock(output, directory);
  if (Has(block, "calibration")) {
    out.calibration = ReadCalibratedDerivative(
        block.At("calibration"), run, "run",
        "the partial run (`derivative`: \"partial\") does not take",
        "the change of the calibrated initial sea level");
  }
  return out;
}

void CheckOtherBlocks(const Json& top, const Run& run,
                      const std::filesystem::path& directory,
                      std::string_view command) {
  if (command != "kernel" && Has(top, "kernels")) {
    const auto kernels = ReadKernelsBlock(top.At("kernels"), run, directory);
    CheckSecondOrderDirection(top, kernels);
  }
  if (command == "tangent") return;
  if (Has(top, "perturbation")) {
    static_cast<void>(
        ReadPerturbationBlock(top.At("perturbation"), run, directory));
  }
  if (Has(top, "tangent")) {
    static_cast<void>(ReadTangentBlock(top.At("tangent"), run, directory));
  }
}

TangentInput ReadTangentInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto allowed = RunMemberNames();
  for (const auto name : {"perturbation", "tangent", "kernels", "output",
                          "checkpoint", "restart", "trajectory"}) {
    allowed.push_back(name);
  }
  AllowTopLevel(top, allowed);
  const auto directory = config.parent_path();
  auto out = TangentInput{};
  out.run = ReadRun(top, directory);
  if (!out.run.gravitating) {
    Fail(top.At("gravity"),
         "is \"none\", and the tangent-linear run of a run without gravity "
         "is not built");
  }
  out.trajectory = ReadTrajectoryBlock(top, out.run, directory);
  if (!Has(top, "perturbation")) {
    Fail(top,
         "has no `perturbation` block, the direction in the run's inputs the "
         "tangent-linear run is along");
  }
  if (!Has(top, "tangent")) {
    Fail(top,
         "has no `tangent` block, which names the files the tangent-linear "
         "run writes (`tangent.output.surface`)");
  }
  // solve's members, checked so that one config serves every command, and
  // left unused
  if (Has(top, "output") || Has(top, "checkpoint") || Has(top, "restart")) {
    static_cast<void>(ReadSolveMembers(top, out.run, directory, false));
  }
  out.perturbation =
      ReadPerturbationBlock(top.At("perturbation"), out.run, directory);
  out.tangent = ReadTangentBlock(top.At("tangent"), out.run, directory);
  // the measurements: the block's own list, or the `kernels` block's
  // where it has none; a `kernels` block is held to the run either way
  auto kernels = std::optional<KernelsBlock>{};
  if (Has(top, "kernels")) {
    kernels = ReadKernelsBlock(top.At("kernels"), out.run, directory);
  }
  if (out.tangent.measurements) {
    out.measurements = *out.tangent.measurements;
    out.measuredBy = "tangent";
  } else if (kernels) {
    out.measurements = std::move(kernels->measurements);
    out.measuredBy = "kernels";
  }
  return out;
}

namespace {

/// The library's pieces of a dimension: the direction's form, the solver
/// of its tangent-linear run and that run's snapshots.
struct Radial {
  using Recorded = Recorded1D;
  using Perturbation = TangentPerturbation;
  using Solver = TangentLinearSolver;
  using Snapshot = Evolution::Snapshot;
  using Trajectory = ForwardTrajectory;
  static constexpr auto lateral = false;
};
struct Lateral {
  using Recorded = Recorded3D;
  using Perturbation = TangentPerturbation3D;
  using Solver = TangentLinearSolver3D;
  using Snapshot = Evolution3D::Snapshot;
  using Trajectory = ForwardTrajectory3D;
  static constexpr auto lateral = true;
};

/// The tangent-linear solvers' options of a run: the block's method, the
/// sea level solves' tolerance and, in 3-D, the step's closure, as the
/// run's own.
TangentLinearSolver::Options OptionsOf(const TangentInput& input, Radial) {
  auto options = TangentLinearSolver::Options{};
  options.method = input.tangent.method;
  options.rtol = input.run.solver.rtol;
  return options;
}
TangentLinearSolver3D::Options OptionsOf(const TangentInput& input, Lateral) {
  auto options = TangentLinearSolver3D::Options{};
  options.method = input.tangent.method;
  options.rtol = input.run.solver.rtol;
  options.max_coupled_iterations = input.run.solver.step_max_iterations;
  options.coupled_rtol = input.run.solver.step_rtol;
  return options;
}

/// A tangent-linear run as far as the files and the summary need it: its
/// snapshots at the library's nodes, the change of the initial sea level
/// it started from (solver units; empty where there is none), where that
/// came from, and, of the linearised calibration, its runs and its last
/// relative update.
template <class Snapshot>
struct Ran {
  std::vector<Snapshot> snapshots;
  std::vector<double> initialSeaLevel;
  std::string origin;
  std::optional<int> calibrationRuns;
  double calibrationUpdate = 0.0;
};

/// Marks a surface file of solve's as a tangent-linear run's: the source
/// and title, what the fields are and by which method, the perturbation's
/// members
/// as described (`perturbation_<member>`), how its ice at the first node
/// was read (`ice_views`, as the kernels' ice files say it) and, of a
/// calibrated run, which derivative it is with the linearised
/// calibration's runs; every field's meaning is said to be its
/// first-order change. The state's ocean function, which a tangent-linear
/// run holds, is left out.
template <class Snapshot>
void MarkTangent(planetmodel::SurfaceFile& file, const TangentInput& input,
                 const Ran<Snapshot>& ran, std::string title) {
  const auto& run = input.run;
  file.source = std::format("sl3d tangent (sl3dpp {})", Version());
  file.title = std::move(title);
  file.variables.erase("ocean_function");
  file.variables.erase("ocean_function_final");
  const auto discrete = input.tangent.method == AdjointMethod::Discrete;
  file.meta.emplace(
      "tangent",
      std::string{"every field is the first-order change, along the "
                  "perturbation, of the field of its name in solve's file of "
                  "the run: "} +
          (discrete ? "the derivative of the stepped run"
                    : "the linearised equations' run (the continuous "
                      "method), which is the derivative of the stepped run "
                      "but along a direction in the viscosity, where it "
                      "converges to it with the step"));
  file.meta.emplace("tangent_method",
                    std::string{MethodWord(input.tangent.method)});
  for (const auto& [name, text] :
       DescribePerturbation(run, input.perturbation)) {
    file.meta.emplace("perturbation_" + name, text);
  }
  if (!input.perturbation.forcing.ice_change.empty()) {
    const auto held = !run.times || run.ocean->at == WindowEnd::Start;
    file.meta.emplace("ice_views", std::string{held ? "state_held"
                                                    : "run_start_moves"});
  }
  if (run.Calibrating()) {
    file.meta.emplace("calibration_derivative",
                      std::string{input.tangent.calibration.total
                                      ? "total"
                                      : "partial"});
  }
  if (ran.calibrationRuns) {
    file.meta.emplace("calibration_tangent_runs",
                      std::vector<double>{
                          static_cast<double>(*ran.calibrationRuns)});
    file.meta.emplace("calibration_tangent_update",
                      std::vector<double>{ran.calibrationUpdate});
  }
  for (auto& [name, variable] : file.variables) {
    const auto meaning = variable.Text("long_name").value_or(name);
    variable.attributes.insert_or_assign(
        "long_name",
        std::string{"the first-order change along the perturbation of: "} +
            meaning);
  }
}

/// The table of the measurements' first-order changes beside a surface
/// file: `<stem>_measurements.txt` in its directory.
std::filesystem::path TableBeside(const std::filesystem::path& surface) {
  return surface.parent_path() /
         (surface.stem().string() + "_measurements.txt");
}

/// The first-order change of each measurement of the config (its
/// `tangent.measurements`, or its `kernels` block's) along the direction:
/// its duals (BuildMeasurement, about the forward run) paired with the tangent-linear run's fields at their
/// nodes, SI, written as a table beside the surface file under a
/// provenance header, a row `name type value_si change_si` each, and
/// listed for the summary.
template <class D>
std::string WriteChanges(const std::filesystem::path& config,
                         const TangentInput& input,
                         const typename D::Recorded& recorded,
                         std::span<const typename D::Snapshot> tangent) {
  const auto& run = input.run;
  const auto& solver = *recorded.solver;
  const auto path = TableBeside(input.tangent.output.surface);
  CreateParent(path);
  auto out = std::ofstream(path);
  out << std::format("# sl3d tangent (sl3dpp {})\n", Version())
      << std::format("# created {}\n", CreatedNow())
      << std::format("# config {}\n", config.string())
      << std::format("# method {}\n", MethodWord(input.tangent.method));
  if (run.Calibrating()) {
    out << std::format("# calibration {}\n", input.tangent.calibration.total
                                                 ? "total"
                                                 : "partial");
  }
  for (const auto& [name, text] :
       DescribePerturbation(run, input.perturbation)) {
    out << std::format("# perturbation {} {}\n", name, text);
  }
  out << std::format("# the measurements of `{}.measurements`: the forward "
                     "value and its first-order change along the\n",
                     input.measuredBy)
      << "# perturbation, SI (what the measurement's kernels give when "
         "paired with the perturbation)\n"
         "# name type value_si change_si\n";
  auto listed = std::string{};
  for (const auto& spec : input.measurements) {
    const auto built = BuildMeasurement<typename D::Snapshot>(
        spec, run, solver, recorded.snapshots, recorded.steps);
    auto change = 0.0;
    for (const auto& [node, measurement] : built.measurements) {
      change += PairedWith(solver, measurement, tangent[node]);
    }
    change *= built.factor;
    out << std::format("{} {} {:.17g} {:.17g}\n", spec.name, spec.word,
                       built.factor * built.value_nd, change);
    listed += std::format("{}{} {:.6g} {}", listed.empty() ? "" : ", ",
                          spec.name, change, built.unit);
  }
  return std::format("; the measurements' first-order changes in {}: {}",
                     path.string(), listed);
}

/// The tangent-linear run about a recorded run along the config's
/// direction, its files written on the first rank, and the summary.
template <class D>
std::string TangentOf(const std::filesystem::path& config,
                      const TangentInput& input, const Json& top,
                      const Run& forcing, const typename D::Recorded& recorded,
                      const typename D::Trajectory& trajectory) {
  const auto& run = input.run;
  const auto& earth = *recorded.earth;
  const auto& solver = *recorded.solver;
  const auto& grid = solver.Grid();
  const auto& steps = recorded.steps;
  const auto& times = steps.times;
  const auto& block = top.At("perturbation");
  const auto& choice = input.tangent.calibration;
  const auto length = solver.SolverScales().Length();

  auto direction = SurfaceDirectionOf<typename D::Perturbation>(
      recorded, input.perturbation, forcing, block, choice.total);
  AddEarthDirection(direction.perturbation, input.perturbation, recorded,
                    block);
  // about a migrating run, its own shorelines (each stage of a two-stage
  // step about its own)
  const auto oceans = run.Migrating()
                          ? OceanFunctions::Of(recorded.snapshots)
                          : OceanFunctions{};
  const auto options = OptionsOf(input, D{});
  auto ran = Ran<typename D::Snapshot>{};
  if (run.Calibrating() && choice.total) {
    // the total derivative: the calibrated initial sea level's change by
    // the linearised calibration, then the run along both
    auto iteration = CalibrationMultiplierOptions{};
    iteration.iterations = choice.iterations;
    iteration.rtol = choice.rtol;
    auto solved = ComputeCalibratedTangent(
        *recorded.evolution, times, direction.perturbation,
        direction.targetSeaLevel, &trajectory, oceans, iteration, options);
    if (!solved.converged) {
      const auto& tangent = top.At("tangent");
      Fail(Has(tangent, "calibration") ? tangent.At("calibration") : tangent,
           std::format("did not find the change of the calibrated initial "
                       "sea level in {} runs (relative update {:.3g}, "
                       "tolerance {:.3g}); raise "
                       "tangent.calibration.iterations",
                       solved.runs, solved.updates.back(), choice.rtol));
    }
    ran.snapshots = std::move(solved.snapshots);
    ran.initialSeaLevel = std::move(solved.initialSeaLevel);
    ran.calibrationRuns = solved.runs;
    ran.calibrationUpdate =
        solved.updates.empty() ? 0.0 : solved.updates.back();
    ran.origin = std::format(
        "the change of the calibrated initial sea level, by the linearised "
        "calibration in {} tangent-linear run(s)",
        solved.runs);
  } else {
    ran.snapshots =
        typename D::Solver(*recorded.evolution, options)
            .Run(times, direction.perturbation, &trajectory, oceans);
    ran.initialSeaLevel = direction.perturbation.initialSeaLevel;
    ran.origin =
        ran.initialSeaLevel.empty()
            ? std::string{"none: the initial sea level is held"}
            : std::string{run.Calibrating()
                              ? "the perturbation's change of the calibrated "
                                "initial sea level"
                              : "the perturbation's change of the state's sea "
                                "level"};
  }

  // the run's steps against the shortest relaxation time, as solve's file
  // states them (collective in 3-D: every rank asks)
  auto relaxation = StepsAgainstRelaxation{};
  if (run.times) {
    relaxation = StepsAgainst(run, solver, times,
                              recorded.op->ShortestRelaxation(), earth,
                              recorded.rheology->BranchCount());
  }
  const auto& output = input.tangent.output;
  auto changes = std::string{};
  if (TheRanks().First()) {
    auto file = planetmodel::SurfaceFile{};
    auto title = std::string{};
    if (run.times) {
      const auto& window = *run.times;
      auto history = std::vector<HistoryNode>{};
      for (auto n = std::size_t{0}; n < window.Size(); ++n) {
        const auto& snapshot = ran.snapshots[steps.library[n]];
        auto kept = HistoryNode{};
        kept.node = window.nodes[n];
        KeepClosure(kept, snapshot);
        KeepFields(kept, snapshot, solver);
        history.push_back(std::move(kept));
      }
      auto seaLevel = ran.initialSeaLevel;
      if (seaLevel.empty()) seaLevel.assign(grid.FieldSize(), 0.0);
      for (auto& v : seaLevel) v *= length;
      file = HistoryFile(run, config, grid, history, seaLevel, ran.origin,
                         *recorded.world, earth,
                         LoadsAtNodes(forcing, solver, direction.loads.get()),
                         nullptr, relaxation);
      title = "first-order changes of the viscoelastic GIA histories along a "
              "perturbation";
    } else {
      auto world = *recorded.fixed;
      if (run.ocean) {
        world.change = direction.iceChange;
        if (world.change.empty()) world.change.assign(grid.FieldSize(), 0.0);
      }
      auto closure = HistoryNode{};
      KeepClosure(closure, ran.snapshots.front());
      file = StaticFile(config, run, world, grid, "",
                        AnswerOf(solver, ran.snapshots.front()),
                        direction.loads ? direction.loads->At(0.0)
                                        : SurfaceLoads::Node{});
      file.meta.emplace("lateral", std::vector<long long>{D::lateral ? 1 : 0});
      if (D::lateral) {
        file.meta.emplace("moduli_reference",
                          std::string{run.model.moduli_file_radial
                                          ? "file_radial"
                                          : "lateral_mean"});
      }
      file.meta.emplace("step_solver",
                        std::string{StepSolverName(run, D::lateral)});
      file.meta.emplace("step_iterations",
                        std::vector<long long>{closure.stepIterations});
      file.meta.emplace("solve_iterations",
                        std::vector<long long>{closure.solveIterations});
      file.meta.emplace("step_residual",
                        std::vector<double>{closure.stepResidual});
      title = D::lateral
                  ? "first-order changes of the static sea level problem on a "
                    "laterally heterogeneous elastic Earth along a "
                    "perturbation"
                  : "first-order changes of the static sea level problem on "
                    "the elastic Earth along a perturbation";
    }
    AddAskedFields(file, output, grid);
    MarkTangent(file, input, ran, std::move(title));
    CreateParent(output.surface);
    planetmodel::WriteSurfaceFile(output.surface, file);
    if (!input.measurements.empty()) {
      changes = WriteChanges<D>(config, input, recorded, ran.snapshots);
    }
  }

  auto members = std::string{};
  for (const auto& [name, text] :
       DescribePerturbation(run, input.perturbation)) {
    members += (members.empty() ? "" : ", ") + name;
  }
  auto derivative = std::string{};
  if (run.Calibrating() && !choice.total) {
    derivative = "; partial derivative, the calibrated initial sea level held";
  } else if (run.Calibrating()) {
    derivative =
        "; total derivative, the initial sea level calibrated again" +
        (run.Migrating()
             ? std::format(", its change in {} tangent-linear run(s)",
                           ran.calibrationRuns.value_or(0))
             : std::string{});
  }
  return std::format(
      "{}; wrote {} (the {}tangent-linear run along the perturbation's {}{}){}",
      recorded.read, output.surface.string(),
      input.tangent.method == AdjointMethod::Continuous ? "continuous " : "",
      members, derivative, changes);
}

}  // namespace

std::string RunTangent(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto input = ReadTangentInput(config);
  const auto earth = BuildEarth(input.run, top);
  PlaceSeismicSources(input.run, earth, top);
  FindFileJumps(input.run);
  const auto& run = input.run;
  // the perturbation's forcing, readied against the Earth as the run's
  const auto forcing = PerturbedForcingOf(run, input.perturbation, earth,
                                          top.At("perturbation"));
  // its model files held to the Earth before the forward run: a radial
  // run (the static problem's on a radial Earth too) takes radial ones
  CheckEarthDirection(input.perturbation, earth,
                      run.problem == Problem::Static ? earth.LateralModuli()
                                                     : earth.Lateral(),
                      top.At("perturbation"));
  const auto meaning = Meaning(run, earth);
  const auto& ranks = TheRanks();
  const auto tangent1D = [&](const Recorded1D& recorded) {
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(recorded.snapshots);
    return TangentOf<Radial>(config, input, top, forcing, recorded,
                             trajectory);
  };
  const auto tangent3D = [&](const Recorded3D& recorded) {
    return TangentOf<Lateral>(config, input, top, forcing, recorded,
                              *recorded.trajectory);
  };
  // the grids' members of the solver pass on a radial model where the
  // config's kernels are lateral, as in solve: the sweep's, unused here
  const auto transforms = LateralKernelsAsked(top);
  if (run.problem == Problem::Static) {
    if (earth.LateralModuli()) {
      return WithRecordedStatic3D(run, input.trajectory, top, earth, meaning,
                                  tangent3D);
    }
    RefuseThreeDSolver(top, "the static problem on a radial Earth",
                       transforms);
    // one process's run: on several ranks, the first rank's alone
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return WithRecordedStatic1D(run, input.trajectory, top, earth, meaning,
                                tangent1D);
  }
  if (!earth.Lateral()) {
    RefuseThreeDSolver(top, "a 1-D run (a radial model)", transforms);
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return WithRecorded1D(run, input.trajectory, top, earth, meaning,
                          tangent1D);
  }
  return WithRecorded3D(run, input.trajectory, top, earth, meaning, tangent3D);
}

}  // namespace sl3dpp::app
