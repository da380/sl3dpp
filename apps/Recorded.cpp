/**
 * @file Recorded.cpp
 * @brief The recorded forward run of the derivative commands and the
 * measurements read off it (Recorded.hpp).
 */
#include "Recorded.hpp"

#include <complex>
#include <format>
#include <stdexcept>
#include <utility>

#include "Ranks.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"

namespace sl3dpp::app {

namespace {

using Type = MeasurementSpec::Type;

/// The surface pairing a^2 int dual field dOmega, nothing where either is
/// absent (a toroidal potential a run does not have, say).
double PairedOrNothing(const SurfaceGrid& grid, double a,
                       std::span<const double> dual,
                       std::span<const double> field) {
  if (dual.empty() || field.empty()) return 0.0;
  return PairedValue(grid, a, dual, field);
}

}  // namespace

template <class Snapshot>
Built BuildMeasurement(const MeasurementSpec& spec, const Run& run,
                       const GRDSolver& solver,
                       std::span<const Snapshot> snapshots,
                       const StepGrid& steps) {
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();
  const auto& scales = solver.SolverScales();
  const auto at = [&steps](std::size_t node) { return steps.library.at(node); };
  auto out = Built{};
  auto single = [&](FieldChannel channel, std::vector<double> dual,
                    auto field_of) {
    out.nodes = {spec.node};
    out.value_nd =
        PairedValue(grid, a, dual, field_of(snapshots[at(spec.node)]));
    out.measurements.emplace(at(spec.node),
                             FieldMeasurement(channel, std::move(dual)));
  };
  const auto sea = [](const Snapshot& s) -> std::span<const double> {
    return s.relativeSeaLevelChange;
  };
  const auto displacement = [](const Snapshot& s) -> std::span<const double> {
    return s.displacement;
  };
  const auto potential = [](const Snapshot& s) -> std::span<const double> {
    return s.potential;
  };
  const auto length = [&] {
    out.factor = scales.Length();
    out.objective = Dimensions{.length = 1};
    out.unit = "m";
  };
  const auto potentialUnit = [&] {
    out.factor = scales.Factor(PotentialDimensions);
    out.objective = PotentialDimensions;
    out.unit = "m^2 s^-2";
  };
  switch (spec.type) {
    case Type::RelativeSeaLevelCap:
    case Type::RelativeSeaLevelPoint:
      length();
      single(FieldChannel::SeaLevel,
             spec.type == Type::RelativeSeaLevelCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             sea);
      break;
    case Type::DisplacementCap:
    case Type::DisplacementPoint:
      length();
      single(FieldChannel::Displacement,
             spec.type == Type::DisplacementCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             displacement);
      break;
    case Type::DisplacementVectorCap:
    case Type::DisplacementVectorPoint: {
      // the radial displacement, and the potentials V and W of the
      // tangential one (W's dual has no degree 1: the displacement in the
      // frame of no net rotation of the surface)
      length();
      auto m = spec.type == Type::DisplacementVectorCap
                   ? CapDisplacementMeasurement(grid, a, spec.site, spec.width,
                                                spec.direction)
                   : PointDisplacementMeasurement(grid, a, spec.site,
                                                  spec.direction);
      const auto& s = snapshots[at(spec.node)];
      out.nodes = {spec.node};
      out.value_nd =
          PairedOrNothing(grid, a, m.displacementDual, s.displacement) +
          PairedOrNothing(grid, a, m.tangentialPotentialDual,
                          s.tangentialPotential) +
          PairedOrNothing(grid, a, m.toroidalPotentialDual,
                          s.toroidalPotential);
      out.measurements.emplace(at(spec.node), std::move(m));
      break;
    }
    case Type::PotentialCap:
    case Type::PotentialPoint:
      potentialUnit();
      single(FieldChannel::Potential,
             spec.type == Type::PotentialCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             potential);
      break;
    case Type::StokesCoefficient:
      potentialUnit();
      single(FieldChannel::Potential,
             CoefficientDual(grid, a, spec.degree, spec.order), potential);
      break;
    case Type::Rotation: {
      out.nodes = {spec.node};
      out.factor = 1.0 / scales.Time();
      out.objective = Dimensions{.time = -1};
      out.unit = "rad s^-1";
      const auto component = static_cast<std::size_t>(spec.component);
      out.value_nd = snapshots[at(spec.node)].angularVelocity[component];
      out.measurements.emplace(at(spec.node),
                               RotationMeasurement(spec.component));
      break;
    }
    case Type::DecayTime: {
      // the site series over the window, in SI metres against the run's
      // unit (kyr in a dated run), the fit's weights becoming node-keyed
      // duals: J in that unit, SI seconds through the factor
      const auto dual = CapMeanDual(grid, a, spec.site, spec.width);
      const auto metres = scales.Length();
      const auto& window = *run.times;
      const auto unit = window.Dated() ? KYR : window.unit.seconds;
      const auto even = window.Dated() && !window.listed;
      const auto step = (window.Start() - window.End()) /
                        (static_cast<int>(window.Size()) - 1);
      auto elapsed = std::vector<double>{};
      auto series = std::vector<double>{};
      auto library = std::vector<std::size_t>{};
      for (auto node = spec.first; node <= spec.last; ++node) {
        elapsed.push_back(even ? step * static_cast<double>(node - spec.first)
                               : (window.Axis(node) - window.Axis(spec.first)) /
                                     unit);
        series.push_back(metres *
                         PairedValue(grid, a, dual, sea(snapshots[at(node)])));
        out.nodes.push_back(node);
        library.push_back(at(node));
      }
      auto fit = DecayTimeFit{};
      out.measurements = DecayTimeMeasurements(
          library, elapsed, series, FieldChannel::SeaLevel, dual,
          {.time_uncertainty = spec.time_uncertainty,
           .value_uncertainty = spec.value_uncertainty_m},
          &fit);
      // the weights are per metre of series: scale the duals so the
      // pairing's J is in the run's unit
      for (auto& [node, measurement] : out.measurements) {
        for (auto& v : measurement.relativeSeaLevelDual) v *= metres;
      }
      out.value_nd = fit.decay_time;
      out.factor = unit;
      out.objective = Dimensions{.time = 1};
      out.unit = "s";
      out.fit = fit;
      break;
    }
  }
  return out;
}


template Built BuildMeasurement<Evolution::Snapshot>(
    const MeasurementSpec&, const Run&, const GRDSolver&,
    std::span<const Evolution::Snapshot>, const StepGrid&);
template Built BuildMeasurement<Evolution3D::Snapshot>(
    const MeasurementSpec&, const Run&, const GRDSolver&,
    std::span<const Evolution3D::Snapshot>, const StepGrid&);

template <class Snapshot>
double PairedWith(const GRDSolver& solver, const Measurement& measurement,
                  const Snapshot& snapshot) {
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();
  auto out = PairedOrNothing(grid, a, measurement.relativeSeaLevelDual,
                             snapshot.relativeSeaLevelChange) +
             PairedOrNothing(grid, a, measurement.displacementDual,
                             snapshot.displacement) +
             PairedOrNothing(grid, a, measurement.potentialDual,
                             snapshot.potential) +
             PairedOrNothing(grid, a, measurement.tangentialPotentialDual,
                             snapshot.tangentialPotential) +
             PairedOrNothing(grid, a, measurement.toroidalPotentialDual,
                             snapshot.toroidalPotential);
  for (auto k = std::size_t{0}; k < 3; ++k) {
    out += measurement.angularVelocityDual[k] * snapshot.angularVelocity[k];
  }
  return out;
}

template double PairedWith<Evolution::Snapshot>(const GRDSolver&,
                                                const Measurement&,
                                                const Evolution::Snapshot&);
template double PairedWith<Evolution3D::Snapshot>(const GRDSolver&,
                                                  const Measurement&,
                                                  const Evolution3D::Snapshot&);

std::string_view MethodWord(AdjointMethod method) {
  return method == AdjointMethod::Discrete ? "discrete" : "continuous";
}

namespace {

/// Where a recorded run's forward run comes from: the trajectory store of
/// `trajectory.directory` that holds this run complete, read back; that
/// store recorded by this run where the directory holds none (or an
/// incomplete one of this run); or, without a directory, the run in memory.
/// `run` is the run's description for the store.
struct Forward {
  std::optional<io::TrajectoryStore> store;  ///< none without a directory
  bool read = false;  ///< swept from a complete store, not run

  /// Surveys `trajectory.directory`, where given, for the run `run`
  /// describes (called only then), and opens its store for reading or
  /// for recording; the store's refusals named at the member.
  Forward(const Json& top, const TrajectoryBlock& block,
          const std::function<io::TrajectoryStore::Description()>& run) {
    if (block.directory.empty()) return;
    const auto described = run();
    if (SurveyTrajectory(top, block, described) ==
        io::TrajectoryStore::Standing::Complete) {
      store = OpenTrajectory(top, block, described);
      read = true;
    } else {
      store = RecordTrajectory(top, block, described);
    }
  }
  /// Whether this run records into the store.
  [[nodiscard]] bool Records() const { return store && !read; }

  /// The summary's account of the store, after the sweeps' (empty
  /// without one).
  [[nodiscard]] std::string Note() const {
    if (!store) return {};
    const auto at = std::format("the trajectory store {} ({})",
                                store->Directory().string(),
                                SizeOf(store->Bytes()));
    return read ? std::format("; the forward run read back from {}", at)
                : std::format("; the forward run recorded to {}", at);
  }
};

/// The static problem's grid: its one node.
StepGrid OneNode() { return {.times = {0.0}, .window = {0}, .library = {0}}; }

}  // namespace

std::string WithRecordedStatic1D(
    const Run& run, const TrajectoryBlock& block, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded1D&)>& then) {
  const auto rheology = ElasticRheologyOf(earth, top);
  const ViscoelasticOperator op(rheology, run.lmax);
  const GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                         ParametersOf(run, earth.material));
  auto world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  const auto times = std::vector<double>{0.0};
  const auto forward = Forward(top, block, [&] {
    return TrajectoryOf(run, 1, times, solver);
  });
  auto options = Evolution::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.record_stride = 1;
  if (forward.read) world.background = forward.store->Start();
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution evolution(op, solver, world.background, options);
  const auto snapshots = forward.read
                             ? forward.store->Snapshots1D()
                             : SolveStatic(run, solver, world, loads, evolution);
  if (!snapshots.front().converged) {
    throw std::runtime_error(
        "the sea level equation did not converge; loosen solver.rtol");
  }
  if (forward.Records()) forward.store->Close();
  auto recorded = Recorded1D{};
  recorded.run = &run;
  recorded.earth = &earth;
  recorded.solver = &solver;
  recorded.steps = OneNode();
  recorded.fixed = &world;
  recorded.loads = loads.get();
  recorded.read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                             InputsOf(loads.get()));
  recorded.rheology = &rheology;
  recorded.op = &op;
  recorded.evolution = &evolution;
  recorded.snapshots = snapshots;
  return then(recorded) + forward.Note();
}

std::string WithRecordedStatic3D(
    const Run& run, const TrajectoryBlock& block, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded3D&)>& then) {
  const auto elastic = ElasticRheologyOf(earth, top);
  const Viscoelastic3D op(elastic, run.lmax,
                          [](double, double, double) { return 0.0; },
                          OperatorOptionsOf(run, earth, elastic));
  // the sea level grid blocks its transforms for the operator's cache
  const GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  const auto times = std::vector<double>{0.0};
  const auto forward = Forward(top, block, [&] {
    return TrajectoryOf(run, 1, times, solver, op);
  });
  auto options = Evolution3D::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  options.record_stride = 1;
  if (forward.read) world.background = forward.store->Start();
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution3D evolution(op, solver, world.background, options);
  const auto snapshots = forward.read
                             ? forward.store->Snapshots3D()
                             : SolveStatic(run, solver, world, loads, evolution);
  if (!snapshots.front().converged) {
    throw std::runtime_error(
        "the sea level equation or the 3-D closure did not converge; loosen "
        "solver.rtol or solver.step_rtol, or raise "
        "solver.step_max_iterations");
  }
  if (forward.Records()) forward.store->Close();
  const auto trajectory =
      forward.store ? forward.store->Trajectory3D(snapshots)
                    : ForwardTrajectory3D::FromSnapshots(snapshots);
  auto recorded = Recorded3D{};
  recorded.run = &run;
  recorded.earth = &earth;
  recorded.solver = &solver;
  recorded.steps = OneNode();
  recorded.fixed = &world;
  recorded.loads = loads.get();
  recorded.read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                             InputsOf(loads.get()));
  recorded.rheology = &elastic;
  recorded.op = &op;
  recorded.evolution = &evolution;
  recorded.snapshots = snapshots;
  recorded.trajectory = &trajectory;
  return then(recorded) + forward.Note();
}

std::string WithRecorded1D(
    const Run& run, const TrajectoryBlock& block, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded1D&)>& then) {
  const auto rheology = RheologyOf(earth, top);
  const ViscoelasticOperator op(rheology, run.lmax);
  const GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                         ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  // the library's grid, a node where the forcing jumps repeated
  const auto steps = StepGridOf(run, solver.SolverScales());
  const auto& times = steps.times;
  const auto stride = block.record_stride;
  const auto forward = Forward(top, block, [&] {
    return TrajectoryOf(run, stride, times, solver);
  });
  auto options = Evolution::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  HoldLand(options, world);
  if (forward.read) {
    world.background = forward.store->Start();
  } else if (run.Calibrating()) {
    // calibrated by runs that record nothing; the recording run follows
    Evolution calibrating(op, solver, world.background, options);
    static_cast<void>(
        CalibrateWorld(run, top, solver, world, calibrating, times));
  }
  options.record_stride = stride;
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution evolution(op, solver, world.background, options);
  // the state calibrated under migrating shorelines, here or by the run
  // the store holds: marked, so that a sweep that would hold the initial
  // sea level where the library has no total derivative is refused
  if (run.Calibrating() && run.Migrating()) evolution.MarkCalibrated();
  const auto drive = DriveOf<Evolution::Forcing>(run, top, solver, world);
  const auto snapshots =
      forward.read ? forward.store->Snapshots1D()
                   : RunDriven(evolution, times, drive, run.Migrating());
  for (const auto& snapshot : snapshots) {
    Check1D(snapshot, WhenOf(run, solver.SolverScales(), snapshot.t),
            run.solver.rtol);
  }
  if (forward.Records()) forward.store->Close();
  auto recorded = Recorded1D{};
  recorded.run = &run;
  recorded.earth = &earth;
  recorded.solver = &solver;
  recorded.world = &world;
  recorded.loads = drive.loads.get();
  recorded.read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                             InputsOf(drive.loads.get()));
  recorded.rheology = &rheology;
  recorded.op = &op;
  recorded.evolution = &evolution;
  recorded.snapshots = snapshots;
  recorded.steps = steps;
  return then(recorded) + forward.Note();
}

std::string WithRecorded3D(
    const Run& run, const TrajectoryBlock& block, const Json& top,
    const BuiltEarth& earth, std::string_view meaning,
    const std::function<std::string(const Recorded3D&)>& then) {
  const auto rheology = RheologyOf(earth, top);
  const Viscoelastic3D op(rheology, run.lmax, DeltaOf(earth),
                          OperatorOptionsOf(run, earth, rheology));
  // the sea level grid blocks its transforms for the operator's cache
  const GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  // the library's grid, a node where the forcing jumps repeated
  const auto steps = StepGridOf(run, solver.SolverScales());
  const auto& times = steps.times;
  const auto stride = block.record_stride;
  const auto forward = Forward(top, block, [&] {
    return TrajectoryOf(run, stride, times, solver, op);
  });
  auto options = Evolution3D::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  HoldLand(options, world);
  if (forward.read) {
    world.background = forward.store->Start();
  } else if (run.Calibrating()) {
    Evolution3D calibrating(op, solver, world.background, options);
    static_cast<void>(
        CalibrateWorld(run, top, solver, world, calibrating, times));
  }
  options.record_stride = stride;
  // recording into the store, each kept state goes to disk as the run
  // passes it, and the sweep fetches it back
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution3D evolution(op, solver, world.background, options);
  // the calibrated state marked, as the radial run's
  if (run.Calibrating() && run.Migrating()) evolution.MarkCalibrated();
  const auto drive = DriveOf<Evolution3D::Forcing>(run, top, solver, world);
  const auto snapshots =
      forward.read ? forward.store->Snapshots3D()
                   : RunDriven(evolution, times, drive, run.Migrating());
  for (const auto& snapshot : snapshots) {
    Check3D(snapshot, WhenOf(run, solver.SolverScales(), snapshot.t),
            run.solver.step_rtol);
  }
  if (forward.Records()) forward.store->Close();
  const auto trajectory =
      forward.store ? forward.store->Trajectory3D(snapshots)
                    : ForwardTrajectory3D::FromSnapshots(snapshots);
  auto recorded = Recorded3D{};
  recorded.run = &run;
  recorded.earth = &earth;
  recorded.solver = &solver;
  recorded.world = &world;
  recorded.loads = drive.loads.get();
  recorded.read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                             InputsOf(drive.loads.get()));
  recorded.rheology = &rheology;
  recorded.op = &op;
  recorded.evolution = &evolution;
  recorded.snapshots = snapshots;
  recorded.trajectory = &trajectory;
  recorded.steps = steps;
  return then(recorded) + forward.Note();
}

}  // namespace sl3dpp::app
