#include "sl3dpp/InitialSeaLevel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sl3dpp/detail/AndersonMixer.hpp"

namespace sl3dpp {

namespace {

/// The weights as a field (one everywhere when empty), refused when off the
/// grid or negative.
std::vector<double> WeightsOf(std::span<const double> weights,
                              std::size_t fieldSize) {
  if (weights.empty()) return std::vector<double>(fieldSize, 1.0);
  if (weights.size() != fieldSize) {
    throw std::invalid_argument("the weights must be a field on the grid");
  }
  if (std::ranges::any_of(weights, [](double w) { return w < 0.0; })) {
    throw std::invalid_argument("the weights must be non-negative");
  }
  return {weights.begin(), weights.end()};
}

double MaxAbs(std::span<const double> field) {
  auto out = 0.0;
  for (const auto v : field) out = std::max(out, std::abs(v));
  return out;
}

/// The weighted rms of a residual, sqrt(int w r^2 dS / int w dS).
double WeightedRms(const SurfaceGrid& grid, std::span<const double> w,
                   std::span<const double> residual) {
  auto squared = std::vector<double>(w.size());
  for (auto i = std::size_t{0}; i < w.size(); ++i) {
    squared[i] = w[i] * residual[i] * residual[i];
  }
  return std::sqrt(grid.Integrate(squared) / grid.Integrate(w));
}

void CheckField(std::span<const double> field, std::size_t fieldSize) {
  if (field.size() != fieldSize) {
    throw std::invalid_argument(
        "the final sea level must be a field on the solver's grid");
  }
}

/// The frozen-shoreline calibration both evolutions share: one run, the
/// exact answer where the weights are positive, the background's ocean
/// function kept as the prescribed one.
template <typename Evolution, typename Result, typename RunFn>
Result CalibrateFrozen(Evolution& evolution, std::span<const double> finalSea,
                       std::span<const double> weights, RunFn&& run) {
  const auto& grid = evolution.Solver().Grid();
  const auto fieldSize = grid.FieldSize();
  CheckField(finalSea, fieldSize);
  const auto w = WeightsOf(weights, fieldSize);
  auto snapshots = run();
  auto state = evolution.BackgroundState();
  const auto& xi = snapshots.back().relativeSeaLevelChange;
  auto residual = std::vector<double>(fieldSize);
  auto applied = std::vector<double>(fieldSize);
  auto left = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    residual[i] = state.seaLevel[i] + xi[i] - finalSea[i];
    applied[i] = w[i] > 0.0 ? residual[i] : 0.0;
    left[i] = w[i] > 0.0 ? 0.0 : residual[i];
    state.seaLevel[i] -= applied[i];
  }
  Result out;
  out.misfits = {WeightedRms(grid, w, residual), WeightedRms(grid, w, left)};
  out.updates = {MaxAbs(applied) / MaxAbs(state.seaLevel)};
  out.converged = true;
  evolution.SetBackgroundState(state);
  out.state = std::move(state);
  out.snapshots = std::move(snapshots);
  return out;
}

/// The misfit both evolutions share; `initialSeaLevelKernel(snapshots,
/// measurement)` is the initial sea level kernel of a sea level measurement
/// at the last node, about the run's own shorelines.
template <typename Misfit, typename Evolution, typename KernelFn>
Misfit MisfitOf(Evolution& evolution, std::span<const double> times,
                const IceHistory& history,
                std::span<const double> finalSeaLevel,
                std::span<const double> weights, bool gradient,
                KernelFn&& initialSeaLevelKernel) {
  const auto& solver = evolution.Solver();
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  CheckField(finalSeaLevel, fieldSize);
  const auto w = WeightsOf(weights, fieldSize);

  Misfit out;
  out.snapshots = RunIceHistory(evolution, times, history);
  const auto& S0 = evolution.BackgroundState().seaLevel;
  const auto& xi = out.snapshots.back().relativeSeaLevelChange;
  out.residual.resize(fieldSize);
  auto squared = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    out.residual[i] = S0[i] + xi[i] - finalSeaLevel[i];
    squared[i] = w[i] * out.residual[i] * out.residual[i];
  }
  const auto r2 = solver.Radius() * solver.Radius();
  out.value = 0.5 * r2 * grid.Integrate(squared);
  if (!gradient) return out;

  out.gradient.resize(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    out.gradient[i] = w[i] * out.residual[i];
  }
  if (history.shorelines == Shorelines::Migrating) {
    // The initial sea level kernel of a sea level measurement w r at the
    // last node, about the run's own shorelines.
    auto measurement = Measurement{};
    measurement.relativeSeaLevelDual = out.gradient;
    const auto kernel = initialSeaLevelKernel(out.snapshots, measurement);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      out.gradient[i] += kernel[i];
    }
  }
  return out;
}

/// The migrating-shoreline iteration both evolutions share;
/// `misfitOf(gradient)` runs from the evolution's background state.
template <typename Result, typename Evolution, typename MisfitFn>
Result CalibrateMigrating(Evolution& evolution,
                          std::span<const double> finalSeaLevel,
                          const CalibrationOptions& options,
                          MisfitFn&& misfitOf) {
  const auto& solver = evolution.Solver();
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  CheckField(finalSeaLevel, fieldSize);
  const auto w = WeightsOf(options.weights, fieldSize);
  const auto background = evolution.BackgroundState();

  auto mixer = detail::AndersonMixer(options.anderson_memory);
  auto guess = background.seaLevel;
  Result out;
  auto misfit = decltype(misfitOf(true)){};
  for (auto n = 0; n < options.iterations; ++n) {
    // Each iterate's shorelines follow from its sea level by flotation.
    out.state =
        solver.MakeState(background.iceThickness, guess, background.forcedLand);
    evolution.SetBackgroundState(out.state);
    misfit = misfitOf(options.method == CalibrationMethod::Gradient);
    out.misfits.push_back(WeightedRms(grid, w, misfit.residual));
    auto image = guess;
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const auto direction = options.method == CalibrationMethod::FixedPoint
                                 ? (w[i] > 0.0 ? misfit.residual[i] : 0.0)
                                 : misfit.gradient[i];
      image[i] -= options.step * direction;
    }
    const auto updated = mixer.Update(guess, image);
    auto change = 0.0;
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      change = std::max(change, std::abs(updated[i] - guess[i]));
    }
    const auto scale = MaxAbs(updated);
    out.updates.push_back(change / scale);
    if (options.on_iteration) {
      options.on_iteration(n + 1, out.misfits.back(), out.updates.back());
    }
    if (change <= options.rtol * scale) {
      out.converged = true;
      break;
    }
    guess = updated;
  }
  // The last state run, with that run, so the two are consistent; the
  // evolution already holds it, and is marked as holding a calibrated
  // state.
  evolution.MarkCalibrated();
  out.snapshots = std::move(misfit.snapshots);
  return out;
}

void CheckIterations(const CalibrationOptions& options) {
  if (options.iterations < 1) {
    throw std::invalid_argument("iterations must be at least one");
  }
}

}  // namespace

namespace {

/// The run of an ice history by either evolution: its left limit, where
/// given, through the evolution's forcing.
template <class EvolutionT>
std::vector<typename EvolutionT::Snapshot> RunHistory(
    EvolutionT& evolution, std::span<const double> times,
    const IceHistory& history) {
  const auto frozen = history.shorelines == Shorelines::Frozen;
  if (!history.before) {
    return frozen ? evolution.Run(times, history.field)
                  : evolution.RunMigrating(times, history.field);
  }
  auto forcing = typename EvolutionT::Forcing{};
  if (frozen) {
    forcing.directLoad = history.field;
    forcing.directLoadBefore = history.before;
    return evolution.Run(times, forcing);
  }
  forcing.iceThicknessBefore = history.before;
  return evolution.RunMigrating(times, history.field, forcing);
}

}  // namespace

std::vector<Evolution::Snapshot> RunIceHistory(
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history) {
  return RunHistory(evolution, times, history);
}

std::vector<Evolution3D::Snapshot> RunIceHistory(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history) {
  return RunHistory(evolution, times, history);
}

FinalSeaLevelMisfit ComputeFinalSeaLevelMisfit(
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights, bool gradient,
    AdjointSolver::Options adjoint) {
  return MisfitOf<FinalSeaLevelMisfit>(
      evolution, times, history, finalSeaLevel, weights, gradient,
      [&](const auto& snapshots, const Measurement& measurement) {
        return AdjointSolver(evolution, std::move(adjoint))
            .FirstOrderKernels(times, {{times.size() - 1, measurement}},
                               nullptr, OceanFunctionsOf(snapshots),
                               StageOceanFunctionsOf(snapshots))
            .initialSeaLevel;
      });
}

FinalSeaLevelMisfit3D ComputeFinalSeaLevelMisfit(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights, bool gradient,
    const AdjointSolver3D::Options& adjoint) {
  return MisfitOf<FinalSeaLevelMisfit3D>(
      evolution, times, history, finalSeaLevel, weights, gradient,
      [&](const auto& snapshots, const Measurement& measurement) {
        return AdjointSolver3D(evolution, adjoint)
            .FirstOrderKernels(times, {{times.size() - 1, measurement}},
                               nullptr, OceanFunctionsOf(snapshots),
                               StageOceanFunctionsOf(snapshots))
            .ice.initialSeaLevel;
      });
}

InitialSeaLevelCalibration CalibrateInitialSeaLevel(
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options) {
  CheckIterations(options);
  if (history.shorelines == Shorelines::Frozen) {
    // With the shorelines fixed the problem is linear and the run does not
    // depend on S0 at all: one run gives the misfit's exact minimiser, and
    // the run from it is the run already made.
    return CalibrateFrozen<Evolution, InitialSeaLevelCalibration>(
        evolution, finalSeaLevel, options.weights,
        [&] { return RunIceHistory(evolution, times, history); });
  }
  return CalibrateMigrating<InitialSeaLevelCalibration>(
      evolution, finalSeaLevel, options, [&](bool gradient) {
        return ComputeFinalSeaLevelMisfit(evolution, times, history,
                                          finalSeaLevel, options.weights,
                                          gradient, options.adjoint);
      });
}

InitialSeaLevelCalibration3D CalibrateInitialSeaLevel(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options) {
  CheckIterations(options);
  if (history.shorelines == Shorelines::Frozen) {
    return CalibrateFrozen<Evolution3D, InitialSeaLevelCalibration3D>(
        evolution, finalSeaLevel, options.weights,
        [&] { return RunIceHistory(evolution, times, history); });
  }
  return CalibrateMigrating<InitialSeaLevelCalibration3D>(
      evolution, finalSeaLevel, options, [&](bool gradient) {
        return ComputeFinalSeaLevelMisfit(evolution, times, history,
                                          finalSeaLevel, options.weights,
                                          gradient, options.adjoint3D);
      });
}

namespace {

/// The iteration both evolutions share; `initialSeaLevelKernel(m)` is the
/// initial sea level kernel of the measurements m by a sweep that forms
/// nothing else it need not.
template <typename KernelFn>
CalibrationMultiplier MultiplierOf(
    std::size_t fieldSize, std::size_t last,
    const std::map<std::size_t, Measurement>& measurements, bool migrating,
    const CalibrationMultiplierOptions& options,
    KernelFn&& initialSeaLevelKernel) {
  if (options.iterations < 1) {
    throw std::invalid_argument("iterations must be at least one");
  }
  if (!(options.rtol > 0.0)) {
    throw std::invalid_argument("rtol must be positive");
  }
  const auto w = WeightsOf(options.weights, fieldSize);
  CalibrationMultiplier out;
  out.multiplier.assign(fieldSize, 0.0);
  out.measurements = measurements;
  if (!migrating) {
    // the run does not depend on its initial sea level: k0 and A vanish
    out.converged = true;
    return out;
  }
  // the measurements less a sea level measurement lambda at the last node
  const auto forcing = [&](std::span<const double> lambda) {
    auto forced = measurements;
    auto& dual = forced[last].relativeSeaLevelDual;
    if (dual.empty()) dual.assign(fieldSize, 0.0);
    if (dual.size() != fieldSize) {
      throw std::invalid_argument(
          "the sea level dual at the last node must be a field on the grid");
    }
    for (auto i = std::size_t{0}; i < fieldSize; ++i) dual[i] -= lambda[i];
    return forced;
  };
  auto mixer = detail::AndersonMixer(options.anderson_memory);
  auto& lambda = out.multiplier;
  for (auto n = 0; n < options.iterations; ++n) {
    auto image = initialSeaLevelKernel(forcing(lambda));
    if (image.size() != fieldSize) {
      throw std::logic_error("the sweep formed no initial sea level kernel");
    }
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      if (!(w[i] > 0.0)) image[i] = 0.0;
    }
    auto change = 0.0;
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      change = std::max(change, std::abs(image[i] - lambda[i]));
    }
    const auto scale = MaxAbs(image);
    const auto update = scale > 0.0 ? change / scale : 0.0;
    out.updates.push_back(update);
    ++out.sweeps;
    if (options.on_iteration) options.on_iteration(n + 1, update);
    if (update <= options.rtol) {
      lambda = std::move(image);
      out.converged = true;
      break;
    }
    lambda = mixer.Update(lambda, image);
  }
  out.measurements = forcing(lambda);
  return out;
}

/// A sweep's options with everything the initial sea level kernel does
/// not need switched off.
template <typename Options>
Options IceOnly(Options adjoint) {
  adjoint.material_kernels = false;
  adjoint.tide_kernels = false;
  adjoint.moment_points.clear();
  return adjoint;
}

void RequireNodes(std::span<const double> times) {
  if (times.empty()) {
    throw std::invalid_argument("the time grid holds no node");
  }
}

template <typename Result, typename Kernels>
Result WithMultiplier(Kernels kernels, CalibrationMultiplier&& multiplier) {
  Result out;
  out.kernels = std::move(kernels);
  out.multiplier = std::move(multiplier.multiplier);
  out.updates = std::move(multiplier.updates);
  out.sweeps = multiplier.sweeps + 1;
  out.converged = multiplier.converged;
  return out;
}

}  // namespace

CalibrationMultiplier SolveCalibrationMultiplier(
    const Evolution& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    const CalibrationMultiplierOptions& options,
    AdjointSolver::Options adjoint) {
  RequireNodes(times);
  const AdjointSolver sweep(evolution, IceOnly(std::move(adjoint)));
  return MultiplierOf(
      evolution.Solver().Grid().FieldSize(), times.size() - 1, measurements,
      !oceanFunctions.empty(), options,
      [&](const std::map<std::size_t, Measurement>& forced) {
        return sweep
            .FirstOrderKernels(times, forced, nullptr, oceanFunctions,
                               stageOceanFunctions)
            .initialSeaLevel;
      });
}

CalibrationMultiplier SolveCalibrationMultiplier(
    const Evolution3D& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    const CalibrationMultiplierOptions& options,
    AdjointSolver3D::Options adjoint) {
  RequireNodes(times);
  const AdjointSolver3D sweep(evolution, IceOnly(std::move(adjoint)));
  return MultiplierOf(
      evolution.Solver().Grid().FieldSize(), times.size() - 1, measurements,
      !oceanFunctions.empty(), options,
      [&](const std::map<std::size_t, Measurement>& forced) {
        return sweep
            .FirstOrderKernels(times, forced, nullptr, oceanFunctions,
                               stageOceanFunctions)
            .ice.initialSeaLevel;
      });
}

CalibratedKernels ComputeCalibratedKernels(
    const Evolution& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    const CalibrationMultiplierOptions& options,
    AdjointSolver::Options adjoint) {
  auto multiplier = SolveCalibrationMultiplier(
      evolution, times, measurements, oceanFunctions, stageOceanFunctions,
      options, adjoint);
  auto kernels = AdjointSolver(evolution, std::move(adjoint))
                     .FirstOrderKernels(times, multiplier.measurements,
                                        trajectory, oceanFunctions,
                                        stageOceanFunctions);
  return WithMultiplier<CalibratedKernels>(std::move(kernels),
                                           std::move(multiplier));
}

CalibratedKernels3D ComputeCalibratedKernels(
    const Evolution3D& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    const CalibrationMultiplierOptions& options,
    const AdjointSolver3D::Options& adjoint) {
  auto multiplier = SolveCalibrationMultiplier(
      evolution, times, measurements, oceanFunctions, stageOceanFunctions,
      options, adjoint);
  auto kernels = AdjointSolver3D(evolution, adjoint)
                     .FirstOrderKernels(times, multiplier.measurements,
                                        trajectory, oceanFunctions,
                                        stageOceanFunctions);
  return WithMultiplier<CalibratedKernels3D>(std::move(kernels),
                                             std::move(multiplier));
}

void RefuseCalibratedSecondOrderKernels() {
  throw std::logic_error(
      "the second-order kernels of a calibrated run under migrating "
      "shorelines (the derivative of its total kernels, the initial sea "
      "level calibrated again) are not built; SecondOrderKernels about the "
      "calibrated state gives the derivative of the partial kernels, the "
      "initial sea level held, when the sweep's "
      "Options::initial_sea_level_held says that is what is wanted");
}

}  // namespace sl3dpp
