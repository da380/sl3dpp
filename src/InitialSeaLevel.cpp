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
  const auto& xi = snapshots.back().seaLevelChange;
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
  const auto& xi = out.snapshots.back().seaLevelChange;
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
    measurement.seaLevelDual = out.gradient;
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
  // evolution already holds it.
  out.snapshots = std::move(misfit.snapshots);
  return out;
}

void CheckIterations(const CalibrationOptions& options) {
  if (options.iterations < 1) {
    throw std::invalid_argument("iterations must be at least one");
  }
}

}  // namespace

std::vector<SeaLevelEvolution::Snapshot> RunIceHistory(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history) {
  return history.shorelines == Shorelines::Frozen
             ? evolution.Run(times, history.field)
             : evolution.RunMigrating(times, history.field);
}

std::vector<SeaLevelEvolution3D::Snapshot> RunIceHistory(
    SeaLevelEvolution3D& evolution, std::span<const double> times,
    const IceHistory& history) {
  return history.shorelines == Shorelines::Frozen
             ? evolution.Run(times, history.field)
             : evolution.RunMigrating(times, history.field);
}

FinalSeaLevelMisfit ComputeFinalSeaLevelMisfit(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights, bool gradient,
    AdjointSolver::Options adjoint) {
  return MisfitOf<FinalSeaLevelMisfit>(
      evolution, times, history, finalSeaLevel, weights, gradient,
      [&](const auto& snapshots, const Measurement& measurement) {
        return AdjointSolver(evolution, std::move(adjoint))
            .IceHistoryKernels(times, {{times.size() - 1, measurement}},
                               nullptr, OceanFunctionsOf(snapshots))
            .initialSeaLevel;
      });
}

FinalSeaLevelMisfit3D ComputeFinalSeaLevelMisfit(
    SeaLevelEvolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights, bool gradient,
    const AdjointSolver3D::Options& adjoint) {
  return MisfitOf<FinalSeaLevelMisfit3D>(
      evolution, times, history, finalSeaLevel, weights, gradient,
      [&](const auto& snapshots, const Measurement& measurement) {
        return AdjointSolver3D(evolution, adjoint)
            .IceHistoryKernels(times, {{times.size() - 1, measurement}},
                               nullptr, OceanFunctionsOf(snapshots))
            .ice.initialSeaLevel;
      });
}

InitialSeaLevelCalibration CalibrateInitialSeaLevel(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options) {
  CheckIterations(options);
  if (history.shorelines == Shorelines::Frozen) {
    // With the shorelines fixed the problem is linear and the run does not
    // depend on S0 at all: one run gives the misfit's exact minimiser, and
    // the run from it is the run already made.
    return CalibrateFrozen<SeaLevelEvolution, InitialSeaLevelCalibration>(
        evolution, finalSeaLevel, options.weights,
        [&] { return evolution.Run(times, history.field); });
  }
  return CalibrateMigrating<InitialSeaLevelCalibration>(
      evolution, finalSeaLevel, options, [&](bool gradient) {
        return ComputeFinalSeaLevelMisfit(evolution, times, history,
                                          finalSeaLevel, options.weights,
                                          gradient, options.adjoint);
      });
}

InitialSeaLevelCalibration3D CalibrateInitialSeaLevel(
    SeaLevelEvolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options) {
  CheckIterations(options);
  if (history.shorelines == Shorelines::Frozen) {
    return CalibrateFrozen<SeaLevelEvolution3D, InitialSeaLevelCalibration3D>(
        evolution, finalSeaLevel, options.weights,
        [&] { return evolution.Run(times, history.field); });
  }
  return CalibrateMigrating<InitialSeaLevelCalibration3D>(
      evolution, finalSeaLevel, options, [&](bool gradient) {
        return ComputeFinalSeaLevelMisfit(evolution, times, history,
                                          finalSeaLevel, options.weights,
                                          gradient, options.adjoint3D);
      });
}

}  // namespace sl3dpp
