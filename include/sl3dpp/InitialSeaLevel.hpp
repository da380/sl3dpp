#pragma once

/**
 * @file InitialSeaLevel.hpp
 * @brief The initial sea level of a run, calibrated so that the run reaches
 * a given final sea level: a port of pyslfp's
 * `SeaLevelEvolution.final_sea_level_misfit` and
 * `calibrate_initial_sea_level` (pyslfp 3.0.0; its docstrings and
 * `pyslfp_theory.tex`, "Calibrating the initial sea level", are the
 * authoritative statements).
 *
 * A thickness-only ice history leaves the initial sea level S0 to be
 * found, and the standard device is to choose it so that the run
 * reproduces the observed present-day sea level S_f. Write the final sea
 * level as S0 + xi(t_f; S0) and the residual as r = S0 + xi(t_f) - S_f,
 * with the weighted misfit J = (1/2) int w r^2 dS.
 *
 * Under frozen shorelines the problem is linear: the shorelines are those
 * of the background state, held fixed, and the run does not depend on S0 at
 * all. One run gives the exact answer, S0 <- S_f - xi(t_f) wherever the
 * weights are positive, by either method, and the calibrated state keeps
 * those shorelines as a prescribed ocean function, so that a frozen run
 * from it reproduces S_f.
 *
 * Under migrating shorelines the initial shorelines follow from S0 by
 * flotation, and each later load depends on S0 through the shorelines that
 * move, so the calibration iterates, by the classical update S0 <- S_f -
 * xi(t_f; S0) (Kendall et al. 2005; Newton's method with the Jacobian
 * I + A replaced by the identity) or by steepest descent with the adjoint
 * gradient (I + A^T) w r (Yu et al. 2025, eq. 87, Section 3.9), the
 * weighted residual plus the initial sea level kernel of a sea level
 * measurement w r at t_f. Either update is accelerated by Anderson mixing.
 *
 * Units are the evolution's: solver-unit fields on its grid, times
 * increasing in SLE units. The 3D evolution is frozen-shoreline only, and
 * so is its calibration.
 */

#include <functional>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"

namespace sl3dpp {

/// How a run treats its shorelines: held at the background state's
/// (`SeaLevelEvolution::Run`) or following flotation (`RunMigrating`).
enum class Shorelines { Frozen, Migrating };

/// An ice history as either shoreline treatment takes it: under frozen
/// shorelines the non-ocean surface load at t (`Run`'s iceLoad, built
/// against the background state's ocean function), under migrating ones
/// the ice thickness at t (`RunMigrating`'s iceThickness). Solver units.
struct IceHistory {
  Shorelines shorelines = Shorelines::Frozen;
  std::function<std::vector<double>(double)> field;

  [[nodiscard]] static IceHistory Frozen(
      std::function<std::vector<double>(double)> iceLoad) {
    return {Shorelines::Frozen, std::move(iceLoad)};
  }
  [[nodiscard]] static IceHistory Migrating(
      std::function<std::vector<double>(double)> iceThickness) {
    return {Shorelines::Migrating, std::move(iceThickness)};
  }
};

/// Runs the history from the evolution's background state, by `Run` or
/// `RunMigrating`.
[[nodiscard]] std::vector<SeaLevelEvolution::Snapshot> RunIceHistory(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history);

/// The misfit of a run's final sea level to a target, and its gradient
/// with respect to the initial sea level.
struct FinalSeaLevelMisfit {
  double value = 0.0;            ///< J = (1/2) int w r^2 dS
  std::vector<double> residual;  ///< r = S0 + xi(t_f) - S_f
  /// dJ / dS0 in the surface pairing, empty when not asked for.
  std::vector<double> gradient;
  std::vector<SeaLevelEvolution::Snapshot> snapshots;  ///< the forward run
};

/**
 * @brief The misfit of the run from the evolution's background state (its
 * sea level is S0) to `finalSeaLevel` at the last node, with `weights`
 * (non-negative; empty is one everywhere).
 *
 * With `gradient`, dJ / dS0: under frozen shorelines the initial sea level
 * kernel vanishes identically, no adjoint sweep runs, and the gradient is
 * the weighted residual; under migrating shorelines it adds the initial
 * sea level kernel of a sea level measurement w r at the last node,
 * linearised about the run's C(t) — one adjoint sweep by `adjoint`
 * (pyslfp uses the discrete transpose, the default here). The gradient is
 * exact away from changes of the initial ocean function itself, where J is
 * not differentiable.
 */
[[nodiscard]] FinalSeaLevelMisfit ComputeFinalSeaLevelMisfit(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights = {}, bool gradient = true,
    AdjointSolver::Options adjoint = {.method = AdjointMethod::Discrete});

enum class CalibrationMethod {
  FixedPoint,  ///< the classical update S0 <- S0 - step w>0 r
  Gradient,    ///< steepest descent S0 <- S0 - step grad J
};

struct CalibrationOptions {
  CalibrationMethod method = CalibrationMethod::FixedPoint;
  /// Non-negative weights on the misfit; empty weights every point equally.
  /// The fixed point updates only where they are positive.
  std::vector<double> weights;
  /// The step length of either update under migrating shorelines.
  double step = 1.0;
  /// The most forward runs under migrating shorelines.
  int iterations = 10;
  /// Convergence under migrating shorelines: the largest update below rtol
  /// times the largest value of S0 (1e-4 is of order a metre on the Earth).
  double rtol = 1e-4;
  /// The depth of the Anderson acceleration; 0 is the plain iteration.
  int anderson_memory = 5;
  /// The adjoint sweep of the gradient method.
  AdjointSolver::Options adjoint{.method = AdjointMethod::Discrete};
  /// Called after each run with its iteration (from 1), its weighted rms
  /// misfit and the relative update it led to; empty by default.
  std::function<void(int iteration, double misfit, double update)>
      on_iteration{};
};

/// The outcome: the calibrated background state and the run from it.
struct InitialSeaLevelCalibration {
  SeaLevelSolver::State state;
  std::vector<SeaLevelEvolution::Snapshot> snapshots;
  /// The weighted rms residual of every run, solver units (frozen: the
  /// run's, then what is left where the weights vanish).
  std::vector<double> misfits;
  /// The largest update of each iteration over the largest value of S0.
  std::vector<double> updates;
  bool converged = false;
};

/**
 * @brief Calibrates the initial sea level so that the run reaches
 * `finalSeaLevel` at the last node, and adopts the calibrated state.
 *
 * The evolution's background state supplies the starting guess (its sea
 * level), the ice thickness at the first node and the forced land. Under
 * frozen shorelines its ocean function is kept as the prescribed one and
 * one run gives the answer; under migrating shorelines each iterate's
 * state takes the flotation ocean function of the guess, and the
 * iteration stops at the tolerance, returning the last state it ran with
 * that run (`converged` false if the iterations ran out first). On return
 * the evolution's background state is the calibrated one, so later runs
 * and adjoints use it.
 *
 * The evolution's options apply to every run: its recording stride keeps
 * the returned run's trajectory, but its streaming callbacks fire for
 * every run, so calibrate with an evolution that streams nothing.
 */
[[nodiscard]] InitialSeaLevelCalibration CalibrateInitialSeaLevel(
    SeaLevelEvolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options = {});

/// The 3D calibration result, as the 1D one.
struct InitialSeaLevelCalibration3D {
  SeaLevelSolver::State state;
  std::vector<SeaLevelEvolution3D::Snapshot> snapshots;
  std::vector<double> misfits;
  std::vector<double> updates;
  bool converged = false;
};

/// The frozen-shoreline calibration of a 3D evolution: one run under the
/// non-ocean `iceLoad`, the same linear answer as in 1D (the lateral
/// structure changes xi(t_f), not the linearity), the evolution adopting
/// the calibrated state. Only `weights` of the options applies.
[[nodiscard]] InitialSeaLevelCalibration3D CalibrateInitialSeaLevel(
    SeaLevelEvolution3D& evolution, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad,
    std::span<const double> finalSeaLevel,
    const CalibrationOptions& options = {});

}  // namespace sl3dpp
