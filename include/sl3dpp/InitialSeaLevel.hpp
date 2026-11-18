#pragma once

/**
 * @file InitialSeaLevel.hpp
 * @brief The initial sea level of a run, calibrated so that the run reaches
 * a given final sea level, as pyslfp's
 * `Evolution.final_sea_level_misfit` and
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
 * increasing in SLE units. The 3D evolution calibrates the same way under
 * either treatment; its gradient comes from the 3D sweep `adjoint3D`
 * selects (the discrete adjoint by default, as in 1D; the continuous one
 * is exact as the time step goes to zero). The gradient method's fixed
 * point does not depend on the sweep's accuracy: the kernel is
 * linear in the residual, so the update vanishes where the residual does.
 *
 * The derivatives of a calibrated run. A calibrated run's initial sea level
 * is not a parameter: it follows the run's parameters p (the ice history,
 * the relaxation times, the moduli) and the target through the converged
 * condition S0 + xi(t_f; p, S0) = S_f on the points P where the weights are
 * positive (S0 held elsewhere). With A = d xi(t_f) / d S0, that condition
 * gives (I + A) dS0 = dS_f - (d xi(t_f) / dp) dp on P, whatever the method
 * that reached it, so for an objective J with the initial sea level kernel
 * k0 = dJ / dS0 (the parameters held)
 *
 *   dJ = (dJ / dp) dp - <lambda, (d xi(t_f) / dp) dp> + <lambda, dS_f>,
 *   lambda = P (k0 - A^T lambda),
 *
 * <.,.> the surface pairing of the duals (Adjoint.hpp). A^T lambda is the
 * initial sea level kernel of a sea level measurement lambda at the last
 * node, so the sweep forced by J's measurements less that measurement gives
 * at once every kernel of the calibrated run (the TOTAL derivatives, where
 * the sweep forced by J's alone gives the PARTIAL ones, the initial sea
 * level held) and, as its initial sea level kernel on P, lambda again: Yu
 * et al. 2025, eqs. (89)-(92), the multiplier SL0-dagger of the constraint
 * entering the adjoint load at the time-reversed present. lambda is the
 * kernel of J in the target S_f. Under frozen shorelines k0 and A vanish,
 * lambda is zero and the partial kernels are the total ones.
 * SolveCalibrationMultiplier finds lambda by iterating that relation, and
 * ComputeCalibratedKernels follows it with the one full sweep. The
 * relation is of the converged state: the kernels are the calibrated
 * run's as far as the calibration's own tolerance holds the condition.
 *
 * The same condition in the forward mode is the LINEARISED CALIBRATION: the
 * tangent-linear run of a calibrated run along a direction dp (and a change
 * dS_f of the target) has the change of the calibrated initial sea level in
 * it, the solution of (I + A) dS0 = dS_f - (d xi(t_f) / dp) dp on P. The
 * right side is the tangent-linear run along dp, the initial sea level
 * held, read at the last node; A dS0 is the run along an initial sea level
 * dS0 alone, read there. SolveCalibrationTangent iterates dS0 = P (dS_f -
 * (d xi(t_f) / dp) dp - A dS0), the transpose of the multiplier's relation
 * and the same iteration (one loop serves both), and
 * ComputeCalibratedTangent follows it with the run along dp and dS0
 * together: the derivative of the calibrated run, whose pairing with an
 * objective's duals is the total kernels' with dp plus lambda's with dS_f.
 */

#include <functional>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"

namespace sl3dpp {

/// How a run treats its shorelines: held at the background state's
/// (`Evolution::Run`) or following flotation (`RunMigrating`).
enum class Shorelines { Frozen, Migrating };

/// An ice history as either shoreline treatment takes it: under frozen
/// shorelines the non-ocean surface load at t (`Run`'s directLoad, built
/// against the background state's ocean function), under migrating ones
/// the ice thickness at t (`RunMigrating`'s iceThickness). Solver units.
/// `before`, where given, is its left limit at a jump, a node repeated in
/// the time grid (Evolution::Forcing::directLoadBefore under frozen
/// shorelines, iceThicknessBefore under migrating ones, whose run then
/// takes a forcing and solves its start).
struct IceHistory {
  Shorelines shorelines = Shorelines::Frozen;
  std::function<std::vector<double>(double)> field;
  std::function<std::vector<double>(double)> before{};

  [[nodiscard]] static IceHistory Frozen(
      std::function<std::vector<double>(double)> directLoad,
      std::function<std::vector<double>(double)> before = {}) {
    return {Shorelines::Frozen, std::move(directLoad), std::move(before)};
  }
  [[nodiscard]] static IceHistory Migrating(
      std::function<std::vector<double>(double)> iceThickness,
      std::function<std::vector<double>(double)> before = {}) {
    return {Shorelines::Migrating, std::move(iceThickness), std::move(before)};
  }
};

/// Runs the history from the evolution's background state, by `Run` or
/// `RunMigrating`.
[[nodiscard]] std::vector<Evolution::Snapshot> RunIceHistory(
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history);
[[nodiscard]] std::vector<Evolution3D::Snapshot> RunIceHistory(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history);

/// The misfit of a run's final sea level to a target, and its gradient
/// with respect to the initial sea level.
struct FinalSeaLevelMisfit {
  double value = 0.0;            ///< J = (1/2) int w r^2 dS
  std::vector<double> residual;  ///< r = S0 + xi(t_f) - S_f
  /// dJ / dS0 in the surface pairing, empty when not asked for.
  std::vector<double> gradient;
  std::vector<Evolution::Snapshot> snapshots;  ///< the forward run
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
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights = {}, bool gradient = true,
    AdjointSolver::Options adjoint = {});

/// The 3D misfit, as the 1D one.
struct FinalSeaLevelMisfit3D {
  double value = 0.0;
  std::vector<double> residual;
  std::vector<double> gradient;
  std::vector<Evolution3D::Snapshot> snapshots;
};

/// The misfit of a 3D run, as the 1D one; under migrating shorelines the
/// initial sea level kernel comes from the 3D sweep `adjoint` selects
/// (the discrete by default) about the run's C(t).
[[nodiscard]] FinalSeaLevelMisfit3D ComputeFinalSeaLevelMisfit(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    std::span<const double> weights = {}, bool gradient = true,
    const AdjointSolver3D::Options& adjoint = {});

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
  /// The adjoint sweep of the gradient method (the discrete by default,
  /// the exact transpose of the run's steps under every scheme, each stage
  /// of Sdirk2 about its own shorelines).
  AdjointSolver::Options adjoint{};
  /// The 3D evolution's sweep for the gradient method, as `adjoint`.
  AdjointSolver3D::Options adjoint3D{};
  /// Called after each run with its iteration (from 1), its weighted rms
  /// misfit and the relative update it led to; empty by default.
  std::function<void(int iteration, double misfit, double update)>
      on_iteration{};
};

/// The outcome: the calibrated background state and the run from it.
struct InitialSeaLevelCalibration {
  GRDSolver::State state;
  std::vector<Evolution::Snapshot> snapshots;
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
 * level), the ice thickness at the first node and the forced land there (a
 * land that changes in time is the evolution's Options::land_held at the
 * later nodes of every run). Under frozen shorelines its ocean function
 * is kept as the prescribed one and one run gives the answer; under
 * migrating shorelines each iterate's
 * state takes the flotation ocean function of the guess, and the
 * iteration stops at the tolerance, returning the last state it ran with
 * that run (`converged` false if the iterations ran out first). On return
 * the evolution's background state is the calibrated one, so later runs
 * and adjoints use it, and under migrating shorelines it is marked so
 * (Evolution::Calibrated).
 *
 * The evolution's options apply to every run: its recording stride keeps
 * the returned run's trajectory, but its streaming callbacks fire for
 * every run, so calibrate with an evolution that streams nothing.
 */
[[nodiscard]] InitialSeaLevelCalibration CalibrateInitialSeaLevel(
    Evolution& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options = {});

/// The 3D calibration result, as the 1D one.
struct InitialSeaLevelCalibration3D {
  GRDSolver::State state;
  std::vector<Evolution3D::Snapshot> snapshots;
  std::vector<double> misfits;
  std::vector<double> updates;
  bool converged = false;
};

/// The calibration of a 3D evolution, as the 1D one. Under frozen
/// shorelines one run gives the same linear answer as in 1D (the lateral
/// structure changes xi(t_f), not the linearity); under migrating ones the
/// iteration is the 1D one with 3D runs, the gradient method's sweep by
/// `options.adjoint3D`. The evolution adopts the calibrated state.
[[nodiscard]] InitialSeaLevelCalibration3D CalibrateInitialSeaLevel(
    Evolution3D& evolution, std::span<const double> times,
    const IceHistory& history, std::span<const double> finalSeaLevel,
    const CalibrationOptions& options = {});

/// The iteration for the calibration's multiplier.
struct CalibrationMultiplierOptions {
  /// The calibration's weights (CalibrationOptions::weights): the
  /// multiplier lives where they are positive, the points whose initial
  /// sea level the calibration set. Empty: every point.
  std::vector<double> weights;
  /// The most sweeps of the iteration.
  int iterations = 20;
  /// Convergence: the largest change of the multiplier between two sweeps
  /// below rtol times its largest value.
  double rtol = 1e-8;
  /// The depth of the Anderson acceleration of the iteration (the mixing
  /// the calibration uses). 0 is the plain iteration, which contracts at
  /// the rate of A, the dependence of the final sea level on the initial
  /// one through the shorelines that move; the relation is linear and A
  /// acts through the cells that flooded or grounded alone, so the mixed
  /// iteration ends in a few sweeps where the plain one takes tens.
  int anderson_memory = 5;
  /// Called after each sweep with its number (from 1) and the relative
  /// change of the multiplier it gave; empty by default.
  std::function<void(int sweep, double update)> on_iteration{};
};

/// The multiplier of a calibrated run's constraint for one objective.
struct CalibrationMultiplier {
  /// lambda, in the duals' surface pairing: dJ / dS_f, the kernel of the
  /// objective in the final sea level the run is calibrated to, zero where
  /// the weights vanish (and everywhere under frozen shorelines).
  std::vector<double> multiplier;
  /// The objective's measurements less a sea level measurement lambda at
  /// the last node: the forcing of the sweep whose kernels are the
  /// calibrated run's.
  std::map<std::size_t, Measurement> measurements;
  /// The relative change of the multiplier at each sweep.
  std::vector<double> updates;
  /// The sweeps run (none under frozen shorelines).
  int sweeps = 0;
  bool converged = false;
};

/**
 * @brief The multiplier lambda = P (k0 - A^T lambda) of the constraint that
 * the run about `evolution`'s background state reaches its target at the
 * last node, for the objective `measurements` define.
 *
 * `oceanFunctions` are the calibrated migrating run's (OceanFunctions::Of
 * its snapshots); empty ocean functions are a frozen run's, whose
 * multiplier is zero and takes no sweep. Each sweep is one of `adjoint`'s
 * method forced by the measurements less the current multiplier at the
 * last node, formed without a trajectory and with the material, tide and
 * moment kernels off (the initial sea level kernel needs none of them): the
 * first, from zero, gives the initial sea level kernel k0 of the objective.
 * By the discrete method the multiplier is that of the stepped run, under
 * every scheme; by the continuous one, of the continuous sweep's
 * approximation to k0 and A^T. `converged` is false when the sweeps ran out
 * first, the last iterate returned.
 */
[[nodiscard]] CalibrationMultiplier SolveCalibrationMultiplier(
    const Evolution& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    AdjointSolver::Options adjoint = {});

/// The multiplier about a 3D run, as the 1D one; collective over the
/// evolution's ranks as the 3D sweeps are, every rank's result alike.
[[nodiscard]] CalibrationMultiplier SolveCalibrationMultiplier(
    const Evolution3D& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    AdjointSolver3D::Options adjoint = {});

/// The kernels of a calibrated run: the total derivatives of the
/// objective, the initial sea level re-calibrated as the parameters
/// change.
struct CalibratedKernels {
  /// The kernels of the sweep forced by the measurements less the
  /// multiplier at the last node. Every kernel in a parameter of the run
  /// is the total one. `initialSeaLevel` is the multiplier again where the
  /// weights are positive, to the iteration's tolerance, and the total
  /// derivative in the initial sea level where they vanish (the points
  /// the calibration holds).
  Kernels kernels;
  std::vector<double> multiplier;  ///< CalibrationMultiplier::multiplier
  std::vector<double> updates;     ///< CalibrationMultiplier::updates
  /// The sweeps run: the multiplier's and the one that formed the kernels.
  int sweeps = 0;
  bool converged = false;
};

/**
 * @brief The total kernels of the objective about a calibrated run:
 * SolveCalibrationMultiplier, then AdjointSolver::FirstOrderKernels with
 * `adjoint` under the measurements it returns (with `trajectory`, the
 * kernels in the relaxation times and the moduli too). Under frozen
 * shorelines (empty ocean functions) the one sweep of the measurements,
 * whose kernels are already the calibrated run's.
 */
[[nodiscard]] CalibratedKernels ComputeCalibratedKernels(
    const Evolution& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    AdjointSolver::Options adjoint = {});

/// The 3D total kernels, as the 1D ones.
struct CalibratedKernels3D {
  Kernels3D kernels;
  std::vector<double> multiplier;
  std::vector<double> updates;
  int sweeps = 0;
  bool converged = false;
};

/// The total kernels about a calibrated 3D run, as the 1D ones; collective
/// as the 3D sweeps are.
[[nodiscard]] CalibratedKernels3D ComputeCalibratedKernels(
    const Evolution3D& evolution, std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    const AdjointSolver3D::Options& adjoint = {});

/// The change of a calibrated run's initial sea level along a direction:
/// the linearised calibration's solution.
struct CalibrationTangent {
  /// dS0, solver units on the grid: where the weights are positive, the
  /// solution of (I + A) dS0 = dS_f - (d xi(t_f) / dp) dp; elsewhere, where
  /// the calibration holds the initial sea level, the direction's own.
  std::vector<double> initialSeaLevel;
  /// The relative change of dS0 at each pass of the iteration.
  std::vector<double> updates;
  /// The tangent-linear runs made: the one along the direction, and one
  /// for each pass of the iteration after the first.
  int runs = 0;
  bool converged = false;
};

/**
 * @brief The change of the calibrated initial sea level of the run about
 * `evolution`'s background state along `direction` and a change
 * `finalSeaLevel` of the sea level the run is calibrated to (solver units
 * on the grid; empty, the target held).
 *
 * The transpose of SolveCalibrationMultiplier, and its iteration: `options`
 * are the multiplier's (the weights that choose the calibrated points, the
 * most passes, the tolerance on the largest change of dS0 over its largest
 * value, the mixing's depth; `on_iteration` is called after each pass).
 * The first run is the tangent-linear run along `direction` by `tangent`,
 * with `trajectory` where the direction needs it; each pass after the first
 * is a run along an initial sea level alone, which needs none. About a
 * frozen run (empty ocean functions) the run does not depend on its initial
 * sea level, A vanishes and dS0 is the right side itself, with no
 * iteration. A direction's own `initialSeaLevel` is the change of the
 * initial sea level where the calibration holds it (the weights zero), and
 * is refused (std::invalid_argument) where it is not zero on the points
 * the calibration sets. An empty direction is a change of the target alone.
 * `converged` is false when the passes ran out first, the last iterate
 * returned.
 */
[[nodiscard]] CalibrationTangent SolveCalibrationTangent(
    const Evolution& evolution, std::span<const double> times,
    const TangentPerturbation& direction,
    std::span<const double> finalSeaLevel,
    const ForwardTrajectory* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    TangentLinearSolver::Options tangent = {});

/// The linearised calibration about a 3D run, as the 1D one; collective
/// over the evolution's ranks as the 3D tangent-linear runs are, every
/// rank's result alike.
[[nodiscard]] CalibrationTangent SolveCalibrationTangent(
    const Evolution3D& evolution, std::span<const double> times,
    const TangentPerturbation3D& direction,
    std::span<const double> finalSeaLevel,
    const ForwardTrajectory3D* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    TangentLinearSolver3D::Options tangent = {});

/// The tangent-linear run of a calibrated run: the derivative of its
/// responses along a direction, the initial sea level re-calibrated.
struct CalibratedTangent {
  /// The run along the direction and the change of the initial sea level
  /// together, as TangentLinearSolver::Run returns it.
  std::vector<Evolution::Snapshot> snapshots;
  std::vector<double> initialSeaLevel;  ///< CalibrationTangent's
  std::vector<double> updates;          ///< CalibrationTangent::updates
  /// The tangent-linear runs made: the linearised calibration's and, about
  /// a migrating run, the one that formed the snapshots.
  int runs = 0;
  bool converged = false;
};

/**
 * @brief The total tangent-linear run about a calibrated run:
 * SolveCalibrationTangent, then TangentLinearSolver::Run along `direction`
 * with the initial sea level it found (the forward mode of
 * ComputeCalibratedKernels: an objective's duals paired with these
 * snapshots are its total kernels paired with the direction, plus its
 * multiplier paired with `finalSeaLevel`). About a frozen run the initial
 * sea level moves nothing, and the snapshots are the run along the
 * direction itself.
 */
[[nodiscard]] CalibratedTangent ComputeCalibratedTangent(
    const Evolution& evolution, std::span<const double> times,
    const TangentPerturbation& direction,
    std::span<const double> finalSeaLevel,
    const ForwardTrajectory* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    TangentLinearSolver::Options tangent = {});

/// The 3D total tangent-linear run, as the 1D one.
struct CalibratedTangent3D {
  std::vector<Evolution3D::Snapshot> snapshots;
  std::vector<double> initialSeaLevel;
  std::vector<double> updates;
  int runs = 0;
  bool converged = false;
};

/// The total tangent-linear run about a calibrated 3D run, as the 1D one;
/// collective as the 3D tangent-linear runs are.
[[nodiscard]] CalibratedTangent3D ComputeCalibratedTangent(
    const Evolution3D& evolution, std::span<const double> times,
    const TangentPerturbation3D& direction,
    std::span<const double> finalSeaLevel,
    const ForwardTrajectory3D* trajectory,
    const OceanFunctions& oceanFunctions,
    const CalibrationMultiplierOptions& options = {},
    TangentLinearSolver3D::Options tangent = {});

/**
 * @brief The second-order kernels of a calibrated migrating run, the
 * derivative along a direction of its total kernels: NOT BUILT, and
 * refused (std::logic_error) naming what is. They need the derivative of
 * the multiplier along the direction, a second linear problem of the
 * first's form forced by the second-order sweep. What is built is the
 * second-order sweep with the initial sea level held
 * (AdjointSolver::SecondOrderKernels, AdjointSolver3D::SecondOrderKernels
 * about the calibrated state), whose kernel derivatives are those of the
 * partial kernels (and, under frozen shorelines, the calibrated run's
 * own: only a migrating run's are refused). Those sweeps call this about
 * an evolution whose state is marked calibrated (Evolution::Calibrated;
 * CalibrateInitialSeaLevel marks it under migrating shorelines) and given
 * ocean functions, unless their Options::initial_sea_level_held states
 * that the partial ones are wanted.
 */
[[noreturn]] void RefuseCalibratedSecondOrderKernels();

}  // namespace sl3dpp
