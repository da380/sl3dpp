#pragma once

/**
 * @file Adjoint.hpp
 * @brief Ice-history sensitivity kernels of the coupled evolution, by the
 * continuous adjoint (the preferred path) or the exact discrete transpose
 * — one interface and result type for both, so every gate and driver is
 * formulation-blind. A port of pyslfp's `adjoint_ice_history` /
 * `continuous_adjoint_ice_history` (whose docstrings, with Yu et al. 2025,
 * are the authoritative statements); the design record is docs/dev/adjoints.md.
 *
 * The discrete adjoint transposes the time-stepped forward map exactly:
 * the gradient identity holds to solver tolerance for any load
 * perturbation on the time grid. The continuous adjoint discretises the
 * continuum adjoint equations — the same form as the forward problem, so
 * between observations the state is stepped by the forward machinery in
 * reversed time, with finite jumps of the adjoint internal variables at
 * each observation. The two differ by the discretise/transpose commute
 * error, which vanishes as the run resolves its rheology: their
 * DISAGREEMENT is a resolution diagnostic for the forward calculation.
 *
 * Conventions: times are the forward driver's (increasing, SLE solver
 * units); measurements are keyed by node index into the time grid; duals
 * are non-dimensional SLE-unit grid densities pairing with the outputs by
 * the surface integral int . dS, the angular-velocity dual by the dot
 * product. The returned kernels pair with load perturbations the same
 * way; densities and rates are per unit SLE time. Frozen shorelines (the
 * forward driver's linear equation) throughout.
 */

#include <array>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "sl3dpp/Evolution.hpp"

namespace sl3dpp {

/**
 * @brief A recorded forward trajectory, for the viscosity kernels: the
 * internal states at every node or at sparse checkpoints, and the
 * per-node load and tide coefficients that let a reverse sweep rebuild
 * the gaps by exact backward steps (ViscoelasticOperator::StepBack).
 * Reconstruction amplifies roundoff by exp(h) per branch and node across
 * a gap, so the recording stride buys memory at the price of trailing
 * digits.
 */
struct ForwardTrajectory {
  std::vector<std::optional<ViscoelasticState>> states;
  /// Per node, the SLE-unit total-load and centrifugal coefficients the
  /// forward run recorded; may be empty when every state is stored.
  std::vector<std::vector<std::complex<double>>> loads;
  std::vector<std::vector<std::complex<double>>> tides;

  /// The trajectory a recording run's snapshots carry (a run with
  /// Options::record_stride set).
  [[nodiscard]] static ForwardTrajectory FromSnapshots(
      std::span<const SeaLevelEvolution::Snapshot> snapshots);
};

/// The duals of one observation node's outputs: the gains of
///   J += int g_xi xi dS + int g_u u dS + int g_phi phi dS + g_omega . omega
/// at that node. Empty vectors are absent duals.
struct Measurement {
  std::vector<double> seaLevelDual;
  std::vector<double> displacementDual;
  std::vector<double> potentialDual;
  std::array<double, 3> angularVelocityDual{0.0, 0.0, 0.0};
};

/**
 * @brief Sensitivity kernels of one objective with respect to the ice
 * history, in three views (docs/dev/adjoints.md §4; Yu et al. 2025 §3.7).
 *
 * DISCRETE view: `loadDuals[n]` is the dual of the direct ice-load grid
 * at time node n — delta J = sum_n int loadDuals[n] delta f_n dS, exact
 * on the time grid (to the commute error for the continuous method);
 * `thickness[n]` the dual of ice thickness there under the grounding
 * factor rho_i (1 - C), with the initial-thickness `baseline` folded
 * into node 0 (valid when the forward load was rho_i (1 - C)(I_n - I_0));
 * `initialSeaLevel` is identically zero under frozen shorelines.
 *
 * DISTRIBUTIONAL view: the kernel is a continuous `density` (per unit
 * time, converging under grid refinement) plus a delta at each
 * observation whose amplitude `instantaneous[i]` is the unrelaxed
 * fingerprint adjoint of that measurement, independent of the grid.
 *
 * QUELLED view (Backus): `rate[k]` pairs with the thickness RATE on
 * interval (times[k], times[k+1]):
 *   delta J = sum_k h_k int rate[k] (dI/dt)_k dS,
 * bounded with finite jumps at observations only; the initial-thickness
 * dual vanishes identically.
 */
struct IceKernels {
  std::vector<double> times;
  std::vector<std::vector<double>> loadDuals;
  std::vector<std::vector<double>> thickness;
  std::vector<double> baseline;
  std::vector<double> initialSeaLevel;
  std::map<std::size_t, std::vector<double>> instantaneous;
  std::vector<std::vector<double>> density;
  std::vector<std::vector<double>> rate;  ///< size times - 1

  /// dJ / d itau(r) per branch, nodal (element, gll) arrays over the
  /// material's full mesh, itau in MATERIAL units; empty unless the
  /// sweep was given the forward trajectory. Discrete: the exact
  /// sensitivity through the scheme-weight derivatives. Continuous: the
  /// multiplier integral int W_k m-dagger / itau (d - m) dt. With the
  /// internal variables at Gauss points both accumulate there and reach
  /// the nodal itau through the interpolation's transpose.
  std::vector<std::vector<double>> itauKernel;

  /// The viscosity kernels dJ / d eta(r) per branch from the itau
  /// kernels: itau = mu / eta gives d itau / d eta = -itau^2 / mu, so an
  /// elastic branch (itau = 0) has no viscosity sensitivity in this
  /// parameterisation. eta in material units.
  [[nodiscard]] std::vector<std::vector<double>> Viscosity(
      const Rheology& rheology) const;
};

enum class AdjointMethod { Continuous, Discrete };

namespace detail {
/// The grounded views and the distributional/quelled assembly shared by
/// the 1D and 3D sweeps: loadDuals complete (lumps folded in), density
/// the continuous part per unit time, lumps the delta amplitudes
/// (load-dual space, by node).
[[nodiscard]] IceKernels AssembleIceViews(
    std::span<const double> oceanFunction, double iceDensity,
    std::span<const double> times, std::vector<std::vector<double>> loadDuals,
    std::vector<std::vector<double>> density,
    std::map<std::size_t, std::vector<double>> lumps);
}  // namespace detail

/**
 * @brief The reverse sweeps over a coupled evolution's setup.
 *
 * Borrows the evolution (operator, sea level solver, surface state, unit
 * ratios, scheme and rotation options) for its lifetime; the sweeps leave
 * the evolution's own state untouched. The ice history enters the frozen
 * linear problem linearly, so no forward run is needed for the ice
 * kernels.
 */
class AdjointSolver {
 public:
  struct Options {
    AdjointMethod method = AdjointMethod::Continuous;
    /// The per-solve tolerance of the SLE (adjoint or forward) solves.
    double rtol = 1e-9;
    int anderson_memory = 5;
  };

  explicit AdjointSolver(const SeaLevelEvolution& evolution, Options options);
  explicit AdjointSolver(const SeaLevelEvolution& evolution)
      : AdjointSolver(evolution, Options{}) {}

  /// The ice-history kernels of the objective the measurements define,
  /// over the forward time grid (increasing, SLE units; at least two
  /// nodes). Measurement keys are node indices into `times`. With a
  /// recorded forward `trajectory` (one entry per node, from a run with
  /// record_stride under the same scheme and loads), the kernels also
  /// carry the inverse-relaxation-time sensitivity.
  [[nodiscard]] IceKernels IceHistoryKernels(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory = nullptr) const;

 private:
  [[nodiscard]] IceKernels DiscreteSweep(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory) const;
  [[nodiscard]] IceKernels ContinuousSweep(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory) const;
  /// The grounded views and the distributional/quelled assembly shared by
  /// the two sweeps (lumps: the delta amplitudes, by node).
  [[nodiscard]] IceKernels AssembleViews(
      std::span<const double> times,
      std::vector<std::vector<double>> loadDuals,
      std::vector<std::vector<double>> density,
      std::map<std::size_t, std::vector<double>> lumps) const;

  const SeaLevelEvolution* evolution_;
  Options options_;
};

}  // namespace sl3dpp
