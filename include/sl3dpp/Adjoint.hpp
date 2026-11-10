#pragma once

/**
 * @file Adjoint.hpp
 * @brief Ice-history sensitivity kernels of the coupled evolution, by the
 * continuous adjoint (the preferred path) or the exact discrete transpose
 * — one interface and result type for both, so every test and driver is
 * formulation-blind. Corresponds to pyslfp's `adjoint_ice_history` /
 * `continuous_adjoint_ice_history` (whose docstrings, with Yu et al. 2025,
 * are the authoritative statements); see docs/adjoints.md.
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
 * way; densities and rates are per unit SLE time. The sweeps linearise
 * about frozen shorelines (the forward driver's linear equation) or, given
 * the ocean function of every node of a migrating run, about that
 * trajectory: the adjoint stays linear with the time-reversed C(t) as its
 * coefficients (Yu et al. 2025, §3.3; the shoreline-motion singular terms
 * cancel), as pyslfp's `ocean_functions`.
 */

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <utility>
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

/// The ocean function of every node of a migrating run (RunMigrating),
/// for the sweeps' linearisation about it; refused for a frozen run's
/// snapshots, which carry none.
[[nodiscard]] std::vector<std::vector<double>> OceanFunctionsOf(
    std::span<const SeaLevelEvolution::Snapshot> snapshots);

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
 * history, in three views (docs/adjoints.md; Yu et al. 2025 §3.7).
 *
 * DISCRETE view: `loadDuals[n]` is the dual of the direct ice-load grid
 * at time node n — delta J = sum_n int loadDuals[n] delta f_n dS, exact
 * on the time grid (to the commute error for the continuous method);
 * `thickness[n]` the dual of ice thickness there under the grounding
 * factor rho_i (1 - C), with the initial-thickness `baseline` folded
 * into node 0 (valid when the forward load was rho_i (1 - C)(I_n - I_0));
 * about a migrating trajectory the grounding factor is the node's own
 * rho_i (1 - C_n), the baseline keeping the initial C. `initialSeaLevel`
 * is the dual of the initial sea level, rho_w sum_n (C_n - C_0)
 * loadDuals[n] (the initial sea level enters each node's load through the
 * shorelines that moved): identically zero under frozen shorelines.
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

  /// dJ / d kappa(r), dJ / d mu(r) and dJ / d rho(r), material units,
  /// pairing with a PointMaterialDirection by the plain sum: the moduli's
  /// at the memory points, (element, point) over the whole mesh, and the
  /// density's at the GLL nodes. Each entry of the moduli's is the
  /// kernel's value at its point times the point's quadrature weight
  /// (MemoryPoints::weight times the element's Jacobian), so dividing by
  /// the weight gives samples of the kernel as a function of radius; no
  /// point is shared between elements, so its jumps across the mesh's
  /// interfaces are kept. The transpose of
  /// TangentPerturbation::material (the branch moduli as fixed fractions
  /// of the unrelaxed modulus, the relaxation times held; the density
  /// with its gravity and the forcings, the sea level engine's gravity
  /// held; docs/adjoints.md). Empty unless Options::material_kernels
  /// asked for them (with a trajectory; Discrete exact, Continuous at the
  /// commute level).
  std::vector<double> kappaKernel;
  std::vector<double> muKernel;
  std::vector<double> rhoKernel;
};

enum class AdjointMethod { Continuous, Discrete };

/**
 * @brief The derivative of an objective's kernels along a direction v of
 * the model parameters: what the second-order adjoint returns, the
 * components of the Hessian's action on v. `thickness` and
 * `initialSeaLevel` are the derivatives of IceKernels' thickness and
 * initial-sea-level views (the same pairings; the thickness view's
 * baseline folded into node 0 as there), `itauKernel` that of the
 * inverse-relaxation-time kernels (per branch, nodal, material units).
 * pyslfp's `KernelDerivatives`.
 */
struct KernelDerivatives {
  std::vector<double> times;
  std::vector<std::vector<double>> thickness;
  std::vector<double> initialSeaLevel;
  std::vector<std::vector<double>> itauKernel;
  /// The derivatives along v of the material kernels (IceKernels::
  /// kappaKernel, muKernel, rhoKernel), in their layout: the material
  /// rows of the Hessian's action on v. Empty unless
  /// Options::material_kernels.
  std::vector<double> kappaKernel;
  std::vector<double> muKernel;
  std::vector<double> rhoKernel;
};

namespace detail {
/// The grounded views and the distributional/quelled assembly shared by
/// the 1D and 3D sweeps: loadDuals complete (lumps folded in), density
/// the continuous part per unit time, lumps the delta amplitudes
/// (load-dual space, by node).
/// With `oceanFunctions` (one per node, about a migrating trajectory) the
/// grounding factors are per node and the initial sea level dual is
/// formed, with the water density `waterDensity`.
[[nodiscard]] IceKernels AssembleIceViews(
    std::span<const double> oceanFunction, double iceDensity,
    std::span<const double> times, std::vector<std::vector<double>> loadDuals,
    std::vector<std::vector<double>> density,
    std::map<std::size_t, std::vector<double>> lumps,
    std::span<const std::vector<double>> oceanFunctions = {},
    double waterDensity = 0.0);
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
    /// Also form the material kernels (IceKernels::kappaKernel, muKernel,
    /// rhoKernel). Discrete: the pairing of each step's dual solution with
    /// the material derivative of the step's force, the exact transpose
    /// of the material tangent. Continuous: the time quadrature of the
    /// continuous adjoint displacement's pairing with the forward
    /// solution and memory, plus the lumped pairings at the observations
    /// — the commute error of the viscosity kernels, falling under
    /// refinement. Both need the trajectory and no rotational feedback,
    /// and cost one unrelaxed solve per node (two for the continuous
    /// form).
    bool material_kernels = false;
    /// Called at the start of each step of the reversed sweep, with the
    /// node it steps back from (the last node first, down to 1): a hook for
    /// progress reports, since a sweep can take as long as the forward run.
    /// Empty by default.
    std::function<void(std::size_t node)> on_node{};
  };

  explicit AdjointSolver(const SeaLevelEvolution& evolution, Options options);
  explicit AdjointSolver(const SeaLevelEvolution& evolution)
      : AdjointSolver(evolution, Options{}) {}

  /// The ice-history kernels of the objective the measurements define,
  /// over the forward time grid (increasing, SLE units; at least two
  /// nodes). Measurement keys are node indices into `times`. With a
  /// recorded forward `trajectory` (one entry per node, from a run with
  /// record_stride under the same scheme and loads), the kernels also
  /// carry the inverse-relaxation-time sensitivity. With
  /// `oceanFunctions`, one grid per node from a migrating run
  /// (OceanFunctionsOf its snapshots), the kernels linearise about that
  /// trajectory instead of the frozen background shorelines, and the
  /// initial sea level kernel is formed; a recorded trajectory must then
  /// come from the same migrating run.
  [[nodiscard]] IceKernels IceHistoryKernels(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory = nullptr,
      std::span<const std::vector<double>> oceanFunctions = {}) const;

  /**
   * @brief The second-order adjoint: the kernels of the objective and
   * their derivative along a direction v, the action of the objective's
   * Hessian on v (pyslfp's `second_order_adjoint_ice_history`, Yu et al.
   * 2025 §4). The sweep that gives the kernels is itself a function of
   * the parameters, through the forward trajectory and the relaxation
   * times; differentiating it along v gives a second sweep of the same
   * form, forced not by the measurement but by the first sweep's state
   * and v's perturbation of the relaxation times, with v's tangent-linear
   * trajectory entering where the forward one enters the kernels. By the
   * discrete method this is the exact derivative of the exact gradient,
   * so the Hessian it gives is symmetric to rounding; by the continuous
   * method it discretises the second-order adjoint equations with the
   * ocean functions held (the terms of shoreline motion omitted).
   *
   * @param tangent v's tangent-linear run about the forward run, recorded
   * at every node (TangentLinearSolver::Options::record), by the same
   * method as this sweep.
   * @param itau v's perturbation of the inverse relaxation times, nodal
   * per branch over the material's mesh (TangentPerturbation::itau);
   * empty for none.
   * @param trajectory The recorded forward trajectory; required here.
   * @param oceanFunctions As for IceHistoryKernels, from the same
   * migrating run as the tangent-linear run.
   * @param material v's perturbation of the moduli
   * (TangentPerturbation::material, the density refused), which the
   * tangent run must have been driven by too; empty for none. A
   * material direction, like Options::material_kernels (which adds the
   * material kernels' derivatives along v, for any v), needs the memory
   * points at the GLL nodes and no rotational feedback. Discrete: the
   * derivative of the effective systems acts on the first sweep's
   * solutions as a source of the derivative duals, and the memory force's
   * variation on the first sweep's strains; the material kernels'
   * derivatives pair the derivative duals with the forward trajectory and
   * the first duals with v's tangent trajectory (docs/adjoints.md
   * §15). Not yet in the continuous sweep.
   */
  [[nodiscard]] std::pair<IceKernels, KernelDerivatives> SecondOrderKernels(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      std::span<const SeaLevelEvolution::Snapshot> tangent,
      const std::vector<std::vector<double>>& itau,
      const ForwardTrajectory& trajectory,
      std::span<const std::vector<double>> oceanFunctions = {},
      const PointMaterialDirection& material = {}) const;

 private:
  [[nodiscard]] std::pair<IceKernels, KernelDerivatives> DiscreteSecondOrder(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      std::span<const SeaLevelEvolution::Snapshot> tangent,
      const std::vector<std::vector<double>>& ditauPoints,
      const ForwardTrajectory& trajectory,
      std::span<const std::vector<double>> oceanFunctions,
      const PointMaterialDirection& material) const;
  [[nodiscard]] std::pair<IceKernels, KernelDerivatives> ContinuousSecondOrder(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      std::span<const SeaLevelEvolution::Snapshot> tangent,
      const std::vector<std::vector<double>>& ditauPoints,
      const ForwardTrajectory& trajectory,
      std::span<const std::vector<double>> oceanFunctions,
      const PointMaterialDirection& material) const;
  [[nodiscard]] IceKernels DiscreteSweep(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory,
      std::span<const std::vector<double>> oceanFunctions) const;
  [[nodiscard]] IceKernels ContinuousSweep(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory* trajectory,
      std::span<const std::vector<double>> oceanFunctions) const;
  /// The material pieces the second-order sweeps share (the variations
  /// of a direction, its forces, the tangent solution's recovery):
  /// detail/AdjointInternals.hpp.
  struct MaterialTools;
  /// The grounded views and the distributional/quelled assembly shared by
  /// the two sweeps (lumps: the delta amplitudes, by node).
  [[nodiscard]] IceKernels AssembleViews(
      std::span<const double> times, std::vector<std::vector<double>> loadDuals,
      std::vector<std::vector<double>> density,
      std::map<std::size_t, std::vector<double>> lumps,
      std::span<const std::vector<double>> oceanFunctions) const;

  const SeaLevelEvolution* evolution_;
  Options options_;
};

}  // namespace sl3dpp
