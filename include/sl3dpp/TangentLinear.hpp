#pragma once

/**
 * @file TangentLinear.hpp
 * @brief The tangent-linear runs of the coupled evolution: the derivative
 * of a run's responses along a perturbation of the ice history, the
 * initial sea level and the inverse relaxation times, about a forward
 * trajectory. Corresponds to pyslfp's `tangent_linear_ice_history`; see
 * docs/adjoints.md.
 *
 * With the trajectory's ocean functions held the forward problem is linear
 * in its loads, so the derivative is the forward problem once more: the
 * shorelines prescribed from the trajectory, the direct load the
 * perturbation's,
 *
 *   delta f_n + rho_i (1 - C_n) delta I_n - rho_i (1 - C_0) delta I_0
 *             + rho_w (C_n - C_0) delta S_0,
 *
 * and, for a perturbation of the relaxation times, the drive of each
 * branch shifted by a known term of the trajectory. Holding the ocean
 * functions is exact in the continuum (the singular terms of shoreline
 * motion cancel, Yu et al. 2025); on the grid the map is piecewise affine
 * in the loads and this is its derivative on each piece. Under frozen
 * shorelines the initial sea level contributes nothing.
 *
 * `method` chooses how the relaxation-time perturbation enters:
 * Discrete differentiates the scheme's weights — the forward mode of the
 * time-stepped map, the exact transpose of AdjointSolver's discrete
 * sweep; Continuous discretises the continuous tangent-linear problem,
 * the drive q_k = (d - m_k) delta itau_k / itau_k entering each step like
 * the strain, the transpose of the continuous sweep. The two coincide in
 * the loads and differ in the relaxation times by the order of the scheme.
 *
 * A perturbation of the material's moduli (`TangentPerturbation::material`)
 * enters as forces on each step's own systems:
 * the variation of the effective system on the full base solution, the
 * memory force of the base memory with the branch moduli's variation,
 * and the start's unrelaxed response; the branch moduli follow as fixed
 * fractions of the unrelaxed modulus and the relaxation times are held.
 *
 * Conventions are the evolution's: times increasing in SLE units, grids in
 * the solver's units, the itau perturbation nodal (element, gll) per
 * branch over the material's mesh in MATERIAL units (the layout of
 * IceKernels::itauKernel, which pairs with it by the plain nodal sum).
 */

#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"

namespace sl3dpp {

/// A direction in the evolution's inputs. Empty members are absent.
struct TangentPerturbation {
  /// Per time node, a perturbation of the non-ocean load a frozen `Run`
  /// was driven by; refused about a migrating trajectory, which is driven
  /// by its thickness.
  std::vector<std::vector<double>> directLoad;
  /// Per time node, a perturbation of the ice thickness, grounded by the
  /// node's ocean function relative to the first node's.
  std::vector<std::vector<double>> iceThickness;
  /// A perturbation of the initial sea level, which enters through the
  /// shorelines that moved: nothing under frozen shorelines.
  std::vector<double> initialSeaLevel;
  /// Per branch, a perturbation of the inverse relaxation times, nodal
  /// over the material's mesh in its units; needs the recorded forward
  /// trajectory.
  std::vector<std::vector<double>> itau;
  /// A perturbation of the material's moduli and density: delta kappa
  /// and delta mu at the memory points and delta rho at the GLL nodes
  /// (PointMaterialDirection), the branch moduli following as fixed
  /// fractions of the unrelaxed modulus and the relaxation times held
  /// (compose with `itau` for a viscosity held instead); a density
  /// direction moves the gravity self-consistently and the forcings with
  /// it, the sea level engine's own gravity and radius held
  /// (SeaLevelParameters::body_from_table; docs/adjoints.md §16). Needs
  /// the recorded trajectory; the moduli are taken under any rotational
  /// feedback, a density direction without it only (the moments of
  /// inertia are not moved).
  PointMaterialDirection material;

  [[nodiscard]] bool Empty() const {
    return directLoad.empty() && iceThickness.empty() &&
           initialSeaLevel.empty() && itau.empty() && material.kappa.empty() &&
           material.mu.empty() && material.rho.empty();
  }
};

/// A pair of directions for a second-order run: the perturbations of the
/// inverse relaxation times along u and v, the second variation
/// itau_uv along both for a parameterisation that is not linear in them
/// (itau delta_u chi delta_v chi for the log viscosity; empty for a
/// linear one), each as TangentPerturbation::itau, and the perturbations
/// of the moduli along u and v (TangentPerturbation::material; the
/// density refused). Under the sharp flotation rule the loads and the
/// initial sea level enter no second derivative, so they are not taken
/// here: their part of u and v is in the recorded tangent runs.
struct SecondOrderDirection {
  std::vector<std::vector<double>> itauU;
  std::vector<std::vector<double>> itauV;
  std::vector<std::vector<double>> itauUV;
  PointMaterialDirection materialU;
  PointMaterialDirection materialV;
};

/**
 * @brief The tangent-linear run over a coupled evolution's setup.
 *
 * Borrows the evolution as AdjointSolver does and leaves its state
 * untouched. The returned snapshots carry the perturbed responses at
 * every node (sea level change, displacement, potential, angular
 * velocity) and, when recording, the tangent-linear internal state with
 * its strain and the node's load and centrifugal coefficients — what a
 * second-order run about this one reads.
 */
class TangentLinearSolver {
 public:
  struct Options {
    AdjointMethod method = AdjointMethod::Continuous;
    double rtol = 1e-9;
    int anderson_memory = 5;
    /// Keep the tangent-linear internal state (with its strain) and the
    /// load and centrifugal coefficients on every snapshot.
    bool record = false;
    /// Called at the start of each step, with the node it steps to.
    std::function<void(std::size_t node)> on_node{};
  };

  explicit TangentLinearSolver(const SeaLevelEvolution& evolution,
                               Options options);
  explicit TangentLinearSolver(const SeaLevelEvolution& evolution)
      : TangentLinearSolver(evolution, Options{}) {}

  /// The derivative of the run over `times` along `direction`. With an
  /// itau perturbation the recorded forward `trajectory` is required (one
  /// entry per node, from a run with record_stride, the gaps refilled by
  /// forward steps under the recorded loads). With `oceanFunctions`, one
  /// grid per node from a migrating run (OceanFunctionsOf its snapshots),
  /// the run linearises about that trajectory instead of the frozen
  /// background shorelines.
  [[nodiscard]] std::vector<SeaLevelEvolution::Snapshot> Run(
      std::span<const double> times, const TangentPerturbation& direction,
      const ForwardTrajectory* trajectory = nullptr,
      std::span<const std::vector<double>> oceanFunctions = {}) const;

  /// The second-order tangent-linear run: the second derivative of the
  /// run's responses along two directions u and v, about the forward
  /// trajectory and the recorded tangent-linear runs of u and v about it
  /// (`Run` with `record`). With the ocean functions held the map is
  /// affine in the loads and the initial sea level, so the second
  /// derivative is non-zero only through the relaxation times and the
  /// moduli: the forward problem with no direct load, the shorelines
  /// prescribed, and the drive of each branch the bilinear term of the
  /// first-order run taken with v's trajectory and u's itau perturbation,
  /// with u's trajectory and v's, with the forward trajectory and
  /// itau_uv, and, by the discrete method, the second derivatives of the
  /// scheme weights with the forward trajectory and both perturbations;
  /// in the moduli, the material forces of u's direction on v's tangent
  /// trajectory (its full solution and its decayed memory with v's itau
  /// drive, which carries the mixed moduli–relaxation-time term) and of
  /// v's on u's, the start's alike (docs/adjoints.md §15). Symmetric
  /// in u and v. pyslfp's `second_order_tangent_linear_ice_history` under
  /// the sharp flotation rule, extended to the moduli.
  [[nodiscard]] std::vector<SeaLevelEvolution::Snapshot> RunSecondOrder(
      std::span<const double> times,
      std::span<const SeaLevelEvolution::Snapshot> tangentU,
      std::span<const SeaLevelEvolution::Snapshot> tangentV,
      const SecondOrderDirection& direction,
      const ForwardTrajectory* trajectory = nullptr,
      std::span<const std::vector<double>> oceanFunctions = {}) const;

  /// The memory increment of a step from `prev` to `next` — two states of
  /// the trajectory with their strains — for a perturbation of the inverse
  /// relaxation times at the memory points, bilinear in the two: the
  /// forward mode of the scheme's weights (Discrete), or the continuous
  /// drive q_k = (d - m_k) delta itau_k / itau_k discretised by the scheme
  /// (Continuous). pyslfp's `_first_order_drive`.
  [[nodiscard]] static std::map<int, std::vector<double>> FirstOrderDrive(
      const ViscoelasticOperator& op, AdjointMethod method, Scheme scheme,
      double dt, const ViscoelasticState& prev, const ViscoelasticState& next,
      const std::vector<std::vector<double>>& ditauPoints);
 private:
  using Direct = std::function<std::vector<double>(std::size_t node)>;
  using Drive =
      std::function<std::map<int, std::vector<double>>(std::size_t node, double dt)>;

  /// The loop the runs share: the forward problem with the trajectory's
  /// ocean functions prescribed, the direct load given per node (empty
  /// where nothing drives it), a memory increment per step and a material
  /// forcing (a start response and a force per step).
  struct MaterialForcing;
  struct MaterialDrive;
  struct MaterialSecondDrive;
  [[nodiscard]] std::vector<SeaLevelEvolution::Snapshot> March(
      std::span<const double> times,
      std::span<const std::vector<double>> oceanFunctions,
      const Direct& directAt, const Drive& drive,
      MaterialForcing* material) const;

  /// The discrete method's term from the second derivatives of the
  /// scheme's weights, paired with the forward trajectory and the two
  /// perturbations of the inverse relaxation times at the memory points.
  /// pyslfp's `_second_weight_drive`.
  [[nodiscard]] static std::map<int, std::vector<double>> SecondWeightDrive(
      const ViscoelasticOperator& op, Scheme scheme, double dt,
      const ViscoelasticState& prev, const ViscoelasticState& next,
      const std::vector<std::vector<double>>& ditauU,
      const std::vector<std::vector<double>>& ditauV);

  /// The material-unit step sizes of the time grid, and the evolution's
  /// column callables of recorded load and centrifugal coefficients.
  [[nodiscard]] static std::vector<double> StepSizes(
      const SeaLevelEvolution& ev, std::span<const double> times);
  [[nodiscard]] static std::function<LoadCallable(
      const std::vector<std::complex<double>>&)>
  LoadColumnsOf(const SeaLevelEvolution& ev);
  [[nodiscard]] static std::function<LoadCallable(
      const std::vector<std::complex<double>>&)>
  TideColumnsOf(const SeaLevelEvolution& ev);

  /// The material drive of a step: the per-dt effective variation, the
  /// force -(dK_eff) x on the base step's full solution plus the memory
  /// force of the base decayed memory with the branch moduli's variation,
  /// and the effective table's Love-number derivatives (the sea level
  /// closure's response channels), cached per dt.
  [[nodiscard]] std::shared_ptr<MaterialDrive> MakeMaterialDrive(
      const PointMaterialDirection& d, const ForwardTrajectory& trajectory,
      std::span<const double> times) const;

  const SeaLevelEvolution* evolution_;
  Options options_;
};

}  // namespace sl3dpp
