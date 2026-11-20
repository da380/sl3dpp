#pragma once

/**
 * @file TangentLinear.hpp
 * @brief The tangent-linear runs of the coupled evolution: the derivative
 * of a run's responses along a perturbation of the ice history, the
 * initial sea level and the inverse relaxation times, about a forward
 * trajectory; see docs/adjoints.md.
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
 * Kernels::itauKernel, a density per unit volume, which pairs with it as
 * VolumePairing(mesh, kernel, perturbation)).
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

namespace detail {
struct StageRef;
}  // namespace detail

/// A direction in the evolution's inputs. Empty members are absent.
///
/// The members given per time node are the direction's values at the
/// nodes, which are all a run reads of its forcing. Under a multi-stage
/// scheme (Sdirk2) a step's inner stage takes 1 - c of the forcing at the
/// step's start node and c of it at the end node (a repeated node's two
/// sides are nodes of their own), and the tangent run takes the same
/// combination of the direction: it is the derivative of the forward run
/// along the direction, and the exact transpose of the discrete adjoint
/// sweep, whose node kernels pair with these values.
struct TangentPerturbation {
  /// Per time node, a perturbation of the direct load inside the water
  /// budget the run was driven by, frozen or migrating.
  std::vector<std::vector<double>> directLoad;
  /// Per time node, perturbations of the run's direct load outside the
  /// water budget and of the height of a layer on the solid surface
  /// (Evolution::Forcing), frozen or migrating: about a migrating
  /// trajectory the layer's place in the flotation is held, as the
  /// shorelines are.
  std::vector<std::vector<double>> directLoadOutsideWaterBudget;
  std::vector<std::vector<double>> solidSurfaceChange;
  /// Per time node, a perturbation of the prescribed tidal potential the
  /// run was driven by, its coefficients on the surface (degree 2 and
  /// up, kLowestTideDegree: a part below is refused), frozen or migrating: about a migrating trajectory the
  /// shorelines are held, the tide's potential moving the sea level
  /// over each node's ocean function.
  std::vector<std::vector<std::complex<double>>> tide;
  /// Per time node, a perturbation of the interior sources acting there
  /// (Evolution::Forcing::sources and stressGluts: each node's point
  /// sources and stress gluts, solver units), frozen or migrating. The
  /// run is linear in them, so no trajectory is needed; the snapshots
  /// carry the toroidal potential.
  std::vector<std::vector<PointSource>> sources;
  std::vector<std::vector<StressGlut>> stressGluts;
  /// Per time node, a perturbation of the ice thickness, grounded by the
  /// node's ocean function relative to the first node's: the load at node
  /// n is rho_i (1 - C_n) dI_n - rho_i (1 - C) dI_0, C the background
  /// state's ocean function, so the background's ice moves with the first
  /// node's. About a migrating run whose start is solved under an ice
  /// change (the ice at the first node other than the background's, the
  /// background held), the derivative along the history alone is this
  /// one plus the run along the direct load rho_i (1 - C) dI_0 at every
  /// node; a direction zero at the first node means the same either way.
  std::vector<std::vector<double>> iceThickness;
  /// A perturbation of the initial sea level, which enters through the
  /// shorelines that moved: nothing under frozen shorelines.
  std::vector<double> initialSeaLevel;
  /// Per branch, a perturbation of the inverse relaxation times, nodal
  /// over the material's mesh in its units; needs the recorded forward
  /// trajectory.
  std::vector<std::vector<double>> itau;
  /// A perturbation of the material's moduli: delta kappa and delta mu
  /// at the memory points (PointMaterialDirection), the branch moduli
  /// following as fixed fractions of the unrelaxed modulus and the
  /// relaxation times held (compose with `itau` for a viscosity held
  /// instead). Needs the recorded trajectory; taken under any rotational
  /// feedback.
  PointMaterialDirection material;

  [[nodiscard]] bool Empty() const {
    return directLoad.empty() && directLoadOutsideWaterBudget.empty() &&
           solidSurfaceChange.empty() && tide.empty() && sources.empty() &&
           stressGluts.empty() && iceThickness.empty() &&
           initialSeaLevel.empty() && itau.empty() && material.Empty();
  }
};

/// A pair of directions for a second-order run: the perturbations of the
/// inverse relaxation times along u and v, the second variation
/// itau_uv along both for a parameterisation that is not linear in them
/// (itau delta_u chi delta_v chi for the log viscosity; empty for a
/// linear one), each as TangentPerturbation::itau, and the perturbations
/// of the moduli along u and v (TangentPerturbation::material). Under
/// the sharp flotation rule the loads and the initial sea level enter no
/// second derivative, so they are not taken here: their part of u and v
/// is in the recorded tangent runs.
struct SecondOrderDirection {
  std::vector<std::vector<double>> itauU;
  std::vector<std::vector<double>> itauV;
  std::vector<std::vector<double>> itauUV;
  PointMaterialDirection materialU;
  PointMaterialDirection materialV;
};

/// The toroidal systems' drive about a source-driven trajectory, for a
/// perturbation of the relaxation times or the moduli.
struct TangentToroidalDrive {
  /// A stage's memory increment and force (the step's one stage under a
  /// one-stage scheme).
  std::function<std::pair<ToroidalViscoelastic::Fields,
                          ToroidalViscoelastic::Fields>(
      const ToroidalViscoelastic& toroidal, const detail::StageRef& stage)>
      step{};
  /// The start's own response (the moduli's).
  std::function<ToroidalViscoelastic::Fields(
      const ToroidalViscoelastic& toroidal)>
      start{};
};

/**
 * @brief The tangent-linear run over a coupled evolution's setup.
 *
 * Borrows the evolution as AdjointSolver does and leaves its state
 * untouched. The returned snapshots carry the perturbed responses at
 * every node (sea level change, displacement, potential, the tangential
 * displacement's potential, angular velocity) and, when recording, the tangent-linear internal state with
 * its strain and the node's load and centrifugal coefficients — what a
 * second-order run about this one reads.
 */
class TangentLinearSolver {
 public:
  struct Options {
    /// Discrete (the default, as AdjointSolver's): the forward mode of
    /// the stepped map; or Continuous (FirstOrderDrive). A second-order
    /// sweep about the run takes the same method as the run. About an
    /// evolution under Sdirk2 the run steps the tangent state by the
    /// scheme's two stages, each driven about the forward stage's states at
    /// the stage's length: by the backward Euler weight's derivative
    /// (Discrete) or by backward Euler's continuous drive (Continuous),
    /// which are one drive, so that the two methods give one run, to
    /// rounding (docs/adjoints.md §1a). Along the forcing or the moduli
    /// (which hold the relaxation times, and so the step's weights) the
    /// two are one run under every scheme.
    AdjointMethod method = AdjointMethod::Discrete;
    double rtol = 1e-9;
    int anderson_memory = 5;
    /// Keep the tangent-linear internal state (with its strain) and the
    /// load and centrifugal coefficients on every snapshot; under a
    /// multi-stage scheme, each inner stage's too (Evolution::Snapshot::
    /// stages, with the stage's state).
    bool record = false;
    /// RunSecondOrder about an evolution whose background state is marked
    /// calibrated under migrating shorelines (Evolution::Calibrated) holds
    /// the initial sea level: it gives the second derivative with the
    /// calibrated initial sea level held, where the calibrated run's own
    /// (the initial sea level calibrated again along both directions) is
    /// not built. It is refused there unless this states that the held one
    /// is wanted. Run holds the initial sea level about any run, and
    /// ComputeCalibratedTangent forms the calibrated run's first derivative
    /// from it (InitialSeaLevel.hpp).
    bool initial_sea_level_held = false;
    /// Called at the start of each step, with the node it steps to.
    std::function<void(std::size_t node)> on_node{};
  };

  /// Refused for an evolution without gravity (Evolution.hpp).
  explicit TangentLinearSolver(const Evolution& evolution,
                               Options options);
  explicit TangentLinearSolver(const Evolution& evolution)
      : TangentLinearSolver(evolution, Options{}) {}

  /// The derivative of the run over `times` along `direction`. With an
  /// itau perturbation the recorded forward `trajectory` is required (one
  /// entry per node, from a run with record_stride, the gaps refilled by
  /// forward steps under the recorded loads). With `oceanFunctions` from
  /// a migrating run (OceanFunctions::Of its snapshots), the run
  /// linearises about that trajectory instead of the frozen background
  /// shorelines; about a migrating run under Sdirk2 each step's first stage
  /// is linearised about the stage's own ocean function
  /// (OceanFunctions::stages), and ocean functions without them are
  /// refused.
  [[nodiscard]] std::vector<Evolution::Snapshot> Run(
      std::span<const double> times, const TangentPerturbation& direction,
      const ForwardTrajectory* trajectory = nullptr,
      const OceanFunctions& oceanFunctions = {}) const;

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
  /// in u and v. The two runs are by this solver's method
  /// (Options::method), and a run of the other is refused; about a
  /// calibrated migrating run the initial sea level is held, and refused
  /// unless Options::initial_sea_level_held.
  [[nodiscard]] std::vector<Evolution::Snapshot> RunSecondOrder(
      std::span<const double> times,
      std::span<const Evolution::Snapshot> tangentU,
      std::span<const Evolution::Snapshot> tangentV,
      const SecondOrderDirection& direction,
      const ForwardTrajectory* trajectory = nullptr,
      const OceanFunctions& oceanFunctions = {}) const;

  /// The memory increment of a step from `prev` to `next` — two states of
  /// the trajectory with their strains — for a perturbation of the inverse
  /// relaxation times at the memory points, bilinear in the two: the
  /// forward mode of the scheme's weights (Discrete), or the continuous
  /// drive q_k = (d - m_k) delta itau_k / itau_k discretised by the scheme
  /// (Continuous). `scheme` is a one-stage scheme (a stage's family under
  /// a multi-stage one, `dt` the stage's length); under backward Euler the
  /// two drives are one and `prev`'s strain is not read.
  [[nodiscard]] static std::map<int, std::vector<double>> FirstOrderDrive(
      const ViscoelasticOperator& op, AdjointMethod method, Scheme scheme,
      double dt, const ViscoelasticState& prev, const ViscoelasticState& next,
      const std::vector<std::vector<double>>& ditauPoints);
 private:
  /// A term of the direction at a stage of a step (detail::StageRef; the
  /// first node is a stage of its own, ending there): a direct load on the
  /// grid, a tide's coefficients.
  using Direct =
      std::function<std::vector<double>(const detail::StageRef& stage)>;
  using Tide = std::function<std::vector<std::complex<double>>(
      const detail::StageRef& stage)>;
  using ToroidalDrive = TangentToroidalDrive;
  /// The interior sources perturbed at a node.
  using Sources = std::function<
      std::pair<std::vector<PointSource>, std::vector<StressGlut>>(
          std::size_t node)>;
  /// A stage's memory increment.
  using Drive = std::function<std::map<int, std::vector<double>>(
      const detail::StageRef& stage)>;

  /// The loop the runs share: the forward problem with the trajectory's
  /// ocean functions prescribed, the direct load given per node (empty
  /// where nothing drives it), a memory increment per step and a material
  /// forcing (a start response and a force per step).
  struct MaterialForcing;
  struct MaterialDrive;
  struct MaterialSecondDrive;
  [[nodiscard]] std::vector<Evolution::Snapshot> March(
      std::span<const double> times, const OceanFunctions& oceanFunctions,
      const Direct& directAt, const Drive& drive, MaterialForcing* material,
      const Tide& tideAt = {}, const Sources& sourcesAt = {},
      const ToroidalDrive& toroidalDrive = {}, const Direct& outsideAt = {},
      const Direct& layerAt = {}) const;

  /// The discrete method's term from the second derivatives of the
  /// scheme's weights, paired with the forward trajectory and the two
  /// perturbations of the inverse relaxation times at the memory points.
  [[nodiscard]] static std::map<int, std::vector<double>> SecondWeightDrive(
      const ViscoelasticOperator& op, Scheme scheme, double dt,
      const ViscoelasticState& prev, const ViscoelasticState& next,
      const std::vector<std::vector<double>>& ditauU,
      const std::vector<std::vector<double>>& ditauV);

  /// The material-unit step sizes of the time grid, and the evolution's
  /// column callables of recorded load and centrifugal coefficients.
  [[nodiscard]] static std::vector<double> StepSizes(
      const Evolution& ev, std::span<const double> times);
  [[nodiscard]] static std::function<LoadCallable(
      const std::vector<std::complex<double>>&)>
  LoadColumnsOf(const Evolution& ev);
  [[nodiscard]] static std::function<LoadCallable(
      const std::vector<std::complex<double>>&)>
  TideColumnsOf(const Evolution& ev);

  /// The material drive of a step: the per-dt effective variation, the
  /// force -(dK_eff) x on the base step's full solution plus the memory
  /// force of the base decayed memory with the branch moduli's variation,
  /// and the effective table's Love-number derivatives (the sea level
  /// closure's response channels), cached per dt.
  [[nodiscard]] std::shared_ptr<MaterialDrive> MakeMaterialDrive(
      const PointMaterialDirection& d, const ForwardTrajectory& trajectory,
      std::span<const double> times) const;

  const Evolution* evolution_;
  Options options_;
};

}  // namespace sl3dpp
