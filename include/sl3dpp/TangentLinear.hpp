#pragma once

/**
 * @file TangentLinear.hpp
 * @brief The tangent-linear runs of the coupled evolution: the derivative
 * of a run's responses along a perturbation of the ice history, the
 * initial sea level and the inverse relaxation times, about a forward
 * trajectory. A port of pyslfp's `tangent_linear_ice_history`; the design
 * record is docs/dev/adjoints.md (tangent-linear section).
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
 * Conventions are the evolution's: times increasing in SLE units, grids in
 * the solver's units, the itau perturbation nodal (element, gll) per
 * branch over the material's mesh in MATERIAL units (the layout of
 * IceKernels::itauKernel, which pairs with it by the plain nodal sum).
 */

#include <functional>
#include <map>
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

  [[nodiscard]] bool Empty() const {
    return directLoad.empty() && iceThickness.empty() &&
           initialSeaLevel.empty() && itau.empty();
  }
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

 private:
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

  const SeaLevelEvolution* evolution_;
  Options options_;
};

}  // namespace sl3dpp
