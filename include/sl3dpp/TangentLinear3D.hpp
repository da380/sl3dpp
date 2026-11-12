#pragma once

/**
 * @file TangentLinear3D.hpp
 * @brief The tangent-linear runs of the coupled 3D (lateral-viscosity)
 * evolution: the derivative of a run's responses along a perturbation of
 * the ice history, the initial sea level and the laterally varying
 * inverse relaxation times, about a recorded trajectory, and the
 * second-order run along two such directions. The 3D form of
 * TangentLinear.hpp (pyslfp has no 3D counterpart); see
 * docs/adjoints.md.
 *
 * With the ocean functions held the coupled step is linear in its loads
 * and its memory, so the tangent-linear step is the forward step once
 * more — the reference effective systems, the within-step lateral fixed
 * point and the sea level closure unchanged — with the shorelines
 * prescribed, the direct load the perturbation's, and each opened step's
 * decayed memory shifted by a drive bilinear in the trajectory and the
 * perturbation of itau(x), pointwise on the memory grid with the
 * pointwise scheme weights: the h-derivatives of the weights (Discrete,
 * the forward mode of the time-stepped map) or the continuous drive
 * q_k = (d - m_k) delta itau_k / itau_k discretised by the scheme
 * (Continuous, the transpose of the continuous 3D sweep). The lateral
 * weight of the fixed point is the trajectory's, so the drive enters the
 * memory force and the end state alike, as in 1D.
 *
 * A lateral perturbation of the moduli (`TangentPerturbation3D::material`)
 * enters each opened step as forces through
 * the grid pipeline (OpenStep::AddForce): the effective system's
 * variation on the full base solution with the step's own lateral
 * weights, the memory force of the base step's decayed memory with the
 * branch moduli's variation, and the bulk term through the scalar
 * pipeline; the start's unrelaxed response alike.
 *
 * Conventions follow Evolution3D: times increasing in SLE units,
 * grids in the solver's units; an itau perturbation is per branch over
 * the coupling's grid values ([FieldOffset(node) + x], the layout of
 * Kernels3D::itauKernel3D) in MATERIAL units, and the log-viscosity
 * perturbation delta-delta(x) maps to it by delta itau_k = -itau_k(x)
 * delta-delta(x) (ItauOfLogViscosity).
 */

#include <complex>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution3D.hpp"

namespace sl3dpp {

/// A direction in the 3D evolution's inputs. Empty members are absent;
/// `itau` and `logViscosity` may both be given (they add). The material
/// direction is MaterialDirection3D (Adjoint3D.hpp).
struct TangentPerturbation3D {
  std::vector<std::vector<double>> directLoad;    ///< per node, frozen runs
  /// Per node, perturbations of a frozen run's direct load outside the
  /// water budget and of the height of a layer on the solid surface
  /// (Evolution3D::Forcing).
  std::vector<std::vector<double>> directLoadOutsideWaterBudget;
  std::vector<std::vector<double>> solidSurfaceChange;
  /// Per node, a perturbation of the prescribed tidal potential of a
  /// frozen run, its coefficients on the surface (no degree-0 part).
  std::vector<std::vector<std::complex<double>>> tide;
  /// Per node, a perturbation of the interior sources acting there
  /// (Evolution3D::Forcing::sources and stressGluts, solver units), about
  /// a frozen run; the run is linear in them.
  std::vector<std::vector<PointSource>> sources;
  std::vector<std::vector<StressGlut>> stressGluts;
  std::vector<std::vector<double>> iceThickness;  ///< per node
  std::vector<double> initialSeaLevel;            ///< migrating runs
  /// Per branch over the grid values, material units.
  std::vector<std::vector<double>> itau;
  /// Over the grid values: delta itau_k = -itau_k(x) logViscosity(x).
  std::vector<double> logViscosity;
  /// The moduli, over the grid values of the memory nodes
  /// (docs/adjoints.md); needs the recorded trajectory. Distributed like
  /// the run.
  MaterialDirection3D material;

  [[nodiscard]] bool Empty() const {
    return directLoad.empty() && directLoadOutsideWaterBudget.empty() &&
           solidSurfaceChange.empty() && tide.empty() && sources.empty() &&
           stressGluts.empty() && iceThickness.empty() &&
           initialSeaLevel.empty() && itau.empty() && logViscosity.empty() &&
           material.Empty();
  }
};

/// Two directions for a second-order run: the itau perturbations along
/// u and v over the grid values, the second variation itau_uv for a
/// parameterisation that is not linear in itau (itau(x) delta_u delta_v
/// for the log viscosity, SecondVariationOfLogViscosity; empty for a
/// linear one), and the perturbations of the moduli along u and v
/// (TangentPerturbation3D::material).
struct SecondOrderDirection3D {
  std::vector<std::vector<double>> itauU;
  std::vector<std::vector<double>> itauV;
  std::vector<std::vector<double>> itauUV;
  MaterialDirection3D materialU;
  MaterialDirection3D materialV;
};

class TangentLinearSolver3D {
 public:
  struct Options {
    AdjointMethod method = AdjointMethod::Continuous;
    /// The SLE's own tolerance per pass, and the combined fixed point's
    /// controls (as Evolution3D::Options).
    double rtol = 1e-9;
    int anderson_memory = 5;
    int max_coupled_iterations = 60;
    double coupled_rtol = 1e-9;
    double anderson_rate = 0.5;
    /// Keep the tangent-linear state (with its strain) and the load and
    /// centrifugal coefficients on every snapshot: what a second-order
    /// run reads.
    bool record = false;
    std::function<void(std::size_t node)> on_node{};
  };

  explicit TangentLinearSolver3D(const Evolution3D& evolution,
                                 Options options);
  explicit TangentLinearSolver3D(const Evolution3D& evolution)
      : TangentLinearSolver3D(evolution, Options{}) {}

  /// The derivative of the run over `times` along `direction`. With an
  /// itau or log-viscosity perturbation the recorded forward `trajectory`
  /// is required (the states between kept nodes rebuilt by forward steps
  /// under the recorded loads, with the run's own iteration). With
  /// `oceanFunctions` from a migrating run the run linearises about that
  /// trajectory. Distributed over radius like the evolution, collective.
  [[nodiscard]] std::vector<Evolution3D::Snapshot> Run(
      std::span<const double> times, const TangentPerturbation3D& direction,
      const ForwardTrajectory3D* trajectory = nullptr,
      std::span<const std::vector<double>> oceanFunctions = {}) const;

  /// The second-order run along u and v, about the forward trajectory and
  /// the recorded tangent-linear runs of u and v (`Run` with `record`):
  /// the forward problem with no direct load, the shorelines prescribed,
  /// and the drive the first-order drive taken with v's trajectory and u's
  /// itau perturbation, with u's trajectory and v's, with the forward
  /// trajectory and itau_uv, and, by the discrete method, the weights'
  /// second derivatives with the forward trajectory and both; in the
  /// moduli, the material forces of u's direction on v's tangent
  /// trajectory (its full solution and its decayed memory with v's itau
  /// drive) and of v's on u's, the start's alike (docs/adjoints.md
  /// §15). Symmetric in u and v; the 3D form of
  /// TangentLinearSolver::RunSecondOrder.
  [[nodiscard]] std::vector<Evolution3D::Snapshot> RunSecondOrder(
      std::span<const double> times,
      std::span<const Evolution3D::Snapshot> tangentU,
      std::span<const Evolution3D::Snapshot> tangentV,
      const SecondOrderDirection3D& direction,
      const ForwardTrajectory3D* trajectory = nullptr,
      std::span<const std::vector<double>> oceanFunctions = {}) const;

  /// The itau perturbation of a log-viscosity perturbation over the grid
  /// values, per branch: -itau_k(x) field(x) (eta = eta_ref exp(delta)).
  [[nodiscard]] static std::vector<std::vector<double>> ItauOfLogViscosity(
      const Viscoelastic3D& op, std::span<const double> field);
  /// The second variation of itau along two log-viscosity perturbations:
  /// itau_k(x) u(x) v(x).
  [[nodiscard]] static std::vector<std::vector<double>>
  SecondVariationOfLogViscosity(const Viscoelastic3D& op,
                                std::span<const double> u,
                                std::span<const double> v);

  /// The memory increment of a step from `prev` to `next` — two states of
  /// a trajectory with their strains — for a perturbation of itau over
  /// the grid values: the 1D FirstOrderDrive pointwise.
  [[nodiscard]] static std::vector<GridCoupling::TensorField> FirstOrderDrive(
      const Viscoelastic3D& op, AdjointMethod method, Scheme scheme, double dt,
      const Viscoelastic3D::State& prev, const Viscoelastic3D::State& next,
      const std::vector<std::vector<double>>& ditau);

 private:
  using Direct = std::function<std::vector<double>(std::size_t node)>;
  using Tide =
      std::function<std::vector<std::complex<double>>(std::size_t node)>;
  /// The interior sources perturbed at a node.
  using Sources = std::function<
      std::pair<std::vector<PointSource>, std::vector<StressGlut>>(
          std::size_t node)>;
  using Drive = std::function<std::vector<GridCoupling::TensorField>(
      std::size_t node, double dt)>;

  /// The material forcing of a march (TangentLinear3D.cpp): a start
  /// response and a force per step, from a direction (MaterialDrive3D:
  /// the three forces of a material direction, from the forward cursor)
  /// or from a pair of directions and tangent runs (MaterialSecondDrive3D).
  struct MaterialForcing3D;
  struct MaterialDrive3D;
  struct MaterialSecondDrive3D;
  class ForwardCursor3D;
  [[nodiscard]] std::vector<Evolution3D::Snapshot> March(
      std::span<const double> times,
      std::span<const std::vector<double>> oceanFunctions,
      const Direct& directAt, const Drive& drive,
      const MaterialForcing3D* material, const Tide& tideAt = {},
      const Sources& sourcesAt = {}, const Direct& outsideAt = {},
      const Direct& layerAt = {}) const;
  /// A material drive after the checks (the fields on the grid values).
  [[nodiscard]] std::shared_ptr<MaterialDrive3D> MakeMaterialDrive3D(
      const MaterialDirection3D& direction,
      const ForwardTrajectory3D* trajectory, ForwardCursor3D* cursor) const;

  [[nodiscard]] static std::vector<GridCoupling::TensorField> SecondWeightDrive(
      const Viscoelastic3D& op, Scheme scheme, double dt,
      const Viscoelastic3D::State& prev, const Viscoelastic3D::State& next,
      const std::vector<std::vector<double>>& ditauU,
      const std::vector<std::vector<double>>& ditauV);

  /// The material-unit step sizes, the forward run's fixed-point
  /// controls, and the evolution's column callables of recorded
  /// coefficients: what the forward cursor rebuilds states with.
  [[nodiscard]] std::vector<double> StepSizes(std::span<const double> times) const;
  [[nodiscard]] Viscoelastic3D::IterationControl Control() const;
  [[nodiscard]] std::function<Viscoelastic3D::LoadCallable3D(
      const std::vector<std::complex<double>>&)>
  LoadColumnsOf() const;
  [[nodiscard]] std::function<Viscoelastic3D::LoadCallable3D(
      const std::vector<std::complex<double>>&)>
  TideColumnsOf() const;

  const Evolution3D* evolution_;
  Options options_;
};

}  // namespace sl3dpp
