#pragma once

/**
 * @file ToroidalViscoelastic.hpp
 * @brief The toroidal partner of ViscoelasticOperator: the radial model's
 * toroidal systems stepped through time under forces.
 *
 * A radial model's toroidal motion is decoupled from its spheroidal
 * motion, from gravity and from the sea level: each degree from 1 up is
 * the scalar W(r) of the displacement -W r x grad_1 Y_lm, forced by
 * whatever acts on it (a stress glut; a toroidal traction). Its
 * deviatoric strain has two driving scalars at each memory point, the
 * toroidal twins of the spheroidal M and S,
 *
 *   M = W,   S = r dW/dr - W,
 *
 * with the same force weights k^2 (k^2 - 2) and k^2, k^2 = l (l + 1), and
 * each branch's internal variables follow them by the same recurrence as
 * the spheroidal ones (Viscoelastic.hpp): the memory decays over the
 * step, the effective system (the operator's own flavour of material and
 * point shear) is solved under the memory's force and the step's own,
 * and the end strain closes the update. The schemes, the memory points
 * and the truncation are the operator's.
 *
 * Arrays follow ViscoelasticOperator's layouts: forces and solutions are
 * (dof, column) column-major over a degree's toroidal dofs, the columns
 * the 2 (l + 1) real ones of a real field (the real block of orders
 * 0..l, then the imaginary); memory and strain are (branch, scalar,
 * element, point, column) over the elements of the degree's system.
 * Everything is in the material's units.
 */

#include <map>
#include <memory>
#include <span>
#include <vector>

#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp {

/// The forces of interior sources on a radial model's degree systems:
/// per degree, (dof, column) over the 2 (l + 1) real columns of a real
/// field (the real block of orders 0..l, then the imaginary), material
/// units.
struct RadialSourceForces {
  std::map<int, std::vector<double>> spheroidal, toroidal;
};

class ToroidalViscoelastic {
 public:
  using Fields = std::map<int, std::vector<double>>;

  /// The internal variables at a time, with the strain of the solution
  /// they go with (the trapezoid's start strain).
  struct State {
    double t = 0.0;
    Fields m;
    Fields strain;
  };

  /// The operator is borrowed and must outlive this.
  explicit ToroidalViscoelastic(const ViscoelasticOperator& op);
  ~ToroidalViscoelastic();

  [[nodiscard]] int LMax() const;
  /// The unrelaxed system of a degree (for its dof map and size).
  [[nodiscard]] const ToroidalOperator& Unrelaxed(int l) const;

  /// No memory and no strain, degrees 1..lmax, 2 (l + 1) columns each.
  [[nodiscard]] State ZeroState() const;

  /// The unrelaxed systems solved under the state's memory and `force`
  /// (a degree absent from it takes none).
  [[nodiscard]] Fields SolveElastic(const State& state,
                                    const Fields& force) const;
  /// The state's strain cache set from a solution consistent with it.
  void SetStrain(State& state, const Fields& solutions) const;

  /// One step of dt ending under `force`: the new state, with its
  /// strain, and the solutions at the end. The trapezoid needs the
  /// state's strain (SetStrain, or a previous step's).
  /// `memory`, when given, is added to the step's decayed memory (the
  /// drive of a tangent-linear step, FirstOrderDrive's).
  [[nodiscard]] std::pair<State, Fields> Step(const State& state, double dt,
                                              Scheme scheme, const Fields& force,
                                              const Fields* memory = nullptr) const;

  /// The memory increment of a step from `prev` to `next` (two states of
  /// a trajectory, with their strains) for a perturbation of the inverse
  /// relaxation times at the memory points: the toroidal twin of
  /// TangentLinearSolver::FirstOrderDrive, discrete (the scheme weights'
  /// derivatives) or continuous.
  [[nodiscard]] Fields FirstOrderDrive(
      bool discrete, Scheme scheme, double dt, const State& prev,
      const State& next,
      const std::vector<std::vector<double>>& ditauPoints) const;

  /// The surface value of W per degree and column.
  [[nodiscard]] Fields Surface(const Fields& solutions) const;

  // -- the continuous adjoint's pairings --------------------------------------
  //
  // The adjoint toroidal state is a State of this class stepped in
  // reversed time. Both factors of each pairing are coefficients of real
  // fields in the natural columns, so the sum over orders counts m > 0
  // twice and the imaginary column of m = 0 not at all.

  /// Adds scale times the multiplier integrand of the kernel in the
  /// inverse relaxation times, W m-dagger (d - m) / itau per branch at
  /// the memory points of the whole mesh, for an adjoint state and the
  /// forward state at a node.
  void AccumulateItau(const State& adjoint, const State& forward, double scale,
                      std::vector<std::vector<double>>& kernel) const;
  /// Adds weight times the pairing of the kernel in the shear modulus:
  /// an adjoint toroidal displacement with the forward solution and the
  /// forward memory at a node.
  void AccumulateMaterial(const Fields& adjointSolution, const State& forward,
                          const Fields& forwardSolution, double weight,
                          std::vector<double>& mu) const;

  /// The jump of the internal variables by the strain of `solutions` over
  /// each branch's relaxation time, m_k += itau_k d: the continuous
  /// adjoint's jump at an observation. The strain cache is cleared.
  void Jump(State& state, const Fields& solutions) const;

  // -- the moduli ---------------------------------------------------------------
  //
  // A perturbation delta mu of the shear modulus at the memory points of
  // the whole mesh (the layout of PointMaterialDirection::mu), the branch
  // moduli following as their fixed fractions of it and the relaxation
  // times held, as in the spheroidal material tangent (docs/adjoints.md
  // §13). The toroidal systems know no bulk modulus.

  /// A step of the base run again: its decayed memory and its solutions.
  struct BaseStep {
    Fields decayed;
    Fields solutions;
  };
  [[nodiscard]] BaseStep Redo(const State& prev, double dt, Scheme scheme,
                              const Fields& force) const;

  /// The force of delta mu on the step: minus the effective system's
  /// variation on the base step's solution, plus the force of its decayed
  /// memory with the branch moduli's variation.
  [[nodiscard]] Fields MaterialForce(double dt, Scheme scheme,
                                     std::span<const double> dmu,
                                     const BaseStep& base) const;
  /// The start's response to delta mu: the unrelaxed systems under minus
  /// their variation on the base start solution.
  [[nodiscard]] Fields StartMaterialResponse(std::span<const double> dmu,
                                             const Fields& baseStart) const;

  // -- the transpose, for the discrete adjoint ---------------------------------

  /// The duals of the internal variables and of the strain a step starts
  /// from, in State's layouts.
  struct Dual {
    Fields m;
    Fields strain;
  };
  [[nodiscard]] Dual ZeroDual() const;

  /// The transpose of the step that ended at a node: `surfaceDual` is the
  /// dual of the node's surface W per degree and column (a degree absent
  /// has none). The duals are carried back to the step's start, and the
  /// dual solutions returned are the duals of the step's forces.
  /// With `pairing`, the step's part of the kernel in the inverse
  /// relaxation times is added to it: the duals paired with the forward
  /// step's states through the scheme weights' derivatives, per branch
  /// at the memory points of the whole mesh (the discrete sweep's
  /// accumulation, before its map to the nodes).
  struct ItauPairing {
    const State* prev = nullptr;
    const State* next = nullptr;
    std::vector<std::vector<double>>* kernel = nullptr;
  };
  /// With `material`, the step's part of the kernel in the shear modulus
  /// is added to it, at the memory points of the whole mesh: the dual
  /// solution paired with MaterialForce's derivative, the base step
  /// redone from `prev` under `force`.
  struct MaterialPairing {
    const State* prev = nullptr;
    const Fields* force = nullptr;
    std::vector<double>* mu = nullptr;
  };
  [[nodiscard]] Fields StepTransposed(
      Dual& dual, double dt, Scheme scheme, const Fields& surfaceDual,
      const ItauPairing* pairing = nullptr,
      const MaterialPairing* material = nullptr) const;
  /// The transpose of the unrelaxed start (SolveElastic on a zero state,
  /// then SetStrain): the duals of the start's forces.
  /// With `baseStart` and `mu`, the start's part of the kernel in the
  /// shear modulus is added.
  [[nodiscard]] Fields StartTransposed(const Dual& dual,
                                       const Fields& surfaceDual,
                                       const Fields* baseStart = nullptr,
                                       std::vector<double>* mu = nullptr) const;

 private:
  struct Degree;
  [[nodiscard]] const Degree& Deg(int l) const;
  /// The effective system of a degree for a (scheme, dt), kept on the
  /// operator's flavour.
  [[nodiscard]] const ToroidalOperator& Effective(Scheme scheme, double dt,
                                                  int l) const;

  const ViscoelasticOperator* op_;
  mutable std::map<int, std::unique_ptr<Degree>> degrees_;
};

}  // namespace sl3dpp
