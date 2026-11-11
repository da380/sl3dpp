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
 * element, point, column) over the degree's memory window. Everything is
 * in the material's units.
 */

#include <map>
#include <memory>
#include <vector>

#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp {

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
  [[nodiscard]] std::pair<State, Fields> Step(const State& state, double dt,
                                              Scheme scheme,
                                              const Fields& force) const;

  /// The surface value of W per degree and column.
  [[nodiscard]] Fields Surface(const Fields& solutions) const;

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
  [[nodiscard]] Fields StepTransposed(Dual& dual, double dt, Scheme scheme,
                                      const Fields& surfaceDual) const;
  /// The transpose of the unrelaxed start (SolveElastic on a zero state,
  /// then SetStrain): the duals of the start's forces.
  [[nodiscard]] Fields StartTransposed(const Dual& dual,
                                       const Fields& surfaceDual) const;

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
