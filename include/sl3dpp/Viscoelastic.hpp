#pragma once

/**
 * @file Viscoelastic.hpp
 * @brief The radial viscoelastic core: state, slaved degree solves,
 * steppers. A port of pyslfp's `viscoelastic.operator`, non-dimensional
 * throughout.
 *
 * The state is the internal variables alone; the displacement is slaved
 * through elastic solves (AdGIA's layering). Per branch and degree the
 * spheroidal reduction of Al-Attar & Tromp (2014), Appendix D, leaves
 * three radial scalars per coefficient column, stored in the order
 * (M, R, S) with driving strains
 *
 *   e_M = V,  e_R = r d_r U - U + (1/2) k^2 V,  e_S = r d_r V - V + U,
 *
 * and force weights k^2 (k^2 - 2), 4/3 and k^2 (pyslfp's
 * docs/planning/gia_notes.md records the algebra). Evaluating anything —
 * displacement, a step — contains quasi-static degree solves: the system
 * assembled with the unrelaxed or a per-step effective shear modulus,
 * forced by the load channels and the memory force
 *
 *   b_mem(U', V') = sum_k int mu_k [ (4/3) R_k e_R(U', V')
 *                   + k^2 S_k e_S(U', V') + k^2 (k^2 - 2) M_k V' ] dr.
 *
 * Steppers: backward Euler, exponential Euler, and the exponential
 * trapezoid (default; second order, no stability limit, one solve per
 * step), with the weight algebra of StepWeights.hpp and one factorised
 * banded system per degree and (scheme, dt) flavour, cached and reused.
 * Each degree's columns are independent right-hand sides: one column for
 * a Love-number history, 2 (l + 1) for the full set of spherical-harmonic
 * orders (the real and imaginary blocks of the m >= 0 coefficients).
 *
 * At degree 0, e_M vanishes with the V dof and the S scalar carries zero
 * force weight, so its drive is zeroed; fluid elements have no
 * displacement dofs and no active branches, so their strain gathers to
 * zero harmlessly.
 *
 * Array layouts (all row-major unless stated): a state's per-degree
 * internal variables have shape (K, 3, ne, ngll, ncols) over the degree's
 * element subset; strains (3, ne, ngll, ncols); degree solutions are
 * column-major (ndof, ncols) as DegreeOperator::Solve returns them.
 */

#include <complex>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/StepWeights.hpp"

namespace sl3dpp {

/// A load channel: columns of a degree's coefficient forcing at a time,
/// or std::nullopt where the channel is silent at that degree.
using LoadCallable =
    std::function<std::optional<std::vector<double>>(int l, double t)>;

/// The internal variables at one time. The strain cache holds the last
/// solve's strain scalars when the displacement is consistent with (m, t);
/// a load jump at exactly t must be followed by Invalidate().
struct ViscoelasticState {
  std::map<int, std::vector<double>> m;  ///< (K, 3, ne, ngll, ncols)
  double t = 0.0;
  std::map<int, std::vector<double>> strain;  ///< cache; empty = none

  [[nodiscard]] bool HasStrain() const { return !strain.empty(); }
  void Invalidate() { strain.clear(); }
};

class ViscoelasticStep;

class ViscoelasticOperator {
 public:
  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
  };

  /// The rheology is copied into immutable shared storage, so the
  /// operator never dangles; every degree 0..lmax is carried.
  ViscoelasticOperator(const Rheology& rheology, int lmax, Options options);
  ViscoelasticOperator(const Rheology& rheology, int lmax)
      : ViscoelasticOperator(rheology, lmax, Options{}) {}
  ~ViscoelasticOperator();  // out of line: unique_ptr of incomplete types
  ViscoelasticOperator(ViscoelasticOperator&&) noexcept;
  ViscoelasticOperator& operator=(ViscoelasticOperator&&) noexcept;

  [[nodiscard]] const Rheology& GetRheology() const { return *rheology_; }
  [[nodiscard]] const Material& BaseMaterial() const {
    return rheology_->BaseMaterial();
  }
  [[nodiscard]] int LMax() const { return lmax_; }

  /// The undisturbed state at t = 0 with ncols(l) columns per degree.
  [[nodiscard]] ViscoelasticState ZeroState(
      const std::function<int(int)>& ncols) const;
  [[nodiscard]] ViscoelasticState ZeroState(int ncols = 1) const {
    return ZeroState([ncols](int) { return ncols; });
  }

  // -- the slaved solve
  // --------------------------------------------------------

  /// The quasi-static solutions consistent with (m, t): the unrelaxed
  /// systems forced by the loads at state.t and the memory force of m.
  [[nodiscard]] std::map<int, std::vector<double>> SolveElastic(
      const ViscoelasticState& state, const LoadCallable& load = {},
      const LoadCallable& tide = {}) const;

  /// The driving strain scalars of degree solutions.
  [[nodiscard]] std::map<int, std::vector<double>> Strain(
      const std::map<int, std::vector<double>>& solutions) const;

  /// The surface value of U, V or phi per degree and column, zero where
  /// the component has no dof.
  [[nodiscard]] std::map<int, std::vector<double>> Surface(
      const std::map<int, std::vector<double>>& solutions,
      Component component) const;

  /// Ensures the state carries a strain cache consistent with (m, t)
  /// under the given loads, solving once if not, and returns it.
  const std::map<int, std::vector<double>>& PrimeStrain(
      ViscoelasticState& state, const LoadCallable& load = {},
      const LoadCallable& tide = {}) const;

  // -- steppers
  // -----------------------------------------------------------------

  /// One step from state.t under the load channels (which the schemes
  /// read at the start and/or end time). Returns the new state (with its
  /// strain cache for the implicit schemes) and the last degree solutions
  /// (consistent with the new state; for the exponential Euler, the
  /// start-of-step solutions).
  [[nodiscard]] std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
  Step(const ViscoelasticState& state, double dt,
       Scheme scheme = Scheme::ExponentialTrapezoid,
       const LoadCallable& load = {}, const LoadCallable& tide = {}) const;

  /// The exact algebraic inverse of Step for the trapezoid and backward
  /// Euler: the state one dt EARLIER with consistent solutions and
  /// strain cache (the load channels given are the earlier time's). The
  /// state must carry its strain cache. Dissipation runs the other way,
  /// so roundoff is amplified by exp(h) per branch and node; a
  /// checkpointing driver budgets the accumulated h between stores.
  [[nodiscard]] std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
  StepBack(const ViscoelasticState& state, double dt,
           Scheme scheme = Scheme::ExponentialTrapezoid,
           const LoadCallable& load = {}, const LoadCallable& tide = {}) const;

  /// Opens an implicit or trapezoid step before the end-time loads are
  /// known: the sea-level driver reads the step's memory responses,
  /// closes its equilibrium with the matching effective Love numbers, and
  /// completes the step with ViscoelasticStep::Finish.
  [[nodiscard]] ViscoelasticStep StartStep(
      const ViscoelasticState& state, double dt,
      Scheme scheme = Scheme::ExponentialTrapezoid,
      const LoadCallable& load = {}, const LoadCallable& tide = {}) const;

  /// One trial trapezoid step with the embedded ETD1 error estimate:
  /// err = RMS of (m - m_hat) / (atol + rtol max|m|), accepted when
  /// err <= 1, dt_next = dt clamp(0.9 / sqrt(err), 0.2, 4). Returns
  /// (state, solutions, err, dt_next), the first empty on rejection.
  struct AdaptiveResult {
    std::optional<ViscoelasticState> state;
    std::map<int, std::vector<double>> solutions;
    double error = 0.0;
    double dtNext = 0.0;
  };
  [[nodiscard]] AdaptiveResult AdaptiveStep(
      ViscoelasticState& state, double dt, double rtol = 1e-4,
      const LoadCallable& load = {}, const LoadCallable& tide = {}) const;

  // -- effective solids
  // ----------------------------------------------------------

  /// The material of a step's effective elastic solve: branch k weighted
  /// by 1/(1 + h_k) (backward Euler) or phi1(h_k) (trapezoid).
  [[nodiscard]] Material EffectiveMaterial(Scheme scheme, double dt) const;

  /// The Love numbers of the step's effective solid (SI, as
  /// ComputeLoveNumbers), cached per (scheme, dt); and the unrelaxed
  /// solid's, cached once.
  [[nodiscard]] const LoveNumbers& EffectiveLoveNumbers(Scheme scheme,
                                                        double dt) const;
  [[nodiscard]] const LoveNumbers& ElasticLoveNumbers() const;

  /// The explicit stability scale of the rheology.
  [[nodiscard]] double MinRelaxationTime() const {
    return rheology_->MinRelaxationTime();
  }

  /// The inertia moment sqrt(4 pi) int rho U r^3 dr of a degree-0
  /// solution array, per column.
  [[nodiscard]] std::vector<double> InertiaMoment(
      std::span<const double> solution0, int ncols) const;

 private:
  friend class ViscoelasticStep;

  /// Per-degree workspace: the unrelaxed system, strain operators and
  /// branch data restricted to the degree's element subset.
  struct Degree;
  /// A (scheme, dt) flavour: its effective material, factorised systems
  /// and cached unit responses.
  struct Flavour;

  [[nodiscard]] Degree& Deg(int l) const;
  [[nodiscard]] Flavour& FlavourOf(Scheme scheme, double dt) const;
  [[nodiscard]] const DegreeOperator& EffectiveSystem(Flavour& flavour,
                                                      int l) const;
  /// The flavour's unit load and tide responses at one degree, columns
  /// (load, tide); flavour nullptr means the unrelaxed system.
  [[nodiscard]] const std::vector<double>& UnitResponses(Flavour* flavour,
                                                         int l) const;
  void AddChannels(std::vector<double>& x, int l, Flavour* flavour, double t,
                   const LoadCallable& load, const LoadCallable& tide,
                   int ncols) const;
  [[nodiscard]] std::map<int, std::vector<double>> StartStrain(
      const ViscoelasticState& state, const LoadCallable& load,
      const LoadCallable& tide) const;
  /// Per-branch nodal weight arrays of a flavour, over the whole mesh.
  struct BranchWeights {
    std::vector<double> e, alpha, beta;  // backward Euler keeps w in e
  };
  [[nodiscard]] const std::vector<BranchWeights>& SchemeWeightsOf(
      Scheme scheme, double dt) const;

  std::shared_ptr<const Rheology> rheology_;  ///< owned, immutable
  int lmax_;
  Options options_;
  mutable std::map<int, std::unique_ptr<Degree>> degrees_;
  mutable std::list<std::pair<std::pair<int, double>, std::unique_ptr<Flavour>>>
      flavours_;  // LRU, most recent at front
  mutable std::map<std::pair<int, double>, std::vector<BranchWeights>> weights_;
  mutable std::map<std::pair<int, double>, LoveNumbers> effectiveLove_;
  mutable std::vector<double> unitUnrelaxed_;  // placeholder, see cpp
  mutable std::map<int, std::vector<double>> unrelaxedUnits_;
};

/// One opened implicit or trapezoid step. On construction the start state
/// is decayed to the step's memory state m~ (e m^n + alpha d^n for the
/// trapezoid, m^n / (1 + h) for backward Euler), which is fixed; the
/// end-time loads enter linearly on top.
class ViscoelasticStep {
 public:
  [[nodiscard]] double EndTime() const { return t_; }

  /// The degree solutions of the memory state alone (zero load) through
  /// the effective systems, cached.
  [[nodiscard]] const std::map<int, std::vector<double>>& MemorySolutions()
      const;

  /// The surface U, V or phi of the memory response, per degree/column.
  [[nodiscard]] std::map<int, std::vector<double>> MemorySurface(
      Component component) const;

  /// The inertia moment of the degree-0 memory response, per column: the
  /// memory field's contribution to the trace of the inertia tensor,
  /// which the axial rotational feedback needs.
  [[nodiscard]] std::vector<double> InertiaMoment() const;

  /// The Love numbers characterising this step's instantaneous response.
  [[nodiscard]] const LoveNumbers& EffectiveLove() const;

  /// Completes the step under the end-time loads: one effective solve per
  /// degree, the state update, and solutions consistent with it.
  [[nodiscard]] std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
  Finish(const LoadCallable& load = {}, const LoadCallable& tide = {}) const;

 private:
  friend class ViscoelasticOperator;
  ViscoelasticStep(const ViscoelasticOperator& op,
                   const ViscoelasticState& state, double dt, Scheme scheme,
                   const LoadCallable& load, const LoadCallable& tide);

  const ViscoelasticOperator* op_;
  Scheme scheme_;
  double dt_;
  double t_;
  std::map<int, int> ncols_;
  std::map<int, std::vector<double>> mem_;  // the decayed memory state
  mutable std::map<int, std::vector<double>> memorySolutions_;
};

}  // namespace sl3dpp
