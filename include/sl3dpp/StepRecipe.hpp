#pragma once

/**
 * @file StepRecipe.hpp
 * @brief A time step as a recipe of implicit stages, each a one-stage step
 * the operators already take: the engine the evolutions step by (1D and
 * 3D, with gravity and without, the 1D toroidal systems beside them), the
 * continuous adjoint sweeps too, and the operators' own Step under a
 * multi-stage scheme.
 *
 * A stage of a step of length h from t_n is a one-stage step of its
 * family (ExponentialTrapezoid or BackwardEuler) of length scale h, ending
 * at t_n + c h: opened (ViscoelasticOperator::StartStep, Viscoelastic3D::
 * StartStep) from a memory state, closed by the driver's equilibrium at
 * the stage's time (the sea level equation, the coupled fixed point, the
 * load channels and the sources there), and finished. Its memory state
 * is a constant combination of the step's start state (index 0) and the
 * end states of the stages before (index i >= 1, the i-th stage's). Every
 * recipe ends with a stage at c = 1, whose end state is the node's (the
 * schemes are stiffly accurate).
 *
 * The recipes (RecipeOf):
 *  - ExponentialTrapezoid, BackwardEuler: one stage {c = 1, scale = 1}
 *    of the scheme itself from the start state; the driver's one-stage
 *    loop, operation for operation.
 *  - Sdirk2 (Alexander 1977; Hairer & Wanner, Solving ODEs II, IV.6):
 *    the tableau [gamma, 0; 1 - gamma, gamma] with gamma = 1 - 1/sqrt 2,
 *    two backward Euler stages of length gamma h. The first from m_n to
 *    t_n + gamma h; the second from (1 - r) m_n + r Y_1 to t_n + h,
 *    r = (1 - gamma) / gamma, which is m_n + (1 - gamma) h f(Y_1) with
 *    h f(Y_1) = (Y_1 - m_n) / gamma from the first stage. Second order,
 *    L-stable (its amplification (1 + (1 - 2 gamma) z) / (1 - gamma z)^2
 *    vanishes as z -> -infinity), stiffly accurate. Both stages are one
 *    effective operator, backward Euler's at gamma h (one factorisation
 *    per step size, the stage lengths formed alike so that they key the
 *    same cached operator), and neither reads a start strain: a step
 *    starts from the node's internal variables alone, so a checkpoint at
 *    a node is a complete restart and the run needs no other history.
 *  - A step of zero length, under every scheme: one stage of length zero
 *    (a jump of the forcing; StepWeights.hpp at h = 0), the elastic solve
 *    with the memory held.
 *
 * What the drivers do at a stage time inside a step: the forcing is read
 * there from its callbacks (at c = 1 at the node, as the node reads it,
 * a repeated node's left limit included); under migrating shorelines the
 * stage solves the nonlinear sea level equation there; under interior
 * sources the stage carries their forces there and steps the toroidal
 * systems through the same recipe. The trajectory a run records is its
 * nodes': a stage state between nodes is not kept.
 */

#include <cstddef>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "sl3dpp/StepWeights.hpp"

namespace sl3dpp {

/// One implicit stage of a step of length h (see the file comment).
struct RecipeStage {
  /// The stage ends at t_n + c h.
  double c = 1.0;
  /// The stage is a step of length scale h.
  double scale = 1.0;
  /// The one-stage scheme it is a step of: ExponentialTrapezoid or
  /// BackwardEuler.
  Scheme family = Scheme::ExponentialTrapezoid;
  /// Its memory state, sum_j a_j X_j over (j, a_j): X_0 the step's start
  /// state, X_i (i >= 1) the i-th stage's end state.
  std::vector<std::pair<int, double>> start{{0, 1.0}};
};

/// A step's stages, in order; the last ends at c = 1.
struct StepRecipe {
  std::vector<RecipeStage> stages;
};

/// Sdirk2's gamma, 1 - 1/sqrt 2: the stages' length over the step's and
/// the first stage's end time within it.
inline constexpr double kSdirk2Gamma = 0.29289321881345247560;

/// Whether a scheme steps in more than one stage (Sdirk2): it then has no
/// one set of step weights and no one effective operator, and a step of
/// it is opened stage by stage.
[[nodiscard]] constexpr bool IsMultiStage(Scheme scheme) {
  return scheme == Scheme::Sdirk2;
}

/// The one-stage scheme a step of zero length is opened with: the scheme
/// itself, or for a multi-stage scheme backward Euler (any scheme's
/// zero-length step is the elastic solve with the memory held).
[[nodiscard]] constexpr Scheme ZeroLengthScheme(Scheme scheme) {
  return IsMultiStage(scheme) ? Scheme::BackwardEuler : scheme;
}

/// The recipe of a step of length h >= 0 under a scheme: one stage for
/// the one-stage implicit schemes and for any step of zero length, two
/// for Sdirk2. ExponentialEuler, explicit, has none (refused).
[[nodiscard]] StepRecipe RecipeOf(Scheme scheme, double h);

/// The length of a stage of a step of length h: h itself at scale one,
/// else scale h, so that stages of equal scale have equal lengths to the
/// bit (and key one cached operator).
[[nodiscard]] constexpr double StageLength(const RecipeStage& stage, double h) {
  return stage.scale == 1.0 ? h : stage.scale * h;
}

/// The memory arrays of states combined: sum_j a_j x_j per key, over
/// (a_j, x_j) whose maps hold the same keys with arrays of the same sizes
/// (the per-degree layouts of ViscoelasticState::m and ToroidalViscoelastic
/// ::State::m).
template <class Fields>
[[nodiscard]] Fields CombineFields(
    std::span<const std::pair<double, const Fields*>> terms) {
  auto out = *terms.front().second;
  for (auto& [key, values] : out) {
    const auto a = terms.front().first;
    for (auto& v : values) v *= a;
  }
  for (const auto& [a, fields] : terms.subspan(1)) {
    for (auto& [key, values] : out) {
      const auto& add = fields->at(key);
      for (auto i = std::size_t{0}; i < values.size(); ++i) {
        values[i] += a * add[i];
      }
    }
  }
  return out;
}

/// One step of length h by a recipe, from `start`. `stage(from, s, length,
/// index)` takes the stage s (the index-th, from 0) of the given length
/// from the memory state `from` and returns its end state; `combine(terms,
/// s)` forms a stage's memory state from (coefficient, state) terms where
/// the stage starts from more than one state, or from one with a
/// coefficient other than one (a stage that starts from one state as it
/// is starts from it, uncopied); `release(state)` takes back each such
/// combined state once its stage is done and each stage's end state but
/// the last once the step is done (an operator's arrays recycled, say).
/// A combined state's time is set to t_n + (c - scale) h, so that its
/// stage ends at t_n + c h, and the step's end state's to t_n + h. State
/// has a member `t`. Returns the last stage's end state.
template <class State, class Stage, class Combine, class Release>
[[nodiscard]] State StepByRecipe(const StepRecipe& recipe, double h,
                                 const State& start, Stage&& stage,
                                 Combine&& combine, Release&& release) {
  auto done = std::vector<State>{};
  done.reserve(recipe.stages.size());
  auto source = [&](int j) -> const State& {
    return j == 0 ? start : done.at(static_cast<std::size_t>(j - 1));
  };
  for (auto index = std::size_t{0}; index < recipe.stages.size(); ++index) {
    const auto& s = recipe.stages[index];
    const auto length = StageLength(s, h);
    if (s.start.size() == 1 && s.start.front().second == 1.0) {
      done.push_back(stage(source(s.start.front().first), s, length, index));
      continue;
    }
    auto terms = std::vector<std::pair<double, const State*>>{};
    for (const auto& [j, a] : s.start) terms.emplace_back(a, &source(j));
    auto from =
        combine(std::span<const std::pair<double, const State*>>(terms), s);
    from.t = start.t + (s.c - s.scale) * h;
    done.push_back(stage(from, s, length, index));
    release(std::move(from));
  }
  auto out = std::move(done.back());
  done.pop_back();
  for (auto& state : done) release(std::move(state));
  out.t = start.t + h;
  return out;
}

/// A stage's memory state on a second chain stepped beside the one
/// StepByRecipe steps (the toroidal systems beside the spheroidal, under
/// one set of stage forces): from the chain's start state and the end
/// states of its stages before, as StepByRecipe forms it; a combined
/// state is formed into `held` and returned from there.
template <class State, class Combine>
[[nodiscard]] const State& StageStart(const RecipeStage& stage, double h,
                                      const State& start,
                                      std::span<const State> stages,
                                      Combine&& combine, State& held) {
  auto source = [&](int j) -> const State& {
    return j == 0 ? start : stages[static_cast<std::size_t>(j - 1)];
  };
  if (stage.start.size() == 1 && stage.start.front().second == 1.0) {
    return source(stage.start.front().first);
  }
  auto terms = std::vector<std::pair<double, const State*>>{};
  for (const auto& [j, a] : stage.start) terms.emplace_back(a, &source(j));
  held = combine(std::span<const std::pair<double, const State*>>(terms));
  held.t = start.t + (stage.c - stage.scale) * h;
  return held;
}

/// StepByRecipe with nothing to take back.
template <class State, class Stage, class Combine>
[[nodiscard]] State StepByRecipe(const StepRecipe& recipe, double h,
                                 const State& start, Stage&& stage,
                                 Combine&& combine) {
  return StepByRecipe(recipe, h, start, std::forward<Stage>(stage),
                      std::forward<Combine>(combine), [](State&&) {});
}

namespace detail {

/// The refusal of a derivative of the stepped map (the discrete adjoint,
/// the tangent-linear run, the second-order sweeps) under a multi-stage
/// scheme: not built. Its exact form runs through each step's stages and
/// needs their states, which a run's trajectory (its nodes') does not
/// hold. The continuous adjoint steps the adjoint by the same stages: the
/// one to state (Options::method), the discrete being the default.
inline void RefuseDerivativeOfMultiStage(Scheme scheme, std::string_view what) {
  if (!IsMultiStage(scheme)) return;
  throw std::invalid_argument(std::format(
      "{} is not built for Sdirk2: its exact derivative of the stepped map "
      "runs through each step's two stages and needs their states, which "
      "the trajectory (the nodes' states) does not hold. The continuous "
      "adjoint (Options::method AdjointMethod::Continuous) steps the "
      "adjoint by the same stages",
      what));
}

/// The refusal of a recording with gaps for a sweep about a run under a
/// multi-stage scheme: a state between two kept ones is rebuilt by forward
/// steps under the recorded node coefficients, and a step of Sdirk2 closes
/// its first stage under the forcing at its own time, which the recording
/// does not hold.
inline void RefuseGapsUnderMultiStage(Scheme scheme, bool gaps) {
  if (!IsMultiStage(scheme) || !gaps) return;
  throw std::invalid_argument(
      "a sweep about a run under Sdirk2 reads the forward state at every "
      "node: record it at every node (record_stride 1). A state between "
      "kept ones is rebuilt by forward steps under the recorded node loads, "
      "and Sdirk2's step closes its first stage under the forcing at its "
      "own time, which the trajectory does not hold");
}

}  // namespace detail

}  // namespace sl3dpp
