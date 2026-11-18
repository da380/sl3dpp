#pragma once

/**
 * @file detail/StepStages.hpp
 * @brief What the derivatives of the stepped map (the discrete adjoint,
 * the tangent-linear runs, the second-order sweeps) share in taking a step
 * stage by stage (StepRecipe.hpp), and what the continuous sweeps read of
 * a step's stages (their quadrature weights, a migrating run's shorelines
 * inside a step). Implementation detail.
 *
 * A stage is a one-stage step of its family from a memory state that is a
 * constant combination of the step's start state and the stages' end
 * states before, closed under its own forcing. So the
 * derivative of a step is its stages' derivatives chained through the
 * combinations: forwards, a tangent state is stepped by the recipe as the
 * forward state is, each stage driven by the weights' derivatives at the
 * stage's length about the forward stage's states; backwards, the stages
 * are transposed last to first, each the one-stage step's own transpose,
 * and the dual of a stage's memory state is handed to the states it
 * combines, times their coefficients (TransposeByRecipe).
 *
 * The forcing inside a step. A run reads its forcing at its nodes alone:
 * a stage that ends inside a step (c < 1) takes (1 - c) of the forcing at
 * the step's start node and c of it at the end node (detail::AtStage of
 * TimeGrid.hpp, the one place the forward runs decide it). The derivative
 * runs hold the forcing by its node values and take the same combination:
 * a direction's value at such a stage is (1 - c) of the start node's and
 * c of the end node's (AtStage here), and the dual of the stage's forcing
 * is handed to the two nodes with the same weights (FoldOnNodes). The
 * tangent-linear run is so the derivative of the forward run along any
 * direction in the node values, and the adjoint sweep its transpose and
 * the run's exact gradient in them, whatever the forcing does between
 * nodes, which no run reads.
 */

#include <cstddef>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sl3dpp/StepRecipe.hpp"

namespace sl3dpp::detail {

/// One stage of the step ending at a node, as a derivative run takes it.
struct StageRef {
  std::size_t node = 0;   ///< the step's end node
  std::size_t index = 0;  ///< the stage within the step, from 0
  std::size_t count = 1;  ///< the step's stages
  double length = 0.0;    ///< the stage's length, material time
  double c = 1.0;         ///< the stage ends at fraction c of the step
  Scheme family = Scheme::ExponentialTrapezoid;
  /// Whether the stage ends at the node (the step's last stage): it then
  /// reads the node's forcing, measurements and shorelines.
  [[nodiscard]] bool Last() const { return index + 1 == count; }
};

/// The stages of the step of length h ending at a node, in order. A
/// derivative run takes a step of several stages only where every stage is
/// a backward Euler step, which reads no start strain: a combined memory
/// state has none.
[[nodiscard]] inline std::vector<StageRef> StagesOf(const StepRecipe& recipe,
                                                    double h,
                                                    std::size_t node) {
  auto out = std::vector<StageRef>{};
  const auto count = recipe.stages.size();
  for (auto i = std::size_t{0}; i < count; ++i) {
    const auto& stage = recipe.stages[i];
    if (count > 1 && stage.family != Scheme::BackwardEuler) {
      throw std::logic_error(
          "a derivative of a step of several stages takes backward Euler "
          "stages alone");
    }
    out.push_back({.node = node,
                   .index = i,
                   .count = count,
                   .length = StageLength(stage, h),
                   .c = stage.c,
                   .family = stage.family});
  }
  return out;
}

/// The transpose of StepByRecipe. `dual` is the dual of the step's end
/// state on entry and of its start state on return. `stage(ref, d)` is a
/// stage's own transpose: it takes the dual of the stage's end state in
/// `d` and leaves there the dual of the stage's memory state.
/// `scaled(a, d)` is a d as a new dual and `add(to, a, d)` adds a d to
/// one, for the transposed combinations. A one-stage step from its start
/// state is the stage's transpose alone, on `dual` in place.
template <class Dual, class Stage, class Scaled, class Add>
void TransposeByRecipe(const StepRecipe& recipe, std::span<const StageRef> refs,
                       Dual& dual, Stage&& stage, Scaled&& scaled, Add&& add) {
  const auto count = recipe.stages.size();
  if (count == 1 && recipe.stages.front().start.size() == 1 &&
      recipe.stages.front().start.front() == std::pair<int, double>{0, 1.0}) {
    stage(refs.front(), dual);
    return;
  }
  // the duals of the step's start state (0) and of each stage's end state
  auto ends = std::vector<std::optional<Dual>>(count + 1);
  ends[count] = std::move(dual);
  for (auto i = count; i-- > 0;) {
    if (!ends[i + 1].has_value()) {
      throw std::logic_error(
          "a stage's end state feeds a later stage or is the step's own");
    }
    auto d = std::move(*ends[i + 1]);
    ends[i + 1].reset();
    stage(refs[i], d);
    for (const auto& [j, a] : recipe.stages[i].start) {
      auto& to = ends[static_cast<std::size_t>(j)];
      if (to.has_value()) {
        add(*to, a, d);
      } else {
        to = scaled(a, d);
      }
    }
  }
  dual = std::move(*ends[0]);
}

/// A direction's value at a stage of the step ending at node n, from its
/// node values (the file comment): the node's at c = 1, else (1 - c) of
/// node n - 1's and c of node n's. `at(node)` gives a node's value, a
/// vector that may be empty (none there).
template <class At>
[[nodiscard]] auto AtStage(const StageRef& ref, At&& at) {
  auto end = at(ref.node);
  if (ref.Last()) return end;
  const auto start = at(ref.node - 1);
  if (end.empty() && start.empty()) return end;
  if (end.empty()) end.assign(start.size(), {});
  for (auto i = std::size_t{0}; i < end.size(); ++i) {
    end[i] *= ref.c;
    if (!start.empty()) end[i] += (1.0 - ref.c) * start[i];
  }
  return end;
}

/// The states of a step's stages, in the stages' order: each stage's
/// memory state and its end state.
template <class State>
struct StageStates {
  std::vector<State> starts, ends;
};

/// The states of the stages of a step of length h, from the step's start
/// and end states: `innerEnd(index, from)` gives the end state of an inner
/// stage from its memory state (a recorded one, or the stage repeated),
/// and `combine(terms)` a combined memory state as StepByRecipe forms it.
/// A combined state's time is set as StepByRecipe sets it.
template <class State, class InnerEnd, class Combine>
[[nodiscard]] StageStates<State> StatesOfStages(const StepRecipe& recipe,
                                                double h, const State& start,
                                                State end, InnerEnd&& innerEnd,
                                                Combine&& combine) {
  auto out = StageStates<State>{};
  const auto count = recipe.stages.size();
  out.starts.reserve(count);
  out.ends.reserve(count);
  for (auto i = std::size_t{0}; i < count; ++i) {
    const auto& stage = recipe.stages[i];
    auto source = [&](int j) -> const State& {
      return j == 0 ? start : out.ends[static_cast<std::size_t>(j - 1)];
    };
    if (stage.start.size() == 1 && stage.start.front().second == 1.0) {
      out.starts.push_back(source(stage.start.front().first));
    } else {
      auto terms = std::vector<std::pair<double, const State*>>{};
      for (const auto& [j, a] : stage.start) terms.emplace_back(a, &source(j));
      auto from =
          combine(std::span<const std::pair<double, const State*>>(terms));
      from.t = start.t + (stage.c - stage.scale) * h;
      out.starts.push_back(std::move(from));
    }
    if (i + 1 == count) {
      out.ends.push_back(std::move(end));
    } else {
      out.ends.push_back(innerEnd(i, out.starts[i]));
    }
  }
  return out;
}

/// The quadrature weights of a recipe's stages: a step is
/// m_n+1 = m_n + h sum_i b_i f(Y_i), Y_i the i-th stage's end state, so
/// b_i h is the share of the step its i-th stage carries (Sdirk2's
/// 1 - gamma and gamma; a one-stage recipe's 1). Each state is m_n plus
/// h times a combination of the stages' f: a stage's memory state takes
/// its sources' combinations, its end state adds scale times its own f;
/// b is the last stage's.
[[nodiscard]] inline std::vector<double> StageQuadratureWeights(
    const StepRecipe& recipe) {
  const auto count = recipe.stages.size();
  // the coefficients of each stage's end state on the stages' f
  auto ends = std::vector<std::vector<double>>{};
  for (auto i = std::size_t{0}; i < count; ++i) {
    const auto& stage = recipe.stages[i];
    auto row = std::vector<double>(count, 0.0);
    for (const auto& [j, a] : stage.start) {
      if (j == 0) continue;  // the step's start state carries none
      const auto& from = ends.at(static_cast<std::size_t>(j - 1));
      for (auto k = std::size_t{0}; k < count; ++k) row[k] += a * from[k];
    }
    row[i] += stage.scale;
    ends.push_back(std::move(row));
  }
  return ends.back();
}

/// The ocean function at fraction s of a step (0 at its start node, 1 at
/// its end node) about a migrating run: the run solved for its shorelines
/// at the step's two nodes and at its inner stages, which end at their
/// fractions c of the step (`inner`, in the recipe's order); between those
/// times it is taken linear in time. At a time where the run solved, that
/// solve's ocean function itself; elsewhere a combination of the two
/// either side, a coefficient between 0 and 1 in a cell whose shoreline
/// moved between them. The continuous adjoint's reversed stages, which
/// end where the run has no stage, close under it.
[[nodiscard]] inline std::vector<double> OceanFunctionWithinStep(
    const StepRecipe& recipe, double s, const std::vector<double>& start,
    std::span<const std::vector<double>> inner, const std::vector<double>& end) {
  if (inner.size() + 1 != recipe.stages.size()) {
    throw std::logic_error(
        "a step's inner stages and their ocean functions do not match");
  }
  auto samples = std::vector<std::pair<double, const std::vector<double>*>>{
      {0.0, &start}};
  for (auto i = std::size_t{0}; i < inner.size(); ++i) {
    samples.emplace_back(recipe.stages[i].c, &inner[i]);
  }
  samples.emplace_back(1.0, &end);
  for (auto k = std::size_t{1}; k < samples.size(); ++k) {
    const auto [s0, a] = samples[k - 1];
    const auto [s1, b] = samples[k];
    if (s > s1 && k + 1 < samples.size()) continue;
    if (s <= s0) return *a;
    if (s >= s1) return *b;
    const auto w = (s - s0) / (s1 - s0);
    auto out = *a;
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = (1.0 - w) * (*a)[i] + w * (*b)[i];
    }
    return out;
  }
  return end;
}

/// The inner stages of the step ending at a node: those a multi-stage
/// scheme takes before the step's last, none for a step of zero length.
[[nodiscard]] inline std::size_t InnerStageCount(Scheme scheme,
                                                 std::span<const double> times,
                                                 std::size_t node) {
  if (node == 0 || !IsMultiStage(scheme)) return 0;
  return RecipeOf(scheme, times[node] > times[node - 1] ? 1.0 : 0.0)
             .stages.size() -
         1;
}

/// The checks a derivative run makes of the inner stages' ocean functions
/// (OceanFunctions::stages): about a migrating run under a multi-stage
/// scheme, one grid per inner stage of every step; none otherwise needed.
/// A list shorter than the nodes gives the nodes past its end none, which
/// is right where their steps have no inner stage (of zero length).
inline void CheckStageOceanFunctions(
    Scheme scheme, std::span<const double> times, bool migrating,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    std::size_t fieldSize) {
  if (!migrating || !IsMultiStage(scheme)) return;
  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto inner = InnerStageCount(scheme, times, node);
    const auto given = node < stageOceanFunctions.size()
                           ? stageOceanFunctions[node].size()
                           : std::size_t{0};
    if (given != inner) {
      throw std::invalid_argument(std::format(
          "about a migrating run under Sdirk2 each step's first stage is "
          "linearised about its own shorelines: give the stages' ocean "
          "functions (OceanFunctions::stages, which OceanFunctions::Of the "
          "run's snapshots fills). The step ending at node {} has {} inner "
          "stage{}, {} given",
          node, inner, inner == 1 ? "" : "s", given));
    }
    if (given == 0) continue;  // and none to index past the given nodes
    for (const auto& C : stageOceanFunctions[node]) {
      if (C.size() != fieldSize) {
        throw std::invalid_argument(
            "the stages' ocean functions must be fields on the solver's "
            "grid");
      }
    }
  }
}

/// The transpose of AtStage: a stage's kernel (the dual of a forcing term
/// the stage took) handed to the step's two nodes, (1 - c) to
/// node n - 1 and c to node n. `perNode` holds every node's kernel; an
/// empty one is sized as the stage's first.
template <class T>
void FoldOnNodes(std::vector<std::vector<T>>& perNode, std::size_t node,
                 double c, const std::vector<T>& stageKernel) {
  if (stageKernel.empty()) return;
  for (const auto& [to, a] :
       {std::pair{node - 1, 1.0 - c}, std::pair{node, c}}) {
    auto& kernel = perNode[to];
    if (kernel.empty()) kernel.assign(stageKernel.size(), T{});
    for (auto i = std::size_t{0}; i < kernel.size(); ++i) {
      kernel[i] += a * stageKernel[i];
    }
  }
}

}  // namespace sl3dpp::detail
