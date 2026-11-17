#pragma once

/**
 * @file detail/StepStages3D.hpp
 * @brief The stages of a recorded 3D run's steps again (StepRecipe.hpp),
 * for the derivative runs about it and the rebuild of a state between
 * kept ones: the 3D twin of what detail/TrajectoryCursor.hpp does for a
 * radial run. Implementation detail.
 *
 * A stage is repeated from its memory state under the channels the
 * forward run closed it with, which the recording keeps (the node's own
 * for a step's last stage, ForwardTrajectory3D::stages for an inner one)
 * and, under interior sources, the sources' forces there: the stage's
 * within-step fixed point under known channels, by the forward run's own
 * iteration, with no sea level equation to close. The state is the run's
 * to that iteration's tolerance.
 */

#include <complex>
#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/detail/StepStages.hpp"

namespace sl3dpp::detail {

/// What closed a stage of the step ending at a node on the recorded 3D
/// run (detail::StageClosure's twin); `forces` is null without sources.
struct StageClosure3D {
  const std::vector<std::complex<double>>* load = nullptr;
  const std::vector<std::complex<double>>* tide = nullptr;
  const DegreeFields* forces = nullptr;
};
[[nodiscard]] inline StageClosure3D ClosureOf(
    const ForwardTrajectory3D& trajectory, const StageRef& ref) {
  if (ref.Last()) {
    return {&trajectory.loads[ref.node], &trajectory.tides[ref.node],
            trajectory.sourceForces.empty()
                ? nullptr
                : &trajectory.sourceForces[ref.node]};
  }
  const auto& closure = trajectory.stages[ref.node][ref.index];
  return {&closure.load, &closure.tide,
          closure.sourceForces ? &*closure.sourceForces : nullptr};
}

/// The states of a step's stages by reference (the grid states are large):
/// each stage's memory state and its end state, the ones that are neither
/// the step's start nor its end held in `owned` (each at its own address,
/// which a copy or a move of this leaves in place).
struct StagePointers3D {
  using Owned = std::vector<std::shared_ptr<Viscoelastic3D::State>>;
  std::vector<const Viscoelastic3D::State*> starts, ends;
  Owned owned;
  /// Holds a state and gives its address.
  static const Viscoelastic3D::State* Hold(Owned& owned,
                                           Viscoelastic3D::State&& state) {
    return owned
        .emplace_back(
            std::make_shared<Viscoelastic3D::State>(std::move(state)))
        .get();
  }
};

/// The stages of a recorded 3D run's steps again.
struct ForwardStages3D {
  using State = Viscoelastic3D::State;
  using LoadOf = std::function<Viscoelastic3D::LoadCallable3D(
      const std::vector<std::complex<double>>&)>;

  const Viscoelastic3D* op = nullptr;
  const ForwardTrajectory3D* trajectory = nullptr;
  Scheme scheme = Scheme::ExponentialTrapezoid;
  Viscoelastic3D::IterationControl control{};
  /// A recorded coefficient vector as the operator's load and tide
  /// channels.
  LoadOf loadOf, tideOf;

  /// Refuses a trajectory of a run under a multi-stage scheme that does
  /// not give every step's inner stages' closures; `dts` the steps'
  /// lengths.
  void Check(std::span<const double> dts) const {
    if (!IsMultiStage(scheme)) return;
    for (auto node = std::size_t{1}; node <= dts.size(); ++node) {
      const auto inner = RecipeOf(scheme, dts[node - 1]).stages.size() - 1;
      const auto given = node < trajectory->stages.size()
                             ? trajectory->stages[node].size()
                             : std::size_t{0};
      if (given == inner) continue;
      throw std::invalid_argument(std::format(
          "the trajectory of a run under Sdirk2 gives the recorded "
          "coefficients of each step's first stage (ForwardTrajectory3D::"
          "stages, from the recording run's snapshots): the step ending at "
          "node {} has {} inner stage{}, the trajectory gives {}. A state "
          "inside a step, and one between kept ones, is the stage repeated "
          "under them",
          node, inner, inner == 1 ? "" : "s", given));
    }
  }

  /// One stage of the step ending at ref.node again, from its memory
  /// state.
  [[nodiscard]] State Stage(const StageRef& ref, const State& from) const {
    const auto node = ref.node;
    auto step = op->StartStep(from, ref.length, ref.family,
                              loadOf(trajectory->loads[node - 1]),
                              tideOf(trajectory->tides[node - 1]));
    const auto closure = ClosureOf(*trajectory, ref);
    if (closure.forces != nullptr && !closure.forces->spheroidal.empty()) {
      step.AddDegreeForces(closure.forces->spheroidal,
                           closure.forces->toroidal);
    }
    auto [strain, info] =
        step.Close(loadOf(*closure.load), tideOf(*closure.tide), control);
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "rebuilding the forward state at node {}{} did not converge: {}",
          node, ref.Last() ? "" : " (a stage inside its step)",
          info.Describe(control.rtol)));
    }
    return std::move(step).Finish(std::move(strain));
  }

  /// The run's step of length dt ending at a node again, from the state
  /// at the node below: a state between kept ones.
  [[nodiscard]] State Step(std::size_t node, double dt,
                           const State& from) const {
    const auto recipe = RecipeOf(scheme, dt);
    const auto refs = StagesOf(recipe, dt, node);
    return StepByRecipe(
        recipe, dt, from,
        [&](const State& s, const RecipeStage&, double, std::size_t index) {
          return Stage(refs[index], s);
        },
        [&](std::span<const std::pair<double, const State*>> terms,
            const RecipeStage&) { return op->StageState(terms); },
        [&](State&& spent) { op->Recycle(std::move(spent)); });
  }

  /// The states of the stages of the step ending at a node, from the
  /// step's start and end states (which stay the caller's): an inner
  /// stage's end state the stage repeated, a combined memory state as the
  /// forward run formed it (its strain cache an earlier stage's, which a
  /// backward Euler stage's weights multiply by zero).
  [[nodiscard]] StagePointers3D Stages(const StepRecipe& recipe,
                                       std::span<const StageRef> refs,
                                       double dt, const State& start,
                                       const State& end) const {
    return PointersOfStages(
        *op, recipe, dt, start, end,
        [&](std::size_t index, const State& from,
            StagePointers3D::Owned& owned) -> const State* {
          return StagePointers3D::Hold(owned, Stage(refs[index], from));
        });
  }

  /// The same from given inner end states: `innerEnd(index, from, owned)`
  /// gives an inner stage's end state, one it holds or one it puts in
  /// `owned`.
  template <class InnerEnd>
  [[nodiscard]] static StagePointers3D PointersOfStages(
      const Viscoelastic3D& op, const StepRecipe& recipe, double dt,
      const State& start, const State& end, InnerEnd&& innerEnd) {
    auto out = StagePointers3D{};
    const auto count = recipe.stages.size();
    for (auto i = std::size_t{0}; i < count; ++i) {
      const auto& stage = recipe.stages[i];
      auto source = [&](int j) -> const State& {
        return j == 0 ? start : *out.ends[static_cast<std::size_t>(j - 1)];
      };
      if (stage.start.size() == 1 && stage.start.front().second == 1.0) {
        out.starts.push_back(&source(stage.start.front().first));
      } else {
        auto terms = std::vector<std::pair<double, const State*>>{};
        for (const auto& [j, a] : stage.start) terms.emplace_back(a, &source(j));
        auto from = op.StageState(terms);
        from.t = start.t + (stage.c - stage.scale) * dt;
        out.starts.push_back(StagePointers3D::Hold(out.owned, std::move(from)));
      }
      out.ends.push_back(i + 1 == count ? &end
                                        : innerEnd(i, *out.starts[i], out.owned));
    }
    return out;
  }
};

/// Refuses a 3D tangent-linear run that is not recorded at every node with
/// its strain and, under a multi-stage scheme, at every inner stage
/// (TangentLinearSolver3D::Options::record).
inline void CheckRecordedTangent3D(
    std::span<const Evolution3D::Snapshot> tangent, Scheme scheme,
    std::span<const double> times) {
  if (tangent.size() != times.size()) {
    throw std::invalid_argument(
        "the tangent-linear run must give one snapshot per time node");
  }
  for (auto node = std::size_t{0}; node < tangent.size(); ++node) {
    const auto& snap = tangent[node];
    if (!snap.internalState.has_value() ||
        !snap.internalState->strain.has_value()) {
      throw std::invalid_argument(
          "the tangent-linear run must be recorded at every node with its "
          "strain (TangentLinearSolver3D::Options::record)");
    }
    const auto inner = InnerStageCount(scheme, times, node);
    auto recorded = snap.stages.size() == inner;
    for (const auto& stage : snap.stages) {
      recorded = recorded && stage.internalState.has_value() &&
                 stage.internalState->strain.has_value();
    }
    if (!recorded) {
      throw std::invalid_argument(std::format(
          "the tangent-linear run about a run under Sdirk2 must be recorded "
          "at every step's first stage too (TangentLinearSolver3D::Options::"
          "record, by a run under the same scheme): the step ending at node "
          "{} is not",
          node));
    }
  }
}

/// a F1 + b F2 over the forces of two sets of sources on the 3D degree
/// systems (a degree absent from one has none there).
[[nodiscard]] inline DegreeFields Blend(double a, const DegreeFields& f1,
                                        double b, const DegreeFields& f2) {
  auto out = DegreeFields{};
  auto add = [](std::map<int, std::vector<std::complex<double>>>& to,
                double factor,
                const std::map<int, std::vector<std::complex<double>>>& from) {
    for (const auto& [l, f] : from) {
      auto& sum = to[l];
      if (sum.empty()) sum.assign(f.size(), {});
      for (auto i = std::size_t{0}; i < f.size(); ++i) sum[i] += factor * f[i];
    }
  };
  add(out.spheroidal, a, f1.spheroidal);
  add(out.spheroidal, b, f2.spheroidal);
  add(out.toroidal, a, f1.toroidal);
  add(out.toroidal, b, f2.toroidal);
  return out;
}

}  // namespace sl3dpp::detail
