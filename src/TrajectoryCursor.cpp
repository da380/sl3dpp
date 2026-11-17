#include "sl3dpp/detail/TrajectoryCursor.hpp"

#include <algorithm>
#include <cassert>
#include <format>
#include <stdexcept>
#include <utility>

#include "sl3dpp/StepRecipe.hpp"

namespace sl3dpp::detail {

TrajectoryCursor::TrajectoryCursor(const ViscoelasticOperator& op,
                                   const ForwardTrajectory& trajectory,
                                   std::vector<double> dts, Scheme scheme,
                                   ChannelOf loadOf, ChannelOf tideOf)
    : op_(&op),
      trajectory_(&trajectory),
      dts_(std::move(dts)),
      scheme_(scheme),
      loadOf_(std::move(loadOf)),
      tideOf_(std::move(tideOf)) {
  const auto& states = trajectory.states;
  assert(!states.empty() && states.front().has_value());
  assert(trajectory.loads.size() == states.size() &&
         trajectory.tides.size() == states.size());
  assert(dts_.size() + 1 == states.size());
  if (IsMultiStage(scheme)) {
    // every step of nonzero length gives its inner stages' closures
    for (auto node = std::size_t{1}; node < states.size(); ++node) {
      const auto inner = RecipeOf(scheme, dts_[node - 1]).stages.size() - 1;
      const auto given = node < trajectory.stages.size()
                             ? trajectory.stages[node].size()
                             : std::size_t{0};
      if (given == inner) continue;
      throw std::invalid_argument(std::format(
          "the trajectory of a run under Sdirk2 gives the recorded "
          "coefficients of each step's first stage (ForwardTrajectory::"
          "stages, from the recording run's snapshots): the step ending at "
          "node {} has {} inner stage{}, the trajectory gives {}. A state "
          "inside a step, and one between kept ones, is the stage repeated "
          "under them",
          node, inner, inner == 1 ? "" : "s", given));
    }
  }
  first_ = states.size();  // nothing held
}

const ViscoelasticState& TrajectoryCursor::State(std::size_t node) {
  const auto& states = trajectory_->states;
  if (node >= states.size()) {
    throw std::out_of_range("the node is off the trajectory");
  }
  if (held_.empty() || node < first_) {
    // from the kept state at or below the node, holding what is stepped
    auto start = node;
    while (!states[start].has_value()) --start;
    held_.clear();
    inner_.clear();
    held_.push_back(Kept(start));
    first_ = start;
    Advance(node, true);
  } else if (node > Last()) {
    Advance(node, false);
  }
  while (Last() > node) held_.pop_back();
  return held_.back();
}

ViscoelasticState TrajectoryCursor::Kept(std::size_t node) const {
  auto out = *trajectory_->states[node];
  op_->PrimeStrain(out, loadOf_(trajectory_->loads[node]),
                   tideOf_(trajectory_->tides[node]));
  return out;
}

void TrajectoryCursor::Advance(std::size_t node, bool hold) {
  for (auto j = Last() + 1; j <= node; ++j) {
    if (trajectory_->states[j].has_value()) {
      held_.push_back(Kept(j));
    } else if (IsMultiStage(scheme_) && dts_[j - 1] != 0.0) {
      // the run's step from node j - 1, stage by stage, each finished
      // under its own recorded coefficients
      const auto recipe = RecipeOf(scheme_, dts_[j - 1]);
      held_.push_back(StepByRecipe(
          recipe, dts_[j - 1], held_.back(),
          [&](const ViscoelasticState& from, const RecipeStage& stage,
              double length, std::size_t index) {
            const auto last = index + 1 == recipe.stages.size();
            auto end = Stage(j, index, stage, length, last, from);
            // an inner stage's end, kept with the gap's states for Step
            if (hold && !last) inner_[j].push_back(end);
            return end;
          },
          [](std::span<const std::pair<double, const ViscoelasticState*>> terms,
             const RecipeStage&) { return CombinedState(terms); }));
    } else {
      // the run's step from node j - 1, finished under node j's
      // coefficients
      held_.push_back(op_->Step(held_.back(), dts_[j - 1], scheme_,
                                loadOf_(trajectory_->loads[j]),
                                tideOf_(trajectory_->tides[j]))
                          .first);
    }
    if (!hold) {
      while (held_.size() > 2) {
        held_.pop_front();
        ++first_;
      }
    }
  }
}

ViscoelasticState TrajectoryCursor::Stage(std::size_t node, std::size_t index,
                                          const RecipeStage& stage,
                                          double length, bool last,
                                          const ViscoelasticState& from) const {
  // (a backward Euler stage reads no start strain, so its opening
  // channels are not read)
  auto step = op_->StartStep(from, length, stage.family,
                             loadOf_(trajectory_->loads[node - 1]),
                             tideOf_(trajectory_->tides[node - 1]));
  if (last) {
    if (!trajectory_->sourceForces.empty()) {
      step.AddForce(trajectory_->sourceForces[node].spheroidal);
    }
    return step
        .Finish(loadOf_(trajectory_->loads[node]),
                tideOf_(trajectory_->tides[node]))
        .first;
  }
  const auto& closure = trajectory_->stages[node][index];
  if (closure.sourceForces) step.AddForce(closure.sourceForces->spheroidal);
  return step.Finish(loadOf_(closure.load), tideOf_(closure.tide)).first;
}

StageClosure ClosureOf(const ForwardTrajectory& trajectory,
                       const StageRef& ref) {
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

TangentRecord TangentRecordOf(std::span<const Evolution::Snapshot> tangent,
                              const StageRef& ref) {
  const auto& snap = tangent[ref.node];
  if (ref.Last()) {
    return {&*snap.internalState, &snap.loadCoefficients,
            &snap.tideCoefficients,
            snap.sourceForces ? &*snap.sourceForces : nullptr};
  }
  const auto& stage = snap.stages[ref.index];
  return {&*stage.internalState, &stage.loadCoefficients,
          &stage.tideCoefficients,
          stage.sourceForces ? &*stage.sourceForces : nullptr};
}

StageStates<ViscoelasticState> TangentStepStates(
    std::span<const Evolution::Snapshot> tangent, const StepRecipe& recipe,
    double dt, std::size_t node) {
  const auto& start = *tangent[node - 1].internalState;
  return StatesOfStages(
      recipe, dt, start, *tangent[node].internalState,
      [&](std::size_t index, const ViscoelasticState&) {
        return *tangent[node].stages[index].internalState;
      },
      [&](std::span<const std::pair<double, const ViscoelasticState*>> terms) {
        return CombinedWithZeroStrain(terms, start);
      });
}

void CheckRecordedTangent(std::span<const Evolution::Snapshot> tangent,
                          Scheme scheme, std::span<const double> times) {
  if (tangent.size() != times.size()) {
    throw std::invalid_argument(
        "the tangent-linear run must give one snapshot per time node");
  }
  for (auto node = std::size_t{0}; node < tangent.size(); ++node) {
    const auto& snap = tangent[node];
    if (!snap.internalState.has_value() || !snap.internalState->HasStrain()) {
      throw std::invalid_argument(
          "the tangent-linear run must be recorded at every node "
          "(TangentLinearSolver::Options::record)");
    }
    const auto inner = InnerStageCount(scheme, times, node);
    auto recorded = snap.stages.size() == inner;
    for (const auto& stage : snap.stages) {
      recorded = recorded && stage.internalState.has_value() &&
                 stage.internalState->HasStrain();
    }
    if (!recorded) {
      throw std::invalid_argument(std::format(
          "the tangent-linear run about a run under Sdirk2 must be recorded "
          "at every step's first stage too (TangentLinearSolver::Options::"
          "record, by a run under the same scheme): the step ending at node "
          "{} is not",
          node));
    }
  }
}

ViscoelasticState CombinedWithZeroStrain(
    std::span<const std::pair<double, const ViscoelasticState*>> terms,
    const ViscoelasticState& like) {
  auto out = CombinedState(terms);
  for (const auto& [l, d] : like.strain) {
    out.strain.emplace(l, std::vector<double>(d.size(), 0.0));
  }
  return out;
}

TrajectoryCursor::StepStates TrajectoryCursor::Step(std::size_t node) {
  if (node == 0 || node >= trajectory_->states.size()) {
    throw std::out_of_range("a step ends at a node after the first");
  }
  // the later state copied: the reference lasts until the next request
  auto end = State(node);
  const auto start = State(node - 1);
  const auto dt = dts_[node - 1];
  const auto recipe = RecipeOf(scheme_, dt);
  auto combine =
      [&](std::span<const std::pair<double, const ViscoelasticState*>> terms) {
        return CombinedWithZeroStrain(terms, start);
      };
  // the inner stages' ends: the gap's rebuild's where it took the step,
  // else the stages again
  auto rebuilt = std::vector<ViscoelasticState>{};
  if (const auto found = inner_.find(node); found != inner_.end()) {
    rebuilt = std::move(found->second);
  }
  inner_.erase(inner_.lower_bound(node), inner_.end());
  return StatesOfStages(
      recipe, dt, start, std::move(end),
      [&](std::size_t index, const ViscoelasticState& from) {
        if (index < rebuilt.size()) return std::move(rebuilt[index]);
        const auto& stage = recipe.stages[index];
        return Stage(node, index, stage, StageLength(stage, dt), false, from);
      },
      combine);
}

}  // namespace sl3dpp::detail
