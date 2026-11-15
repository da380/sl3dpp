#include "sl3dpp/detail/TrajectoryCursor.hpp"

#include <algorithm>
#include <cassert>
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
  RefuseGapsUnderMultiStage(
      scheme, std::ranges::any_of(
                  states, [](const auto& state) { return !state.has_value(); }));
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

}  // namespace sl3dpp::detail
