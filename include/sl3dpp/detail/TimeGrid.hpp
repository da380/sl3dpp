#pragma once

/**
 * @file detail/TimeGrid.hpp
 * @brief The time grid every run and every derivative run steps over.
 * Internal.
 *
 * Non-decreasing, with a node repeated at most once: times[k + 1] ==
 * times[k] is a jump of the forcing at that time, node k its left limit and
 * node k + 1 its right, joined by a zero-length step (the static elastic
 * solve with the internal variables held, StepWeights.hpp at h = 0).
 * A run's first node is the right limit at times[0] already (the
 * instantaneous elastic response to the forcing there), so a repeat there
 * is refused, unless the grid continues a run from a checkpoint, whose
 * own node may be a left limit.
 */

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sl3dpp/PointSource.hpp"

namespace sl3dpp::detail {

/// Refuses a time grid that is empty, decreasing anywhere, holds a time
/// three times, or (unless `resumed`) repeats its first node.
inline void CheckTimeGrid(std::span<const double> times, bool resumed = false) {
  if (times.empty()) {
    throw std::invalid_argument("times must hold at least one entry");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] < times[i - 1]) {
      throw std::invalid_argument(
          "times must be non-decreasing (a node repeated once is a jump of "
          "the forcing: its left limit, then its right)");
    }
    if (times[i] == times[i - 1] && i >= 2 && times[i - 1] == times[i - 2]) {
      throw std::invalid_argument(
          "a time is repeated at most once: a jump of the forcing is its "
          "left limit and its right");
    }
  }
  if (!resumed && times.size() > 1 && times[1] == times[0]) {
    throw std::invalid_argument(
        "a run's first node is not repeated: it is the right limit at its "
        "time already, the instantaneous elastic response to the forcing "
        "there");
  }
}

/// Whether node n is the first of a repeated pair, the left limit of a
/// jump of the forcing at its time.
[[nodiscard]] inline bool IsLeftLimit(std::span<const double> times,
                                      std::size_t n) {
  return n + 1 < times.size() && times[n + 1] == times[n];
}

/// A forcing term at node n: its left limit `before` at the first node of
/// a repeated pair where one is given, `plain` elsewhere (and there, the
/// term continuous at that time); `fallback` where `plain` is absent. The
/// one place a march decides which side of a jump a node reads.
template <class Callback, class Value = std::invoke_result_t<Callback, double>>
[[nodiscard]] Value AtNode(std::span<const double> times, std::size_t n,
                           const Callback& plain, const Callback& before,
                           Value fallback = {}) {
  const auto t = times[n];
  if (before && IsLeftLimit(times, n)) return before(t);
  return plain ? plain(t) : std::move(fallback);
}

/// Refuses a forcing's left limit given without its term (Evolution::
/// Forcing, Evolution3D::Forcing), which would be read at the first node
/// of a pair alone, and the ice thickness's under frozen shorelines
/// (`migrating` false), whose ice is a direct load.
template <class Forcing>
void CheckLeftLimits(const Forcing& forcing, bool migrating) {
  const auto refuse = [](bool given, bool term, const char* name) {
    if (given && !term) {
      throw std::invalid_argument(std::string{name} +
                                  " is a left limit of a term the forcing "
                                  "does not give");
    }
  };
  refuse(static_cast<bool>(forcing.directLoadBefore),
         static_cast<bool>(forcing.directLoad), "directLoadBefore");
  refuse(static_cast<bool>(forcing.directLoadOutsideWaterBudgetBefore),
         static_cast<bool>(forcing.directLoadOutsideWaterBudget),
         "directLoadOutsideWaterBudgetBefore");
  refuse(static_cast<bool>(forcing.solidSurfaceChangeBefore),
         static_cast<bool>(forcing.solidSurfaceChange),
         "solidSurfaceChangeBefore");
  refuse(static_cast<bool>(forcing.tideBefore),
         static_cast<bool>(forcing.tide), "tideBefore");
  refuse(static_cast<bool>(forcing.sourcesBefore),
         static_cast<bool>(forcing.sources), "sourcesBefore");
  refuse(static_cast<bool>(forcing.stressGlutsBefore),
         static_cast<bool>(forcing.stressGluts), "stressGlutsBefore");
  if (forcing.iceThicknessBefore && !migrating) {
    throw std::invalid_argument(
        "iceThicknessBefore is the left limit of a migrating run's ice "
        "thickness; under frozen shorelines the ice is a direct load, its "
        "left limit directLoadBefore");
  }
}

/// A forcing term's value at fraction c of a step from its values at the
/// step's two nodes, (1 - c) of the start node's and c of the end node's:
/// fields and coefficient vectors entry by entry (an empty one is none,
/// zero), interior sources as the start node's with their moments times
/// 1 - c beside the end node's with theirs times c (the forces are linear
/// in the moments, so that is the same combination of the forces).
template <class T>
[[nodiscard]] std::vector<T> BetweenNodes(std::vector<T> start,
                                          std::vector<T> end, double c) {
  if (start.empty() && end.empty()) return end;
  if (end.empty()) end.assign(start.size(), T{});
  if (!start.empty() && start.size() != end.size()) {
    throw std::invalid_argument(
        "a forcing term's values at the two nodes of a step are of "
        "different sizes");
  }
  for (auto i = std::size_t{0}; i < end.size(); ++i) {
    end[i] *= c;
    if (!start.empty()) end[i] += (1.0 - c) * start[i];
  }
  return end;
}
[[nodiscard]] inline std::vector<PointSource> BetweenNodes(
    std::vector<PointSource> start, std::vector<PointSource> end, double c) {
  auto scale = [](PointSource& source, double a) {
    auto& m = source.moment;
    for (auto* v : {&m.rr, &m.tt, &m.pp, &m.rt, &m.rp, &m.tp}) *v *= a;
  };
  for (auto& source : start) scale(source, 1.0 - c);
  for (auto& source : end) scale(source, c);
  start.insert(start.end(), end.begin(), end.end());
  return start;
}
[[nodiscard]] inline std::vector<StressGlut> BetweenNodes(
    std::vector<StressGlut> start, std::vector<StressGlut> end, double c) {
  auto scale = [](StressGlut& glut, double a) {
    auto& m = glut.moment;
    for (auto* field : {&m.rr, &m.tt, &m.pp, &m.rt, &m.rp, &m.tp}) {
      for (auto& v : *field) v *= a;
    }
  };
  for (auto& glut : start) scale(glut, 1.0 - c);
  for (auto& glut : end) scale(glut, c);
  start.insert(start.end(), std::make_move_iterator(end.begin()),
               std::make_move_iterator(end.end()));
  return start;
}

/// A forcing term at a stage of the step ending at node n: at c = 1 the
/// node's read (AtNode, its left limit at the first node of a repeated
/// pair). A stage that ends inside the step (c < 1) reads no time of its
/// own: it takes (1 - c) of the term at the step's start node, as the
/// step from there reads it (`plain` at that node's time: the right limit
/// at a repeated pair's second node), and c of the term at its end node,
/// as the node reads it (BetweenNodes). A run therefore depends on its
/// forcing through the values at its nodes alone, under every scheme, and
/// the derivative runs, which hold the forcing by those values, take the
/// same combination (detail/StepStages.hpp). The one place a stage
/// decides what it reads.
template <class Callback, class Value = std::invoke_result_t<Callback, double>>
[[nodiscard]] Value AtStage(std::span<const double> times, std::size_t n,
                            double c, const Callback& plain,
                            const Callback& before, Value fallback = {}) {
  if (c == 1.0) return AtNode(times, n, plain, before, std::move(fallback));
  if (!plain) return fallback;
  return BetweenNodes(plain(times[n - 1]), AtNode(times, n, plain, before), c);
}

}  // namespace sl3dpp::detail
