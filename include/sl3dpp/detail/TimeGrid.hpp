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
#include <type_traits>
#include <utility>

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

}  // namespace sl3dpp::detail
