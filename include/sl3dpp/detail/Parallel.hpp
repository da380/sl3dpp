#pragma once

/**
 * @file detail/Parallel.hpp
 * @brief GSHTrans's parallel loops, taking sl3dpp's `parallel` flag.
 * Internal.
 *
 * @details sl3dpp threads through GSHTrans's forms (GSHTrans/Parallel.hpp);
 * the functions here only turn an `Options::parallel` flag into the
 * execution policy they take. `ForEach` is for element-wise passes over
 * large arrays (static blocks, the same block to each thread on every pass
 * over a range); `ForEachTask` for per-degree tasks of uneven cost (handed
 * out dynamically). The bodies must write only what their index owns. Inside
 * an active parallel region the loops run on the calling thread. A body may
 * throw: the exception rethrown after the loop is the one a sequential loop
 * would raise, the lowest failing index's, every index below it having run.
 */

#include <GSHTrans/Parallel.hpp>
#include <cstddef>
#include <utility>

namespace sl3dpp::detail {

/// The execution policy for an `Options::parallel` flag: OpenMP's thread
/// count when set, one thread otherwise.
inline GSHTrans::Execution PolicyOf(bool parallel) {
  return parallel ? GSHTrans::Execution::Parallel()
                  : GSHTrans::Execution::Sequential();
}

/// GSHTrans::ForEach: body(i) for i in [0, count), in static blocks.
template <typename Body>
void ForEach(bool parallel, std::size_t count, Body&& body) {
  GSHTrans::ForEach(PolicyOf(parallel), static_cast<std::ptrdiff_t>(count),
                    std::forward<Body>(body));
}

/// GSHTrans::ForEachTask: body(i) for i in [0, count), handed out
/// dynamically. The task counts here are degrees, so the body takes an
/// int; the conversion from GSHTrans's index is made here, once.
template <typename Body>
void ForEachTask(bool parallel, int count, Body&& body) {
  GSHTrans::ForEachTask(PolicyOf(parallel), count,
                        [&](std::ptrdiff_t i) { body(static_cast<int>(i)); });
}

}  // namespace sl3dpp::detail
