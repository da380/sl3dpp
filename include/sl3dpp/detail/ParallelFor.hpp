#pragma once

/**
 * @file detail/ParallelFor.hpp
 * @brief OpenMP loops: over independent tasks, safe to throw from; and
 * over the elements of large arrays. Internal.
 */

#include <cstddef>
#include <exception>

namespace sl3dpp::detail {

/// Runs body(i) for i in [0, count): under OpenMP with a dynamic schedule
/// when `parallel`, in order otherwise. An exception thrown by a body is
/// caught inside the region (one leaving it would terminate the program)
/// and the first one caught is rethrown after the loop. The bodies must
/// be independent: each writes only its own outputs.
template <typename Body>
void ParallelFor(int count, bool parallel, Body&& body) {
  std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic) if (parallel)
  for (auto i = 0; i < count; ++i) {
    try {
      body(i);
    } catch (...) {
#pragma omp critical(sl3dpp_parallel_for)
      if (!failure) failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}

/// Runs body(i) for i in [0, count): under OpenMP's static schedule when
/// `parallel`, in order otherwise. For element-wise passes over large
/// arrays: every pass over the same range hands each thread the same
/// block. The body must not throw and must write only element i's
/// outputs, so the result does not depend on the thread count.
template <typename Body>
void ParallelElements(std::size_t count, bool parallel, Body&& body) {
#pragma omp parallel for schedule(static) if (parallel)
  for (auto i = std::size_t{0}; i < count; ++i) body(i);
}

}  // namespace sl3dpp::detail
