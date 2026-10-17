#pragma once

/**
 * @file detail/ParallelFor.hpp
 * @brief An OpenMP loop over independent tasks that is safe to throw from.
 * Internal.
 */

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
  for (int i = 0; i < count; ++i) {
    try {
      body(i);
    } catch (...) {
#pragma omp critical(sl3dpp_parallel_for)
      if (!failure) failure = std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
}

}  // namespace sl3dpp::detail
