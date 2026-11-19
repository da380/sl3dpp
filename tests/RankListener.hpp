#pragma once

// A test over several ranks pairs its barriers and collectives rank by
// rank. A rank that leaves a test's body early (an exception thrown, or a
// fatal assertion) makes none of the calls the others then wait in, and
// its next test's calls pair with theirs: the program hangs, or fails far
// from the cause. This listener ends every rank at the first fatal failure
// on any, the failing rank saying what failed.

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace sl3dpp::test {

class AbortEveryRankOnAFatalFailure : public ::testing::EmptyTestEventListener {
  // (the test's name is kept from its start: a part's result is reported
  // under googletest's own lock, which asking it for the running test
  // would take again)
  void OnTestStart(const ::testing::TestInfo& test) override {
    name_ = std::string{test.test_suite_name()} + "." + test.name();
  }
  void OnTestPartResult(const ::testing::TestPartResult& result) override {
    if (!result.fatally_failed()) return;
    auto rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "[rank " << rank << "] fatal failure in " << name_ << "\n"
              << (result.file_name() != nullptr ? result.file_name() : "")
              << ":" << result.line_number() << "\n"
              << result.message() << "\n"
              << "every rank is ended: the other ranks would wait on this "
                 "one's collectives"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::string name_;
};

/// Appends the listener; on every rank, after InitGoogleTest.
inline void EndEveryRankOnAFatalFailure() {
  ::testing::UnitTest::GetInstance()->listeners().Append(
      new AbortEveryRankOnAFatalFailure);
}

/// Holds the world's rank count to the one the launcher was asked for
/// (SL3DPP_TEST_RANKS, which the tests' CMake sets beside mpiexec's count),
/// and ends the program where they differ. An mpiexec that does not match
/// the MPI library starts each process as a world of one rank, and every
/// test would then pass without the ranks it is there to exercise. After
/// MPI_Init; a program started without the variable is not held.
inline void RequireTheRanksAskedFor() {
  const auto* asked = std::getenv("SL3DPP_TEST_RANKS");
  if (asked == nullptr) return;
  auto size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size == std::atoi(asked)) return;
  std::cerr << "the launcher was asked for " << asked
            << " ranks and MPI_COMM_WORLD holds " << size
            << ": mpiexec and the MPI library do not match" << std::endl;
  MPI_Abort(MPI_COMM_WORLD, 1);
}

}  // namespace sl3dpp::test
