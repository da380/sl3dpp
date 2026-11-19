#pragma once

// A test over several ranks pairs its barriers and collectives rank by
// rank. A rank that leaves a test's body early (an exception thrown, or a
// fatal assertion) makes none of the calls the others then wait in, and
// its next test's calls pair with theirs: the program hangs, or fails far
// from the cause. This listener ends every rank at the first fatal failure
// on any, the failing rank saying what failed.

#include <gtest/gtest.h>
#include <mpi.h>

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

}  // namespace sl3dpp::test
