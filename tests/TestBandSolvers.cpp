// The banded factorisations and solves against LAPACK's, which they
// transcribe: the same factors, pivots and solutions to rounding, and the
// same failure indices.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "sl3dpp/detail/BandSolvers.hpp"

extern "C" {
void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab,
             const int* ldab, int* info);
void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs,
             const double* ab, const int* ldab, double* b, const int* ldb,
             int* info);
void dgbtrf_(const int* m, const int* n, const int* kl, const int* ku,
             double* ab, const int* ldab, int* ipiv, int* info);
void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku,
             const int* nrhs, const double* ab, const int* ldab,
             const int* ipiv, double* b, const int* ldb, int* info);
}

namespace {

using sl3dpp::detail::BandCholeskyFactor;
using sl3dpp::detail::BandCholeskySolve;
using sl3dpp::detail::BandLuFactor;
using sl3dpp::detail::BandLuSolve;

constexpr auto kTol = 1e-12;

std::vector<double> Uniform(std::size_t count, unsigned seed) {
  auto engine = std::mt19937{seed};
  auto uniform = std::uniform_real_distribution<double>{-1.0, 1.0};
  auto values = std::vector<double>(count);
  for (auto& v : values) v = uniform(engine);
  return values;
}

/// The largest difference relative to the largest magnitude of `want`.
double RelativeDifference(const std::vector<double>& got,
                          const std::vector<double>& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

/// A symmetric band matrix in upper storage, the diagonal raised by
/// `shift` (positive definite once shift exceeds 2 kd).
std::vector<double> SymmetricBand(int n, int kd, double shift, unsigned seed) {
  const auto ld = static_cast<std::size_t>(kd) + 1;
  auto ab = Uniform(ld * n, seed);
  for (auto j = 0; j < n; ++j) {
    ab[static_cast<std::size_t>(j) * ld + kd] += shift;
    for (auto i = j - kd; i < 0; ++i) {  // outside the matrix
      ab[static_cast<std::size_t>(j) * ld + kd + i - j] = 0.0;
    }
  }
  return ab;
}

/// A general band matrix in LU storage (kl rows of room on top), its
/// diagonal scaled by `diagonal` so that small values force interchanges.
std::vector<double> GeneralBand(int n, int kl, int ku, double diagonal,
                                unsigned seed) {
  const auto ld = static_cast<std::size_t>(2 * kl + ku + 1);
  const auto kv = kl + ku;
  auto values = Uniform(ld * n, seed);
  auto ab = std::vector<double>(ld * n, 0.0);
  for (auto j = 0; j < n; ++j) {
    for (auto i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); ++i) {
      const auto at = static_cast<std::size_t>(j) * ld + kv + i - j;
      ab[at] = values[at] * (i == j ? diagonal : 1.0);
    }
  }
  return ab;
}

void ExpectCholeskyMatchesLapack(int n, int kd, int nrhs, unsigned seed) {
  auto ours = SymmetricBand(n, kd, 2.0 * kd + 1.0, seed);
  auto lapack = ours;
  const auto ldab = kd + 1;
  auto info = 0;
  dpbtrf_("U", &n, &kd, lapack.data(), &ldab, &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(BandCholeskyFactor(ours, n, kd), 0);
  EXPECT_LT(RelativeDifference(ours, lapack), kTol);

  auto x = Uniform(static_cast<std::size_t>(n) * nrhs, seed + 1);
  auto want = x;
  dpbtrs_("U", &n, &kd, &nrhs, lapack.data(), &ldab, want.data(), &n, &info);
  ASSERT_EQ(info, 0);
  BandCholeskySolve(ours, n, kd, x, nrhs);
  EXPECT_LT(RelativeDifference(x, want), kTol);
}

void ExpectLuMatchesLapack(int n, int kl, int ku, int nrhs, unsigned seed) {
  auto ours = GeneralBand(n, kl, ku, 0.05, seed);
  auto lapack = ours;
  const auto ldab = 2 * kl + ku + 1;
  auto pivots = std::vector<int>(n), ipiv = std::vector<int>(n);
  auto info = 0;
  dgbtrf_(&n, &n, &kl, &ku, lapack.data(), &ldab, ipiv.data(), &info);
  ASSERT_EQ(info, 0);
  ASSERT_EQ(BandLuFactor(ours, n, kl, ku, pivots), 0);
  auto interchanges = 0;
  for (auto j = 0; j < n; ++j) {
    EXPECT_EQ(pivots[j], ipiv[j] - 1) << "column " << j;
    interchanges += pivots[j] != j;
  }
  EXPECT_GT(interchanges, 0);  // the test exercises the pivoting
  EXPECT_LT(RelativeDifference(ours, lapack), kTol);

  auto x = Uniform(static_cast<std::size_t>(n) * nrhs, seed + 1);
  auto want = x;
  dgbtrs_("N", &n, &kl, &ku, &nrhs, lapack.data(), &ldab, ipiv.data(),
          want.data(), &n, &info);
  ASSERT_EQ(info, 0);
  BandLuSolve(ours, n, kl, ku, pivots, x, nrhs);
  EXPECT_LT(RelativeDifference(x, want), kTol);
}

}  // namespace

TEST(BandSolvers, CholeskyMatchesLapack) {
  ExpectCholeskyMatchesLapack(200, 14, 4, 1);
  ExpectCholeskyMatchesLapack(60, 5, 1, 2);
  ExpectCholeskyMatchesLapack(4, 7, 2, 3);  // the band wider than the matrix
  ExpectCholeskyMatchesLapack(1, 0, 1, 4);
}

TEST(BandSolvers, LuMatchesLapack) {
  ExpectLuMatchesLapack(200, 14, 14, 4, 5);  // the operators' shape
  ExpectLuMatchesLapack(80, 3, 6, 2, 6);
  ExpectLuMatchesLapack(80, 6, 3, 1, 7);
  ExpectLuMatchesLapack(6, 9, 9, 2, 8);  // the band wider than the matrix
}

// The failure index is LAPACK's: the order of the first leading minor that
// is not positive.
TEST(BandSolvers, CholeskyReportsTheIndefiniteMinor) {
  const auto n = 40, kd = 4;
  auto ours = SymmetricBand(n, kd, 2.0 * kd + 1.0, 9);
  ours[static_cast<std::size_t>(17) * (kd + 1) + kd] = -1.0;
  auto lapack = ours;
  const auto ldab = kd + 1;
  auto info = 0;
  dpbtrf_("U", &n, &kd, lapack.data(), &ldab, &info);
  EXPECT_GT(info, 0);
  EXPECT_EQ(BandCholeskyFactor(ours, n, kd), info);
}

TEST(BandSolvers, LuReportsTheZeroPivot) {
  const auto n = 30, kl = 3, ku = 3;
  auto ours = GeneralBand(n, kl, ku, 1.0, 10);
  const auto ld = static_cast<std::size_t>(2 * kl + ku + 1);
  constexpr auto zero = 11;  // the column that vanishes
  for (auto i = zero - ku; i <= zero + kl; ++i) {
    ours[static_cast<std::size_t>(zero) * ld + kl + ku + i - zero] = 0.0;
  }
  auto lapack = ours;
  const auto ldab = static_cast<int>(ld);
  auto pivots = std::vector<int>(n), ipiv = std::vector<int>(n);
  auto info = 0;
  dgbtrf_(&n, &n, &kl, &ku, lapack.data(), &ldab, ipiv.data(), &info);
  EXPECT_GT(info, 0);
  EXPECT_EQ(BandLuFactor(ours, n, kl, ku, pivots), info);
}
