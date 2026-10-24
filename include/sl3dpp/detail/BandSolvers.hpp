#pragma once

/**
 * @file detail/BandSolvers.hpp
 * @brief Banded Cholesky and banded LU factorisations and solves, in
 * LAPACK's band storage. Internal.
 *
 * The per-degree systems are factorised and solved inside the library's
 * OpenMP loops, one degree per thread, so these must not call a BLAS:
 * OpenBLAS's pthread build crashes when many threads call it at once
 * (its fixed table of memory buffers runs out; OpenBLAS issue #5958), and
 * its OpenMP build livelocks when nested. The routines follow LAPACK's
 * unblocked reference algorithms (dpbtf2, dgbtf2 and the solves of dpbtrs,
 * dgbtrs), which LAPACK itself runs below a bandwidth of 32, with the same
 * storage and the same pivot choices, so the results agree with LAPACK's to
 * rounding. The band is narrow (3 nGll - 1 at most, 14 at the default five
 * GLL nodes), where BLAS calls cost more than the arithmetic they do.
 */

#include <span>

namespace sl3dpp::detail {

/// Factorises a symmetric positive definite band matrix as A = U^T U, in
/// place, as LAPACK's dpbtrf with uplo "U". `ab` holds the upper triangle
/// by columns, A(i, j) at ab[j (kd + 1) + kd + i - j] for
/// max(0, j - kd) <= i <= j, and receives U in the same places. Returns 0,
/// or k + 1 when the leading minor of order k + 1 is not positive (or is
/// NaN), `ab` then partly overwritten.
int BandCholeskyFactor(std::span<double> ab, int n, int kd);

/// Solves A X = B from BandCholeskyFactor's U, as dpbtrs: B is n by nrhs,
/// column-major, and is overwritten by X.
void BandCholeskySolve(std::span<const double> ab, int n, int kd,
                       std::span<double> b, int nrhs);

/// Factorises a square band matrix with kl sub- and ku superdiagonals as
/// A = P L U with partial pivoting, in place, as LAPACK's dgbtrf. `ab` has
/// 2 kl + ku + 1 rows, A(i, j) at ab[j (2 kl + ku + 1) + kl + ku + i - j],
/// its first kl rows being room for U's fill-in. Row j was interchanged
/// with row pivots[j] (zero-based, unlike LAPACK's). Returns 0, or k + 1
/// when U(k, k) is exactly zero, the factorisation completed regardless.
int BandLuFactor(std::span<double> ab, int n, int kl, int ku,
                 std::span<int> pivots);

/// Solves A X = B from BandLuFactor's factors, as dgbtrs with trans "N": B
/// is n by nrhs, column-major, and is overwritten by X.
void BandLuSolve(std::span<const double> ab, int n, int kl, int ku,
                 std::span<const int> pivots, std::span<double> b, int nrhs);

}  // namespace sl3dpp::detail
