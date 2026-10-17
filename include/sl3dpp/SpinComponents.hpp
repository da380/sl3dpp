#pragma once

/**
 * @file SpinComponents.hpp
 * @brief The degree <-> spin-component algebra of the 3D memory coupling
 * (docs/stage4.md §2–3), extracted from the Fortran's `memdot_bvec_all`
 * and kept in its ordering and normalisation.
 *
 * A trace-free symmetric tensor on the sphere is carried as five complex
 * components at spins (-2, -1, 0, +1, +2):
 *
 *   A[0] = A(--), A[1] = A(-0), A[2] = A(-+) = (1/2) A(00),
 *   A[3] = A(0+), A[4] = A(++).
 *
 * `DeviatoricComponents` gives the components of the deviatoric strain of
 * one (l, m) column from the radial scalars (U, V, W) and their r d_r
 * derivatives at a node; `ForceRowsFromComponents` gives the memory-force
 * rows the (weighted) components exert on the spheroidal (U, V) and
 * toroidal (W) degree equations. The caller supplies the quadrature: the
 * pointwise rows (u, v, w) pair with the test value at node q under the
 * weight jac w_q r_q mu_q, and the derivative rows (du, dv, dw) pair with
 * the test function's D(q, i) under the weight w_q r_q mu_q (one factor
 * of r is already inside the row).
 *
 * Reality: for a real tensor the grid component fields satisfy
 * A(++) = conj(A(--)), A(0+) = -conj(A(-0)) — the spin-1 pair
 * anti-conjugates under conj(Y^N_lm) = (-1)^(N-m) Y^(-N)_(l,-m) — and
 * A(-+) is real; the tests pin these down.
 *
 * In the 1D spheroidal limit (W = 0, uniform weights) the composition of
 * the two maps reproduces the degree-basis memory force of the ported
 * stepper exactly — the reduction identity the tests pin down constant by
 * constant, and the heart of stage 4's zero-lateral gate.
 */

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>

namespace sl3dpp {

/// The radial scalars of one (l, m) column at one node: U, V, W and
/// their r d_r derivatives.
struct DegreeNodeValues {
  std::complex<double> U{}, V{}, W{};
  std::complex<double> rdU{}, rdV{}, rdW{};
};

/// The five deviatoric strain components at the node; the +-2 rows vanish
/// at l = 1 and only the spin-0 component survives at l = 0.
[[nodiscard]] inline std::array<std::complex<double>, 5> DeviatoricComponents(
    int l, const DegreeNodeValues& v, double r) {
  constexpr std::complex<double> i(0.0, 1.0);
  const auto zeta2 = l * (l + 1.0);
  const auto zeta = std::sqrt(zeta2);
  auto A = std::array<std::complex<double>, 5>{};
  A[2] = (v.rdU - v.U + 0.5 * zeta2 * v.V) / (3.0 * r);
  if (l >= 1) {
    const auto tangential = v.rdV - v.V + v.U;
    const auto twist = v.W - v.rdW;
    A[1] = zeta * (tangential + i * twist) / (2.0 * std::numbers::sqrt2 * r);
    A[3] = zeta * (tangential - i * twist) / (2.0 * std::numbers::sqrt2 * r);
  }
  if (l >= 2) {
    const auto factor = 0.5 * zeta * std::sqrt(zeta2 - 2.0) / r;
    A[0] = factor * (v.V - i * v.W);
    A[4] = factor * (v.V + i * v.W);
  }
  return A;
}

/// The components of a tensor given by its spheroidal degree-basis
/// scalars (M, R, S) — an internal-variable state rather than a strain;
/// the map is the same as the strain's with (e_M, e_R, e_S) replaced by
/// (M, R, S) and no toroidal part.
[[nodiscard]] inline std::array<std::complex<double>, 5> ComponentsFromScalars(
    int l, std::complex<double> M, std::complex<double> R,
    std::complex<double> S, double r) {
  const auto zeta2 = l * (l + 1.0);
  const auto zeta = std::sqrt(zeta2);
  auto A = std::array<std::complex<double>, 5>{};
  A[2] = R / (3.0 * r);
  if (l >= 1) {
    A[1] = zeta * S / (2.0 * std::numbers::sqrt2 * r);
    A[3] = A[1];
  }
  if (l >= 2) {
    const auto factor = 0.5 * zeta * std::sqrt(zeta2 - 2.0) / r;
    A[0] = factor * M;
    A[4] = A[0];
  }
  return A;
}

/// The memory-force rows of (weighted) components at one node: see the
/// file comment for the quadrature weights the caller applies.
struct MemoryForceRows {
  std::complex<double> u{}, v{}, w{};     // pointwise rows
  std::complex<double> du{}, dv{}, dw{};  // derivative rows (carry one r)
};

[[nodiscard]] inline MemoryForceRows ForceRowsFromComponents(
    int l, std::span<const std::complex<double>, 5> A, double r) {
  constexpr std::complex<double> i(0.0, 1.0);
  const auto zeta2 = l * (l + 1.0);
  const auto zeta = std::sqrt(zeta2);
  const auto root = std::sqrt(std::max(zeta2 - 2.0, 0.0));
  MemoryForceRows rows;
  rows.u = std::numbers::sqrt2 * zeta * (A[1] + A[3]) - 4.0 * A[2];
  rows.v = zeta * root * (A[0] + A[4]) + 2.0 * zeta2 * A[2] -
           std::numbers::sqrt2 * zeta * (A[1] + A[3]);
  rows.w = i * (zeta * root * (A[0] - A[4]) -
                std::numbers::sqrt2 * zeta * (A[1] - A[3]));
  rows.du = 4.0 * r * A[2];
  rows.dv = std::numbers::sqrt2 * zeta * r * (A[1] + A[3]);
  rows.dw = i * std::numbers::sqrt2 * zeta * r * (A[1] - A[3]);
  return rows;
}

}  // namespace sl3dpp
