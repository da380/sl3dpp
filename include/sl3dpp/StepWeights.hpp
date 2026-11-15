#pragma once

/**
 * @file StepWeights.hpp
 * @brief The per-step weight algebra of the viscoelastic time-stepping
 * schemes, algebraically identical to pyslfp's `viscoelastic.weights` and
 * AdGIA's `detail::ExponentialTrapezoidWeights`, so that the
 * cross-benchmarks isolate the spatial discretisation.
 *
 * Everything is a function of h = dt / tau, evaluated per node and per
 * branch. The exponential trapezoid update of an internal variable m
 * driven by a strain scalar d is
 *
 *   m^{n+1} = e m^n + alpha d^n + beta d^{n+1},
 *
 * with e = exp(-h), phi1 = (1 - e) / h, alpha = phi1 - e and
 * beta = 1 - phi1; phi1 is series-expanded below |h| = 1e-3, where the
 * direct quotient loses digits. The backward Euler update is
 * m^{n+1} = (m^n + h d^{n+1}) / (1 + h), and the exponential Euler update
 * is the trapezoid's with the strain frozen at the start. Eliminating
 * m^{n+1} from the step's equilibrium leaves an effective shear modulus
 * mu_inf + sum_k w_k mu_k with w = 1/(1 + h) for backward Euler and
 * w = 1 - beta = phi1 for the trapezoid.
 *
 * An elastic node is h = 0 (tau infinite, stored as itau = 0): e = 1,
 * alpha = beta = 0, both effective weights 1. The weights lie in (0, 1]
 * where evaluated; anything interpolating them between nodes must clamp
 * to that interval. Every step runs forwards, h >= 0: a step at -dt
 * would invert the update exactly, but amplify rounding by exp(h), so a
 * sweep that needs a state between two recorded ones rebuilds it by
 * forward steps from the earlier one (detail/TrajectoryCursor.hpp).
 *
 * Every scheme here but one steps in one stage with these weights. The
 * exception, Sdirk2, steps in two backward Euler stages (StepRecipe.hpp);
 * it has no weights of its own, and what takes a scheme's weights or its
 * one effective operator refuses it at a step of nonzero length.
 *
 * Why two stages. Where a memory point's h is large, the trapezoid's e
 * and alpha vanish and its update is backward Euler's, m+ = d+: the point
 * is slaved to the end strain. The slow, buoyancy-driven modes that carry
 * the observables involve such points (a weak layer, a fast branch), and
 * at steps long against their relaxation times the trapezoid then decays
 * those modes at backward Euler's rate, first order, most visibly after
 * an abrupt change of the forcing (a seismic onset, an unloading, the
 * continuous adjoint's impulse at a measurement). No one-solve scheme of
 * this form is both second order on those modes and damped at the stiff
 * points (second order forces e -> -1 there, an undamped sawtooth); two
 * implicit stages are: Sdirk2 is second order, L-stable and stiffly
 * accurate, at two solves of one effective operator a step. It is the
 * scheme for steps long against the shortest relaxation time (the forward
 * run after abrupt forcing, the continuous kernels); at steps short
 * against it the trapezoid, exact for a strain linear in a step, is the
 * closer at half the cost (docs/design.md §5).
 */

#include <cmath>
#include <stdexcept>

namespace sl3dpp {

/// The time-stepping schemes. The first three are one-stage schemes with
/// the weights below: the exponential trapezoid (second order, the
/// default), backward Euler (first order) and the exponential Euler
/// (explicit in the strain). Sdirk2 is Alexander's (1977) two-stage
/// singly diagonally implicit Runge-Kutta scheme, gamma = 1 - 1/sqrt 2:
/// second order, L-stable and stiffly accurate, its stages backward Euler
/// steps of length gamma dt (StepRecipe.hpp).
enum class Scheme {
  ExponentialTrapezoid,
  BackwardEuler,
  ExponentialEuler,
  Sdirk2
};

/// The (e, alpha, beta) of one node and branch; also carries the
/// h-derivatives when asked for them.
struct TrapezoidWeights {
  double e = 1.0;
  double alpha = 0.0;
  double beta = 0.0;
};

namespace detail {

inline constexpr auto kSeriesSwitch = 1e-3;

/// (1 - exp(-h)) / h: the quotient away from zero, the alternating series
/// below the switch, exact at h = 0.
inline double Phi1(double h, double e) {
  if (std::abs(h) < kSeriesSwitch) {
    return 1.0 - h / 2.0 + h * h / 6.0 - h * h * h / 24.0 +
           h * h * h * h / 120.0;
  }
  return (1.0 - e) / h;
}

/// d phi1 / dh = (e - phi1) / h, likewise.
inline double DPhi1(double h, double e, double phi1) {
  if (std::abs(h) < kSeriesSwitch) {
    return -0.5 + h / 3.0 - h * h / 8.0 + h * h * h / 30.0 -
           h * h * h * h / 144.0;
  }
  return (e - phi1) / h;
}

/// d^2 phi1 / dh^2 = -(e + 2 dphi1) / h, likewise.
inline double D2Phi1(double h, double e, double dphi1) {
  if (std::abs(h) < kSeriesSwitch) {
    return 1.0 / 3.0 - h / 4.0 + h * h / 10.0 - h * h * h / 36.0 +
           h * h * h * h / 168.0;
  }
  return -(e + 2.0 * dphi1) / h;
}

}  // namespace detail

/// The exponential trapezoid weights at h = dt / tau >= 0. The
/// effective-modulus weight is 1 - beta = phi1.
[[nodiscard]] inline TrapezoidWeights ExponentialTrapezoidWeights(double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  const auto e = std::exp(-h);
  const auto phi1 = detail::Phi1(h, e);
  return {e, phi1 - e, 1.0 - phi1};
}

/// The backward Euler weight 1 / (1 + h), h >= 0: both the effective
/// modulus weight and the factor on the start state in the memory force.
[[nodiscard]] inline double BackwardEulerWeight(double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  return 1.0 / (1.0 + h);
}

/// (de, dalpha, dbeta), the h-derivatives of the trapezoid weights at
/// h >= 0; the effective weight's derivative is dphi1 = -dbeta. A
/// viscosity sensitivity pairs these with dh = dt d(1/tau).
[[nodiscard]] inline TrapezoidWeights ExponentialTrapezoidWeightDerivatives(
    double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  const auto e = std::exp(-h);
  const auto phi1 = detail::Phi1(h, e);
  const auto dphi1 = detail::DPhi1(h, e, phi1);
  return {-e, dphi1 + e, -dphi1};
}

/// d/dh of the backward Euler weight: -1 / (1 + h)^2.
[[nodiscard]] inline double BackwardEulerWeightDerivative(double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  return -1.0 / ((1.0 + h) * (1.0 + h));
}

/// (d2e, d2alpha, d2beta), the second h-derivatives of the trapezoid
/// weights at h >= 0: d2e = e, d2alpha = d2phi1 - e, d2beta = -d2phi1. A
/// second derivative in the viscosity needs these at fixed dt.
[[nodiscard]] inline TrapezoidWeights ExponentialTrapezoidWeightSecondDerivatives(
    double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  const auto e = std::exp(-h);
  const auto phi1 = detail::Phi1(h, e);
  const auto dphi1 = detail::DPhi1(h, e, phi1);
  const auto d2phi1 = detail::D2Phi1(h, e, dphi1);
  return {e, d2phi1 - e, -d2phi1};
}

/// d^2/dh^2 of the backward Euler weight: 2 / (1 + h)^3.
[[nodiscard]] inline double BackwardEulerWeightSecondDerivative(double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  return 2.0 / ((1.0 + h) * (1.0 + h) * (1.0 + h));
}

}  // namespace sl3dpp
