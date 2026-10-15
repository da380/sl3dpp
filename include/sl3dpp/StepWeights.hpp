#pragma once

/**
 * @file StepWeights.hpp
 * @brief The per-step weight algebra of the viscoelastic time-stepping
 * schemes: a port of pyslfp's `viscoelastic.weights`, itself a port of
 * AdGIA's `detail::ExponentialTrapezoidWeights`. The three must stay
 * algebraically identical for the cross-benchmarks to isolate spatial
 * discretisation.
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
 * to that interval. Negative h gives the EXACT inverse trapezoid step
 * (the update at -dt with the start and end strains swapped), which a
 * checkpointing reverse sweep uses; dissipation then amplifies roundoff
 * by exp(|h|).
 */

#include <cmath>
#include <stdexcept>

namespace sl3dpp {

/// The implicit/exponential time-stepping schemes.
enum class Scheme { ExponentialTrapezoid, BackwardEuler, ExponentialEuler };

/// The (e, alpha, beta) of one node and branch; also carries the
/// h-derivatives when asked for them.
struct TrapezoidWeights {
  double e = 1.0;
  double alpha = 0.0;
  double beta = 0.0;
};

namespace detail {

inline constexpr double kSeriesSwitch = 1e-3;

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

}  // namespace detail

/// The exponential trapezoid weights at h = dt / tau; negative h gives the
/// exact inverse step's. The effective-modulus weight is 1 - beta = phi1.
[[nodiscard]] inline TrapezoidWeights ExponentialTrapezoidWeights(double h) {
  const double e = std::exp(-h);
  const double phi1 = detail::Phi1(h, e);
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
  const double e = std::exp(-h);
  const double phi1 = detail::Phi1(h, e);
  const double dphi1 = detail::DPhi1(h, e, phi1);
  return {-e, dphi1 + e, -dphi1};
}

/// d/dh of the backward Euler weight: -1 / (1 + h)^2.
[[nodiscard]] inline double BackwardEulerWeightDerivative(double h) {
  if (h < 0.0) throw std::invalid_argument("h = dt / tau must be >= 0");
  return -1.0 / ((1.0 + h) * (1.0 + h));
}

}  // namespace sl3dpp
