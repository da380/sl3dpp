#pragma once

/**
 * @file Measurements.hpp
 * @brief Measurement duals and derived data: the spatial duals that turn
 * a snapshot's fields into numbers, the `Measurement` entries that hand
 * the same functionals to the adjoint solvers, and the decay time of a
 * site's series as a fitted, kernel-ready datum.
 *
 * A measurement is a linear functional of one observation node's
 * outputs,
 *   J = a^2 int g xi dOmega + a^2 int g_u u dOmega
 *     + a^2 int g_phi phi dOmega + g_omega . omega,
 * read forward from a snapshot by `PairedValue`, and handed to
 * `AdjointSolver` / `AdjointSolver3D` as the duals of `Measurement`.
 * The duals here carry the 1 / a^2 of the physical surface integral, so
 * a cap mean comes out in the field's own unit and a coefficient dual
 * reads the field's expansion coefficient; everything is in the
 * caller's units (the drivers' solver units, with `a` the solver's
 * radius).
 *
 * The decay time follows Al-Attar & Tromp (2014, Appendix C), the
 * machinery of pyslfp's `fit_decay_time`: the series is fitted as
 * a e^(-s t) + b, the model times joining the parameters under Gaussian
 * penalties towards the nominal ones where the timing uncertainty is
 * non-zero, and the datum's exact sensitivity to the series comes from
 * differentiating the fit's stationarity, turning the fitted 1 / s into
 * per-node weights on the series — behind which every kernel driver
 * applies unchanged. Only the ratio of the two uncertainties shapes the
 * fit and the weights. The window must hold a decay: rebound series
 * look exponential near the centre of a former ice sheet and only after
 * the melting has ceased, and a window constraining no decay is
 * refused.
 */

#include <array>
#include <cstddef>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp {

// -- spatial duals ----------------------------------------------------------

/// A Gaussian cap in great-circle distance about `site`, of the given
/// `width` in radians, normalised to unit physical surface integral on a
/// sphere of radius `a`: paired with a field it reads the cap-weighted
/// mean, in the field's own unit.
[[nodiscard]] std::vector<double> CapMeanDual(const SurfaceGrid& grid, double a,
                                              GeographicPoint site,
                                              double width);

/// The band-limited point evaluation at `site`: the grid's reproducing
/// kernel sum_l (2l + 1) P_l(cos gamma) / (4 pi a^2) over the grid's
/// band. Paired with a band-limited field it reads its value at the
/// site exactly; against rough data it rings, and a cap mean is the
/// steadier datum.
[[nodiscard]] std::vector<double> PointDual(const SurfaceGrid& grid, double a,
                                            GeographicPoint site);

/// The dual reading a field's real orthonormal coefficient of degree l
/// and order m (the pyshtools convention of ShConventions.hpp: m >= 0
/// the cosine term, m < 0 the sine term of |m|): Y_lm / a^2, built by
/// synthesis so the pairing is the coefficient exactly. A GRACE-style
/// Stokes coefficient of the potential differs from the paired value by
/// a constant of the geodetic normalisation alone.
[[nodiscard]] std::vector<double> CoefficientDual(const SurfaceGrid& grid,
                                                  double a, int l, int m);

/// The physical surface integral a^2 int dual field dOmega: the forward
/// value of the measurement the dual defines.
[[nodiscard]] double PairedValue(const SurfaceGrid& grid, double a,
                                 std::span<const double> dual,
                                 std::span<const double> field);

// -- measurements -----------------------------------------------------------

/// The field a surface dual pairs with.
enum class FieldChannel { SeaLevel, Displacement, Potential };

/// The `Measurement` of one node whose named channel carries `dual`.
[[nodiscard]] Measurement FieldMeasurement(FieldChannel channel,
                                           std::vector<double> dual);

/// A component of the angular-velocity perturbation: X and Y drive
/// polar motion, Z the length of day.
enum class AngularVelocityComponent { X = 0, Y = 1, Z = 2 };

/// The `Measurement` reading one angular-velocity component.
[[nodiscard]] Measurement RotationMeasurement(
    AngularVelocityComponent component);

// -- the decay time ---------------------------------------------------------

/// The uncertainties the decay-time estimator is built with. Both are
/// required: only their ratio shapes the fit and the weights, and there
/// is no default that survives a change of units. A zero
/// `time_uncertainty` pins the model times to the nominal ones.
struct DecayTimeOptions {
  double time_uncertainty = 0.0;  ///< one-sigma timing error, >= 0
  double value_uncertainty = 0.0; ///< one-sigma series error, > 0
};

/// A fitted decay curve a e^(-s t) + b and the datum's exact
/// sensitivity to its data: `weights[j]` is the derivative of
/// `decay_time` with respect to the series value at the j-th time, from
/// the implicit differentiation of the fit's stationarity. Times, the
/// decay time and the weights share the caller's time unit;
/// `model_times` are the fitted times when the timing uncertainty freed
/// them, the nominal ones otherwise.
struct DecayTimeFit {
  double decay_time = 0.0;
  double amplitude = 0.0;
  double offset = 0.0;
  double misfit = 0.0;
  std::vector<double> model_times;
  std::vector<double> weights;
};

/// Fits `values` ~ a e^(-s times) + b and returns the decay time 1 / s
/// with its sensitivity to the values. `times` must increase, with at
/// least four entries, measured from any origin in the unit the decay
/// time is wanted in. Refused (std::invalid_argument) when the inputs
/// are malformed or the window constrains no decay, and
/// (std::runtime_error) when the fit does not reach stationarity.
[[nodiscard]] DecayTimeFit FitDecayTime(std::span<const double> times,
                                        std::span<const double> values,
                                        const DecayTimeOptions& options);

/// The decay time of a site's series as a measurement map for the
/// adjoint drivers: `series[j]` is the paired value of `dual` with the
/// channel's field at time node `nodes[j]` of the evolution (times
/// increasing along the window), and the returned map holds, at each
/// node, the fit's weight times the dual on the channel. `fit`, when
/// given, receives the fit itself.
[[nodiscard]] std::map<std::size_t, Measurement> DecayTimeMeasurements(
    std::span<const std::size_t> nodes, std::span<const double> times,
    std::span<const double> series, FieldChannel channel,
    std::span<const double> dual, const DecayTimeOptions& options,
    DecayTimeFit* fit = nullptr);

}  // namespace sl3dpp
