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
enum class FieldChannel {
  SeaLevel,
  Displacement,
  Potential,
  /// The potential V of the tangential displacement.
  TangentialPotential
};

/// The `Measurement` of one node whose named channel carries `dual`.
[[nodiscard]] Measurement FieldMeasurement(FieldChannel channel,
                                           std::vector<double> dual);

// -- displacement measurements ----------------------------------------------
//
// A measurement of the surface displacement is a vector weight w on the
// surface, J = a^2 int w . u dOmega, with u = U r + grad_1 V - r x grad_1 W
// (the gradient that of the unit sphere, V and W lengths). Its radial
// part is the dual of U; its tangential part gives the duals of V and W.
// In the continuous sweeps the weight is an adjoint traction, -w. The
// weight is given as a vector field on the grid, or, to read the
// displacement at one point along one direction, by the point and the
// direction.
//
// The frame. A rigid rotation of a solid region costs no strain, so the
// toroidal degree 1 of a solution is fixed only by the solver's gauge
// (ToroidalOperator), and a horizontal displacement is defined only
// relative to a frame. The measurements here read the displacement in
// the frame of NO NET ROTATION OF THE SURFACE: the degree-1 part of W at
// the surface, the rigid rotation u = c x r, is left out of their duals,
// so that they do not depend on the gauge and their adjoint tractions
// apply no torque. NetRotation, WithoutNetRotation and
// RigidRotationPotential move a toroidal potential between frames.
// BaselineMeasurement and AngleMeasurement read changes of length and of
// angle between sites, which no rigid motion alters and no frame enters.

/// A vector field on the surface by its components to the east, to the
/// north and upwards, each a field on the grid; an empty component is
/// absent.
struct SurfaceVectorField {
  std::vector<double> east;
  std::vector<double> north;
  std::vector<double> up;
};

/// The displacement u = U r + grad_1 V - r x grad_1 W as a vector field,
/// from the fields U, V and W on the grid (an empty one is absent). The
/// tangential part goes through the canonical components
/// u^(+-) = sqrt(l (l + 1) / 2) (V_lm +- i W_lm) of the generalised
/// spherical harmonics of upper index +-1 (Phinney & Burridge 1973;
/// Dahlen & Tromp 1998, appendix C), so the grid must carry upper index 1
/// (SurfaceGrid::Options::nmax).
[[nodiscard]] SurfaceVectorField DisplacementField(
    const SurfaceGrid& grid, std::span<const double> U,
    std::span<const double> V, std::span<const double> W);

/// The rigid rotation in a toroidal potential W on the grid: the vector
/// c, in W's unit, of its degree-1 part c . r, whose displacement
/// -r x grad_1 (c . r) is c x r (r the unit radial vector; c over the
/// radius is the rotation's angle). Cartesian components, z the polar
/// axis and x through the zero of longitude.
[[nodiscard]] std::array<double, 3> NetRotation(const SurfaceGrid& grid,
                                                std::span<const double> W);

/// The toroidal potential c . r of the rigid rotation u = c x r, on the
/// grid: added to a W it moves the displacement into the frame that
/// turns by -c against the one it was in.
[[nodiscard]] std::vector<double> RigidRotationPotential(
    const SurfaceGrid& grid, std::array<double, 3> c);

/// W less its rigid rotation: the toroidal potential in the frame of no
/// net rotation of the surface, in which the measurements below read
/// the displacement.
[[nodiscard]] std::vector<double> WithoutNetRotation(
    const SurfaceGrid& grid, std::span<const double> W);

/// The `Measurement` of the displacement against a vector weight w,
/// J = a^2 int w . u dOmega, the displacement taken in the frame of no
/// net rotation of the surface (the dual of W has no degree 1): the
/// upward component is the dual of U, and
/// the tangential ones, expanded in their canonical components
/// w^(+-)_lm, give the duals of V and W algebraically, with coefficients
/// sqrt(l (l + 1) / 2) (w^+ + w^-)_lm and -i sqrt(l (l + 1) / 2)
/// (w^+ - w^-)_lm. The grid must carry upper index 1. The weight carries
/// the 1 / a^2 of the physical surface integral, as the scalar duals do.
[[nodiscard]] Measurement DisplacementMeasurement(
    const SurfaceGrid& grid, const SurfaceVectorField& weight);

/// A direction at a point of the surface, by its components to the
/// east, to the north and upwards; it need not be a unit vector (a
/// line-of-sight direction is one, and the three axes read the
/// components).
struct SurfaceDirection {
  double east = 0.0;
  double north = 0.0;
  double up = 0.0;
};

/// The `Measurement` of the displacement at `site` along `direction`,
/// band-limited as PointDual is: paired with a band-limited displacement
/// it reads direction . u at the site exactly, in the displacement's
/// unit. The upward part is PointDual on the radial displacement; the
/// horizontal part is the derivative of the grid's reproducing kernel
/// along the direction, on V and (turned a quarter) on W, the latter
/// from degree 2: the displacement is read in the frame of no net
/// rotation of the surface.
[[nodiscard]] Measurement PointDisplacementMeasurement(
    const SurfaceGrid& grid, double a, GeographicPoint site,
    SurfaceDirection direction);

/// The `Measurement` of the change of the straight-line distance between
/// two sites of the surface, e . (u(to) - u(from)) with e the unit
/// vector from the one to the other, in the displacement's unit. No
/// rigid motion changes it, so it needs no frame. The sites must differ.
[[nodiscard]] Measurement BaselineMeasurement(const SurfaceGrid& grid, double a,
                                              GeographicPoint from,
                                              GeographicPoint to);

/// The `Measurement` of the change of the angle at `vertex` between the
/// straight lines to `first` and to `second`, in radians when the
/// displacement is in the unit of `a`. No rigid motion changes it. The
/// three sites must not lie on one line.
[[nodiscard]] Measurement AngleMeasurement(const SurfaceGrid& grid, double a,
                                           GeographicPoint vertex,
                                           GeographicPoint first,
                                           GeographicPoint second);

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
/// `time_uncertainty` pins the model times to the nominal ones. Each is in
/// the unit of what it qualifies, and the fit refuses a time uncertainty
/// not smaller than the window's span, or a value uncertainty not smaller
/// than the series' range: the marks of an uncertainty in the wrong unit.
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
/// are malformed, an uncertainty is out of scale (DecayTimeOptions) or
/// the window constrains no decay, and
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
