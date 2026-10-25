#pragma once

/**
 * @file SeaLevel.hpp
 * @brief The sea level engine: the gravitationally consistent sea level
 * equation — linear, generalised, or nonlinear with migrating shorelines,
 * with the rotational feedbacks — for whatever Love numbers it is given.
 * A port of pyslfp's `core.sea_level` (its docstrings and
 * `pyslfp_theory.tex` are the authoritative statements), on GSHTrans
 * fields.
 *
 * This is an internal component and works non-dimensionally throughout
 * (the SI user boundary is Fingerprint.hpp): the constructor converts the
 * SI Love table and parameters once, under Scales::Geophysical of the
 * table's radius, and every method's fields, coefficients and scalars are
 * non-dimensional under SolverScales().
 *
 * Numerical scheme, as in pyslfp: both solvers iterate on the surface
 * load; within each iteration the sea level change is synthesised from
 * the load coefficients with the combined factor -(h_l + k_l / g), the
 * rotational feedback is solved in closed form at degree 2 (transverse at
 * order 1, axial at order 0 with its degree-0 part through the axial Love
 * numbers), and mass is conserved by a uniform shift over the oceans. The
 * linear fixed point is accelerated by Anderson mixing on the load
 * coefficients; the nonlinear solver recomputes the flotation ocean
 * function each pass (Kendall-style migration). The adjoint solve is the
 * exact transpose of the generalised solve's input channels, run through
 * the same mixer.
 *
 * The rotation block's algebra is pyslfp's verbatim, carried on the real
 * (cos, sin) coefficient pairs of the pyshtools basis; the conversions to
 * GSHTrans's complex basis live in ShConventions.hpp and are orthogonal
 * for the pairing used, so the adjoint identities survive them.
 */

#include <array>
#include <complex>
#include <span>
#include <vector>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp {

/// Physical parameters of the rotating body, SI (pyslfp's
/// `EarthModelParameters` defaults). The sea-floor radius, surface gravity
/// and G are replaced by the Love table's values on construction of the
/// solver (pyslfp's `with_body`), so the body and its response agree.
struct SeaLevelParameters {
  double equatorial_radius = 6378137.0;
  double polar_radius = 6356752.0;
  double mean_radius = 6371000.0;
  double mean_sea_floor_radius = 6368000.0;
  double mass = 5.974e24;
  double gravitational_acceleration = 9.825652323;
  double gravitational_constant = GravitationalConstantSI;
  double equatorial_moment_of_inertia = 8.0096e37;
  double polar_moment_of_inertia = 8.0359e37;
  double rotation_frequency = 7.27220521664304e-05;
  double water_density = 1000.0;
  double ice_density = 917.0;
};

/// The parameters with the difference of the principal moments set from a
/// model's hydrostatic fluid limit (Mitrovica et al. 2005):
/// C - A = (1 + delta) k_hyd Omega^2 a^5 / (3 G), the equatorial moment
/// adjusted and the polar kept. With delta > 0 the transverse feedback is
/// stable for any rheology of the model. k_hyd is FluidTidalNumber's.
[[nodiscard]] SeaLevelParameters WithHydrostaticMoments(
    SeaLevelParameters parameters, double fluid_tidal_number,
    double non_hydrostatic_fraction = 0.008);

/// The sea level of a state from its topography T and ice thickness I, by
/// pyslfp's construction, which the sl3d program applies to the states it
/// reads: -T over the ocean floor (T < 0) and -T + I over land, raised by
/// the floating fraction (ice density / water density) of any ice standing
/// on the ocean floor. T and I as pyslfp's `export_ice_history` writes them
/// from ICE-5G/6G/7G; any consistent units (SI for Fingerprint, the
/// solver's for SeaLevelSolver::MakeState). Refused when the sizes differ.
[[nodiscard]] std::vector<double> SeaLevelFromTopography(
    std::span<const double> topography, std::span<const double> iceThickness,
    const SeaLevelParameters& parameters = {});

/// Which parts of the rotational feedback are solved. The transverse fixed
/// point requires moments of inertia sustaining more oblateness than the
/// tidal response (see WithHydrostaticMoments); Axial keeps only the
/// component along the rotation axis.
enum class RotationalFeedback { None, Axial, Full };

class SeaLevelSolver {
 public:
  struct Options {
    double rtol = 1e-9;
    int max_iterations = 1000;
    /// Iterates retained by the Anderson acceleration; 0 is plain Picard.
    int anderson_memory = 5;
  };

  /// The grid is copied in; the SI table and parameters are converted to
  /// the solver's non-dimensional scheme here, once.
  SeaLevelSolver(SurfaceGrid grid, const LoveNumbers& table,
                 SeaLevelParameters parameters);

  [[nodiscard]] const SurfaceGrid& Grid() const { return grid_; }

  /// The non-dimensional scheme everything inside is stated under.
  [[nodiscard]] const Scales& SolverScales() const { return scales_; }

  /// The SI parameters this solver was built with (as given, before the
  /// body was taken from the Love table): what a driver building solvers
  /// for other Love tables of the same body must reuse.
  [[nodiscard]] const SeaLevelParameters& Parameters() const {
    return parameters_;
  }

  // Non-dimensional parameters internal consumers need.
  [[nodiscard]] double WaterDensity() const { return waterDensity_; }
  [[nodiscard]] double IceDensity() const { return iceDensity_; }
  [[nodiscard]] double SurfaceGravity() const { return g_; }
  [[nodiscard]] double Radius() const { return radius_; }

  /// The stability margin of the transverse (polar wander) fixed point;
  /// it must be positive for RotationalFeedback::Full to be valid.
  [[nodiscard]] double TransverseRotationDenominator() const {
    return rotationDenominator_;
  }

  /// The transverse margin against a given degree-2 tidal number: a
  /// viscoelastic evolution must check the model's hydrostatic fluid
  /// limit (FluidTidalNumber), which long steps approach, not just the
  /// elastic kt2 the denominator above carries.
  [[nodiscard]] double TransverseMarginFor(double tidalNumber) const {
    return 1.0 - inertiaFactor_ * tidalNumber * rotationFactor_;
  }

  /// The centrifugal potential coefficients of an angular velocity
  /// change (non-dimensional): degree 2 at orders 0 and 1, and the
  /// degree-0 coefficient of the (r/a)^2 mean that an l = 0 tide channel
  /// carries.
  [[nodiscard]] std::vector<std::complex<double>> CentrifugalCoefficients(
      const std::array<double, 3>& omega) const;

  /// The exact transpose of CentrifugalCoefficients: a coefficient dual
  /// (paired by the L2 coefficient pairing of real fields) to the
  /// angular-velocity dual (paired by the dot product). The adjoint
  /// sweeps' psi-dagger exit.
  [[nodiscard]] std::array<double, 3> CentrifugalDual(
      std::span<const std::complex<double>> tideDual) const;

  // -- the surface state ------------------------------------------------------

  /**
   * @brief A snapshot of ice thickness and sea level (non-dimensional),
   * with the ocean function the flotation condition rho_w SL - rho_i I > 0
   * implies and its area. `forcedLand` marks points held at C = 0
   * whatever the flotation says (pyslfp masks the Caspian this way); the
   * state keeps it, so the nonlinear solve's migrating shorelines respect
   * it too (empty: none).
   *
   * The ocean function need not be the flotation one: a state may carry a
   * prescribed C (the frozen shorelines of a calibrated initial state,
   * InitialSeaLevel.hpp), with oceanArea its integral. The linear solves
   * read C and its area only, never the sea level; the nonlinear solve
   * starts from the state's C and re-derives it by flotation.
   */
  struct State {
    std::vector<double> iceThickness;
    std::vector<double> seaLevel;
    std::vector<double> oceanFunction;
    double oceanArea = 0.0;  ///< int C dS, radius factor included
    std::vector<char> forcedLand;
  };

  [[nodiscard]] State MakeState(std::vector<double> iceThickness,
                                std::vector<double> seaLevel,
                                std::span<const char> forcedLand = {}) const;

  // -- the generalised linear solve
  // ---------------------------------------------

  /// The forcings, all non-dimensional; an empty vector is an absent
  /// channel. Grids are fields over the grid points; responses are ready
  /// coefficient-space responses joined with no Love number applied (a
  /// viscoelastic step's memory field).
  struct GeneralisedForcing {
    std::vector<double> directLoad;
    std::vector<double> displacementLoad;
    std::vector<double> potentialLoad;
    std::vector<std::complex<double>> displacementResponse;
    std::vector<std::complex<double>> potentialResponse;
    double inertiaMomentChange = 0.0;
    std::array<double, 3> angularMomentumChange{0.0, 0.0, 0.0};
    /// An override of the state's ocean function (linearising about a
    /// migrating-shoreline trajectory, or a smooth benchmark C).
    std::vector<double> oceanFunctionOverride;
    RotationalFeedback rotation = RotationalFeedback::Full;
  };

  struct Response {
    std::vector<double> seaLevelChange;
    std::vector<double> displacement;
    std::vector<double> potential;
    std::array<double, 3> angularVelocity{0.0, 0.0, 0.0};
    int iterations = 0;
    double error = 0.0;
    bool converged = true;
  };

  [[nodiscard]] Response SolveGeneralised(const State& state,
                                          const GeneralisedForcing& forcing,
                                          Options options) const;
  [[nodiscard]] Response SolveGeneralised(
      const State& state, const GeneralisedForcing& forcing) const {
    return SolveGeneralised(state, forcing, Options{});
  }

  /// The standard linear sea level equation for a surface mass load.
  [[nodiscard]] Response Solve(const State& state,
                               std::vector<double> directLoad,
                               RotationalFeedback rotation,
                               Options options) const;
  [[nodiscard]] Response Solve(
      const State& state, std::vector<double> directLoad,
      RotationalFeedback rotation = RotationalFeedback::Full) const {
    return Solve(state, std::move(directLoad), rotation, Options{});
  }

  // -- the adjoint of the generalised solve
  // ---------------------------------------

  /// The duals of the outputs: densities paired with the output fields by
  /// the surface integral, and the angular-velocity dual by the dot
  /// product. Must use the ocean function and rotation setting of the
  /// forward solve being transposed.
  struct AdjointForcing {
    std::vector<double> seaLevelDual;
    std::vector<double> displacementDual;
    std::vector<double> potentialDual;
    std::array<double, 3> angularVelocityDual{0.0, 0.0, 0.0};
    std::vector<double> oceanFunctionOverride;
    RotationalFeedback rotation = RotationalFeedback::Full;
  };

  /// The duals of the generalised solve's input channels: the direct load
  /// (a grid density), the two response channels (coefficients, paired by
  /// the L2 coefficient pairing) and the inertia moment.
  struct AdjointResponse {
    std::vector<double> directLoadDual;
    std::vector<std::complex<double>> displacementResponseDual;
    std::vector<std::complex<double>> potentialResponseDual;
    double inertiaMomentDual = 0.0;
    int iterations = 0;
    double error = 0.0;
    bool converged = true;
  };

  [[nodiscard]] AdjointResponse SolveAdjoint(const State& state,
                                             const AdjointForcing& forcing,
                                             Options options) const;
  [[nodiscard]] AdjointResponse SolveAdjoint(
      const State& state, const AdjointForcing& forcing) const {
    return SolveAdjoint(state, forcing, Options{});
  }

  // -- the nonlinear (migrating shoreline) solve
  // ------------------------------------

  struct NonlinearForcing {
    std::vector<double> iceThicknessChange;
    std::vector<double> sedimentThicknessChange;
    std::vector<double> dynamicSeaLevelChange;
    /// Non-dimensional; negative means the default 2300 kg m^-3.
    double sedimentDensity = -1.0;
    std::vector<std::complex<double>> displacementResponse;
    std::vector<std::complex<double>> potentialResponse;
    double inertiaMomentChange = 0.0;
    RotationalFeedback rotation = RotationalFeedback::Full;
  };

  struct NonlinearResponse {
    State finalState;
    Response response;
  };

  [[nodiscard]] NonlinearResponse SolveNonlinear(
      const State& state, const NonlinearForcing& forcing,
      Options options) const;
  [[nodiscard]] NonlinearResponse SolveNonlinear(
      const State& state, const NonlinearForcing& forcing) const {
    Options options;
    options.max_iterations = 50;
    return SolveNonlinear(state, forcing, options);
  }

 private:
  struct Rotation;  // the degree-2/degree-0 feedback block, in SeaLevel.cpp

  /// int f dS = a^2 int f dOmega.
  [[nodiscard]] double IntegrateSurface(std::span<const double> field) const;

  SurfaceGrid grid_;
  Scales scales_;
  SeaLevelParameters parameters_;  // as given (SI)
  // Non-dimensional Love arrays, degree 0..lmax: channel sums and splits.
  std::vector<double> h_, k_, hU_, kU_, hPhi_, kPhi_;
  std::vector<double> slcFactor_;  // -(h + k / g), per degree
  // Degree-2 and axial numbers.
  double h2_ = 0, k2_ = 0, ht2_ = 0, kt2_ = 0;
  double hc_ = 0, kc_ = 0, mu_ = 0, mphi_ = 0, mc_ = 0;
  // Non-dimensional parameters and the rotation factors (pyslfp's names).
  double g_ = 0, waterDensity_ = 0, iceDensity_ = 0, radius_ = 0;
  double rotationFactor_ = 0, inertiaFactor_ = 0;
  double axialRotationFactor_ = 0, axialInertiaFactor_ = 0;
  double uniformRotationFactor_ = 0;
  double inverseInertiaDifference_ = 0, inversePolarInertia_ = 0;
  double traceFactor_ = 0, shellFactor_ = 0;
  double rotationDenominator_ = 0, axialRotationDenominator_ = 0;
};

}  // namespace sl3dpp
