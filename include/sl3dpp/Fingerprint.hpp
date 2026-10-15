#pragma once

/**
 * @file Fingerprint.hpp
 * @brief Elastic sea-level fingerprints: the public, SI face of the sea
 * level engine, pyslfp-shaped (plan §1) so that sl3dpp can one day serve
 * as a faster backend for pyslfp.
 *
 * An elastic earth responds instantaneously, so a fingerprint is one
 * equilibrium solve of the sea level equation against a fixed background
 * state. SI in, SI out: ice thicknesses and sea levels in metres, loads in
 * kg m^-2, the returned sea level change, displacement in metres, the
 * potential perturbation in m^2 s^-2 and the angular velocity change in
 * rad s^-1. Everything inside is the non-dimensional SeaLevelSolver.
 */

#include <array>
#include <span>
#include <vector>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp {

class Fingerprint {
 public:
  struct Options {
    SeaLevelParameters parameters = {};
    bool parallel_transforms = false;
    double rtol = 1e-9;
    int anderson_memory = 5;
  };

  /// A fingerprint solver at degree lmax for an SI Love table (from
  /// ComputeLoveNumbers or LoveNumbers::FromFile). No state is set yet.
  Fingerprint(int lmax, const LoveNumbers& table, Options options);
  Fingerprint(int lmax, const LoveNumbers& table)
      : Fingerprint(lmax, table, Options{}) {}

  [[nodiscard]] const SurfaceGrid& Grid() const { return solver_.Grid(); }
  [[nodiscard]] const SeaLevelSolver& Solver() const { return solver_; }

  // -- the background state (SI metres)
  // -----------------------------------------

  /// Sets the background state from ice thickness and sea level
  /// (topography) in metres over the grid points; `forcedLand` marks
  /// points treated as land whatever the flotation condition says.
  void SetState(std::span<const double> iceThickness,
                std::span<const double> seaLevel,
                std::span<const char> forcedLand = {});

  /// pyslfp's analytical testing state (`AnalyticalIceModel`): smooth
  /// super-Gaussian continents and ice caps, solvable at low degree
  /// without coastline aliasing. `date` in kiloyears before present (0 the
  /// present day, 21 the glacial maximum).
  void SetAnalyticTestState(double date = 0.0);

  /// The state's ocean function (1 over oceans) and the background fields,
  /// SI where dimensional.
  [[nodiscard]] std::span<const double> OceanFunction() const {
    return state_.oceanFunction;
  }
  [[nodiscard]] std::vector<double> IceThickness() const;
  [[nodiscard]] std::vector<double> SeaLevel() const;

  // -- loads (SI)
  // -----------------------------------------------------------------

  /// The surface mass load (kg m^-2) of an ice thickness change (metres):
  /// rho_ice (1 - C) Delta I, grounded ice only.
  [[nodiscard]] std::vector<double> LoadFromIceThicknessChange(
      std::span<const double> iceThicknessChange) const;

  /// The surface mass load of a sea level change: rho_water C Delta SL.
  [[nodiscard]] std::vector<double> LoadFromSeaLevelChange(
      std::span<const double> seaLevelChange) const;

  // -- solves (SI)
  // ------------------------------------------------------------------

  struct Result {
    std::vector<double> seaLevelChange;       ///< m
    std::vector<double> displacement;         ///< m
    std::vector<double> potential;            ///< m^2 s^-2
    std::array<double, 3> angularVelocity{};  ///< rad s^-1
    int iterations = 0;
    double error = 0.0;
    bool converged = true;
  };

  /// The fingerprint of a surface mass load (kg m^-2).
  [[nodiscard]] Result Solve(
      std::span<const double> directLoad,
      RotationalFeedback rotation = RotationalFeedback::Full) const;

  /// The fingerprint of an ice thickness change (metres).
  [[nodiscard]] Result SolveForIceChange(
      std::span<const double> iceThicknessChange,
      RotationalFeedback rotation = RotationalFeedback::Full) const;

  /// The nonlinear solve with migrating shorelines for an ice thickness
  /// change (metres); the background state is left untouched (call
  /// SetState with the outputs to step).
  [[nodiscard]] Result SolveNonlinear(
      std::span<const double> iceThicknessChange,
      RotationalFeedback rotation = RotationalFeedback::Full) const;

 private:
  [[nodiscard]] std::vector<double> ToDimensionless(
      std::span<const double> field, Dimensions dimensions) const;
  [[nodiscard]] Result FromResponse(SeaLevelSolver::Response response) const;

  SeaLevelSolver solver_;
  SeaLevelSolver::State state_;
  double rtol_;
  int andersonMemory_;
};

}  // namespace sl3dpp
