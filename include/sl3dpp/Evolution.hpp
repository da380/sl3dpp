#pragma once

/**
 * @file Evolution.hpp
 * @brief Forward GIA: ice history in, sea level, displacement, potential
 * and rotation histories out. A port of pyslfp's
 * `viscoelastic.evolution.SeaLevelEvolution`: each implicit step is opened
 * before its end-time loads are known, its memory responses and inertia
 * moment enter the sea level equation through the generalised response
 * channels, the SLE closes the step's equilibrium with the matching
 * per-(scheme, dt) effective Love numbers, and the step finishes under the
 * converged load and centrifugal channels.
 *
 * Internal component, non-dimensional: the radial operator works in the
 * material's scheme and the sea level solver in its own, with the unit
 * ratios carried here exactly as pyslfp does. Times and the ice-load
 * callable are in the SLE solver's units; the returned fields likewise.
 * The linear (frozen shoreline) equation is used throughout, with the
 * background state's ocean function.
 */

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp {

class SeaLevelEvolution {
 public:
  struct Options {
    Scheme scheme = Scheme::ExponentialTrapezoid;
    RotationalFeedback rotation = RotationalFeedback::Full;
    double rtol = 1e-9;
    int anderson_memory = 5;
    /// Keep every stride-th node's internal state on its snapshot (and
    /// the last; 0 keeps none); the per-node load and centrifugal
    /// coefficients are kept whenever the stride is set, so a reverse
    /// sweep can reconstruct the gaps by exact backward steps.
    int record_stride = 0;
  };

  /// One node of the history: fields in the SLE solver's units on its
  /// grid; the load and tide coefficient vectors are in the MATERIAL's
  /// units (what the radial operator consumed).
  struct Snapshot {
    double t = 0.0;
    std::vector<double> seaLevelChange;
    std::vector<double> displacement;
    std::vector<double> potential;
    std::array<double, 3> angularVelocity{};
    std::optional<ViscoelasticState> internalState;
    std::vector<std::complex<double>> loadCoefficients;
    std::vector<std::complex<double>> tideCoefficients;
  };

  /// The operator and solver are borrowed for the evolution's lifetime;
  /// the SLE grid's lmax must match the operator's.
  SeaLevelEvolution(const ViscoelasticOperator& op,
                    const SeaLevelSolver& solver, SeaLevelSolver::State state,
                    Options options);
  SeaLevelEvolution(const ViscoelasticOperator& op,
                    const SeaLevelSolver& solver, SeaLevelSolver::State state)
      : SeaLevelEvolution(op, solver, std::move(state), Options{}) {}

  /// Integrates over the time grid (SLE units, increasing; include any
  /// breakpoints of the ice history). `iceLoad(t)` is the non-ocean
  /// surface load (grounded ice, sediment) at t, a grid field in the SLE
  /// solver's units, continuous for t > times[0]; the start is the
  /// instantaneous elastic response to iceLoad(times[0]). Returns one
  /// snapshot per node, the first included, and leaves the final internal
  /// state readable through FinalState().
  [[nodiscard]] std::vector<Snapshot> Run(
      std::span<const double> times,
      const std::function<std::vector<double>(double)>& iceLoad);

  [[nodiscard]] const ViscoelasticState& FinalState() const { return vstate_; }

 private:
  friend class AdjointSolver;  // reuses the operator/solver/state/ratios

  /// A degree-column callable over a coefficient vector in material
  /// units: degree l maps to its 2 (l + 1) columns, the real block then
  /// the imaginary block of orders 0..l.
  [[nodiscard]] LoadCallable Columns(
      const std::vector<std::complex<double>>& coefficients,
      double factor) const;
  /// The inverse packing of per-degree surface columns, times a factor.
  [[nodiscard]] std::vector<std::complex<double>> Coefficients(
      const std::map<int, std::vector<double>>& perDegree, double factor) const;

  const ViscoelasticOperator* op_;
  const SeaLevelSolver* solver_;
  SeaLevelSolver::State state_;
  Options options_;
  ViscoelasticState vstate_;
  // Unit ratios, material <-> SLE schemes (pyslfp's names).
  double toNdLength_ = 1.0, toNdPotential_ = 1.0, toNdMoment_ = 1.0;
  double toMatDensity_ = 1.0, toMatPotential_ = 1.0, toMatTime_ = 1.0;
};

}  // namespace sl3dpp
