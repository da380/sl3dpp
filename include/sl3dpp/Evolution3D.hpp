#pragma once

/**
 * @file Evolution3D.hpp
 * @brief Forward GIA with lateral viscosity: the coupled loop of plan §3 —
 * one combined within-step fixed point over the ocean load, the rotational
 * feedback and the lateral viscosity term, under one Anderson mixer.
 *
 * Per step, the opened 3D step (Viscoelastic3D::OpenStep) holds the
 * decayed memory; each pass of the combined iteration computes the
 * memory + lateral responses for the current strain iterate, hands their
 * surface values and inertia moment to the sea level solver as ready
 * response channels (the SLE closes the load and rotation with the
 * reference flavour's effective Love numbers), rebuilds the total load
 * and centrifugal channels from the SLE's answer, and updates the strain
 * through the step's reference solves. With no lateral perturbation the
 * channels are strain-independent, the loop converges on its first pass,
 * and the history equals the 1D coupled evolution's — the coupled
 * zero-lateral gate.
 *
 * Units follow SeaLevelEvolution: times, the ice-load callable and the
 * returned fields are in the SLE solver's non-dimensional scheme; the
 * radial operator works in the material's, with the ratios carried here.
 * The m >= 0 SLE coefficients and the full-m material columns are bridged
 * by the Hermitian symmetry of real fields.
 */

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp {

class SeaLevelEvolution3D {
 public:
  struct Options {
    Scheme scheme = Scheme::ExponentialTrapezoid;
    RotationalFeedback rotation = RotationalFeedback::Full;
    /// The SLE's own tolerance per pass.
    double rtol = 1e-9;
    int anderson_memory = 5;
    /// The combined (strain) fixed point.
    int max_coupled_iterations = 60;
    double coupled_rtol = 1e-9;
    /// Keep every stride-th node's internal state on its snapshot (and
    /// the last; 0 keeps none); the per-node load and centrifugal
    /// coefficients are kept whenever the stride is set. (No 3D
    /// backward reconstruction exists yet: a reverse sweep needs
    /// stride 1.)
    int record_stride = 0;
  };

  struct Snapshot {
    double t = 0.0;
    std::vector<double> seaLevelChange;
    std::vector<double> displacement;
    std::vector<double> potential;
    std::array<double, 3> angularVelocity{};
    int coupledIterations = 0;
    double coupledResidual = 0.0;
    bool converged = true;
    /// The internal variables (no strain cache: one elastic solve under
    /// the recorded load and centrifugal coefficients regenerates it).
    std::optional<Viscoelastic3D::State> internalState;
    std::vector<std::complex<double>> loadCoefficients;
    std::vector<std::complex<double>> tideCoefficients;
  };

  /// The operator and solver are borrowed for the evolution's lifetime;
  /// the SLE grid's lmax must match the operator's.
  SeaLevelEvolution3D(const Viscoelastic3D& op, const SeaLevelSolver& solver,
                      SeaLevelSolver::State state, Options options);
  SeaLevelEvolution3D(const Viscoelastic3D& op, const SeaLevelSolver& solver,
                      SeaLevelSolver::State state)
      : SeaLevelEvolution3D(op, solver, std::move(state), Options{}) {}

  /// As SeaLevelEvolution::Run: times increasing in SLE units,
  /// iceLoad(t) the non-ocean surface load on the SLE grid.
  [[nodiscard]] std::vector<Snapshot> Run(
      std::span<const double> times,
      const std::function<std::vector<double>(double)>& iceLoad);

  [[nodiscard]] const Viscoelastic3D::State& FinalState() const {
    return vstate_;
  }

 private:
  friend class AdjointSolver3D;  // reuses the operator/solver/state/ratios

  using Complex = std::complex<double>;

  /// A degree-column callable over m >= 0 coefficients, the negative
  /// orders restored by the Hermitian symmetry of a real field.
  [[nodiscard]] Viscoelastic3D::LoadCallable3D Columns(
      const std::vector<Complex>& coefficients, double factor) const;
  /// m >= 0 coefficients from full-m surface columns, times a factor.
  [[nodiscard]] std::vector<Complex> Coefficients(
      const std::map<int, std::vector<Complex>>& perDegree,
      double factor) const;

  const Viscoelastic3D* op_;
  const SeaLevelSolver* solver_;
  SeaLevelSolver::State state_;
  Options options_;
  Viscoelastic3D::State vstate_;
  double toNdLength_ = 1.0, toNdPotential_ = 1.0, toNdMoment_ = 1.0;
  double toMatDensity_ = 1.0, toMatPotential_ = 1.0, toMatTime_ = 1.0;
};

}  // namespace sl3dpp
