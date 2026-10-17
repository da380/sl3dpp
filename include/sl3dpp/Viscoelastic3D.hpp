#pragma once

/**
 * @file Viscoelastic3D.hpp
 * @brief The 3D-viscosity stepper: grid-held internal variables, pointwise
 * step weights from laterally varying relaxation times, and the
 * within-step lateral iteration (docs/stage4.md §5 — the research content
 * of the project, plan §3).
 *
 * The viscosity is eta(x) = eta_bar(r) exp(delta(x)): the base (radial)
 * Rheology is the REFERENCE, whose per-(scheme, dt) effective operators
 * are factorised once and reused, and delta is the band-limited lateral
 * log-perturbation, sampled on the memory nodes' grid; every branch's
 * inverse relaxation time is scaled by exp(-delta(x)).
 *
 * The state is the five-component memory tensor per branch on the grid —
 * a collocation representation: the pointwise update
 * m+ = e(x) m + alpha(x) d^n + beta(x) d^{n+1} is exact in angle at the
 * grid points, and the only spatial approximation is the memory-force
 * quadrature, dealiased by the grid's oversampling.
 *
 * The step eliminates m+ as in 1D, leaving the laterally varying
 * effective shear mu_inf + sum_k (1 - beta_k(x)) mu_k(r). Writing
 * K(x) = K_ref + dK and moving dK to the right-hand side turns the
 * lateral term into a pointwise augmentation of the memory tensor:
 *
 *   K_ref u = F + b( m~_k(x) + (beta_k(x) - beta_ref_k(r)) d(u) ),
 *
 * so each pass of the within-step iteration is one trip through the
 * (gated) GridCoupling pipeline plus the reference banded back-solves,
 * spheroidal and toroidal. The iteration is Anderson-accelerated
 * (memory 5 by default; 0 gives plain Picard, optionally damped), and
 * one loop, OpenStep::Converge, serves this class's Step, the sea level
 * coupling and the adjoint sweep. Its convergence against viscosity
 * contrast is the stage-4 research measurement (stage4.md §5a).
 *
 * With delta = 0 the lateral term vanishes identically and a step
 * reproduces the 1D two-phase stepper exactly — the zero-lateral step
 * gate. Load channels are prescribed here (complex m >= 0 columns of real
 * fields); the sea-level coupling drives this class through the same
 * channels.
 */

#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp {

class Viscoelastic3D {
 public:
  using Complex = std::complex<double>;
  /// A load channel: the complex m >= 0 columns (l + 1, index m; the
  /// field is real, so m < 0 is implied) at a time, or std::nullopt
  /// where silent.
  using LoadCallable3D =
      std::function<std::optional<std::vector<Complex>>(int l, double t)>;

  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
    double oversampling = 1.5;
    bool parallel = false;
    bool matrix_kernel = false;
    /// The within-step iteration: Anderson-accelerated (0 = plain
    /// Picard), with optional damping of the Picard part.
    int max_iterations = 40;
    double rtol = 1e-11;
    double damping = 1.0;
    int anderson_memory = 5;
  };

  /// The reference rheology is copied (shared, immutable); delta is
  /// sampled as delta(r, theta, phi) over the memory nodes' grid.
  Viscoelastic3D(const Rheology& rheology, int lmax,
                 const std::function<double(double, double, double)>& delta,
                 Options options);
  Viscoelastic3D(const Rheology& rheology, int lmax,
                 const std::function<double(double, double, double)>& delta)
      : Viscoelastic3D(rheology, lmax, delta, Options{}) {}
  /// Shares an existing grid (lmax-matched, spin coverage >= 2) instead
  /// of building one — the way several operators at one lmax, or a
  /// driver that also owns the grid, avoid repeating its construction.
  Viscoelastic3D(const Rheology& rheology, int lmax, const SurfaceGrid& grid,
                 const std::function<double(double, double, double)>& delta,
                 Options options);
  ~Viscoelastic3D();  // out of line: unique_ptr of an incomplete type
  Viscoelastic3D(Viscoelastic3D&&) noexcept;
  Viscoelastic3D& operator=(Viscoelastic3D&&) noexcept;

  [[nodiscard]] const GridCoupling& Coupling() const { return coupling_; }
  [[nodiscard]] int LMax() const { return lmax_; }
  [[nodiscard]] const Rheology& GetRheology() const { return *rheology_; }
  [[nodiscard]] const Options& GetOptions() const { return options_; }

  /// The unrelaxed solid's Love numbers (SI), cached.
  [[nodiscard]] const LoveNumbers& ElasticLove() const {
    return reference_.ElasticLoveNumbers();
  }

  /// The grid-held state: one five-component tensor per branch, plus the
  /// strain cache of the last consistent solve.
  struct State {
    double t = 0.0;
    std::vector<GridCoupling::TensorField> m;
    std::optional<GridCoupling::TensorField> strain;
  };
  [[nodiscard]] State ZeroState() const;

  /// Per-degree solutions: spheroidal complex m >= 0 columns
  /// (ndof x (l + 1), column-major) and toroidal likewise (degrees >= 2).
  struct Solutions {
    std::map<int, std::vector<Complex>> spheroidal;
    std::map<int, std::vector<Complex>> toroidal;
  };

  /// The quasi-static solutions consistent with (m, t): the unrelaxed
  /// reference systems forced by the loads and the memory force of m.
  /// (The unrelaxed operator is laterally uniform — viscosity enters only
  /// through the stepping — so no iteration is needed here.)
  [[nodiscard]] Solutions SolveElastic(const State& state,
                                       const LoadCallable3D& load = {},
                                       const LoadCallable3D& tide = {}) const;

  /// Ensures the state carries a strain cache consistent with (m, t)
  /// under the given loads, solving once if not (the 1D operator's
  /// PrimeStrain; a recording run needs the start node's cache).
  void PrimeStrain(State& state, const LoadCallable3D& load = {},
                   const LoadCallable3D& tide = {}) const;

  struct StepInfo {
    int iterations = 0;
    double residual = 0.0;
    bool converged = true;
  };

  /// The within-step fixed point's controls: at most max_iterations
  /// passes, converged when the relative sup-norm change of the strain is
  /// at most rtol; damping scales the Picard part, and anderson_memory 0
  /// means plain (damped) Picard.
  struct IterationControl {
    int max_iterations = 40;
    double rtol = 1e-11;
    double damping = 1.0;
    int anderson_memory = 5;
  };

  /// One implicit step (ExponentialTrapezoid or BackwardEuler) under the
  /// load channels; returns the new state (with its strain cache), the
  /// end-time solutions, and the iteration diagnostics.
  struct StepResult {
    State state;
    Solutions solutions;
    StepInfo info;
  };
  [[nodiscard]] StepResult Step(const State& state, double dt,
                                Scheme scheme = Scheme::ExponentialTrapezoid,
                                const LoadCallable3D& load = {},
                                const LoadCallable3D& tide = {}) const;

  /// The surface value of U, V or phi per degree: complex m >= 0 columns.
  [[nodiscard]] std::map<int, std::vector<Complex>> Surface(
      const Solutions& solutions, Component component) const;

  /**
   * @brief One opened implicit step, for drivers that close the step's
   * equilibrium themselves (the sea level coupling): the decayed memory
   * m~(x) and the weight fields are fixed on construction, and the
   * driver alternates Respond / its own load closure, then finishes.
   */
  class OpenStep {
   public:
    [[nodiscard]] double EndTime() const { return t_; }
    [[nodiscard]] const GridCoupling::TensorField& StartStrain() const {
      return d0_;
    }

    /// The solutions of the (memory + lateral(d)) force through the
    /// reference effective systems, with NO load channels: what enters a
    /// sea level solver as ready response channels.
    [[nodiscard]] Solutions Respond(const GridCoupling::TensorField& d) const;

    /// Adds load/tide channels to zero-channel solutions by linearity
    /// (cached unit responses of the effective systems).
    void AddChannels(Solutions& solutions, const LoadCallable3D& load,
                     const LoadCallable3D& tide) const;

    /// The strain field of solutions, for the next pass.
    [[nodiscard]] GridCoupling::TensorField StrainOf(
        const Solutions& solutions) const;

    /// One pass of a driver's within-step iteration: the strain the
    /// step's solutions take for a given strain (Respond, the driver's own
    /// load closure, AddChannels, StrainOf).
    using Pass = std::function<GridCoupling::TensorField(
        const GridCoupling::TensorField&)>;

    /// The within-step fixed point d = pass(d) from the start strain,
    /// Anderson-mixed in place: the converged (or, when the budget runs
    /// out, the last) strain and the diagnostics. The pass runs once per
    /// iteration, so whatever it records is from the final pass.
    [[nodiscard]] std::pair<GridCoupling::TensorField, StepInfo> Converge(
        const Pass& pass, const IterationControl& control) const;

    /// The inertia moment of the degree-0, m = 0 column of solutions.
    [[nodiscard]] double InertiaMoment(const Solutions& solutions) const;

    /// The effective Love numbers of this step's reference solid (SI).
    [[nodiscard]] const LoveNumbers& EffectiveLove() const;

    /// Completes the step: m+ = m~ + beta(x) d, with the strain cached.
    [[nodiscard]] State Finish(GridCoupling::TensorField d) const;

   private:
    friend class Viscoelastic3D;
    const Viscoelastic3D* op_ = nullptr;
    Scheme scheme_{};
    double dt_ = 0.0, t_ = 0.0;
    GridCoupling::TensorField d0_;
    std::vector<GridCoupling::TensorField> memory_;
    std::vector<std::vector<double>> beta_, dbeta_;
  };

  /// Opens a step: the start strain comes from the state's cache or a
  /// consistent solve under the start-time channels.
  [[nodiscard]] OpenStep StartStep(const State& state, double dt,
                                   Scheme scheme = Scheme::ExponentialTrapezoid,
                                   const LoadCallable3D& load = {},
                                   const LoadCallable3D& tide = {}) const;

 private:
  friend class AdjointSolver3D;  // the sweeps transpose these internals

  struct Flavour;  // reference effective systems per (scheme, dt)

  [[nodiscard]] const DegreeOperator& Spheroidal(int l) const;
  [[nodiscard]] const ToroidalOperator& Toroidal(int l) const;
  [[nodiscard]] Flavour& FlavourOf(Scheme scheme, double dt) const;
  [[nodiscard]] const DegreeOperator& FlavourSpheroidal(Flavour& flavour,
                                                        int l) const;
  [[nodiscard]] const ToroidalOperator& FlavourToroidal(Flavour& flavour,
                                                        int l) const;
  /// d(x) of per-degree solutions, through the pipeline.
  [[nodiscard]] GridCoupling::TensorField StrainField(
      const Solutions& solutions,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;
  /// Solve all degrees of both blocks for a combined weighted tensor,
  /// adding the load channels.
  [[nodiscard]] Solutions SolveAll(
      const GridCoupling::SpectralTensor& weighted, double t,
      const LoadCallable3D& load, const LoadCallable3D& tide,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;

  std::shared_ptr<const Rheology> rheology_;
  int lmax_;
  Options options_;
  GridCoupling coupling_;
  ViscoelasticOperator reference_;         // the radial machinery and tables
  std::vector<std::vector<double>> itau_;  // per branch, (node x grid)
  std::vector<std::vector<double>> muBranch_;  // per branch, per node
  mutable std::map<int, std::unique_ptr<DegreeOperator>> spheroidal_;
  mutable std::map<int, std::unique_ptr<ToroidalOperator>> toroidal_;
  mutable std::map<std::pair<int, double>, std::unique_ptr<Flavour>> flavours_;
};

}  // namespace sl3dpp
