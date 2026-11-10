#pragma once

/**
 * @file Viscoelastic3D.hpp
 * @brief The 3D-viscosity stepper: grid-held internal variables, pointwise
 * step weights from laterally varying relaxation times, and the
 * within-step lateral iteration (docs/lateral-coupling.md).
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
 * GridCoupling pipeline plus the reference banded back-solves,
 * spheroidal and toroidal. The iteration is Anderson-accelerated
 * (memory 5 by default; 0 gives plain Picard, optionally damped), and
 * one loop, OpenStep::Converge, serves this class's Step, the sea level
 * coupling and the adjoint sweep. Its convergence against viscosity
 * contrast is measured in docs/lateral-coupling.md. The same step is
 * a symmetric positive definite system in u, and Options::solver =
 * ConjugateGradients closes it by preconditioned conjugate gradients
 * instead (K_ref the preconditioner; OpenStep::Solve), with a driver's
 * sea level closure as an outer iteration around warm-started inexact
 * solves (OpenStep::Nest): fewer iterations than passes everywhere
 * measured and no contrast envelope (docs/adjoints.md §13n).
 *
 * Lateral ELASTIC moduli (Options::moduli) enter the same weight: the
 * shear's relative perturbation s(x) adds -s mu_eff(x) to it and scales
 * the memory force's branch moduli by (1 + s), and a bulk perturbation
 * weights the dilatation, carried as the iterate's scalar companion;
 * the unrelaxed start solve is then the conjugate-gradient solve of
 * detail/LateralSolve.hpp (docs/adjoints.md §13k).
 *
 * With delta = 0 and no lateral moduli the lateral term vanishes
 * identically and a step reproduces the 1D two-phase stepper exactly. The memory nodes are the
 * Gauss points of the viscous elements or, with Options::memory_points,
 * their GLL nodes (MemoryPoints.hpp); the 1D reference takes the same
 * set, so the identity holds for either. Load channels are prescribed
 * here (complex m >= 0 columns of real fields); the sea-level coupling
 * drives this class through the same channels.
 */

#include <complex>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "sl3dpp/Collectives.hpp"
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
    /// Whether the step runs threaded, on OpenMP's thread count
    /// (OMP_NUM_THREADS; 1 for a serial run): the transforms (GSHTrans)
    /// and the per-degree loops of force assembly, solves and strain
    /// recovery. false forces serial, for a caller already inside its own
    /// parallel region, say. With the loop kernel (matrix_kernel false) the
    /// step is bitwise independent of the thread count: the loops write
    /// only what their index owns, the transforms are thread-count
    /// independent, and the mixer's sums run over blocks fixed by the
    /// vector's length.
    bool parallel = true;
    bool matrix_kernel = false;
    /// The cache the transforms block for (SurfaceGrid::Options).
    std::size_t cache_bytes = 0;
    /// The within-step iteration (IterationControl): Picard, Anderson-
    /// accelerated when it contracts too slowly (anderson_memory 0 = plain
    /// Picard throughout), with optional damping of the Picard part.
    int max_iterations = 40;
    double rtol = 1e-11;
    double damping = 1.0;
    int anderson_memory = 5;
    double anderson_rate = 0.5;
    /// How Step closes a step's linear system: the within-step fixed
    /// point above (Picard, Anderson), or preconditioned conjugate
    /// gradients on the laterally weighted effective systems with the
    /// reference effective solve as the preconditioner (OpenStep::Solve;
    /// solve_rtol on the relative residual in the preconditioned energy
    /// norm, solve_max_iterations the budget; docs/adjoints.md §13n).
    /// Drivers that close a step themselves (the evolutions, the sweeps)
    /// read the same choice.
    enum class StepSolver { FixedPoint, ConjugateGradients };
    StepSolver solver = StepSolver::FixedPoint;
    double solve_rtol = 1e-11;
    int solve_max_iterations = 200;
    /// A driver's outer iteration with the solve inside (OpenStep::Nest)
    /// solves each pass inexactly: to solve_outer_factor times the
    /// residual its warm start had (never below solve_rtol), so the inner
    /// work follows the outer progress.
    double solve_outer_factor = 1e-2;
    /// The degree-1 toroidal gauge (ToroidalOperator::Options::
    /// pin_fraction): where in each solid region the pinned dof sits.
    /// Nothing well defined depends on it; it is here so that tests can
    /// show that.
    double toroidal_pin_fraction = 0.0;
    /// Where the internal variables sit within each radial element:
    /// Gauss points (the default; ~20% fewer memory nodes and transforms,
    /// the step an exact elimination still; MemoryPoints.hpp), or GLL
    /// nodes.
    MemoryPointSet memory_points = MemoryPointSet::Gauss;
    /// Lateral perturbations of the ELASTIC moduli, relative and sampled
    /// like delta: the unrelaxed shear mu_U(x) = mu_U(r) (1 + shear(r,
    /// theta, phi)), every branch modulus scaled alike (the branch
    /// fractions and the relaxation times held, as the material
    /// derivatives hold them), and kappa(x) = kappa(r) (1 + bulk(r, theta,
    /// phi)); either may be empty. With any set the coupling carries,
    /// beside the viscous elements, every solid element where a
    /// perturbation is non-zero at some memory point (the functions are
    /// sampled on a probe of the whole solid first), the
    /// unrelaxed start solve is the preconditioned conjugate-gradient
    /// solve of detail/LateralSolve.hpp on the reference systems
    /// (elastic_rtol, elastic_max_iterations), and the within-step
    /// iterate carries the dilatation beside the deviatoric strain (the
    /// lateral bulk term). The step itself stays one fixed point: the
    /// lateral moduli join the lateral viscosity in its pointwise weight.
    /// Distributed over radial segments like the rest (the start solve's
    /// conjugate gradients pair each process's share; docs/adjoints.md
    /// §13k, §13m).
    struct LateralModuli {
      std::function<double(double, double, double)> shear;
      std::function<double(double, double, double)> bulk;
    };
    LateralModuli moduli{};
    double elastic_rtol = 1e-12;
    int elastic_max_iterations = 200;
    /// Radial cuts for every degree system, the reference operator's
    /// included (ViscoelasticOperator::Options::radial_cuts).
    std::vector<int> radial_cuts{};
    /// The degrees the transforms may be cut to with depth
    /// (GridCoupling::Options::transform_degrees): each memory node is
    /// transformed, and delta sampled, on a grid of the least of these
    /// covering its element's degrees. Empty: every node at lmax.
    std::vector<int> transform_degrees{};
    /// Distributed over the segments of radial_cuts (Distributed.hpp's
    /// ShareMesh fills these): the segments this process owns, contiguous,
    /// and the collectives over the processes. The process then holds the
    /// memory nodes of its segments' elements alone (the state, the
    /// strain), solves only its segments of every degree system (each
    /// solution holding its segments and the cut nodes, zero elsewhere),
    /// and every step, solve and Surface is collective; its strain
    /// coefficients and memory are its own nodes'. Empty collectives: one
    /// process, every segment.
    std::vector<int> segments{};
    Collectives collectives{};
  };

  /// The reference rheology is copied (shared, immutable); delta is
  /// sampled as delta(r, theta, phi) over each memory node's grid.
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
  /// The elements a coupling must carry for these lateral moduli: the
  /// viscous ones and those where a perturbation is non-zero at some
  /// memory point (empty when no moduli are set: the rheology's rule).
  [[nodiscard]] static std::vector<char> ActiveElementsFor(
      const Rheology& rheology, int lmax, const SurfaceGrid& grid,
      const Options& options);
  ~Viscoelastic3D();  // out of line: unique_ptr of an incomplete type
  Viscoelastic3D(Viscoelastic3D&&) noexcept;
  Viscoelastic3D& operator=(Viscoelastic3D&&) noexcept;

  [[nodiscard]] const GridCoupling& Coupling() const { return coupling_; }
  [[nodiscard]] int LMax() const { return lmax_; }
  [[nodiscard]] const Rheology& GetRheology() const { return *rheology_; }
  [[nodiscard]] const Options& GetOptions() const { return options_; }
  /// The radial reference: the 1D operator on the same memory points.
  [[nodiscard]] const ViscoelasticOperator& Reference() const {
    return reference_;
  }

  /// The unrelaxed solid's Love numbers (SI), cached.
  [[nodiscard]] const LoveNumbers& ElasticLove() const {
    return reference_.ElasticLoveNumbers();
  }
  /// Whether lateral moduli are set, and their sampled fields over the
  /// coupling's grid values: the relative shear perturbation, the
  /// absolute bulk perturbation delta kappa(x) (material units), and the
  /// absolute unrelaxed shear perturbation delta mu_U(x); empty where
  /// not set.
  /// Whether lateral moduli were given: the same on every rank of a
  /// distributed run, whose segment may hold no node where they are
  /// non-zero (its sampled arrays then empty), since the solve's route
  /// is collective.
  [[nodiscard]] bool HasLateralModuli() const { return lateralModuli_; }
  /// The scalar companions of this operator's iterate: the dilatation's
  /// block (-1 without a lateral bulk modulus).
  [[nodiscard]] int DilatationBlock() const { return dilatationBlock_; }
  [[nodiscard]] std::size_t CompanionCount() const { return companions_; }
  /// Weights a grid tensor's scalar companions by the step's negative
  /// lateral weights (-delta kappa on the dilatation): what Respond and
  /// the transposed passes apply.
  void WeightCompanions(GridCoupling::TensorField& field) const;
  [[nodiscard]] std::span<const double> ShearPerturbation() const {
    return shear_;
  }
  [[nodiscard]] std::span<const double> BulkPerturbation() const {
    return bulk_;
  }
  [[nodiscard]] std::span<const double> UnrelaxedShearPerturbation() const {
    return shearUnrelaxed_;
  }
  /// The unrelaxed shear modulus at each memory node of the reference
  /// (the long-term modulus interpolated plus the branches' at the
  /// points).
  [[nodiscard]] std::span<const double> UnrelaxedShearAtNodes() const {
    return muUnrelaxed_;
  }
  /// The reference's bulk modulus at each memory node, likewise.
  [[nodiscard]] std::span<const double> BulkModulusAtNodes() const {
    return kappaAtNodes_;
  }
  /// The factor 1 + s(x) every branch modulus carries at a grid value
  /// under the lateral shear (1 without).
  [[nodiscard]] double ShearFactor(std::size_t gridValue) const {
    return shear_.empty() ? 1.0 : 1.0 + shear_[gridValue];
  }
  /// The diagnostics of the last unrelaxed start solve under lateral
  /// moduli (SolveElastic / ElasticStrain / PrimeStrain / a StartStep
  /// without a cache): its iteration count and residual.
  struct ElasticSolveInfo {
    int iterations = 0;
    double residual = 0.0;
    bool converged = true;
    /// The relative residual of the start (the reference response's, or
    /// a warm start's) before the first iteration.
    double initial = 0.0;
  };
  [[nodiscard]] ElasticSolveInfo LastElasticSolve() const {
    return lastElasticSolve_;
  }

  /// The grid-held state: one five-component tensor per branch, plus the
  /// strain cache of the last consistent solve.
  struct State {
    double t = 0.0;
    std::vector<GridCoupling::TensorField> m;
    /// The strain cache: its grid values and, from a step or a priming
    /// solve, the coefficients they synthesise from (a step starts from
    /// those rather than analysing the grid values; either alone is
    /// enough to start from).
    std::optional<GridCoupling::TensorField> strain;
    std::optional<GridCoupling::SpectralTensor> strainCoefficients;
    /// Drops the strain cache, as after changing m by hand.
    void Invalidate() {
      strain.reset();
      strainCoefficients.reset();
    }
  };
  [[nodiscard]] State ZeroState() const;

  /// Per-degree solutions: spheroidal complex m >= 0 columns
  /// (ndof x (l + 1), column-major) and toroidal likewise (degrees >= 1;
  /// degree 1 in the operator's pinned gauge, its rigid rotations being
  /// gauge).
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

  /// The strain consistent with (m, t) under the given loads: one
  /// elastic solve. The unrelaxed operator is laterally uniform, so no
  /// iteration, and a step's end strain satisfies unrelaxed equilibrium
  /// with its new memory, so this regenerates the strain of a state
  /// recorded with its internal variables only.
  [[nodiscard]] GridCoupling::TensorField ElasticStrain(
      const State& state, const LoadCallable3D& load = {},
      const LoadCallable3D& tide = {}) const;

  /// Ensures the state carries a strain cache consistent with (m, t)
  /// under the given loads, solving once if not (the 1D operator's
  /// PrimeStrain).
  void PrimeStrain(State& state, const LoadCallable3D& load = {},
                   const LoadCallable3D& tide = {}) const;

  struct StepInfo {
    int iterations = 0;
    /// The last pass's residual; NaN when a pass went non-finite, which
    /// ends the iteration unconverged on the last finite iterate.
    double residual = 0.0;
    bool converged = true;
    /// The passes Anderson mixed.
    int mixedPasses = 0;
    /// Under StepSolver::ConjugateGradients, the conjugate-gradient
    /// iterations in all (the step's one solve, or a nested closure's
    /// solves summed).
    int solveIterations = 0;
  };

  /// The within-step fixed point's controls: at most max_iterations
  /// passes, converged when the relative sup-norm change of the strain's
  /// coefficients is at most rtol; damping scales the Picard part.
  ///
  /// The iteration starts as Picard, which is enough when the lateral
  /// term is weak and costs no history. Once a pass shrinks the residual
  /// by less than anderson_rate (0.5 by default, about where Anderson's
  /// fewer passes start to pay for its dearer ones), Anderson mixing with
  /// anderson_memory differences takes over for the rest of the step
  /// (docs/lateral-coupling.md). anderson_rate <= 0 mixes from the
  /// first pass; anderson_memory 0 never mixes.
  struct IterationControl {
    int max_iterations = 40;
    double rtol = 1e-11;
    double damping = 1.0;
    int anderson_memory = 5;
    double anderson_rate = 0.5;
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
  /// Distributed, collective: every process gets the values (the surface
  /// segment's process holds them).
  [[nodiscard]] std::map<int, std::vector<Complex>> Surface(
      const Solutions& solutions, Component component) const;

  /// The inertia moment of the degree-0, m = 0 column of solutions (zero
  /// without one); distributed, collective (each process's share summed).
  [[nodiscard]] double InertiaMoment(const Solutions& solutions) const;

 private:
  struct Flavour;  // reference effective systems per (scheme, dt)

 public:
  /**
   * @brief One opened implicit step, for drivers that close the step's
   * equilibrium themselves (the sea level coupling): the decayed memory
   * m~(x), the weight fields and the memory force they fix are set on
   * construction, and the driver alternates Respond / its own load
   * closure, then finishes.
   *
   * A step is a linear system in the degree solutions, (K_eff -
   * P^T W S) x = f: the reference effective systems K_eff, the lateral
   * weight W(x) of the viscosity's and the moduli's departures from the
   * reference applied pointwise to the strain S x (and the bulk
   * companion), the forces f the fixed memory force, any added, and the
   * channels'. Respond / StrainOf iterate it as a fixed point on the
   * strain; Forces / Solve close it by conjugate gradients on x, the
   * operator symmetric positive definite (the 3D effective stiffness)
   * and K_eff^-1 the preconditioner.
   *
   * The within-step iterate is the strain's COEFFICIENTS (a
   * SpectralTensor, 5 L^2 reals per node), not its values on the
   * oversampled grid: the grid strain exists only inside Respond and at
   * Finish, and the mixing and the convergence test run on the
   * coefficients, about 4.5 times fewer values.
   *
   * The step keeps its large buffers from pass to pass (Respond's grid
   * tensor, the analysed force, the iterate StrainOf fills), so an open
   * step is used from one thread at a time; its own loops run threaded
   * under Options::parallel. It holds its effective systems itself, so
   * the operator's cache may drop them (it keeps the four most recently
   * used (scheme, dt)) while the step is open.
   */
  class OpenStep {
   public:
    OpenStep() = default;
    OpenStep(const OpenStep&) = default;
    OpenStep& operator=(const OpenStep&) = default;
    OpenStep(OpenStep&&) noexcept = default;
    OpenStep& operator=(OpenStep&&) noexcept = default;
    /// Hands the step's large arrays back to the operator for the next
    /// StartStep (those Finish moved into the new state excepted), so a
    /// step must not outlive its operator.
    ~OpenStep();

    [[nodiscard]] double EndTime() const { return t_; }
    /// The start strain's coefficients: where the iteration starts.
    [[nodiscard]] const GridCoupling::SpectralTensor& StartStrain() const {
      return s0_;
    }

    /// The solutions of the (memory + lateral(d)) force through the
    /// reference effective systems, with NO load channels, for the strain
    /// with the given coefficients: what enters a sea level solver as
    /// ready response channels.
    [[nodiscard]] Solutions Respond(
        const GridCoupling::SpectralTensor& strain) const;

    /// Adds load/tide channels to zero-channel solutions by linearity
    /// (cached unit responses of the effective systems).
    void AddChannels(Solutions& solutions, const LoadCallable3D& load,
                     const LoadCallable3D& tide) const;

    /// The strain coefficients of solutions, for the next pass: into
    /// `out` (a tensor of this step's layout, from an earlier call, is
    /// zeroed and refilled without reallocating; an empty one is sized),
    /// or into a new tensor.
    void StrainOf(const Solutions& solutions,
                  GridCoupling::SpectralTensor& out) const;
    [[nodiscard]] GridCoupling::SpectralTensor StrainOf(
        const Solutions& solutions) const;

    using Fields = std::map<int, std::vector<Complex>>;
    /// The step's degree forces f: the fixed memory force (and the forces
    /// added) assembled through the effective systems, plus the load and
    /// tide channels' at the end time; spheroidal and toroidal, every
    /// degree present (distributed, this process's shares).
    [[nodiscard]] std::pair<Fields, Fields> Forces(
        const LoadCallable3D& load, const LoadCallable3D& tide) const;
    /// The step's operator on solutions, (K_eff - P^T W S) x.
    [[nodiscard]] std::pair<Fields, Fields> Apply(const Solutions& x) const;
    /// The step's solutions for given forces by preconditioned conjugate
    /// gradients (Options::solve_rtol, solve_max_iterations), from the
    /// reference response or, given, a start (the previous solution of a
    /// nearby system: a driver's outer iteration warm-starts each
    /// solve). The solutions carry whatever channels the forces did.
    /// Unconverged, the last iterate is returned with converged false.
    [[nodiscard]] std::pair<Solutions, ElasticSolveInfo> Solve(
        const Fields& spheroidal, const Fields& toroidal,
        const Solutions* start = nullptr, double rtol = 0.0,
        double reduction = 0.0) const;

    /// A driver's own closure on the step's solutions (the sea level
    /// equation on the surface response, say): given the CHANNEL-FREE
    /// solutions it sets its loads and returns the step's load and tide
    /// channels for the next solve.
    using Closure = std::function<std::pair<LoadCallable3D, LoadCallable3D>(
        const Solutions& responses)>;
    /// The coupled problem with the solve inside: each pass strips the
    /// last solution's channels, hands the closure the channel-free
    /// solutions, and solves the step under the channels it returns,
    /// warm-started from the last solution with the new channels (the
    /// Picard iterate) and inexactly (Options::solve_outer_factor). The
    /// outer residual is the warm start's: how far the last solutions
    /// with the new channels are from solving the step, relative to its
    /// forces in the solve's measure; converged when it is at most
    /// control.rtol. The outer map is Picard, Anderson-mixed on the
    /// channel-free solutions as Converge mixes the strain (the control's
    /// memory and rate). The first pass's channel-free solutions are the
    /// start strain's response. The closure runs once per pass, so
    /// whatever it records is from the last pass, consistent with the
    /// returned strain to the tolerance.
    [[nodiscard]] std::pair<GridCoupling::SpectralTensor, StepInfo> Nest(
        const Closure& closure, const IterationControl& control,
        Solutions* solutions = nullptr) const;
    /// The step under KNOWN channels, closed as Options::solver says: the
    /// fixed point of Respond / AddChannels / StrainOf under the control,
    /// or the one conjugate-gradient solve.
    [[nodiscard]] std::pair<GridCoupling::SpectralTensor, StepInfo> Close(
        const LoadCallable3D& load, const LoadCallable3D& tide,
        const IterationControl& control) const;

    /// One pass of a driver's within-step iteration: the strain
    /// coefficients the step's solutions take for given ones (Respond, the
    /// driver's own load closure, AddChannels, StrainOf).
    using Pass = std::function<GridCoupling::SpectralTensor(
        const GridCoupling::SpectralTensor&)>;

    /// The relative sup-norm change of the coefficients from now to next
    /// (Converge's measure: maxima of squared magnitudes, the same on any
    /// thread count; collective), or NaN where next is not finite.
    [[nodiscard]] double RelativeChange(const GridCoupling::SpectralTensor& next,
                                        const GridCoupling::SpectralTensor& now) const;

    /// The within-step fixed point s = pass(s) from the start strain,
    /// Anderson-mixed in place: the converged (or, when the budget runs
    /// out, the last) strain coefficients and the diagnostics. The pass
    /// runs once per iteration, so whatever it records is from the final
    /// pass.
    [[nodiscard]] std::pair<GridCoupling::SpectralTensor, StepInfo> Converge(
        const Pass& pass, const IterationControl& control,
        const GridCoupling::SpectralTensor* start = nullptr) const;

    /// The inertia moment of the degree-0, m = 0 column of solutions;
    /// distributed, collective (each process's share summed).
    [[nodiscard]] double InertiaMoment(const Solutions& solutions) const;

    /// The effective Love numbers of this step's reference solid (SI).
    [[nodiscard]] const LoveNumbers& EffectiveLove() const;

    /// Adds an increment to the step's decayed memory, one tensor per
    /// branch on the grid: it joins the fixed memory force and the end
    /// state alike — the drive of a tangent-linear step (the 1D
    /// ViscoelasticStep::AddMemory, pointwise). The first pass's force is
    /// rebuilt on the next Respond.
    void AddMemory(const std::vector<GridCoupling::TensorField>& increment);

    /// Adds a force to the step: an analysed tensor (the spectral
    /// coefficients of a weighted grid tensor, as the fixed memory force
    /// is) joining the fixed force, and per-degree spheroidal forces
    /// (the solutions' layout) added after the assembly — the drive of a
    /// tangent-linear step in the material (TangentLinearSolver3D): the
    /// effective system's variation and the memory force of the branch
    /// moduli's variation as the tensor, a bulk modulus perturbation as
    /// the degree forces. The first pass's force is rebuilt on the next
    /// Respond.
    void AddForce(const GridCoupling::SpectralTensor& analysed,
                  const std::map<int, std::vector<Complex>>& spheroidal);

    /// Completes the step for the final strain coefficients: the strain d
    /// on the grid, m+ = m~ + beta(x) d, with d and its coefficients
    /// cached. On a step that is done with
    /// (std::move(step).Finish(std::move(strain))) the step's own buffers
    /// and the coefficients become the new state instead of being copied.
    [[nodiscard]] State Finish(
        const GridCoupling::SpectralTensor& strain) const&;
    [[nodiscard]] State Finish(GridCoupling::SpectralTensor strain) &&;

   private:
    friend class Viscoelastic3D;
    friend class AdjointSolver3D;  // the discrete transpose reads the weights
    friend class TangentLinearSolver3D;  // the material drive reads the memory
    /// P^T W S x: the lateral force of solutions through the step's
    /// weight (the fixed force excluded), in the step's buffers.
    [[nodiscard]] std::pair<Fields, Fields> LateralForce(const Solutions& x) const;
    /// The step's effective systems and their unit responses (cached in
    /// the flavour), for the discrete transpose.
    [[nodiscard]] const DegreeOperator& Spheroidal(int l) const;
    [[nodiscard]] const ToroidalOperator& Toroidal(int l) const;
    [[nodiscard]] const std::vector<double>& Units(int l) const;
    const Viscoelastic3D* op_ = nullptr;
    std::shared_ptr<Flavour> flavour_;
    Scheme scheme_{};
    double dt_ = 0.0, t_ = 0.0;
    GridCoupling::SpectralTensor s0_;
    /// The analysed fixed memory force, sum_k mu_k(r) m~_k(x).
    GridCoupling::SpectralTensor fixed_;
    /// The lateral weight, sum_k mu_k(r) (beta_k(x) - beta_ref_k(r)).
    std::vector<double> weight_;
    std::vector<GridCoupling::TensorField> memory_;
    std::vector<std::vector<double>> beta_;
    /// Buffers reused from pass to pass: Respond's grid tensor and its
    /// analysed force, and a spent iterate for StrainOf to fill.
    mutable GridCoupling::TensorField grid_;
    mutable GridCoupling::SpectralTensor analysed_, spare_;
    /// Whether analysed_ holds the start strain's force, formed in
    /// StartStep from the grid strain it had, for the first Respond.
    mutable bool startForce_ = false;
    /// Per-degree spheroidal forces added after the assembly (AddForce).
    std::map<int, std::vector<Complex>> extraSpheroidal_;
  };

  /// Opens a step: the start strain comes from the state's cache or a
  /// consistent solve under the start-time channels. A step of length
  /// ZERO is the instantaneous (unrelaxed) problem as an open step: the
  /// weights are the identity (beta = 0, m~ = m, the effective systems
  /// the unrelaxed ones) and the within-step fixed point closes the
  /// instantaneous elastic response under lateral moduli — what the
  /// evolution's start, the sweeps' starts, equilibria and jumps need
  /// there (docs/adjoints.md §13k). Its grid-sized
  /// arrays (the decayed memory, the weights, the workspace) come from the
  /// arrays recycled since the last StartStep where there are any, and are
  /// allocated otherwise.
  [[nodiscard]] OpenStep StartStep(const State& state, double dt,
                                   Scheme scheme = Scheme::ExponentialTrapezoid,
                                   const LoadCallable3D& load = {},
                                   const LoadCallable3D& tide = {}) const;

  /// Hands back a state the caller is done with, so that the next
  /// StartStep reuses its tensors instead of allocating and zero-filling
  /// fresh ones: an evolution recycles each state a step replaces (a
  /// closed step hands back its own arrays itself). A state merely dropped
  /// is simply freed; arrays of another shape are dropped. Thread-safe.
  void Recycle(State&& state) const;

  /// Frees the arrays kept for reuse, as at the end of a run.
  void ReleaseScratch() const;

 private:
  friend class AdjointSolver3D;  // the sweeps transpose these internals
  friend class TangentLinearSolver3D;  // the forward-mode drives

  /// The arrays kept for reuse by StartStep (Recycle, ~OpenStep): at most
  /// one step's worth, under their own lock.
  [[nodiscard]] GridCoupling::TensorField TakeTensor() const;
  [[nodiscard]] std::vector<double> TakeField() const;
  void Give(GridCoupling::TensorField&& tensor) const;
  void Give(std::vector<double>&& field) const;
  std::unique_ptr<std::mutex> scratchMutex_ = std::make_unique<std::mutex>();
  mutable std::vector<GridCoupling::TensorField> spareTensors_;
  mutable std::vector<std::vector<double>> spareFields_;

  /// The lazily filled caches below (and the reference operator's) are
  /// filled under this lock, so the const methods may be called from
  /// several threads at once.
  [[nodiscard]] std::unique_lock<std::mutex> LockCaches() const {
    return std::unique_lock(*cacheMutex_);
  }
  [[nodiscard]] const DegreeOperator& Spheroidal(int l) const;
  [[nodiscard]] const ToroidalOperator& Toroidal(int l) const;
  /// The (scheme, dt) flavour, from a cache of the four most recently used
  /// (a variable-dt run would otherwise keep every dt's factorisations).
  [[nodiscard]] std::shared_ptr<Flavour> FlavourOf(Scheme scheme,
                                                   double dt) const;
  [[nodiscard]] const DegreeOperator& FlavourSpheroidal(Flavour& flavour,
                                                        int l) const;
  [[nodiscard]] const ToroidalOperator& FlavourToroidal(Flavour& flavour,
                                                        int l) const;
  /// The strain coefficients consistent with (m, t) under the given loads:
  /// one elastic solve (ElasticStrain before its synthesis).
  [[nodiscard]] GridCoupling::SpectralTensor ElasticStrainCoefficients(
      const State& state, const LoadCallable3D& load,
      const LoadCallable3D& tide) const;
 public:
  /// The unrelaxed reference systems (the Love-number operators of the
  /// rheology's material), built on first use.
  [[nodiscard]] const DegreeOperator& UnrelaxedSpheroidal(int l) const {
    return Spheroidal(l);
  }
  [[nodiscard]] const ToroidalOperator& UnrelaxedToroidal(int l) const {
    return Toroidal(l);
  }
  /// A zero spectral tensor shaped for this operator's iterate: with the
  /// scalar companions of its lateral moduli (the dilatation's).
  [[nodiscard]] GridCoupling::SpectralTensor ZeroSpectralFor() const;
  /// The spheroidal forces of a weighted tensor: the deviatoric assembly
  /// plus, where the tensor carries a scalar companion, the dilatation's.
  [[nodiscard]] std::map<int, std::vector<Complex>> SpheroidalForces(
      const GridCoupling::SpectralTensor& weighted,
      const std::function<const DegreeOperator&(int)>& spheroidal) const;
  /// The lateral part of the UNRELAXED operator on solutions: the force
  /// of delta mu_U(x) on the deviatoric strain and of delta kappa(x) on
  /// the dilatation (per degree, spheroidal and toroidal).
  [[nodiscard]] std::pair<std::map<int, std::vector<Complex>>,
                          std::map<int, std::vector<Complex>>>
  LateralElasticForce(const Solutions& x) const;
  /// The unrelaxed systems under the lateral moduli solving the given
  /// forces (every degree present): the preconditioned conjugate-gradient
  /// solve, recording LastElasticSolve.
  [[nodiscard]] Solutions LateralElasticSolve(
      const std::map<int, std::vector<Complex>>& sphForces,
      const std::map<int, std::vector<Complex>>& torForces) const;
  /// (K - L) x for reference systems K and a lateral force L x: the bands
  /// on the complex columns as real pairs, threaded over the degrees;
  /// distributed, the shares assembled in one collective and the
  /// products masked back to shares.
  [[nodiscard]] std::pair<std::map<int, std::vector<Complex>>,
                          std::map<int, std::vector<Complex>>>
  ApplyLateralOperator(
      const Solutions& x,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal,
      const std::function<std::pair<std::map<int, std::vector<Complex>>,
                                    std::map<int, std::vector<Complex>>>(
          const Solutions&)>& lateralForce) const;
  /// Preconditioned conjugate gradients on (K - L) x = f for reference
  /// systems K (the preconditioner, solving shares) and a lateral force
  /// L x, distributed like the rest (each process's shares, the pairings
  /// summed): what the unrelaxed solve and the step's solve share.
  [[nodiscard]] std::pair<Solutions, ElasticSolveInfo> ConjugateGradients(
      const std::map<int, std::vector<Complex>>& sphForces,
      const std::map<int, std::vector<Complex>>& torForces,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal,
      const std::function<std::pair<std::map<int, std::vector<Complex>>,
                                    std::map<int, std::vector<Complex>>>(
          const Solutions&)>& lateralForce,
      double rtol, int maxIterations, const Solutions* start,
      double reduction = 0.0) const;
  /// The UNRELAXED systems solving per-degree forces: the reference
  /// systems alone, or under lateral moduli the conjugate-gradient solve
  /// (the symmetric operator, so the transposed solve as well).
  [[nodiscard]] Solutions SolveUnrelaxed(
      const std::map<int, std::vector<Complex>>& sphForces,
      const std::map<int, std::vector<Complex>>& torForces) const;
  /// Adds the load/tide channels' forces at time t per degree (the unit
  /// load and tide vectors times the amplitudes) to spheroidal forces.
  void AddChannelForces(
      std::map<int, std::vector<Complex>>& forces, double t,
      const LoadCallable3D& load, const LoadCallable3D& tide,
      const std::function<const DegreeOperator&(int)>& spheroidal) const;
  /// The strain coefficients of per-degree solutions, fresh or added to
  /// a spectral tensor.
  [[nodiscard]] GridCoupling::SpectralTensor StrainCoefficients(
      const Solutions& solutions,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;
  void AddStrainCoefficients(
      GridCoupling::SpectralTensor& spectral, const Solutions& solutions,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;
 private:
  /// Solve all degrees of both blocks for a combined weighted tensor.
  [[nodiscard]] Solutions SolveAll(
      const GridCoupling::SpectralTensor& weighted,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;
  /// The solves alone: per-degree spheroidal and toroidal forces (complex
  /// m >= 0 columns) through the given systems.
  [[nodiscard]] Solutions SolveForces(
      const std::map<int, std::vector<Complex>>& sphForces,
      const std::map<int, std::vector<Complex>>& torForces,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      const std::function<const ToroidalOperator&(int)>& toroidal) const;
  /// Adds load/tide channels at time t by linearity, through the
  /// spheroidal systems' unit load and tide responses, solved once per
  /// degree into `units` and kept there.
  void AddUnitChannels(
      Solutions& solutions, double t, const LoadCallable3D& load,
      const LoadCallable3D& tide,
      const std::function<const DegreeOperator&(int)>& spheroidal,
      std::map<int, std::vector<double>>& units) const;

  std::shared_ptr<const Rheology> rheology_;
  int lmax_;
  Options options_;
  GridCoupling coupling_;
  ViscoelasticOperator reference_;         // the radial machinery and tables
  std::vector<std::vector<double>> itau_;  // per branch, (node x grid)
  std::vector<std::vector<double>> muBranch_;  // per branch, per node
  std::vector<std::vector<double>> itauRef_;   // per branch, per node
  /// The lateral moduli (Options::moduli), sampled: the relative shear
  /// perturbation and delta mu_U(x) over the grid values, delta kappa(x)
  /// and its negative (the step's weight) likewise, and the reference's
  /// unrelaxed shear per node; empty where not set.
  bool lateralModuli_ = false;
  std::vector<double> shear_, shearUnrelaxed_, bulk_, negativeBulk_;
  std::vector<double> muUnrelaxed_, kappaAtNodes_;
  int dilatationBlock_ = -1;
  std::size_t companions_ = 0;
  mutable ElasticSolveInfo lastElasticSolve_;
  mutable std::map<int, std::unique_ptr<DegreeOperator>> spheroidal_;
  mutable std::map<int, std::unique_ptr<ToroidalOperator>> toroidal_;
  mutable std::map<int, std::vector<double>> unrelaxedUnits_;  // (ndof x 2)
  mutable std::list<std::pair<std::pair<int, double>, std::shared_ptr<Flavour>>>
      flavours_;  // LRU, most recent at front
  std::unique_ptr<std::mutex> cacheMutex_ = std::make_unique<std::mutex>();
};

}  // namespace sl3dpp
