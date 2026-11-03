# Adjoints: discrete and continuous

*Drafted 16 October 2026 from pyslfp's recent adjoint work
(`viscoelastic/evolution.py`, commits `d215e3f`..`5479ce9`: discrete and
continuous adjoints, quelling, 1D viscous kernels, checkpointing) and
Yu et al. 2025 (equation references below are to that paper). pyslfp is
the porting template for the 1D path; the 3D path is sl3dpp's own.
Sections 6a and 6b record what is built in 3D and what is designed but
not built.*

## 1. Both formulations, continuous preferred (decided 16 Oct 2026)

Two adjoints are implemented as options, as pyslfp and AdGIA carry:

- **Discrete**: the exact transpose of the time-stepped forward map.
  Gradient identity to solver tolerance for any perturbation on the
  date grid; gated by dot-product tests at machine precision. Being
  the exact transpose of THIS code's step, it is also the
  porting-bug detector against pyslfp's (algebraically identical)
  implementation.
- **Continuous**: the adjoint equations of the continuum problem,
  discretised. They have the same form as the forward problem (Yu et
  al. §3.3), so between observations the adjoint state is stepped by
  the SAME machinery as a zero-load forward run in reversed time, with
  finite jumps of the adjoint internal variables at each observation
  (eq. 80). It never sees the stepper's internals, so it survives a
  change of scheme — and of discretisation.

**The continuous adjoint is the preferred production path** (David,
16 Oct 2026): the discrete adjoint gives a numerically exact gradient,
but not necessarily for the right problem — the two differ by the
discretise/transpose commute error, which is large precisely when the
run is under-resolved. For a good discretisation they agree to
practical accuracy, so their DISAGREEMENT is a resolution diagnostic
for the forward calculation itself, and both are kept callable behind
one interface and one result type so every gate runs on both alike.

## 2. Scope

1. **1D kernels** (the pyslfp port): ice-history kernels for objectives
   J = Σ_i ∫ g_i · (ξ, u, φ)(t_i) dS + g_ω·ω(t_i), by both sweeps;
   inverse-relaxation-time (itau) kernels, both ways, with the
   viscosity mapping; checkpointing (record stride + exact backward
   steps); the kernel views of §4.
2. **3D kernels**: the continuous adjoint of the coupled 3D
   evolution and the grid viscosity kernel μ_k m†:(d − m)/itau (§6,
   §6a); the discrete transpose of the reduced pipeline is designed
   (§6b) but not built.
   **Both kernel granularities are deliverables** (David, 16 Oct 2026):
   the full 3D viscosity kernel on the (memory nodes × angular grid),
   AND the spherically averaged 1D radial kernel as pyslfp provides —
   the latter both from the 1D sweeps directly and as the spherical
   average of the 3D kernel, whose agreement in the zero-lateral limit
   is a free consistency gate between the two pipelines.
3. Deferred when this was drafted: linearisation about a
   migrating-shoreline trajectory (pyslfp's `ocean_functions`), until
   sl3dpp's forward evolution grew migration. It was built on 25 October
   2026 with the migrating evolutions, 1D and 3D: the sweeps'
   `oceanFunctions` argument (design.md §7). Decay-time functionals (pyslfp
   `functionals.py`), a front end over Measurement, were ported later
   (`DecayTimeMeasurements` in `Measurements.hpp`).

## 3. The 1D sweeps (pyslfp's algebra, verbatim)

The authoritative statements are pyslfp `evolution.py` docstrings and
code; what follows fixes the sl3dpp mapping only.

- **Discrete** (`adjoint_ice_history`): one reversed sweep carrying the
  duals (m†, d†) per degree. Per step: the duals of the end solve
  gather through the strain transpose (`Degree::DualFromScalars` — the
  plain transpose of `Degree::Strain` under the nodal pairing); the
  unit load/tide responses contract them to load/tide coefficient
  duals; those join the measurement duals through the engine adjoint
  (`SeaLevelSolver::SolveAdjoint`, the sea-level solver's transpose); its channel
  duals force one symmetric effective solve per degree; the weight
  recursion closes m† and d†. The start is the transposed
  instantaneous elastic solve. Yu et al. §3; pyslfp lines 1554–1663.
- **Continuous** (`continuous_adjoint_ice_history`): reversed zero-load
  stepping by `StartStep`/`Finish` with the memory responses through
  the SLE's response channels — exactly the forward loop with no ice
  load. At an observation: the instantaneous adjoint fingerprint via
  the forward channel mapping (ζ_d = g_ξ, ζ^u = −g_u, ζ^φ = −g g_φ,
  L = −g (g_ω − Ψ* g_φ), with Ψ* the adjoint of the centrifugal
  operator), then the radial jump m† += itau d̃† with d̃† the strain of
  the unrelaxed solve under the converged (λ, ψ†) and the SPLIT load
  channels (`DegreeOperator::Load(LoadPart::Force/Potential)`) carrying
  displacement and potential measurements. The smooth kernel density is
  the memory-only adjoint equilibrium's sea level, kept one-sided
  across each observation (the density jumps there).
- **itau kernels.** Discrete: per step, the h-derivatives of the scheme
  weights (`ExponentialTrapezoidWeightDerivatives`,
  `BackwardEulerWeightDerivative`) pair the sweep's duals with the
  recorded forward internal variables and strains, and the effective
  modulus's dependence pairs the transposed and forward solutions
  through the branch bilinear (pyslfp lines 1603–1627). Continuous:
  dJ/d itau(r) = ∫ W_k m†/itau (d − m) dt over the reversed nodes with
  the split trapezoid (the adjoint state jumps at observations; the
  self-adjoint channel mapping poses the adjoint in the g-weighted
  energy form, so the carried radial state is g × the true multiplier —
  hence a division by g in material units). `IceKernels::Viscosity`
  maps to dJ/dη via dη = −η² d(itau)/μ per branch, as pyslfp.
- **Checkpointing.** The forward `record_stride` already stores the
  internal state every stride-th node and the per-node load/tide
  coefficients always; a `TrajectoryCursor` serves states strictly
  backwards, reconstructing gaps by `StepBack` under the recorded
  loads. *(17 Oct 2026: recordings keep the internal variables only, as
  in pyslfp and AdGIA; the cursor regenerates each stored state's strain
  by one elastic solve under that node's recorded loads, and the 3D
  sweep does the same per node.)* Roundoff grows as exp(Σh) across a gap — the stride buys
  memory at the price of trailing digits (pyslfp `_TrajectoryCursor`).

## 4. Result type and views

`IceKernels` carries the same three views as pyslfp, so the gates and
any inversion driver are formulation-blind:

- **Discrete view**: `loadDuals[n]` (dual of the direct load grid at
  node n), `thickness[n]` (grounding factor ρ_i (1−C) applied, the
  baseline −ρ_i (1−C) Σ_n loadDuals folded into node 0), `baseline`,
  `initialSeaLevel` (≡ 0 under frozen shorelines). Exact pairing on
  the time grid.
- **Distributional view** (Yu et al. §3.7, eq. 81): per-time `density`
  (converges under grid refinement) plus `instantaneous[i]` delta
  amplitudes — the unrelaxed fingerprint adjoint of each measurement,
  independent of the time grid.
- **Quelled view** (Backus): `rate[k]`, the kernel against the
  thickness RATE on interval k — the cumulative tail Σ_{n>k} of the
  thickness duals; bounded, with finite jumps at observations only;
  the initial-thickness dual vanishes identically.
- `itauKernel` per branch over (element, gll) when a trajectory is
  supplied, and `Viscosity(rheology)` mapping it to dJ/dη.

Conventions (sl3dpp's, not pyslfp's): times are the forward driver's —
increasing, SLE solver units; measurements are keyed by NODE INDEX into
the time grid (no float matching); duals are non-dimensional SLE-unit
densities pairing by ∫ · dS on the solver grid, the angular-velocity
dual by the dot product; densities/rates are per unit SLE time.

## 5. Architecture

- `include/sl3dpp/Adjoint.hpp` + `src/Adjoint.cpp`: `Measurement`,
  `IceKernels`, `ForwardTrajectory` (built from recorded snapshots),
  and the sweeps as methods of `SeaLevelEvolution`
  (`AdjointIceHistory`, `ContinuousAdjointIceHistory`), mirroring
  pyslfp's placement — the evolution already holds the operator,
  solver, state and unit ratios both sweeps need.
- Internals access: `ViscoelasticOperator` befriends a single
  `detail::AdjointAccess` defined in Adjoint.cpp (the pattern of
  `ViscoelasticStep`); nothing speculative is made public. The degree
  workspace already has the transposes (`DualFromScalars`,
  `momentRow`); it grows `ApplyWeights` (factored out of
  `MemoryForce`) and a surface-dual placement; the operator grows
  cached split unit responses and nodal weight derivatives beside
  `SchemeWeightsOf`.
- The 3D sweeps follow in `Evolution3D`/`Viscoelastic3D` once 1D is
  gated (§6).

## 6. The 3D path

- **Continuous first** (and preferred): the adjoint equations' form is
  again the forward's, so the reversed sweep reuses the coupled
  machinery of Evolution3D — zero-load reversed stepping with the
  within-step lateral iteration intact, jumps at observations through
  the unrelaxed 3D solve, the lateral term entering the fixed point
  exactly as forward. The viscosity kernel is the 3D form of Yu et al.
  2025's multiplier integral, as in pyslfp and the 1D sweep (not the
  Fortran's rate-form kernel expression): per branch and memory point,
  μ_k m†:(d − m)/itau, accumulated over the reversed nodes with the
  split trapezoid. *(Corrected 17 Oct 2026: this paragraph first planned
  the contraction μ s (d − m):(d† − m†) with metric signs from
  GSHTrans's tensor types; as built, §6a, the contraction pairs the
  adjoint memory with the forward d − m, and its metric on the reduced
  storage is fixed by the 1D limit.)* The reality reduction gives the derived spins by sign·conj for
  free at the contraction, and the adjoint fields are real tensors too.
- **Discrete option**: the exact transpose of the reduced pipeline —
  Scatter/batched-synthesis/pointwise-multiply/analysis/force-rows in
  reverse, the m ≥ 0 column reduction and the spin orbit relation
  transposed consistently (the pipeline ordering was fixed with this
  in mind), and the transpose of the converged within-step fixed point
  solved by the same Anderson-mixed iteration on the transposed
  system (as the SLE adjoint already does). This is the
  machine-precision gate for the 3D machinery; whether it is also
  worth production use the diagnostics will say.
- Gate chain: zero-lateral limit — the 3D sweeps must reproduce the 1D
  sweeps' kernels (machine precision for the discrete, tight for the
  continuous); dot-product identity for the discrete 3D transpose; FD
  checks on the grid viscosity kernel; discrete-vs-continuous ladder
  over lateral contrast (the resolution diagnostic in action — a
  research measurement alongside `lateral-coupling.md` §5a).

## 6a. The 3D continuous sweep as built

`Adjoint3D.hpp/cpp`: `AdjointSolver3D` (continuous only; the discrete
transpose is not built, §6b), `ForwardTrajectory3D` (full recording via
`SeaLevelEvolution3D::Options::record_stride` — no 3D backward
reconstruction yet, so the viscosity kernels demand stride 1; since 17
Oct, performance chunk 10, the sweep rebuilds the gaps by forward steps
from the kept states, so any stride serves), and
`Kernels3D`: the 1D `IceKernels` views, `itauKernel3D` (per branch, an
ANGULAR DENSITY over memory node × coupling grid, radial quadrature
folded in), its angular integral `itauKernelSpherical` (the 1D bridge),
and `logViscosityKernel3D` = −itau(x)·K (the operator's delta
parameterisation).

The pieces, each mirroring a gated 1D counterpart:
- the observation JUMP is the 1D one (the unrelaxed problem is
  laterally uniform): the elastic adjoint fingerprint's channels drive
  the reference unit/split responses, the strain enters the pipeline
  via `AddStrainOfSolution` + `Synthesise`, and
  m† += itau(x)·d̃†(x) pointwise with the laterally varying itau;
- between observations the reversed stepping is Evolution3D's coupled
  pass loop verbatim with no ice load (`OpenStep::Respond` /
  `AddChannels` / `StrainOf` / `Finish` under the Anderson mixer);
- the kernel contraction's null-basis metric is FIXED by the 1D limit:
  on the reduced storage, C(x) = r²·[12·A₀B₀ + 8 Re(A₋₁B̄₋₁) +
  4 Re(A₋₂B̄₋₂)] — derived by requiring the angle-uniform Parseval
  reduction to reproduce the force weights (4/3, k², k²(k²−2)) through
  `ComponentsFromScalars`, and verified by the zero-lateral gate;
- the covariant pairing ratio of the 1D sweep (the pyslfp
  units-covariance fix) carries over unchanged.

GATES (TestAdjoint3D): zero-lateral — ice kernels and the spherical
reduction equal the 1D continuous sweep's at 1e-6 through the whole
pipeline; lateral — the full 3D log-viscosity kernel against central
differences of the coupled forward along an analytic direction, at the
commute level (< 10%) in the resolved regime and improving under step
refinement; and the canonical TAYLOR TEST (added 16 Oct 2026 at
David's prompt): |J(δ+hp) − J| = O(h) and the gradient-corrected
remainder |J(δ+hp) − J − h⟨K,p⟩| at second order on the
curvature-dominated end of the h-sweep, monotone, and inside the
commute floor (0.1·h·|⟨K,p⟩|) everywhere. The machine-grade Taylor
test — second order to solver precision at every h — belongs to the
discrete transpose (§6b).

## 6b. The discrete 3D transpose: design (not built)

The step is LINEAR throughout, so its exact transpose is mechanical
once the pairings are fixed. Write the forward within-step pass (the
code of Evolution3D::Run and OpenStep) as d' = Exit K⁻¹ [Entry(m̃ +
Δβ d) + units·closure(f, S K⁻¹ Entry(...))] with
- Entry(T) = SpheroidalForce/ToroidalForce(Analyse(μ_k-scaled T)) — the
  grid entry with branch moduli,
- Exit(u) = Synthesise(AddStrainOfSolution(u)) — the strain exit,
- K the reference effective systems (symmetric), S the surface reads,
  closure the (linear) generalised SLE, Δβ = diag(β_k(x) − β_ref_k(r)).

*Note (17 Oct 2026, performance chunk 3; `walkthrough-3d-step.md` §4).*
The forward code now iterates on the strain's COEFFICIENTS s, not its
grid values d. A pass is s' = AddStrainOfSolution(K⁻¹[fixed +
Analyse(W · Synthesise(s)) + channels]), where fixed = Analyse(Σ_k μ_k
m̃_k) is formed once per step and W = Σ_k μ_k Δβ_k(x) is a single weight
field. The step map, and so everything in this section, is unchanged.
Only the code's split has moved: Exit now stops at the coefficients,
Synthesise opens the next pass, and Entry's constant part is applied
once per step. A transposed iteration can likewise run on coefficient
duals.

**The load-bearing duality**: Entry = Exitᵀ under the ENERGY pairing
⟨S, T⟩_W0 = Σ_node Wq(node) · r² [12 S₀T₀ + 8 Re(S₋₁T̄₋₁) +
4 Re(S₋₂T̄₋₂)] on the reduced components against the REAL dof pairing
with m ≥ 0 column weights (1, 2, 2, ...) — the SAME metric the
continuous kernel contraction derived from the 1D force weights, and
which the zero-lateral kernel gate already validated. With duals
carried in this pairing, the transposed pass is the forward pass with
the force/strain pair swapped and the SLE replaced by its
adjoint:

  passᵀ(y): x† = Entry-with-Δβμ-weights(y) … K⁻¹ (symmetric) …
            unit-response contraction → SLE ADJOINT → surface-dual +
            moment-row injections → K⁻¹ … Exit → Δβ-weighted tensor.

The transposed step: w = d† + Σ_k β_k(x) m†_k; solve y = Mᵀy + w by the
SAME Anderson-mixed iteration (the transpose of the converged fixed
point is the transposed system's solution, as the SLE adjoint already
does); then one application of the constant part's transpose yields
m̃†_k (→ the e/α recursion closing m† and d†, pointwise fields) and the
node's load dual; the measurement duals enter the forcing w through one
SLE-adjoint chain per node, as in the 1D discrete sweep.

**Implementation order, micro-gated** (each pinned by a randomized
adjointness identity at lmax 4 with Hermitian data BEFORE the next
layer — the reduced pipeline's ±m bookkeeping lives inside
Scatter/Gather and must be verified, not trusted):
  i.   ⟨Exit(x), y⟩_W0 = ⟨x, Entry(W0-images of y)⟩_dof — the
       force/strain duality with every quadrature factor;
  ii.  the pass transpose: ⟨dNext(d), y⟩_W0 = ⟨d, passᵀ(y)⟩_W0 with
       the SLE closure in the loop (random smooth state);
  iii. the single-step transpose: the dot-product identity of one
       coupled Step against (m†, d†) duals;
  iv.  the sweep: gradient identity at solver tolerance (the forward's
       coupled_rtol is the floor), zero-lateral == the 1D DISCRETE
       sweep, machine-grade Taylor test (second order at every h), and
       the continuous-vs-discrete contrast ladder (the §5a-style
       research measurement).

## 7. Implementation order and gates

a. **Discrete 1D sweep** (ice kernels, full Measurement). GATES:
   dot-product/gradient identity at solver tolerance against the
   forward driver for random perturbations on the time grid; pyslfp
   fixture comparison (identical algebra — near machine precision).
b. **Continuous 1D sweep**. GATES: agreement with (a) at
   discretisation accuracy, tightening under dt refinement; the
   singular-part separation and quelled-view identities (pyslfp's
   `test_kernel_singular_parts_are_separated`, `test_quelled_rate_kernels`).
c. **itau/viscosity kernels + checkpointing**, both sweeps. GATES: FD
   kernel checks; checkpointed == full-record to the exp(Σh) budget;
   pyslfp fixtures.
d. **3D continuous sweep + grid viscosity kernel** (§6). GATES:
   zero-lateral limit; FD checks; Δt refinement.
e. **3D discrete transpose**. GATES: dot-product at machine precision;
   zero-lateral == 1D discrete; the contrast-ladder diagnostic.

Fixtures: `scripts/make_kernel_fixtures.py` driving pyslfp, committed
under `tests/data/` with provenance headers, like the other fixtures.

## 8. The tangent-linear runs (built 3 November 2026)

The forward mode, ported from pyslfp's `tangent_linear_ice_history`
(its docstring and `docs/planning/viscoelastic_operator_plan.md`, "The
tangent-linear problem", are the authoritative statements; the first
step of the second-derivative work, whose plan is pyslfp's). With the
trajectory's ocean functions held the forward problem is linear in its
loads, so the derivative along (δf, δI, δS₀, δitau) is the forward
problem once more:

- the shorelines prescribed from the trajectory (`oceanFunctionOverride`
  on every generalised solve, as the sweeps do);
- the direct load δfₙ + ρ_i (1 − Cₙ) δIₙ − ρ_i (1 − C₀) δI₀ +
  ρ_w (Cₙ − C₀) δS₀, with the start the instantaneous elastic response to
  its first node (zero for thickness and sea level directions);
- per step, the opened step's decayed memory state shifted by a drive
  (`ViscoelasticStep::AddMemory`), bilinear in the trajectory's (m, d) at
  the step's ends and δitau at the memory points (the forward map of
  `ItauKernelsAtNodes`' transpose, `ItauPerturbationAtPoints`):
  - Discrete: dh (de m⁰ + dα d⁰ + dβ d¹) for the trapezoid, dw dh (m⁰ − d¹)
    for backward Euler, dh = dt δitau — the forward mode of the scheme
    weights, whose reverse mode is the discrete sweep's itau pairing;
  - Continuous: α q⁰ + β q¹ (trapezoid) or (1 − w) q¹ (backward Euler)
    with qⁿ = (dⁿ − mⁿ_k) δitau_k/itau_k — the continuous drive entering
    like the strain, whose transpose is the continuous sweep.

The two coincide in the loads and differ in itau by the order of the
scheme. The forward trajectory is served forwards by `ForwardCursor`:
stored states where the recording kept them (strain regenerated by one
elastic solve under the recorded loads), exact forward steps under the
recorded loads between — which amplify nothing, unlike the reverse
cursor's backward reconstruction. With `Options::record` every snapshot
keeps the tangent-linear state with its strain and the node's load and
centrifugal coefficients: what the second-order runs read.

Weights: `StepWeights.hpp` gained the second h-derivatives (d2e = e,
d2α = d2φ₁ − e, d2β = −d2φ₁ with d2φ₁ = −(e + 2 dφ₁)/h and its series
below the switch; 2/(1 + h)³ for backward Euler), cached by the operator
beside the weights and their first derivatives (`WeightsOfOrder`).

GATES (`TestTangentLinear`): linear in the loads at 1e-8 against the
difference of two forward runs; the discrete pair exact transposes — the
objective of the run along (δf, δitau) equals the pairing of the load
duals (node 0 included) and the itau kernel at 1e-8, each argument alone
too, and a stride-2 trajectory reproduces the full one at 1e-10; the
continuous drive converging to the discrete at the scheme's order (gap
< 0.2, falling by > 3 as the steps halve, η = 1e22 Pa s, 10 kyr ramp);
about a migrating marine-cap trajectory, the run along thickness and
initial-sea-level perturbations that flip no flotation cell equal to
central differences of migrating runs at 1e-7, both methods, and the
kernels' thickness and initial-sea-level views pairing to the same
number; the initial sea level inert under frozen shorelines.

**The second-order run** (`RunSecondOrder`, built the same day; pyslfp's
`second_order_tangent_linear_ice_history` under the sharp flotation
rule): F''(x)[u, v] as the forward problem with no direct load, the
shorelines prescribed, and the drive the sum of the first-order drive
taken with v's recorded tangent trajectory and u's δitau, with u's
trajectory and v's δitau, with the forward trajectory and the second
variation itau_uv (empty for a parameterisation linear in itau), and, by
the discrete method, the second derivatives of the scheme weights with
the forward trajectory and both perturbations (dh_u dh_v (d2e m⁰ +
d2α d⁰ + d2β d¹), or d2w dh_u dh_v (m⁰ − d¹)). With the ocean functions
held the map is affine in the loads and the initial sea level, so those
directions enter only through the tangent trajectories. The smoothed
flotation rule's shoreline terms (pyslfp's `flotation_curvature`) are
not ported: sl3dpp has no smoothed rule. Both runs share one loop
(`March`: prescribed shorelines, a direct load per node, a memory
increment per step).

GATES: the second-order run along (u, v) is the derivative, at the base
point along u, of the first-order run along v — central differences
with the loads moved by ε δu and the rheology by ε δitau_u converge at
second order (ratio in 3–5) and Richardson lands on it at 1e-6, both
schemes — and is symmetric in u and v at 1e-10; it vanishes identically
without a viscosity perturbation, about frozen and migrating runs alike;
by the continuous method it converges to the discrete one at the
scheme's order (gap 0.11 → 0.027 from 17 to 33 nodes over 10 kyr at
η = 1e22; the coarser grids, h ≈ 1, are pre-asymptotic); unrecorded
tangent runs are refused.

## 9. The second-order adjoint (built 3 November 2026)

`AdjointSolver::SecondOrderKernels` (`AdjointSecondOrder.cpp`; pyslfp's
`second_order_adjoint_ice_history` and `_second_order_continuous_sweep`
under the sharp flotation rule, Yu et al. 2025 §4): the kernels of the
objective and their derivative along a direction v — the action of the
objective's Hessian on v, returned as `KernelDerivatives` (the thickness
and initial-sea-level views' derivatives in the kernels' own pairings,
and the itau kernel's). Inputs: the recorded forward trajectory, v's
tangent-linear run recorded at every node by the same method, v's
δitau, and a migrating run's ocean functions. The sweep that gives the
kernels is itself a function of the parameters, through the forward
trajectory and the relaxation times; differentiating it along v gives a
second sweep of the same form, forced by the first sweep's state and
δitau, with v's tangent trajectory entering where the forward one enters
the kernels.

- **Discrete** (the exact derivative of the exact gradient; the Hessian
  symmetric to rounding): per reversed step, beside the first sweep's
  (m†, d†), the derivative duals (dm†, dd†). The first sweep's solve is
  the first-order one; the second sweep's dual of the step's solution is
  dd† + Σ B dm† + Bh dh m† through the strain transpose plus the source
  Σ Bh dh W s_y (the effective system's derivative acting on the first
  sweep's solution), contracted with the unit responses into load and
  tide duals and passed through the SLE adjoint with no measurement; its
  radial solve closes dm̃† = dm† + W d s_y. The pairings g (the itau
  kernel) and dg (its derivative: the same expression with each factor
  replaced in turn by its derivative — the dual state's, the tangent
  trajectory's (v m⁰, v d⁰, v d¹), and the weights' second derivatives
  times dh) accumulate at the memory points; the recursion closes both
  states (dm† = E dm̃† + Eh dh m̃†, dd† = Σ A dm̃† + Ah dh m̃†). The start
  is the transposed instantaneous solve for both dual states. Weights in
  the trapezoid's (E, A, B) form, backward Euler as (w, 0, 1 − w).
- **Continuous**: the adjoint state and its derivative are both stepped
  by the forward machinery in reversed time; the derivative state's step
  carries the continuous first-order drive built from the adjoint
  trajectory (`TangentLinearSolver::FirstOrderDrive` with the primed
  pre-step state and the post-step state); at an observation both jump
  by the instantaneous adjoint strain, over itau for the state and by
  δitau for its derivative; the viscosity pairing is differentiated with
  the derivative state and v's tangent trajectory in place, less
  (δitau/itau) times the first-order pairing. Assembly as the first
  sweep's (split node weights, lumps on the kernels only).

The shoreline terms of pyslfp's smoothed flotation rule are not ported.
The sweeps' shared pieces (the trajectory cursor, dual packers, node
weights, measurement check) moved from `Adjoint.cpp` into
`detail/AdjointInternals.hpp`.

GATES (`TestSecondOrderAdjoint`): the kernels the second-order sweep
carries equal the first-order sweep's at 1e-12, both methods, about
frozen and migrating runs; along v, central differences of the gradient
paired with u converge at second order to the derivatives paired with u
and Richardson lands on them at 1e-6, both schemes; ⟨H u, w⟩ = ⟨H w, u⟩
at 1e-9 and equals ⟨g, F''[u, w]⟩ from the second-order tangent-linear
run at 1e-9, frozen and migrating; the continuous derivatives converge
to the discrete block by block (ice, viscosity) and their mismatch with
the continuous second-order run falls at the scheme's order (17 → 33
nodes, η = 1e22), the discrete pair matching at 1e-9 at every spacing.

## 10. The 3D tangent-linear runs (built 3 November 2026)

`TangentLinearSolver3D` (`TangentLinear3D.hpp/cpp`; sl3dpp's own — pyslfp
has no 3D): the derivative of the coupled lateral-viscosity evolution
along a direction in the direct load, the ice thickness, the initial sea
level and the laterally varying inverse relaxation times, and the
second-order run along two directions. The direction in itau is per
branch over the coupling's grid values (the layout of
`Kernels3D::itauKernel3D`), or a log-viscosity field δδ(x) which maps
to it by δitau_k = −itau_k(x) δδ (`ItauOfLogViscosity`; the second
variation of the log parameterisation, itau u v, by
`SecondVariationOfLogViscosity`).

The algebra is the 1D one pointwise. The coupled step is linear in its
loads and its memory with coefficients that depend on itau(x) through
the pointwise weights alone: the decayed memory e(x) m + α(x) d⁰, the
lateral weight Σ μ_k (β_k(x) − β_ref) in the fixed point's force, and
the finish m⁺ = m̃ + β(x) d. Differentiating each in itau(x) gives
terms that together are exactly the 1D drive dh (de m⁰ + dα d⁰ + dβ d¹)
added to the decayed memory — `OpenStep::AddMemory`, which raises the
fixed memory force Σ μ_k m̃_k and the end state alike, so the within-step
iteration, the reference effective systems and the sea level closure
run unchanged (the lateral weight W δd is the trajectory's). The
continuous drive α q⁰ + β q¹, q = (d − m) δitau/itau, is pointwise too.
The forward trajectory is served forwards by `ForwardCursor3D`: kept
states as recorded, the gaps rebuilt by forward coupled steps under the
recorded loads with the run's own iteration (the adjoint's rebuild, in
the forward direction). Snapshots are `SeaLevelEvolution3D`'s, recording
the tangent state with its strain for the second-order run.

GATES (`TestTangentLinear3D`, lmax 8): zero-lateral — with δ = 0 at the
GLL memory points the 3D run equals the 1D run along loads and an
angle-uniform itau perturbation at 1e-7, both methods; with lateral
structure on, linear in the loads at 1e-8 against two coupled runs; the
discrete run along a log-viscosity direction p is the exact derivative
of the coupled forward (central differences with the operator's δ moved
by ε p converge at second order, Richardson to 1e-6), and the continuous
run pairs with the continuous 3D kernel ⟨K_δ, p⟩ at the commute level
(< 10 %); the second-order run along loads and log-viscosity directions
(with itau_uv) is the derivative along u of the first-order run along v
at the moved operator (second order, Richardson to 1e-6), symmetric at
1e-9, and zero without a viscosity perturbation.

Next: the 3D second-order adjoint (the continuous second-order sweep on
`Evolution3D`, the derivative adjoint state stepped with the continuous
drive of the adjoint trajectory, pointwise — Hessian-vector products of
3D viscosity objectives), then the forward perturbations in the elastic
parameters and the density.

