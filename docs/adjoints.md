# Adjoints: discrete and continuous

*The adjoints and kernels of sl3dpp, 1D and 3D. The 1D path follows
pyslfp's adjoint machinery (`viscoelastic/evolution.py`: discrete and
continuous adjoints, quelling, 1D viscous kernels, checkpointing) and
Yu et al. 2025 (equation references below are to that paper); the 3D
path is sl3dpp's own. Sections 6a and 6b give the 3D continuous sweep
and the design of the discrete 3D transpose; §8–§17 the tangent-linear
runs, the second-order adjoints, the discrete 3D transpose (§12), the
material derivatives and kernels, and their distribution over radius.
The overview is [design.md](design.md) §7.*

## 1. Both formulations, continuous preferred

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

**The continuous adjoint is the preferred production path**: the
discrete adjoint gives a numerically exact gradient, but not
necessarily for the right problem — the two differ by the
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
   §6a), and the discrete transpose of the reduced pipeline, designed in
   §6b and built (§12).
   **Both kernel granularities are deliverables**:
   the full 3D viscosity kernel on the (memory nodes × angular grid),
   AND the spherically averaged 1D radial kernel as pyslfp provides —
   the latter both from the 1D sweeps directly and as the spherical
   average of the 3D kernel, whose agreement in the zero-lateral limit
   is a free consistency gate between the two pipelines.
3. Linearisation about a migrating-shoreline trajectory (pyslfp's
   `ocean_functions`), 1D and 3D: the sweeps' `oceanFunctions` argument
   (design.md §7). Decay-time functionals (pyslfp `functionals.py`), a
   front end over Measurement: `DecayTimeMeasurements` in
   `Measurements.hpp`.

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
  loads. Recordings keep the internal variables only, as in pyslfp and
  AdGIA; the cursor regenerates each stored state's strain by one elastic
  solve under that node's recorded loads, and the 3D sweep does the same
  per node. Roundoff grows as exp(Σh) across a gap — the stride buys
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
  supplied, and `Viscosity(rheology)` mapping it to the viscosity's.

### The kernels' convention

Every kernel with respect to a field in the Earth — the relaxation
times', the viscosity's, the moduli's, in the time-stepped and the
static problems, their derivatives in the second-order sweeps — is a
**density per unit volume**, held as samples of a function at the
points where the parameter lives. With K the kernel and δ the
perturbation of the parameter,

    dJ = ∫ K δ dV.

In 3D the samples are K(x) on each memory node's grid, and

    dJ ≈ Σ_node w_node r_node² ∫ K δ dΩ

(`VolumePairing(coupling, K, δ)`; w the node's radial quadrature
weight). In 1D the samples are K(r) at the memory points or the nodes,
and for the spherically symmetric perturbation δ(r)

    dJ = 4π ∫ K(r) δ(r) r² dr ≈ Σ_p W_p K(r_p) δ(r_p),

W_p = 4π r_p² times the point's radial quadrature weight
(`MemoryPoints::VolumeWeights`, `VolumePairing(points, mesh, K, δ)`).
Neither the r² nor the 4π is in the kernel: both belong to the volume.
So a 1D kernel is the same function as the 3D kernel of the same
model, and a 3D kernel's reduction per node (`…Spherical`,
`SphericalMean`) is its mean over the sphere, the kernel of an
angle-uniform perturbation.

The values are non-dimensional in the material's scheme: the
objective's unit per unit of the parameter and of volume. The programs
write them in SI, the objective's unit per parameter unit per m³.

The discrete and the continuous sweeps both approximate these
functions. The sweeps accumulate the gradient with respect to each
point value, which is the kernel times the point's volume weight, and
divide the weights out on return; the pairing above with the weights
restored is then the exact discrete derivative for the discrete
sweeps. A node at the centre has no volume: its sample is the element's
polynomial through the other nodes and its gradient entry is not kept.
The entry is zero except for the nodal relaxation-time kernel of Gauss
memory points in an element at the centre that relaxes, where the
pairing misses the sensitivity to the value at the centre alone (about
1e-5 of that element's).
The kernels with respect to the ice history are fields on the surface
grid and pair by ∫ · dS (below).

Conventions (sl3dpp's, not pyslfp's): times are the forward driver's —
increasing, SLE solver units; measurements are keyed by NODE INDEX into
the time grid (no float matching); duals are non-dimensional SLE-unit
densities pairing by ∫ · dS on the solver grid, the angular-velocity
dual by the dot product; densities/rates are per unit SLE time.

## 5. Architecture

- `Adjoint.hpp`: `Measurement`, `IceKernels`, `KernelDerivatives`,
  `ForwardTrajectory` (built from recorded snapshots) and
  `AdjointSolver`, which borrows a `SeaLevelEvolution` (the operator,
  solver, state and unit ratios the sweeps need). Its
  `IceHistoryKernels` runs either 1D sweep by `Options::method`
  (`Adjoint.cpp`); its `SecondOrderKernels` the second-order sweeps
  (`AdjointSecondOrder.cpp`, §9). The pieces they share (the reverse
  trajectory cursor, the dual packers, the node weights, the measurement
  check) are in `detail/AdjointInternals.hpp`.
- `Adjoint3D.hpp`: `ForwardTrajectory3D`, `Kernels3D`,
  `KernelDerivatives3D` and `AdjointSolver3D` over a
  `SeaLevelEvolution3D`. `IceHistoryKernels` and `SecondOrderKernels` run
  the continuous sweep, one core in `Adjoint3D.cpp` (§6a, §11);
  `Options::method = Discrete` the discrete transpose in
  `AdjointDiscrete3D.cpp` (§12).
- `TangentLinear.hpp`, `TangentLinear3D.hpp`: `TangentLinearSolver` and
  `TangentLinearSolver3D`, the forward mode (§8, §10).
- Internals access is by friendship; nothing speculative is made
  public. `ViscoelasticOperator` befriends `AdjointSolver`,
  `AdjointSolver3D` (the 3D continuous sweep's jump) and
  `TangentLinearSolver`, and `ViscoelasticStep` befriends
  `TangentLinearSolver` (`Viscoelastic.hpp`); `Viscoelastic3D` and its
  `OpenStep` befriend `AdjointSolver3D` and `TangentLinearSolver3D`
  (`Viscoelastic3D.hpp`); each evolution befriends its adjoint and
  tangent-linear solvers. The degree workspace
  (`detail/ViscoelasticDegree.hpp`) carries the transposes
  (`DualFromScalars`, `momentRow`, `ApplyWeights`); the operator caches
  the split unit responses (`SplitUnitResponses`) and the scheme weights
  with their h-derivatives (`SchemeWeightsOf`, `WeightsOfOrder`).

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
  split trapezoid. The contraction pairs the adjoint memory with the
  forward d − m, and its metric on the reduced storage is fixed by the 1D
  limit (§6a), not taken from GSHTrans's tensor types. The reality
  reduction gives the derived spins by sign·conj for free at the
  contraction, and the adjoint fields are real tensors too.
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

`Adjoint3D.hpp/cpp`: `AdjointSolver3D` (the continuous sweep; the
discrete transpose is built too, §12), `ForwardTrajectory3D` (a recording
at any `SeaLevelEvolution3D::Options::record_stride`: the sweep rebuilds
the gaps by forward steps from the kept states), and
`Kernels3D`: the 1D `IceKernels` views, `itauKernel3D` (per branch, the
density per unit volume over memory node × coupling grid, §4), its mean
over the sphere `itauKernelSpherical` (the 1D bridge),
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
refinement; and the canonical Taylor test: |J(δ+hp) − J| = O(h) and the
gradient-corrected remainder |J(δ+hp) − J − h⟨K,p⟩| at second order on
the curvature-dominated end of the h-sweep, monotone, and inside the
commute floor (0.1·h·|⟨K,p⟩|) everywhere. The machine-grade Taylor
test, second order to solver precision at every h, is the discrete
transpose's (§12).

## 6b. The discrete 3D transpose: design (built, §12)

The step is LINEAR throughout, so its exact transpose is mechanical
once the pairings are fixed. Write the forward within-step pass (the
code of Evolution3D::Run and OpenStep) as d' = Exit K⁻¹ [Entry(m̃ +
Δβ d) + units·closure(f, S K⁻¹ Entry(...))] with
- Entry(T) = SpheroidalForce/ToroidalForce(Analyse(μ_k-scaled T)) — the
  grid entry with branch moduli,
- Exit(u) = Synthesise(AddStrainOfSolution(u)) — the strain exit,
- K the reference effective systems (symmetric), S the surface reads,
  closure the (linear) generalised SLE, Δβ = diag(β_k(x) − β_ref_k(r)).

*Note.* The forward code iterates on the
strain's COEFFICIENTS s, not its grid values d. A pass is
s' = AddStrainOfSolution(K⁻¹[fixed + Analyse(W · Synthesise(s)) +
channels]), where fixed = Analyse(Σ_k μ_k m̃_k) is formed once per step
and W = Σ_k μ_k Δβ_k(x) is a single weight field. The step map, and so
everything in this section, is the same; only the code's split differs:
Exit stops at the coefficients, Synthesise opens the next pass, and
Entry's constant part is applied once per step. A transposed iteration can likewise run on coefficient
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

The gates of the transpose as built are in §12.

## 7. The pieces and their gates

a–d are described in §3 and §6a, e in §12.

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
e. **3D discrete transpose** (§12). GATES: dot-product at machine
   precision; zero-lateral == 1D discrete; the contrast-ladder
   diagnostic.

Fixtures: `scripts/make_kernel_fixtures.py` driving pyslfp, committed
under `tests/data/` with provenance headers, like the other fixtures.

## 8. The tangent-linear runs

The forward mode, ported from pyslfp's `tangent_linear_ice_history`
(its docstring is the authoritative statement). With the
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

Weights: `StepWeights.hpp` carries the second h-derivatives (d2e = e,
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

**The second-order run** (`RunSecondOrder`; pyslfp's
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

## 9. The second-order adjoint

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
weights, measurement check) are in `detail/AdjointInternals.hpp`.

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

## 10. The 3D tangent-linear runs

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

## 11. The 3D second-order adjoint

`AdjointSolver3D::SecondOrderKernels`: the continuous sweep of §6a runs
as one core (`Sweep`) that carries, when given v's recorded 3D
tangent run and δitau(x), a second carried state — the derivative of
the adjoint state along v — beside the first. The derivative state is
stepped by the same reversed coupled machinery, its step driven by the
continuous first-order drive of the adjoint trajectory
(`TangentLinearSolver3D::FirstOrderDrive` with the primed pre-step
adjoint state and the post-step one, through `OpenStep::AddMemory`);
both states jump at an observation by the instantaneous adjoint strain,
over itau(x) and by δitau(x); and the pointwise pairing is
differentiated — the derivative state against the forward d − m, the
adjoint state against v's tangent d − m, less (δitau/itau) times the
pairing itself — with the same null-basis metric. The result is
`KernelDerivatives3D`: the thickness and initial-sea-level views'
derivatives, the itau kernel's derivative on the grid and its spherical
reduction, and the log-viscosity kernel's derivative
−(δitau K + itau dK), the Hessian-vector product of the log
parameterisation. `IceHistoryKernels` is the same sweep with no
tangent run, so the first-order gates guard it too.

GATES (`TestSecondOrderAdjoint3D`, lmax 8): zero-lateral — with δ = 0 at
the GLL points the kernels and the derivatives (thickness views,
spherical reductions of the itau kernel and its derivative) equal the
1D continuous second-order sweep's at 1e-6 along a direction
angle-uniform in itau; lateral — the carried kernels equal the
first-order sweep's at 1e-10, and ⟨H u, w⟩ against ⟨H w, u⟩, against
⟨g, F''[u, w]⟩ from the 3D second-order run (the log parameterisation's
uv term accounted for), and against central differences of the
continuous kernels along u, all agree below 10 % at 9 nodes and improve
at 17, on the η = 1e22 mantle — the continuous formulation's commute
level, as the first-order 3D gates. The machine-grade version is the
discrete 3D second-order sweep (§14); the first-order discrete 3D
transpose is §12.

## 12. The discrete 3D transpose as built

`AdjointSolver3D` with `Options::method = Discrete`
(`AdjointDiscrete3D.cpp`): the exact transpose of the coupled
lateral-viscosity time-stepped map, the §6b design realised with no new
low-level transposes. The load-bearing duality holds as stated there:
under the energy pairing ⟨S, T⟩ = Σ_node Wq r² ∫dΩ [12 S₀T₀ +
8 Re(S₋₁T̄₋₁) + 4 Re(S₋₂T̄₋₂)] on grid tensors and the Hermitian m ≥ 0
pairing (m > 0 weighted twice) on degree columns, Entry (force assembly
of a tensor's analysis) is Exitᵀ (synthesis of the strain of a
solution), the pointwise weight W and the reference systems K are
self-adjoint, and the sea level equation has its exact adjoint. So the
transposed pass is the forward pass with the force and strain legs
swapped and the SLE solve replaced by its adjoint:

    z ↦ A( d_e† + W ⊙ Exit( K⁻¹ [ Entry(z) + Surfᵀ SLE_adj( Uᵀ Entry(z), g ) ] ) ),

run on the spectral dual z = A(y) by the step's own iteration
(`OpenStep::Converge` with a start, opened on the forward step's start
state for its weight field, effective systems and unit responses),
with the measurement duals g a constant source in every pass as the
direct load is in the forward; one more pass at the converged dual gives
the byproducts. Per reversed step: d_e† = d† + Σ β_k(x) m†_k (the finish
transposed); the transposed fixed point; the load dual f† =
directLoadDual + synth(σ†)/r²; m̃†_k = m†_k + μ_k Exit(x†) (the fixed
memory force transposed); the itau kernel as the energy pairing of m̃†
with dt (dE m + dA d⁰ + dB d¹) — the tangent-linear drive's derivative —
as an angular density (Wq r² folded in); and the recursion m† = E m̃†,
d† = Σ A m̃† pointwise. The start is the transposed instantaneous elastic
solve on the unrelaxed systems. The column duals carry no halving or
doubling: the Hermitian column pairing matches the solver's real-field
coefficient pairing (unlike the 1D real columns, which halve and double).
Distributed over radius (§17). Its second-order form is
§14 (the continuous one is §11).

GATES (`TestDiscreteAdjoint3D`, lmax 8, lateral structure on): the
pairing identity with the discrete tangent-linear run at 1e-8 for both
schemes, along loads (node 0 included) and a log-viscosity direction,
each alone and together, and a stride-2 trajectory giving the same at
1e-9; zero-lateral at the GLL points equal to the 1D discrete sweep at
1e-8 (load duals and the spherical reduction of the itau kernel); the
machine-grade Taylor test — |J(δ + h p) − J − h⟨K_δ, p⟩| falls at second
order (rate in 1.9–2.1) at every h from 0.2 to 0.025, Richardson on
central differences landing at 1e-7; and the continuous sweep's gap to
the discrete one falling by > 2.5 as the steps halve, in the loads and
in the log-viscosity kernel (the resolution diagnostic in 3D).

## 13. Material derivatives and kernels

New design, not a port (pyslfp has no material derivatives): the static
elastic problem (§13a), its lateral Born response (§13b), the
time-stepped tangents in the moduli (§13c, §13d), their exact transposes,
the material kernels (§13e, §13f), the continuous kernels (§13g) and the
static kernels (§13h).

### 13a. The static elastic problem

The parameters are the material's moduli at its nodes —
`MaterialDirection`: δκ, δμ on the mesh's (element, gll) nodes in
material units, either empty — and the derivative is taken of the
computed map through `Material::Shifted(d, ε)`, whose linearisation
`Material::Variation(d)` is exact: the moduli's variation (A, C, F, L, N
from δκ, δμ). The density, gravity and the fluid flags are held; the
density is not a parameter of the derivatives (§13l).

`DegreeOperator` assembles its band from a `Coefficients` set
(`AssembleBand`), the base one from the material and the variation's
from `VariationCoefficients`: the elastic block's moduli replaced by
their variations, with the gravitational couplings, the fluid
stratification, the fluid-solid interface terms, the exterior closures
and the point shear (fixed branches) carrying none.
`ApplyVariation(dm, x)` is (dK·dm) x; the forcings do not depend on the
moduli. `ComputeLoveNumberDerivatives(material, lmax, direction)` solves
dx = −K⁻¹ dK x per degree and forcing and reads the surface off as the
numbers are; it returns a LoveNumbers-shaped table of derivatives
(surface_gravity, radius and G zero).

GATES (`TestMaterialDerivatives`, prem8 at lmax 6): the variation
operator against central differences of K(m ± εd)x, Richardson to 1e-9
at every degree, and a direction of the wrong size refused; the
Love-number derivatives against central differences of the shifted
material's numbers for directions in μ, κ alone and together,
Richardson to 1e-7 per kind of number for every channel, degree and
axial number, the differences second order where resolved; the
reciprocity identities differentiated (g δh_φ = δk_u and the two others)
at 1e-10; an empty direction gives zeros.

### 13b. The lateral Born response

`ElasticPerturbation3D` (`ElasticPerturbation3D.hpp/cpp`): the
first-order (Born) response of the static gravitating elastic problem to
lateral δμ(x) and δκ(x) over the grid values. A lateral δμ enters the
bilinear form exactly as the lateral viscosity weight enters a step —
a pointwise weight on the deviatoric strain, fed back through the force
assembly (`Entry(δμ ⊙ Exit(x))`), which in the angle-uniform limit is
the 1D shear block exactly — and δκ weights the dilatation r div u =
X + Y, carried by a scalar companion of `GridCoupling`
(`SynthesiseScalar`/`AnalyseScalar` in the spin-0 layouts,
`AddDilatationOfSolution`, `DilatationForce`: the bulk term κ(X+Y)(X'+Y')
of the degree form, pointwise in angle). Then δx = −K⁻¹ (dK) x with the
reference (radial, unrelaxed) systems, spheroidal and toroidal: a
lateral perturbation couples the degrees by the selection rules of the
products and couples spheroidal to toroidal motion. The nodes are the
GLL nodes of every solid element above the one holding the centre (the
spherical strain components divide by r); they come from a throwaway
Maxwell rheology over the isotropic Voigt average of the material (a
branch may not be active where the material is anisotropic), while the
reference systems and the perturbation's meaning — an isotropic
increment of whatever anisotropy the material has — are the material's
own. `SolveLoad` gives the reference response to a surface mass load by
its m ≥ 0 coefficients; `Force`, `Respond`, `Solve`, `Surface`,
`SurfaceToroidal` and the Hermitian `Pair` complete the interface. No
δρ (§13l).

GATES (`TestElasticPerturbation3D`, prem8 at lmax 6): angle-uniform
δμ(r) and δκ(r) reproduce §13a's exact 1D Love-number derivatives
(U, V, φ at the surface, every degree) at 1e-8 through the whole
pipeline, with no coupling into other orders or toroidal motion; the
perturbation force is symmetric under the Hermitian pairing at 1e-10 for
lateral μ and κ directions, alone and together; a single-harmonic
direction f(r) P_L^M cos Mφ couples a degree-l order-m load only to the
degrees |l − L|..l + L and orders m ± M, spheroidal and toroidal alike,
with the toroidal response present. The lateral response beyond the
angle-uniform limit rests on the pipeline (gated through the viscosity
path) and these structural identities; its independent check is the
cross-code comparison with AdGIA, which is what this is for.

### 13c. The material tangent, 1D

`TangentPerturbation::material` (a `PointMaterialDirection`: δκ and δμ
at the memory points) runs the 1D
tangent-linear along a direction of the moduli with the branch moduli as
fixed fractions f_k = μ_k/μ_U of the unrelaxed modulus at the points and
the relaxation times held (compose with `itau` for a viscosity held
instead). A perturbation of the moduli is a field at the memory points
that enters by being integrated against the strain and the dilatation
there, the rule of the 3D stepper's lateral moduli; the forward problem
takes the same perturbation as `ViscoelasticOperator::Options::moduli`,
and the tangent is its derivative. At the GLL points the points are the
nodes and the direction is a nodal one (`AtMemoryPoints` carries a nodal
direction to the points). The material enters a step in two places and
the start in one, and each is a force on the step's own systems:

- the effective system K_eff of a step (whose shear at the points is
  μ_U − Σ_k μ_k (1 − w_k) with w_k the scheme's weight there): its
  variation along δμ_eff = δμ (1 − Σ_k f_k (1 − w_k)), δκ at the points,
  applied to the FULL base solution x_n of the step
  (`DegreeOperator::ApplyPointVariation`), negated;
- the memory force Σ_k μ_k ⟨m̃_k, ·⟩ of the base decayed memory m̃ =
  e m + α d₀ with δμ_k = f_k δμ in place of μ_k;
- the start's unrelaxed solve: −K_U⁻¹ (δK_U) x₀ about the base start
  solution, whose surface values close the start's sea level solve and
  whose strain is added to the primed start strain (the trapezoid's
  α d₀ reads it).

Both step terms go in through `ViscoelasticStep::AddForce` (a force per
degree added to the memory force before the effective solve; the cached
memory solutions invalidated), so the step's memory surface — what the
sea level closure already reads — carries them, and the closure needs no
Love-table derivative: acting on the full solution, the first term holds
the material derivative of the load response too (adding δh_eff σ as a
response channel would double count). The scheme weights depend on itau
and dt only, so the memory recursion is unchanged.

Rotational feedback, axial or full, needs nothing further for the
moduli. The base solution the force acts on holds the response to the
centrifugal potential, so the force carries the tidal Love numbers'
derivative as it carries the load numbers'; and the rotation block of
the sea level solve closes about the material response through the
channels it reads for the memory — the degree-2 surface displacement
and potential and the degree-0 inertia moment, the start's unrelaxed
response included — with the base model's numbers in its closed form.

GATE (`TangentLinear.MaterialDirectionMatchesDifferences`): along
a smooth direction in κ and μ on the solid, both schemes, the tangent
run matches central differences of the coupled run on the shifted
material — `Material::Shifted` with its rheology rebuilt at the same
fractions and times — at second order (ratio in (3, 5)) with Richardson
landing at 1e-6 of the derivative. The steps are 1% and 0.5% of the
moduli, wide enough for the truncation error to stand above the solves'
1e-12 noise: the functional moves by parts in 1e-12 per unit step, and
at narrower steps the ratio check reads noise.
With rotational feedback on, axial and full, the same holds along the
moduli (the same test and its 3D twin), the discrete kernels stay the
tangent's transpose (`Adjoint.MaterialKernelsAreTheTangentTranspose`,
`DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent`), the continuous
ones converge to them, and the second-order blocks are exact and
symmetric (`SecondOrderAdjoint.MaterialBlocksAreExactAndSymmetric`,
`SecondOrderAdjoint3D.DiscreteMaterialHessianIsExactUnderRotation`).
`TangentLinear.MaterialDirectionMatchesDifferencesAtGaussPoints` holds
the same at the default memory points, along a direction in κ and μ at
the Gauss points, against differences of the run with `Options::moduli`
perturbed.

### 13d. The material tangent, 3D

`TangentPerturbation3D::material` (`MaterialDirection3D`: δκ(x), δμ(x)
over the coupling's grid values, material units) runs the 3D tangent-linear along a lateral direction of
the moduli, with the same meaning as in 1D (branch moduli as fixed
fractions, relaxation times held). The direction lives on the memory
nodes, which is where the operator's lateral structure lives: an
elastic region to be perturbed laterally takes a branch of negligible
inverse relaxation time, so that it carries memory nodes. The three
terms are the 1D ones through the grid pipeline, entering the opened
step by `OpenStep::AddForce` (an analysed tensor joining the fixed
memory force, and per-degree spheroidal forces added after the
assembly), so the within-step fixed point and the sea level closure
carry them as they carry the memory force:

- the effective system's variation on the full base solution — the
  Born force of §13b with the weight δμ_eff(x) = δμ(x)(1 − Σ_k f_k
  β_k(x)), the step's OWN lateral weights β_k(x) (a lateral viscosity
  makes the effective modulus's variation lateral even for a radial δμ)
  — on the base node's grid strain (the cursor's state carries it; a
  step's converged solution satisfies the unrelaxed equilibrium with its
  new memory, so the base solutions a bulk direction needs come from one
  unrelaxed solve under the recorded loads);
- the memory force of the BASE step's decayed memory m̃ = e m + α d₀
  with δμ_k = f_k δμ: a base step opened on the cursor's previous state
  (the tangent's own step holds the tangent's memory, which is not it);
- the bulk term through the scalar pipeline (`DilatationForce` of
  −δκ times the dilatation of the base solutions);

and the start's unrelaxed response −K_U⁻¹ (δK_U) x₀ by the same
pipeline, closing the start's sea level solve and seeding the start
strain. Rotational feedback as in 1D (§13c): the closure reads the
material response's surface values and inertia moment, the start's
included.

GATES (`TestTangentLinear3D`): `MaterialDirectionMatchesDifferences`
— with lateral viscosity ON, along an angle-uniform direction in κ and
μ on the memory nodes, the 3D tangent matches central differences of
the coupled 3D run on the shifted material at second order with
Richardson at 1e-6, both schemes (the lateral β_k(x) factor exercised),
and `MaterialDirectionMatchesDifferencesAtGaussPoints` the same at the
default memory nodes against the run with `Options::moduli` perturbed;
`ZeroLateralMaterialMatchesThe1DRun` — with δ = 0 the 3D run equals the
1D material tangent at 1e-7, both schemes, at the GLL and at the Gauss
points, and a lateral direction of the same angular mean is a different
run. The
lateral part of the shear term is the pipeline of §13b and of the
viscosity path; its exact gate is the pairing identity with the discrete
3D material kernels (`DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent`,
§13f).

### 13e. The material kernels, 1D

`AdjointSolver::Options::material_kernels` has the discrete sweep form
`IceKernels::kappaKernel` and `muKernel` — the kernels in κ(r) and
μ(r) over the full mesh in material units, per unit volume (§4): the
exact transpose of
`TangentPerturbation::material`. The sweep already solves, per step,
the transposed effective system for the dual y_n of the step's full
solution (the duals of the next memory through the strain transpose,
the sea level closure's response duals, the inertia dual); the material
enters the tangent as forces on that solve, so the kernels are the
pairings of y_n with each force's material derivative:

- −(dK_eff) x_n: the gradient of the degree form in the moduli at the
  memory points for the pair (y_n, x_n), `DegreeOperator::PointGradient`
  — the dilatation product for κ and the deviatoric-scalar form the
  memory force uses for μ, at each point with its quadrature weight,
  over EVERY element of the system — the shear scaled by the effective
  factor 1 − Σ_k f_k (1 − w_k) at the points; x_n from one unrelaxed
  solve under the node's memory and recorded loads;
- the memory force of the base decayed memory m̃ with δμ_k = f_k δμ:
  Σ_k f_k W_q Σ_s w_s d_s(y_n) m̃_{k,s}, on the memory window;
- the start: the transposed unrelaxed solve under the start duals and
  the start's response duals, paired with the base start solution.

The gradient lives on `DegreeOperator`, not on the operator's per-degree
workspace, because that workspace covers the memory window only (the
relaxing elements) and the kernels span the elastic lithosphere too.
Discrete exact, continuous at the commute level (below, §13g); needs the
trajectory; formed under any rotational feedback; the bulk pairing costs
one unrelaxed solve per step. The kernels in κ and μ are held at the memory
points as samples of the kernel per unit volume, and pair with a
`PointMaterialDirection` by `VolumePairing` (§4); their jumps across
the mesh's interfaces are kept, since no point is shared between
elements. As functions the kernels of the two memory point sets agree
to 1e-6 or better (`Adjoint.MaterialKernelsAreFunctionsOfRadius`).

GATES (`TestAdjoint`): `MaterialGradientIsTheTransposeOfTheVariation`
— for random solution pairs of every degree the gradient paired with a
random direction equals the pair's product with §13a's variation
operator at 1e-12; `MaterialKernelsAreTheTangentTranspose` — the
kernels paired with a smooth direction in κ and μ equal the material
tangent's objective change at 1e-10, both schemes, with the memory
points at the GLL nodes and at the Gauss points, and at the GLL points
central differences on the shifted material land on them at second
order with Richardson at 1e-6; the refusals.

### 13f. The material kernels, 3D

`AdjointSolver3D::Options::material_kernels` has the discrete 3D sweep
form `Kernels3D::kappaKernel3D` and `muKernel3D` — the kernels in κ(x)
and μ(x) in the itau kernel's layout, per unit volume (§4) — and their
means over the sphere, the
exact transpose of `TangentPerturbation3D::material`. The transposed
fixed point already yields, per step, the dual solution y_n of the
step's full solution (the force of the end strain's dual, the lateral
weight's transpose and the closure's response duals through the
effective systems); the material tangent adds its forces to that very
solve (an analysed tensor into the fixed force, per-degree bulk
forces), so the kernels are the pairings of y_n with the tangent's grid
tensors:

- the shear: W_q r² [−(1 − Σ_k f_k β_k(x)) C(e_n, d_n) + Σ_k f_k C(e_n,
  m̃_k)] with C the (12, 8, 4) energy contraction the itau kernel uses,
  e_n the dual solution's grid strain, d_n the recorded forward strain
  and m̃_k the base decayed memory the opened step holds;
- the bulk: −W_q D(y_n) D(x_n), the dilatations through the scalar
  companion (`AddDilatationOfSolution`, `SynthesiseScalar`), x_n from
  one unrelaxed solve under the node's memory and recorded loads;
- the start: the transposed unrelaxed solve under the start duals and
  the start's response duals (one more solve than the ice kernels
  need), paired with the base start solution.

Discrete exact, continuous at the commute level (below, §13g); needs the
trajectory; distributed (§17).

GATES (`TestDiscreteAdjoint3D`): `MaterialKernelsPairWithTheTangent` —
with lateral viscosity on, the kernels paired with LATERAL directions in
κ and μ (a radial profile times caps and harmonics) equal the 3D
material tangent's objective change at 1e-8, κ alone, μ alone and
together, both schemes: the exact gate of the lateral shear term of
tangent and kernel alike; the spherical reductions pair with
angle-uniform directions as the full kernels do; the refusals.
`ZeroLateralMaterialKernelsMatchThe1D` — with δ = 0 the spherical
reductions equal the 1D discrete material kernels at the memory nodes
at 1e-7, both schemes.

The derivative machinery in the material is closed in both dimensions:
Love-number derivatives (§13a), the lateral Born response (§13b), the
time-stepped tangents (§13c, §13d) and their exact transposes, the κ and
μ kernels (§13e, §13f), each gated against the previous rung.

### 13g. The continuous material kernels, 1D and 3D

`material_kernels` works under `AdjointMethod::Continuous` too. The
continuous adjoint problem has the forward form, so at a node the
adjoint displacement λ is the unrelaxed solve under the adjoint
memory and the closed adjoint load (what `smoothEquilibrium` closes),
and the true multiplier is λ times the itau kernel's units constant
(and, in 1D, its Hermitian column weights: m = 0 real once, its
imaginary part not at all, m > 0 twice — derived from the itau pairing,
whose calibration fixes the multiplier). The kernels are then

- the time quadrature, with the itau kernel's trapezoid half-weights on
  both sides of each node, of −(the material gradient of λ and the
  forward solution x_n) + (the memory force's variation with the FULL
  forward memory m_n): no effective factor and no decayed memory, those
  being the discrete form's;
- plus, at each observation, the lumped pairing of the instantaneous
  adjoint solution (the jump's, kept from `applyJump`) with x_n and m_n,
  without a time weight: the delta the adjoint displacement carries
  there, whose jump in the adjoint memory the viscosity path already
  took.

In 3D the same with the grid strains' energy contraction and the
dilatations through the scalar companion (as the discrete 3D kernels),
no column weights (the grid pairing is the full real one). Cost: one
more unrelaxed solve per node than the discrete form.

GATES: `Adjoint.ContinuousMaterialKernelsAgreeAndConverge` and
`DiscreteAdjoint3D.ContinuousMaterialKernelsAgreeAndConverge` — ONE
problem at every resolution (a ramp of fixed fields, measurements at the
same times), against the exact discrete kernels paired with a direction
(lateral in 3D, lateral viscosity on): at 2.5 kyr steps the gap is 0.9%
(trapezoid) and 5.6% / 0.5% (backward Euler, 1D / 3D), falling under
refinement at the schemes' orders (1D trapezoid 0.95% → 0.24% → 0.06%
over 5, 9, 17 nodes; backward Euler 5.6% → 2.9% → 1.5%).

### 13h. The static kernels

For the static elastic problem the kernel is the start term alone.
`ComputeLoveNumberKernels` (`LoveNumbers.hpp`): per channel (load force and potential pieces,
tangential, tide) and degree, dh_l/dκ(r), dh_l/dμ(r) and the l and k
alike, nodal over the full mesh, by the unit surface duals' solutions
paired with the forcings' through `DegreeOperator::MaterialGradient`;
gated against `ComputeLoveNumberDerivatives` along a direction at 1e-10
for every channel and degree
(`MaterialDerivatives.LoveNumberKernelsPairToTheDerivatives`). In 3D
`ElasticPerturbation3D::StaticKernels(y, x)` (the transpose of its
`Force`, per unit volume on its grid) with `SolveSurfaceDual` for
the adjoint solution of a surface measurement; gated as the transpose
of `Force` under the Hermitian pairing at 1e-10 and by the dual solve
being the adjoint of the surface readout
(`ElasticPerturbation3D.StaticKernelsAreTheTransposeOfTheForce`), and
under a lateral background with both solutions from the full solve
(§13j) against central differences of the measurement at 1e-5
(`KernelsUnderLateralStructureMatchFiniteDifferences`). The 3D elastic
benchmark against AdGIA can thus compare kernels as well as responses,
at the heterogeneous model itself.

### 13j. The full 3D static solve

The laterally heterogeneous static problem itself, (K_ref + dK) x = f,
not only its Born linearisation: `ElasticPerturbation3D::SolveLateral`.
The method is that of Maitra & Al-Attar (2019) and Myhill, Maitra &
Al-Attar (2026) for Poisson's equation in aspherical planets, carried
to the elastic operator: the full operator's action is matrix-free —
the reference degree systems' `Apply` (banded, per degree and order)
plus the lateral force of §13b through the grid pipeline (`Apply(x,
δμ, δκ)`) — and the reference solve K_ref⁻¹, block diagonal over the
degrees with the factorisations already held, is the preconditioner.
The operator is symmetric (§13b's gate) and, for perturbations that
leave the moduli positive, definite, so the iteration is preconditioned
conjugate gradients, started from the reference response x₀ = K_ref⁻¹
f; the first preconditioned residual is then the Born correction, so
the Born response is the method's own first step and the iteration
continues to the exact answer. The toroidal degree-1 rigid rotations,
which the pinned reference solve removes and the lateral force never
excites, stay out. The residual reported is the preconditioned energy
norm relative to the forcing's, ⟨r, K_ref⁻¹ r⟩/⟨f, K_ref⁻¹ f⟩ under the
Hermitian pairing; `SolveInfo` carries its history, the count and a
flag should a curvature come out non-positive (the operator pair then
not definite, the iteration continued on magnitudes).

The forcing is any `Forces` pair: `LoadForces` (a surface mass load),
`TideForces` (an external potential of every degree and order through
`DegreeOperator::Tide(power)` — the static body tide of a laterally
heterogeneous body, which is what the app will expose),
`SurfaceDualForces` (a measurement's dual), or their sum, with toroidal
forces allowed. Because the operator is symmetric, the solve of a
measurement's dual forces under the lateral structure is the adjoint
solution at that structure, and `-StaticKernels(y, x)` with both
solutions from the lateral solve are the measurement's kernels in μ(x)
and κ(x) at the laterally heterogeneous model: the static path's
adjoint, discrete-exact, under 3D structure.

GATES (`TestElasticPerturbation3D`, prem8 at lmax 6): the converged
solution's full, unpreconditioned residual at 1e-9 of the forcing; its
departure from the reference-plus-Born response second order in the
contrast (halving a 10% lateral δμ, δκ quarters it: ratios 4.0
spheroidal, 3.9 toroidal), the first-order check against §13b; an
angle-uniform 10% shift of an isotropic reference (the Voigt average of
PREM — the shifted material must keep the reference's anisotropy, so the
reference is made isotropic) equals the exact 1D solve on the shifted
material at 1e-9 at every degree and order of a load and of a tide,
with no toroidal motion; the kernels under a lateral background (10% in
μ and κ) paired with a lateral direction equal the central difference of
a surface potential measurement through the full solve at 1e-5 (the
difference's own error). Convergence to 1e-10 against a degree-2 shear
pattern of 5, 10, 20 and 40 per cent of the local shear modulus takes
6, 7, 9 and 15 iterations, geometric throughout; the solve of the
10% kernel gate converges in about 20. Cost on 16 threads (the
disabled `TimingAtHigherDegree`, `SL3DPP_LMAX`): a 10% degree-2
pattern and a cap-like load at lmax 16, 32 and 64 take 0.4, 2.4 and
15 s for six iterations, the lateral force (the grid transforms) the
whole of it; the reference solve is 0.07 s at lmax 64. Realistic
contrasts (10% or less) are well inside what the reference
preconditioner handles, as the papers found for Poisson's equation.

Not built: aspherical geometry by particle relabelling (the papers'
mapping ξ, which turns the geometry into laterally varying tensorial
coefficients of the same form), lateral
δρ (the hydrostatic background does not admit it, §13l), and
distribution of the static solve (OpenMP over the grid pipeline only).

The program: `sl3d response` (apps/ResponseCommand.cpp) runs this problem
on a model file's lateral moduli (io::ModelModuli:
the relative perturbations of its `mu` and `kappa` about their lateral
means or a stored radial reference) through the stepper's conjugate
gradients on a branchless rheology, distributed over radial segments,
with the radial model's response from ElasticPerturbation3D beside it;
it writes the surface fields, the surface coefficients in the
benchmarks' exchange form, and the response at the memory nodes as a
model file (io::WriteNodeFields, io::ResponseFieldsAtNodes) for
planetmodel's cutaways. ExampleStatic3D and `configs/response.json` run
it on the synthetic 3D elastic Earth the build writes
(MakeSyntheticModuli, io::WriteModuli). With a `times` block the same
command steps the viscoelastic response through time instead, its first
node this problem's solve (docs/sl3d-configs.md).

### 13k. Lateral moduli in the 3D stepper

The same lateral δμ(x), δκ(x) in the quasi-static viscoelastic
problem: `Viscoelastic3D::Options::moduli`, relative perturbations
sampled like the log-viscosity — μ_U(x) = μ̄_U(r)(1 + s(x)) with every
branch modulus scaled alike (fractions and relaxation times held, the
material derivatives' convention), κ(x) = κ̄(r)(1 + b(x)). With them
set the coupling carries every solid element above the centre
(`GridCoupling::Options::all_solid`; an elastic element's memory is
inert), so a perturbation can live in the lithosphere or the inner
core as well as the Maxwell mantle. The fluid is not covered: the
fluid form (κ, ρ, g; no deviatoric strain) has no grid pipeline, so a
lateral κ in the outer core is not representable.

The step is one fixed point (§5 of lateral-coupling.md): the
lateral moduli join the lateral viscosity in its pointwise weight. The
effective shear of the shifted body is (1 + s)(μ̄_U − Σ_k μ̄_k β_k(x)),
so the weight gains −s(x) μ̄_eff(x) with μ̄_eff(x) = μ̄_U − Σ_k μ̄_k β_k(x)
the reference moduli under the LOCAL weights (`StartStep`), the fixed
memory force gains the factor (1 + s(x)) on every branch modulus, and
m⁺ = m̃ + β d is as before (β is the relaxation time's). The bulk term
is new in kind: a lateral δκ weights the dilatation, which the
deviatoric iterate does not carry, so `SpectralTensor`/`TensorField`
carry an optional scalar companion (`scalar`: the dilatation's
coefficients and grid values), which `Synthesise`/`Analyse` carry when
present, `StrainOf` fills (`AddDilatationOfSolution` beside the strain),
`Respond` weights by −δκ(x) and assembles through `DilatationForce`
(`Viscoelastic3D::SpheroidalForces`), and the mixer and residual treat
as a fourth block. A strain cache under a lateral bulk modulus must
carry it (a cache without it is rebuilt by a solve); the serialiser
writes it.

The UNRELAXED solve (a step's start without a cache, `SolveElastic`,
`ElasticStrain`, `PrimeStrain`) is laterally heterogeneous, so it
is the preconditioned conjugate-gradient solve of §13j on the
unrelaxed reference systems — shared code, `detail/LateralSolve.hpp`,
which `ElasticPerturbation3D::SolveLateral` also uses — with the
memory and channel forces as the right-hand side
(`LateralElasticSolve`, `AddChannelForces`; `elastic_rtol` 1e-12;
`LastElasticSolve` reports it). Where the material is anisotropic the
perturbation is an isotropic increment of its Voigt shear; at the
memory points the reference's unrelaxed shear is the interpolated
long-term modulus plus the branches' at the points (`UnrelaxedShearAtNodes`).

GATES (`TestLateralModuli3D`, PREM's Voigt average with a Maxwell
mantle, lmax 6): smooth 10% shifts of shear and bulk over the whole
solid above the centre element at the GLL points equal the 1D stepper
on the shifted rheology at 8e-12 (U, V, φ at every degree and order,
no toroidal motion), both schemes, two steps from rest; at Gauss
points a per-element constant 10% shift of the Maxwell region (where
every shear term is a point term) is exact likewise at 9e-12; the
unrelaxed solve of a zero state under a load with angle-varying shear
and bulk equals the static lateral solve on the same fields at 1e-9,
toroidal motion included; with lateral moduli and lateral viscosity on
together the step self-converges at second order (ratio 3.98). The
within-step passes grow from 2 (zero lateral) to about 10 at 10%
moduli contrast — the lateral elastic term is a stronger coupling than
mild viscosity and contracts under Picard at roughly 0.1 a pass; the
step by conjugate gradients (§13n) is the alternative.

**The sweeps under lateral moduli.** Three things change
for the tangent, the adjoints and the second-order sweeps, and nothing
else: every branch modulus the sweeps pair with or differentiate
through carries the shear factor 1 + s(x) (`Viscoelastic3D::
ShearFactor`: the itau kernels' μ_k m†:(d − m), the discrete
recursion's μ_k e† terms, the derivative dual's, the lateral weight's
derivative Σ μ_k dβ dh; the fractions f_k are unchanged, so every
material drive and pairing is); every unrelaxed solve of forces is the
lateral one (`SolveUnrelaxed`: the material drives' responses, the
tangent solutions, the start duals); and the discrete transposed fixed
point carries the dilatation's block (its `entry` adds the dilatation's
force where the iterate carries the scalar companion, its `exit`'s
strain carries the dilatation, weighted by −δκ(x) in the pass).

The one structural change is the INSTANTANEOUS problem. The unrelaxed
operator is laterally heterogeneous, so the elastic sea-level
problem at a node — the forward run's start, the tangent's node 0, the
discrete sweep's start transpose, the continuous sweep's jumps and
equilibria — is a coupled fixed point like a step's, not a Love-number
closure with response channels. `Viscoelastic3D::StartStep` accepts
dt = 0: a zero-length step (β = 0, m̃ = m, the effective systems the
unrelaxed ones, the lateral weight −δμ_U), whose within-step fixed point
with the sea level engine IS the instantaneous coupled solve. Under
lateral moduli the evolution's start is such a step (`SeaLevelEvolution3D`,
one coupled pass for the start and the steps), the tangent's
node 0 likewise with the material drive's start forces on it
(`MaterialForcing3D::ApplyStart`), the discrete sweep runs its step
transpose down to n = 0 with dt = 0 instead of its start block (the
channel duals at a lateral start are the unit load and tide vectors
paired with the lateral solution, `lateralChannelDuals`), and the
continuous sweep's jumps and equilibria are zero-length reversed steps
(`instantaneous`, `lateralJump`, `lateralJumpDerivative`, and
`reversedStep(…, 0.0, …)` for the equilibria, the derivative state's
with v's material source). Without lateral moduli the Love-number
closures are kept.

The evolution's start is also a solve of its own,
`SeaLevelEvolution3D::Instantaneous(load)`: the instantaneous elastic
response of the sea level problem to a direct load, the linear equation
on the background state's shorelines coupled to the rotational feedback,
by the zero-length step's fixed point (or the nested closure of §13n)
under lateral moduli and by the unrelaxed Love numbers without — the
elastic sea-level fingerprint of a laterally heterogeneous Earth, which
`sl3d sea-level` solves without a `times` block on a lateral moduli
file. `Run`'s first node is the same code (`Start`), so `Run(times,
load)[0]` is `Instantaneous(load(times[0]), times[0])`; it is distributed
over radial segments as the run is. The check of the transverse
rotational feedback against the model's fluid limit, which a run
through time makes for its long steps, does not apply to it.

GATES under lateral moduli (10% shear, 5% bulk, angle-varying, with
lateral viscosity): the coupled evolution with angle-uniform moduli
equals the evolution on the shifted rheology at every node including
the start, rotation on, at 1e-9 (`LateralModuli3D.
CoupledEvolutionMatchesTheShiftedRheology`); the discrete sweep pairs
with the tangent-linear run at 1e-8 for loads and a log-viscosity
direction, both schemes (`DiscreteAdjoint3D.PairsWithTheTangentUnderLateralModuli`);
the discrete κ(x), μ(x) kernels pair with the material tangent at 1e-8
(`MaterialKernelsPairWithTheTangentUnderLateralModuli`); the tangent
along a lateral material direction matches central differences of the
coupled run with the background moduli shifted, Richardson at 1e-6
(`TangentLinear3D.MaterialDirectionMatchesDifferencesUnderLateralModuli`);
the discrete second-order sweep is symmetric and adjoint to the
second-order run to 1e-21 relative (`SecondOrderAdjoint3D.
DiscreteMaterialHessianIsExactUnderLateralModuli`); the continuous
sweep's gap to the discrete closes under refinement at ratio 4 for
loads and log viscosity (`ContinuousGapClosesUnderLateralModuli`), and
the continuous material
kernels and continuous second-order sweep converge likewise
(`ContinuousMaterialKernelsAgreeUnderLateralModuli`,
`ContinuousMaterialHessianConvergesUnderLateralModuli`). The
instantaneous response equals the run's first node to the bit, with and
without lateral moduli, rotation off and full, both step solvers
(`LateralModuli3D.InstantaneousIsTheRunsFirstNode`); with the moduli set
but zero it is the radial fingerprint, to the bit
(`InstantaneousWithoutLateralStructureIsTheRadialFingerprint`); 10%
shear and 5% bulk perturbations move it by 2.3e-2 of the sea level's
largest magnitude, the two step solvers agreeing at 1e-9
(`InstantaneousUnderLateralModuliDepartsFromTheRadialFingerprint`).

Not covered: lateral δκ in the fluid. Distribution over radial
segments: §13m.

### 13l. Density: not offered

sl3dpp offers no derivatives with respect to the density, radial or
lateral: its material directions, tangents and kernels are in the moduli
(and, through the rheology, the relaxation times), the density a fixed
part of the forward model. A lateral density would not even be
consistent with the forward problem. The equations sl3dpp solves are
those of a hydrostatically pre-stressed body: the reference gravity
g(r), the reference potential and its Hessian, and the stratification
r²(dρ/dr)/g all come from a radial density in equilibrium. A laterally
varying density is not in hydrostatic equilibrium with a radial gravity
field. What the field does — ρ(x) multiplying the displacement in the
gravitational couplings while g, the reference potential and the
stratification stay radial, the self-attraction and the pre-stress of
δρ neglected — is a modelling choice with no consistent limit inside a
spherically layered reference: making the gravity self-consistent
(carrying Φ as a variable of the whole problem, equilibrium and
first-order together) still leaves the pressure wrong, because ∇p =
ρ∇φ₀ has no solution on a spherically layered reference with a lateral
ρ; the consistent routes are an aspherical figure (particle
relabelling, not built) or a non-hydrostatic pre-stress, which is
AdGIA's new theory. So 3D density belongs on the AdGIA side, and
sl3dpp says only what it can defend: the moduli laterally, the density
radially.

### 13m. The lateral solves over radial segments

The stepper's lateral moduli run over radial segments like the rest
of it (docs/radial-partitioning.md). The grid pipeline is local to
each process's nodes — the lateral weights and the companions are per
node — so what is distributed is the conjugate-gradient start solve:
its operator's reference part assembles the whole vector from the
held shares (`MaskToShare` then the sum collective, one per degree),
applies the band and masks the product back to the share, the layout
the force shares have; its pairings are each process's part (its
solution, holding its segments and the cut nodes, against the shares)
summed by the collective (`detail::LateralSystem::sum`); the channel
forces take the unit vectors as shares. The static 3D problem over ranks
is the stepper on a branchless rheology (`Rheology::Create(material,
{})` with `moduli` set; `SolveElastic` is then the static lateral
solve); `ElasticPerturbation3D` itself stays single-process.

GATES: two and three threads-as-ranks (`RadialRanks.
LateralModuliStepsMatchTheOneProcessSteps`): the start solve's
iteration count, two lateral steps' surface responses and memory
slices, and the branchless static solve, against the one process at
1e-11 (measured 1e-13); the continuous and discrete sweeps on three
ranks with lateral moduli and the axial rotational feedback
(`LateralModuliAdjointsMatchTheOneProcessSweeps`): ice kernels and
viscosity-kernel slices at 1e-10 (measured 6e-13 and 2e-13); the
instantaneous response on three ranks, on the Maxwell and on the
branchless rheology, both step solvers, at 1e-10 (measured 2e-16 by the
fixed point and 1.2e-12 by the conjugate gradients;
`InstantaneousResponsesMatchTheOneProcess`); under
mpiexec on two and three ranks
(`Distributed.LateralModuliStepsMatchTheOneProcessSteps`), likewise.

**Cost, and the active elements.** A step's pass on 16
threads at lmax 64 (prem8, Maxwell mantle, the `DISABLED_StepTiming`
probe of `TestLateralModuli3D`): 0.66 s with lateral viscosity alone
(1108 memory nodes), 0.96 s with a lateral shear over the whole solid
(1700 nodes: the lithosphere, crust and inner core join); the one
scalar companion of a lateral bulk modulus adds two scalar transforms a
pass, and a scalar transform costs about as much as a spin one. The
coupling therefore carries, beside the viscous
elements, only the solid elements where a perturbation is non-zero at
some memory point (`Viscoelastic3D::ActiveElementsFor`, a probe of the
whole solid sampling the functions; `GridCoupling::Options::
active_elements`), so a perturbation confined to the mantle costs no
lithosphere or inner-core nodes.

Two properties of the distribution, which the program's `response`
command relies on. Whether an operator has lateral moduli is a
property of the RUN, not of a rank: with the moduli confined to the
upper mantle, an inner rank's segment holds no element where they are
non-zero, its coupling no node, and its sampled arrays are empty — but
every rank must take the lateral route, since the solve is collective
(`HasLateralModuli` reports the option, not the samples;
`RadialRanks.ConfinedLateralModuliMatchTheOneProcessStatic` holds a
node-less rank to the one-process answer). And the rank cuts balance
the transform work of the coupling's nodes, which for a branchless
rheology the rheology does not know: the response command weights the
active elements the moduli make (`OperatorOptions` takes them), or the
partition is degenerate and the conjugate gradients stall.

### 13n. The step by conjugate gradients, and the nested closure

A 3D step is a linear system in the degree solutions,

    (K_eff − Pᵀ W S) x = f,

K_eff the reference effective systems of the (scheme, dt), W(x) the
step's pointwise lateral weight (the viscosity's Σ μ̄_k(β_k(x) −
β_ref,k) and the moduli's −s μ̄_eff, with the bulk companion −κ̄ b on
the dilatation), S the strain of the solutions, Pᵀ the assembly, and
f the fixed memory force plus any added forces and the channels'. The
operator is the 3D effective stiffness: symmetric positive definite
whatever the contrast, since the effective modulus μ_U(1+s) −
Σ μ_k(1+s)β_k(x) is positive pointwise. The within-step fixed point
(§13k, lateral-coupling.md §5) iterates it as x = K_eff⁻¹(f + Pᵀ W S x)
on the strain — Picard, Anderson-mixed — which contracts only as fast
as the lateral term is small against the reference: at 100x contrasts
and dt ≥ 2τ it takes 65–150 passes, and at dt = 10τ it fails.
`Viscoelastic3D::Options::solver = ConjugateGradients` closes the same
system by the preconditioned conjugate gradients of §13j instead
(`detail/LateralSolve.hpp`, the routine the static solve and the
unrelaxed start solve use), K_eff⁻¹ the preconditioner, from the
Picard iterate (the reference response, or a warm start): guaranteed
to converge, and in practice in fewer iterations than the fixed point
takes passes everywhere measured. An iteration is one grid pipeline
(synthesis, weight, analysis, assembly) like a pass, plus the reference
band products on the columns (hand-written, threaded over the degrees:
a BLAS call inside the OpenMP loops starts a second thread pool on the
server)
and the preconditioner's reference solve, plus the Krylov vector
algebra; at lmax 16, where the pipeline is 4 ms, an iteration costs
about 1.3 passes, and the ladder's second steps still take less time
at every contrast (lateral-coupling.md §5a''); at production sizes the
pipeline dominates both. `OpenStep::Forces` builds
f, `OpenStep::Apply` the operator, `OpenStep::Solve` the solve
(`solve_rtol` on the relative residual in the preconditioned energy
norm √(⟨r, K_eff⁻¹ r⟩/⟨f, K_eff⁻¹ f⟩), `solve_max_iterations` the
budget), `OpenStep::Close` a step under known channels either way, and
`Step` reads the option. Distributed, the operator is the §13m one
(shares assembled by the sum collective per degree, the band applied,
the product masked back; pairings summed), so the step's solve runs
over radial segments like the start solve.

**The closure outside.** The sea level equation makes the loads depend
on the solutions. The fixed point folded both into one iteration on
the strain. With the solve inside, `OpenStep::Nest` runs the driver's
closure (the sea level engine on the surface response, as before: a
`Closure` maps the CHANNEL-FREE solutions to the next channels) as an
outer iteration: pass n hands the closure the last solution less its
channels, solves the step under the new channels warm-started from the
last solution WITH the new channels (the Picard iterate, which already
carries the 1D Love-number response to the load change), and the
outer residual is that warm start's residual before the solve — how
far the last solutions with the new channels are from solving the
step, relative to its forces, in the solve's own measure. Converged
when it is at most `coupled_rtol`. The inner solves are inexact: each
reduces its start's residual by `solve_outer_factor` (1e-2) at least,
never below `solve_rtol`, so the inner work follows the outer
progress. The outer map is Picard, Anderson-mixed on the channel-free
solutions by the same adaptive rule as the strain fixed point (the
`IterationControl`'s memory and rate); in the cases measured the outer
map contracts by ~1e-3 a pass and Picard never hands over. The outer
contraction is the 3D departure of the load response from the 1D Love
numbers the engine closes with, times the ocean's feedback — small
unless the lateral structure is strong at the surface.

**Who uses what.** The evolution's steps and its lateral start (the
zero-length step, `Instantaneous` too), the tangent-linear run's steps
and start, the
continuous sweep's reversed steps and its instantaneous problems
(equilibria, jumps) all go through `Nest` under the option; the
forward regenerations in the sweeps (a kept trajectory's steps rebuilt
under recorded loads) through `Close`. The DISCRETE transpose follows
the option too. It transposes the converged linear solve, which is the
same system whichever way the forward solved it, and its operator is
the forward step's own: in the dual solutions y the transposed step is

    (K_eff − Pᵀ W S) y = b₀ + Surfᵀ SLE†(Fᵀ r, g),

b₀ the forces of the dual strain's source (and any added), F the unit
load and tide forces, Surfᵀ the surface duals placed on the surface
rows and the degree-0 moment row, and r = y less the reference response
of the surface duals — the reference response of the forces, so Fᵀ r is
the forces' contraction with the unit responses, what the transposed
fixed point hands the engine's adjoint. Under the fixed point the pass
of §13f iterates it on the dual strain; under conjugate gradients the
sea level adjoint is the outer closure on r exactly as `Nest` runs the
forward's (the solve is `OpenStep::Solve` on the same step, warm-started
from r with the new surface duals' reference response, inexact by
`solve_outer_factor`, the outer residual the warm start's, Anderson by
the control's rule), for the first dual state and the second-order
sweep's derivative state alike. Distributed, Fᵀ r is summed from each
rank's rows.

**Measured (lateral-coupling.md §5a'').** The §5a ladder at lmax 16,
the second step: at 3x, 8/13/17 iterations against 12/21/21 passes at
dt/τ = 0.5/2/10; at 10x, 12/23/34 against 20/29/50; at 100x, 23/46/90
against 26/65/fail; at 1000x, 42/83/183 against 49/138/fail. The
nested evolution at a 3x viscosity contrast with 10%/5% moduli and the
full rotational feedback (`LateralModuli3D.NestedConjugateGradient
EvolutionMatchesTheFixedPoint`, agreement to 1e-9 at every node): 5
passes and 8 iterations at the start against the fixed point's 10
passes, 7 passes and 20–22 iterations a step against 20 — the nested
closure costs about 1.4 passes' worth per fixed-point pass at this
weak contrast (each outer pass also pays one operator application for
its residual and one reference solve for its scale), and the fixed
point stays the default. Where the fixed point grinds or fails, the
conjugate-gradient step is the one to use.

GATES: `Lateral.ConjugateGradientStepMatchesTheFixedPoint` (10x,
dt = 2τ, two steps, solutions to 1e-9, and the operator applied to the
fixed point's solution returns the step's forces to 1e-9),
`LateralModuli3D.ConjugateGradientStepMatchesTheFixedPointUnderModuli`
(viscosity, shear and bulk together), the nested evolution gate above,
`TangentLinear3D.MaterialDirectionMatchesDifferencesUnderConjugate
GradientSteps` (the tangent along a lateral material direction against
central differences of nested runs, Richardson at 1e-6),
`DiscreteAdjoint3D.ContinuousMaterialKernelsAgreeUnderConjugate
GradientSteps` (the continuous sweep's reversed steps and instantaneous
problems nested, the discrete transpose nested too, the gap closing
under refinement), `DiscreteAdjoint3D.PairsWithTheTangentUnderConjugate
GradientSteps`, `DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent
UnderConjugateGradientSteps` and `SecondOrderAdjoint3D.DiscreteMaterial
HessianIsExactUnderConjugateGradientSteps` (the nested transpose exact:
the kernels pair with the nested tangent and the Hessian with the
second-order run at 1e-8), and the two radial-rank gates and the
mpiexec lateral gate run the steps and the evolution on the
conjugate-gradient solver. Probe: `Lateral.DISABLED_StepSolverLadder`.

## 14. The discrete 3D second-order adjoint

`AdjointSolver3D::SecondOrderKernels` under `AdjointMethod::Discrete`
(AdjointDiscrete3D.cpp; the 3D form of the 1D discrete second-order sweep
of §9): the exact derivative of the exact gradient of §12 along a
direction v, from v's recorded DISCRETE tangent-linear run and its
δitau(x). The transposed sweep carries the derivative dual state
(dm†_k, dd†) beside (m†_k, d†), and every piece of §12 is differentiated
in place:

- the dual of the end strain: dd† + Σ_k β_k(x) dm†_k + Σ_k dβ_k(x) dh_k(x)
  m†_k, with dh_k = dt δitau_k(x) and dβ_k the weight's h-derivative
  (dB of the trapezoid's (E, A, B), −dw for backward Euler);
- the derivative state's transposed fixed point: the SAME iteration as
  the first's (reference systems, lateral weight W(x), the SLE adjoint
  with no measurement duals), with its source dde + dW(x) e, where
  dW = Σ_k μ_k(r) dβ_k(x) dh_k(x) is the derivative of the step's
  lateral weight Σ_k μ_k (β_k(x) − β^ref_k(r)) and e the first dual
  strain — the only term with no 1D counterpart in form (in 1D it is the
  effective system's derivative acting on the first sweep's solution,
  Σ_k B_h dh μ_k S y; here the reference is held and the lateral weight
  carries all of the itau dependence, so the derivative lands pointwise);
- the dual of the decayed memory, dm̃†_k = dm†_k + μ_k e', with e' the
  derivative dual strain; the itau pairing's derivative, pointwise in the
  energy contraction: ⟨dm̃†, dE m + dA d⁰ + dB d¹⟩ + ⟨m̃†, dE vm + dA vd⁰
  + dB vd¹⟩ + dh ⟨m̃†, d²E m + d²A d⁰ + d²B d¹⟩ with (vm, vd⁰, vd¹) the
  tangent run's states; and the recursion dm† = E dm̃† + dE dh m̃†,
  dd† = Σ_k (A dm̃†_k + dA dh m̃†_k);
- the start: the transposed instantaneous solve for both dual states.

Returns `KernelDerivatives3D` as the continuous sweep does: the thickness
and initial-sea-level views of the derivative load duals (no lumps, as in
1D), the itau kernel's derivative with its spherical reduction, and the
log-viscosity kernel's derivative −(δitau K + itau dK). Cost: two
transposed fixed points per node instead of one. Distributed (§17);
the material kernels and their derivatives alongside (§15).

GATES (`TestSecondOrderAdjoint3D`, lmax 8, prem8):
`DiscreteZeroLateralMatchesThe1DDiscreteSweep` — with δ = 0 at the GLL
points the kernels and derivatives equal the 1D discrete second-order
sweep's at 1e-7 (thickness views, spherical reductions of the itau
kernel and its derivative). `DiscreteLateralHessianIsExact` — lateral
structure on (δ of amplitude 0.6, η = 1e22, 9 nodes over 10 kyr, a
sea-level measurement at the end): the carried kernels equal the
first-order discrete sweep's bitwise; ⟨H u, w⟩ against ⟨H w, u⟩ agree at
1.2e-13 (gate 1e-8); against ⟨g, F''[u, w]⟩ from the discrete 3D
second-order tangent-linear run (the uv term of the log parameterisation
accounted for) at 1.5e-13 (gate 1e-8); and central differences of the
discrete first-order kernels along u (δ and the thickness moved
together, w's itau direction taken at the moved operator) paired with w
converge at second order (ratio 3.99994 over eps 0.02 → 0.01), Richardson
landing at 7.6e-11 (gate 1e-6). The continuous sweep's gates (§11) are
unchanged; the two are the same problem's commute pair, as in 1D.

## 15. The material in the second-order runs

The directions u, v of the second-order machinery (§8–§12, §14) gain a
part in the moduli — `MaterialDirection` (1D, nodal) / `MaterialDirection3D`
(3D, grid values): δκ, δμ with the branch moduli as fixed fractions and
the relaxation times held — so that the Hessian's
action covers every pairing of (ice, itau, κ, μ): `SecondOrderDirection::
materialU/materialV`, `AdjointSolver::SecondOrderKernels(..., material)`,
`KernelDerivatives::kappaKernel/muKernel` (and the 3D forms with their
spherical reductions). New design, no pyslfp counterpart. Everything
runs under any rotational feedback. The
moduli's directions and kernels are held at the memory points, GLL or
Gauss, in 1D (§13c, §13e) as on the 3D coupling's memory nodes. The 3D
forms are distributed (§17).

**The second-order tangent run** (`TangentLinearSolver::RunSecondOrder`,
3D alike). The step map is affine in the moduli, so the second variation
is the bilinear pairing of each direction's material forces with the
other's tangent trajectory: for the ordered pairs (a, b) = (u, v), (v, u)
with a material in a, the forces −dK_eff[δm_a] x'_b + Σ_k f_k δμ_a
⟨m̃'_{k,b} + drive_b⟩ enter each step through `AddForce` (1D) /
`OpenStep::AddForce` (3D), where x'_b is b's FULL tangent solution at the
node — recovered from b's recorded snapshot by one unrelaxed solve under
its loads and memory, plus b's own unrelaxed material force's response
−K_U⁻¹(dK_U[δm_b] x_n − MemVar(δμ_b, m⁺_n)) on the base when b has one
(the unrelaxed and effective forms of the material force are equal, the
end solution satisfying unrelaxed equilibrium with its new memory) — and
m̃'_b + drive_b is b's decayed tangent memory with b's own itau drive, read
off a step opened on b's tangent state (`mem_` / `memory_` after
`AddMemory`): the drive's dB d¹ part is the mixed moduli–relaxation-time
term ∂²K_eff/∂μ∂h x_n in disguise. The start response is a's unrelaxed
variation on b's start solution, summed over the pairs. In 3D the shear
weight is δμ_a (1 − Σ_k f_k β_k(x)) with the BASE step's lateral weights
and the strain is the recorded tangent strain; `ForwardCursor3D`
serves the node before its current one, so the itau and material drives
of one step may ask for (node − 1, node) in either order.

**The discrete sweeps** (1D `DiscreteSecondOrder`, 3D `DiscreteSweep`).
By ⟨H v, u⟩ = ⟨λ, forcing''(u, v)⟩: the parts of forcing'' linear in u's
tangent trajectory transpose into SOURCES of the derivative dual state,
the rest pair with the first duals directly. Along a material v: the
dual of the full tangent solution receives −dK_eff[δm_v] y_n (the
derivative of the effective systems on the first sweep's solution; in 3D
the Born force on the first dual strain with the step's lateral weights,
plus the bulk term), added BEFORE the unit-response contraction so that it
reaches the sea level adjoint — no table derivative is needed, for the
same reason the forward tangent needs none; the derivative dual of the
decayed memory receives f_k δμ_v S y_n (the memory force's variation on
the first dual strain), which then flows through the kernel pairing's
B-weight terms (written as the decayed-memory duals against the end
strains) and the E/A recursion; the start likewise with the unrelaxed
systems. The material kernels' derivatives are the pairings of the
derivative duals with the forward trajectory plus the first duals with
v's tangent trajectory — its full solution (recovered as above) and its
decayed memory with v's full discrete drive (whose dB d¹ part is the
effective factor's derivative along v's itau) — and the start terms
alike. Exact: the Hessian symmetric and equal to the second-order run to
the solvers' tolerance.

**The continuous sweeps** (1D `ContinuousSecondOrder`, 3D `Sweep`). The
second-order adjoint equations are the derivative of the continuous
adjoint equations, which have the forward form, so along a material v the
derivative state is driven by v's material forces on the ADJOINT
trajectory — the tangent's drive with the adjoint states as the base: on
each reversed step's end solution and decayed memory (`AddForce` on the
derivative state's step, the 3D shear weight with the step's own lateral
weights), on the equilibrium solutions at the observations (an extra
response channel of the closure), and on the instantaneous jump
solutions, whose fingerprint moves with the unrelaxed table — carried as
the response channel of v's unrelaxed variation on the jump solution (the
forward start's pattern), its sea level the derivative LUMP of the
derivative duals (a material v is the one direction whose derivative
duals carry lumps). Three further pieces: the
itau pairing's branch weight μ_k moves with δμ (the term δμ_v/μ_U times
the pairing); the derivative state's strain after a jump must carry v's
material response (what `StartStep` would regenerate lacks it; the
trapezoid's α d⁰ reads it — without it the convergence to the discrete
derivatives drops to first order); and the material kernels' derivatives
pair the derivative displacement with the forward trajectory and the
adjoint displacement with v's tangent solution and END memory (no drive:
the continuous pairing is with the full forward memory). At the commute
level, falling at the scheme's order under refinement.

GATES. 1D: `TangentLinear.SecondOrderRunInTheMaterial` — u, v in the
loads, itau and the moduli (1 % of the moduli): the second-order run
against central differences of the first-order run on the shifted
material (the rheology rebuilt at the same fractions, the inverse times
and loads moved), second order with Richardson at 1e-6, both schemes,
symmetric, the moduli–loads block non-zero;
`SecondOrderAdjoint.MaterialBlocksAreExactAndSymmetric` — the discrete
sweep with the material kernels on: carried kernels at 1e-12,
⟨H u, w⟩ = ⟨H w, u⟩ = ⟨g, F''[u, w]⟩ at 1e-9, central differences of the
full gradient along w (material shifted, itau and thickness moved) paired
with u at second order, Richardson at 1e-6, both schemes;
`SecondOrderAdjoint.ContinuousConvergesToDiscrete` carries the
material blocks: ice, viscosity and material gaps and the mismatch with
the continuous second-order run all fall by > 3 from 17 to 33 nodes
(along a material-only v the gaps are 0.06 %, 0.23 %, 0.04 % at 17 nodes,
ratio 4 over 17 → 33 → 65). 3D:
`TangentLinear3D.SecondOrderRunInTheMaterial` (loads, lateral log
viscosity, angle-uniform moduli; lateral viscosity on; second order,
Richardson 1e-6, symmetric);
`SecondOrderAdjoint3D.DiscreteMaterialHessianIsExactAndSymmetric`
(lateral directions in everything: carried kernels at 1e-10, symmetry and
the adjoint identity with the discrete second-order run at 1e-8);
`DiscreteZeroLateralMatchesThe1DDiscreteSweep` and
`ZeroLateralMatchesThe1DSweep` with an angle-uniform material v and
the material kernels (their spherical reductions and derivatives against
the 1D at 1e-7 / 1e-6); `ContinuousMaterialHessianConverges` (lateral
directions, the continuous blocks against the discrete and the mismatch
with the continuous run below 10 % at 9 nodes and smaller at 17).

## 17. The derivatives over radial segments

The derivative paths run under the radial distribution of
[radial-partitioning.md](radial-partitioning.md), as the forward step,
the continuous 3D adjoint and the 3D tangent-linear runs do:

- **The discrete 3D transpose** (first and second order, §12, §14, with
  the material of §15): each rank transposes its own memory nodes' part
  of every pass; the duals of the load and centrifugal coefficients are
  the ranks' PARTIAL contractions of the unit responses (whole) with the
  forces (held in shares, each rank its own rows) summed over the ranks;
  every rank solves the sea level adjoint alike from them and rank 0's
  answers serve all, a failure failing on all; the surface duals and the
  degree-0 moment row join the forces on the rank holding their rows
  (`DegreeOperator::Holds`); the transposed fixed point's residual is the
  ranks' maximum, as the forward's. The forward step's conventions
  (`SeaLevelEvolution3D::March`) transposed.
- **The 3D material tangent, the 3D material kernels of both sweeps and
  the material directions of the 3D second-order sweeps** need nothing
  of their own: their pieces — the Born force assembled on the rank's own
  nodes, the bulk term's dilatation through the scalar companion, the
  unrelaxed responses through `SolveForces`, the closures' extra response
  channels through the collective `Surface` and the summed moment — are
  the forward step's.
- **Threads.** The derivative loops are threaded through
  `detail::ForEach` / `ForEachTask` (the 3D sweeps over nodes
  and grid values, the 1D sweeps over degrees), as are the static Born
  perturbation's pointwise loops.

GATES: `RadialRanks.DerivativesMatchTheOneProcess` (threads as ranks,
three ranks, the 1e22 mantle with lateral viscosity, GLL memory nodes):
the 3D tangent-linear runs of both methods along loads, a lateral log
viscosity and lateral moduli; the first-order sweeps of both methods
with the material kernels; the second-order sweeps of both methods along
that direction with the kernels' derivatives — the ice views on every
rank and each rank's slices of the 3D kernels and derivatives against
the one process on the same cuts, all at 1e-11.
`Distributed.DerivativesMatchTheOneProcess` is the same under mpiexec
(TestDistributed), on 2, 3 and 4 ranks.
Not distributed: the static Born perturbation (`ElasticPerturbation3D`),
a small static problem run at benchmark size on one process.
