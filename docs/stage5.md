# Stage 5 design: adjoints — discrete and continuous

*Drafted 16 October 2026 from pyslfp's recent adjoint work
(`viscoelastic/evolution.py`, commits `d215e3f`..`5479ce9`: discrete and
continuous adjoints, quelling, 1D viscous kernels, checkpointing) and
Yu et al. 2025 (equation references below are to that paper). pyslfp is
the porting template for the 1D path; the 3D path is this stage's new
content. This records the design before implementation.*

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
2. **3D kernels** (new): the continuous adjoint of the coupled 3D
   evolution and the grid viscosity kernel μ s (d−m):(d†−m†); the
   discrete transpose of the reduced pipeline as the option.
   **Both kernel granularities are deliverables** (David, 16 Oct 2026):
   the full 3D viscosity kernel on the (memory nodes × angular grid),
   AND the spherically averaged 1D radial kernel as pyslfp provides —
   the latter both from the 1D sweeps directly and as the spherical
   average of the 3D kernel, whose agreement in the zero-lateral limit
   is a free consistency gate between the two pipelines.
3. Deferred within the stage: linearisation about a migrating-shoreline
   trajectory (pyslfp's `ocean_functions`) waits until sl3dpp's forward
   evolution grows migration (the SLE's nonlinear solve exists; the
   driver doesn't use it yet); decay-time functionals
   (pyslfp `functionals.py`) are a front end over Measurement and port
   separately if needed.

## 3. The 1D sweeps (pyslfp's algebra, verbatim)

The authoritative statements are pyslfp `evolution.py` docstrings and
code; what follows fixes the sl3dpp mapping only.

- **Discrete** (`adjoint_ice_history`): one reversed sweep carrying the
  duals (m†, d†) per degree. Per step: the duals of the end solve
  gather through the strain transpose (`Degree::DualFromScalars` — the
  plain transpose of `Degree::Strain` under the nodal pairing); the
  unit load/tide responses contract them to load/tide coefficient
  duals; those join the measurement duals through the engine adjoint
  (`SeaLevelSolver::SolveAdjoint`, the stage-2 transpose); its channel
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
  loads. Roundoff grows as exp(Σh) across a gap — the stride buys
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
  exactly as forward. The viscosity kernel is then the pointwise grid
  contraction μ s (d − m):(d† − m†) accumulated at the memory nodes
  over the reversed nodes with the split trapezoid; the null-basis
  metric signs (g(+,−) = −1 pairings across the component slots) come
  from GSHTrans's tensor types, not hand-rolled weights (stage4.md
  §8). The reality reduction gives the derived spins by sign·conj for
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
  research measurement alongside stage 4's §5a table).

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

Fixtures: `scripts/make_stage5_fixtures.py` driving pyslfp, committed
under `tests/data/` with provenance headers, per the stage-0 mechanics.
