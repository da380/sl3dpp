# Stage 4 design: 3D viscosity — the pseudospectral lateral coupling

*Drafted 15 October 2026 from the old Fortran (`sl3d/src/module_gia.f90`,
`module_mat.f90`), plan §3 and pyslfp's `gia_notes.md`. This records the
algebra before it is implemented; equation references are Al-Attar &
Tromp (2014, GJI) Appendix D. The within-step lateral iteration is the
research content of the project (plan §3, open question 1).*

## 1. Scope and gates (plan §6, stage 4)

GSHTrans spin-2 machinery; grid-held memory variables; pointwise weight
fields; the within-step lateral iteration; the netCDF 3D model reader
(blocked on the planetmodel format work, §2.5 of the plan). Gates: the
zero-lateral limit against the 1D stepper at machine precision; the old
sl3d (`visco_test`, `sea_level_3d`) in the explicit/weak-contrast limit;
Δt self-convergence; AdGIA on a common model — the headline benchmark.

## 2. The canonical components and the degree ↔ spin maps

The trace-free symmetric memory/strain tensors are carried as five
complex components in the Fortran's ordering and normalisation
(`memdot_bvec_all`), spins N = (−2, −1, 0, +1, +2):

    A1 = A(--),  A2 = A(-0),  A3 = A(-+) = (1/2) A(00),  A4 = A(0+),
    A5 = A(++).

A real tensor gives the Hermitian pairs A5 = conj-pattern of A1 and
A4 of A2 under m -> −m; the Fortran stores all five complex and so, to
begin with, shall we (GSHTrans handles tensor components as complex
fields at each spin; exploiting the symmetry halves the transforms and
is an optimisation to take once the plain path is gated).

Per degree (ζ² = l(l+1)), from the spheroidal (U, V) and toroidal (W)
radial scalars and their r-derivatives, the deviatoric strain components
are (l ≥ 2; at l = 1 the ±2 rows vanish; at l = 0 only A3 survives):

    A1 = (1/2) ζ sqrt(ζ²−2) (V − iW) / r
    A2 = ζ ( r∂rV − V + U + i(W − r∂rW) ) / (2 sqrt(2) r)
    A3 = ( r∂rU − U + (1/2) ζ² V ) / (3 r)
    A4 = ζ ( r∂rV − V + U − i(W − r∂rW) ) / (2 sqrt(2) r)
    A5 = (1/2) ζ sqrt(ζ²−2) (V + iW) / r

In the 1D limit (W = 0, no lateral variation) these reduce to the
degree-basis scalars of the ported stepper: A3 ∝ e_R, A2 = A4 ∝ e_S,
A1 = A5 ∝ e_M, with the squared map constants reproducing the force
weights (4/3, k², k²(k²−2)) — the consistency identity the zero-lateral
gate (§7a below) rests on.

## 3. The memory force from the components

Given the five component fields of (weighted) memory Alm per (l, m) and
radial node, the force on the degree systems (`memdot_bvec_all`'s bvec
loops; integrand factors jac·w·r·μ_k, plus the derivative terms through
hprime with an extra r):

    spheroidal U-row:  sqrt(2) ζ (A2 + A4) − 4 A3,  deriv: 4 r A3
    spheroidal V-row:  ζ sqrt(ζ²−2)(A1 + A5) + 2ζ² A3
                       − sqrt(2) ζ (A2 + A4),       deriv: sqrt(2) ζ r (A2 + A4)
    toroidal  W-row:  i[ ζ sqrt(ζ²−2)(A1 − A5)
                       − sqrt(2) ζ (A2 − A4) ],     deriv: i sqrt(2) ζ r (A2 − A4)

This is the same dual-from-scalars structure as the 1D `Degree`
workspace, extended to the toroidal block and to complex coefficient
columns.

## 4. The toroidal degree operator (a stage-1 gap, needed here)

Lateral viscosity couples spheroidal and toroidal motion (the W-row
above), so stage 4 adds the toroidal per-degree system, in its
transversely isotropic form (Myhill & Al-Attar; verified against the
DSpecM1D assembly, whose two base matrices expand to exactly this):

    A_tor(W, W') = ∫ [ ζ² L (r∂rW − W)(r∂rW' − W')
                      + ζ²(ζ²−2) N W W' ] dr,

the TI analogue of the spheroidal D76 pattern (L with the shear-gradient
scalar, N with the (ζ²−2) term), reducing to the Fortran's `amat_tor` at
L = N = μ. Solid regions only — fluid carries no toroidal dof, so the solid
sections decouple into independent blocks (one banded factorisation
with absent fluid dofs covers them; the blocks are SPD separately).
No gravity, no fluid terms, no DtN. At l = 1 the form is singular
(rigid rotation W ∝ r: both terms vanish), to be handled as the
Fortran does — check `get_depth`/`gvn_tor` use at l = 1 during
implementation; the expected resolution is dropping l = 1 toroidal
response into the angular-momentum bookkeeping of the rotation
feedback. Verification: reciprocity/symmetry, an analytic homogeneous-
sphere toroidal response, and the Fortran once the time-dependent
cross-checks start.

## 5. The step with lateral viscosity

Viscosity η(x) = η̄(r) exp(δ(x)) with δ the (band-limited) log
perturbation (plan §2.5: the file stores log viscosity). Per branch,
itau(x) nodal on the grid; per (scheme, Δt) the weight FIELDS
e(x), α(x), β(x) from h(x) = Δt·itau(x) — StepWeights pointwise, exactly
as the 1D code does per node.

Split (plan §3): μ_eff(x) = μ̄_eff(r) + δμ_eff(x), with μ̄_eff built from
the reference τ̄(r) (laterally averaged η̄ by default) and factorised per
degree by the existing flavour machinery. The within-step iteration, one
combined fixed point under the SLE's existing Anderson mixer:

    K̄_eff u^{(j+1)} = F_load(ocean, rotation | u^{(j)})
                      + b_mem(m̃(x))                       [grid pipeline]
                      − b_lat(u^{(j)})                     [grid pipeline]

where b_lat is the pseudospectral application of δK_eff: synthesise the
strain components of u^{(j)} to the grid, multiply pointwise by
δμ_eff(x) = Σ_k (φ1(h_k(x)) − φ1(h̄_k(r))) μ_k(x), analyse, and assemble
through §3's force rows — the SAME pipeline as the memory force, with a
coefficient field in place of the memory state. The memory state
m̃(x) = e(x) m + α(x) d^n is pointwise on the grid. After convergence,
m⁺ = m̃ + β(x) d^{n+1} pointwise.

The ocean-load and rotation fixed points fold in as today (the SLE's
load iteration), so the combined iteration costs one banded back-solve
per degree plus a bounded number of spin transforms per pass.
Fallbacks if contraction fails under strong contrasts (plan §3): damped
Picard, a better reference profile (per-degree or worst-case), IMEX.

## 6. Data layout, dealiasing, memory

- Memory state: per branch, 5 complex component fields over (viscous
  radial memory nodes × angular grid). Radial memory nodes are the GLL
  nodes of viscous elements (the Gauss-point reduction of plan §5.5 is
  the planned ~20% saving and an independent-discretisation benchmark
  axis, added after the GLL path is gated).
- Grid: one `SurfaceGrid` with nMax = 2 and `ForBand` oversampling 1.5
  (the 3/2 rule) for the pointwise products — an improvement over the
  Fortran, which never dealiased (plan §2.1).
- Transforms are issued **batched** (one call per spin component, the
  radial nodes as the batch — GSHTrans guarantees exact equality with the
  per-field calls) with the BLAS matrix kernel and threaded execution as
  options. Measured on the PREM-deck Maxwell body (15 Oct 2026, 8
  threads): a full synthesis + analysis pass over all five components,
  lmax 128, 1630 graded-mesh viscous nodes: 51.6 s naive single-field
  loop kernel → 12.9 s batched loop → **5.0 s batched matrix kernel** —
  about 10x; lmax 64 (1170 nodes): 0.8 s. The Hermitian m >= 0 reduction
  (halving the batch) and the Gauss-point memory nodes (~20% fewer)
  compound on top.
- Budget at lmax 256 (grid ~385 × 770 with 1.5× oversampling), graded
  mesh (~600 radial nodes, ~450 viscous): 5 comps × 450 × 385 × 770
  complex ≈ 11 GB per branch... **the oversampled grid is for products
  only**: m is band-limited at lmax, stored as coefficients or on the
  unoversampled grid (257 × 515 ≈ 4.8 GB complex per branch), with
  synthesis to the big grid per step. Final layout to be settled against
  measurements; the Wigner table adds 2.6–5.2 GB (plan §2.4).

## 7. Implementation order and gates

a. **Spin machinery**: `SurfaceGrid` grows spin-N transforms (complex
   fields, N = −2..2); the degree ↔ spin maps of §2 and force rows of §3
   as a `TensorPipeline` beside the 1D `Degree` workspace. GATE: with
   zero lateral perturbation (δ = 0, weights uniform in angle), one full
   grid-pipeline step must reproduce the 1D stepper's state and
   solutions at machine precision, degree by degree — the round trip
   through synthesis/multiply/analysis is then exact up to transform
   round-off. This is the decisive correctness gate and needs no
   external reference.
b. **Toroidal operator** (§4), with its own tests.
c. **Lateral weights and the within-step iteration** (§5). GATES: Δt
   self-convergence at order 2 with lateral viscosity on; the old sl3d
   in the weak-contrast limit (tolerances set by its explicit stepping
   and lack of dealiasing); contraction diagnostics over a contrast
   ladder (the research measurements).
d. **AdGIA on a common model** — the headline benchmark (plan §1).

## 5a. Measured within-step convergence (15 Oct 2026, first numbers)

PREM-deck Maxwell mantle, lmax 16, a sin(theta)cos(phi) log-viscosity
pattern of amplitude A (peak-to-peak contrast e^{2A}), unit loads on all
degrees, trapezoid steps; iterations to 1e-11 on the second step:

    contrast   dt/tau:   0.5      2       10
    3x (plain Picard)     12      22       67
    3x (Anderson 5)       10      17       23
    10x    Picard         21      54     FAIL (oscillates)
    10x    Anderson       14      29       52
    100x   Picard         64     FAIL    FAIL
    100x   Anderson       27      68     FAIL*
    1000x  Anderson       49     147     FAIL*
    10^4x  Anderson       92     FAIL*   FAIL*

(*) the Anderson failures are slow grinds (residual 1e-3..1e-9 at the
200-iteration cap), not divergence — unlike Picard's oscillation at
residual ~1.8 — so a larger budget, a better reference profile
(per-degree or worst-case, plan §3) or IMEX would extend the envelope.
Headline: Anderson mixing roughly squares the practical contrast ceiling
at fixed dt, and one-decade contrasts at dt = 10 tau — far beyond the old
explicit code's dt <= tau_min limit — converge in ~50 passes. At the
saturated extreme (h >> 1 everywhere) the lateral term vanishes and the
iteration trivialises, as it should.

## 5b. The Fortran gate (visco_test)

The old code's `visco_test` is the weak-contrast external gate: PREM
(`prem.simp.vm1`, natural splines), the analytic lateral pattern
s = s(r) exp(0.3 (Y_2,-2 + Y_2,2)) (our delta is minus that, being a
log-viscosity perturbation), a constant zonal load RATE
sigma_dot_l = exp(-2 pi (l+1)/10.5) sqrt((2l+1)/4 pi) x 5515 / 30 yr
(SI), explicit RK2 at dt = 50 yr against our trapezoid at the same dt.
The driver was patched (I/O only: an lmax argument and coefficient
output, `tests/data/stage4_fortran_visco.txt` at lmax 32); the physics
is untouched, so the comparison pits the two codes' discretisations —
rate-form explicit stepping on a coarse uniform mesh against implicit
stepping on the graded mesh — against each other at physical accuracy.
Measured (lmax 8, laptop, 15 Oct 2026): the two codes agree at
3-13 x 1e-4 relative in the surface U and phi coefficients over the full
5 kyr history for every degree l >= 2 — independent discretisations in
space AND time, with the lateral viscosity on, agreeing at observational
accuracy. The drift grows slowly with time in phi, consistent with the
Fortran's explicit RK2 truncation at dt/tau ~ 0.5. Degree 1 is
excluded from the comparison: sl3dpp works in the centre-of-mass frame,
while the Fortran's l = 1 handling was a known loose end (deferred until
rotation was added, which it never was), so its l = 1 output is neither
our frame nor a settled one — the driver reports it separately for the
record.

Detailed runs are server work (the original always was): the kit is
`benchmarks/fortran_gate` plus `scripts/run_fortran_gate.sh`, which
builds both repos and runs the comparison at a chosen lmax; the lmax-32
fixture is committed for reference. The sl3d driver patch (I/O only)
must be committed in that repo for the kit to work from a fresh clone.

## 7a. Nonlinear Maxwell rheologies (readily buildable here)

Stress-dependent relaxation times — AdGIA's `RelaxationLaw`
(τ_k = τ_k0(x) F_k(state), Crawford et al. 2017 Appendix A) — fit this
stage's machinery with almost no new structure: τ already varies
pointwise on the grid, the local deviatoric stress invariant is
computable from the (d − m) component fields the pipeline already
synthesises, and re-evaluating the weight fields e(x), α(x), β(x) from
the updated τ(x, state) folds the nonlinearity into the SAME within-step
fixed point as the lateral iteration (one more contribution to the
Anderson-mixed residual). pyslfp's notes defer exactly this to the 3-D
code (`gia_notes.md`: "it belongs with lateral viscosity variations").
Planned as an option on `Rheology` (a per-branch law hook), implemented
once the linear lateral iteration is gated, and benchmarked against
AdGIA's `RelaxationLaw` path.

## 8. Open questions (beyond plan §8)

- l = 1 toroidal null space handling (§4).
- Whether the SLE needs toroidal surface channels: no first-order
  coupling to sea level or gravity (toroidal motion is divergence-free
  and radial-displacement-free at the surface), so the toroidal block
  enters through the memory coupling alone; surface toroidal velocity is
  an observable output only.
- The reference profile τ̄(r): laterally averaged log-viscosity vs
  worst-case; possibly per-degree. To be measured in (c).
- Hermitian-pair exploitation in the transforms (halves the spin
  work), once gated — **via GSHTrans's tensor layer, not by hand**
  (noted 15 Oct 2026): `TensorField`'s Reality parameter and its orbit
  table (`Tensor/Orbits.hpp`) encode exactly the pair structure derived
  here (sign-and-conjugation relations between component slots, the
  anti-conjugating spin-1 pair included), so the reduced storage and
  transforms come with the signs tested in the library. Kernel
  implications checked: the kernel contraction is pointwise on the grid,
  where negative-spin fields reconstruct by sign * conj for free; the
  adjoint fields are real tensors too; the one discipline is that the
  discrete adjoint must transpose the reduced pipeline consistently
  (as with the Gauss-point memory migration).
- GSHTrans's field algebra (noted 15 Oct 2026): not needed for the
  stepping pipeline (plain batched transforms + hand loops), but the
  stage-5 kernel accumulation is a pointwise tensor contraction
  mu s (d - m):(d^+ - m^+) whose null-basis metric signs (the g(+,-) = -1
  pairings across the component slots) the library's tensor types encode
  — use them there rather than hand-rolling the contraction weights.
