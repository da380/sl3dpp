# 3D viscosity: the pseudospectral lateral coupling

*The algebra of the pseudospectral lateral coupling, after the old
Fortran (`sl3d/src/module_gia.f90`, `module_mat.f90`), with the
within-step convergence and the Fortran gate; equation references are
Al-Attar & Tromp (2014, GJI) Appendix D. The within-step lateral
iteration is the research content of the project
([design.md](design.md) §6).*

## 1. Scope and gates

GSHTrans spin-2 machinery; grid-held memory variables; pointwise weight
fields; the within-step lateral iteration; the 3D model reader
(`io::ModelViscosity`, over planetmodel's model file). Gates: the
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
A4 of A2 under m -> −m. The Fortran stores all five complex; sl3dpp
stores spins −2 and −1 as complex fields and spin 0 as a real field,
and reconstructs +1 and +2 by the orbit relation, which halves the
transforms (§8).

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
gate (§7, item a) rests on.

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

## 4. The toroidal degree operator

Lateral viscosity couples spheroidal and toroidal motion (the W-row
above), so the 3D problem needs the toroidal per-degree system, in its
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
(rigid rotation W ∝ r: both terms vanish).

**Degree 1 is included** ([design.md](design.md) §8). The
rigid rotations, one per order in each solid region that fluid separates
(mantle, inner core), form a null space that is removed by convention.
The right-hand side is torque-free, so the singular system is consistent.
Which frame does not matter, so a convenient gauge is used (one pinned
dof per solid region). What matters is that measured,
benchmarked and tested quantities are invariant under the rigid
rotations, which is also the condition for the adjoint equations to
exist. The l = 1 toroidal response cannot be dropped into the
angular-momentum bookkeeping of the rotation feedback: differential
degree-1 rotation carries strain through W − r∂rW, and lateral viscosity
drives it. `ToroidalOperator` accepts l = 1, pinning one
dof per solid region (its top node by default), and the toroidal force,
solves and strain recovery run from l = 1. The Fortran factors the
unconstrained degree-1 form by banded LU (`aa_tor_1`) and fixes no frame,
so it is no reference for this mode.

Verification: reciprocity/symmetry, an analytic homogeneous-sphere
toroidal response, and the Fortran gate (§5b, l ≥ 2).

## 5. The step with lateral viscosity

Viscosity η(x) = η̄(r) exp(δ(x)) with δ the (band-limited) log
perturbation (formed from a planetmodel model file's viscosity by
`io::ModelViscosity`). Per branch,
itau(x) nodal on the grid; per (scheme, Δt) the weight FIELDS
e(x), α(x), β(x) from h(x) = Δt·itau(x) — StepWeights pointwise, exactly
as the 1D code does per node.

Split ([design.md](design.md) §6): μ_eff(x) = μ̄_eff(r) + δμ_eff(x), with μ̄_eff built from
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

The ocean-load and rotation fixed points fold in through the SLE's
load iteration, so the combined iteration costs one banded back-solve
per degree plus a bounded number of spin transforms per pass. Under
strong contrasts: damped Picard, or the step by conjugate gradients
(§5a''); a better reference profile (per-degree or worst-case) and IMEX
are not built.

## 5a. Measured within-step convergence

PREM-deck Maxwell mantle, lmax 16, a sin(theta)cos(phi) log-viscosity
pattern of amplitude A (peak-to-peak contrast e^{2A}), unit loads on all
degrees, trapezoid steps; iterations to 1e-11 on the second step, the
iterate the strain's coefficients and the internal variables at Gauss
points, over the history depth and adaptive mixing. Every step-1 result
is checked: `s1` marks a case whose first step failed, so the second is
not reported. DIV is a residual above 1e-2 (or non-finite) at the 200-pass cap, cap a
smaller one. "always" mixes from the first pass at the given depth;
"adaptive" is the default iteration (below), with the passes it mixed.

    contrast dt/tau | Picard | always m=1    2    3    5    8 | adaptive m=5 | adaptive m=8
         3x    0.5  |   12   |          10   10   10   10   10 |   12 (0)     |   12 (0)
         3x    2.0  |   21   |          19   17   17   16   15 |   21 (0)     |   21 (0)
         3x   10.0  |   45   |          27   25   22   21   20 |   21 (11)    |   21 (11)
        10x    0.5  |   20   |          15   15   14   14   14 |   20 (0)     |   20 (0)
        10x    2.0  |   44   |          42   33   32   28   25 |   28 (17)    |   27 (16)
        10x   10.0  |   s1   |          s1   71   68   50   43 |   50 (46)    |   43 (39)
       100x    0.5  |   63   |          38   30   31   25   25 |   25 (22)    |   25 (22)
       100x    2.0  |   s1   |         141  111   85   68   59 |   68 (66)    |   58 (56)
       100x   10.0  |   s1   |          s1   s1   s1   s1   s1 |   s1         |   s1
      1000x    0.5  |  193   |          91   68   72   48   48 |   48 (46)    |   48 (46)
      1000x    2.0  |   s1   |          s1   s1   s1  147  127 |  147 (145)   |  125 (123)
      1000x   10.0  |   s1   |          s1   s1   s1   s1   s1 |   s1         |   s1

Depth 2-3 gets most of the gain in mild cases and falls well behind at
100x and above; depth 8 is somewhat better at the hard end for 18
history vectors against 12. Picard never converges where Anderson does
not. Anderson mixing roughly squares the practical contrast ceiling at
fixed dt, and one-decade contrasts at dt = 10 tau — far beyond the old
explicit code's dt <= tau_min limit — converge in ~50 passes. At the
saturated extreme (h >> 1 everywhere) the lateral term vanishes and the
iteration trivialises, as it should.

What it costs (lmax 64, 8 threads, the laptop): at the benchmark's weak
contrast (1.26x, 50 yr steps) every depth takes 7 passes a step, while
a pass costs 0.62 s under Picard against 0.87 s under depth-5 Anderson
(0.81 at depth 2), whose history also holds 1.9 GB more at peak. The
mixer is memory-bound: an update moves ~3 GB. So Anderson pays only
where it saves more than ~30% of the passes, which in the table is
where Picard contracts by less than about half a pass.

Hence the default iteration (`Viscoelastic3D::IterationControl`,
`anderson_rate = 0.5`): Picard until a pass shrinks the residual by less
than half, then Anderson (depth 5) for the rest of the step. It is plain
Picard wherever Picard is quick, and pass for pass Anderson wherever it
is not, the switch costing a pass or two. `anderson_rate = 0` mixes
from the first pass.

A non-finite pass ends the iteration unconverged, with a NaN residual,
on the last finite iterate.

**Resolution of the strong contrasts.** Are the hard cases resolved at
lmax 16, or is part of their difficulty truncation? The log-viscosity
is degree 1, but the lateral weight beta(h e^-delta) - beta(h) it enters
through has every degree. With the radial mesh fixed, loads on degrees
<= 8 only, and lmax 16, 24 and 32 (default iteration):

    contrast dt/tau | weight's amplitude above   | passes on step 2 | U (l <= 8) against lmax 32
                    |   16      24      32       |  16   24   32    |   lmax 16   lmax 24
        10x    0.5  | 9e-14   1e-14   1e-14      |  20   20   20    |   6e-16     5e-16
        10x    2.0  | 2e-11   1e-14   1e-14      |  28   28   28    |   9e-14     2e-15
       100x    0.5  | 4e-08   4e-12   1e-14      |  25   25   26    |   7e-12     1e-13
       100x    2.0  | 7e-07   7e-10   4e-13      |  70   69   69    |   3e-10     7e-14
      1000x    0.5  | 6e-06   3e-08   5e-11      |  48   49   49    |   1e-08     3e-13
      1000x    2.0  | 3e-05   5e-07   5e-09      | 154  154  155    |   3e-08     1e-11

The pass counts do not depend on lmax, so the hard cases are hard in
the continuum (the lateral term against the radial reference), not by
truncation; and the low-degree response at lmax 16 is good to 3e-8 even
at 1000x. Nothing here can go negative for want of resolution: itau(x)
is the reference times exp(-delta) and the step weights come from it
pointwise on the grid, so the moduli stay positive whatever the
truncation. What this smooth pattern does not test is a field with sharp
lateral edges (cratons, slabs): its weight spectrum decays slowly, so
the pseudospectral products alias and the solution needs a higher lmax,
and a delta supplied as a truncated expansion overshoots (exaggerating
the contrast, though exp keeps it positive).

## 5a'. Lateral elastic moduli in the same fixed point

A lateral perturbation of the unrelaxed shear, s(x) relative, and of the
bulk modulus, b(x) relative, enters the step through the same weight as
the viscosity: W(x) = Σ_k μ̄_k (β_k(x) − β_ref,k(r)) − s(x) (μ̄_U − Σ_k
μ̄_k β_k(x)), the fixed memory force Σ_k μ̄_k (1 + s(x)) m̃_k, and a
scalar companion −κ̄ b(x) D(u) on the dilatation, assembled through the
bulk pipeline of the static perturbation. The unrelaxed start solve
becomes a preconditioned conjugate-gradient solve on the reference
systems. Gates and the within-step cost are in adjoints.md §13k.

## 5a''. The step by conjugate gradients

The step is a symmetric positive definite system in the degree
solutions (adjoints.md §13n), and `Viscoelastic3D::Options::solver =
ConjugateGradients` closes it by preconditioned conjugate gradients
with the reference effective solve as the preconditioner, from the
Picard iterate. The §5a ladder (lmax 16, the second step, to 1e-11 in
each method's own measure), the default fixed point's passes (Anderson-
mixed ones in brackets) and the conjugate-gradient iterations, with the
second step's wall-clock time on the idle laptop (16 threads):

    contrast dt/tau | fixed point passes (mixed)  time | conjugate gradients  time
         3x    0.5  |    12 (  0)               0.1 s  |     8               0.1 s
         3x    2.0  |    21 (  0)               0.1 s  |    13               0.1 s
         3x   10.0  |    21 ( 13)               0.1 s  |    17               0.1 s
        10x    0.5  |    20 (  0)               0.1 s  |    12               0.1 s
        10x    2.0  |    29 ( 22)               0.2 s  |    23               0.1 s
        10x   10.0  |    50 ( 48)               0.3 s  |    34               0.2 s
       100x    0.5  |    26 ( 21)               0.1 s  |    23               0.1 s
       100x    2.0  |    65 ( 62)               0.5 s  |    46               0.3 s
       100x   10.0  |    FAIL (step 1)                 |    90               0.5 s
      1000x    0.5  |    49 ( 46)               0.3 s  |    42               0.3 s
      1000x    2.0  |   138 (136)               0.7 s  |    83               0.5 s
      1000x   10.0  |    FAIL (step 1)                 |   183               1.0 s

Fewer iterations than passes everywhere, less time at every row, and
convergent where the fixed point's first step diverges. The iteration
count grows like the square root of the contrast's effect on the
conditioning, as conjugate gradients should; nothing here needs a
better reference profile. An iteration is a pass's grid pipeline plus
the reference band products on the columns (hand-written, threaded
over the degrees), the preconditioner's solve and the Krylov algebra:
at this size (a 4 ms pipeline) about 1.3 passes
(`Lateral.DISABLED_StepSolverProfile` times the pieces), and at
production sizes, where the pipeline is 0.6 s a pass, the two cost
alike. With
sea level the engine's closure runs as an outer iteration around
warm-started inexact solves (`OpenStep::Nest`), which at weak contrasts
costs about 1.4 times the fixed point (adjoints.md §13n for the
numbers), so the fixed point stays the default and the conjugate-
gradient step is the choice for strong contrasts. Probe:
`Lateral.DISABLED_StepSolverLadder`.

## 5b. The Fortran gate (visco_test)

The old code's `visco_test` is the weak-contrast external gate: PREM
(`prem.simp.vm1`, natural splines), the analytic lateral pattern
s = s(r) exp(0.3 (Y_2,-2 + Y_2,2)) (our delta is minus that, being a
log-viscosity perturbation), a constant zonal load RATE
sigma_dot_l = exp(-2 pi (l+1)/10.5) sqrt((2l+1)/4 pi) x 5515 / 30 yr
(SI), explicit RK2 at dt = 50 yr against our trapezoid at the same dt.
The driver was patched (I/O only: an lmax argument and coefficient
output, `tests/data/fortran_visco_test.txt` at lmax 32); the physics
is untouched, so the comparison pits the two codes' discretisations —
rate-form explicit stepping on a coarse uniform mesh against implicit
stepping on the graded mesh — against each other at physical accuracy.
Measured (lmax 8): the two codes agree at
3-13 x 1e-4 relative in the surface U and phi coefficients over the full
5 kyr history for every degree l >= 2 — independent discretisations in
space AND time, with the lateral viscosity on, agreeing at observational
accuracy. The drift grows slowly with time in phi, consistent with the
Fortran's explicit RK2 truncation at dt/tau ~ 0.5. Degree 1 is
excluded from the comparison: sl3dpp works in the centre-of-mass frame,
while the Fortran's l = 1 handling is unsettled, so its l = 1 output is
neither our frame nor a settled one — the driver reports it separately.

Detailed runs are server work: the kit is
`benchmarks/fortran_gate` plus `scripts/run_fortran_gate.sh`, which
builds both repos and runs the comparison at a chosen lmax; the lmax-32
fixture is committed for reference. The kit needs an sl3d checkout
carrying the driver patch (I/O only).

## 6. Data layout, dealiasing, memory

- Memory state: per branch, the reduced components (spins −2 and −1
  complex, spin 0 real; §8) over (radial memory nodes × angular grid).
  Radial memory nodes are the Gauss points of viscous elements by
  default, ~20% fewer than the GLL nodes, which remain selectable
  ([design.md](design.md) §5).
- Grid: one `SurfaceGrid` with nMax = 2 and `ForBand` oversampling 1.5
  (the 3/2 rule) for the pointwise products — an improvement over the
  Fortran, which never dealiased.
- Transforms are issued **batched** (one call per spin component, the
  radial nodes as the batch — GSHTrans guarantees exact equality with the
  per-field calls), threaded, with GSHTrans's BLAS matrix kernel as an
  option. Measured on the PREM-deck Maxwell body (8 threads, the memory
  nodes at the GLL points): batching takes a full synthesis + analysis
  pass over the five components at lmax 128 (1630 graded-mesh viscous
  nodes) from 51.6 s (single-field calls) to 12.9 s; over the reduced
  components the pass is 4.9 s with the loop kernel and 4.4 s with the
  matrix kernel at lmax 128, and about 0.6 s at lmax 64 (1170 nodes),
  where the two kernels are alike. Grid construction is 0.1 s at lmax
  128. The matrix kernel is opt-in, its gain being marginal.
- Budget: the internal variables are held on the coupling's oversampled
  grid at every memory node (a collocation representation), so one
  branch's tensor is about 0.7 GB at lmax 64, 4 GB at 128, 21 GB at 256
  and 98 GB at 512 ([design.md](design.md) §6); the Wigner table adds
  2.6–5.2 GB at lmax 256.

## 7. Components and gates

a. **Spin machinery**: `SurfaceGrid` carries spin-N transforms (complex
   fields, N = −2..2); the degree ↔ spin maps of §2 and force rows of §3
   are `GridCoupling`, with `SpinComponents.hpp`, beside the 1D `Degree`
   workspace. GATE: with
   zero lateral perturbation (δ = 0, weights uniform in angle), one full
   grid-pipeline step reproduces the 1D stepper's state and
   solutions at machine precision, degree by degree — the round trip
   through synthesis/multiply/analysis is then exact up to transform
   round-off. This is the decisive correctness gate and needs no
   external reference.
b. **Toroidal operator** (§4), with its own tests.
c. **Lateral weights and the within-step iteration** (§5). GATES: Δt
   self-convergence at order 2 with lateral viscosity on; the old sl3d
   in the weak-contrast limit (§5b; tolerances set by its explicit
   stepping and lack of dealiasing); contraction diagnostics over a
   contrast ladder (§5a, the research measurements).
d. **AdGIA on a common model** — run from AdGIA's benchmark suite.

## 8. Choices

- l = 1 toroidal: included, with one pinned dof per solid region as the
  gauge (§4, [design.md](design.md) §8).
- The SLE has no toroidal surface channels: there is no first-order
  coupling of toroidal motion to sea level or gravity (it is
  divergence-free and radial-displacement-free at the surface), so the
  toroidal block enters through the memory coupling alone; surface
  toroidal velocity is an observable output only.
- The reference profile τ̄(r) is the laterally averaged log-viscosity by
  default (§5); a worst-case or per-degree reference is not built.
- Hermitian pairs in the transforms. The tensors are real, so
  GridCoupling stores spins -2, -1 as complex fields and spin 0 as a
  real field; spins +1, +2 are reconstructed at the force exit by
  A^{(N)}_{lm} = (-1)^m conj(A^{(-N)}_{l,-m}) — the same orbit relation
  GSHTrans's tensor layer encodes (eq:reality, Tensor/TensorField.hpp;
  its sign table corroborates the convention the tests pin, the
  anti-conjugating spin-1 pair included). GSHTrans's TensorField
  transform batches the components of ONE tensor, while this pipeline's
  dominant win is batching across the ~10^3 radial memory nodes, so the
  pipeline makes cross-node batched calls (in Synthesise/Analyse) over
  the stored spins plus the real N = 0 path. The contract the reduction
  rests on — every degree-column set entering the pipeline is Hermitian,
  i.e. a real field — is stated in GridCoupling.hpp and the gate tests
  feed Hermitian data. Contractions are pointwise on the grid, where the
  derived spins reconstruct by sign * conj for free; adjoint fields are
  real tensors too; the discrete adjoint transposes the reduced pipeline
  consistently ([adjoints.md](adjoints.md) §6b, §12).
- m >= 0 columns. The degree boundary carries l + 1 columns per degree
  (the fields are real; Scatter writes each column's -m twin into the
  full-m spin blocks), halving the banded solves, the boundary loops and
  the solution storage. Public column convention: index m, m >= 0, m = 0
  real.
- Transforms cut with depth: deep memory nodes carry only the degrees
  whose truncation reaches them, and `transform_degrees` transforms node
  blocks by depth, each at its own degree
  ([radial-partitioning.md](radial-partitioning.md) §6).
- The grid (its Wigner table is the gigabytes) is shared: SurfaceGrid is
  a cheap handle to one immutable GSHTrans grid, and GridCoupling /
  Viscoelastic3D accept an existing grid. One grid per resolution: the
  sea level machinery's band-exact scalar grid and the coupling's
  oversampled spin-2 grid are deliberately different samplings.
- GSHTrans's field algebra is not used: the stepping pipeline needs only
  batched transforms and hand loops, and the 3D viscosity kernel's
  contraction μ_k m†:(d − m)/itau carries its metric on the reduced
  storage fixed by the 1D limit ([adjoints.md](adjoints.md) §6a), not
  taken from the library's tensor types.
