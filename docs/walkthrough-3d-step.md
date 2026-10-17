# Walkthrough: one 3D (lateral-viscosity) step, as the code stands

*Written 17 October 2026 against commit 88c58b5 (develop). This is a map
of what the code DOES, with file:line pointers, so it can be checked
against intent. The algebra and its derivation are in `stage4.md`; this
document does not repeat them. Where the code and the docs disagree, or
where something is correct but costly, it is flagged in §8.*

## 1. Who owns what

| Piece | Where | Role in the step |
|---|---|---|
| `SurfaceGrid` | `include/sl3dpp/SurfaceGrid.hpp` | Shared handle to one immutable GSHTrans grid (Wigner table); batched scalar and spin transforms |
| `GridCoupling` | `src/GridCoupling.cpp` | The memory nodes; grid and spectral tensor layouts; `Analyse` / `Synthesise`; force assembly; strain recovery |
| `SpinComponents.hpp` | `include/sl3dpp/SpinComponents.hpp` | The per-(l, m, node) algebra: strain components from (U, V, W), force rows from components |
| `DegreeOperator`, `ToroidalOperator` | `src/DegreeOperator.cpp`, `src/ToroidalOperator.cpp` | Per-degree banded systems: assemble, scale, factorise, solve |
| `ViscoelasticOperator` (`reference_`) | `src/Viscoelastic.cpp` | The 1D machinery: effective materials and Love numbers per (scheme, dt), inertia moment |
| `StepWeights.hpp` | `include/sl3dpp/StepWeights.hpp` | (e, alpha, beta) of the exponential trapezoid / backward Euler as functions of h = dt/tau |
| `Viscoelastic3D` | `src/Viscoelastic3D.cpp` | State, lateral weight fields, the step and its within-step iteration |
| `SeaLevelEvolution3D` | `src/Evolution3D.cpp` | The coupled time loop: strain, ocean load and rotation in one fixed point |
| `AdjointSolver3D` | `src/Adjoint3D.cpp` | Friend of `Viscoelastic3D`; steps the adjoint with the same machinery |

## 2. The data, its layout and its size

**Memory nodes.** The GLL nodes of every element where any rheology
branch is active, kept per (element, node-in-element), so a node shared
by two elements appears twice with its one-sided values
(`src/GridCoupling.cpp:30-41`). The elastic lithosphere and the fluid
core have no memory nodes: they appear in the linear systems only.

**The angular grid.** Gauss–Legendre, built by `ForBand` with
oversampling 1.5. It resolves degree ceil(1.5 L), so it has
ceil(1.5 L) + 1 colatitudes and about twice that many longitudes (the
next fast FFT size at or above 2·ceil(1.5 L) + 1). That is about 97 × 196
at lmax 64 and 193 × 390 at lmax 128.

**Layouts.**

- `TensorField` (grid): `c[0]` = spin −2 (complex), `c[1]` = spin −1
  (complex), `r` = spin 0 (real); each indexed `node * FieldSize + x`,
  colatitude-major. Spins +1 and +2 are never stored: they are the
  conjugate images of −1 and −2, a real tensor being assumed throughout
  (`include/sl3dpp/GridCoupling.hpp`, file comment).
- `SpectralTensor`: spins −2 and −1 over all orders (GSHTrans layout,
  index l² − N² + m + l), spin 0 in the real m ≥ 0 layout
  (l(l+1)/2 + m), each block per node, node-major.
- Degree columns (forces and solutions): m ≥ 0 only, l + 1 complex
  columns per degree, column-major `ndof × (l + 1)`. The m < 0 columns
  are implied by reality and never exist as data.

**Sizes** (memory-node counts from `stage4.md` §6, PREM graded mesh):

| | lmax 64 | lmax 128 |
|---|---|---|
| Memory nodes N_r | ~1170 | ~1630 |
| Grid points N_g | ~19k | ~76k |
| One `TensorField` (N_r · N_g · 40 B) | ~0.9 GB | ~4.9 GB |
| One real weight field (N_r · N_g · 8 B) | ~0.18 GB | ~1.0 GB |
| One `SpectralTensor` | ~0.2 GB | ~1.1 GB |

The grid tensor is about 4.5× the spectral one. The difference is the
1.5² oversampling plus spin 0 being stored real on the grid but m ≥ 0
complex in coefficients.

## 3. Construction: what is built when

`Viscoelastic3D::Viscoelastic3D` (`src/Viscoelastic3D.cpp:69-119`):

1. Builds the `GridCoupling`, on a shared `SurfaceGrid` if one is passed,
   otherwise a new one with spins to 2 and oversampling 1.5, and the 1D
   `ViscoelasticOperator` for the reference (radial) rheology.
2. Samples `exp(-delta(r, theta, phi))` at every memory node and grid
   point, and stores per branch `itau_[k]` = reference itau × that
   factor, a full (node × grid) real field per branch, plus `muBranch_[k]`
   per node. This is the only place delta enters.

Nothing is factorised here. The degree operators are built lazily on
first use and cached in `mutable` maps:

- `Spheroidal(l)` / `Toroidal(l)` (`:133`, `:147`): the UNRELAXED
  reference systems, used by `SolveElastic`.
- `FlavourOf(scheme, dt)` (`:163`): a reference EFFECTIVE material,
  mu_inf + Σ_k (1 − beta_ref,k(r)) mu_k, keyed by the exact
  (scheme, dt) pair. Its per-degree operators (`FlavourSpheroidal` /
  `FlavourToroidal`, `:355`, `:370`) and its unit load/tide responses
  (`flavour.units`) are built on first use too.

The elastic moduli stay 1D. Lateral variation enters ONLY through the
relaxation times, so every linear system in the step is a 1D per-degree
banded system. The lateral part is moved to the right-hand side.

## 4. One step, phase by phase

`Step` (`src/Viscoelastic3D.cpp:641`) is `StartStep`, then the
iteration (`OpenStep::Converge`), then `Finish`. `OpenStep` exposes the
same phases so the sea-level loop (§5) can close the step itself.

The within-step iterate is the strain's COEFFICIENTS, s (a
`SpectralTensor`), not its grid values d. The grid strain exists only
inside `Respond` (synthesised from s and consumed there) and at `Finish`.

### 4.1 StartStep (`:396`)

- **The start strain.** From the state's strain cache if present: its
  grid values d0 are used directly and analysed once for the starting
  coefficients s0 (`:418`). Otherwise from `SolveElastic` (`:326`), the
  memory force of Σ_k mu_k m_k through the unrelaxed reference
  operators plus the load channels, whose strain coefficients are s0
  (`:420`) and whose synthesis is d0 (`:424`). No iteration is needed
  there, because the unrelaxed operator is laterally uniform.
- **The weights, per branch and grid point** (`:450`):
  h(x) = dt · itau(x) gives (e, alpha, beta) from `StepWeights.hpp`.
  The reference beta_ref uses the radial itau at that node. Stored per
  branch for `Finish`: `beta_[k]` = beta(x), a (node × grid) real field.
- **The decayed memory** (`:467`): `memory_[k]` = m̃_k = e(x) m_k +
  alpha(x) d0, pointwise, a full `TensorField` per branch.
- **Summed over branches, once per step:** the single lateral weight
  `weight_` = Σ_k mu_k(r) (beta_k(x) − beta_ref,k(r)) (`:471`), and the
  fixed memory force Σ_k mu_k m̃_k, analysed into `fixed_` (`:477`).

### 4.2 One pass of the within-step iteration

The equation being solved is (header of `Viscoelastic3D.hpp`)

    K_ref u = F_load + b( Σ_k mu_k m̃_k  +  W(x) d(u) ),
    W(x) = Σ_k mu_k (beta_k(x) − beta_ref,k(r)),

and a pass maps strain coefficients s to the next ones.

```
 s (coefficients)                                         s' (coefficients)
   │                                                               ▲
   ▼  [3 batched SHTs]                                             │
 Synthesise → d (grid)                                             │
   │                                                               │
   ▼  [pointwise, serial]                                          │
 W · d                                                             │
   │                                                               │
   ▼  [3 batched SHTs]                                             │
 Analyse, + fixed_ ──► SpectralTensor                              │
   │                                                               │
   ▼  per degree l, per node, per m  [threaded over degrees]       │
 Gather 5 comps → force rows → quadrature into b                   │
   │                                           [threaded over l]   │
   ▼  per degree  [threaded over degrees]                          │
 banded solve, 2(l+1) real RHS ──► x ── + load/tide ──► strain of x
                                     channels          (gather U,V,W,
                                                        r∂r; Scatter)
```

**Respond** (`:481`):

1. *Synthesise* (`:488` → `src/GridCoupling.cpp:72`). Three batched
   inverse transforms (spins −2, −1, 0), each with ALL memory nodes as
   one contiguous batch, into a freshly allocated grid `TensorField`.
2. *Pointwise weight* (`:489`). `GridCoupling::Multiply` by `weight_`,
   in place.
3. *Analyse* (`:490` → `src/GridCoupling.cpp:83`). Three batched forward
   transforms, then `fixed_` is added.
4. *Force assembly* (`SpheroidalForce`, `src/GridCoupling.cpp:245`;
   `ToroidalForce`, `:295`). For each degree, each node in the degree's
   depth subset, and each m ≥ 0: `Gather` reads the five components,
   reconstructing spins +1 and +2 by reality (`:176`).
   `ForceRowsFromComponents` gives the rows. They are added into the
   degree's right-hand side at the node's own dofs (the pointwise rows,
   weight jac·w_q·r) and at every node of its element (the derivative
   rows, through D(q, j)).
5. *Solves* (`SolveAll`, `src/Viscoelastic3D.cpp:216`). Per degree, the
   complex columns are split into real pairs and solved as 2(l+1) real
   right-hand sides through the FLAVOUR operator: `dpbtrs` (banded
   Cholesky) or `dgbtrs` (the LU fallback), at `:267`.
   Toroidal runs from l = 1, degree 1 in a pinned gauge (§8, item 1).

The three per-degree loops (force assembly, solves, strain recovery)
run as one OpenMP task per degree, largest degrees first, when
`Options::parallel` is set (`detail::ParallelFor`). Each first looks up
every degree's operator serially, filling the lazy caches, so the tasks
only read shared data and each writes only its own degree's outputs.
The results are therefore bitwise the same at any thread count.

**AddChannels** (`:508`). The load and tide enter by linearity:
amplitude × the cached unit response of the flavour operator (solved
once per degree per flavour, `:527`).

**StrainOf → StrainCoefficients** (`:549` → `:176`). Per degree,
`AddStrainOfSolution` (`src/GridCoupling.cpp:201`) gathers U, V, W at
each node through the operators' dof maps and forms r∂r through the
element's derivative matrix. `DeviatoricComponents` gives the five
components, and `Scatter` writes them into the (l, m) and (l, −m)
slots of a fresh `SpectralTensor`: the next iterate, with no transform.

**Degree truncation.** Each degree acts only on its element subset
(`firstElementOf_`, `src/GridCoupling.cpp:48`; `FirstNodeFor`). Deep
nodes therefore carry zero spectral content at high degrees. The
transforms still run every node at full lmax.

### 4.3 Convergence and mixing (`OpenStep::Converge`, `:596-639`)

One loop serves all three drivers: `Step` here, the sea-level loop (§5)
and the adjoint's reverse sweep (`src/Adjoint3D.cpp`). Each supplies its
own pass (`Respond`, its own load closure, `AddChannels`, `StrainOf`) and
an `IterationControl`.

- Residual: max |s' − s| / max |s'| over every stored coefficient at
  every node (a relative sup-norm on the coefficients, `:618`), compared
  on squared magnitudes. Until 17 October (chunk 3) it was the same norm
  on the grid values; the two agree in what they accept to within the
  tolerance, and the results to round-off (§7).
- Defaults (`Viscoelastic3D.hpp`, `Options`): rtol 1e-11, at most 40
  passes, damping 1, Anderson memory 5. The sea-level loop and the
  adjoint pass their own options, without damping.
- Mixing: `detail::AndersonMixer::UpdateInPlace` (`:633`) works directly
  on s's three coefficient blocks and overwrites s with the mixed
  iterate, with no copies and two sweeps over the data. The mixer keeps
  ring buffers of 5 iterate differences and 5 residual differences,
  plus the previous iterate and residual: 12 spectral-tensor-sized
  vectors, allocated afresh for each step.
- Not converging is not an error here. `StepInfo.converged = false` is
  returned and the step finishes with the last iterate. The adjoint
  sweep throws instead.

### 4.4 Finish (`:575`)

The final coefficients are synthesised to the grid strain d (`:577`),
then m⁺_k = m̃_k + beta_k(x) d, pointwise. d is cached in the new state
as its strain, so the next `StartStep` skips the elastic solve (and
analyses d once for its starting coefficients).

Transforms per step: two batches per pass, plus three per step (the
start coefficients, the fixed memory force, and `Finish`'s synthesis).

### 4.5 What is live during a step (one branch, lmax 64)

| Array | Count | Size |
|---|---|---|
| `itau_` | K | 0.18 GB each |
| State m, strain cache | K + 1 tensors | 0.9 GB each |
| `memory_` | K tensors | 0.9 GB each |
| `beta_`, `weight_` | K + 1 real fields | 0.18 GB each |
| `Respond`'s grid strain d | 1 tensor | 0.9 GB |
| s, s', `fixed_`, s0, `Respond`'s analysed tensor | 5 spectral | 0.2 GB each |
| Anderson buffers | 12 spectral-sized | 0.2 GB each |

That is about 8 GB during the iteration; `Finish` briefly adds the next
state's two grid tensors. The measured peak RSS is 10.9 GB
(`step_timing`, lmax 64, one branch; 22.9 GB at the baseline, 19.6 GB
after chunk 2). The grid tensors of the state and the decayed memory are
now the bulk of it.

## 5. The coupled loop with sea level (`SeaLevelEvolution3D::Run`)

`src/Evolution3D.cpp:63-199`. Per time step (`:158`):

1. `StartStep` under the previous step's total load σ and centrifugal
   potential ψ as channels.
2. Outer passes, through `OpenStep::Converge` (`:167-191`). The pass:
   `Respond(s)` gives the zero-channel solutions. Their surface U and Φ
   per degree and the inertia moment feed a full SLE solve,
   `SolveGeneralised` (`:180`), under the step's effective Love numbers,
   and that solve runs its OWN inner Anderson iteration. The new σ and ψ
   go back in through `AddChannels` (`:185`), and `StrainOf` gives s'.
   Residual and mixing are §4.3's.
3. `Finish` (`:192`).

So strain, ocean load and rotation converge together in one outer fixed
point, and each outer pass contains a complete inner SLE solve.
Recording (`record_stride > 0`) stores the whole `Viscoelastic3D::State`
(K + 1 grid tensors) in each kept snapshot (`:149`). The 3D adjoint
needs every step kept (`ForwardTrajectory3D`, `Adjoint3D.hpp`).

## 6. What pins it down (the gates)

- `SpinMachinery.ReductionToTheDegreeBasisMemoryForce`: the
  component/force-row algebra reduces to the 1D degree-basis force,
  constant by constant.
- `SpinMachinery.RealTensorsHaveConjugateComponentFields`: the
  reality relations the reduced storage rests on.
- `Stage4Gate.ZeroLateralPipelineMatchesTheDegreeBasisStepper`
  (`TestGridCoupling.cpp`) and
  `Stage4Gate.ZeroLateralStepMatchesTheDegreeBasisStepper`
  (`TestViscoelastic3D.cpp`): delta = 0 reproduces the 1D stepper at
  machine precision.
- `Stage4.LateralStepSelfConvergesAtSecondOrder`: dt self-convergence
  with lateral viscosity on.
- `Stage4Gate.ZeroLateralCoupledEvolutionMatchesThe1DLoop` and
  `Stage4.CoupledLateralEvolutionRunsAndConservesMass`
  (`TestEvolution3D.cpp`): the coupled loop.
- External: the Fortran `visco_test` gate (`stage4.md` §5b), degrees
  l ≥ 2 only, at lmax 8.

Any restructuring of §4 (iteration variable, chunking, threading) must
keep these green. The zero-lateral gates are the decisive ones.

## 7. Where the time goes (17 October profile)

`fortran_gate` at lmax 64, one thread, gdb sampling, 285 samples over
90 s (early steps only):

| Share | What |
|---|---|
| ~30% | `AndersonMixer::Update` |
| ~19% | `Step`'s own loops: flatten/unflatten, `hypot` in the residual, freeing tensors |
| ~13% | `ZeroTensor` memsets (fresh tensors every pass) |
| ~15% | the transforms |
| ~9% | force assembly + banded solves + strain recovery |

**Baseline phase timings** (`benchmarks/step_timing`, 17 Oct 2026,
develop at 88c58b5, laptop: AMD Ryzen AI 7 PRO 350, one socket; lmax 64,
1225 memory nodes, grid 97 × 195, 7 passes per step). Seconds per pass:

| Phase | 1 thread | 8 threads |
|---|---|---|
| analyse (3 batched SHTs) | 0.67 | 0.26 |
| synthesise (3 batched SHTs) | 0.76 | 0.36 |
| force assembly | 0.13 | 0.12 |
| banded solves | 0.21 | 0.21 |
| strain recovery | 0.12 | 0.11 |
| pointwise (in `Respond`) | 0.29 | 0.32 |
| iteration overhead (residual, copies, mixing) | 2.48 | 2.35 |
| **whole pass** | **4.64** | **3.72** |

Peak RSS 22.9 GB in both. Eight threads speed up only the transforms
(2.3×), so the pass speeds up 1.25×, and the serial overhead grows from
53% to 63% of it.

After chunk 2 (one shared loop, the mixer in place): the overhead is
1.14 s (1 thread) and 1.12 s (8 threads), the whole pass 3.30 s and
2.52 s (1.41× and 1.48× faster), and the peak RSS 19.6 GB. The final
state is bitwise unchanged.

After chunk 3 (the iterate in coefficients, the fixed force once per
step): the whole pass is 2.60 s (1 thread) and 1.66 s (8 threads), 1.8×
and 2.2× faster than the baseline, and the peak RSS 10.9 GB. Per pass
at 8 threads: transforms 0.63 s, the serial degree loop (force + solves
+ strain) 0.44 s, pointwise 0.10 s, the rest 0.50 s (of which about
0.15 s is the three per-step transforms, spread over 7 passes). Results
agree with chunk 2 to round-off (surface sums to 2e-16 at lmax 16 and
32; the Fortran gate table unchanged), the zero-lateral run is
unchanged, and the coupled sea-level loop converges in one pass fewer
per step on the lateral example.

After chunk 4 (the per-degree loops threaded): one thread is unchanged
(2.62 s per pass); on 8 threads the pass is 1.37 s, 2.7× faster than the
baseline. The degree loop on 8 threads: force 0.029 s (4.5×), solves
0.079 s (3.4×), strain recovery 0.052 s (2.3×). What remains of an
8-thread pass: transforms 0.63 s, serial overhead 0.52 s (mixing,
residual, the per-step pointwise loops and transforms), degree loop
0.16 s, pointwise 0.04 s. Results are bitwise unchanged: with the
transforms serial and the degree loops on 8 threads, every run
reproduces the one-thread state.

After chunk 7 (internal variables at the Gauss points, `step_timing ...
loop gauss`; 17 Oct 2026, 8 threads, lmax 64): 980 memory nodes for
1225, one TensorField 0.74 GB for 0.93; per pass 1.14 s for 1.34 (analyse
0.21 for 0.24, synthesise 0.31 for 0.36, solves 0.070 for 0.089, strain
recovery 0.044 for 0.053, overhead 0.42 for 0.51; the force assembly
unchanged at 0.028, its point rows now spread over the element's
nodes); peak RSS 9.55 GB for 11.69. Surface sums agree with the GLL run
to 8e-13 (U) and 3e-11 (phi); since the shear split, which changes the
discrete elastic solid at Gauss points at the quadrature level, to 2e-10
(U) and 5e-11 (phi), at the same cost. Gauss points are now the default;
with GLL points selected every result is bitwise as before chunk 7.

Re-run with `OMP_NUM_THREADS=N build/benchmarks/step_timing <deck>
<lmax> [steps] [repeats] [loop|matrix] [gauss|gll]` after each change
(Gauss points by default).

## 8. Things to check against your intent

Ordered roughly by how much they matter. "Discrepancy" means the code
differs from the Fortran or from the docs; "cost" means correct but
expensive.

1. **Fixed (17 Oct, chunk 5): degree-1 toroidal was omitted.** The
   toroidal force, solves and strain recovery started at l = 2, so the
   degree-1 toroidal response was never assembled or solved. Only rigid
   rotation (W ∝ r) is strain-free: differential degree-1 toroidal
   motion carries strain through the twist term W − r∂rW in A(−0) and
   A(0+) (`DeviatoricComponents`), and lateral viscosity drives it (the
   omission followed an expectation in `stage4.md` §4, since corrected).
   Now `ToroidalOperator` accepts l = 1 and fixes the gauge by pinning
   one dof per solid region (the region's top node by default;
   `Options::pin_fraction`), and the three loops run from l = 1. Tests:
   the rigid rotations are the null space, a torque-free right-hand side
   is solved in full (pinned rows included), the memory force is
   torque-free, and moving the pin changes only the degree-1 rigid
   rotation, nothing well defined. Effect on the lateral example: 0.25%
   of the lateral-minus-radial sea-level difference. (The Fortran solves
   this degree with an LU on the unconstrained singular form, so it
   fixes no frame; its gate pattern does not excite the mode, by
   symmetry.)
2. **Fixed (17 Oct, chunk 2): the iteration was described wrongly.**
   The `Viscoelastic3D.hpp` file comment said "damped Picard here …
   Anderson mixing is the recorded fallback", while the code defaults
   to Anderson memory 5. The comment now matches the code.
3. **Behaviour: threading.** `Options::parallel` defaults to true
   (since 17 Oct; it used to default to false, so nothing was threaded
   unless a caller asked). It threads the transforms and (since chunk 4)
   the per-degree loops, on `OMP_NUM_THREADS` threads; the pointwise
   loops and the mixing stay serial. The examples therefore run threaded,
   and the test harness pins each test to one thread.
4. **Fixed (17 Oct, chunk 3): the iterate was the full grid tensor.**
   Mixing and the residual now run on the strain's coefficients (~4.5×
   fewer values), with the grid strain formed only inside `Respond` and
   at `Finish`.
5. **Cost: allocation every pass.** `Respond`'s grid strain and
   analysed tensor and `StrainOf`'s coefficients are fresh allocations
   per pass, and the mixer's buffers are allocated afresh per step. (The
   flatten/unflatten copies and the mixer's per-pass temporaries went in
   chunk 2.)
6. **Fixed (17 Oct, chunk 3): the fixed part was recombined every
   pass.** Σ_k mu_k m̃_k is now analysed once per step (`fixed_`), and a
   pass multiplies d by the single weight field W(x).
7. **Thread safety: the lazy caches.** `Spheroidal`, `Toroidal`,
   `FlavourOf`, `FlavourSpheroidal`/`FlavourToroidal` and
   `flavour.units` insert into `mutable` maps from `const` methods.
   `conventions.md` says const solve/apply methods must be safe in
   parallel loops; these are not. The threaded degree loops (chunk 4)
   therefore look every operator up serially first, but calling these
   methods concurrently from outside is still unsafe.
8. **Fixed (17 Oct, chunk 2): duplicated loop.** The residual and
   mixing loop existed three times (`Step`, `SeaLevelEvolution3D::Run`
   and the adjoint's reverse sweep). All three now call
   `OpenStep::Converge`.
9. **Cost: unit responses re-solved.** `SolveAll`'s channel path
   (`src/Viscoelastic3D.cpp:299`, used by `SolveElastic` and so
   `PrimeStrain` and an uncached `StartStep`) re-solves the two unit
   responses on every call. `AddChannels` caches them.
10. **Cost: weight fields recomputed every step.** `StartStep` evaluates
    `exp` per (branch, node, grid point) each step, even at constant dt.
    They depend only on (dt, itau).
11. **Memory: flavours keyed by exact dt.** `flavours_` is keyed by the
    (scheme, dt) double and never evicted. A variable-dt run factorises
    and keeps a full set of per-degree operators per distinct dt.
12. **Memory: trajectories store grid states.** Each kept snapshot holds
    K + 1 grid tensors (~1.8 GB at lmax 64, one branch), and the 3D
    adjoint needs every step. A 100-step run at lmax 64 is ~180 GB.
13. **Cost: force-assembly access pattern.** For each degree, the
    force loop walks node by node through the node-major spectral
    layout, jumping one coefficient block per node, and the strain
    recovery scatters back the same way. GSHTrans's `Layered/RadialMajor.hpp`
    provides the transposed layout meant for per-(l, m) radial work.
14. **Choice: the stopping norm.** A relative sup-norm over all grid
    values is dominated by the largest strains, usually near the
    surface. It is a reasonable choice, but it is a choice.
15. **Choice (recorded): full lmax at every node.** Depth-variable lmax
    was considered and declined (`stage4.md` §8); deep nodes run full
    transforms for content that is truncated to zero there.
16. **Behaviour (accepted): threaded runs are not bitwise
    reproducible.** With more than one thread, the step's results vary
    from run to run at round-off. The cause (found 17 Oct, chunk 4) is
    FFTW planning: `SurfaceGrid` builds its GSHTrans grid with
    `FFTWpp::Measure`, each thread makes its own FFTW plan, and
    `FFTW_MEASURE` chooses algorithms by timing, which differs between
    threads planning at once. With `FFTWpp::Estimate`, eight-thread runs
    were identical in a test. One-thread runs are reproducible.
    Agreement to round-off is all that is required (David, 17 Oct), so
    nothing is to change; bitwise regression checks use one thread.
17. **Cost (negligible, deferred): unforced solid regions.** A solid
    region separated by fluid and carrying no active rheology branch
    (the elastic inner core, usually) has no memory force, so its
    toroidal response is exactly zero: its block is solved for nothing
    (and pinned at l = 1). Dropping such regions from the toroidal dof
    map would be exact, but the graded truncation already excludes the
    inner core above l ≈ 10, so the saving is about 1e-4 of a pass
    (David, 17 Oct). A viscous inner core with lateral variation could
    drive its own toroidal motion, so the rule would be "no active
    branch", not "inner core".
18. **Fixed (17 Oct, chunk 6): stored states carried the strain.** A
    recorded state held the strain beside the internal variables, half of
    every checkpoint with one branch. As in pyslfp and AdGIA, recordings
    now keep only the internal variables, and the adjoints regenerate a
    node's strain with one elastic solve under its recorded load and
    centrifugal coefficients (`Viscoelastic3D::ElasticStrain`; the 1D
    `TrajectoryCursor` primes each stored state it serves). The unrelaxed
    operator is laterally uniform, so no iteration, and the end-of-step
    strain satisfies unrelaxed equilibrium with the new memory (David,
    17 Oct). On the 1D kernel example (lmax 128): peak memory 32.9 →
    20.2 GB; viscosity kernels unchanged to 1e-9 (the forward coupled
    tolerance); time 63.5 → 87.6 s, one serial 1D elastic solve per node
    per sweep. Threading the 1D operator's degree loops (the same day,
    `ViscoelasticOperator::Options::parallel`) brought it to 48.3 s on 8
    threads, below the original, and the 1D forward example from 55.1 to
    24.8 s; results on one thread are unchanged. In 3D a regeneration
    costs about one pass, against ~7 per reverse step.
    `Options::record_strain` (both evolutions, default false) keeps the
    strain as well, trading memory back for the regeneration solves;
    tests show the kernels agree either way.
19. **Cost (planned): Gauss-point memory nodes** (`plan.md` §5.5). The
    memory lives at the GLL nodes, doubled at shared element boundaries;
    the N = ngll − 1 interior Gauss points match the GLL rule's exactness
    for the memory-force integral with ~20% fewer points (and ~20% fewer
    radial nodes in the transforms). A switchable option, GLL staying the
    default for the pyslfp gates; τ, the step weights, their
    h-derivatives and the kernel accumulators move together. In
    progress (David, 17 Oct): 7a, done, makes the effective operators
    take an explicit shear term at the memory points (needed for an
    exact elimination at Gauss points; `MemoryPoints.hpp`); 7b-i, done,
    moves the 1D forward (`ViscoelasticOperator::Options::memory_points`):
    80% of the memory, an exact elimination still, and surface responses
    within 1.4e-8 of GLL at 600 km elements, 2.4e-10 at 300 km; 7b-ii,
    done, moves the 1D adjoints (kernels accumulate at the points and
    reach the nodal itau through the interpolation's transpose; nodal
    kernels within 1e-6 of GLL's); 7c, done, moves the 3D
    (`Viscoelastic3D::Options::memory_points`; at lmax 64 on 8 threads
    20% fewer memory nodes, a 14% faster pass, 18% less peak memory).
    Gauss points are now the default, 1D and 3D; GLL stays selectable
    for like-for-like comparisons with the Fortran and pyslfp. Switching
    exposed an instability of the first form at long steps (h = dt/tau of
    a few hundred and up), fixed by splitting the shear energy: the
    long-term part at the nodes, each branch's at the points, so every
    term of the effective operator stays positive (plan.md §5.5).
20. **Cost (small tidy-up): the 1D internal variables span each
    degree's whole subset**, from its first element to the surface, so
    they are stored (as zeros) in the lithosphere and, at low degrees,
    the core: ~10–12% of the 1D state overall, up to ~25–29% at low
    degrees (PREM Maxwell body, lmax 64 and 128). The 3D path already
    keeps only active elements (`GridCoupling` memory nodes: 1225 of
    1725 mesh nodes at lmax 64).
21. **Deferred: per-branch active sets.** With several branches, every
    branch stores memory over the union of active elements; only worth
    restricting once multi-branch rheologies are in use (David, 17 Oct).
