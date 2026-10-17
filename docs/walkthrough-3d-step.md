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
(`src/GridCoupling.cpp:27-38`). The elastic lithosphere and the fluid
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

`Viscoelastic3D::Viscoelastic3D` (`src/Viscoelastic3D.cpp:56-106`):

1. Builds the `GridCoupling`, on a shared `SurfaceGrid` if one is passed,
   otherwise a new one with spins to 2 and oversampling 1.5, and the 1D
   `ViscoelasticOperator` for the reference (radial) rheology.
2. Samples `exp(-delta(r, theta, phi))` at every memory node and grid
   point, and stores per branch `itau_[k]` = reference itau × that
   factor, a full (node × grid) real field per branch, plus `muBranch_[k]`
   per node. This is the only place delta enters.

Nothing is factorised here. The degree operators are built lazily on
first use and cached in `mutable` maps:

- `Spheroidal(l)` / `Toroidal(l)` (`:120`, `:134`): the UNRELAXED
  reference systems, used by `SolveElastic`.
- `FlavourOf(scheme, dt)` (`:148`): a reference EFFECTIVE material,
  mu_inf + Σ_k (1 − beta_ref,k(r)) mu_k, keyed by the exact
  (scheme, dt) pair. Its per-degree operators (`FlavourSpheroidal` /
  `FlavourToroidal`, `:292`, `:307`) and its unit load/tide responses
  (`flavour.units`) are built on first use too.

The elastic moduli stay 1D. Lateral variation enters ONLY through the
relaxation times, so every linear system in the step is a 1D per-degree
banded system. The lateral part is moved to the right-hand side.

## 4. One step, phase by phase

`Step` (`src/Viscoelastic3D.cpp:574`) is `StartStep`, then the
iteration (`OpenStep::Converge`), then `Finish`. `OpenStep` exposes the
same phases so the sea-level loop (§5) can close the step itself.

### 4.1 StartStep (`:331`)

- **The start strain d0.** Taken from the state's strain cache if
  present; otherwise from `SolveElastic` (`:263`): the memory force of
  Σ_k mu_k m_k through the unrelaxed reference operators, plus the load
  channels. No iteration is needed there, because the unrelaxed operator
  is laterally uniform.
- **The weight fields, per branch and grid point** (`:374`):
  h(x) = dt · itau(x) gives (e, alpha, beta) from `StepWeights.hpp`.
  The reference beta_ref uses the radial itau at that node. Stored:
  `beta_[k]` = beta(x) and `dbeta_[k]` = beta(x) − beta_ref(r), both
  (node × grid) real fields.
- **The decayed memory** (`:391`): `memory_[k]` = m̃_k = e(x) m_k +
  alpha(x) d0, pointwise, a full `TensorField` per branch.

### 4.2 One pass of the within-step iteration

The equation being solved is (header of `Viscoelastic3D.hpp`)

    K_ref u = F_load + b( Σ_k mu_k ( m̃_k + dbeta_k(x) d(u) ) ),

and a pass maps a strain field d to the next one.

```
 d (grid)                                                       d' (grid)
   │                                                               ▲
   ▼  [pointwise, serial]                         [3 batched SHTs] │
 combined = Σ_k mu_k (m̃_k + dbeta_k · d)          Synthesise        │
   │                                                               │
   ▼  [3 batched SHTs]                                             │
 Analyse ──► SpectralTensor                     SpectralTensor ◄───┤
   │                                                  ▲            │
   ▼  per degree l, per node, per m  [serial]         │ [serial]   │
 Gather 5 comps → force rows → quadrature into b      │            │
   │                                                  │            │
   ▼  per degree  [serial]                            │            │
 banded solve, 2(l+1) real RHS ──► x ── + load/tide ──► strain of x
                                     channels          (gather U,V,W,
                                                        r∂r; Scatter)
```

**Respond** (`:401`):

1. *Pointwise combine* (`:416`). `combined = Σ_k mu_k (m̃_k + dbeta_k d)`
   over every node and grid point, into a freshly allocated
   `TensorField`.
2. *Analyse* (`:426` → `src/GridCoupling.cpp:80`). Three batched
   forward transforms (spins −2, −1, 0), each with ALL memory nodes as
   one contiguous batch.
3. *Force assembly* (`SpheroidalForce`, `src/GridCoupling.cpp:242`;
   `ToroidalForce`, `:285`). For each degree, each node in the degree's
   depth subset, and each m ≥ 0: `Gather` reads the five components,
   reconstructing spins +1 and +2 by reality (`:173`).
   `ForceRowsFromComponents` gives the rows. They are added into the
   degree's right-hand side at the node's own dofs (the pointwise rows,
   weight jac·w_q·r) and at every node of its element (the derivative
   rows, through D(q, j)).
4. *Solves* (`SolveAll`, `src/Viscoelastic3D.cpp:178`). Per degree, the
   complex columns are split into real pairs and solved as 2(l+1) real
   right-hand sides through the FLAVOUR operator: `dpbtrs` (banded
   Cholesky) or `dgbtrs` (the LU fallback), at `:201` and `:249`.
   Toroidal runs from l = 2 (see §8, item 1).

**AddChannels** (`:437`). The load and tide enter by linearity:
amplitude × the cached unit response of the flavour operator (solved
once per degree per flavour, `:456`).

**StrainOf → StrainField** (`:478` → `:161`). Per degree,
`AddStrainOfSolution` (`src/GridCoupling.cpp:198`) gathers U, V, W at
each node through the operators' dof maps and forms r∂r through the
element's derivative matrix. `DeviatoricComponents` gives the five
components, and `Scatter` writes them into the (l, m) and (l, −m)
slots. Then `Synthesise` runs three batched inverse transforms back to
a fresh grid `TensorField`.

**Degree truncation.** Each degree acts only on its element subset
(`firstElementOf_`, `src/GridCoupling.cpp:45`; `FirstNodeFor`). Deep
nodes therefore carry zero spectral content at high degrees. The
transforms still run every node at full lmax.

### 4.3 Convergence and mixing (`OpenStep::Converge`, `:524-572`)

One loop serves all three drivers: `Step` here, the sea-level loop (§5)
and the adjoint's reverse sweep (`src/Adjoint3D.cpp`). Each supplies its
own pass (`Respond`, its own load closure, `AddChannels`, `StrainOf`) and
an `IterationControl`.

- Residual: max |d' − d| / max |d'| over every stored grid value at
  every node (a relative sup-norm, `:548`), compared on squared
  magnitudes.
- Defaults (`Viscoelastic3D.hpp`, `Options`): rtol 1e-11, at most 40
  passes, damping 1, Anderson memory 5. The sea-level loop and the
  adjoint pass their own options, without damping.
- Mixing: `detail::AndersonMixer::UpdateInPlace` (`:566`) works directly
  on d's three component arrays and overwrites d with the mixed iterate,
  with no copies and two sweeps over the data. The mixer keeps ring
  buffers of 5 iterate differences and 5 residual differences, plus the
  previous iterate and residual: 12 grid-tensor-sized vectors, allocated
  afresh for each step.
- Not converging is not an error here. `StepInfo.converged = false` is
  returned and the step finishes with the last iterate. The adjoint
  sweep throws instead.

### 4.4 Finish (`:504`)

m⁺_k = m̃_k + beta_k(x) d, pointwise. The final d is cached in the new
state as its strain, so the next `StartStep` skips the elastic solve.

### 4.5 What is live during a step (one branch, lmax 64)

| Array | Count | Size |
|---|---|---|
| `itau_` | K | 0.18 GB each |
| State m, strain cache | K + 1 tensors | 0.9 GB each |
| `d0_`, `memory_` | 1 + K tensors | 0.9 GB each |
| `beta_`, `dbeta_` | 2K real fields | 0.18 GB each |
| d, d', Respond's `combined` | 3 tensors | 0.9 GB each |
| Anderson buffers | 12 tensor-sized | 0.9 GB each |

That is about 18 GB, against a measured peak RSS of 19.6 GB
(`step_timing`, lmax 64, one branch; 22.9 GB before the flatten copies
and the mixer's temporaries went). The mixer's buffers are about
two-thirds of it.

## 5. The coupled loop with sea level (`SeaLevelEvolution3D::Run`)

`src/Evolution3D.cpp:63-199`. Per time step (`:158`):

1. `StartStep` under the previous step's total load σ and centrifugal
   potential ψ as channels.
2. Outer passes, through `OpenStep::Converge` (`:167-191`). The pass:
   `Respond(d)` gives the zero-channel solutions. Their surface U and Φ
   per degree and the inertia moment feed a full SLE solve,
   `SolveGeneralised` (`:180`), under the step's effective Love numbers,
   and that solve runs its OWN inner Anderson iteration. The new σ and ψ
   go back in through `AddChannels` (`:185`), and `StrainOf` gives d'.
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
state is bitwise unchanged. Re-run with
`OMP_NUM_THREADS=N build/benchmarks/step_timing <deck> <lmax>` after
each change.

## 8. Things to check against your intent

Ordered roughly by how much they matter. "Discrepancy" means the code
differs from the Fortran or from the docs; "cost" means correct but
expensive.

1. **Discrepancy: degree-1 toroidal is omitted.** *Decided 17 Oct 2026:
   it must be included, with its rigid-rotation null space removed by
   convention (`plan.md` §4, §8.9, §9).* `ToroidalForce` starts
   at l = 2 (`src/GridCoupling.cpp:292`), so the degree-1 toroidal force
   is never assembled or solved. Only rigid rotation (W ∝ r) is strain-free:
   differential degree-1 toroidal motion carries strain through the
   twist term W − r∂rW in A(−0) and A(0+) (`SpinComponents.hpp`,
   `DeviatoricComponents`), and lateral viscosity can excite it. The
   Fortran solves it with a banded LU on `aa_tor_1`
   (`sl3d/src/module_gia.f90:503-506`, `module_mat.f90:58-62`), but on
   the unconstrained singular form, so it fixes no frame. The omission
   follows an expectation recorded in `stage4.md` §4, since corrected.
   The Fortran gate compares l ≥ 2 only, and this enters other degrees
   only through the lateral coupling, so no gate would have caught it.
2. **Fixed (17 Oct, chunk 2): the iteration was described wrongly.**
   The `Viscoelastic3D.hpp` file comment said "damped Picard here …
   Anderson mixing is the recorded fallback", while the code defaults
   to Anderson memory 5. The comment now matches the code.
3. **Behaviour: nothing in the 3D step is threaded unless the caller
   asks.** `Options::parallel` defaults to false, and with it the
   transforms run sequentially. Only `benchmarks/fortran_gate` sets it;
   no example or test does.
   Every other loop in the step is serial regardless.
4. **Cost: the iterate is the full grid tensor.** Mixing and the
   residual run on (node × oversampled grid) values. Since d' is always
   `Synthesise` of a spectral tensor and mixing is linear, the same
   fixed point can be iterated on the spectral strain (~4.5× smaller),
   with the grid strain formed only inside a pass and at `Finish`.
5. **Cost: allocation every pass.** `Respond` (`combined`),
   `Synthesise` (output) and `Analyse` (`ZeroSpectral`) are fresh
   allocations of tensor size per pass, and the mixer's buffers are
   allocated afresh per step. (The flatten/unflatten copies and the
   mixer's per-pass temporaries went in chunk 2.)
6. **Cost: the fixed part is recombined every pass.** `Respond` re-forms
   and re-analyses Σ_k mu_k m̃_k each pass, though it is constant within
   the step. By linearity it could be analysed once, leaving one weight
   field Σ_k mu_k dbeta_k(x) times d per pass.
7. **Thread safety: the lazy caches.** `Spheroidal`, `Toroidal`,
   `FlavourOf`, `FlavourSpheroidal`/`FlavourToroidal` and
   `flavour.units` insert into `mutable` maps from `const` methods.
   `conventions.md` says const solve/apply methods must be safe in
   parallel loops; these are not. Any degree-parallel loop must build
   them first.
8. **Fixed (17 Oct, chunk 2): duplicated loop.** The residual and
   mixing loop existed three times (`Step`, `SeaLevelEvolution3D::Run`
   and the adjoint's reverse sweep). All three now call
   `OpenStep::Converge`.
9. **Cost: unit responses re-solved.** `SolveAll`'s channel path
   (`src/Viscoelastic3D.cpp:217`, used by `SolveElastic` and so
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
