# sl3dpp: plan for a C++ port of the sl3d GIA codes

*Drafted 15 October 2026, from a survey of sl3d (Fortran), AdGIA, pyslfp,
planetmodel and the GSHTrans/GaussQuad/Interpolation/FFTWpp/NumericConcepts
stack. This is the planning record for the new code; it should be updated as
decisions are made.*

## 1. Purpose and scope

A new C++ package for forward and adjoint GIA modelling in 1D elastic and 3D
viscoelastic earth models, functionally equivalent to the old Fortran codes in
`~/dev/sl3d` (gia1d and sl3d) but **not** a feature-level clone. It adopts the
refined formulations now used in AdGIA and pyslfp:

- implicit (exponential trapezoid) time stepping of the internal variables, not
  the old explicit rate formulation;
- rotational feedback following pyslfp precisely (absent from the old sl3d
  forward problem);
- the generalised sea level equation with Anderson acceleration, proper mass
  conservation and both frozen and migrating shorelines;
- discrete adjoints with proper checkpointing.

The primary motivations are benchmarking against AdGIA (the main project) and
keeping the spectral methods alive as an independent, fast tool. Target:
shared-memory OpenMP, no MPI (see §5.4).

## 2. What the source codes provide

### 2.1 sl3d / gia1d (Fortran) — the algorithmic reference

- Radial discretisation: 1D GLL spectral elements (ngll = 5), per-degree banded
  systems. Spheroidal block (φ, U, V interleaved; φ only in fluid regions, with
  fluid–solid interface terms), toroidal block (W, solid sections only).
  Exterior/interior Poisson via Robin conditions; degree-dependent bottom
  truncation `rl = R(1 − 20/(l+0.5))`. Factorised once per degree, reused for
  the whole run; 2l+1 orders solved as multiple right-hand sides.
- Time stepping (the part being replaced): explicit midpoint RK2 on the rate
  equations, with memory variables `m` (5 canonical deviatoric components) held
  **on the spatial grid** at every GLL node and `ṁ = s(θ,φ,r)(d − m)`
  evaluated pseudospectrally (generalised SH, spin N = −2..2). Viscosity never
  enters the matrices, only the RHS — that is how the old code dodges lateral
  coupling in the operator, at the price of dt capped by the minimum Maxwell
  time (2.5–50 yr step tables).
- Sea level: rate form; ocean function from the flotation test
  `C = H(ρ_w·SL − ρ_i·I)` recomputed every stage (migrating shorelines);
  grounded-ice loading `B = (1−C)İ`; ocean-load fixed point per stage; uniform
  mass-conservation shift. No rotation.
- Adjoints: self-adjoint backward run with frozen forward ocean function,
  static jump sources at observation times, viscosity kernel
  `∫ μ s (d−m):(d†−m†) dt` and ice kernels; full forward history in RAM.
- Known defects not to copy: banded Cholesky on the gravitating operator (use
  LDLᵀ/LU); convergence tested on a single coefficient; integer-division bug in
  snapshot interpolation (`adj_kern_visc_full`); hard-coded 21 ka adjoint
  timing (wrong for ICE-6G/7G); debug writes in the RK loop; no dealiasing;
  complex storage of real fields; dead modules (`module_sphexp*`,
  `module_kern_stress`); memory allocated over non-viscous elements.

### 2.2 pyslfp — the formulation reference (most direct template)

pyslfp has already translated gia1d's physics into the modern formulation; the
authoritative statements are `pyslfp/docs/theory/pyslfp_theory.tex` and the
planning notes `gia_notes.md` / `gia_adjoint_plan.md`. Key pieces to port:

- **Step algebra** (`viscoelastic/weights.py`, itself a port of AdGIA's
  `ExponentialTrapezoidWeights`): with h = Δt/τ, e = e^{−h},
  φ₁ = (1−e)/h (series for small h), α = φ₁ − e, β = 1 − φ₁,
  the exponential-trapezoid update is `m⁺ = e·m + α·dⁿ + β·dⁿ⁺¹`, giving an
  **effective modulus** μ_eff = μ∞ + Σ_k φ₁(h_k) μ_k and memory state
  m̃ = e·m + α·dⁿ. Backward Euler and ETD1 variants as alternatives/error
  estimators. Second order, no stability limit, one solve per step.
- **Two-phase step** (`viscoelastic/operator.py`): memory solve
  K_eff⁻¹ b_mem(m̃) per degree; coupled SLE solve with per-Δt effective
  responses, with the memory surface field entering as response channels;
  finish via cached rank-one unit load/tide solutions; then dⁿ⁺¹ and m⁺.
  Factorisations cached per (scheme, Δt). Unrelaxed elastic solve at t₀ (needed
  for second order). Adaptive stepping via embedded ETD1.
- **Generalised SLE** (`core/sea_level.py`): fixed point on the load with
  Anderson mixing (Walker–Ni, memory 5); extra channels for displacement /
  potential loads and ready responses, inertia-moment and angular-momentum
  changes, and an ocean-function override for linearising about a C(t)
  trajectory; exact quadrature surface integrals; Kendall-style nonlinear solve
  for migrating shorelines.
- **Rotation** (`_rotation`; theory §7): first-order quasi-static Liouville,
  no Chandler wobble; transverse (TPW) and axial (LOD) solved in closed form
  inside each SLE iteration; specified (observed) equilibrium moments; the five
  axial numbers (h_c, k_c, m_u, m_φ, m_c); the fluid-tidal-number stability
  guard (`with_hydrostatic_moments`, δ = 0.008). **Port verbatim.**
- **Discrete adjoint** (`viscoelastic/evolution.py::adjoint_ice_history`):
  reverse sweep with (m†, d†); strain transpose; engine adjoint SLE
  (`adjoint_generalised_solve`, the exact reverse of the forward path, dualising
  the full degree-2 objects); one symmetric effective solve per step. Gated by
  dot-product tests at machine precision. The continuous adjoint exists as a
  resolution diagnostic (discrete ~1e−13 vs continuous %-level when
  under-resolved).
- **Checkpointing**: `record=M` stores internal state every M nodes (σ, ψ
  coefficients always); gaps rebuilt by the **exact algebraic inverse step**
  (α(−h) = −β(h)/e(h) for the trapezoid; m = (1+h)m′ − h d′ for BE); round-off
  grows like exp(Σh) across a gap.
- **Radial solver** (`love_numbers/`): GLL weak form (A&T 2014 eq. D76 in
  X = r∂U, Y = 2U − k²V, S = U + r∂V − V), transversely isotropic moduli,
  fluid stratification terms, DtN at surface and sub-mesh base, graded
  per-degree sub-mesh truncated at `a(1 − ε^{1/(l+1)})`, proper l = 0, 1
  handling (centre-of-mass frame at l = 1), banded Cholesky with LU fallback,
  reciprocity checks. `DegreeSystem` in `love_numbers/assembly.py` is the
  direct template for the C++ per-degree operator.

### 2.3 AdGIA — the design reference

- Layering: the problem knows only how to do an elastic solve; the rheology
  owns material data; the viscoelastic operator owns the internal variables and
  drives the problem through a small interface. Gravity, ocean and rotation sit
  inside the problem (implicit at tⁿ⁺¹), so the viscoelastic layer never sees a
  potential. Mirror this.
- State is **m only**; displacement is slaved through one elastic solve, with a
  cache keyed on (m, t).
- `ForwardTrajectory` stores m per step in memory; adjoint is the zero-load
  twin problem run in reversed time with the same stepper; `KernelAccumulator`
  consumes (forward, adjoint) pairs during the sweep so the adjoint trajectory
  is never stored.
- Conventions worth copying: namespace + `detail`; PascalCase methods /
  trailing-underscore members (MFEM style); nested `Options` structs with
  defaults; `std::function` hooks (force hook, step corrector, time sync);
  GoogleTest with analytic, reciprocity and finite-difference checks; heavy
  Doxygen citing equations; non-dimensional units.
- AdGIA is MPI/MFEM with no OpenMP; sl3dpp is OpenMP with no MPI — the codes
  stay complementary and their discretisations independent, which is a feature
  for benchmarking.

### 2.4 The toolkit

- **GSHTrans** (C++23, header-only): generalised SH transforms at any spin N on
  a Gauss–Legendre grid, with **exactly the old Fortran's normalisation**
  (Y^N_lm = √((2l+1)/4π) d^l_{Nm} e^{imφ}, Condon–Shortley, forward =
  ∫ conj(Y) f dΩ). Spin-weighted / tensor / layered field algebra,
  `RadialMajor` coefficient layout, `ForBand(lBand, nMax, oversampling)` for
  dealiased products (1.5 = 3/2 rule), batched transforms, per-call
  `Execution::Parallel`, optional BLAS matrix kernel (~6× batched forward).
  Degree ceiling ~1827 in double (ample). Memory: Wigner table 325–648 MB at
  lmax 256, 2.6–5.2 GB at lmax 512 (matrix vs loop kernel).
  Caveats: batches cannot mix N (loop over spin components); real fields only
  at N = 0 (tensor components handled as complex with Hermitian symmetry — as
  the Fortran did); radial operators there are conveniences, not a
  discretisation.
- **GaussQuad**: GLL/Gauss–Jacobi rules and Legendre polynomial utilities —
  the radial element machinery.
- **Interpolation**: cubic/Akima splines with `Piecewise`/`SplitAtRepeats`
  (layered models with repeated-radius discontinuities map directly), complex
  ordinates (spline spectral lines), thread-safe `CubicSplineSystem`.
- **FFTWpp / NumericConcepts**: reached only through GSHTrans.
- Version hygiene: GaussQuad and Interpolation pin NumericConcepts to a commit
  while GSHTrans/FFTWpp track `main` — **sl3dpp must pin all five tags** (and
  can override via `FETCHCONTENT_SOURCE_DIR_<NAME>` for local development).

### 2.5 planetmodel — model formats and generation (division of labour)

planetmodel (Python, numpy/scipy core; used directly by pyslfp, and by AdGIA's
meshing/benchmark tooling) owns **model definition, generation and export**;
sl3dpp has only **lightweight readers**. What exists today:

- `Model` = `Geometry` (skeleton radii, named layers/interfaces, mapping) plus
  named `Field`s per layer; `PREM` as exact polynomials; `Scales` units with
  `nondimensionalised()`/`geophysical()`; `RadialMesh` (GLL, ngll = 5);
  rheology read from fields (a `viscosity` field ⇒ Maxwell in shear,
  τ = η/μ; `frozen()` for complex moduli). Prony branches live in pyslfp
  (`Rheology.from_model`), not planetmodel.
- **1D file format: the deck** (`deck.py`) — whitespace text, radius first,
  one column per field, repeated radius = discontinuity, NaN = absent field;
  MINEOS header variant; custom `DeckFormat`s, including a GIA format with a
  viscosity column (tutorial 11). This is the stable 1D interchange format and
  what sl3dpp's 1D reader targets.
- **3D**: in-memory only (`AnalyticField`, gridded `Sample` with layout
  (radial node, colatitude, longitude), GL or equiangular grids, poles
  excluded; real orthonormal SH, no CS phase — the pyshtools convention).
- **netCDF model file: designed but not built.** planetmodel's (gitignored)
  `docs/stage3.md` §3 records the agreed design (netCDF4/CF-styled, real
  fields, skeleton + layer names + scales + angular grid + one variable per
  field in `Sample` node layout, read-back path to a `Model`), but variable and
  attribute names are not yet frozen and no writer exists.
- **No tomography ingestion and no viscosity-scaling chain.** S20RTS appears
  nowhere; the old Fortran's `visc_3d_from_vs` (S20RTS → dlnvs via Woodhouse
  radial splines → dlnρ → dT via a thermal-expansion polynomial →
  s = s0·exp(0.01·dT)) is the seed to port into planetmodel properly. AdGIA's
  `earth_data/README.md` anticipates exactly this ("for conversion into
  viscosity … as the need arises"), so the addition serves both codes.
- Precedent for a C++ reader of a planetmodel format with a frozen schema
  string: `AdGIA::MeshManifest` (reads `planetmodel.mesh.manifest/5`).

**planetmodel workstream** (Python; prerequisite for sl3dpp stage 5, useful to
AdGIA regardless):

1. Freeze the netCDF 3D model spec (names, dims, attributes, a schema string
   like the manifest's) and build writer + reader.
2. Tomography ingestion (S20RTS-style SH coefficients with radial splines →
   fields on a model).
3. The dlnvs → (dlnρ, dT) → viscosity scaling chain, with vocabulary additions
   (and possibly a temperature dimension / relaxation-time names).
4. Decide the lateral representation in the file. The stage-3 design chose the
   gridded `Sample` layout; store the **log**-viscosity perturbation (the
   exponentiated field is not band-limited), let sl3dpp analyse/resample to its
   own grid and lmax via GSHTrans.

**Internal representation is sl3dpp's own.** The C++ model classes are
designed to suit this project (GLL nodal fields, sections, spectral layouts),
not to mirror planetmodel's object model. File formats — the deck, the planned
netCDF model file, the ice histories — are strictly an IO boundary that the
readers translate across. The planetmodel additions below are built as part of
the broader process (they are needed anyway, not least by AdGIA), and the
early stages need no real models at all: analytic/synthetic models (uniform
layers, prescribed perturbations) carry development and testing until the
format work lands.

**sl3dpp reader scope** (deliberately thin):

- Deck reader (plain text, stdlib only) for 1D models — the canonical input.
  A tiny legacy reader for the old `prem.simp.vm1` format is worth keeping for
  benchmark parity with the Fortran.
- netCDF reader (netCDF-C, a light dependency already needed for the ice
  histories) for 3D fields on the planetmodel grid: read grid + fields,
  convert SH conventions, resample to the run grid.
- Ice histories: port `module_ice`'s hard-won ICE-5G/6G/7G quirks table
  (field-name differences, grid offsets), or read AdGIA's `earth_data`
  exports — decide when the format work in (1) settles.
- Convention conversion utilities at the IO boundary: planetmodel/pyslfp real
  orthonormal no-CS-phase SH ↔ GSHTrans complex fully-normalised CS basis.
  Internally sl3dpp uses GSHTrans's basis throughout (identical to the old
  Fortran's, which helps stage-by-stage verification).

## 3. The one real design problem: implicit stepping with 3D viscosity

In 1D, μ_eff(r; Δt) is radial: degrees stay decoupled and per-(scheme, Δt)
factorisation caching (pyslfp) is enough. With s(θ,φ,r), the effective modulus
is laterally varying and the implicit operator couples all (l,m) — which is
exactly what the old explicit formulation was dodging.

**Planned resolution:** split K_eff = K̄_eff + δK_eff, where K̄_eff is built
from a 1D reference profile τ̄(r) (e.g. laterally averaged) and factorised per
degree, and δK_eff (the lateral perturbation) is applied pseudospectrally on
the RHS (synthesis → pointwise multiply by the laterally varying branch
weights → analysis). Iterate uⁿ⁺¹ together with the ocean-load and rotation
fixed point — one combined within-step iteration under one Anderson mixer,
with the lateral term entering through the generalised SLE's existing
response channels. Each iteration = one banded back-solve per degree plus a
few transforms; factorisations reused across the run at fixed Δt; no
Maxwell-time step limit (≈50 yr steps throughout instead of 2.5 yr where the
asthenosphere demands it).

The pointwise memory update m⁺ = e(x)m + α(x)dⁿ + β(x)dⁿ⁺¹ uses weights
evaluated on the grid from h(x) = Δt/τ(x) — the same kind of pointwise grid
operation as the old s(x) multiply.

**Open question (the research content of the project):** convergence of the
within-step iteration under strong lateral viscosity contrasts (s scales
exponentially with temperature). Encouraging: φ₁(h) ∈ (0, 1] bounds each
branch's contribution by its modulus; Anderson extends the basin. Fallbacks to
keep in the design: damping; a better reference profile (e.g. per-degree or
worst-case); IMEX (reference implicit, lateral explicit). Demonstrating where
this works is itself a benchmark-paper result alongside the AdGIA comparison.

## 4. Formulation summary (what gets implemented)

- Quasi-static, compressible, self-gravitating, per-degree radial operators:
  spheroidal + toroidal, fluid regions, interface terms, DtN/Robin exterior
  and truncation conditions, proper l = 0 and l = 1 — following pyslfp's
  `DegreeSystem`, with LDLᵀ (LU fallback), never plain Cholesky.
- Internal-variable Maxwell/Prony rheology; exponential trapezoid default,
  BE and ETD1 variants; adaptive stepping optional. Implement ntau > 1
  properly (nearly free with the weight algebra).
- Generalised SLE with Anderson, response channels, exact-quadrature
  integrals, mass conservation; frozen shorelines as the production path,
  Kendall-style migration available; rotation (transverse + axial) per pyslfp.
- Pseudospectral 3D coupling at spin N = −2..2 via GSHTrans, dealiased with
  `ForBand` oversampling (an improvement over the Fortran, which never
  dealiased).
- Discrete adjoint = exact transpose of the step; reverse sweep with (m†, d†);
  jump sources at observations; kernels (viscosity, ice, τ-derivative)
  accumulated during the sweep; record-every-M checkpointing with exact
  inverse-step reconstruction, plus an optional disk tier (retire the old
  all-in-RAM history).
- Inversion drivers last, or deferred to a pygeoinf-style interface.

## 5. Architecture

### 5.1 Package

New repo `~/dev/sl3dpp`. C++23 (forced by GSHTrans), header + src split,
static library `sl3dpp` with thin executable drivers, GoogleTest. AdGIA's
naming and layout conventions (namespace `sl3dpp` with `detail`; PascalCase
methods, trailing-underscore members; nested `Options` structs; `std::function`
hooks).

### 5.2 Classes (mirroring pyslfp's decomposition)

| Component | Responsibility | Template |
| --- | --- | --- |
| `EarthModel`, `RadialMesh` | deck reading, splines, non-dimensionalisation, GLL mesh, layer/section classification, reference gravity, degree-dependent truncation | planetmodel `RadialMesh`, Fortran `module_model` |
| `DegreeOperator` | per-degree spheroidal/toroidal assembly, banded factorisation, multi-RHS solves | pyslfp `DegreeSystem` |
| `Rheology`, `StepWeights` | Prony branches, trapezoid/BE/ETD1 weight algebra | pyslfp `weights.py`, AdGIA `rheology.hpp` |
| `SeaLevelEquation` | generalised SLE, Anderson, rotation block, mass conservation, shorelines | pyslfp `core/sea_level.py` |
| `ViscoelasticOperator`, `Step` | two-phase step, effective operators per (scheme, Δt), memory force, 3D lateral iteration | pyslfp `operator.py`, AdGIA `viscoelastic.hpp` |
| `Trajectory`, `AdjointSolver`, `Kernels` | checkpointing with inverse step, reverse sweep, kernel accumulation | pyslfp `evolution.py`, AdGIA `adjoint.hpp` |
| `IceHistory` | netCDF ICE-NG readers, grid interpolation, time interpolation | Fortran `module_ice` |
| `io/` | deck + netCDF model readers, SH convention converters, output writers | §2.5 |

### 5.3 Discretisation choice

Hand-rolled GLL spectral elements + LAPACK banded solves, **not** MFEM. The
per-degree systems are small banded complex/real systems with interleaved
fields and many RHS; MFEM buys little in 1D, adds a heavyweight dependency,
and keeping sl3dpp's discretisation independent of AdGIA's strengthens the
benchmark. (Serial MFEM remains the documented fallback if API familiarity
ever outweighs this.)

### 5.4 Parallelism

OpenMP only. Parallel axes: degrees (independent banded factorisations and
solves), latitudes (pointwise grid operations), and GSHTrans's own
`Execution::Parallel` for transforms. MPI — including radial-block
partitioning — is rejected: the radial direction is where the serial banded
solves live, the pseudospectral coupling is global in (l,m), and any
distribution becomes transpose-heavy like a parallel FFT. Memory fits shared
machines: at lmax = 256 the Wigner table is ~2.6 GB (matrix kernel) plus a few
GB of viscous-element state.

### 5.5 Conventions

- SH basis: GSHTrans native (complex, fully normalised, CS phase) internally;
  converters to the real pyshtools convention at the IO/benchmark boundary.
- Non-dimensionalisation, signs and Love-number conventions from
  `pyslfp_theory.tex` (ξ = −u − (φ+ψ)/g + c; g = ∂_rΦ > 0; centre-of-mass
  frame; body (b, g, G) consistent with the radial model).
- Memory variables stored only over viscous elements; 5 canonical deviatoric
  components on the grid (3D) / spectral (1D path).

## 6. Staging

Each stage verifiable against an existing code — this is what makes the port
largely mechanical and automatable stage by stage.

| Stage | Content | Verification gate |
| --- | --- | --- |
| 0 | repo scaffolding, CMake with pinned deps, conventions doc; deck reader (+ legacy format), non-dimensionalisation, `RadialMesh`, reference gravity | g, φ₀ vs Fortran `grav_cal` and planetmodel `mesh1d.gravity` |
| 1 | `DegreeOperator`: spheroidal + toroidal, fluids, interfaces, truncation, LDLᵀ/LU, multi-RHS | elastic generalised Love numbers vs pyslfp table (near machine precision); reciprocity relations |
| 2 | elastic fingerprints: generalised SLE + Anderson + rotation + shorelines; adjoint SLE solve | vs pyslfp elastic fingerprints; dot-product test |
| 3 | 1D viscoelastic stepping: weights, effective operators per (scheme, Δt), memory force, two-phase step, unrelaxed t₀ solve, adaptive option | vs pyslfp GIA driver and gia1d; measured order 2 |
| 4 | 3D viscosity: GSHTrans spin-2 machinery, grid-pointwise weights, within-step lateral iteration (**the research stage**); netCDF 3D model reader | vs old sl3d (`visco_test`, `sea_level_3d`) in the explicit/weak-contrast limit; Δt self-convergence; **AdGIA on a common model — the headline benchmark** |
| 5 | adjoints: reverse sweep, checkpointing + inverse step, viscosity/ice/τ kernels | dot-product tests at machine precision; FD kernels; discrete-vs-continuous adjoint diagnostic |
| 6 | inversion drivers or pygeoinf interface; ice-history IO polish; docs, examples | end-to-end reproduction of a Fortran inversion case |

Parallel Python workstream in planetmodel (before stage 4): netCDF model
format, tomography ingestion, viscosity scaling chain (§2.5).

## 7. Dependencies and build

- Required: GSHTrans (→ NumericConcepts, GaussQuad, FFTWpp), Interpolation,
  LAPACK/BLAS, FFTW3 (all three precisions, via FFTWpp), netCDF-C, OpenMP.
- FetchContent with `FIND_PACKAGE_ARGS` and per-dependency tag variables, all
  pinned; `FETCHCONTENT_SOURCE_DIR_<NAME>` overrides for local sibling
  checkouts during development.
- GCC 13+ / Clang 18+, CMake ≥ 3.24, GoogleTest for tests.

## 8. Open questions

1. Convergence strategy and limits for the within-step lateral iteration (§3):
   reference-profile choice, Anderson vs damped Picard, IMEX fallback.
2. Lateral representation in the planetmodel netCDF file (gridded Sample
   layout per the stage-3 design vs SH coefficients per radial node) and the
   frozen variable/attribute spec.
3. Whether degree-dependent radial truncation keeps the old `20/(l+0.5)` rule
   or adopts pyslfp's `a(1 − ε^{1/(l+1)})` grading (lean: pyslfp's).
4. ntau > 1 / Prony vocabulary: whether planetmodel grows relaxation-time
   fields or sl3dpp derives branches from `viscosity` as pyslfp does (lean:
   the latter until a multi-branch model is actually needed).
5. Ice-history interchange: read the reduced netCDF files from `sl3d/data`
   directly (port `module_ice`) vs standardise on AdGIA `earth_data` exports.
6. Config: namelist-style run files served the Fortran well; AdGIA uses C++
   setters + `OptionsParser`. Lean: `Options` structs in the library, a small
   TOML/INI-style run file in the drivers.
