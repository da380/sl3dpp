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

Alongside the core GIA problem, **simple elastic sea-level fingerprints are a
first-class public capability** (stage 2), as in pyslfp: a user can compute a
fingerprint for a load without touching the viscoelastic machinery. The API
is kept pyslfp-shaped so that sl3dpp could later serve as a faster backend
for pyslfp — that binding is explicitly **not** part of this plan, but
nothing in stage 2's interface should preclude it.

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
- **GSHTrans is a live target** (noted 15 Oct 2026): recently updated with
  pseudo-spectral work in mind, and where sl3dpp needs something it cannot
  do, updates can be planned. Running wishlist:
  1. `GaussLegendreGrid` passthrough of the base's `nPhi` parameter
     (requesting `nlon = 2 lmax + 1`). Low priority: benchmark comparisons
     are done in coefficient space and through grid-independent
     functionals, not by matching grids (decided 15 Oct 2026) — but an
     explicit `nPhi` may still be wanted for IO-grid interop one day.
     (Reading nPhi needs nothing: `NumberOfLongitudes()` exists — the
     open item is only choosing it at construction.)
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

A candidate input source for (2): the curated collection at
https://github.com/shuleyu/seismic-tomography-models wraps many published
tomographic models in one consistent netCDF format (noted 15 Oct 2026) —
the natural feed for the ingestion chain and a format worth aligning the
reader with.

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
- **Degree-1 toroidal motion is part of the 3D problem** (decided 17 Oct
  2026, §9). Lateral viscosity drives it: differential degree-1 rotation
  carries strain through W − r∂rW. The toroidal block therefore runs over
  l ≥ 1, not l ≥ 2. Its quasi-static operator is singular. Rigid rotation
  (W ∝ r) carries no strain, which gives one null vector per order m in
  each solid region that fluid separates from the rest (the mantle and the
  inner core). These are removed by convention, as the centre-of-mass
  frame removes the degree-1 spheroidal translations:
  - The singular system is consistent. Memory stresses are symmetric and
    do no work on a rigid rotation, and the surface loads and potentials
    have no toroidal part, so the right-hand side is torque-free.
  - The convention only picks the representative. Rigid rotation carries
    no strain, so the memory evolution, every other degree and sea level
    do not depend on it. Only the degree-1 toroidal displacement itself
    does.
  - Which convention does not matter at all, so the convenient one is
    used (decided 17 Oct 2026, §9): pin one W dof per solid region. The
    pinned matrix is SPD, so the banded Cholesky stands, and since the
    right-hand side is consistent the pinned solve solves the full
    system. No projection is needed.
  - **What matters is what is measured.** Only quantities invariant under
    the null space are well defined, and this is a requirement, not a
    nicety: it is the solvability condition of the adjoint equations
    (the measurement's dual must annihilate the rigid modes, or the
    singular adjoint system has no solution), and it is the rule for
    benchmarks and tests. Relative distances are fine; relative
    displacements are not. Sea level, radial displacement, the
    potential and the rotation vector do not see the degree-1 toroidal
    rigid rotations (no radial part, no mass moved), so the current
    measurement set (`Measurement`, `Adjoint.hpp`) qualifies. A
    horizontal displacement at a point does not; a horizontal observable
    must be built to be invariant, such as a change in distance between
    sites.
  - The degree-1 spheroidal translations are different (David, 17 Oct
    2026). They are fixed by the centre-of-mass frame: the surface
    degree-1 potential dof is removed, so there is no external degree-1
    field and the centre of mass of Earth and load stays at the origin.
    That frame is physical, not arbitrary, so quantities relative to it
    are well defined. The degree-1 radial displacement the code computes
    is the displacement relative to the centre of mass (u − u_CM·r̂ is
    translation-invariant, and u_CM = 0 in this frame), and likewise the
    potential and sea level may be measured. The binding constraint is
    on horizontals, which see the arbitrary rotation gauge. (A first,
    stricter reading the same day, allowing only u + φ/g at degree 1 and
    enforcing it in the adjoint solvers, was reverted.)
  - Gates: the right-hand side torque-free at round-off; strain, sea
    level and every other invariant output identical (to round-off)
    whichever dof is pinned, while W itself changes only by the rigid
    mode; the zero-lateral gates unchanged (with no lateral variation
    the toroidal block is not excited).
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
| `EarthModel`, `RadialMesh` | deck reading, splines, non-dimensionalisation, GLL mesh, layer/section classification, reference gravity | planetmodel `RadialMesh`, Fortran `module_model` |
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

**One global radial mesh.** There is a single GLL mesh shared by all degrees
and by the grid fields (memory variables, lateral viscosity). The 3D coupling
moves fields pseudospectrally between (l,m) space and the (radial node ×
angular grid) — that requires every degree's coefficients to live on the same
radial nodes, so per-degree sub-meshes (pyslfp's choice, viable only because
pyslfp has no lateral coupling) are ruled out. Degree-dependent truncation is
a per-degree *DOF subset* of the global mesh, with the truncation radius
snapped to an element boundary — as the Fortran did, which also helps
benchmark parity. The radius heuristic (`20/(l+0.5)` vs `ε^{1/(l+1)}`
grading) is an option chosen by testing (open question 3).

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
- **Memory-variable point set** (raised 15 Oct 2026): m currently lives at
  the same ngll GLL nodes as the displacement (pyslfp/Fortran parity). A
  plain order drop would lose accuracy — the drives e_M, e_R, e_S are
  degree-N polynomials per element (the differentiated terms carry the
  affine factor r) — but storing m at the N interior **Gauss** points
  instead is not an order drop: the N-point Gauss rule matches the
  (N+1)-point GLL rule's 2N-1 exactness for the memory-force integral, m
  is only ever sampled/updated/integrated at its own points (no inf-sup
  issue: m is local and undifferentiated), and boundary doubling
  disappears. Saves ~20% of the internal-variable storage and, at stage 4,
  ~20% of the radial nodes driving the spin-weighted transforms in the
  memory update. Plan (revised 15 Oct 2026): keep GLL storage as the
  default through stages 3 and 5, where the machine-tight pyslfp gates
  (identical algebra) are the porting-bug detector; implement the
  Gauss-point set as a switchable option around stage 4. The divergence
  from pyslfp is then a **benchmarking asset, not a cost**: machine-level
  agreement between algebraically identical codes checks the port, but
  what the community values is independent implementations with
  *different* discretisations agreeing at observational accuracy — the
  Gauss-point variant adds exactly such an axis, cheaply. Kernels fit the
  reduced set naturally: the viscosity kernel is accumulated pointwise at
  the memory points and integrated with the matching (equally exact)
  rule, and the machine-precision discrete-adjoint identity is a property
  of the discretisation's self-consistency, not of the point set — the
  one discipline being that τ, the step weights, their h-derivatives and
  the kernel accumulators all migrate to the Gauss points together.
  *(Added 17 Oct 2026.)* One more term must move with them: the implicit
  step eliminates m⁺ = m̃ + β d(u) into equilibrium, giving
  K_eff = K_U − Σ_k Dᵀ W (μ_k β_k) D at the memory points. At the GLL
  nodes that is the relaxed-moduli assembly (the stiffness's own
  quadrature points), which is how `EffectiveMaterial` builds it; at Gauss
  points the relaxation term (degree 2N, exact in neither rule) must be
  assembled at the Gauss points to keep the elimination exact. Done as
  performance chunk 7a: `MemoryPoints` (GLL or Gauss reference geometry)
  and a `Relaxation` term that `DegreeOperator`, `ToroidalOperator` and
  `ComputeLoveNumbers` subtract; with GLL points it reproduces
  `EffectiveMaterial`'s operators and Love numbers to round-off (the
  gate), and the Gauss and GLL effective Love numbers differ by 8e-9 at
  600 km elements, 1e-10 at 300 km, 4e-12 at 150 km.
  *(Revised 17 Oct 2026, with Gauss points made the default.)* That
  form is unstable for long steps: as a branch relaxes fully (β → 1) the
  shear left in K_eff is the GLL quadrature of μ|ε|² less the Gauss
  quadrature of the same energy, which has no definite sign, and a PREM
  Maxwell mantle grew from round-off for h = dt/τ of a few hundred and
  up. The cure is to split the energy as the continuum does,
  ∫ μ_∞|ε|² + Σ_k ∫ μ_k |ε − m_k|²: the long-term part at the GLL nodes,
  each branch's at the memory points where m_k lives. Then
  K_eff = K(μ_∞) + Σ_k Dᵀ W (μ_k w_k) D, w_k = 1 − β_k the unrelaxed
  fraction, positive term by term for any step, and the elastic operator
  (w = 1) is K(μ_∞) + Σ_k Dᵀ W μ_k D. The term is now a `PointShear`
  the operators add to the long-term material; at GLL points it is still
  `EffectiveMaterial`'s assembly to round-off. The price: at Gauss points
  the discrete elastic solid itself differs from the GLL one, at the
  quadrature level (2e-10 in the surface sums at lmax 64).

## 6. Staging

Each stage verifiable against an existing code — this is what makes the port
largely mechanical and automatable stage by stage.

**Where references disagree, pyslfp is authoritative for the elastic
problem.** Elastic Love numbers from an updated Fortran build (an abortive
sl3d update, not itself a reference) sit ~1% off pyslfp — more than
numerics, cause not chased, and possibly shared by sl3d suitably isolated —
so the stage-1/2 gates compare against pyslfp, and Fortran comparisons are
reserved for the time-dependent stages (3–5), at tolerances judged per
case.

**Cross-library field comparisons are made in coefficient space** (via the
documented basis conversion) **or through grid-independent functionals**
(rotation vector, mean sea level, surface integrals), never by matching
grids point for point (decided 15 Oct 2026). Tight gates use smooth,
band-limited analytic inputs — including a smooth ocean-function override
in the SLE — so results are grid-independent to spectral accuracy; the
sharp flotation ocean function is compared separately at the coarser
tolerance its coastline quadrature permits, and documented as such.

**Verification mechanics** (decided 15 Oct 2026): reference values are
produced by small Python scripts in `scripts/` that drive pyslfp/planetmodel
and dump committed fixtures into `tests/data/` (text/JSON, with a provenance
header recording the generating script, library versions/commits and date);
the GoogleTests read the fixtures. Tests therefore need no Python, and
regeneration is reproducible. Fortran references are captured the same way
(a documented run of the old code, output committed as a fixture).

| Stage | Content | Verification gate |
| --- | --- | --- |
| 0 | repo scaffolding, CMake with pinned deps, conventions doc; deck reader (+ legacy format), non-dimensionalisation, `RadialMesh`, reference gravity | g, φ₀ vs Fortran `grav_cal` and planetmodel `mesh1d.gravity` |
| 1 | `DegreeOperator`: spheroidal + toroidal, fluids, interfaces, truncation, LDLᵀ/LU, multi-RHS | elastic generalised Love numbers vs pyslfp table (near machine precision); reciprocity relations |
| 2 | elastic fingerprints as a standalone public capability (§1): generalised SLE + Anderson + rotation + shorelines; adjoint SLE solve | vs pyslfp elastic fingerprints; dot-product test |
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
3. ~~Whether degree-dependent radial truncation keeps the old `20/(l+0.5)`
   rule or adopts pyslfp's `a(1 − ε^{1/(l+1)})` grading.~~ **Structurally
   resolved** (15 Oct 2026, see §9): one global GLL mesh, truncation by
   per-degree DOF subsetting snapped to element boundaries — never per-degree
   sub-meshes. The truncation-*radius* heuristic stays open: implement both
   formulas as options and choose by testing.
4. ntau > 1 / Prony vocabulary: whether planetmodel grows relaxation-time
   fields or sl3dpp derives branches from `viscosity` as pyslfp does (lean:
   the latter until a multi-branch model is actually needed).
5. Ice-history interchange: read the reduced netCDF files from `sl3d/data`
   directly (port `module_ice`) vs standardise on AdGIA `earth_data` exports.
6. Config: namelist-style run files served the Fortran well; AdGIA uses C++
   setters + `OptionsParser`. Lean: `Options` structs in the library, a small
   TOML/INI-style run file in the drivers.
7. **Second-order adjoints** (raised 15 Oct 2026): Yu et al. 2025
   (AdGIA's `doc/Elasticity/ggae388.pdf`) details them and they are
   planned for AdGIA; a matching addition here is worth considering once
   the first-order adjoints (stage 5) are gated, but it is not a
   priority — this code is not aimed at production-scale inversion runs.
8. **Transversely isotropic viscoelasticity** (raised 15 Oct 2026; pyslfp
   keeps viscoelasticity isotropic). The defining physical condition: a
   Maxwell-type TI material relaxes towards a state supporting **no
   deviatoric stress**, so the long-time stiffness is an isotropic-stress
   projection (σ∞ = κ* tr ε I). That condition alone should pin the
   structure of the relaxation; open residuals are (a) the relaxed bulk
   response κ* when the TI tensor couples volumetric and deviatoric parts
   (hydrostatic strain in a TI medium produces deviatoric stress, so
   relaxation shifts the apparent bulk modulus), and (b) whether different
   moduli carry different relaxation time-scales, i.e. several internal
   variables/branches rather than one — representable in the planned
   ntau > 1 weight algebra. Not to be solved now, but stage 3's
   `Rheology`/`StepWeights` interface must not preclude moduli-resolved
   branches. Whatever is settled here is intended to translate back to
   pyslfp/AdGIA.
9. ~~**The degree-1 toroidal frame** (raised 17 Oct 2026, §4).~~
   **Resolved** (17 Oct 2026, David): the choice does not matter at all,
   so the convenient one is used (one pinned dof per solid region; the
   inner core needs its own, because quasi-statically the fluid outer
   core transmits no tangential traction). What matters instead is that
   every measured, benchmarked or tested quantity is invariant under the
   rigid rotations (§4).
10. ~~**Gauge-dependent measurements in the existing tests**~~
   **Withdrawn** (17 Oct 2026). The tests' displacement and potential
   measurements have degree-1 content, but at degree 1 they are
   relative to the centre of mass, a physical frame, so they are well
   defined (§4). An enforcement briefly added the same day (rejecting
   degree-1 displacement and potential duals not of the form u + φ/g,
   with the tests and the stage-5 fixture changed to suit) was reverted.
   A check belongs with any future horizontal-displacement measurement
   type, which must be rotation-invariant (§4).

## 9. Decision log

- **15 Oct 2026** — No frequency-domain machinery in this library: the time
  domain keeps every operator real, so the degree solves stay real
  factorisations with [Re | Im] right-hand-side pairs (the cheap corner —
  a complex solve against a real matrix only adds arithmetic). Complex
  moduli / complex-frequency Love numbers (normal-mode or Laplace-domain
  work) would need a complex DegreeOperator flavour and are out of scope.
- **15 Oct 2026** — Elastic sea-level fingerprints are a first-class public
  capability (stage 2), pyslfp-shaped, so sl3dpp can later serve as a faster
  pyslfp backend; the binding itself is outside this plan (§1).
- **15 Oct 2026** — Stage-4 design recorded in `docs/stage4.md`: the five
  canonical spin components, the degree ↔ spin maps and memory-force rows
  extracted from the Fortran; the toroidal degree operator added to
  stage-4 scope in its TI form (Myhill & Al-Attar, verified against
  DSpecM1D); nonlinear Maxwell rheologies (stress-dependent τ, AdGIA's
  `RelaxationLaw`, Crawford et al. 2017 App. A) added as a readily
  buildable stage-4 option — the stress invariant comes from the grid
  components the pipeline already carries, and the nonlinearity folds
  into the same within-step fixed point.
- **15 Oct 2026** — Stage-2 gate calibration: with everything band-limited
  (smooth ocean override) the SLE solutions agree with pyslfp at ~2e-9 in
  coefficient space; with the sharp flotation ocean function the solution
  is grid-determined only to the coastline quadrature — pyslfp's own DH vs
  GLQ spread at lmax 32 is 0.44% in ocean area, 0.049 of the largest
  sea-level coefficient, and a factor ~2 in the (ill-determined, nearly
  zonal) transverse rotation. sl3dpp sits inside that spread; the sharp
  gate's tolerances are set to it.
- **15 Oct 2026** — pyslfp is the authoritative elastic benchmark; a ~1%
  Love-number discrepancy against an updated Fortran build is known and
  unexplained (§6). TI viscoelasticity added as open question 7: relaxation
  defined by vanishing relaxed deviatoric stress, possibly with
  per-modulus time-scales; stage 3's interface must keep it representable.

- **15 Oct 2026** — Verification gates run on committed fixtures generated by
  Python scripts in `scripts/` (see §6); no Python at test time.
- **15 Oct 2026** — One global GLL radial mesh for all degrees and grid
  fields; per-degree truncation is DOF subsetting snapped to element
  boundaries, never a separate sub-mesh (see §5.3). Rationale: the
  pseudospectral 3D coupling needs a single radial node set. Truncation-radius
  heuristic left open as a tested option.
- **15 Oct 2026** — Units: SI in, SI out, non-dimensional everything
  inside. Public APIs convert at the call boundary (readers
  non-dimensionalise on entry, writers/returns re-dimensionalise on exit);
  components exchange non-dimensional numbers (`EarthModel::Dimensionless`).
  Scheme and constants are planetmodel's `Scales.geophysical` (G = 1).
- **16 Oct 2026** — Cross-project convention (David): pyslfp, AdGIA and
  sl3dpp should share ONE internal non-dimensionalisation. Found the same
  day: pyslfp's EarthModel scales (T ≈ 805 s, g_nd ≈ 1) deviate from
  planetmodel's documented `geophysical` (T ≈ 1648 s, G_nd = 1), which
  sl3dpp follows; the non-dimensional potential scales as T², a clean
  4π/3 between the codes, invisible to every stage 0–4 gate. Until the
  convention is unified (a pyslfp-side change, David's call), cross-code
  comparisons of potentials and frequencies anchor through SI per
  channel; raw non-dimensional values are never compared.
- **16 Oct 2026** — Examples (stage 6, pulled forward as features settle):
  example programs per feature with linked Python plotting scripts
  (cartopy or pyslfp for maps), a poetry environment for the plotting
  side with the setup documented, and the scripts run **in the build
  tree, never from source** (no outputs polluting the source tree).
- **16 Oct 2026** — Stage 5 implements BOTH the discrete and the continuous
  adjoint as options behind one interface and result type, with the
  **continuous adjoint the preferred production path** (design in
  `docs/stage5.md`). Rationale: the discrete adjoint is a numerically exact
  gradient, but of the discretised problem — exact-but-possibly-wrong
  precisely when the run is under-resolved; for a good discretisation the
  two agree to practical accuracy, so their disagreement is itself the
  resolution diagnostic. pyslfp's recent adjoint work (discrete +
  continuous + quelling + 1D viscous kernels + checkpointing) is the
  porting template for the 1D path.
- **15 Oct 2026** — Conventions recorded in `docs/conventions.md`;
  `.clang-format` taken verbatim from AdGIA. AdGIA supplies naming/layering
  only: the idiom is modern C++23 (concepts, ranges, `std::expected`, value
  semantics), not AdGIA's MFEM-constrained style.
- **17 Oct 2026** — Degree-1 toroidal motion is included in the 3D
  problem (David): it is a physical mode that lateral viscosity drives,
  and its rigid-rotation null space is removed by convention (§4; the
  frame is open question §8.9). This reverses the expectation recorded in
  `stage4.md` §4, which was to drop it, and which the code follows:
  `GridCoupling::ToroidalForce` and the toroidal solves start at l = 2.
  That omission is to be fixed before the AdGIA benchmark. The Fortran is
  no reference for this mode: it factors the unconstrained, singular
  degree-1 form by banded LU (`aa_tor_1`, `module_mat.f90:58-62`;
  `amat_tor` applies no constraint), so it fixes no frame.
- **17 Oct 2026** — The degree-1 toroidal frame does not matter (David):
  the convenient gauge is used, one pinned dof per solid region (§4,
  §8.9). The real requirement is on measurements: only quantities
  invariant under the rigid rotations are well defined. It is the
  existence condition for the adjoint equations, and the rule for
  benchmarks and tests (relative distances are fine, relative
  displacements are not).
- **17 Oct 2026** — The measurement rule at degree 1 (David): the
  spheroidal translations are fixed by the centre-of-mass frame, a
  physical frame, so quantities relative to the centre of mass (radial
  displacement, potential, sea level) are well defined and may be
  measured. The toroidal rotations are fixed by an arbitrary pin, so a
  horizontal displacement at a point is gauge: horizontal observables
  must be rotation-invariant. A first, stricter reading (only u + φ/g
  at degree 1, enforced in the adjoint solvers) was reverted the same
  day (§4, §8.10).
- **17 Oct 2026** — Degree-1 toroidal implemented (performance chunk 5):
  `ToroidalOperator` accepts l = 1, pinning one dof per solid region
  (the top node by default; `pin_fraction` moves it), and the toroidal
  force, solves and strain recovery run from l = 1. Gated: rigid
  rotations are the null space; the memory force is torque-free; the
  pinned solve solves the full system; moving the pin changes only the
  degree-1 rigid rotation. Effect on the lateral example: 0.25% of the
  lateral-minus-radial sea-level difference; the Fortran gate is
  unchanged (its pattern cannot excite the mode, by symmetry).
- **17 Oct 2026** — Recorded forward states keep the internal variables
  only, as in pyslfp and AdGIA (David): the adjoints regenerate a node's
  strain by one elastic solve under its recorded loads (performance
  chunk 6). Halves the stored state with one branch; costs one elastic
  solve per node per sweep.
- **17 Oct 2026** — Save or recompute (David): `record_strain` on both
  evolutions keeps the strain with the internal variables on request;
  recompute (internal variables only) stays the default.
- **17 Oct 2026** — Gauss-point memory in the 1D forward (performance
  chunk 7b-i): `ViscoelasticOperator::Options::memory_points`, GLL the
  default and bitwise unchanged. The step stays an exact elimination
  (the regenerated strain matches the step's to 1e-11), the state holds
  80% of the memory, and surface responses agree with GLL to 1.4e-8 at
  600 km elements, 2.4e-10 at 300 km (a coupled sea level run, 5e-12).
  The 1D adjoints refuse Gauss points until 7b-ii.
- **17 Oct 2026** — Gauss-point memory in the 1D adjoints (performance
  chunk 7b-ii): both sweeps index the memory points, and the itau
  kernels, accumulated there, reach the nodal itau through the
  interpolation's transpose (masked where the clamp at zero acts), so
  `IceKernels::itauKernel` keeps its nodal layout and the viscosity map
  is unchanged. The discrete gradient identity holds to 1e-8 at Gauss
  points, FD/Richardson confirms the kernel under both schemes, and the
  nodal kernels agree with GLL's to 1.1e-6 (discrete) and 7e-7
  (continuous). The 3D evolution keeps the GLL reference until 7c.
- **17 Oct 2026** — Gauss-point memory in the 3D path (performance chunk
  7c): `Viscoelastic3D::Options::memory_points` sets the coupling's radial
  memory nodes and the 1D reference alike. The strain is interpolated to
  the points and the force rows spread back through the same Lagrange
  values; the reference effective operators carry the relaxation term;
  the 3D itau is the reference's point itau times exp(-delta) at the
  point's radius. The zero-lateral gate holds at Gauss points to
  round-off (8e-16 on the step, tighter than the 2-4e-12 that separates
  the point sets, so the gate discriminates), coupled runs and their
  continuous kernels included; with lateral structure the point sets
  agree to 1e-11 and the log-viscosity kernel meets FD as at GLL. At lmax
  64 on 8 threads: 20% fewer nodes, the pass 14% faster, peak RSS 9.55 GB
  for 11.69. GLL stays the default (bitwise unchanged); whether to switch
  is David's call.
- **17 Oct 2026** — Gauss points become the default for the internal
  variables (`memory_points` in ViscoelasticOperator, GridCoupling and
  Viscoelastic3D; `step_timing` likewise), David's decision after 7c. The
  displacement and every nodal field stay on the GLL nodes; only where
  the internal variables live changes. GLL stays selectable, and the
  tests that read the GLL layout directly or exercise the relaxed-moduli
  identity pin it explicitly (the pyslfp and Fortran gates pass at Gauss
  points within their tolerances, so they now check the default path).
  Making it the default exposed an instability of the 7a/7b form at long
  steps (§5.5, revised): fixed by the shear split, the effective
  operators built on the long-term material plus a `PointShear` (the
  renamed, sign-flipped `Relaxation`) of the branches' unrelaxed moduli
  at the points, the elastic operators likewise. A regression test steps
  h = 2000 400 times: Gauss stays within 1.7e-9 of GLL, where the old
  form reached 0.5. The split also brought the point sets closer
  elsewhere (1D surface gap 1.2e-9 at 600 km elements, 1.9e-11 at 300 km,
  from 1.4e-8 and 2.4e-10; lateral 3D steps 5.5e-12 from 1.2e-11). Both
  point sets grow alike near h = 1e4 over hundreds of steps on PREM: a
  property of the model or the scheme there, not of the point set, and
  not investigated further. The 1D operator now holds its points by
  pointer, so the per-degree workspaces' references survive a move of
  the operator (a latent dangling pointer since 7b-i).
- **17 Oct 2026** — The 3D step's serial remainder threaded (performance
  chunk 8). The step's own loops (weights and decay, the pointwise
  weight, the fixed-force add, the residual, the damping, the memory
  update, `SolveElastic`'s weighting) run element-wise threaded
  (`detail::ParallelElements`, a static schedule), and the Anderson
  mixer's two sweeps are threaded with per-thread partial sums added in
  thread order (`AndersonMixer(memory, parallel)`; the 1D users keep the
  serial default). The mixer stores dX + dF in place of dX, the one form
  its update reads. The open step keeps its buffers (`Respond`'s grid
  tensor and analysed force, the spent iterate `StrainOf` refills;
  `GridCoupling` gained into-buffer `Synthesise`/`Analyse`), and
  `std::move(step).Finish` turns them into the new state. `SolveAll` only
  solves; `SolveElastic` adds channels through the cached path
  `AddChannels` uses. The 3D adjoint's itau accumulation is threaded
  over memory nodes. At one thread every result is bitwise unchanged; at
  lmax 64 on 8 threads a pass is 0.76–0.79 s for 1.05 (Gauss), 0.94 for
  1.34 (GLL). `StartStep` (0.82 s a step) is now the largest overhead.
- **17 Oct 2026** — Adaptive mixing in the 3D within-step iteration
  (performance chunk 9). Each step starts as Picard; Anderson (depth 5)
  takes over for the rest of the step once a pass shrinks the residual
  by less than `IterationControl::anderson_rate` (0.5; also on
  `Viscoelastic3D`, `SeaLevelEvolution3D` and `AdjointSolver3D`
  options). Measured (`stage4.md` §5a, revisited): pure Picard wherever
  Picard is quick, which at the benchmark's weak contrast saves 30% of
  the pass time and 1.9 GB at lmax 64, and pass for pass Anderson where
  it is not; `anderson_rate` 0 restores the old iteration bitwise. A
  give-up-for-Picard safeguard was tried and dropped (it gave Anderson up
  in its early lag, and Picard then diverged); the case that motivated it
  was an artefact of a real bug, now fixed: the residual's maxima passed
  over NaNs, so an all-NaN iterate read as converged. A non-finite pass
  now ends the iteration unconverged. A resolution check (lmax 16/24/32,
  contrasts to 1000x) shows the pass counts independent of lmax and the
  low-degree response at lmax 16 good to 3e-8.
- **17 Oct 2026** — Checkpointed 3D trajectories and three small items
  (performance chunk 10). (1) The 3D reverse sweep takes a recording at
  any stride: the states between two kept ones are rebuilt by forward
  steps from the lower under the recorded loads (the forward run's own
  iteration), held, their strain as coefficients, while the sweep passes
  through: N / stride + stride states. Fixed stride (David), not revolve.
  Kernels agree with a full recording's to 1.5e-12; ice kernels bitwise.
  At production sizes (one state 21 GB at lmax 256, 98 GB at 512; ~60
  steps of 500 yr over 21-26 kyr) lmax 256 fits a large node, lmax 512
  will want states on disk or a memory-budgeted schedule. The rebuild
  steps forward, so no backward-integration stability limits the stride.
  (2) `Viscoelastic3D::State` carries the converged strain coefficients
  (`strainCoefficients`, `Invalidate()`), so a step starts without
  analysing its grid strain; and `StartStep` forms the first pass's force
  from the grid strain in hand, so the first `Respond` transforms
  nothing: two transforms fewer a step, results changed at round-off.
  (3) Cache robustness: the flavours are an LRU of four, open steps hold
  their own, and the lazy caches fill under one lock, so const methods
  may run concurrently. (4) Huge pages documented (README, Running) and
  set in `run_fortran_gate.sh`. lmax 64, 8 threads: 0.59 s a pass for
  0.70, 0.54 s with huge pages.
- **17 Oct 2026** — Tidy-ups (performance chunk 11). The 1D internal
  variables span each degree's memory window (`MemoryElements`: its
  elements from the first to the last with a relaxing node), no longer
  its whole subset: the 1D state 10-20% smaller (lmax 64-256), every
  result bitwise unchanged. The 1D adjoint sweeps run their per-degree
  loops threaded (the continuous sweep's kernels bitwise, the discrete
  sweep's itau kernel at round-off: each degree's terms are summed before
  joining the kernel, in degree order whatever the thread count); the
  1D kernels example runs 31 s for 48 on 8 threads. The radial-major
  force layout was considered and deferred (~9% of a pass). The AAA
  sweep follows as its own change.
- **17 Oct 2026** — The almost-always-auto sweep (conventions.md): 1375
  explicitly typed locals and loop variables across src, include, tests,
  examples and benchmarks rewritten to `auto`, each checked by the compiler
  to deduce exactly the old type before the checks were removed; every
  output byte-identical. Left explicit: struct members, initialisers of a
  different type (a deliberate conversion), multi-line declarations, and
  loops with non-literal starts.

- **21 Oct 2026** — The E1 tier of the cross-code benchmark hierarchy
  (planned in `~/dev/planning/benchmark-plan.md`, orchestrated from
  `AdGIA/benchmarks/`): `DegreeOperator` gains `gravitating = false`,
  dropping the gravity and potential terms and the phi dofs (U, V alone;
  U at l = 0), on a solid model, l != 1 excluded (a surface traction
  there has a net force against the rigid translations); the unit
  tractions `Traction()` / `TractionTangential()` (`Load(Force)` is g
  times `Traction()` on the full problem); and
  `ComputeTractionNumbers(material, lmax, gravitating)` returning the
  SI surface response per unit traction, mirroring pyslfp's
  `traction_numbers` exactly. Gated by `TestTractionNumbers`: identical
  discrete problem against pyslfp fixtures at 1e-10
  (`make_e1_fixtures.py`, homogeneous and two-solid decks from AdGIA's
  ladder), the degree-0 closed form -a/(3 kappa), reciprocity
  h_v = l(l+1) l_u, and the g-tie to `ComputeLoveNumbers`.
