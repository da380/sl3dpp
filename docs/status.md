# Status and aims

*A dated note, 5 November 2026, on where sl3dpp stands and what it is
for, written as the code moves onto a server for its optimisation pass.
It is the in-repo counterpart of the planning notes kept outside the
repository (`~/dev/planning/` on David's machines: `status-2026-11-03.md`,
`hessians-plan-2026-11-05.md`, `relabelling-sl3dpp-2026-11-04.md`,
`gshtrans-notes-2026-11-04.md`, `apps-plan.md`), which are not
distributed with the code. Update it when the aims change; the dated
records of how each piece was built stay in [docs/dev](dev/README.md).*

## 1. The role of the code

sl3dpp is one of three sea-level / GIA codes developed together:

- **AdGIA**, a 3D finite-element GIA code on MFEM, the main focus of the
  programme, whose forward methods are at the benchmarking stage.
- **sl3dpp**, this code: the spectral reference for AdGIA in 1D, and the
  second code able to carry laterally varying structure, so the only
  cross-check of AdGIA's 3D results.
- **pyslfp**, the released Python elastic sea-level fingerprint package,
  the elastic oracle; stable, and the source of several of this code's
  fixtures.

The programme is a hierarchical forward benchmark (elastic → viscoelastic
loading → the sea-level equation), followed by the adjoints. Tiers E1
(non-gravitating elastic), E2 (gravitating Love numbers) and V1
(viscoelastic Love histories) are done across all three codes; the two
spectral codes agree to about 1e-11, and are the time-domain reference
for AdGIA. Two papers are planned: the forward one (AdGIA's fluid
closure theory and numerics, with a full 3D GIA benchmark between AdGIA
and sl3dpp as its capstone: an ICE-7G history on a PREM-grade Maxwell
deck meshed at its discontinuities, rotation on, a tomography-scaled 3D
viscosity ramped up from a smooth, low-amplitude start), then the
adjoints, with every tier gaining an adjoint rung. Inverse problems are
posed through pygeoinf; the C++ codes are back ends for the core
computations, not hosts for objectives.

## 2. What is built (develop, 5 November 2026)

The README's "What it computes" is current. In the order the pieces were
built (the records are in `docs/dev/adjoints.md` and
`docs/dev/lateral-coupling.md`, sections as cited):

- 1D and 3D viscoelastic evolution with the sea-level equation, rotation,
  frozen or migrating shorelines, initial sea-level calibration; the
  `sl3d` program (`love-numbers`, `fingerprint`, `evolve1d`, `evolve`,
  `kernel1d`, `kernel`, `static`), configs in `docs/sl3d-configs.md`.
- Adjoints, continuous and the exact discrete transpose, 1D and 3D
  (adjoints.md §3–§6, §12).
- Tangent-linear runs and second-order adjoints (Hessian-vector products
  as derivatives of the kernels), 1D and 3D, both methods (§8–§11, §14).
- Material derivatives and kernels in κ and μ, first and second order,
  1D and 3D (§13, §15); the density direction in the time-stepped 1D
  problem, first order (§16).
- The 3D elastic extension: the full static solve under lateral δμ, δκ
  (`ElasticPerturbation3D::SolveLateral`, conjugate gradients
  preconditioned by the 1D solve, after Maitra & Al-Attar 2019 and
  Myhill et al. 2026; loads, body tides, adjoint kernels at the 3D
  model, §13j); lateral moduli in the 3D stepper and in all its
  adjoints (§13k); the step by conjugate gradients as an alternative to
  the fixed point (lateral-coupling.md §5a'', adjoints.md §13n; the
  fixed point stays the default, CG is for strong contrasts).
- Distribution over radial segments of the 3D step, its adjoints, the
  tangent-linear runs, the discrete transposes, the material kernels and
  the lateral solves of the stepper (§17, §13m; `docs/radial-partitioning.md`).
  Gated by threads-as-ranks and under `mpiexec` at 1e-11 to 1e-13.
- Pins: planetmodel ≥ 1.5 (Maxwell deck format), pyslfp ≥ 4; all pyslfp
  fixtures regenerated under 4.0.0.

The suite is 268 tests, green serially; the MPI gates are green under
`mpiexec` on 2–4 ranks.

**Decisions that shape the code** (David, 4–5 November 2026):

- Every feature serves the static elastic problem and the quasi-static
  viscoelastic one alike; the app exposes both (`static` is the static
  3D command). When a viscoelastic feature is added its static
  counterpart goes in the same pass.
- No 3D density. A lateral δρ pipeline was built and removed: with the
  gravity, potential and stratification held radial it is
  hydrostatically inconsistent, and that theory belongs to AdGIA's
  non-hydrostatic work (adjoints.md §13l). The 1D self-consistent radial
  density derivatives stay.
- Lateral moduli live in the solid only; the fluid core's κ has no
  lateral pipeline.
- No BLAS or LAPACK calls inside the OpenMP loops: the banded solves are
  the code's own (`detail/BandSolvers.hpp`), because a threaded BLAS's
  worker pool clashes with the library's threads (the server's OpenBLAS
  crash of `docs/dev/multicore-scaling.md` §1). Keep it so.

## 3. What is not built

- Second order in the density (needs the form's second variation);
  density derivatives in 3D (declined, above).
- The static Born perturbation (`ElasticPerturbation3D`) runs on one
  process; the distributed static route is the stepper on a branchless
  rheology.
- A prescribed tidal-potential history: `DegreeOperator::Tide(l)` is a
  general per-degree channel, but the evolution and the app drive it
  from rotation only.
- Nonlinear (power-law) and Burgers rheologies, wanted as tier-V rungs
  (lateral-coupling.md §7a sketches the power-law step).
- An objective layer (`MeasurementSet` / `Objective` /
  `HessianVectorProduct`, a `sl3d hessian` command) and Python bindings
  for pygeoinf: planned, recorded, not current work (the Hessians plan;
  `docs/dev/pygeoinf-integration.md` is the earlier sketch).
- Particle relabelling (aspherical boundaries by a referential map):
  a think-piece exists, parked; the open design question is the fluid's
  level-border closure under the map.

## 4. Aims, in order

1. **The optimisation pass, on the server.** The derivative and lateral
   machinery was built serially on an 8-core laptop and distributed
   afterwards; its performance has only been measured there (adjoints.md
   §13m; lateral-coupling.md §5a''). On the two-socket node: repeat the
   sweeps of `docs/dev/multicore-scaling.md` on the current code (thread
   placement, one rank per socket, the 3D step's phases by thread count),
   then the lateral pipeline, which is transform-bound (at lmax 64 the
   lateral force is the whole of a 2.5 s CG iteration). The levers,
   recorded in the GSHTrans note: a fused synthesise–weight–analyse
   transform (the memory traffic, not the flops, is the cost), a real
   spin transform producing the m ≥ 0 coefficients directly, a ragged
   batch for the depth-truncated node blocks, and the CG iteration's
   own algebra. GSHTrans changes go upstream with sl3dpp as the first
   consumer. Measure before and after; keep the determinism gates (the
   degree loops are bitwise independent of the thread count).
2. **The 3D elastic benchmark against AdGIA.** `sl3d static` writes the
   exchange-format surface file for a Gaussian cap load and a tidal
   potential on a laterally varying model (`examples/ExampleStatic3D`,
   `configs/static.json`); AdGIA's lateral-perturbation check compares
   the exact heterogeneous response, not just Born. This is the first
   cross-code 3D rung and the model for the later ones.
3. **The 3D viscoelastic rungs (tier V-lat).** Lateral viscosity, then
   lateral moduli, between the two codes, without and then with the
   sea-level equation; the capstone 3D GIA benchmark of §1 at the end.
   Burgers and power-law rheologies join tier V on the way, through
   planetmodel's rheology fields.
4. **The adjoint rungs** of every tier, once the forward ladder is
   complete: AdGIA builds its tangent-linear and second-order machinery
   then, and sl3dpp's existing sweeps are the reference.
5. **The objective layer and bindings**, when inversions through
   pygeoinf need them (§3).

## 5. Working on the server

- Build as the README says; `-DSL3DPP_WITH_MPI=ON` with
  `-DMPI_CXX_COMPILER=<the server's mpicxx>` for the distributed parts,
  `-DSL3DPP_BUILD_BENCHMARKS=ON` for `step_timing`, `distributed_step`
  and the cross-code exchange writers. The siblings are downloaded at
  their pins unless `FETCHCONTENT_SOURCE_DIR_<NAME>` points at a checkout
  (do that for GSHTrans while changing it).
- Threads and placement: README "Threads and where they run" and
  `docs/dev/multicore-scaling.md` §2–§4. One process per socket with the
  socket's threads is the shape the radial partition was built for.
- Do not let a test run take the whole node: a moderate `ctest -j`, a
  capped `OMP_NUM_THREADS`, a few threads per rank under `mpiexec`.
  Timing probes run alone.
- The exchange files of the benchmark programme (`e2_*_sl3dpp.txt` and
  the like) are machine-independent text; the reference values live in
  AdGIA's benchmark tree, not here.
