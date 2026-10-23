# sl3dpp conventions

Companion to [design.md](design.md). AdGIA supplies the *naming and layering*
conventions and pyslfp the formulation; the C++ itself is modern C++23
throughout — AdGIA's sometimes old-fashioned idiom is MFEM's doing, not a
model to copy. Items marked **(provisional)** await confirmation.

## Code style

- Formatting: `clang-format` with the repo's `.clang-format`, taken verbatim
  from AdGIA (Google-based, 2-space indent, 80 columns, attached braces).
- Namespace `sl3dpp`, with implementation helpers in `sl3dpp::detail`.
- PascalCase types and methods (`DegreeOperator`, `Factorise()`), MFEM-style
  trailing-underscore private members (`lMax_`, `mesh_`), snake_case locals.
- File names PascalCase matching the principal class (`RadialMesh.hpp` /
  `RadialMesh.cpp`), one principal class per header, under `include/sl3dpp/`
  with non-template implementation in `src/`.
- Doxygen comments on every public entity, citing equations by source —
  `pyslfp_theory.tex §7`, A&T 2014 eq. D76 — rather than restating
  derivations. Comments state constraints and ownership, not narration.

### Modern-C++ idiom (C++23, GSHTrans-style)

- **Value semantics by default.** Classes own their data; copies and moves
  are the defaulted ones where possible. No raw `new`/`delete`; owning
  pointers only as `std::unique_ptr` and only where identity or pimpl truly
  demands it. Non-owning views are `std::span` / `std::string_view`.
- **Concepts over inheritance.** Customisation points are constrained
  templates (NumericConcepts is already in the stack) or `std::function`
  hooks (AdGIA's force-hook / corrector / time-sync pattern) — no virtual
  hierarchies for physics components.
- **Ranges.** Iteration uses `std::ranges` / `std::views` (`zip`,
  `enumerate`, `iota`) rather than index juggling, as GSHTrans does, except
  where an OpenMP pragma needs a plain loop.
- **`Options` structs** are aggregates with member initialisers, taken by
  value and usable with designated initialisers:
  `Solver(mesh, {.tol = 1e-10, .max_iterations = 50})`.
- `enum class` always; `[[nodiscard]]` on factories, solves and queries;
  `constexpr`/`static constexpr` wherever meaningful.
- **Not a template library, and double throughout**:
  the banded solves live on LAPACK's d-routines, so precision templating
  would buy nothing real; `double` is the honest choice. Templates are for
  constraints and small utilities, not a precision axis.
- **Almost-always-auto**: `auto` is the default for locals and loop
  variables, with the type on the right where it needs saying
  (`auto x = Thing{...}`); spell a type only where it changes behaviour
  (a deliberate conversion, a span over an owning return). Existing code
  migrates opportunistically, not in sweeps.
- **Errors:** `std::expected<T, E>` with a small error enum/struct for
  recoverable, caller-decidable failures (parsing a deck, a factorisation
  that may legitimately fail); exceptions (`std::runtime_error` or a thin
  `sl3dpp::Error`) for unrecoverable misuse and IO disasters; `assert` for
  internal invariants. Solver non-convergence is a returned diagnostic, not
  an exception. Messages via `std::format`.
- Toolchain floor is GCC 13 / Clang 18: `std::expected` and `std::format`
  are in; `std::mdspan` is **not** (libstdc++ gained it in GCC 15 — tested
  absent under the machine's g++-14 too). Strided multi-dimensional views
  get a small `detail` view type until the floor rises (swap for
  `std::mdspan` then). Clang 18 under-reports `__cpp_concepts`, which makes
  libstdc++ hide `<expected>`; CMake and `.clangd` carry the standard
  `-D__cpp_concepts=202002L` workaround until the floor is Clang 19.

## Units and non-dimensionalisation

**SI in, SI out — non-dimensional everything inside.** A user never sees
the scales: anything read is non-dimensionalised at the moment it enters
(file readers, SI construction options), anything written or returned to
the user is re-dimensionalised at the moment it leaves, and everything
between components is non-dimensional. Concretely: `EarthModel`'s public
methods take and return SI, converting at the call boundary, while
`EarthModel::Dimensionless` is the non-dimensional view the library's own
components (meshes, operators, steppers) work in without ever converting.

The scheme is planetmodel's `Scales.geophysical`: length scale = the body's
radius `b`, density scale = Earth mean density `ρ̄` (mass scale `ρ̄ b³`),
time scale fixed by `G ρ̄ T² = 1`. Internally `G = 1` and the surface radius
is 1. Constants (`G`, `ρ̄`) are shared with planetmodel exactly
(`Scales.hpp`), so fixture comparisons carry no unit skew.

## Signs and physical conventions

From `pyslfp/docs/theory/pyslfp_theory.tex` (the authoritative statement):

- Gravity `g = ∂_r Φ > 0`; potential perturbation sign conventions as there.
- Sea level `ξ = −u − (φ+ψ)/g + c`.
- Centre-of-mass frame at degree 1 (removes the spheroidal translations).
- Degree-1 toroidal rigid rotations: removed by a convenient gauge (one
  pinned dof per solid region); which one does not matter (`design.md`
  §8).
- Don't measure things that are gauge: measured, benchmarked and tested
  quantities must be well defined, which is also the existence condition
  for the adjoint equations (`design.md` §8). At degree 1 the spheroidal
  translations are fixed by the centre-of-mass frame, a physical frame,
  so quantities relative to it are well defined: radial displacement,
  potential and sea level as the code computes them. The degree-1
  toroidal rotations are fixed by an arbitrary pin, so a horizontal
  displacement at a point is gauge: horizontal observables must be
  rotation-invariant (relative distances are fine, relative
  displacements are not).
- Love-number normalisations and the body triple (b, g, G) consistent with
  the radial model, per `pyslfp_theory.tex`.
- Rotation: first-order quasi-static Liouville per pyslfp §7, ported
  verbatim (including the δ = 0.008 hydrostatic-moment guard).

## Spherical harmonics

GSHTrans's native basis **everywhere inside the library**: complex, fully
normalised, Condon–Shortley phase,
`Y^N_lm = √((2l+1)/4π) d^l_{Nm} e^{imφ}`, forward transform
`∫ conj(Y) f dΩ` — identical to the old Fortran's, which makes comparing
intermediate quantities direct. The real orthonormal no-CS-phase (pyshtools) convention
used by planetmodel/pyslfp exists only in `io/` converters at the file and
benchmark boundary. Real scalar fields use `RealValued` at N = 0; tensor
components are complex with Hermitian symmetry. Products are dealiased via
`ForBand` oversampling (3/2 rule).

## Radial mesh

One global GLL mesh (default ngll = 5) shared by all degrees and all grid
fields. Per-degree systems act on a contiguous DOF subset, truncation radius
snapped to an element boundary; the radius heuristic is an `Options` choice
(`design.md` §3). Memory variables are stored only over viscous
elements.

## Testing

- GoogleTest; one `TestX.cpp` per component, registered via
  `sl3dpp_add_test` in `tests/CMakeLists.txt`.
- Four test kinds, used wherever applicable: analytic (closed-form cases),
  structural (reciprocity, symmetry, dot-product/adjoint identities),
  convergence (measured order), and **fixture comparison** against
  pyslfp/planetmodel/Fortran references.
- Fixtures live in `tests/data/`, generated by scripts in `scripts/`
  (Python, driving the sibling checkouts). Every fixture opens with a
  provenance header: generating script, source-library commit, date,
  tolerances. Fixtures are committed; tests never invoke Python.
- Adjoint code is gated by dot-product tests at machine precision, following
  pyslfp.

## Parallelism

OpenMP only. Parallel axes: degree loops (independent factorisations/solves),
latitude loops (pointwise grid operations), and GSHTrans's own
`Execution::Parallel`. No nested parallelism; classes are not thread-safe
unless documented, but `const` solve/apply methods must be safe to call from
parallel loops.

Threaded by default: the `parallel` options default
on, and the thread count is OpenMP's (`OMP_NUM_THREADS`; 1 for a serial
run). `parallel = false` forces serial, for a caller already inside its own
parallel region. Threaded runs agree with serial ones to round-off, which
is all that is required; the test harness runs each test on one thread
(`tests/CMakeLists.txt`).

## Dependencies

All six sibling libraries (header-only) are pinned by commit in the
top-level `CMakeLists.txt` (`SL3DPP_<NAME>_TAG`); local development via
`FETCHCONTENT_SOURCE_DIR_<NAME>`. Moving a pin is a deliberate, reviewed
change, made with `scripts/bump_dependencies.py`; the libraries themselves
track each other's `main`, and the Upstream workflow builds sl3dpp against
those weekly.

## Test cost

Tests run at small degree (lmax <= 8) and few steps: the gates are
identities and convergence orders, which small problems pin as tightly
as large ones. High-degree or long-duration 3D runs are benchmark or
server work (`benchmarks/`, `scripts/run_fortran_gate.sh`), never suite
tests.
