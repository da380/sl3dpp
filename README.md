# sl3dpp

A C++23 library for forward and adjoint modelling of glacial isostatic
adjustment (GIA) and sea-level change on a self-gravitating, rotating,
compressible viscoelastic Earth: spherically symmetric (1D) or with laterally
varying viscosity (3D). It succeeds the Fortran codes
[sl3d](https://github.com/da380/sl3d) and [gia1d](https://github.com/da380/sl3d) , written by Ophelia Crawford
and David Al-Attar with contributions from Andrew Lloyd and Frank Syvret.

Developed as part of the grant NE/V010433/1, NSFGEO-NERC: Adjoint tomography
of mantle viscosity using deglacial sea level observations.

## What it computes

- **Elastic response.** Load and tidal Love numbers of transversely isotropic
  radial models, from per-degree spheroidal and toroidal systems; elastic
  sea-level fingerprints.
- **The sea-level equation.** The generalised sea-level equation, with
  displacement and potential response channels, frozen or migrating
  shorelines, exact mass conservation, rotational feedback (none, axial, or
  full: polar wander and length-of-day change), and its exact adjoint.
- **1D viscoelastic evolution.** Generalised Maxwell (Prony) rheologies,
  stepped implicitly: exponential trapezoid (second order, the default) or
  backward Euler, with exponential Euler and an adaptive step controller
  also available. Step sizes are not limited by the shortest Maxwell time. A
  coupled driver runs an ice history through the sea-level equation.
- **3D viscoelastic evolution.** Viscosity varying laterally as
  η(r, θ, φ) = η̄(r) exp(δ(r, θ, φ)) about a radial reference. The lateral
  term is applied pseudospectrally with dealiased generalised spherical
  harmonic transforms ([GSHTrans](https://github.com/da380/GSHTrans)), and
  each implicit step solves a fixed point about the reference (Picard,
  Anderson-accelerated where it converges slowly). A coupled driver runs an
  ice history through the sea-level equation.
- **Sensitivity kernels.** Kernels of sea-level, displacement, potential and
  rotation measurements with respect to the ice history (as load duals,
  densities in time, and rate kernels) and to viscosity (the inverse
  relaxation time of each branch, and in 3D the log-viscosity field), by the
  continuous adjoint (1D and 3D) or the exact discrete transpose (1D). The
  forward run can be recorded at any stride; the reverse sweep rebuilds the
  states in between.

## Library overview

| Header | Contents |
| --- | --- |
| `Deck.hpp` | Tabulated radial models: planetmodel decks (`GiaFormat`), mineos (`MineosFormat`), custom column layouts, and the legacy sl3d/gia1d `prem.simp.vm1` format (`Deck::ReadLegacySl3d`) |
| `EarthModel.hpp`, `Scales.hpp` | The splined radial model, its layering and gravity, and non-dimensionalisation |
| `RadialMesh.hpp`, `Material.hpp` | The spectral-element radial mesh (GLL, graded so that each degree uses a subset of it), and the material on its nodes |
| `Rheology.hpp` | Relaxing shear branches: a Maxwell body from a viscosity profile, a model's own viscosity column, or general branches |
| `DegreeOperator.hpp`, `ToroidalOperator.hpp`, `LoveNumbers.hpp` | Per-degree spheroidal and toroidal systems; Love numbers |
| `SurfaceGrid.hpp`, `ShConventions.hpp`, `Geography.hpp` | The Gauss–Legendre surface grid and its transforms; spherical-harmonic conventions; geographic coordinates |
| `Fingerprint.hpp` | Elastic sea-level fingerprints, SI in and out |
| `SeaLevel.hpp` | The sea-level solver (linear, generalised, nonlinear) and its adjoint |
| `Viscoelastic.hpp`, `Evolution.hpp` | The 1D viscoelastic operator and its time steps; the coupled 1D evolution |
| `Viscoelastic3D.hpp`, `GridCoupling.hpp`, `Evolution3D.hpp` | The 3D operator and its time steps, the grid coupling, and the coupled 3D evolution |
| `Adjoint.hpp`, `Adjoint3D.hpp` | Adjoint sweeps and kernels, 1D and 3D |

## Using the library

A 1D run, from a planetmodel GIA deck to sea-level histories:

```cpp
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

// The Earth model: elastic structure and viscosity from the deck (SI).
const auto deck = Deck::Read("earth.deck", GiaFormat()).value();
const auto model = EarthModel::FromDeck(deck).value();
const auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
const auto material = Material::Create(model, mesh).value();
const auto rheology = Rheology::FromModel(material, model).value();

// The viscoelastic operator and the sea-level solver.
const auto op = ViscoelasticOperator(rheology, lmax);
const auto solver = SeaLevelSolver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                                   SeaLevelParameters{});
// The present-day ice thickness and sea level, on solver.Grid().
const auto state = solver.MakeState(iceThickness, seaLevel);

// The evolution under an ice-load history iceLoad(t), sampled at `times`.
auto evolution = SeaLevelEvolution(op, solver, state, {});
const auto snapshots = evolution.Run(times, iceLoad);
// Each snapshot: sea-level change, displacement, potential, angular velocity.
```

The operator and solver are borrowed by the evolution and must outlive it.
The other entry points follow the same pattern:

- **Fingerprints.** `Fingerprint(lmax, ComputeLoveNumbers(material, lmax))`,
  then `SetState(iceThickness, seaLevel)` and `SolveForIceChange(change)`,
  all in SI.
- **3D.** `Viscoelastic3D(rheology, lmax, delta)`, with `delta(r, theta, phi)`
  the log-viscosity perturbation, driven by `SeaLevelEvolution3D` as above.
- **Kernels.** Run the evolution with `{.record_stride = n}`, build a
  `ForwardTrajectory` (or `ForwardTrajectory3D`) from the snapshots, and call
  `AdjointSolver(evolution).IceHistoryKernels(times, measurements, &trajectory)`
  (`AdjointSolver3D` in 3D), with `measurements` the duals of the observed
  quantities, keyed by time node.

Model construction returns `std::expected`; solver failures are reported in
the returned diagnostics or thrown.

### Units and conventions

- Decks are read in SI, and every internal component works in
  non-dimensional units. `Fingerprint` and `LoveNumbers` are SI at their
  interface. The time-dependent drivers (`SeaLevelEvolution`,
  `SeaLevelEvolution3D`, `AdjointSolver`, `AdjointSolver3D`) take and return
  times, loads and fields in the sea-level solver's units: convert with
  `solver.SolverScales()`.
- Spherical harmonics are complex, fully normalised, with the Condon–Shortley
  phase (GSHTrans's convention); `ShConventions.hpp` converts to and from real
  coefficients.
- Fields live on the solver's Gauss–Legendre grid, colatitude-major.
- Details, including sign conventions, are in
  [docs/conventions.md](docs/conventions.md).

## Building

### Requirements

A C++23 compiler (GCC 13+ or Clang 18+), CMake 3.24+, pkg-config, OpenMP,
FFTW3 in all three precisions (double, single and long double), BLAS and
LAPACK, and netCDF-C. On Debian or Ubuntu 24.04, for example:

```
sudo apt install g++ cmake pkg-config libfftw3-dev libopenblas-dev liblapack-dev libnetcdf-dev
```

The header-only dependencies (GSHTrans, GaussQuad, Interpolation, FFTWpp,
NumericConcepts) and, for the tests, GoogleTest are downloaded from GitHub
at pinned commits when the build is first configured, unless an installed
copy is found; the first configure therefore needs network access.

### Configure, build and test

```
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

The build type defaults to `Release`. The build produces the static library
`build/libsl3dpp.a`, the tests in `build/tests`, the example drivers in
`build/examples` (run them from there), and, when enabled, the benchmarks in
`build/benchmarks`. Nothing is written into the source tree.

| CMake option | Default | Effect |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | The standard CMake build types |
| `SL3DPP_BUILD_TESTS` | `ON` | The test suite |
| `SL3DPP_BUILD_EXAMPLES` | `ON` | The example drivers |
| `SL3DPP_BUILD_BENCHMARKS` | `OFF` | `step_timing` and `fortran_gate` |
| `SL3DPP_PYSLFP_DIR` | a sibling `pyslfp` | A pyslfp checkout, from which the build extracts the examples' present-day Earth state (needed by the fingerprint, evolution and kernel examples) |

To build against local checkouts of the header-only dependencies rather than
the pinned downloads, set `FETCHCONTENT_SOURCE_DIR_<NAME>` for each (the
comment at the top of `CMakeLists.txt` lists them).

## Examples

Four drivers, each with a plotting script, are built into `build/examples`
and run from there:

| Driver | What it computes |
| --- | --- |
| `ExampleFingerprint` | The elastic fingerprint of a Greenland melt (lmax 256) |
| `ExampleEvolution` | 1D GIA under a West Antarctic melt and hold (lmax 128) |
| `ExampleLateralViscosity` | A 3D run with lateral viscosity against its radial counterpart |
| `ExampleIceKernels` | Ice-history and viscosity kernels of a tide-gauge measurement |

See [examples/README.md](examples/README.md) for the inputs, the plotting
environment and the outputs.

## Running

- **Threads.** The library runs threaded by default (OpenMP); set
  `OMP_NUM_THREADS`. One-thread runs are bitwise reproducible; threaded runs
  agree to round-off.
- **Memory.** In 3D each relaxing branch holds its internal variables on a
  grid at every radial memory node: one such tensor is 0.7 GB at lmax 64,
  4 GB at 128, 21 GB at 256 and 98 GB at 512, and a step holds several.
  Viscosity kernels need the forward states: recording every `n`-th of `N`
  time nodes holds about `N/n + n` of them during the sweep, fewest near
  `n = √N`.
- **Huge pages.** Fresh memory is faulted in one 4 KB page at a time by the
  allocating thread. Where transparent huge pages are on request only
  (`/sys/kernel/mm/transparent_hugepage/enabled` reads `[madvise]`), glibc
  2.35+ can request them:

  ```
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 OMP_NUM_THREADS=8 ./ExampleLateralViscosity
  ```

  At lmax 64 on 8 threads this makes the 3D run about 10% faster, with
  unchanged results.
- **Memory points.** The internal variables live at the Gauss points of each
  radial element by default; `memory_points = MemoryPointSet::Gll` places
  them at the GLL nodes, as the Fortran codes and pyslfp do, for direct
  comparison.

## Tests and benchmarks

`ctest --test-dir build` runs the unit and comparison tests, including
fixtures generated from pyslfp and the Fortran sl3d (`tests/data`); each test
runs on one thread.

With `-DSL3DPP_BUILD_BENCHMARKS=ON`:

- `step_timing <deck> <lmax> [steps] [repeats] [loop|matrix] [gauss|gll]`
  times each phase of a 3D step and reports the peak memory.
- `fortran_gate` compares a 3D lateral-viscosity run with the Fortran sl3d;
  `scripts/run_fortran_gate.sh` builds both codes and runs the comparison.

## Documentation

- [docs/conventions.md](docs/conventions.md): units, signs, harmonic
  conventions and code style.
- [docs/walkthrough-3d-step.md](docs/walkthrough-3d-step.md): how a 3D step
  is computed, with pointers into the code and its costs.
- [The 3D coupling](docs/stage4.md) and [the adjoints](docs/stage5.md):
  their formulations, with convergence measurements.
- [docs/plan.md](docs/plan.md): design notes and the record of decisions.

## References

- Al-Attar, D. & Tromp, J., 2014. Sensitivity kernels for viscoelastic
  loading based on adjoint methods, *Geophys. J. Int.*, 196(1), 34–77.
- Crawford, O., Al-Attar, D., Tromp, J. & Mitrovica, J.X., 2017. Forward
  and inverse modelling of post-seismic deformation, *Geophys. J. Int.*,
  208(2), 845–876.
- Crawford, O., Al-Attar, D., Tromp, J., Mitrovica, J.X., Austermann, J.
  & Lau, H.C.P., 2018. Quantifying the sensitivity of post-glacial sea
  level change to laterally varying viscosity, *Geophys. J. Int.*,
  214(2), 1324–1363.
- Crawford, O., 2019. *On the Viscoelastic Deformation of the Earth*,
  Ph.D. thesis, University of Cambridge.
- Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. & Lloyd, A.J.,
  2024. Reciprocity and sensitivity kernels for sea level fingerprints,
  *Geophys. J. Int.*, 236(1), 362–378.
- Lloyd, A.J., Crawford, O., Austermann, J., Hoggard, M.J., Richards,
  F.D. & Syvret, F., 2024. GIA imaging of 3D mantle viscosity based on
  palaeo sea level observations — part I: sensitivity kernels for an
  Earth with laterally varying viscosity, *Geophys. J. Int.*, 236(2),
  1139–1171.
- Yu, Z., Al-Attar, D., Syvret, F. & Lloyd, A.J., 2025. Application of
  first- and second-order adjoint methods to glacial isostatic
  adjustment incorporating rotational feedbacks, *Geophys. J. Int.*,
  240(1), 329–348.
