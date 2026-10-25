# sl3dpp

[![CI](https://github.com/da380/sl3dpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/da380/sl3dpp/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-da380.github.io%2Fsl3dpp-blue)](https://da380.github.io/sl3dpp/)
[![Licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue)](https://github.com/da380/sl3dpp/blob/main/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)](https://en.cppreference.com/w/cpp/23)

sl3dpp models glacial isostatic adjustment (GIA) and sea-level change on a
self-gravitating, rotating, compressible viscoelastic Earth, spherically
symmetric (1D) or with laterally varying viscosity (3D), forward and
adjoint. It comes as a program, `sl3d`, which runs models from JSON configs
without any C++, and as the C++23 library behind it, for codes that need
more. It succeeds the Fortran codes sl3d and gia1d (both in
[da380/sl3d](https://github.com/da380/sl3d)), written by Ophelia Crawford
and David Al-Attar with contributions from Andrew Lloyd and Frank Syvret.

Developed partially funded by the grant NE/V010433/1, NSFGEO-NERC: Adjoint tomography
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
  coupled driver runs an ice history through the sea-level equation, with
  frozen or migrating shorelines.
- **The initial sea level.** Calibrated so that a run reaches a given final
  (present-day) sea level: in one linear step under frozen shorelines, or
  iteratively under migrating ones, by the classical update or the adjoint
  gradient.
- **3D viscoelastic evolution.** Viscosity varying laterally as
  η(r, θ, φ) = η̄(r) exp(δ(r, θ, φ)) about a radial reference. The lateral
  term is applied pseudospectrally with dealiased generalised spherical
  harmonic transforms ([GSHTrans](https://github.com/da380/GSHTrans)), and
  each implicit step solves a fixed point about the reference (Picard,
  Anderson-accelerated where it converges slowly). A coupled driver runs an
  ice history through the sea-level equation, with frozen or migrating
  shorelines.
- **Sensitivity kernels.** Kernels of sea-level, displacement, potential and
  rotation measurements with respect to the ice history (as load duals,
  densities in time, and rate kernels) and to viscosity (the inverse
  relaxation time of each branch, and in 3D the log-viscosity field), by the
  continuous adjoint (1D and 3D) or the exact discrete transpose (1D). The
  forward run can be recorded at any stride; the reverse sweep rebuilds the
  states in between.

## Getting started

### Requirements

A C++23 compiler (GCC 13+ or Clang 18+), CMake 3.24+, pkg-config, OpenMP,
FFTW3 in all three precisions (double, single and long double), BLAS and
LAPACK, and netCDF-C. On Debian or Ubuntu 24.04, for example:

```
sudo apt install g++ cmake pkg-config libfftw3-dev libopenblas-dev liblapack-dev libnetcdf-dev
```

The first configure downloads the sibling libraries from GitHub, so it needs
network access. The examples' plotting and the ICE-7G download use Python
3.12+ through [poetry](https://python-poetry.org).

### Build

```
cmake -S . -B build
cmake --build build -j
```

This builds the `sl3d` program (`build/apps/sl3d`), the library, the
examples and the tests (`ctest --test-dir build` runs them); the options are
under *Building in detail*, below.

### Run a first model

The build sets up a config for every `sl3d` command, on a synthetic Earth it
writes itself, so nothing needs downloading:

```
cd build/examples/configs
../../apps/sl3d fingerprint fingerprint.json   # one command
./run_all.sh                                   # every command in turn
```

The outputs land in `out/`. The configs there are yours to edit — the build
copies them only when they are missing and never overwrites them — so change
the degree, the dates, the ice change or the measurements and run again. To
draw the outputs, and to run on the real ice history (ICE-7G, through
[pyslfp](https://github.com/da380/pyslfp)):

```
cd build/examples
poetry install --no-root                  # the examples' Python environment
poetry run python plot_app.py             # a figure per command
poetry run python get_ice7g.py            # writes ice7g.nc (about 24 MB the first time)
(cd configs && ./run_all.sh)              # now also runs the *_ice7g.json configs
```

[examples/configs/README.md](examples/configs/README.md) says what each
config does.

## The sl3d program

`build/apps/sl3d` (installed by `cmake --install`) runs the library's
common capabilities from JSON configs:

```
sl3d love-numbers run.json     # the SI Love-number table of a radial model
sl3d fingerprint run.json      # an elastic fingerprint on a present-day state
sl3d evolve run.json           # a 3D (lateral-viscosity) GIA run, restartable
sl3d evolve1d run.json         # its 1D counterpart
sl3d kernel run.json           # adjoint kernels of configured measurements
                               #   ("trajectory": {"directory": ...} spills
                               #    the recorded states to disk for large lmax)
sl3d kernel1d run.json         # its 1D counterpart
```

Every config member is documented in the configuration reference,
[docs/sl3d-configs.md](docs/sl3d-configs.md).

### Configs

Configs are strict JSON read with planetmodel's reader: an unknown or
misspelt member is refused naming the file, the member and the fault,
and relative paths resolve against the config's directory. A
fingerprint's present-day state is a planetmodel surface file holding
`ice_thickness` and `topography` (pyslfp's `export_ice_history` writes
one from ICE-5G/6G/7G), and the fingerprint comes back as a surface
file of SI fields with the angular-velocity perturbation in its
metadata. For example:

```json
{
  "model": "earth.deck",
  "lmax": 256,
  "state": {"file": "ice7g.nc", "date_kyr_bp": 0.0},
  "change": {"type": "cap_fraction", "latitude": 72, "longitude": 320,
             "width_deg": 16, "fraction": -0.1},
  "rotation": "full",
  "output": "fingerprint.nc"
}
```

### Inputs

- **Earth models** are tabulated decks: planetmodel's GIA format (with a
  viscosity column, which the 1D commands use), MINEOS, or a six-column
  elastic table.
- **Present-day states and ice histories** are planetmodel surface files
  holding `ice_thickness` and `topography`. pyslfp's `export_ice_history`
  writes them from ICE-5G, ICE-6G or ICE-7G (`get_ice7g.py` in the examples
  does so), and any code can write one with planetmodel.
- **3D viscosity**, for `evolve` and `kernel`, is a planetmodel model file
  holding a `viscosity` field, as planetmodel's conversion of a tomography
  model writes it.

### Outputs

`love-numbers` writes a plain-text table; `fingerprint`, `evolve` and
`evolve1d` write planetmodel surface files of SI fields, with the angular
velocity in their metadata; `kernel` and `kernel1d` write a directory per
run: the ice-history kernels of each measurement as surface files, its
viscosity kernels (a table in 1D, a planetmodel model file in 3D), and the
measured values. Every output records the program, its version and the
config that made it, and planetmodel's Python readers read the surface and
model files (see below).

### Long runs

- A `checkpoint` section makes `evolve` and `evolve1d` write their state as
  they go and resume from it when rerun; with `max_walltime_min` they stop
  cleanly before a batch system's time limit, so a chain of jobs running
  one config completes the run.
- `kernel` can spill the recorded forward states to disk (`trajectory`),
  the lever on memory at large degree.
- *Running*, below, has the threads and memory a run needs.

`-DSL3DPP_BUILD_APPS=OFF` leaves the program out of a build.

## Working with planetmodel and pyslfp

sl3dpp reads and writes the file formats of
[planetmodel](https://github.com/da380/planetmodel), and its formulation is
[pyslfp](https://github.com/da380/pyslfp)'s, so the three work as one
toolchain: models and data prepared in Python, run by `sl3d` or the
library, and the results read back in Python. Both are on PyPI
(`pip install "planetmodel[netcdf]" pyslfp`); the examples' poetry
environment has them.

- **Preparing inputs in Python.**
  - *Earth models:* planetmodel builds radial models and writes them as
    decks (`planetmodel.deck.write_deck`).
  - *3D viscosity:* planetmodel builds laterally varying models — from a
    seismic tomography model through its conversion laws
    (`planetmodel.conversion`), say — and writes them as model files
    (`planetmodel.modelfile.write_model`), which `sl3d evolve` and
    `io::ModelViscosity` read.
  - *Ice histories and present-day states:* pyslfp's `export_ice_history`
    (in `pyslfp.ice`) writes ICE-5G, ICE-6G or ICE-7G as surface files, and
    `planetmodel.surfacefile.write_surface` writes any field history of
    one's own (a synthetic load, say).
- **Reading the outputs in Python.** Every `sl3d` output is a planetmodel
  file or a plain table: `planetmodel.surfacefile.read_surface` reads the
  fingerprint and evolution outputs as histories in time, and
  `planetmodel.modelfile.read_model` the 3D kernels, for analysis and
  plotting (`examples/python/plot_app.py` does both).
- **In C++.** Linking `sl3dpp::sl3dpp` brings planetmodel's headers with it,
  so a code can read and write the same files beside the library
  (`<planetmodel/surfacefile.hpp>`, `<planetmodel/modelfile.hpp>`,
  `<planetmodel/deck.hpp>`). The library's `io/` classes connect them to its
  own types: `io::FieldHistory` serves a surface file's fields on the solver
  grid at any time, `io::ModelViscosity` turns a model file's viscosity into
  the radial reference and the lateral perturbation, `io::WriteKernels`
  writes 3D kernels as a model file, and `Deck::Read` reads decks.
  `SeaLevelFromTopography` builds a state's sea level from an ICE-xG
  topography and ice thickness. `examples/MakeSyntheticEarth.cpp` writes a
  surface file, and `examples/Common.hpp` reads one.
- **pyslfp as the reference.** sl3dpp's Love numbers, fingerprints, 1D
  evolution and kernels are tested against pyslfp (`tests/data`). Exchange
  quantities between the two in SI, through files: their internal units
  differ.

## Running

### Threads and where they run

sl3dpp is threaded with OpenMP throughout, and how the threads are placed
matters as much as how many there are. `lscpu` (on Linux) gives a
machine's layout: its sockets, cores per socket, threads per core and NUMA
nodes.

- **One thread per physical core.** OpenMP starts one thread per logical
  CPU by default, which on a machine with simultaneous multithreading
  (hyperthreading: two logical CPUs to a core) is two threads per core. The
  two share the core's floating-point units, so for sl3dpp the second
  gains little and adds to every synchronisation. Set `OMP_NUM_THREADS` to
  the number of physical cores, or fewer.
- **Pin the threads.** Unpinned threads are moved between cores by the
  operating system, and on a machine with several sockets (NUMA nodes)
  away from the memory they first touched, which then crosses between
  sockets on every access. Pinning keeps each thread on one core:

  ```
  export OMP_NUM_THREADS=<physical cores> OMP_PLACES=cores OMP_PROC_BIND=close
  ```

  On a two-socket node this made an 8-thread 3D run 2.7 times faster. The
  variables apply to every OpenMP program started from that shell, so set
  them per run, or in a job script, where other codes want other settings.
- **One NUMA node per run.** `close` puts threads on neighbouring cores,
  filling one socket before the next; `spread` divides them between the
  sockets. sl3dpp's large arrays live in the memory of whichever socket
  first touched them, so threads on another socket reach across for them,
  and at present a 3D run is fastest on the cores of one NUMA node (`lscpu`
  lists them; usually one socket). On a two-socket node with 32 cores per
  socket, a 3D step at lmax 128 ran 1.4 times faster on one socket's 32
  cores (`close`) than on all 64, and `close` beat `spread` at 16 and 32
  threads. To use a whole node, run one job per NUMA node, each with its
  memory kept there too (when the job fits in that node's memory):

  ```
  export OMP_NUM_THREADS=<cores per NUMA node> OMP_PLACES=cores OMP_PROC_BIND=close
  numactl --cpunodebind=0 --membind=0 sl3d evolve first.json &
  numactl --cpunodebind=1 --membind=1 sl3d evolve second.json &
  ```
- **Small problems want fewer threads.** The parallel loops run over
  degrees (lmax + 1 of them), latitude rows and radial nodes, and below a
  degree of a few dozen there is too little work to share among the cores
  of a large node: more threads then make a run slower, not faster.
- **Cutting the transforms with depth.** `Viscoelastic3D::Options::
  transform_degrees` (sl3d's `coupled.transform_degrees`) transforms each
  memory node only to the degrees its solution carries there under the
  degree truncation. At lmax 128, the list 32, 48, ..., 112 made a step
  1.5 times faster and cut its peak memory from 34 to 24 GB, with the
  surface response unchanged to round-off under a smooth viscosity
  pattern; a deep viscosity rougher than the solution there would want
  the listed degrees higher. Below about lmax 30 every mantle node needs
  lmax and nothing is cut.
- **Measuring.** `step_timing` (built with `-DSL3DPP_BUILD_BENCHMARKS=ON`)
  times each phase of a 3D step and reports the peak memory. Run it at the
  production degree over a few thread counts and both bindings, and take
  the fastest per-pass time (it needs about 34 GB at lmax 128):

  ```
  for bind in close spread; do for threads in 16 32 64; do
    OMP_NUM_THREADS=$threads OMP_PLACES=cores OMP_PROC_BIND=$bind \
      build/benchmarks/step_timing tests/data/prem.simp.vm1 128 2 2 \
      > step_${bind}_${threads}.txt
  done; done
  grep -H "per pass" step_*.txt
  ```

- **Reproducibility.** One-thread runs are bitwise reproducible; threaded
  runs agree with them to round-off.
- **BLAS.** Any BLAS works on any number of cores, OpenBLAS's pthread and
  OpenMP builds alike, with no environment variables to set: the library's
  threaded loops make no BLAS calls (the per-degree banded solves are its
  own).

### Memory

- **Size.** In 3D each relaxing branch holds its internal variables on a
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

  The 3D step reuses its large arrays from step to step, so they are
  faulted in once per run: at lmax 64 on 8 threads huge pages now make the
  3D run about 1.5% faster (it was 10% when every step allocated afresh),
  with unchanged results.
- **Memory points.** The internal variables live at the Gauss points of each
  radial element by default; `memory_points = MemoryPointSet::Gll` places
  them at the GLL nodes, as the Fortran codes and pyslfp do, for direct
  comparison.

## Examples

The examples are built into `build/examples` and run from there. They need
no data: the build writes a synthetic Earth (`synthetic_earth.nc`, a toy
planet over a glacial cycle from 21 ka) for them, and each also runs on
ICE-7G (`--state ice7g.nc`, or the `*_ice7g.json` configs).

- **The `sl3d` configs** in `examples/configs`, one per command, as in
  *Run a first model*, above.
- **Seven C++ drivers** of the library, each with a plotting script:

| Driver | What it shows |
| --- | --- |
| `ExampleLoveNumbers` | Load and tidal Love numbers of a radial model |
| `ExampleFingerprint` | The elastic sea-level fingerprint of a Greenland melt |
| `ExampleEvolution` | 1D GIA under a West Antarctic melt and hold |
| `ExampleLateralViscosity` | A 3D run with lateral viscosity against its radial twin |
| `ExampleModelViscosity` | The deglaciation on a 3D viscosity read from a planetmodel model file |
| `ExampleIceKernels` | Ice-history and viscosity kernels of a tide gauge near the melting Greenland ice (1D) |
| `ExampleViscosityKernels3D` | 3D log-viscosity kernels of a sea-level and a decay-time measurement, written as a model file |

See [examples/README.md](examples/README.md) for the inputs, the Python
environment, the ICE-7G route, the run times and the outputs.

## The library

Each `sl3d` command is a few calls into the library, which offers more: any
rheology, measurement or load history a code can build, the operators and
solvers on their own, and the adjoints of all of them.

### Overview

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
| `Viscoelastic.hpp`, `Evolution.hpp` | The 1D viscoelastic operator and its time steps; the coupled 1D evolution, frozen or migrating shorelines |
| `InitialSeaLevel.hpp` | The final sea level misfit and its gradient; the initial sea level calibrated against a final one |
| `Viscoelastic3D.hpp`, `GridCoupling.hpp`, `Evolution3D.hpp` | The 3D operator and its time steps, the grid coupling, and the coupled 3D evolution, frozen or migrating shorelines |
| `Adjoint.hpp`, `Adjoint3D.hpp` | Adjoint sweeps and kernels, 1D and 3D |
| `Measurements.hpp` | Measurement duals (cap means, point values, expansion coefficients, rotation components) and the decay time of a site's series as a fitted, kernel-ready datum |

### Using the library

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

### Using it from CMake

`cmake --install build --prefix <prefix>` installs the library, its headers,
the `sl3d` program and a CMake package, with the sibling libraries' own
packages beside it. A code then links the one target either way:

```cmake
find_package(sl3dpp REQUIRED)      # an installed copy, on CMAKE_PREFIX_PATH
# or
include(FetchContent)
FetchContent_Declare(sl3dpp
  GIT_REPOSITORY https://github.com/da380/sl3dpp.git
  GIT_TAG        <release tag or commit>)
FetchContent_MakeAvailable(sl3dpp)

target_link_libraries(my_code PRIVATE sl3dpp::sl3dpp)
```

The target carries the C++23 requirement and its dependencies, planetmodel
among them, so a code can read and write planetmodel files beside it.

## Building in detail

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
| `SL3DPP_BUILD_TESTS` | on at top level | The test suite |
| `SL3DPP_BUILD_APPS` | on at top level | The `sl3d` program |
| `SL3DPP_BUILD_EXAMPLES` | on at top level | The example drivers |
| `SL3DPP_BUILD_BENCHMARKS` | `OFF` | `step_timing`, `fortran_gate` and the cross-code exchange writers |
| `SL3DPP_BUILD_DOCS` | `OFF` | The `sl3dpp_docs` target: the API reference, with Doxygen, in `build/docs/html` |
| `SL3DPP_INSTALL` | `ON` | The install rules and the CMake package |
| `SL3DPP_WITH_MPI` | `OFF` | The MPI-distributed parts (`Distributed.hpp`: the 3D step and adjoint over radial segments, one per rank; `sl3d evolve` and `kernel` under `mpiexec`), their tests (run under `mpiexec`) and the `distributed_solves` and `distributed_step` benchmarks; defines `SL3DPP_HAVE_MPI` for the library and its users |

"At top level" means when sl3dpp is the project being built rather than a
dependency of another, which then gets only the library.

With `SL3DPP_WITH_MPI`, CMake's FindMPI picks the MPI; with several
installed, name one by its compiler wrapper,
`-DMPI_CXX_COMPILER=/path/to/mpicxx`. An MPI whose library directory also
holds netCDF, HDF5 or zlib under the system's names (a PETSc install that
downloaded them, say) would shadow the system's at run time: point FindMPI
at the MPI library alone instead, with
`-DMPI_CXX_SKIP_MPICXX=ON -DMPI_CXX_INCLUDE_PATH=<dir of mpi.h>
-DMPI_CXX_LIB_NAMES=mpi -DMPI_mpi_LIBRARY=<path of libmpi.so>
-DMPIEXEC_EXECUTABLE=<its mpiexec>`, where the two directories hold the MPI
headers and library and nothing else (symbolic links will do).

To build against local checkouts of the sibling libraries rather than the
downloads, set `FETCHCONTENT_SOURCE_DIR_<NAME>` for each (the comment at the
top of `CMakeLists.txt` lists them).

The sibling libraries (GSHTrans, GaussQuad, Interpolation, FFTWpp,
NumericConcepts and planetmodel) and, for the tests, GoogleTest are
downloaded from GitHub when the build is first configured, unless an
installed copy is found. Each sibling is pinned to a commit
(`SL3DPP_<NAME>_PIN` in `CMakeLists.txt`), so a commit of sl3dpp always
builds against the same sources, and an existing build tree follows the pins
when they move; `scripts/bump_dependencies.py` moves them, and CI builds
against the libraries' current `main` weekly. One build tree can override a
pin, as `-DSL3DPP_GSHTRANS_TAG=main`, say; an empty value returns it to the
pin.

## Tests and benchmarks

`ctest --test-dir build` runs the unit and comparison tests, including
fixtures generated from pyslfp and the Fortran sl3d (`tests/data`); each test
runs on one thread.

With `-DSL3DPP_BUILD_BENCHMARKS=ON`:

- `step_timing <deck> <lmax> [steps] [repeats] [loop|matrix] [gauss|gll]
  [degrees]` times each phase of a 3D step and reports the peak memory;
  `steps` 0 times the phases alone (the quick comparison), and `degrees`
  (such as `32,48,64`) cuts the transforms with depth.
- `fortran_gate` compares a 3D lateral-viscosity run with the Fortran sl3d;
  `scripts/run_fortran_gate.sh` builds both codes and runs the comparison.
- `traction_numbers <deck> <lmax> <out.txt> [nongrav|grav]` writes the E1
  traction numbers of a deck as a cross-code exchange file, for the
  benchmark harness in AdGIA's `benchmarks/elastic/nongrav/`.
- `love_numbers <deck> <lmax> <out.txt>` writes the E2 conventional Love
  numbers (load primed, geodetic tidal) the same way, for
  `benchmarks/elastic/grav/`.
- `love_histories <deck> <lmax> <taus> <out.txt> [steps] [dtmax]` writes
  Maxwell load Love-number histories under a Heaviside load (taus in
  seconds per layer, `inf` elastic), for the V1 tier in
  `benchmarks/viscoelastic/love/`.
- `substructure_timing <deck> <lmax> [repeats] [parts...]` times the degree
  solves substructured over radial segments against the direct ones, and
  `mpiexec -n <ranks> distributed_solves <deck> <lmax>` (with
  `SL3DPP_WITH_MPI`) the same solves distributed over ranks
  ([docs/dev/radial-substructuring.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/radial-substructuring.md)).
- `mpiexec -n <ranks> distributed_step <deck> <lmax> [repeats] [degrees]`
  (with `SL3DPP_WITH_MPI`) times the phases of the 3D step distributed
  over ranks, one radial segment each, with OpenMP threads inside: run it
  as ranks x threads over a node (`OMP_NUM_THREADS=4 mpiexec -n 2
  -bind-to core:4 ...` with MPICH).

## Documentation

The API reference and these pages are published at
<https://da380.github.io/sl3dpp/>, built from `main` by Doxygen
(`-DSL3DPP_BUILD_DOCS=ON`, target `sl3dpp_docs`, builds them locally).

- [docs/sl3d-configs.md](docs/sl3d-configs.md): the `sl3d` program's
  configs, command by command.
- [docs/design.md](docs/design.md): what sl3dpp is built on and how it
  works: formulation, discretisation, the adjoints, units, the comparison
  rules for cross-code work, validation and known limitations.
- [docs/radial-partitioning.md](docs/radial-partitioning.md): the 3D
  problem split over radial segments: the substructured degree solves and
  their distributed form, the transforms cut with depth, the balance of
  the cuts and the communication, and the step, adjoint and `sl3d` over
  MPI ranks.
- [docs/conventions.md](docs/conventions.md): units, signs, harmonic
  conventions and code style.

The developer notes in [docs/dev](https://github.com/da380/sl3dpp/blob/main/docs/dev) are dated records kept from
the code's development, not maintained as reference: the derivations and
convergence measurements of [the 3D coupling](https://github.com/da380/sl3dpp/blob/main/docs/dev/lateral-coupling.md)
and [the adjoints](https://github.com/da380/sl3dpp/blob/main/docs/dev/adjoints.md), and a
[walkthrough of one 3D step](https://github.com/da380/sl3dpp/blob/main/docs/dev/walkthrough-3d-step.md) with pointers
into the code and its costs, and the first
[multicore scaling measurements](https://github.com/da380/sl3dpp/blob/main/docs/dev/multicore-scaling.md) with the
plans that follow from them.

## Licence

BSD 3-Clause; see [LICENSE](https://github.com/da380/sl3dpp/blob/main/LICENSE).

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
