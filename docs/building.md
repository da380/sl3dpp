# Building sl3dpp

The requirements, the dependencies and how they are fetched and pinned,
the build options, the MPI build, the tests and benchmarks, and using
the library from another CMake project. Running what the build makes is
in [running.md](running.md).

## Requirements

A C++23 compiler (GCC 13+ or Clang 18+), CMake 3.24+, pkg-config, OpenMP,
FFTW3 in all three precisions (double, single and long double), BLAS and
LAPACK, and netCDF-C. On Debian or Ubuntu 24.04, for example:

```
sudo apt install g++ cmake pkg-config libfftw3-dev libopenblas-dev liblapack-dev libnetcdf-dev
```

The first configure downloads the sibling libraries from GitHub, so it needs
network access. The examples' plotting and the ICE-7G download use Python
3.12+ through [poetry](https://python-poetry.org), with the versions of
pyslfp and planetmodel that `examples/python/pyproject.toml` names.

Clang 18 reports a `__cpp_concepts` value that makes libstdc++ hide
`<expected>`; the library target carries the usual workaround
(`-D__cpp_concepts=202002L`) for Clang before 19, for its consumers too.

## Build

```
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

The build type defaults to `Release` when sl3dpp is the project being
built. The build produces the static library `build/libsl3dpp.a`, the
`sl3d` program `build/apps/sl3d`, the tests in `build/tests`, the example
drivers in `build/examples` (run them from there) and, when enabled, the
benchmarks in `build/benchmarks`. Nothing is written into the source tree.

| CMake option | Default | Effect |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | The standard CMake build types |
| `SL3DPP_BUILD_TESTS` | on at top level | The test suite |
| `SL3DPP_BUILD_APPS` | on at top level | The `sl3d` program |
| `SL3DPP_BUILD_EXAMPLES` | on at top level | The example drivers, the synthetic Earths they run on, and the configs copied into `build/examples/configs` |
| `SL3DPP_RUN_EXAMPLES` | `OFF` | Runs every example driver at a small degree as part of the build (the target `sl3dpp_example_outputs`; [examples/README.md](../examples/README.md)) |
| `SL3DPP_BUILD_BENCHMARKS` | `OFF` | `step_timing`, `fortran_gate`, `substructure_timing` and the cross-code exchange writers (below) |
| `SL3DPP_BUILD_DOCS` | `OFF` | The `sl3dpp_docs` target: the API reference and these pages, with Doxygen, in `build/docs/html` |
| `SL3DPP_INSTALL` | `ON` | The install rules and the CMake package |
| `SL3DPP_WITH_MPI` | `OFF` | The MPI-distributed parts (below) |

"At top level" means when sl3dpp is the project being built rather than a
dependency of another, which then gets only the library.

## Dependencies

The sibling libraries, GSHTrans (grids and generalised spherical harmonic
transforms), GaussQuad, Interpolation, FFTWpp, NumericConcepts and
planetmodel (decks, model and surface files, units, JSON configs), are
fetched from GitHub by CMake's FetchContent when the build is first
configured, unless an installed copy is found (`FIND_PACKAGE_ARGS`).
planetmodel is fetched whole and built from its `cpp/` directory. The
tests fetch GoogleTest (v1.15.2) likewise. OpenMP, BLAS and LAPACK are
found by CMake's own modules, and netCDF-C by planetmodel, which looks for
its CMake package first and pkg-config after. What each library is used
for is in [design.md](design.md) §13.

- **Pins.** Each sibling is pinned to a commit (`SL3DPP_<NAME>_PIN` in
  `CMakeLists.txt`), so a commit of sl3dpp always builds against the same
  sources. The pins are plain variables, so an existing build tree follows
  them when they move. `scripts/bump_dependencies.py` moves them, every
  library to its current `main` by default or the ones named to a branch,
  tag or commit; the Upstream workflow (`.github/workflows/upstream.yml`)
  builds and tests against every library's `main` weekly.
- **Overrides.** One build tree can override a pin with the matching cache
  variable, as `-DSL3DPP_GSHTRANS_TAG=main`, say; an empty value returns it
  to the pin.
- **Local checkouts.** To build against working copies of the siblings
  rather than the downloads, set `FETCHCONTENT_SOURCE_DIR_<NAME>` for each
  (`NUMERICCONCEPTS`, `GAUSSQUAD`, `INTERPOLATION`, `FFTWPP`, `GSHTRANS`,
  `PLANETMODEL`; the comment at the top of `CMakeLists.txt` lists them).
  An installed copy found by `FIND_PACKAGE_ARGS` is used before the
  checkout; `-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` makes the build
  take the checkouts and downloads alone.
- **planetmodel 1.6.0.** The `fault_file` seismic term of `sl3d`, and
  the library's `FaultSourcesOf`, read planetmodel's fault file
  (`planetmodel.fault/1`) and lay its slice quadrature; the `"prony"` and
  `"burgers"` deck formats (`PronyFormat`, `BurgersFormat`) are read by
  planetmodel's own readers; and the viscosities and relaxation times of
  a deck are joined by planetmodel's logarithmic kinds. These are the C++
  interfaces of planetmodel's 1.6.0 release. planetmodel states no feature
  macros, so the configure step compiles a probe of them against the
  planetmodel it builds with, at every configure, and defines
  `SL3DPP_PLANETMODEL_1_6` for the library and its users where it
  compiles (the status line says which); `sl3dpp/Fault.hpp` defines
  `SL3DPP_HAVE_FAULT_FILE` where that holds and
  `<planetmodel/faultfile.hpp>` is found. Against an older planetmodel
  the build and its tests pass without them: `sl3d` refuses the term and
  the formats, naming the planetmodel it was built with, and a deck's
  columns are joined by that planetmodel's kinds. Until the pin reaches
  planetmodel 1.6.0, build against a checkout that has it:

  ```sh
  cmake -B build -DFETCHCONTENT_SOURCE_DIR_PLANETMODEL=<checkouts>/planetmodel
  ```

## The MPI build

`-DSL3DPP_WITH_MPI=ON` compiles the MPI-distributed parts
(`Distributed.hpp`: the 3D step and its derivatives over radial segments,
one per rank; `sl3d solve` and `kernel` under `mpiexec`), their tests and
the `distributed_solves` and `distributed_step` benchmarks, and defines
`SL3DPP_HAVE_MPI` for the library and its users. How the problem is split
is in [radial-partitioning.md](radial-partitioning.md), and running over
ranks in [running.md](running.md).

CMake's FindMPI picks the MPI; with several installed, name one by its
compiler wrapper, `-DMPI_CXX_COMPILER=/path/to/mpicxx`. An MPI whose library
directory also holds netCDF, HDF5 or zlib under the system's names (a PETSc
install that downloaded them, say) would shadow the system's at run time:
point FindMPI at the MPI library alone instead, with
`-DMPI_CXX_SKIP_MPICXX=ON -DMPI_CXX_INCLUDE_PATH=<dir of mpi.h>
-DMPI_CXX_LIB_NAMES=mpi -DMPI_mpi_LIBRARY=<path of libmpi.so>
-DMPIEXEC_EXECUTABLE=<its mpiexec>`, where the two directories hold the MPI
headers and library and nothing else (symbolic links will do).

## Tests

`ctest --test-dir build` runs the unit and comparison tests, including
fixtures generated from pyslfp and the Fortran sl3d (`tests/data`). Each
test runs on one OpenMP thread unless it sets its own count, so `ctest -j`
does not oversubscribe the machine. With `SL3DPP_WITH_MPI` the MPI tests run under `mpiexec`:
`TestDistributed` on 2, 3 and 4 ranks and `TestAppsDistributed` on 2 and
3, one thread per rank. The testing conventions are in
[conventions.md](conventions.md).

## Benchmarks

With `-DSL3DPP_BUILD_BENCHMARKS=ON`:

- `step_timing <deck> <lmax> [steps] [repeats] [loop|matrix] [gauss|gll]
  [degrees]` times each phase of a 3D step and reports the peak memory;
  `steps` 0 times the phases alone (the quick comparison), and `degrees`
  (such as `32,48,64`) cuts the transforms with depth
  ([running.md](running.md) has its use).
- `fortran_gate` compares a 3D lateral-viscosity run with the Fortran sl3d;
  `scripts/run_fortran_gate.sh` builds both codes and runs the comparison
  ([lateral-coupling.md](lateral-coupling.md) §5b).
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
  `mpiexec -n <ranks> distributed_solves <deck> <lmax> [repeats]` (with
  `SL3DPP_WITH_MPI`) the same solves distributed over ranks
  ([radial-partitioning.md](radial-partitioning.md)).
- `mpiexec -n <ranks> distributed_step <deck> <lmax> [repeats] [degrees]`
  (with `SL3DPP_WITH_MPI`) times the phases of the 3D step distributed
  over ranks, one radial segment each, with OpenMP threads inside: run it
  as ranks x threads over a node (`OMP_NUM_THREADS=4 mpiexec -n 2
  -bind-to core:4 ...` with MPICH).

## Documentation

`-DSL3DPP_BUILD_DOCS=ON` adds the target `sl3dpp_docs`, which builds the
API reference and the pages under `docs/` and `examples/` with Doxygen
into `build/docs/html`, the README as the main page
(`cmake --build build --target sl3dpp_docs`). The same pages are
published from `main` at <https://da380.github.io/sl3dpp/>.

## Using it from CMake

`cmake --install build --prefix <prefix>` installs the library, its headers,
the `sl3d` program and a CMake package, with the sibling libraries' own
packages beside it (`-DSL3DPP_BUILD_APPS=OFF` leaves the program out of a
build). A code then links the one target either way:

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
among them, so a code can read and write planetmodel files beside it
([library.md](library.md)); with `SL3DPP_WITH_MPI` it carries MPI and
`SL3DPP_HAVE_MPI` too. An installed package satisfies a request for its
own minor version only (before 1.0 a minor release may change the API).
With a planetmodel that has no
install rules, the library cannot be exported, and the install holds the
`sl3d` program alone, the configure step saying so.
