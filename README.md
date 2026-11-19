# sl3dpp

[![CI](https://github.com/da380/sl3dpp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/da380/sl3dpp/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-da380.github.io%2Fsl3dpp-blue)](https://da380.github.io/sl3dpp/)
[![Licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue)](https://github.com/da380/sl3dpp/blob/main/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)](https://en.cppreference.com/w/cpp/23)

sl3dpp models glacial isostatic adjustment (GIA), sea-level change and the
deformation of a self-gravitating, rotating, compressible viscoelastic
Earth, spherically symmetric (1D) or laterally heterogeneous (3D), forward
and adjoint. It comes as a program, `sl3d`, which runs models from JSON
configs without any C++, and as the C++23 library behind it, for codes
that need more. It succeeds the Fortran codes sl3d and gia1d (both in
[da380/sl3d](https://github.com/da380/sl3d)), written by Ophelia Crawford
and David Al-Attar with contributions from Andrew Lloyd and Frank Syvret.

Development partially funded by the grants:
- NE/V010433/1, NSFGEO-NERC: Adjoint tomography
of mantle viscosity using deglacial sea level observations;
- NE/X013804/1: Next Generation Modelling of Glacial Isostatic Adjustment.

## What it computes

- **Two problems.** The static problem, one elastic solve at an instant
  (a sea-level fingerprint, a co-seismic response), and the quasi-static
  problem, stepped through time on generalised Maxwell (Prony) rheologies
  (Maxwell and Burgers bodies among them) by implicit schemes whose step
  is not limited by the Maxwell times.
- **1D or 3D.** A radial model, isotropic or transversely isotropic (with
  its Love numbers), or one with a lateral viscosity η = η̄(r) exp(δ(r, θ, φ)),
  lateral bulk and shear moduli, or both, applied pseudospectrally with
  [GSHTrans](https://github.com/da380/GSHTrans)'s dealiased transforms; a
  3D step, and the static 3D solve, is closed by an Anderson-accelerated
  fixed point or by conjugate gradients, about the radial reference.
- **With an ocean or without.** The generalised sea-level equation from a
  state (the sea level and the ice at one instant), mass conserving, with
  frozen or migrating shorelines and the initial sea level calibrated
  against a present-day one; or the solid body's response alone.
- **Forcings, in any mix.** An ice change, surface loads, a prescribed tide
  and seismic sources (moment tensors, planar faults, and faults given by
  their slip in a planetmodel fault file, static or in time), with rotational
  feedback none, axial or full; with `"gravity": "none"`, the radial
  forward problem without self-gravitation, under surface loads (their
  weight, a normal traction; no degree-1 part) and seismic sources.
- **Derivatives.** Adjoint kernels of sea-level, displacement, potential
  and rotation measurements, exactly discrete (`sl3d kernel`'s default)
  or continuous, 1D and 3D,
  in the ice history, surface loads, a tide, the initial sea level, the
  viscosity (in 3D the log-viscosity field), the moduli κ and μ and the
  sources' moment tensors; the kernels in lateral structure (the log
  viscosity and the moduli as fields) about a radial Earth by the radial
  sweeps alone, with no 3D solve, as about a 3D one by the 3D sweeps;
  the kernels of a run with a calibrated initial sea level as total
  derivatives, the calibration followed (with the kernel in the
  present-day sea level it is held to);
  tangent-linear runs along a direction in any input, given as the
  inputs are (`sl3d tangent`), the forward mode whose transpose the
  discrete kernels are, of a calibrated run the total derivative too;
  and second-order adjoints (Hessian-vector products), which `sl3d
  kernel` writes as the kernels' derivatives along such a direction.
  None is taken in
  the density
  ([docs/adjoints.md](docs/adjoints.md) §13l).
- **At scale.** OpenMP threads; a trajectory store on disk, from which the
  reverse sweeps run without the forward problem; checkpoints; and,
  optionally, the 3D runs over MPI ranks, split over radius: the step and
  its lateral solves, the static problem on lateral moduli, both
  adjoints, the tangent-linear runs and the material kernels
  ([docs/radial-partitioning.md](docs/radial-partitioning.md)).

## Getting started

A C++23 compiler (GCC 13+ or Clang 18+), CMake 3.24+, pkg-config, OpenMP,
FFTW3 (all three precisions), BLAS, LAPACK and netCDF-C; on Debian or
Ubuntu 24.04:

```
sudo apt install g++ cmake pkg-config libfftw3-dev libopenblas-dev liblapack-dev libnetcdf-dev
cmake -S . -B build
cmake --build build -j
```

The first configure downloads the sibling libraries from GitHub
([docs/building.md](docs/building.md) has the options and the MPI build).
The build makes `build/apps/sl3d`, the library, the examples and the tests
(`ctest --test-dir build`), and a config for every `sl3d` command on a
synthetic Earth it writes itself:

```
cd build/examples/configs
../../apps/sl3d solve sea_level_fingerprint.json   # one config
./run_all.sh                                       # every config in turn
```

Each config's outputs land in `out/<config>/`, and the copies there are
yours to edit. Plotting them and running on ICE-7G take the examples'
poetry environment ([examples/README.md](examples/README.md)).

## The sl3d program

```
sl3d love-numbers run.json     # the SI Love-number table of a radial model
sl3d solve run.json            # the static or the quasi-static problem
sl3d kernel run.json           # adjoint kernels of the config's measurements
sl3d tangent run.json          # the run's first-order change along a perturbation
```

A config is strict JSON (an unknown member is refused, naming the file,
the member and the fault) made of blocks, `discretisation`, `model`,
`times`, `forcing`, `ocean`, `solver`, `checkpoint`, `trajectory`,
`output`, `kernels`, `perturbation` and `tangent`, under a required
`problem`. The elastic
fingerprint of a Greenland melt on ICE-7G's present day, for one:

```json
{
  "problem": "static", "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 256},
  "forcing": {"ice_change": [{"cap_fraction": {"latitude": 72, "longitude": 320,
                                               "width_deg": 16, "fraction": -0.1}}]},
  "ocean": {"date_kyr_bp": 0.0, "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "rotation": "full", "output": {"surface": "fingerprint.nc"}
}
```

With `"shorelines": "migrating"` in `ocean` the shorelines follow
flotation; `"quasi_static"` with a `times` block, the state at an end of
the window (`ocean.at`) and an ice history runs GIA through time; without
`ocean` the body has none; a lateral `model.viscosity` or `model.moduli`
file makes the run 3D; a planetmodel prony or burgers deck
(`"format": "prony"`, `"burgers"`), or `model.burgers` on a Maxwell deck,
gives generalised Maxwell and Burgers bodies; a `kernels` block serves
`sl3d kernel`, and a `perturbation` block, a direction in the inputs
written as the inputs are, `sl3d tangent`. Outputs,
planetmodel files and plain tables, record the program, its version and
the config. Every member is in [docs/sl3d-configs.md](docs/sl3d-configs.md),
and threads, memory, MPI and long runs in [docs/running.md](docs/running.md).

## Working with planetmodel and pyslfp

sl3dpp reads and writes [planetmodel](https://github.com/da380/planetmodel)'s
files, and pyslfp builds its Earth models with planetmodel and writes its
ice histories as planetmodel files, so the three are one toolchain
(`pip install "planetmodel[netcdf]" pyslfp`). planetmodel writes decks
(`deck.write_deck`; a model's prony and burgers decks,
`conversion.prony.prony_deck` and `burgers_deck`), 3D models from tomography (`conversion`,
`modelfile.write_model`) and field histories (`surfacefile.write_surface`),
pyslfp's `ice.export_ice_history` writes ICE-5G, 6G or 7G, and
`surfacefile.read_surface` and `modelfile.read_model` read the outputs.
sl3dpp is tested against pyslfp (`tests/data`); their internal units
differ, so exchange quantities in SI, through files.

## Examples and the library

Nine C++ drivers of the library, each with a plotting script, and the
`sl3d` configs run in `build/examples` on the synthetic Earths the build
writes, or on ICE-7G ([examples/README.md](examples/README.md),
[examples/configs/README.md](examples/configs/README.md)). Each `sl3d`
command is a few calls into the library, which offers more: the operators
and solvers on their own, any rheology, measurement or forcing a code can
build, and the derivatives of all of them. [docs/library.md](docs/library.md)
has its headers, a first program, and its units and conventions.

## Documentation

Published, with the API reference, at <https://da380.github.io/sl3dpp/>.

- [docs/sl3d-configs.md](docs/sl3d-configs.md): the `sl3d` configs, command by command.
- [docs/running.md](docs/running.md): threads, memory, MPI ranks and long runs.
- [docs/building.md](docs/building.md): dependencies, options, tests, benchmarks, CMake use.
- [docs/library.md](docs/library.md): the library's headers and how to use them.
- [docs/design.md](docs/design.md): formulation, discretisation, units, validation, limitations.
- [docs/adjoints.md](docs/adjoints.md): the adjoints, tangent-linear runs and material kernels.
- [docs/lateral-coupling.md](docs/lateral-coupling.md): the 3D lateral coupling and its gates.
- [docs/radial-partitioning.md](docs/radial-partitioning.md): the 3D problem over radial segments and MPI ranks.
- [docs/trajectory-store.md](docs/trajectory-store.md): the trajectory store's on-disk format.
- [docs/conventions.md](docs/conventions.md): code style and testing conventions.
- [examples/README.md](examples/README.md), [examples/configs/README.md](examples/configs/README.md): the examples.

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
