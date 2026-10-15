# sl3dpp

A C++ port of the [sl3d](https://github.com/da380/sl3d) Fortran codes:
forward and adjoint modelling of glacial isostatic adjustment in 1D elastic
and 3D viscoelastic earth models, with sensitivity kernels for viscosity and
ice history. Functionally equivalent to the old codes, but built on the
refined formulations of AdGIA and pyslfp: implicit (exponential trapezoid)
time stepping, rotational feedback, the generalised sea level equation, and
checkpointed discrete adjoints.

The plan, including the division of labour with
[planetmodel](https://github.com/da380/planetmodel) for model formats and
generation, is in [docs/plan.md](docs/plan.md). Work is staged, with each
stage benchmarked against pyslfp, the Fortran codes, or AdGIA (see §6 there).

## Building

Requires GCC 13+ or Clang 18+, CMake ≥ 3.24, FFTW3 (all three precisions),
LAPACK/BLAS, netCDF-C and OpenMP. The sibling header-only libraries
(GSHTrans, GaussQuad, Interpolation, FFTWpp, NumericConcepts) are fetched at
pinned commits, or found if installed:

```
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

To develop against local checkouts instead of the pinned fetches, see the
`FETCHCONTENT_SOURCE_DIR_*` comment at the top of `CMakeLists.txt`.
