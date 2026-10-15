# sl3dpp

Forward and adjoint modelling of glacial isostatic adjustment in 1D
elastic and 3D viscoelastic earth models. A C++ port of the original
Fortran codes ([sl3d](https://github.com/da380/sl3d), gia1d) written by
Ophelia Crawford and David Al-Attar, with contributions from Andrew Lloyd
and Frank Syvret.

Relative to the original codes:

- Implicit time stepping of the internal variables (exponential
  trapezoid, second order; backward Euler and exponential Euler
  variants; an embedded adaptive controller). Step sizes are not limited
  by the shortest Maxwell time.
- Rotational feedback: first-order quasi-static polar wander and
  length-of-day changes, including the axial Love numbers.
- The generalised sea level equation with Anderson-accelerated load
  iteration, exact mass conservation, displacement/potential load and
  response channels, frozen and migrating shorelines, and its exact
  adjoint. Elastic sea-level fingerprints as a standalone capability.
- Lateral viscosity in an implicit step, through a within-step iteration
  about a radial reference operator; dealiased generalised
  spherical-harmonic transforms
  ([GSHTrans](https://github.com/da380/GSHTrans)), batched with a BLAS
  kernel.
- Transversely isotropic spheroidal and toroidal operators; per-degree
  systems solved on subsets of one global radial mesh with graded
  resolution.
- Model input through planetmodel deck formats and a reader for the
  legacy `prem.simp.vm1` format. SI units at every public interface.

## References

The theory and its development, in order:

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
