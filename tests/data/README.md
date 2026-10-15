# Test fixtures

- `prem.simp.vm1` — copied verbatim from `sl3d/data/prem.simp.vm1`
  (da380/sl3d) on 15 Oct 2026: the legacy gia1d/sl3d 1D model format
  (title; `nknot ntau`; rows `r rho kappa[GPa] mu[GPa] log10(viscosity)`,
  0 meaning no viscosity). Used by the legacy-reader tests and, later, for
  benchmark parity with the Fortran codes.

Generated fixtures (planetmodel/pyslfp references) carry a provenance
header naming the generating script under `scripts/`, the source-library
commit and the date.

- `stage4_fortran_visco.txt` — `visccoefs.out` from the original Fortran
  `sl3d/visco_test` (da380/sl3d, driver patched for coefficient output and
  an lmax argument; physics untouched), run 15 Oct 2026 at lmax 32 on
  `prem.simp.vm1`: the analytic lateral viscosity s = s(r) exp(0.3 (Y_2,-2
  + Y_2,2)), a constant zonal load rate, explicit RK2 at dt = 50 yr over
  5 kyr. Rows: output index (t = 1000 yr each), l, m, then Re/Im of
  surface U (m) and phi (m^2/s^2), SI.
  Degree 1 of that fixture reflects the Fortran's unfinished l = 1
  handling (a known loose end in the original, deferred with rotation)
  and is not compared; the gate uses l >= 2 (see
  `benchmarks/fortran_gate.cpp`).
