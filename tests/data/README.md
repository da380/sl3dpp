# Test fixtures

- `prem.simp.vm1` — copied verbatim from `sl3d/data/prem.simp.vm1`
  (da380/sl3d) on 15 Oct 2026: the legacy gia1d/sl3d 1D model format
  (title; `nknot ntau`; rows `r rho kappa[GPa] mu[GPa] log10(viscosity)`,
  0 meaning no viscosity). Used by the legacy-reader tests and, later, for
  benchmark parity with the Fortran codes.

Generated fixtures (planetmodel/pyslfp references) carry a provenance
header naming the generating script under `scripts/`, the source-library
commit and the date.
