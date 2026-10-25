# Test fixtures

The reference data the tests hold the C++ to: models, references computed
by planetmodel and pyslfp, and output of the original Fortran sl3d.

## Generated fixtures

The scripts in `scripts/` write these, in that directory's poetry
environment (pyslfp and planetmodel as published on PyPI; see
`scripts/pyproject.toml`):

```sh
cd scripts && poetry install --no-root
poetry run python make_love_number_fixtures.py
```

Each generated file carries a header naming its script, the libraries that
computed it (their commits, for the fixtures made from checkouts; their
versions, for those made from PyPI) and the date. Every script reads its
inputs from, and writes into, this directory unless given `--data-dir`: to
regenerate a fixture without replacing the committed one, copy this
directory somewhere, pass that as `--data-dir`, and compare before copying
anything back.

| File | Script | Read by | What it holds |
| --- | --- | --- | --- |
| `prem8.elastic.deck` | `make_love_number_fixtures.py` | most tests, the examples | PREM without its ocean, 8 knots per layer, as a six-column elastic deck (rho, vpv, vsv, vph, vsh, eta) |
| `pyslfp_love_numbers.txt` | `make_love_number_fixtures.py` | `TestFixtures`, `TestSeaLevel` | the generalised Love numbers of `prem8.elastic.deck` to degree 32 |
| `prem8.rho.deck` | `make_gravity_fixtures.py` | `TestFixtures` | PREM's density alone |
| `planetmodel_gravity.txt` | `make_gravity_fixtures.py` | `TestFixtures` | the mass, moment of inertia and gravity of `prem8.rho.deck` |
| `pyslfp_generalised_sea_level.txt` | `make_sea_level_fixtures.py` | `TestFixtures` | a generalised sea-level solve with every input band-limited, so grid-independent (the tight gate) |
| `pyslfp_fingerprint.txt` | `make_sea_level_fixtures.py` | `TestFixtures` | a fingerprint on the analytic state's sharp ocean, grid-sensitive (the loose gate) |
| `pyslfp_stepper.txt` | `make_stepper_fixtures.py` | `TestFixtures` | Love-number histories of a Maxwell mantle under a held load, by both steppers, and the step's effective Love numbers |
| `pyslfp_evolution.txt` | `make_evolution_fixture.py` | `TestEvolution` | the coupled 1D evolution of the analytic state under a linear melt |
| `pyslfp_kernels.txt` | `make_kernel_fixtures.py` | `TestAdjoint` | ice-history (load-dual) and inverse-relaxation-time kernels of that evolution, by the discrete and continuous adjoints |
| `pyslfp_migrating.txt` | `make_migrating_fixture.py` | `TestInitialSeaLevel` | a migrating-shoreline run under a melting marine cap (its initial fields as coefficients, the per-node sea level change and ocean function), the gradient of a final sea level misfit, and both calibrations' misfit histories |
| `decay_time.txt` | `make_decay_fixtures.py` | `TestMeasurements` | the decay-time estimator on a deterministic noisy series |
| `e1_homogeneous.deck`, `e1_two_solid.deck` | `make_e1_fixtures.py` | `TestTractionNumbers` | the solid models of AdGIA's benchmark ladder, full precision |
| `e1_traction_numbers.txt` | `make_e1_fixtures.py` | `TestTractionNumbers` | their traction numbers, without and with gravity |
| `viscosity_log.nc`, `viscosity_mean.nc`, `viscosity_full3d.nc`, `viscosity_radial.nc` | `make_model_viscosity_fixtures.py` | `TestModelViscosity` (and `viscosity_log.nc` `TestApps`) | the `TestViscoelastic3D` body's viscosity as planetmodel model files: a zero-mean log perturbation, one with a lateral mean, stored fully 3D, and radial alone |
| `viscosity_probes_log.txt`, `viscosity_probes_mean.txt`, `viscosity_probes_full3d.txt` | `make_model_viscosity_fixtures.py` | `TestModelViscosity` | planetmodel's own reading of those files at fixed off-grid points |

## From the Fortran sl3d

- `prem.simp.vm1` — copied verbatim from `sl3d/data/prem.simp.vm1`
  (da380/sl3d) on 15 Oct 2026: the legacy gia1d/sl3d 1D model format
  (title; `nknot ntau`; rows `r rho kappa[GPa] mu[GPa] log10(viscosity)`,
  0 meaning no viscosity). Used by the legacy-reader tests and by the
  Fortran gate (`benchmarks/fortran_gate.cpp`).
- `fortran_visco_test.txt` — `visccoefs.out` from the original Fortran
  `sl3d/visco_test` (da380/sl3d, driver patched for coefficient output and
  an lmax argument; physics untouched), run 15 Oct 2026 at lmax 32 on
  `prem.simp.vm1`: the analytic lateral viscosity s = s(r) exp(0.3 (Y_2,-2
  + Y_2,2)), a constant zonal load rate, explicit RK2 at dt = 50 yr over
  5 kyr. Rows: output index (t = 1000 yr each), l, m, then Re/Im of
  surface U (m) and phi (m^2/s^2), SI. No test reads it; it is the record
  of the comparison in `docs/dev/lateral-coupling.md`, which
  `scripts/run_fortran_gate.sh` reruns at any lmax.
  Degree 1 of that fixture reflects the Fortran's unfinished l = 1
  handling (a known loose end in the original, deferred with rotation)
  and is not compared; the gate uses l >= 2 (see
  `benchmarks/fortran_gate.cpp`).
