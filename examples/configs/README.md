# sl3d example configs

A config or more per `sl3d` command, each a working example of the program's
JSON input, made of the blocks the commands share (the members are
documented in `docs/sl3d-configs.md`). The
build copies this directory to `build/examples/configs` when the files are
missing and never overwrites them, so the copies there are yours to edit
and experiment with; delete one (or the directory) and reconfigure to
restore the original. Run the configs from there, where their relative
paths reach the examples' inputs in `../data/` (`synthetic_earth.nc` and
`synthetic_moduli.nc`, written by the build, `prem8.elastic.deck`,
`prem8.gia.deck` and `viscosity_log.nc`) and each config's outputs land in
`out/<config>/`, `<config>` being its file name without `.json`:

```sh
cd build/examples/configs
./run_all.sh                       # every config, stopping at a failure
../../apps/sl3d sea-level sea_level_1d.json   # or one at a time
```

The synthetic configs need no data and run in about 40 s together on four
threads (`OMP_NUM_THREADS=4`); CI runs them with `run_all.sh`. They are
sized for that, so their degrees are low; raise `lmax` and `nodes` for
sharper results. `poetry run python plot_app.py`, from `build/examples`,
draws each config's figures from `out/<config>/` into
`figures/sl3d/<config>/` (examples/README.md lists them).

## The synthetic Earth

- **`love_numbers.json`** — the load and tidal Love numbers of the prem8
  deck to degree 256: `out/love_numbers/love_numbers.txt`.
- **`sea_level_fingerprint.json`** — `sea-level` without a `times` block:
  the elastic sea-level fingerprint of melting a tenth of the synthetic
  Greenland ice, at lmax 128, with full rotational feedback:
  `out/sea_level_fingerprint/sea_level_fingerprint.nc`.
- **`sea_level_1d.json`** — `sea-level` with a `times` block: the synthetic
  deglaciation from 21 ka to the present through the 1D viscoelastic Earth
  of `../data/prem8.gia.deck`, at lmax 32 in quarter-kiloyear steps, with
  full rotational feedback (the moments of inertia from the model's
  hydrostatic limit, which the full feedback needs):
  `out/sea_level_1d/sea_level_1d.nc`. It also writes checkpoints every six
  nodes to `out/sea_level_1d/checkpoints`, so a rerun resumes rather than
  starting again; a rerun after the config or an input has changed is
  refused instead, until `"restart": false` is set or the directory removed.
- **`sea_level_1d_calibrated.json`** — that deglaciation with migrating
  shorelines, its initial sea level calibrated against the synthetic Earth's
  present day rather than read at 21 ka (the device a thickness-only ice
  history needs), by the classical update in a few runs (the `ocean` block):
  `out/sea_level_1d_calibrated/sea_level_1d_calibrated.nc`, which also holds
  the calibrated `initial_sea_level` and the `ocean_function` at every node.
  The marine margins of the synthetic ice sheets flood as they melt.
- **`sea_level_3d.json`** — the same deglaciation with the laterally varying
  viscosity of `../data/viscosity_log.nc` (a planetmodel model file on the
  same body, a zero-mean log perturbation about 1e21 Pa s), at lmax 16 in
  half-kiloyear steps: `out/sea_level_3d/sea_level_3d.nc`. The `model`
  block's viscosity file is what makes the run 3D; the deck's viscosity
  column is not used, the model file's is.
- **`sea_level_3d_calibrated.json`** — that 3D run with migrating shorelines
  and its initial sea level calibrated against the present day, as
  `sea_level_1d_calibrated.json` is in 1D:
  `out/sea_level_3d_calibrated/sea_level_3d_calibrated.nc`. Each pass of the
  3D fixed point solves the nonlinear sea level equation; the calibration
  takes four runs (about 20 s on 8 threads).
- **`kernel1d.json`** — adjoint kernels in 1D of five measurements of that
  deglaciation (its forward run read from the same blocks as
  `sea_level_1d.json`'s): the present sea level averaged over Hudson Bay,
  the sea level at a far-field point at 10 ka, the length of day (the axial
  rotation), the degree-2 zonal Stokes coefficient J2, and the decay time of
  the Hudson Bay series over the last 7 kyr. Each gets its ice-history
  kernels and its radial viscosity kernel: `out/kernel1d/`.
- **`kernel.json`** — the 3D counterpart for the Hudson Bay sea level and
  decay time, with the log-viscosity kernels of the 3D viscosity:
  `out/kernel/`.
- **`response.json`** — the instantaneous elastic response of the
  synthetic 3D elastic Earth (`../data/synthetic_moduli.nc`, written by the
  build: the PREM-like moduli with a stiff craton and a soft ridge in the
  upper mantle) to a 2 km ice-equivalent cap over Hudson Bay and a unit
  degree-2 tide, at lmax 16, by conjugate gradients on the radial
  systems: `out/response/response.nc` (the surface displacement, potential
  and the radial model's beside them), `out/response/response.txt` (the
  surface coefficients in the cross-code exchange form) and
  `out/response/response_volume.nc` (the response inside the body, for
  planetmodel's cutaways). The forcing's `load` and `tide` are lists of
  terms, summed.
- **`response_viscoelastic.json`** — the viscoelastic response of the
  synthetic 3D Earth, the lateral viscosity of `../data/viscosity_log.nc`
  with the lateral moduli of `../data/synthetic_moduli.nc`, to the same
  Hudson Bay cap loaded linearly over 6 kyr and removed over the next one
  (its `history_kyr`) and a constant unit degree-2 tide, from 0 to 12 kyr in
  half-kyr steps at lmax 16, by the 3D stepper:
  `out/response_viscoelastic/response_viscoelastic.nc` (the surface response
  at every node, the radial reference model's beside it) and exchange tables
  at four nodes,
  `out/response_viscoelastic/response_viscoelastic_{node}.txt` (the start,
  the peak load, the unloaded state and the end). A `times` block is what
  makes a `response` viscoelastic; it takes about 5 s on four threads.
- **`response1d_disk.json`** — a disk load of 8 degrees over Hudson Bay, the
  mass of about 2 km of ice, held for 10 kyr and removed over the next half,
  on the radial viscoelastic Earth of `../data/prem8.gia.deck` (no lateral
  files, so the 1D stepper), from 0 to 20 kyr in half-kyr steps at lmax 64:
  `out/response1d_disk/response1d_disk.nc`. The disk is the exact expansion
  to lmax, not a sampled step. The figure shows the displacement against
  distance from the centre, the peripheral bulge growing under the load and
  the uplift after it; about 5 s on four threads.
- **`response1d_tide.json`** — a degree-2 tidal potential switched on and
  held on the same radial Earth, at unevenly spaced nodes out to 500 kyr
  (`times.at_kyr`), with an exchange table at every node:
  `out/response1d_tide/response1d_tide.nc` and
  `out/response1d_tide/response1d_tide_{node}.txt`. The coefficients over
  the applied potential are the tidal Love numbers through time, from the
  elastic values at the start towards the fluid limit, which the figure
  draws; a fraction of a second.

`../data/prem8.gia.deck` is `../data/prem8.elastic.deck` in planetmodel's
GIA format, with a Maxwell mantle of 1e21 Pa s (the solid layers between
3480 and 6151 km) and `nan` elsewhere: the radial runs through time take
their viscosity from this column.

## ICE-7G

`sea_level_fingerprint_ice7g.json` (10% of Greenland's ice, lmax 256),
`sea_level_1d_ice7g.json` (26 ka to the present at lmax 64, a node on every
stored ICE-7G date) and `sea_level_1d_ice7g_calibrated.json` (that run with
migrating shorelines and its initial sea level calibrated against the
present day: the misfit falls from about 150 m to a few millimetres in five
runs, as pyslfp's tutorial 10 reports, and about 45 s) are the same runs on
ICE-7G. They read
`../data/ice7g.nc`, which `python get_ice7g.py` writes from `build/examples`
(see `examples/README.md`); `run_all.sh` runs them once that file exists
and skips them otherwise.
