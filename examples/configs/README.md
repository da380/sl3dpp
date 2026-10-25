# sl3d example configs

One config per `sl3d` command, each a working example of the program's
JSON input (the members are documented in `docs/sl3d-configs.md`). The
build copies this directory to `build/examples/configs` when the files are
missing and never overwrites them, so the copies there are yours to edit
and experiment with; delete one (or the directory) and reconfigure to
restore the original. Run the configs from there, where their relative
paths reach the inputs beside the example
programs (`../synthetic_earth.nc`, written by the build, `../prem8.elastic.deck`
and `../viscosity_log.nc`) and their outputs land in `out/`:

```sh
cd build/examples/configs
./run_all.sh                       # every config, stopping at a failure
../../apps/sl3d evolve1d evolve1d.json   # or one at a time
```

The synthetic configs need no data and run in about 15 s together on four
threads (`OMP_NUM_THREADS=4`); CI runs them with `run_all.sh`. They are
sized for that, so their degrees are low; raise `lmax` and `nodes` for
sharper results. `poetry run python plot_app.py`, from `build/examples`,
draws a figure per command from `out/`.

## The synthetic Earth

- **`love_numbers.json`** — the load and tidal Love numbers of the prem8
  deck to degree 256: `out/love_numbers.txt`.
- **`fingerprint.json`** — the elastic sea-level fingerprint of melting a
  tenth of the synthetic Greenland ice, at lmax 128, with full rotational
  feedback: `out/fingerprint.nc`.
- **`evolve1d.json`** — the synthetic deglaciation from 21 ka to the
  present through the 1D viscoelastic Earth of `prem8.gia.deck`, at lmax 32
  in quarter-kiloyear steps, with full rotational feedback (the moments of
  inertia from the model's hydrostatic limit, which the full feedback
  needs): `out/evolve1d.nc`. It also writes checkpoints every six nodes to
  `out/checkpoints_evolve1d`, so a rerun resumes rather than starting again;
  a rerun after the config or an input has changed is refused instead,
  until `"restart": false` is set or the directory removed.
- **`evolve1d_calibrated.json`** — that deglaciation with migrating
  shorelines, its initial sea level calibrated against the synthetic
  Earth's present day rather than read at 21 ka (the device a
  thickness-only ice history needs), by the classical update in a few runs:
  `out/evolve1d_calibrated.nc`, which also holds the calibrated
  `initial_sea_level` and the `ocean_function` at every node. The marine
  margins of the synthetic ice sheets flood as they melt.
- **`evolve.json`** — the same deglaciation with the laterally varying
  viscosity of `../viscosity_log.nc` (a planetmodel model file on the same
  body, a zero-mean log perturbation about 1e21 Pa s), at lmax 16 in
  half-kiloyear steps: `out/evolve.nc`. The deck's viscosity column is not
  used; the model file's is.
- **`evolve_calibrated.json`** — that 3D run with migrating shorelines and
  its initial sea level calibrated against the present day, as
  `evolve1d_calibrated.json` is in 1D: `out/evolve_calibrated.nc`. Each
  pass of the 3D fixed point solves the nonlinear sea level equation; the
  calibration takes four runs (about 20 s on 8 threads).
- **`kernel1d.json`** — adjoint kernels in 1D of five measurements of that
  deglaciation: the present sea level averaged over Hudson Bay, the sea
  level at a far-field point at 10 ka, the length of day (the axial
  rotation), the degree-2 zonal Stokes coefficient J2, and the decay time
  of the Hudson Bay series over the last 7 kyr. Each gets its ice-history
  kernels and its radial viscosity kernel: `out/kernels1d/`.
- **`kernel.json`** — the 3D counterpart for the Hudson Bay sea level and
  decay time, with the log-viscosity kernels of the 3D viscosity:
  `out/kernels3d/`.

`prem8.gia.deck` is `../prem8.elastic.deck` in planetmodel's GIA format,
with a Maxwell mantle of 1e21 Pa s (the solid layers between 3480 and
6151 km) and `nan` elsewhere: the 1D commands take their viscosity from
this column.

## ICE-7G

`fingerprint_ice7g.json` (10% of Greenland's ice, lmax 256),
`evolve1d_ice7g.json` (26 ka to the present at lmax 64, a node on every
stored ICE-7G date) and `evolve1d_ice7g_calibrated.json` (that run with
migrating shorelines and its initial sea level calibrated against the
present day: the misfit falls from about 150 m to a few millimetres in five
runs, as pyslfp's tutorial 10 reports, and about 45 s) are the same runs on
ICE-7G. They read
`../ice7g.nc`, which `python get_ice7g.py` writes in `build/examples` (see
`examples/README.md`); `run_all.sh` runs them once that file exists and
skips them otherwise.
