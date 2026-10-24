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
- **`evolve.json`** — the same deglaciation with the laterally varying
  viscosity of `../viscosity_log.nc` (a planetmodel model file on the same
  body, a zero-mean log perturbation about 1e21 Pa s), at lmax 16 in
  half-kiloyear steps: `out/evolve.nc`. The deck's viscosity column is not
  used; the model file's is.
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

`fingerprint_ice7g.json` (10% of Greenland's ice, lmax 256) and
`evolve1d_ice7g.json` (26 ka to the present at lmax 64, a node on every
stored ICE-7G date) are the same runs on ICE-7G. They read
`../ice7g.nc`, which `python get_ice7g.py` writes in `build/examples` (see
`examples/README.md`); `run_all.sh` runs them once that file exists and
skips them otherwise.
