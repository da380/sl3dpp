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

The synthetic configs need no data and run in about 45 s together on four
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
- **`sea_level_fingerprint_3d.json`** — that fingerprint on the synthetic
  3D elastic Earth (`../data/synthetic_moduli.nc` on
  `../data/prem8.elastic.deck`, as `response.json`): without a `times`
  block, a moduli file that varies laterally makes the fingerprint 3D,
  the instantaneous response of the 3D sea level problem, its
  zero-length step closed by the fixed point, at lmax 32 (the moduli
  file's own degree):
  `out/sea_level_fingerprint_3d/sea_level_fingerprint_3d.nc`, which also
  holds the radial model's fingerprint of the same melt
  (`*_radial_model`). The stiff craton under North America and the soft
  ridge change the sea level by a few centimetres at most, against a
  near-field fall of about 12 m; about 3 s and 0.9 GB on four threads.
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
- **`response_tide_3d.json`** — the lunar body tide on the synthetic 3D
  elastic Earth (`../data/synthetic_moduli.nc` on
  `../data/prem8.elastic.deck`), at lmax 32:
  `out/response_tide_3d/response_tide_3d.nc` (the 3D and the radial model's
  surface responses) and `out/response_tide_3d/response_tide_3d.txt`. The
  forcing is an instantaneous degree-2 tidal potential with (2,0), (2,1)
  and (2,2) coefficients: the Moon (GM = 4.9028e12 m^3 s^-2) at its mean
  distance d = 384400 km, at declination 20° N over longitude 100° W, whose
  potential at the model's surface (a = 6368 km) has the amplitude
  A = GM a^2 / d^3 = 3.50 m^2 s^-2; its coefficients in the exchange
  table's harmonics are c_2m = -A (4 pi / 5) conj(Y_2m(70°, 260°)), the
  physical potential (-A under the Moon): (2,0) 1.8009, (2,1)
  -0.3793 + 2.1511 i and (2,2) 2.8196 + 1.0263 i m^2 s^-2. It is
  illustrative, one instant of a Moon placed by hand, not an ephemeris,
  and leaves out the Sun. The radial model's bulge reaches 0.21 m under the
  Moon (h2 = 0.604); the lateral moduli change the response by 0.04 mm at
  most, and the 3D model puts 7.0e-5 of its rms into degrees other than 2,
  which the radial model cannot: the lateral coupling, which the figure
  draws by degree. lmax 32 is the moduli file's own degree. The difference
  between the 3D and the radial response changes by 0.5% between lmax 16
  and 32 (0.4% of it lies above degree 16) and by 0.4% between 32 and 48
  (0.06% above degree 32). Beyond degree 15 its spectrum sits at a floor
  of about 1e-5 mm, which is the moduli file's linear lateral interpolation
  onto the coupling's grid, not its structure (under a cubic lateral rule
  the floor is hundreds of times lower). About 3 s and 0.9 GB on four
  threads.
- **`response_tide_oscillating.json`**, with
  **`response_tide_oscillating_short.json`** and
  **`response_tide_oscillating_long.json`** — a degree-2 tide oscillating
  on the radial viscoelastic Earth of `../data/prem8.gia.deck`, whose
  mantle's Maxwell times (viscosity over shear modulus) run from 0.11 to
  0.43 kyr: the (2,0) coefficient of 1 m^2 s^-2 times sin(2 pi t / P),
  with P = 1 kyr (0.1 kyr and 10 kyr in the other two), over six periods
  at 24 nodes a period, at lmax 4:
  `out/<config>/<config>.nc` and one exchange table at the last crest of
  the potential. They are the output of `make_tide_history.py` (in
  `build/examples`), whose amplitude history `history_kyr` has a point at
  every node, where the run sees the forcing. With a unit potential the degree-2 coefficients
  are the Love numbers at the forcing's frequency; over the last cycle at
  P = 1 kyr |h2| = 0.811 lagging the potential by 23.0° and |k2| = 0.409
  lagging by 24.1°, at 0.1 kyr |k2| = 0.301 lagging by 4.0°, and at
  10 kyr |k2| = 0.787 lagging by 12.2°: the lag is largest between the
  elastic and the fluid limits. Against 96 nodes a period, 24 put k2
  within 0.7% and its lag within 0.4°. A fraction of a second each.

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

## The toy planet

These read the decks of a three-layer "toy Mars" (a fluid core, a 1e21
Pa s Maxwell mantle and a 150 km elastic lid; radius 3390 km, surface
gravity 3.66 m s^-2), which `python make_planet_deck.py` writes from
`build/examples` into `../data/planet.gia.deck` and
`../data/planet.elastic.deck`; `run_all.sh` runs them once those exist
and skips them otherwise.

- **`response_tide_planet.json`** — the Sun's tide on the toy planet,
  elastic and radial, at lmax 32:
  `out/response_tide_planet/response_tide_planet.nc` and
  `out/response_tide_planet/response_tide_planet.txt`. The potential is
  the Sun's (GM = 1.32712e20 m^3 s^-2) at Mars's mean distance, 1.524 AU,
  at declination 25° N (about Mars's obliquity) over longitude 0°:
  A = 0.1287 m^2 s^-2 at the surface, so (2,0) 0.04735, (2,1) 0.09571 and
  (2,2) -0.10263 m^2 s^-2 (real, the Sun over the prime meridian),
  illustrative as the lunar tide above is. The figure reads the planet's
  tidal Love numbers off the exchange table, each order's coefficients
  over the potential's: h2 = 0.3390 and k2 = 0.1823, the three orders
  agreeing to rounding, and a bulge of about 12 mm.
- **`response_tide_planet_relax.json`** — a unit degree-2 potential
  switched on and held on the planet's Maxwell deck, as
  `response1d_tide.json` is on the Earth's, at the same nodes out to
  500 kyr: the Love numbers relax from h2 = 0.34 and k2 = 0.18 to 1.83 and
  0.99, most of the way between 1 and 10 kyr, several times later than
  the Earth's under the same viscosity.
