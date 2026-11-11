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

The synthetic configs need no data and run in about a minute together on four
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
- **`response_seismic.json`** — the co-seismic and post-seismic response
  of the radial GIA deck to a thrust fault off north-east Japan (a 400 by
  150 km plane of scalar moment 4e22 N m, laid out as 8 by 3 point
  sources): `forcing.seismic` in a run through time, the first node the
  co-seismic response and the later ones the relaxation, to 2 kyr at lmax
  64. `out/response_seismic/response_seismic.nc` holds the radial
  displacement, the tangential potential V, the toroidal potential W and
  the potential at every node. About 5 s on two threads.
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
- **`sea_level_fingerprint_melt.json`** with **`response_melt.json`** — what
  the ocean adds: one ice change through both commands, elastic, on the
  radial Earth of `../data/prem8.elastic.deck` at lmax 128. The
  fingerprint melts a `cap_thickness` cap, 1 km at the centre of the
  synthetic North America (50° N, 100° W) and 3° wide, without rotation
  (so that what differs is the ocean alone); `response` applies the load
  of the same melt, a `cap` of 917 kg m^-3 times -1 km. The fingerprint
  loads only grounded ice, 917 x the change x (1 - C), and the cap lies
  18° from the nearest coast, so the ocean function cuts 5e-8 of its mass:
  the two loads agree to 1e-7 of their peak (the fingerprint writes
  `ice_change` and `ocean_function`, from which this is checked). The melt,
  6.4e17 kg, raises the ocean by 1.50 m on average (from -1.2 m near the
  melt to +1.78 m far from it). The ice's load alone lifts the cap's
  centre by 25.9 m; the meltwater's load changes the displacement by
  -21 to +14 cm, pressing the ocean floor down by 11 cm on average and
  lifting the cap's centre by a further 12 cm. A fraction of a second
  each.
- **`response_cap_weak.json`** and **`response_cap_strong.json`** — the
  same Hudson Bay-sized cap (10° wide, 1.834e6 kg m^-2) switched on at 0,
  held, and removed over 6 to 6.5 kyr, from 0 to 12 kyr in half-kyr steps
  at lmax 16, on the synthetic 3D viscoelastic Earth
  (`../data/viscosity_log.nc` on `../data/prem8.gia.deck`), at two places:
  over weak mantle at (0°, 90° E) and over stiff mantle at (0°, 0°). The
  file's log perturbation is a sin^2(theta) cos(2 phi) pattern about
  1e21 Pa s, strongest in the upper mantle: ln(eta / eta_ref) = -0.35 and
  +0.35 at the two centres at 400 km depth (0.70e21 and 1.42e21 Pa s), a
  factor of two between them. Each writes the radial reference model's
  response beside its own (`displacement_radial_model`), the same at both
  places. Under the load the weak site subsides faster (by 21 m more than
  the radial model after 1 kyr, the stiff one 19 m less, of about 255 m)
  and after the load is removed it rebounds faster (20 m left at 12 kyr,
  against 32 m in the radial model and 45 m at the stiff site). About 3 s
  each on four threads, 13 to 15 fixed-point iterations a step.
- **`response_load_cap.json`**, with **`response_load_field.json`** and
  **`response_load_coefficients.json`** — one load history given three
  ways, on the radial viscoelastic Earth of `../data/prem8.gia.deck` from 0
  to 12 kyr in quarter-kyr steps at lmax 32: a Gaussian cap (60° N, 85° W,
  5° wide, 1.834e6 kg m^-2) with the history `[[2, 0], [3, 1], [7, 1],
  [7.5, 0]]` (`response_load_cap`, no inputs but the deck); the same cap
  sampled into a surface file on a time axis, `../data/load_cap_history.nc`,
  read as a `field` term (`response_load_field`); and its coefficient
  table, `../data/load_cap_lm.txt`, as a `coefficients` term with the
  cap's history (`response_load_coefficients`). The two files come from
  `make_load_field.py` and `make_load_coefficients.py` (examples/README.md),
  and `run_all.sh` runs these two configs once they exist. The file's own
  time axis runs from 0 to 5.5 kyr (the load grows over its first
  kiloyear, is held to 5 and removed by 5.5); `"time_origin_kyr": 2` reads
  it at t - 2 kyr when the run's clock says t, so the file's time 0 is the
  run's 2 kyr, and before it and after 7.5 kyr the file's first and last
  (zero) loads are held. The coefficient table holds the LOAD's
  coefficients in the exchange table's harmonics (an exchange table holds
  the response's U and Phi, which are not a load). The field file is
  sampled on the run's own Gauss–Legendre grid, so the run reads it
  without interpolation, and the table is that field's expansion to the
  run's lmax: the three responses agree to 1e-14 of their largest values.
  The `load` written by the coefficient run is synthesised from the table
  (band-limited), so it differs from the sampled cap's by about 1% at the
  centre while the loads' coefficients, and so the responses, agree. About
  1 s each.
- **The time-step study**, `timestep_{et,be}_n<N>.json` (sea-level) and
  `timestep_{discrete,continuous}_n<N>.json` (kernel1d) — one problem at
  several step sizes: the synthetic deglaciation from 16 to 6 ka on the
  radial Earth of `../data/prem8.gia.deck` (Maxwell times 0.11 to
  0.43 kyr) at lmax 16, without rotation, with the sea level averaged over
  a 4° cap at Hudson Bay measured at 10 ka (mid-run) and at 6 ka (the
  end). N is the number of steps over the 10 kyr window, each N putting a
  node at 10 ka; `et` and `be` are the forward run by the exponential
  trapezoid and by backward Euler, `discrete` and `continuous` the
  kernels by each adjoint (the exponential trapezoid forward). The configs
  here are the short family, N = 10, 20 and 40 against N = 160, which
  `run_all.sh` runs (about 8 s on four threads); `timestep_study.py`, from
  `build/examples`, writes and runs the whole study, N = 5 to 80 (2 kyr to
  1/8 kyr) against N = 320 (1/32 kyr), in about 15 s, and prints its
  errors; `plot_app.py` draws them. Against the converged reference, at
  1 kyr steps (about nine of the shortest Maxwell times) the sea level is
  wrong by 0.5–0.6% (relative L2 over the sphere) under the exponential
  trapezoid and 0.7–1.0% under backward Euler, and the measured values by
  0.5–0.8%; the viscosity kernel is wrong by 3.3–4.0% by the discrete
  adjoint and 13–15% by the continuous one. The errors, at 2, 1, 1/2, 1/4
  and 1/8 kyr: sea level by the exponential trapezoid 1.7e-2, 6.2e-3,
  2.0e-3, 5.5e-4, 1.4e-4 (at 10 ka; 1.1e-2 to 1.1e-4 at 6 ka), by backward
  Euler 2.1e-2, 1.0e-2, 5.5e-3, 2.8e-3, 1.4e-3 (1.4e-2 to 1.1e-3 at 6 ka);
  the kernel at 6 ka by the discrete adjoint 7.7e-2, 4.0e-2, 1.7e-2,
  5.5e-3, 1.4e-3, by the continuous 0.33, 0.16, 5.9e-2, 1.8e-2, 4.9e-3.
  Backward Euler is first order throughout; the exponential trapezoid and
  both adjoints reach second order (1.9–2.0) only once the step comes down
  to the Maxwell times, and converge more slowly above them. At the
  reference step the two adjoints' kernels agree to 4e-4. The lesson: at
  a given step the kernels are several times less accurate than the
  fields they are kernels of (here five to eight times by the discrete
  adjoint and twenty to thirty by the continuous one). The discrete adjoint is the exact
  gradient of the time-stepped model, not the exact gradient of the
  Earth's: it inherits the time step's error, and more of it than the
  fields carry. A step that serves the forward run may not serve the
  kernels; refine it until the kernel stops changing.

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

## PREM with S40RTS on ICE-7G

These run on the real Earth: ICE-7G (`../data/ice7g.nc`), and PREM without
its ocean, a 100 km elastic lithosphere and Maxwell mantles of 5e20 Pa s
above 670 km and 5e21 Pa s below (`../data/earth_custom.gia.deck` and
`../data/earth_custom.elastic.deck`, written by `make_earth_deck.py`), with
S40RTS's lateral structure built on that deck by `make_lateral_model.py`
(the commands are in examples/README.md, ICE-7G). `run_all.sh` runs them
once all these files exist, and skips them otherwise; the 3D run through
time takes minutes.

- `../data/earth_custom_moduli_s40rts.nc` (`--field moduli --kind
  tomography --lmax 64`): the shear and bulk moduli from S40RTS's dlnVs,
  with dlnVp = dlnVs / 2 at fixed density, on the mantle below the Moho.
  dln mu has an rms of 1.7% over the mantle (4.5% at 100 km depth, 0.8%
  at 1200 km), and dln kappa about a third of dlnVs (0.27 to 0.40 of it
  with depth), its correlation with dln mu 0.999 or more at every depth.
- `../data/earth_custom_viscosity_s40rts.nc` (`--field viscosity --kind
  tomography --renormalise`, lmax 32): S40RTS's dlnVs through sl3d's
  conversion chain (its density scaling and expansivity to a temperature,
  then eta = eta_0 exp(-0.01 dT)), renormalised so that the lateral mean
  of the viscosity is the deck's at every radius. ln(eta / eta_ref) about
  the deck's viscosity runs from -2.98 to +3.57 (a factor of 20 below and
  35 above), with an rms of 0.62 (0.07 at the core-mantle boundary, 0.56
  at 150 km and up to 0.70 in the mid-mantle); renormalising lowers the
  geometric mean by up to 27%. No lid: with this deck's 100 km elastic
  lithosphere, the stiffest mantle beneath it is 5.7 times the deck's
  5e20 Pa s (2.9e21 Pa s), and `--lid` at 5e21 Pa s finds no lid anywhere.

- **`sea_level_fingerprint_3d_ice7g.json`** — the fingerprint of
  `sea_level_fingerprint_ice7g.json` (10% of ICE-7G's Greenland ice
  melted, full rotational feedback) on the elastic deck with the S40RTS
  moduli, at lmax 64: the 3D fingerprint, its zero-length step closed by
  the fixed point in 9 iterations (residual 1.2e-10), with the radial
  model's beside it (`*_radial_model`). The ocean rises by 0.62 m on
  average (to 0.83 m far from Greenland) and falls by up to 4.5 m at
  Greenland's coast. The lateral moduli change the sea level by up to
  7 cm over the interior of Greenland, under the ice (stiffer mantle,
  less rebound), and by a few millimetres over the ocean (within 3.3 mm
  over 99% of it, -15 to +8 mm along Greenland's coasts). Measured with
  `/usr/bin/time -v`: 27 s and 4.2 GB peak on four threads, 25 s and
  4.5 GB on eight; at lmax 128 (eight threads) 3 min 24 s and 31 GB, the
  same 9 iterations. Between lmax 64 and 128 the 3D minus radial
  difference changes by 16% of its rms (7% in the degrees both hold).
- **`sea_level_3d_ice7g.json`** with **`sea_level_1d_ice7g_custom.json`**
  — GIA through time on ICE-7G, 26 ka to the present in half-kyr steps
  (a node on every stored date) at lmax 32, full rotational feedback and
  the hydrostatic moments, on the Maxwell deck with S40RTS's lateral
  viscosity (`"reference": "file_radial"`, the deck's own viscosity, which
  the renormalisation made the 3D model's mean) and its lateral moduli;
  and the deck's radial run of the same (`sea_level_1d_ice7g_custom`),
  which is the 3D run's radial reference model. The lateral structure
  changes the sea level change from 26 ka to the present by -30 to +19 m
  (an rms of 3.5 m over the sphere, of the change's own 153 m), most
  under the former Cordilleran and Laurentide ice, and by up to 74 m at
  some time in the run. In relative sea level it lowers Hudson Bay's by
  13 m at 26 ka and raises it by 9 m at 8 ka, lowers the Gulf of
  Bothnia's by 5 m at 11 ka and raises Barbados's by 3 m at 26 ka. Each
  step is the combined fixed point to `step_rtol` 1e-8, which the first
  step reaches in about 60 iterations (the default budget is 60, so the
  config raises `step_max_iterations`); `"step": "conjugate_gradients"`
  converges within 40 passes a step and gives the same sea level to
  1.5e-5 m, in somewhat more time. The 3D run takes 4.4 minutes and
  1.6 GB on four threads, the radial one 1.5 s; at lmax 48 it does not
  finish in 15 minutes.


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
