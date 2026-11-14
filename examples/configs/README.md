# sl3d example configs

A config or more per `sl3d` command, each a working example of the program's
JSON input, made of the blocks the commands share (the members are
documented in `docs/sl3d-configs.md`). The `sea_level_*` configs are sea
level problems, with an ocean (but `sea_level_no_ocean.json`); the
`no_ocean*` configs are problems of a body's response without an ocean;
the `timestep_*` configs are the time-step study (below); the `kernel*`
and `timestep_{discrete,continuous}_*` configs, which hold a `kernels`
block, run as `sl3d kernel`. Every other one but `love_numbers.json` runs
as `sl3d solve`. The build
copies this directory to `build/examples/configs`, and copies a file again
when its source has changed and its copy has not been edited, so the
copies there are yours to edit and experiment with (an edited copy is
kept, and the configure step says so); delete one (or the directory) and
reconfigure to restore the original. Run the configs from there, where
their relative
paths reach the examples' inputs in `../data/` (`synthetic_earth.nc` and
`synthetic_moduli.nc`, written by the build, `prem8.elastic.deck`,
`prem8.gia.deck`, `prem8.postseismic.deck` and `viscosity_log.nc`) and
each config's outputs land in
`out/<config>/`, `<config>` being its file name without `.json`:

```sh
cd build/examples/configs
./run_all.sh                       # every config, stopping at a failure
../../apps/sl3d solve sea_level_1d.json   # or one at a time
```

The synthetic configs need no data and run in about five minutes
together on four threads (`OMP_NUM_THREADS=4`), three of them the
seismic runs; CI runs them with `run_all.sh`. They are
sized for that, so their degrees are low; raise `discretisation.lmax`
and `times.nodes` for sharper results. `poetry run python plot_app.py`,
from `build/examples`, draws each config's figures from `out/<config>/` into
`figures/sl3d/<config>/` (examples/README.md lists them).

A run through time gives its nodes in one of two forms
(docs/sl3d-configs.md, The times block). Dated, as the deglaciations
here: `start_kyr_bp`, `end_kyr_bp` and `nodes`, or the dates listed in
`at_kyr_bp`, its terms' times dates too (`history_kyr_bp`,
`onset_kyr_bp`), its files read by date and its state at an end of the
window (`ocean.at`). Counting forwards on the run's own clock, as
`sea_level_postseismic.json` and the `no_ocean*` runs through time: a
`unit` (`"kyr"`, `"yr"` or `"day"`) with `start`, `end` and `nodes` or the
times listed in `at`, its terms' times on that clock (`history`, `onset`,
a load file's `time_origin`), its state, where it has one, a snapshot of
its files at `ocean.date_kyr_bp`, and its output's nodes in the unit
(`times`, `time_unit`) rather than in dates (`dates_kyr_bp`). The static
problem reads a load file with a time axis at `date_kyr_bp`, or at `date`
with its `unit` on the file's own axis.

The `no_ocean*` configs have no `ocean` block, so the body has no ocean,
and give `"rotation": "none"`, so the response is the solid body's alone
(`no_gravity_postseismic.json` and `no_gravity_load.json` are such
responses without gravity too);
those on lateral structure set `solver.step_rtol` 1e-10 and
`step_max_iterations` 200, tighter than the defaults (1e-9, 60).

## The synthetic Earth

- **`love_numbers.json`** — the load and tidal Love numbers of the prem8
  deck to degree 256: `out/love_numbers/love_numbers.txt`.
- **`sea_level_fingerprint.json`** — the static problem:
  the elastic sea-level fingerprint of melting a tenth of the synthetic
  Greenland ice, its state (the ice and the topography) read at the
  present day (`ocean.date_kyr_bp` 0) and the melt a `cap_fraction` term
  of the ice change (`forcing.ice_change`), at lmax 128, with full
  rotational feedback:
  `out/sea_level_fingerprint/sea_level_fingerprint.nc`.
- **`sea_level_fingerprint_3d.json`** — that fingerprint on the synthetic
  3D elastic Earth (`../data/synthetic_moduli.nc` on
  `../data/prem8.elastic.deck`, as `no_ocean.json`): in the static
  problem, a moduli file that varies laterally makes the fingerprint 3D,
  the instantaneous response of the 3D sea level problem, its
  zero-length step closed by the fixed point, at lmax 32 (the moduli
  file's own degree):
  `out/sea_level_fingerprint_3d/sea_level_fingerprint_3d.nc`, which also
  holds the radial model's fingerprint of the same melt
  (`*_radial_model`). The stiff craton under North America and the soft
  ridge change the sea level by a few centimetres at most, against a
  near-field fall of about 12 m; about 3 s and 0.9 GB on four threads.
- **`sea_level_static_migrating.json`** — the static problem under
  migrating shorelines (`ocean.shorelines`): the elastic
  response to the whole synthetic deglaciation, its state the synthetic
  Earth at 21 ka (`ocean.date_kyr_bp` 21) and the change a `thickness`
  term read at the present, so that the ice at the solve is the present
  day's and the shorelines follow its flotation, at lmax 128 with full
  rotational feedback:
  `out/sea_level_static_migrating/sea_level_static_migrating.nc`, which
  holds the ocean function the solve reaches (`ocean_function_final`)
  beside the state's. The ocean rises by 80 m on average over the 21 ka
  ocean, and the marine margins of the ice sheets flood: the ocean covers
  83.75% of the surface at the solve against 83.33% at 21 ka. A fraction
  of a second.
- **`sea_level_static_coseismic.json`** — the static problem forced by a
  fault alone, no ice change (`forcing.seismic`): the
  co-seismic sea level change of a thrust off the synthetic South
  America's west coast (centred at 22° S, 74.5° W, 25 km deep), a 600 by
  150 km plane of scalar moment 2e22 N m dipping 18° east under the
  continent, laid out as 12 by 3 point sources, on the present day's
  state at lmax 96 with full rotational feedback:
  `out/sea_level_static_coseismic/sea_level_static_coseismic.nc`.
  Averaged over Gaussian caps of 2°, the measure the seismic examples are
  converged in (Convergence in lmax, below), the sea floor rises by up to
  0.16 m over the fault and moves by up to 0.87 m towards the trench, the
  sea level falling by the uplift; the ocean's water is conserved, the
  sea level change integrating to 1e-17 of its magnitude over the ocean.
  The file holds the tangential and toroidal potentials (the horizontal
  motion) beside the rest, and, as the config's `output` asks, the
  horizontal displacement's east and north components
  (`"fields": ["horizontal_displacement"]`) and every field of the
  response smoothed over a 2° Gaussian cap (`"smoothing": {"kind":
  "gaussian", "width_deg": 2}`, as `<field>_smoothed`), which the figure
  (`coseismic.png`) draws, with the point values along a profile across
  the fault for their ripples. About 2.5 s and 1.2 GB on four threads.
- **`sea_level_static_megathrust.json`** — the static problem forced by a
  fault given by its slip, a planetmodel fault file (`forcing.seismic`'s
  `fault_file`): a curved megathrust off the synthetic South America, its
  trench an arc of 940 km near 74° W from 24° S to 16° S (a small circle
  about a pole inland), the interface descending east at 10° at the trench
  steepening to 25° at its base, 50 km deep, and a thrust of 12 m at its
  middle tapered to nothing at every edge, about 2.4e22 N m on this deck
  (Mw 8.9), on the present day's state on the post-seismic Earth
  `../data/prem8.postseismic.deck` at lmax 96 with full rotational
  feedback: `out/sea_level_static_megathrust/sea_level_static_megathrust.nc`.
  `make_megathrust.py` (examples/README.md) writes the file,
  `../data/megathrust.nc`, through planetmodel's swept patch; it needs
  planetmodel's fault module (planetmodel after 1.5.1), and `run_all.sh`
  runs this config only once the file exists. The file holds the slip
  alone: sl3d lays a quadrature of 36 by 6 cells over the patch, a point at
  each cell's centre, and forms each point's moment from the deck's moduli
  there, the slip and the patch's normal. Its positions are on the Earth's
  mean sphere, 6371 km, and the deck ends at 6368 km (PREM without its
  ocean), so the config places the fault by depth (`"place_by":
  "depth"`). In 2° means the sea floor rises by up to 0.13 m over the
  fault and moves by up to 0.88 m towards the trench, the sea level
  falling by the uplift; the ocean's water is conserved, the sea level
  change integrating to 5e-17 of its magnitude over the ocean. The figure
  (`coseismic.png`) is `sea_level_static_coseismic`'s, the fault's outline
  and trench drawn from the file. The 216 points lie at 102 radii (the
  patch is a surface of revolution about the trench's pole, its points
  pairing across the arc's middle), and the sources at one radius are
  formed together: about 10 s and 1.2 GB on four threads, against 20 s
  with every point formed on its own.
- **`sea_level_postseismic.json`** — that thrust's post-seismic
  relaxation, the quasi-static problem counting forwards
  on the run's own clock in years (`"times": {"unit": "yr", "at": [...]}`,
  23 uneven nodes from 0 to 300 yr, each step after the first year a
  quarter to a third of the time elapsed) rather than in dates: the fault
  acts from the run's start (`"onset": 0`), the state is a snapshot of
  the synthetic Earth's files at the present day (`ocean.date_kyr_bp` 0,
  `"at": "start"`), on the post-seismic Earth `../data/prem8.postseismic.deck`
  (a 50 km elastic lid over a weak asthenosphere, below) at
  lmax 96 with full rotational feedback and the hydrostatic moments:
  `out/sea_level_postseismic/sea_level_postseismic.nc`, its time axis the
  run's clock in seconds and its nodes in years in the metadata (`times`,
  `time_unit`). The first node is the co-seismic answer of
  `sea_level_static_coseismic.json` to 1.6 mm, and to 0.2 mm in 2° means
  (this deck's viscous layers are isotropic, the elastic deck's
  transversely isotropic between 24 and 220 km). Then the asthenosphere relaxes, and the upper mantle after it.
  In 2° means the uplift over the fault, 0.150 m at the event, grows by
  6 mm in the first year, to 0.205 m by 10 years and 0.319 m at a
  century, and falls back to 0.233 m by 300 years; 250 km down dip, on
  the coast, it rises from 0.052 m to 0.209 m at 140 years and is 0.189 m
  at 300; 1000 km up dip it goes from +1 mm to -27 mm at 140 years and
  -22 mm at 300. The relative sea level follows minus the uplift. The
  file holds the east and north components and every field smoothed over
  2° at every node, as `sea_level_static_coseismic.json`'s, and the
  figure (`postseismic.png`) draws the relaxation as those 2° means, in maps
  and at three places on a time axis linear about the event and
  logarithmic after it, the deck's Maxwell times marked. On nodes three
  times as close the means move by at most 3.2 mm, 1.8% of their change
  (at lmax 64). About 46 s and 4.0 GB on four threads.
- **`no_ocean_postseismic.json`** — that thrust without an ocean and
  without rotational feedback, on the same Earth, nodes and lmax: the
  solid body's response, `out/no_ocean_postseismic/no_ocean_postseismic.nc`
  (the displacement, the tangential, toroidal and gravitational
  potentials, the east and north components, and each smoothed over 2°),
  and the twin with gravity of the next. In 2° means over
  the fault the uplift is 0.147 m at the event, 0.299 m at its peak (75
  years) and 0.200 m at 300 years; the horizontal motion up dip there
  0.851 m at the event and 0.701 m at 300 years, and 250 km down dip it
  grows from 0.631 m to 1.046 m. About 47 s on four threads.
- **`no_gravity_postseismic.json`** — the problem without gravity
  (`"gravity": "none"`): the same thrust on the same Earth, nodes and
  lmax, without an ocean, a potential or a rotational feedback, the
  degree systems taking the displacement alone and stopping at the
  core-mantle boundary:
  `out/no_gravity_postseismic/no_gravity_postseismic.nc` (the
  displacement, the tangential and toroidal potentials and the east and
  north components, each smoothed over 2° too; no potential and no sea
  level change) and exchange tables of the displacement at the
  first and last nodes, `out/no_gravity_postseismic/no_gravity_postseismic_{node}.txt`.
  Against its twin with gravity, `no_ocean_postseismic.json`, in 2° means
  over the fault: 0.153 m of uplift at the event against 0.147 m, 0.356 m
  at its peak (100 years) against 0.299 m, and 0.297 m at 300 years
  against 0.200 m; 250 km down dip 0.238 m at 300 years against 0.170 m.
  Without self-gravitation and buoyancy nothing but the elastic lid holds
  the relaxing surface down, and the uplift grows the more; the
  horizontal motion hardly differs (0.700 m up dip over the fault at 300
  years against 0.701 m). The figure comparing the two,
  `with_and_without_gravity/postseismic.png`, draws them on one panel.
  About 35 s on four threads.
- **`sea_level_1d.json`** — the quasi-static problem (its
  `times` block the window, its state, the topography and the ice, read at
  the window's start, `"at": "start"`, and the synthetic history a
  `thickness` term of the ice change): the synthetic
  deglaciation from 21 ka to the present through the 1D viscoelastic Earth
  of `../data/prem8.gia.deck`, at lmax 32 in quarter-kiloyear steps, with
  full rotational feedback (the moments of inertia from the model's
  hydrostatic limit, which the full feedback needs):
  `out/sea_level_1d/sea_level_1d.nc`. It also writes checkpoints every six
  nodes to `out/sea_level_1d/checkpoints`, so a rerun resumes rather than
  starting again; a rerun after the config or an input has changed is
  refused instead, until `"restart": false` is set or the directory removed
  (a rebuild leaves the synthetic Earth as it was unless its content
  changed, so its checkpoints stay the run's).
- **`sea_level_1d_calibrated.json`** — that deglaciation with migrating
  shorelines, its initial sea level calibrated against the synthetic Earth's
  present day rather than read at 21 ka (the device a thickness-only ice
  history needs), by the classical update in a few runs (the `ocean` block:
  its state, the present day's, `"at": "end"`, and `calibration`):
  `out/sea_level_1d_calibrated/sea_level_1d_calibrated.nc`, which also holds
  the calibrated `initial_sea_level` and the `ocean_function` at every node.
  The marine margins of the synthetic ice sheets flood as they melt.
- **`sea_level_tide_migrating.json`** — that deglaciation with migrating
  shorelines from the state at 21 ka (`"at": "start"`) under a degree-2
  zonal tidal potential beside the ice (`forcing.tide`), switched on over
  the window to 30 m^2 s^-2 by the present (its `history_kyr_bp`), at lmax
  32 in kiloyear steps: `out/sea_level_tide_migrating/sea_level_tide_migrating.nc`.
  The tide enters each node's nonlinear sea level solve as it enters the
  linear one, its potential moving the sea surface and so the flotation
  that places the shorelines. Its equilibrium surface, -psi / g, falls by
  nearly 3 m from pole to equator, but by the present the body has relaxed
  nearly to the fluid response, which follows the equipotential: the
  tide changes the relative sea level by -0.27 to +0.12 m, and moves no
  shoreline of the synthetic Earth's steep coasts; the shorelines the
  figure (`sea_level.png`) shows moved are the deglaciation's. About 1 s
  on two threads.
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
- **`kernel1d.json`** (`sl3d kernel`) — adjoint kernels in 1D of five
  measurements of that deglaciation (its forward run read from the same
  blocks as `sea_level_1d.json`'s, the measurements in its `kernels`
  block): the present sea level averaged over Hudson Bay, the sea level at
  a far-field point at 10 ka, the length of day (the axial rotation), the
  degree-2 zonal Stokes coefficient J2, and the decay time of the Hudson
  Bay series over the last 7 kyr. Each gets its ice-history kernels and
  its radial viscosity kernel (`"parameters": ["ice", "viscosity"]`):
  `out/kernel1d/`. About 5 s on four threads.
- **`kernel.json`** (`sl3d kernel`) — the 3D counterpart for the Hudson Bay
  sea level and decay time, with the log-viscosity kernels of the 3D
  viscosity: `out/kernel/`. About 3 s on four threads.
- **`kernel1d_lateral.json`** (`sl3d kernel`) — the same two measurements
  about the radial Earth (`kernel.json`'s deck without its viscosity file),
  with `"kernels": {"lateral": true}`: the kernels in the log viscosity and
  in the moduli as fields over the memory nodes' grids, by the radial
  sweeps with the grid expansion (no 3D solve), written as `kernel.json`'s
  are (`<name>_viscosity_kernels.nc`, `<name>_moduli_kernels.nc`) beside the
  radial profiles they average to (`<name>_viscosity_kernel.txt`,
  `<name>_moduli_kernel.txt`): `out/kernel1d_lateral/`. The figure
  (`kernel1d_lateral/viscosity_kernels.png`) maps the log-viscosity
  kernel at the depth where it is largest, cuts a section through Hudson
  Bay and draws its lateral mean and rms against radius. About 1.3 s on
  four threads.
- **`kernel_static_moduli.json`** (`sl3d kernel`) — the static problem
  without an ocean on the radial `../data/prem8.elastic.deck` at lmax 32,
  under a disk of 1e5 kg m^-2 over Hudson Bay: the kernels in the load and
  in the moduli kappa and mu of the uplift averaged over a 2° cap at its
  centre and of the northward displacement 10° south of it (a vector
  displacement, through the tangential potential). Its `output` block is
  `solve`'s, so `sl3d solve` on the same config writes the response the
  kernels are of: `out/kernel_static_moduli/`. About 0.15 s.
- **`kernel_postseismic.json`** (`sl3d solve`, then `sl3d kernel`) —
  `sea_level_postseismic`'s thrust off northern Chile (as 6 by 2 patches,
  at lmax 32) on the post-seismic Earth, on the run's own clock in years,
  from the event to a century after it at eight nodes: the kernels, by the
  discrete adjoint, in the moment tensors of
  the fault's patches at every node and in the viscosity, of a GPS
  station's eastward and upward displacement 20 years on and of a tide
  gauge's sea level a century on: `out/kernel_postseismic/`. The one config
  serves both commands, the workflow of a run swept for many
  measurements: `sl3d solve` writes the run (`run.nc`) and records it into
  the trajectory store `out/kernel_postseismic/trajectory`, and `sl3d
  kernel` sweeps from the store without running the forward problem
  again; a kernel config of other measurements against the same store
  (the same blocks, another `kernels` block) would too. A run under
  seismic sources is recorded at every node, each state with its strain,
  its toroidal state and the sources' forces: 8 state records of 86 MB
  and 8 surface records of 104 kB, 655 MiB in all, the nodes few to keep
  the store small. On four threads the solve takes about 1 s without
  recording and 1.2 to 2 s recording (the disk's), and the kernels 3 to
  4 s, from the store or with the forward run in memory, the same to
  rounding (about 1e-14 relative; two processes may plan their FFTs
  differently, and the tests hold the two bitwise within one process).
  The steps are long against the asthenosphere's Maxwell time: on 33
  nodes the measurements move by 0.4 to 2.1% and the viscosity kernels
  by 4% (the GPS station's) and 9% (the gauge's), a kernel carrying more
  of the time step's error than the field it is of (the time-step study,
  below). The figure (`kernels.png`)
  draws the moment kernels per patch, held from the onset, and their
  contraction with the fault's own mechanism, which gives back each
  measurement's value with the fault's moment.
- **`no_ocean.json`** — the instantaneous elastic response of the
  synthetic 3D elastic Earth (`../data/synthetic_moduli.nc`, written by the
  build: the PREM-like moduli with a stiff craton and a soft ridge in the
  upper mantle) without an ocean to a 2 km ice-equivalent cap over Hudson
  Bay and a unit degree-2 tide, at lmax 16, the static problem's
  zero-length step closed by the fixed point: `out/no_ocean/no_ocean.nc`
  (the surface displacement, potential and the radial model's beside
  them), `out/no_ocean/no_ocean.txt` (the surface coefficients in the
  cross-code exchange form) and `out/no_ocean/no_ocean_volume.nc` (the
  response inside the body, for planetmodel's cutaways). The forcing's
  `load` and `tide` are lists of terms, summed.
- **`no_ocean_viscoelastic.json`** — the viscoelastic response of the
  synthetic 3D Earth, the lateral viscosity of `../data/viscosity_log.nc`
  with the lateral moduli of `../data/synthetic_moduli.nc`, to the same
  Hudson Bay cap loaded linearly over 6 kyr and removed over the next one
  (its `history`) and a constant unit degree-2 tide, from 0 to 12 kyr in
  half-kyr steps on the run's own clock (`"unit": "kyr"`) at lmax 16, by
  the 3D stepper:
  `out/no_ocean_viscoelastic/no_ocean_viscoelastic.nc` (the surface response
  at every node, the radial reference model's beside it) and exchange tables
  at four nodes,
  `out/no_ocean_viscoelastic/no_ocean_viscoelastic_{node}.txt` (the start,
  the peak load, the unloaded state and the end). The quasi-static problem
  (`"problem": "quasi_static"` with a `times` block) is what makes the
  response viscoelastic; it takes about 5 s on four threads.
- **`no_ocean_seismic.json`** — the co-seismic and post-seismic response
  of the post-seismic Earth to a thrust fault off north-east Japan (a 400
  by 150 km plane of scalar moment 4e22 N m, laid out as 8 by 3 point
  sources): `forcing.seismic` in a run through time, the first node the
  co-seismic response and the later ones the relaxation, on the 23 nodes
  of `sea_level_postseismic.json` to 300 years at lmax 96.
  `out/no_ocean_seismic/no_ocean_seismic.nc` holds the radial
  displacement, the tangential potential V, the toroidal potential W, the
  potential and the east and north components at every node, each
  smoothed over 2° too, which the figure (`postseismic.png`)
  draws as `sea_level_postseismic`'s without an ocean: in 2° means the
  uplift over the fault, 0.171 m at the event, peaks at 0.393 m at 75
  years and is 0.240 m at 300, and 1000 km up dip the surface moves 0.49
  m towards the fault over the three centuries. About 38 s on four
  threads.
- **`no_ocean_1d_disk.json`** — a disk load of 8 degrees over Hudson Bay, the
  mass of about 2 km of ice, held for 10 kyr and removed over the next half,
  on the radial viscoelastic Earth of `../data/prem8.gia.deck` (no lateral
  files, so the 1D stepper), from 0 to 20 kyr in half-kyr steps at lmax 64:
  `out/no_ocean_1d_disk/no_ocean_1d_disk.nc`. The disk is the exact expansion
  to lmax, not a sampled step. The figure shows the displacement against
  distance from the centre, the peripheral bulge growing under the load and
  the uplift after it; about 5 s on four threads.
- **`no_ocean_1d_tide.json`** — a degree-2 tidal potential switched on and
  held on the same radial Earth, at unevenly spaced nodes out to 500 kyr
  (`times.at`, in kyr), with an exchange table at every node:
  `out/no_ocean_1d_tide/no_ocean_1d_tide.nc` and
  `out/no_ocean_1d_tide/no_ocean_1d_tide_{node}.txt`. The coefficients over
  the applied potential are the tidal Love numbers through time, from the
  elastic values at the start towards the fluid limit, which the figure
  draws; a fraction of a second.
- **`no_ocean_tide_3d.json`** — the lunar body tide on the synthetic 3D
  elastic Earth (`../data/synthetic_moduli.nc` on
  `../data/prem8.elastic.deck`), at lmax 32:
  `out/no_ocean_tide_3d/no_ocean_tide_3d.nc` (the 3D and the radial model's
  surface responses) and `out/no_ocean_tide_3d/no_ocean_tide_3d.txt`. The
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
- **`no_ocean_tide_oscillating.json`**, with
  **`no_ocean_tide_oscillating_short.json`** and
  **`no_ocean_tide_oscillating_long.json`** — a degree-2 tide oscillating
  on the radial viscoelastic Earth of `../data/prem8.gia.deck`, whose
  mantle's Maxwell times (viscosity over shear modulus) run from 0.11 to
  0.43 kyr: the (2,0) coefficient of 1 m^2 s^-2 times sin(2 pi t / P),
  with P = 1 kyr (0.1 kyr and 10 kyr in the other two), over six periods
  at 24 nodes a period, at lmax 4:
  `out/<config>/<config>.nc` and one exchange table at the last crest of
  the potential. They are the output of `make_tide_history.py` (in
  `build/examples`), whose amplitude history `history` has a point at
  every node, where the run sees the forcing. With a unit potential the
  degree-2 coefficients
  are the Love numbers at the forcing's frequency; over the last cycle at
  P = 1 kyr |h2| = 0.811 lagging the potential by 23.0° and |k2| = 0.409
  lagging by 24.1°, at 0.1 kyr |k2| = 0.301 lagging by 4.0°, and at
  10 kyr |k2| = 0.787 lagging by 12.2°: the lag is largest between the
  elastic and the fluid limits. Against 96 nodes a period, 24 put k2
  within 0.7% and its lag within 0.4°. A fraction of a second each.
- **`sea_level_fingerprint_ice_cap.json`** with **`no_ocean_ice_cap.json`**
  — what the ocean adds: one ice change with an ocean and without one,
  elastic, on the radial Earth of `../data/prem8.elastic.deck` at lmax 128. The
  fingerprint lays a `cap_thickness` cap of ice, 1 km at the centre of the
  synthetic North America (50° N, 100° W) and 3° wide, on the present
  day's bare land (a change may not take ice the state does not hold, so
  the cap is grown rather than melted: a melt is its negative), without
  rotation (so that what differs is the ocean alone); `no_ocean_ice_cap.json`
  applies the load of the same ice to the body without an ocean, a `cap`
  of 917 kg m^-3 times 1 km. The
  fingerprint loads only grounded ice, 917 x the change x (1 - C), and the
  cap lies 18° from the nearest coast, so the ocean function cuts 5e-8 of
  its mass: the two loads agree to 1e-7 of their peak (the fingerprint
  writes `ice_change` and `ocean_function`, from which this is checked).
  The ice, 6.4e17 kg drawn from the ocean, lowers it by 1.50 m on average
  (from +1.2 m near the cap to -1.78 m far from it). The ice's load alone
  presses the cap's centre down by 25.9 m; the water's unloading changes
  the displacement by -14 to +21 cm, lifting the ocean floor by 11 cm on
  average and pressing the cap's centre down by a further 12 cm. A
  fraction of a second each.
- **`no_ocean_cap_weak.json`** and **`no_ocean_cap_strong.json`** — the
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
- **`no_ocean_load_cap.json`**, with **`no_ocean_load_field.json`** and
  **`no_ocean_load_coefficients.json`** — one load history given three
  ways, on the radial viscoelastic Earth of `../data/prem8.gia.deck` from 0
  to 12 kyr in quarter-kyr steps at lmax 32: a Gaussian cap (60° N, 85° W,
  5° wide, 1.834e6 kg m^-2) with the history `[[2, 0], [3, 1], [7, 1],
  [7.5, 0]]` (`no_ocean_load_cap`, no inputs but the deck); the same cap
  sampled into a surface file on a time axis, `../data/load_cap_history.nc`,
  read as a `field` term (`no_ocean_load_field`); and its coefficient
  table, `../data/load_cap_lm.txt`, as a `coefficients` term with the
  cap's history (`no_ocean_load_coefficients`). The two files come from
  `make_load_field.py` and `make_load_coefficients.py` (examples/README.md),
  and `run_all.sh` runs these two configs once they exist. The file's own
  time axis runs from 0 to 5.5 kyr (the load grows over its first
  kiloyear, is held to 5 and removed by 5.5); `"time_origin": 2` reads it
  at t - 2 kyr when the run's clock says t (in its unit, kyr), so the
  file's time 0 is the
  run's 2 kyr, and before it and after 7.5 kyr the file's first and last
  (zero) loads are held, as `"outside_span": "hold"` asks (without it the
  run's nodes outside the file's span are refused). The coefficient table holds the LOAD's
  coefficients in the exchange table's harmonics (an exchange table holds
  the response's U and Phi, which are not a load). The field file is
  sampled on the run's own Gauss–Legendre grid, so the run reads it
  without interpolation, and the table is that field's expansion to the
  run's lmax: the three responses agree to 1e-14 of their largest values.
  The `load` written by the coefficient run is synthesised from the table
  (band-limited), so it differs from the sampled cap's by about 1% at the
  centre while the loads' coefficients, and so the responses, agree. About
  1 s each.
- **`sea_level_no_ocean.json`** — `no_ocean_load_cap.json`'s run dated:
  its window from 12 kyr BP to the present in quarter-kyr steps and the
  cap's history in dates before present (`history_kyr_bp` `[[10, 0], [9,
  1], [5, 1], [4.5, 0]]`), without an `ocean` block, the load's properties
  left at their defaults and no rotational feedback:
  `out/sea_level_no_ocean/sea_level_no_ocean.nc`. Its displacement,
  potential and tangential potential are `no_ocean_load_cap`'s at every
  node to 1e-11 of their largest values (9e-13 for the displacement), the
  rounding of the two clocks' node times, and its `load` at every node to
  2e-15; its relative sea level change is -(u + phi / g), the height of an
  equipotential above the surface. About 1 s.
- **`no_gravity_load.json`** — surface loads in the problem without
  gravity (`"gravity": "none"`), where a load acts by its weight, a normal
  traction g sigma, and must have no degree-1 part (its net force would
  be unbalanced): `no_ocean_load_cap.json`'s cap and its antipode (60° S,
  95° E), with the same history, on the radial Maxwell deck at lmax 32,
  without an ocean, a potential or a rotational feedback, the degree
  systems taking the displacement alone and stopping at the core-mantle
  boundary: `out/no_gravity_load/no_gravity_load.nc` (the displacement,
  the tangential potential and the loads applied; no potential and no sea
  level change) and exchange tables of the displacement at 3, 7 and
  12 kyr, `out/no_gravity_load/no_gravity_load_{node}.txt`. The pair's
  degree-1 part is 4e-16 of its size; a single cap is refused, naming
  `forcing.load`. Without buoyancy the surface under each cap sinks 203 m
  once the load is on and 1018 m by 7 kyr (125 m and 327 m with gravity),
  and after the load is gone it springs back by its elastic part alone,
  792 m down at 12 kyr against 60 m with gravity, as the figure
  comparing it with `no_ocean_load_cap`,
  `with_and_without_gravity/load.png`, draws. Under 1 s on four
  threads.
- **The time-step study**, `timestep_{et,be}_n<N>.json` (solve) and
  `timestep_{discrete,continuous}_n<N>.json` (kernel) — one problem at
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
  adjoint and 13–16% by the continuous one. The errors, at 2, 1, 1/2, 1/4
  and 1/8 kyr: sea level by the exponential trapezoid 1.7e-2, 6.2e-3,
  2.0e-3, 5.5e-4, 1.4e-4 (at 10 ka; 1.1e-2 to 1.1e-4 at 6 ka), by backward
  Euler 2.1e-2, 1.0e-2, 5.5e-3, 2.8e-3, 1.4e-3 (1.4e-2 to 1.1e-3 at 6 ka);
  the kernel at 6 ka by the discrete adjoint 7.7e-2, 4.0e-2, 1.7e-2,
  5.5e-3, 1.4e-3, by the continuous 0.34, 0.16, 6.0e-2, 1.8e-2, 4.9e-3.
  Backward Euler is first order throughout; the exponential trapezoid and
  both adjoints reach second order (1.9–2.0) only once the step comes down
  to the Maxwell times, and converge more slowly above them. At the
  reference step the two adjoints' kernels agree to 4e-4. The lesson: at
  a given step the kernels are several times less accurate than the
  fields they are kernels of (here five to eight times by the discrete
  adjoint and twenty to thirty-three by the continuous one). The discrete adjoint is the exact
  gradient of the time-stepped model, not the exact gradient of the
  Earth's: it inherits the time step's error, and more of it than the
  fields carry. A step that serves the forward run may not serve the
  kernels; refine it until the kernel stops changing.

`../data/prem8.gia.deck` is `../data/prem8.elastic.deck` in planetmodel's
GIA format, with a Maxwell mantle of 1e21 Pa s (the solid layers between
3480 and 6151 km) and `nan` elsewhere: the radial runs through time take
their viscosity from this column.

`../data/prem8.postseismic.deck`, the Earth of the post-seismic configs
(`sea_level_postseismic`, `no_ocean_postseismic`,
`no_gravity_postseismic`, `no_ocean_seismic`, `kernel_postseismic`) and
of `ExampleSeismic`, is PREM without its ocean sampled as
`prem8.elastic.deck` is (8 knots a layer), with a thin elastic lid over
a weak asthenosphere: the lid elastic from the surface (6368 km) to 50
km depth, a boundary inserted there inside PREM's lid, so that the
faults' deep ends (48 km off South America, 41 km off Japan) lie inside
it, the first 2 km above its base; a Maxwell asthenosphere of
1e19 Pa s from there to PREM's 220 km discontinuity (6151 km, 217 km
deep); an upper mantle of 1e20 Pa s to the 670 km discontinuity (667 km
deep) and a lower mantle of 1e22 Pa s to the core. Its viscous layers
are isotropic, PREM's moduli Voigt-averaged (sl3d refuses a transversely
isotropic Maxwell layer), and the lid keeps PREM's. Their Maxwell times
eta / mu, from the deck's own moduli: 4.7 to 4.8 years in the
asthenosphere, 26 to 43 years in the upper mantle, 1080 to 2050 years in
the lower mantle. `python make_postseismic_deck.py`, from
`build/examples`, writes it from planetmodel's PREM and prints those
times; its options make other lids and viscosities (`--out` keeps the
original). The radial mesh pins every boundary of the deck and cuts each
layer into elements no wider than 0.1 a / (lmax + 1), 6.6 km at lmax 96
and 19 km at lmax 32, so the lid's mantle (28.6 km) and the
asthenosphere's part above PREM's 80 km discontinuity (50 to 77 km deep)
take five elements each at lmax 96 and two at lmax 32. A viscosity
contrast of three decades needs no care in the step: the schemes are
implicit, and the post-seismic configs step from 0.1 to 100 years,
across the Maxwell times, at second order (above).

### Convergence in lmax of the seismic examples

sl3d is a global code, and these examples show what its seismic
forcing resolves: the regional and global response (averages over caps
wider than the truncation wavelength, about 40 000 km / lmax), not the
field at kilometre scale about the fault, which is a regional or
half-space code's work.

The thrust off South America is a 600 by 150 km plane reaching from 2 to
48 km depth, laid out as 12 by 3 point sources 50 km apart, the
shallowest row at 9.5 km. A point source at depth d puts its surface
field into degrees up to about a / d, some 670 for that row, and the
point sources are resolved one by one from about lmax 400: the field's
point values near the fault are not converged at any degree these runs
reach, and their converged limit would be the point sources' own
fields, not a smooth fault's. The truncation at lmax rings about the
source, 360/lmax degrees apart. Means over Gaussian caps of 2° (the
field averaged over a cap of 2° about each point) are converged: the
seismic configs write those beside the raw fields (`output.smoothing`,
docs/sl3d-configs.md, Optional and smoothed fields), and their figures
draw them, saying so; the configs run at
lmax 96, where the means are within a few tenths of a per cent of lmax
128 and 256. On the post-seismic deck with an ocean, along a profile
across the fault through its centroid, the co-seismic field (the static
problem; four threads):

| lmax | peak uplift, point | peak horizontal, point | point uplift's ripple beyond 400 km | peak uplift, 2° mean | peak horizontal, 2° mean | time, memory |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.373 m | 1.96 m | 0.116 m (31% of its peak) | 153.3 mm | 852.9 mm | 1.1 s, 0.5 GB |
| 96 | 0.585 m | 2.64 m | 0.226 m (39%) | 159.1 mm | 872.6 mm | 2.6 s, 1.2 GB |
| 128 | 0.726 m | 2.83 m | 0.238 m (33%) | 159.1 mm | 872.6 mm | 4.6 s, 2.2 GB |
| 256 | 1.335 m | 4.30 m | 0.100 m (7%) | 159.1 mm | 872.6 mm | 18.6 s, 9.4 GB |

From one lmax to the next the point values move by 44%, 38% and 46% of
their peak uplift (34%, 28% and 38% of the horizontal's), the 2° means
by 4.0%, 0.20% and 0.04% (2.7%, 0.08% and 0.001%). Through time,
`sea_level_postseismic`'s series at its three places and at the coast
(the tide gauge of `ExampleSeismic`, 22° S, 72° W), the largest change
over the run from one lmax to the next, against the series' largest
value at lmax 128:

| series | 48 to 64 | 64 to 96 | 96 to 128 |
| --- | --- | --- | --- |
| uplift over the fault, point | 164 mm (22%) | 247 mm (33%) | 175 mm (23%) |
| uplift over the fault, 2° mean | 24 mm (7.6%) | 7.5 mm (2.3%) | 0.7 mm (0.2%) |
| uplift 250 km down dip, point | 77 mm (10%) | 304 mm (39%) | 223 mm (29%) |
| uplift 250 km down dip, 2° mean | 10 mm (4.9%) | 6.2 mm (3.0%) | 0.2 mm (0.1%) |
| uplift 1000 km up dip, point | 47 mm (70%) | 79 mm (118%) | 35 mm (52%) |
| uplift 1000 km up dip, 2° mean | 5.3 mm (20%) | 1.8 mm (6.6%) | 0.2 mm (0.6%) |
| sea level at the coast, point | 80 mm (11%) | 309 mm (41%) | 192 mm (26%) |
| sea level at the coast, 2° mean | 11 mm (5.5%) | 6.2 mm (3.1%) | 0.3 mm (0.1%) |

That run takes 12 s and 0.9 GB at lmax 48, 22 s and 1.7 GB at 64, 46 s
and 4.0 GB at 96 and 83 s and 7.3 GB at 128 on four threads: time and
memory grow as lmax squared, held by the sources' forces, a radial
vector over each degree's whole system for every order although a point
source touches one element. lmax 256 would take about 27 GB through
time (and the static problem about 38 GB at lmax 512). The trajectory
store of `kernel_postseismic` grows as lmax cubed (655 MiB at lmax 32,
about 5 GB at 64), so it stays at lmax 32, where its measurements, point
values, are far from converged: it shows the workflow and the kernels'
form, not converged sensitivities. `ExampleSeismic` runs at lmax 64 and
draws 2° means too, within a few per cent of converged there.

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

- **`no_ocean_tide_planet.json`** — the Sun's tide on the toy planet,
  elastic and radial, at lmax 32:
  `out/no_ocean_tide_planet/no_ocean_tide_planet.nc` and
  `out/no_ocean_tide_planet/no_ocean_tide_planet.txt`. The potential is
  the Sun's (GM = 1.32712e20 m^3 s^-2) at Mars's mean distance, 1.524 AU,
  at declination 25° N (about Mars's obliquity) over longitude 0°:
  A = 0.1287 m^2 s^-2 at the surface, so (2,0) 0.04735, (2,1) 0.09571 and
  (2,2) -0.10263 m^2 s^-2 (real, the Sun over the prime meridian),
  illustrative as the lunar tide above is. The figure reads the planet's
  tidal Love numbers off the exchange table, each order's coefficients
  over the potential's: h2 = 0.3390 and k2 = 0.1823, the three orders
  agreeing to rounding, and a bulge of about 12 mm.
- **`no_ocean_tide_planet_relax.json`** — a unit degree-2 potential
  switched on and held on the planet's Maxwell deck, as
  `no_ocean_1d_tide.json` is on the Earth's, at the same nodes out to
  500 kyr: the Love numbers relax from h2 = 0.34 and k2 = 0.18 to 1.83 and
  0.99, most of the way between 1 and 10 kyr, several times later than
  the Earth's under the same viscosity.
