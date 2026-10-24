# The sl3d program: configuration reference

`sl3d` runs the library's common capabilities from a JSON config, without
writing C++. Each run is one command and one config:

```
sl3d <command> <config.json>
```

| Command | What it computes | Writes |
| --- | --- | --- |
| `love-numbers` | the SI Love numbers of a radial model | a plain-text table |
| `fingerprint` | an elastic sea-level fingerprint on a present-day state | a surface file |
| `evolve1d` | 1D viscoelastic GIA under an ice history, on the deck's own viscosity | a surface file of histories |
| `evolve` | its 3D counterpart, the viscosity read from a planetmodel model file | a surface file of histories |
| `kernel1d` | adjoint sensitivity kernels of configured measurements, 1D | a directory of kernel files |
| `kernel` | their 3D counterpart | a directory of kernel files |

## Running sl3d

- `sl3d --help` prints the usage; `sl3d --version` prints the versions of
  sl3dpp and of the planetmodel it was built with, as
  `sl3d (sl3dpp <version>, planetmodel <version>)`.
- A run prints one summary line to standard error, as
  `sl3d <command>: <summary> in <seconds> s`; the summary of each command
  is given with it below.
- Exit status: 0 when the command finished (including an evolve run that
  stopped at its wall-time limit, below), 1 when it failed, with the error
  on standard error as `sl3d <command>: <message>`, and 2 for a usage
  error (a wrong number of arguments or an unknown command).
- The library runs threaded with OpenMP on every core by default; set
  `OMP_NUM_THREADS` to choose the thread count. One-thread runs are bitwise
  reproducible; threaded runs agree to round-off.

## Configs

A config is one JSON object, read strictly by planetmodel's JSON reader:

- A member that the command does not know is refused, so a misspelt member
  is an error rather than silently ignored. A missing required member, a
  value of the wrong type, a non-integer where an integer is wanted, and a
  word outside a member's choices are refused too. The error names the
  config file as given, the member's path within it and the fault, for
  example `run.json: change.width_deg: is not a positive angle`; a JSON
  syntax error names the line and column instead.
- Comments and trailing commas are not JSON and are refused.
- A relative path resolves against the directory holding the config, not
  the working directory; an absolute path is kept. The parent directories
  of an output are created as needed.
- Members whose values are words (`"rotation": "full"`) take exactly the
  words listed in the tables below.

### Units and conventions

- Files are SI throughout: lengths in metres, viscosities in Pa s, times in
  seconds unless a file's own scales say otherwise.
- Dates in configs are in kiloyears before present (`_kyr_bp` members),
  a kiloyear being 1000 Julian years, 3.15576e10 s. A date of D kyr BP
  is the time -D kyr: the time axes of the files sl3d writes are in seconds
  relative to the present, negative in the past.
- Latitudes and longitudes are in degrees, latitude north positive and
  longitude east positive; cap widths are in degrees.
- `rotation` chooses the rotational feedback: `"none"`, `"axial"` (the
  length-of-day change alone) or `"full"` (polar wander as well); the
  default is `"full"` wherever the member is accepted.

### Earth models

`model` names a deck: a table of knots, radius first, in which a repeated
radius marks a layer boundary. `model_format` says which kind:

| `model_format` | Header lines | Columns after the radius |
| --- | --- | --- |
| `"gia"` (the default) | 2 | `rho vpv vsv vph vsh eta viscosity`, planetmodel's GIA deck; viscosity `nan` in elastic and fluid layers |
| `"mineos"` | 3 | `rho vpv vsv qkappa qmu vph vsh eta`, or `rho vp vs qkappa qmu` for an isotropic model |
| `"elastic"` | 0 | `rho vpv vsv vph vsh eta`, the six-column elastic table of the test and example decks |

All values are SI. Blank lines and lines starting with `#` in the table are
skipped. The elastic structure always comes from the deck. `evolve1d` and
`kernel1d` take the viscosity from the deck's `viscosity` column, so they
need a deck that has one (`"gia"`): each layer holding a viscosity is a
Maxwell body, and a deck with no viscosity at all is refused. `evolve` and
`kernel` take the viscosity from the model file named by `viscosity`
instead, and ignore any viscosity column.

The radial mesh is built for the run's `lmax` from the deck alone; there
are no mesh members.

### Present-day states and ice histories

The ice comes in as planetmodel surface files (`planetmodel.surface/1`):
named scalar fields on one angular grid, static or on a time axis.
pyslfp's `export_ice_history` (`from pyslfp.ice import export_ice_history`)
writes one from ICE-5G, ICE-6G or ICE-7G, holding the `ice_thickness` and
`topography` histories in metres at the model's stored dates.

- A present-day state (`state.file`) must hold the fields `ice_thickness`
  and `topography`; an ice history (`ice_history.file`) must hold
  `ice_thickness`. The two may be the same file.
- Fields are resampled once onto the run's Gauss–Legendre grid at `lmax`,
  by each field's own lateral interpolation rule, and converted to SI by
  their declared dimensions (a field declaring none is used as it stands).
- A field on a time axis is read at any date by the file's own rule: a
  `node` field with linear time interpolation (what pyslfp writes) between
  its stored dates. Outside the file's span the first or last stored values
  are held, without a warning, so check that the file covers the dates
  asked of it. A static field is read at every date.
- The sea level of a state is built from its topography T and ice
  thickness I by pyslfp's construction: -T over the ocean (T < 0) and
  -T + I over land, raised by the floating fraction (917/1000) of the ice
  where ice sits on the ocean floor. The ocean function follows from it.
- An ice thickness change loads the Earth only where the ice is grounded:
  the load is 917 kg m^-3 times the thickness change times (1 - C), with C
  the ocean function of the state. The sea level equation is the linear
  one, its shorelines fixed at the state's.

### 3D viscosity models

`evolve` and `kernel` read the viscosity from a planetmodel model file
(`planetmodel.model/1`) with a field named `viscosity`, as planetmodel's
tomography conversion writes. The viscosity is split as
eta(r, theta, phi) = eta_bar(r) exp(delta(r, theta, phi)) about a radial
reference eta_bar, chosen by `viscosity.reference`:

- `"geometric_mean"` (the default): at each radius, the exponential of the
  lateral mean of ln eta, so delta has zero lateral mean. Works for any
  viscosity; needs a grid that carries quadrature weights.
- `"file_radial"`: the file's own radial reference, for a viscosity stored
  as a perturbation of one; refused when that reference varies laterally.

The file's body must be the deck's: its radial ends are checked, and each
layer of the deck the file gives a viscosity must lie within one layer of
the file. Each viscous layer is a Maxwell body; a wholly radial viscosity
runs as the 1D problem would.

## love-numbers

The per-degree Love numbers of the deck's elastic structure, degrees 0 to
`lmax`, in pyslfp's conventions (the generalised Love numbers of Al-Attar
et al. 2024).

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | path | required | the deck |
| `model_format` | word | `"gia"` | `"gia"`, `"mineos"` or `"elastic"` |
| `lmax` | integer >= 0 | required | the largest degree |
| `output` | path | required | the table to write |

The output is a plain-text table under a header of `#` lines: the program
and sl3dpp version, the creation time (UTC), the config, the deck and
`lmax`; then `radius_m` and `surface_gravity_m_s2` of the model's surface;
the five degree-zero axial numbers of the rotational feedback (`h_c`, `k_c`,
`m_u`, `m_phi`, `m_c`); the worst relative violation of the reciprocity
relations (`reciprocity_residual`); and the column names. One row follows
per degree:

```
l h_u l_u k_u h_phi l_phi k_phi h_v l_v k_v h_t l_t k_t
```

The `u` and `phi` channels are the traction and attraction parts of the
response to a unit surface load (kg m^-2), so the conventional load Love
numbers are their sums (h = h_u + h_phi, and likewise l and k); `v` is the
response to a unit tangential traction, and `t` the tidal response to a unit
external potential. h and l are surface displacements in metres and k the
potential perturbation in m^2 s^-2, per unit of the forcing. Degree 1 is in
the centre-of-mass frame.

Summary: `wrote <output> (degrees 0..<lmax>)`.

```json
{
  "model": "earth.deck",
  "lmax": 256,
  "output": "love_numbers.txt"
}
```

```json
{
  "model": "prem8.elastic.deck",
  "model_format": "elastic",
  "lmax": 1024,
  "output": "out/prem8_love_numbers.txt"
}
```

## fingerprint

The elastic sea-level fingerprint of an ice change applied to a present-day
state: one solve of the linear sea level equation, with the Love numbers of
the deck's elastic structure at `lmax`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | path | required | the deck |
| `model_format` | word | `"gia"` | `"gia"`, `"mineos"` or `"elastic"` |
| `lmax` | integer >= 1 | required | the degree of the solve and of its grid |
| `state.file` | path | required | a surface file holding `ice_thickness` and `topography` |
| `state.date_kyr_bp` | number | 0 | the date at which the state is read, kyr BP |
| `change.type` | word | required | `"cap_fraction"` or `"cap_thickness"` |
| `change.latitude` | number | required | the cap's centre, degrees |
| `change.longitude` | number | required | the cap's centre, degrees |
| `change.width_deg` | number > 0 | required | the cap's width w, degrees |
| `change.fraction` | number | required for `cap_fraction` | the fraction of the local ice thickness added (negative to melt) |
| `change.thickness_m` | number | required for `cap_thickness` | the thickness added at the cap's centre, metres (negative to melt) |
| `rotation` | word | `"full"` | `"none"`, `"axial"` or `"full"` |
| `output` | path | required | the surface file to write |

The cap is Gaussian in great-circle distance d from its centre, weighted
exp(-d^2 / (2 w^2)) with no cut-off. A `cap_fraction` change is
`fraction` times the weight times the state's ice thickness, so it acts
only where there is ice; a `cap_thickness` change is `thickness_m` times
the weight. Either loads the Earth only where the ice is grounded. Giving
the amount member of the other type is refused.

The output is a surface file of static SI fields on the run's
Gauss–Legendre grid:

| Field | Unit | Meaning |
| --- | --- | --- |
| `ice_change` | m | the ice thickness change applied (before the grounded-ice mask) |
| `sea_level_change` | m | the sea-level change |
| `displacement` | m | the radial surface displacement |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |
| `ocean_function` | 1 | the ocean function of the state |

Its metadata record the config, deck and state paths, `date_kyr_bp`, the
`rotation`, the angular velocity perturbation as `omega_x_rad_s`,
`omega_y_rad_s` and `omega_z_rad_s` (rad s^-1), and the number of sea
level iterations (`sle_iterations`). A solve that does not converge is an
error.

Summary: `wrote <output> (<n> SLE iterations)`.

```json
{
  "model": "earth.deck",
  "lmax": 128,
  "state": {"file": "ice7g.nc"},
  "change": {"type": "cap_thickness", "latitude": -75, "longitude": 250,
             "width_deg": 8, "thickness_m": -10},
  "output": "fingerprint.nc"
}
```

```json
{
  "model": "earth.deck",
  "model_format": "gia",
  "lmax": 256,
  "state": {"file": "ice7g.nc", "date_kyr_bp": 0.0},
  "change": {"type": "cap_fraction", "latitude": 72, "longitude": 320,
             "width_deg": 16, "fraction": -0.1},
  "rotation": "full",
  "output": "out/greenland_fingerprint.nc"
}
```

## evolve1d and evolve

Viscoelastic GIA under an ice history, through the sea level equation with
rotational feedback, over a window of time. `evolve1d` runs a spherically
symmetric Earth whose viscosity is the deck's own column; `evolve` runs the
laterally varying viscosity of a model file (`viscosity`), each step solving
the 3D fixed point about the radial reference.

The run starts from the state at the window's start: the background is
built from `state.file` read at `times.start_kyr_bp`, and the load at each
time node is the change of grounded ice thickness in `ice_history.file`
from its value at the start. The outputs are changes from the window's
start.

### The run members

These members are shared by `evolve1d`, `evolve`, `kernel1d` and `kernel`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | path | required | the deck |
| `model_format` | word | `"gia"` | `"gia"`, `"mineos"` or `"elastic"` |
| `lmax` | integer >= 1 | required | the degree of the run and of its grid |
| `viscosity.file` | path | required by `evolve` and `kernel`, refused by `evolve1d` and `kernel1d` | the planetmodel model file of the 3D viscosity |
| `viscosity.reference` | word | `"geometric_mean"` | `"geometric_mean"` or `"file_radial"`: the radial reference (3D viscosity models, above) |
| `state.file` | path | required | a surface file holding `ice_thickness` and `topography`, read at the window's start |
| `ice_history.file` | path | required | a surface file holding `ice_thickness`; may be the state's file |
| `times.start_kyr_bp` | number | required | the window's start, kyr BP |
| `times.end_kyr_bp` | number | 0 | the window's end, kyr BP; must be later than the start |
| `times.nodes` | integer >= 2 | required | the number of time nodes, both ends included, evenly spaced |
| `scheme` | word | `"exponential_trapezoid"` | the time step: `"exponential_trapezoid"` (second order) or `"backward_euler"` |
| `rotation` | word | `"full"` | `"none"`, `"axial"` or `"full"` |
| `moments` | word | `"input"` | `"input"`: the Earth's standard moments of inertia; `"hydrostatic"`: their difference set from the model's own hydrostatic fluid limit, which full rotational feedback needs when the standard moments sustain less oblateness than the model's tidal response |
| `rtol` | number | 1e-9 | the relative tolerance of the sea level solves (and, for the kernels, of the adjoint solves) |
| `coupled.rtol` | number | 1e-9 | `evolve` and `kernel`: the relative tolerance of each step's combined fixed point |
| `coupled.max_iterations` | integer | 60 | `evolve` and `kernel`: its iteration limit |

The time nodes are `nodes` evenly spaced dates from `start_kyr_bp` to
`end_kyr_bp`; the step is not limited by the shortest Maxwell time. The
1D commands accept `coupled` but have no use for it. In `evolve`, a node
whose fixed point does not converge within `coupled.max_iterations` is an
error.

### The evolve members

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `checkpoint.directory` | path | required in `checkpoint` | where the checkpoints go |
| `checkpoint.every` | integer >= 1 | required in `checkpoint` | the checkpoint stride, in time nodes |
| `checkpoint.max_walltime_min` | number | 0 | stop cleanly once a checkpoint is written after this many minutes; 0 for no limit |
| `restart` | boolean | `true` | whether to resume from a checkpoint found in `checkpoint.directory`; `false` starts over, replacing the checkpoints there |
| `output` | path | required | the surface file to write |

Without a `checkpoint` section nothing is written until the run finishes.

### Output

A surface file on the run's Gauss–Legendre grid with one slice per time
node, its time axis in seconds relative to the present (negative in the
past). Its fields are SI node fields with linear time interpolation:

| Field | Unit | Meaning |
| --- | --- | --- |
| `sea_level_change` | m | the sea-level change from the window's start |
| `displacement` | m | the radial surface displacement |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |

Its metadata record the config, deck and ice-history paths (and the
viscosity file's, for `evolve`), the node dates as `dates_kyr_bp`, and the
angular velocity perturbation at each node as `omega_x_rad_s`,
`omega_y_rad_s` and `omega_z_rad_s` (rad s^-1).

Summary: `wrote <output> (<n> nodes, <start> to <end> kyr BP)`, ending in
`, resumed` when the run resumed from a checkpoint, and in
`, replacing <k> earlier checkpoint(s)` when a run starting over removed
some.

### Checkpoints and restarts

With a `checkpoint` section, the run writes a checkpoint after every
`checkpoint.every`-th node it steps and after the last node. Each is a
file `node_NNNNNN.ckpt` in `checkpoint.directory`, named by its node,
holding the run's state there and the histories so far. It is written to
a temporary file and renamed into place, so an interrupted write leaves no
partial checkpoint, and only the newest two are kept.

When a run starts and finds a checkpoint in the directory, it resumes from
the newest one, with the histories up to that node restored. A resumed run
reproduces an uninterrupted one (bitwise, on one thread). Rerunning a
finished run whose last checkpoint is at the final node just writes the
output again.

A checkpoint is bound to its run: it records a hash of the members that
decide the computation (everything but `checkpoint`, `restart` and
`output`) and of the contents of the files the config names (the deck, the
state, the ice history and, for `evolve`, the viscosity file). A run is
refused, with an error naming the checkpoint, rather than resuming another
run's state when any of these differ: another time grid, tolerance or
model, or the same config over an input file that has changed since. The
config's layout, its output, the checkpoint stride and the wall-time limit
may all change between a run and its resumption, and the paths may change
as long as the files' contents do not, so a run directory can be moved. A
checkpoint of the other command (`evolve1d` against `evolve`) is refused
too. The files are raw binary, for the machine that wrote them.

With `restart` set to `false` the run starts over: it first removes the
checkpoints it finds in the directory, then writes its own. Only this
program's checkpoint files (`node_NNNNNN.ckpt`, and the `.node_NNNNNN.partial`
it renames) are read, pruned or removed; anything else in the directory is
left alone. Leave `restart` at `true` in a chain of jobs that is meant to
resume.

`checkpoint.max_walltime_min` serves batch systems with a time limit. After
writing a checkpoint, a run that has been stepping for longer than the limit
and has nodes still to go stops cleanly, with the summary
`checkpointed node <n> of <last> to <directory> at the wall-time limit;
rerun the same config to resume`. It exits with status 0 and writes no
output yet, so a script can tell a stopped run from a finished one by the
summary or by whether `output` exists; rerunning the same config resumes
it, so a chain of jobs running one config completes the run. The clock starts with the time stepping, after
the model and files are read.

### Examples

```json
{
  "model": "earth.deck",
  "lmax": 64,
  "state": {"file": "ice7g.nc"},
  "ice_history": {"file": "ice7g.nc"},
  "times": {"start_kyr_bp": 26.0, "nodes": 53},
  "output": "histories.nc"
}
```

```json
{
  "model": "prem8.elastic.deck",
  "model_format": "elastic",
  "lmax": 128,
  "viscosity": {"file": "s40rts_viscosity.nc", "reference": "geometric_mean"},
  "state": {"file": "ice7g.nc"},
  "ice_history": {"file": "ice7g.nc"},
  "times": {"start_kyr_bp": 26.0, "end_kyr_bp": 0.0, "nodes": 105},
  "scheme": "exponential_trapezoid",
  "rotation": "full",
  "moments": "hydrostatic",
  "rtol": 1e-9,
  "coupled": {"rtol": 1e-8, "max_iterations": 80},
  "checkpoint": {"directory": "checkpoints", "every": 4,
                 "max_walltime_min": 1380},
  "restart": true,
  "output": "out/histories_3d.nc"
}
```

The first is an `evolve1d` config; the second needs `evolve`.

## kernel1d and kernel

Adjoint sensitivity kernels of configured measurements. The forward run is
the evolve run of the same run members; then, for each measurement, one
reverse (adjoint) sweep gives its sensitivity to the ice history and to the
viscosity. `kernel1d` runs the 1D problem, by the continuous adjoint or the
exact discrete transpose; `kernel` runs the 3D problem by the continuous
adjoint.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| the run members | | | as for evolve, above |
| `record_stride` | integer >= 1 | 1 | record the forward state at every `record_stride`-th node (and the last); the sweep rebuilds the states in between |
| `adjoint` | word | `"continuous"` | `"continuous"` or `"discrete"`; `kernel` takes `"continuous"` alone |
| `trajectory.directory` | path | none | `kernel` alone: spill the recorded states to this directory instead of holding them in memory |
| `measurements` | array | required | the measurements, at least one (below) |
| `output` | path | required | the directory the kernel files go into |

The kernel commands have no checkpoints. Recording every n-th of N nodes
holds about N/n + n forward states during the sweep, fewest near n = sqrt(N);
in 3D each is large (see the README's notes on memory). With
`trajectory.directory`, `kernel` writes each recorded state to a file
`node_NNNNNN.traj` there as the forward run passes it, and the sweep reads
them back one at a time: the lever for large `lmax`. The files are raw
binary and are left in place after the run. `kernel1d` refuses a
`trajectory` section, its trajectory fitting in memory.

### Measurements

Each element of `measurements` is an object with a `type` and a `name`.
The `name` (by default `measurement_<k>`, with k the element's position
from 0) is the stem of the measurement's files, so names should be distinct
and usable as file names. A measurement taken on one date reads the outputs
at that date, which must be a time node of the run (`times`). Every
measurement is held to the run when the config is read, before anything
runs: a date that is not a node, a coefficient that does not exist at
`lmax`, or a decay window holding fewer than four nodes is refused then,
naming the member, rather than after the forward run.

| `type` | Members | Value (SI) |
| --- | --- | --- |
| `sea_level_cap` | `latitude`, `longitude`, `width_deg`, `date_kyr_bp` | the cap-weighted mean sea-level change, m |
| `sea_level_point` | `latitude`, `longitude`, `date_kyr_bp` | the sea-level change at the point, m |
| `displacement_cap` | `latitude`, `longitude`, `width_deg`, `date_kyr_bp` | the cap-weighted mean radial displacement, m |
| `displacement_point` | `latitude`, `longitude`, `date_kyr_bp` | the radial displacement at the point, m |
| `potential_cap` | `latitude`, `longitude`, `width_deg`, `date_kyr_bp` | the cap-weighted mean potential perturbation, m^2 s^-2 |
| `potential_point` | `latitude`, `longitude`, `date_kyr_bp` | the potential perturbation at the point, m^2 s^-2 |
| `stokes_coefficient` | `degree`, `order`, `date_kyr_bp` | a real orthonormal coefficient of the potential perturbation, m^2 s^-2 |
| `rotation` | `component`, `date_kyr_bp` | a component of the angular velocity perturbation, rad s^-1 |
| `decay_time` | `latitude`, `longitude`, `width_deg`, `window_start_kyr_bp`, `window_end_kyr_bp`, `time_uncertainty_kyr`, `value_uncertainty_m` | the relaxation time of a site's sea-level series, s |

- `latitude`, `longitude` (degrees) and `width_deg` (degrees, > 0) are
  required where listed; `date_kyr_bp` defaults to 0, the present.
- A cap is Gaussian in great-circle distance d from its centre, weighted
  exp(-d^2 / (2 w^2)) with w = `width_deg` and normalised to unit area
  integral, so it reads a weighted mean. A point value is the band-limited
  value at `lmax` (the grid's reproducing kernel); a narrow cap is often the
  steadier datum.
- `stokes_coefficient`: `degree` l and `order` m are required integers,
  0 <= l <= `lmax` and |m| <= l, with m >= 0 the cosine term and m < 0 the sine term of |m| (the pyshtools
  convention); the value is the coefficient of the potential perturbation
  in the orthonormal real harmonics (each squared integrates to 1 over the
  unit sphere), so a GRACE-style geodetic coefficient differs from it by a
  normalisation constant alone.
- `rotation`: `component` is `"x"`, `"y"` or `"z"`; x and y drive polar
  motion, z the length of day.
- `decay_time`: the cap-mean sea-level change at every node from
  `window_start_kyr_bp` (required) to `window_end_kyr_bp` (default 0) is
  fitted by a e^(-t / tau) + b, following Al-Attar & Tromp (2014,
  Appendix C), and the datum is tau. Both window ends must be time nodes,
  the start the earlier, and the window must hold at least four nodes, the
  fit's minimum. `time_uncertainty_kyr` (>= 0, required) and
  `value_uncertainty_m` (> 0, required) are the one-sigma timing and
  series errors; only their ratio shapes the fit, and a zero timing
  uncertainty holds the times at the nodes.

### Output

In the `output` directory:

- `measurements.txt`: under a header of `#` lines (program and version,
  creation time, config), one row per measurement:
  `name type value_si`, the forward value in the SI unit of the table
  above; a `decay_time` row adds `tau_kyr amplitude_m offset_m` from the fit.
- `<name>_ice_kernels.nc`: the ice-history kernels, a surface file on the
  run's Gauss–Legendre grid. Its time axis is every time node (seconds
  relative to the present), and its instants are the measurement's own
  nodes. Each field is a density per square metre of surface, in the
  measurement's SI unit per unit of the perturbation, so that
  dJ = integral of K delta dS, summed over the nodes, intervals or instants
  for the fields that have them:

  | Field | Support | Meaning |
  | --- | --- | --- |
  | `thickness_dual` | node | dJ per metre of grounded ice thickness at each node |
  | `rate_dual` | interval | dJ per metre per second of ice thickness rate on each interval |
  | `instantaneous_dual` | instant | the delta amplitude at each measurement instant |
  | `thickness_baseline_dual` | static | dJ per metre of the initial ice thickness |
  | `initial_sea_level_dual` | static | dJ per metre of the initial sea level |

- `kernel1d`: `<name>_viscosity_kernel.txt`, the radial viscosity kernel of
  each Maxwell branch: under a header, columns `r_m` and `K_branch_<b>`, in
  SI, so that dJ = integral of K(r) delta eta(r) dr with eta in Pa s. The
  rows are the radial mesh's nodes, element by element, so the radius of a
  shared element edge appears twice.
- `kernel`: `<name>_viscosity_kernels.nc`, the 3D log-viscosity kernels as
  a planetmodel model file: a field `K_ln_eta_<b>` per branch, a density
  per unit volume in SI, so that dJ = integral of K delta(ln eta) dV. It is
  written on the angular grid of the run's 3D coupling at the viscous
  elements' nodes, one part per contiguous run of viscous elements, with the
  model's density alongside.

Summary: `wrote <n> measurement(s) of kernels to <output>`.

### Examples

```json
{
  "model": "earth.deck",
  "lmax": 32,
  "state": {"file": "ice7g.nc"},
  "ice_history": {"file": "ice7g.nc"},
  "times": {"start_kyr_bp": 20.0, "nodes": 41},
  "measurements": [
    {"type": "sea_level_point", "name": "fremantle",
     "latitude": -32.05, "longitude": 115.74}
  ],
  "output": "kernels"
}
```

```json
{
  "model": "earth.deck",
  "lmax": 64,
  "state": {"file": "ice7g.nc"},
  "ice_history": {"file": "ice7g.nc"},
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 41},
  "rotation": "full",
  "moments": "hydrostatic",
  "record_stride": 6,
  "adjoint": "discrete",
  "measurements": [
    {"type": "sea_level_cap", "name": "barbados", "latitude": 13.1,
     "longitude": -59.6, "width_deg": 2.0, "date_kyr_bp": 6.0},
    {"type": "displacement_point", "name": "uplift_hudson",
     "latitude": 60.0, "longitude": -85.0},
    {"type": "potential_cap", "name": "geoid_canada", "latitude": 60.0,
     "longitude": -90.0, "width_deg": 10.0},
    {"type": "potential_point", "name": "geoid_point", "latitude": 60.0,
     "longitude": -90.0},
    {"type": "displacement_cap", "name": "uplift_fennoscandia",
     "latitude": 64.0, "longitude": 20.0, "width_deg": 5.0},
    {"type": "stokes_coefficient", "name": "c20", "degree": 2, "order": 0},
    {"type": "rotation", "name": "lod", "component": "z"},
    {"type": "decay_time", "name": "rebound", "latitude": 60.0,
     "longitude": -85.0, "width_deg": 3.0, "window_start_kyr_bp": 8.0,
     "window_end_kyr_bp": 0.0, "time_uncertainty_kyr": 0.0,
     "value_uncertainty_m": 0.5}
  ],
  "output": "out/kernels_1d"
}
```

```json
{
  "model": "prem8.elastic.deck",
  "model_format": "elastic",
  "lmax": 64,
  "viscosity": {"file": "s40rts_viscosity.nc"},
  "state": {"file": "ice7g.nc"},
  "ice_history": {"file": "ice7g.nc"},
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 81},
  "coupled": {"rtol": 1e-9, "max_iterations": 60},
  "record_stride": 9,
  "trajectory": {"directory": "/scratch/trajectory"},
  "measurements": [
    {"type": "sea_level_cap", "name": "barbados", "latitude": 13.1,
     "longitude": -59.6, "width_deg": 2.0, "date_kyr_bp": 10.0}
  ],
  "output": "out/kernels_3d"
}
```

The first two are `kernel1d` configs; the third needs `kernel`.
