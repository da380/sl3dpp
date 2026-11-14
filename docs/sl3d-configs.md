# The sl3d program: configuration reference

`sl3d` runs the library's common capabilities from a JSON config, without
writing C++. Each run is one command and one config:

```
sl3d <command> <config.json>
```

| Command | What it computes | Writes |
| --- | --- | --- |
| `love-numbers` | the SI Love numbers of a radial model | a plain-text table |
| `solve` | the sea level equation, from a state (the sea level and the ice at one instant) under any mix of an ice change, surface loads, a tide and seismic sources, with frozen or migrating shorelines: in the static problem the solve at the state's date on the elastic Earth (the elastic fingerprint of an ice change, the co-seismic sea level change of a fault), in the quasi-static problem the run through time from the state at an end of the window, its nodes dates before present or times on the run's own clock; without a state, the response of a body without an ocean to the loads, the tide and the sources, elastic or viscoelastic, with the rotational feedback as asked; without gravity, a radial body's response to surface loads, by their weight, and seismic sources; 1D or 3D as the model's moduli (and, through time, viscosity) make it | a surface file (of histories, in the quasi-static problem), and optionally coefficient tables and, static, a model file |
| `kernel` | adjoint sensitivity kernels of configured measurements of the problems `solve` solves, read from the same config with a `kernels` block: in the ice history, surface loads, a tide, the moment tensors of seismic sources, the viscosity and the elastic moduli; 1D or 3D as `solve`'s run | a directory of kernel files |

## Running sl3d

- `sl3d --help` prints the usage; `sl3d --version` prints the versions of
  sl3dpp and of the planetmodel it was built with, as
  `sl3d (sl3dpp <version>, planetmodel <version>)`.
- A run prints one summary line to standard error, as
  `sl3d <command>: <summary> in <seconds> s`; the summary of each command
  is given with it below.
- Exit status: 0 when the command finished (including a `solve` run that
  stopped at its wall-time limit, below), 1 when it failed, with the error
  on standard error as `sl3d <command>: <message>`, and 2 for a usage
  error (a wrong number of arguments or an unknown command).
- The library runs threaded with OpenMP on every core by default; set
  `OMP_NUM_THREADS` to choose the thread count. Two runs of one config
  agree to rounding, about 1e-14 relative, threaded or not: many outputs
  come out bitwise the same (the forward run of
  `examples/configs/kernel_postseismic.json`, say), while others need not
  (its kernel tables, the lmax-64 3D ICE-7G fingerprint), each process
  planning its FFTs by timing them (FFTW's measured plans), which may
  round differently from run to run. Within one process the runs are
  bitwise reproducible, which the tests check. Placing the threads, the
  memory a run needs and long runs are in [running.md](running.md).

### Over MPI ranks

Built with the CMake option `SL3DPP_WITH_MPI`, `solve` and `kernel` run
under `mpiexec` over several ranks, the same config on every rank:

```
OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close \
  mpiexec -n 4 -bind-to core:32 sl3d solve sea_level_3d.json
```

(the binding flags are MPICH's; another MPI names them otherwise). The
run is split over radius, one contiguous radial segment per rank, the cuts
balancing the transform work ([radial-partitioning.md](radial-partitioning.md) §7 and §9):
each rank holds and steps its own segment's internal variables, with its
threads inside, and every rank reads the inputs and solves the sea level
equation alike, the first rank's results serving all: the coefficients
of the total load and of the centrifugal and tidal potential, which force
every segment, the surface response and, under migrating shorelines, the
ocean function. A 1D run (the static
problem on a radial Earth, a radial run through time) is the first rank's
alone, as a run of one process, its checkpoints in `checkpoint.directory`
itself; the other ranks return at once. One rank per socket (or per
node) with threads inside is the intended shape. Rank 0 writes every
output and prints the summary, with `on <n> ranks` appended. The radial
reference model written beside a 3D run through time is stepped on rank
0 alone, after the run. A failure on any rank ends the whole job, the
message naming the rank. A 1D `kernel` run is the first rank's alone,
as `solve`'s, and a 3D one refuses on several ranks what its sweeps form
on one process (kernel, below). `love-numbers` refuses several ranks
(exit status 2).

The results agree with a one-process run to round-off of the radially
substructured solves (about 1e-12 relative), not bitwise, and within the
solves' tolerances where an iterative solve closes a pass earlier or
later on the ranks (up to about 1e-10 relative at the default
tolerances of 1e-9). Each rank's
checkpoints go to its own subdirectory, `rank_RRRR` of
`checkpoint.directory`, holding its own segment's state (rank 0's holds
the histories too, the others' each node's iteration counts alone), and
the trajectory store's states go to `rank_RRRR`
of `trajectory.directory` likewise, rank 0 writing its manifest, its start
and its surface records; a store is read over the count of ranks that
recorded it, and the same parts of the mesh, only (The trajectory block,
below). A run resumes from the newest node whose
checkpoint every rank has (of the first rank's newest two), and over the
same number of ranks only: the rank count is part of the checkpoints'
hash, so a resumption over another count is refused like another run's.
Where the ranks have no checkpoint in common (a run stopped while they
wrote its first, say), each rank's checkpoints and any kept initial state
are stale: they are removed and the run starts over, the summary saying
`, replacing <k> earlier checkpoint(s)` where the first rank removed some.

## Configs

A config is one JSON object, read strictly by planetmodel's JSON reader:

- A member that the command does not know is refused, so a misspelt member
  is an error rather than silently ignored. A missing required member, a
  value of the wrong type, a non-integer where an integer is wanted, and a
  word outside a member's choices are refused too. The error names the
  config file as given, the member's path within it and the fault, for
  example `run.json: forcing.ice_change[0].cap_thickness.width_deg: is not a
  positive angle`; a JSON syntax error names the line and column instead.
- Comments and trailing commas are not JSON and are refused.
- A relative path resolves against the directory holding the config, not
  the working directory; an absolute path is kept. The parent directories
  of an output are created as needed.
- Members whose values are words (`"rotation": "full"`) take exactly the
  words listed in the tables below.

### Units and conventions

- Files are SI throughout: lengths in metres, viscosities in Pa s, times in
  seconds unless a file's own scales say otherwise.
- Dates in configs are in kiloyears before present (`_kyr_bp` members:
  the dated `times`, the state's date, the dates of the forcing's terms,
  their histories and onsets, and the measurements' dates), a kiloyear
  being 1000 Julian years, 3.15576e10 s. A date of D kyr BP is the time
  -D kyr: the time axes of the files with dates that sl3d writes are in
  seconds relative to the present, negative in the past.
- Times counted forwards on a run's own clock are the other form, with a
  `unit` beside them (`times.unit`: `"kyr"`, `"yr"`, a Julian year of
  3.15576e7 s, or `"day"`, 86400 s), every time of the run and its terms
  in that unit under unsuffixed names (`start`, `end`, `at`, `history`,
  `onset`, `time_origin`, and the static problem's `date` beside its own
  `unit`). The time axis of the file of such a run is its clock in
  seconds (solve, The run through time). A time in kyr without a unit
  (`start_kyr`, `end_kyr`, `at_kyr`, `history_kyr`, `onset_kyr`,
  `time_origin_kyr`, a field's `date_kyr`) is no form's, and is refused,
  the error naming the member to give.
- Latitudes and longitudes are in degrees, latitude north positive and
  longitude east positive; cap widths and disk radii are in degrees.
- `rotation` chooses the rotational feedback: `"none"`, `"axial"` (the
  length-of-day change alone) or `"full"` (polar wander as well); the
  default is `"full"` wherever the member is accepted, and `"none"` in a
  run without gravity, which takes no other (solve, Without gravity).

### Earth models

`model.deck` names a deck: a table of knots, radius first, in which a
repeated radius marks a layer boundary. `model.format` says which kind
(`love-numbers` takes the two as `model` and `model_format`):

| `model.format` | Header lines | Columns after the radius |
| --- | --- | --- |
| `"maxwell"` (the default; `"gia"` names the same format) | 2 | `rho vpv vsv vph vsh eta viscosity`, planetmodel's Maxwell deck; viscosity `nan` in elastic and fluid layers |
| `"prony"` | 2 (and planetmodel's naming lines) | `rho vpv vsv vph vsh eta`, then `relaxation_strength_k relaxation_time_k` for k = 1 to N: planetmodel's prony deck, a generalised Maxwell body (its format page, `docs/source/formats/prony_deck.rst`); `nan` in every branch column of an elastic or fluid layer |
| `"burgers"` | 2 (and planetmodel's naming lines) | `rho vpv vsv vph vsh eta viscosity kelvin_viscosity kelvin_modulus_fraction`: planetmodel's burgers deck, each layer elastic, a Maxwell body (the viscosity alone) or a Burgers body (all three; `docs/source/formats/burgers_deck.rst`) |
| `"mineos"` | 3 | `rho vpv vsv qkappa qmu vph vsh eta`, or `rho vp vs qkappa qmu` for an isotropic model |
| `"elastic"` | 0 | `rho vpv vsv vph vsh eta`, the six-column elastic table of the test and example decks |

All values are SI. Blank lines and lines starting with `#` in the table are
skipped. A prony or burgers deck is read by planetmodel's own reader of
it, its column line and naming lines and its values held to planetmodel's
checks (a prony deck's strengths in [0, 1] summing to at most one, its
times positive; a burgers deck's layers one of its three bodies), and
needs a build against planetmodel 1.6.0 or later, which reads them; an
older one refuses the format at `model.deck`. The elastic structure always
comes from the deck, a prony or burgers deck's elastic values being the
unrelaxed ones. The quasi-static problem (`solve`, `kernel`) takes the
viscosity from the model file of `model.viscosity` when given, ignoring
any viscosity column, and from the deck otherwise, which then needs a
deck that states a rheology (`"maxwell"`, `"prony"` or `"burgers"`): a
Maxwell deck's layer holding a viscosity is a Maxwell body (and, where
`model.burgers` names it, a Burgers body: The model block, below); a
burgers deck's layers are its Maxwell and Burgers bodies, a Burgers
body's Kelvin element the layer's `kelvin_viscosity` and
`kelvin_modulus_fraction` times its unrelaxed shear modulus (two exact
Prony branches, `BurgersPronyTerms`); a prony deck's layers are its
branches, branch k's modulus the strength times the unrelaxed shear
modulus and its relaxation time the deck's, a strength of 0 leaving it
inactive there, any number of branches and a relaxed modulus (strengths
summing below one) included; strengths summing past one by the rounding
a prony deck allows (1e-8 a branch, the eight digits planetmodel writes)
are taken to sum to one. A deck with no viscosity at all is refused. The
static problem uses no viscosity.

**Joining the knots.** Each column is a function of the radius on each
layer through its knots: the elastic columns (and any other) by the
library's cubic splines, not-a-knot as planetmodel's on a layer of four
knots or more (natural on three, where planetmodel takes the parabola;
two knots are a line under both), and the columns of planetmodel's
viscoelastic decks as planetmodel joins them, by its kinds for them
(`planetmodel::DeckLayers`), so that such a deck read here and by
planetmodel is one body: a viscosity, a Kelvin element's viscosity
and a Prony branch's relaxation time in their logarithm by straight
lines (`log-linear`: geometric between knots, so positive, and following
a viscosity that changes by orders of magnitude within a layer), a
strength and a Kelvin modulus fraction by straight lines. A deck is a
table dense enough in knots that the interpolation between them does not
matter in practice: the kinds are a default for joining knots, not part
of the model, and a column constant through a layer is that constant
under any of them. A deck's elastic and viscous columns, joined thus,
make one body whatever deck states it at the knots: a prony deck of one
branch made from a Maxwell deck (its time the knots' viscosity over the
shear modulus) is the Maxwell deck's body at the knots, and between them
differs at second order in their spacing (the post-seismic deck of the
examples, 8 knots a layer, moves a run to 1000 years by 1.6e-5 of its
fields).

The radial mesh is built for the run's degree, `discretisation.lmax`
(The discretisation block, below), from the deck alone; there are no mesh
members.

### The problem

Every config of `solve` and `kernel` names its problem in the required
top-level member `problem`:

| `problem` | What is solved | `times` |
| --- | --- | --- |
| `"static"` | the problem at one instant: the solve at the state's date on the elastic Earth, or without a state the body's elastic response, and its kernels | refused |
| `"quasi_static"` | the problem stepped through the nodes of a `times` block: GIA through time, or without a state the body's viscoelastic response, and its kernels | required |

The rheology is the `model` block's business, and nothing is implied by
the presence or absence of another block: a config without `problem` is
refused, and so is a static one with a `times` block or a quasi-static
one without. A member only the
quasi-static problem takes (a viscosity file, an amplitude history, an
onset, the time step, the initial sea level's calibration, checkpoints,
the state's end of the window, ...) is refused in the static problem,
and a member only the static problem takes (a term's date; the state's
date, which a run counting forwards takes too) in the quasi-static one,
the error naming the problem the member needs.

### File inputs

A member read from a file is a file input, an object of

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `file` | path | required | the file |
| `format` | word | `"planetmodel_surface"` | the file's type: `"planetmodel_surface"`, a planetmodel surface file, is the only one |
| `variable` | string | the member's own quantity | the variable read: `ice_thickness` for `ocean.ice` and a `thickness` term, `ice_thickness_change` for a `change` term, `sea_level` for `ocean.sea_level`, `topography` for `ocean.topography`, `land` for `ocean.land`, `load` for a `field` load term |
| `outside_span` | word | `"refuse"` | what a read of a field on a time axis at a date outside the file's span does: `"refuse"`, an error; `"hold"`, the field's first or last stored values held there. Refused on a static field (a file without a time axis, or a field of `static` support), which no read leaves |

When the input is read is its member's business: the state's members
(`ocean.sea_level` or `ocean.topography`, `ocean.ice`, `ocean.land`) at
the `ocean` block's instant, `ocean.date_kyr_bp` in the static problem
and in a run counting forwards, and an end of the window, `ocean.at`, in
a dated run (a land on a time axis there under frozen shorelines, and at
every node's date under migrating ones); a term of the ice change, and a
`field` load term, at its own `date_kyr_bp` beside the file in the static
problem (a `field` term at its `date` on the file's own axis instead,
where it gives one) and at each node's date in a dated run; a `field`
term on the run's clock, from its `time_origin`, in a run counting
forwards (The state and the ice change, below; solve, The forcing
block). A variable the file does not hold is refused, the error listing
the variables the file does hold. Conversion from other sources (ICE-NG,
other formats) happens outside sl3d, in pyslfp.

A planetmodel surface file (`planetmodel.surface/1`) holds named scalar
fields on one angular grid, static or on a time axis. pyslfp's
`export_ice_history` (`from pyslfp.ice import export_ice_history`) writes
one from ICE-5G, ICE-6G or ICE-7G, holding the `ice_thickness` and
`topography` histories in metres at the model's stored dates: such a file
is a source of a state, its ice and its topography at one date, and of an
ice history, a `thickness` term of the ice change, at once, and may be
named by all three.

- In the static problem a file with a time axis needs a date (the slice
  read: the state's for the state's members, a term's own for a term),
  and a file without one refuses a date, its fields being the same at
  every date.
- Fields are resampled once onto the run's Gauss–Legendre grid at `lmax`,
  by each field's own lateral interpolation rule, and converted to SI by
  their declared dimensions (a field declaring none is used as it stands).
- A field on a time axis is read at any date within the file's span by
  the file's own rule: a `node` field with linear time interpolation (what
  pyslfp writes) between its stored dates. A read at a date outside the
  span is refused, every read alike (the state at its instant, the
  calibration's target at a calibrated run's end among them, a term at a
  node's date, the land), the error naming the member, the file's span,
  the date asked, in kyr BP or, for a `field` term read on the run's own
  clock or at its `date`, in its unit on the file's own axis, and the read;
  with `"outside_span": "hold"` on the input the field's first or last
  stored values are held there instead (at a calibrated run's end, the
  calibration's target is then the file's last state). A date within
  1e-9 of the span's ends (of the larger of its length and its ends'
  magnitudes, the tolerance a measurement's date is located on the nodes
  with) is read at the end itself, so that rounding (node dates formed
  from the window, a file's axis stored in years or seconds) raises no
  refusal: a run from 21 to 0 kyr BP reads a file spanning 21 to 0 kyr BP
  whatever unit its axis is stored in. The runs through time hold their
  files to their spans when they read them, before the first step. A
  static field is read at every date.
- Inputs naming one file (of one format) read and hold it once.
- A value that is not a number (a surface file marks uncovered points
  so) is refused in the state and in the ice change: fill the file before
  the run.

### The state and the ice change

A run of the sea level equation (`solve`, `kernel`) takes two
kinds of input, kept apart. The STATE is what is on the body at one
instant: the sea level of the sea level equation and the ice, the `ocean`
block's (`ocean.sea_level` or `ocean.topography`, and `ocean.ice`), with
the one instant of the whole block: its date, `ocean.date_kyr_bp`, in the
static problem, and an end of the window, `ocean.at`, in the
quasi-static one; a run counting forwards on its own clock, which has no
calendar, reads its state's files at their date `ocean.date_kyr_bp` too,
a snapshot put at the window's start. The FORCING is what happens: an ice
thickness CHANGE is one kind of it, the sum of the terms of the list
`forcing.ice_change`, beside surface loads, a tide and seismic sources
(solve, The forcing block, below). The ice at a time is the state's ice
plus the change there; it is formed in one place for every use the run
makes of it (the state a run starts from, its loads, its migrating
thickness, the calibration's start and the static problem's change). The
state is optional: without an `ocean` block the body has no
ocean, no ice and no ice change, and the forcing is the loads, the tide
and the sources alone (solve, Without an ocean, below).

- The state's ice (`ocean.ice`) is a file input of the ice thickness,
  required.
- The sea level is exactly one of `ocean.sea_level`, a file input of the
  sea level, used as it stands, and `ocean.topography`, a file input of
  the topography in the ICE-NG convention (the ice surface where the ice
  is grounded, the sea floor under floating ice), converted with the
  state's ice, at the same instant by construction: -T over the ocean
  (T < 0) and -T + I over land, raised by the floating fraction
  (917/1000) of the ice where ice sits on the ocean floor (pyslfp's
  construction). The member names the quantity the file holds; no other
  is tried. The ocean function follows from the sea level.
- The ocean function is decided by flotation, so every depression below
  sea level is ocean, joined to the sea or not. `ocean.land`, optional,
  is a file input of a field that is one where the surface is held as
  land whatever the flotation and zero elsewhere (the landlocked
  depressions: the Caspian and the like), cut at one half. A static field
  is the same at every date: under migrating shorelines those points stay
  land throughout. A field on a time axis is a land that changes in time:
  under migrating shorelines in a dated run it is read at every node's
  date, each node's ocean function holding that node's land (the run's
  start, and a calibration's every run, the first node's); under frozen
  shorelines the ocean function is the state's throughout, and the land
  is read at the state's instant alone, the end of the window `ocean.at`
  names; in the static problem and a run counting forwards at
  `ocean.date_kyr_bp`. pyslfp's `export_ice_history` (4.2 or later)
  writes such a field, `land`, as a static field beside the ice thickness
  and the topography, so one file may be named for all three.
- A `thickness` term of the ice change is a file of ice THICKNESS, an ice
  history: its change is its value less its own value at the state's
  instant, so naming for it the file the state's ice comes from gives the
  ice of that file at every date. A `change` term is a file holding a
  thickness change, taken as it stands; a cap (`cap_thickness`,
  `cap_fraction`) is a Gaussian cap of change, in a run through time held
  or scaled by its amplitude history. A `thickness` term, and a `change`
  file with a time axis, are dated ice histories: they make the problem a
  dated one, and a run counting forwards on its own clock refuses them
  (its ice changes by `change` files without a time axis, held, and by
  caps).
- Under frozen shorelines (`ocean.shorelines`, the default) the sea level
  equation is the linear one on the state's shorelines (the start's
  through time), and an ice thickness change loads the Earth only where
  the ice is grounded: the load is 917 kg m^-3 times the thickness change
  times (1 - C), with C that ocean function, so a change of floating ice
  loads nothing. Under migrating shorelines the equation is the nonlinear
  one: the ice is the state's plus the change, and flotation decides
  where it is grounded and where the ocean is.
- A change may not take more ice than the state holds: the state's ice
  plus the change more negative, at any point of the run's grid at the
  date of the static problem or at any node of a run through time, than
  1e-9 times the state's largest ice thickness (1e-9 m where the state
  holds no ice) is refused at `forcing.ice_change`, the error giving the
  worst deficit in metres, its place and, through time, its date. Where
  a resampled state's own thickness falls below zero it is taken to hold
  none.

The static problem: the state is ICE-7G at 26 ka, and the change the
whole way to the present is a `thickness` term read at the present (its
ICE-7G thickness there less its thickness at 26 ka; with `"shorelines":
"migrating"` in the `ocean` block the ice at the solve is ICE-7G's present
ice, and its shorelines follow flotation):

```json
"ocean": {"date_kyr_bp": 26,
          "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"},
          "land": {"file": "ice7g.nc"}},
"forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc", "date_kyr_bp": 0}}]}
```

The same state with a change from a file of one's own, a thickness change
in metres (its variable `ice_thickness_change` by default):
`"forcing": {"ice_change": [{"change": {"file": "my_melt.nc"}}]}`.

The quasi-static problem: the state at the window's start, and the ICE-7G
history over the window as a `thickness` term, read at every node:
`"ocean": {"at": "start", "topography": {"file": "ice7g.nc"}, "ice":
{"file": "ice7g.nc"}}` with `"forcing": {"ice_change": [{"thickness":
{"file": "ice7g.nc"}}]}`.

### 3D viscosity models

`model.viscosity` names a planetmodel model file (`planetmodel.model/1`)
with a field named `viscosity`, as planetmodel's tomography conversion
writes, for the quasi-static problem of `solve` or `kernel`.
The viscosity is split as
eta(r, theta, phi) = eta_bar(r) exp(delta(r, theta, phi)) about a radial
reference eta_bar, chosen by `model.viscosity.reference`:

- `"geometric_mean"` (the default): at each radius, the exponential of the
  lateral mean of ln eta, so delta has zero lateral mean. Works for any
  viscosity; needs a grid that carries quadrature weights.
- `"file_radial"`: the file's own radial reference, for a viscosity stored
  as a perturbation of one; refused when that reference varies laterally.

The file's body must be the deck's: its radial ends are checked, and each
layer of the deck the file gives a viscosity must lie within one layer of
the file. Each viscous layer is a Maxwell body; a wholly radial viscosity
runs the 1D problem (The model block, below).

## The blocks

`solve` and `kernel` read their configs as blocks, JSON objects each
read by one piece of code for every command that takes it:
`discretisation` (the degree the run is truncated at), `model` (the
deck, its format, the lateral moduli and the 3D viscosity),
`times` (the time nodes of a run through time), `forcing` (what drives
the run), `ocean` (the state, the sea level and the ice at one instant,
the shorelines and the initial sea level's calibration), `solver` (the
solves' tolerances, budgets and transforms, and the time step),
`checkpoint` (a run's checkpoints), `output` (solve's files),
`trajectory` (where the forward run is recorded, for the kernels) and
`kernels` (the measurements and the kernels written), with `problem`,
`gravity`, `rotation`, `moments` and `restart` beside them. A
member a command does not take is refused ("is not a recognised
member"), and so is a block's member given at the top level. For the
block members the top-level reader lists (`lmax`, `sea_level`,
`topography`, `ice`, `date_kyr_bp`, `at`, `shorelines`, `calibration`,
`scheme`, `rtol`, `measurements`, `parameters`, `method`,
`record_stride`, ...)
the error names the member each belongs in (a top-level `sea_level` "is
not a top-level member: it belongs in `ocean.sea_level`"), and for
`state`, `ice_history` and `coupled` the members that hold what they
would; any other is not recognised.

One config serves both commands. `solve` reads a `kernels` block with the
same reader as `kernel`, so that a fault in it is found before the run,
and does not otherwise use it; `kernel` reads `output`, `checkpoint` and
`restart` with `solve`'s readers, held to the run as `solve` holds them,
and writes no surface file, exchange table or checkpoint.

Which command takes which:

| Block or member | `solve` | `kernel` |
| --- | --- | --- |
| `problem` | `"static"` or `"quasi_static"`, required | as `solve` |
| `model.deck`, `model.format` | yes | yes |
| `model.viscosity`, `model.burgers` | quasi-static | as `solve` |
| `model.moduli` | yes | yes |
| `discretisation` | required: `lmax` | as `solve` |
| `times` | quasi-static, required: dates before present, or times counted forwards in a `unit` | as `solve` |
| `forcing` | `ice_change`, `load`, `tide` and `seismic`, any mix (`ice_change` with an ocean alone) | as `solve` |
| `ocean` | optional: the state and the shorelines; static, at `date_kyr_bp`; quasi-static, at `at`, with the calibration (counting forwards, at `"at": "start"` and `date_kyr_bp`); absent, a body without an ocean | as `solve` |
| `rotation`, `moments` | yes | yes |
| `gravity` | `"full"` or `"none"`; `"none"` a radial run under seismic sources alone | `"full"` alone: `"none"` is refused |
| `solver.rtol` | yes | yes (the adjoint solves' too) |
| `solver.step_rtol`, `step_max_iterations` | in a 3D run (the static problem's on lateral moduli too) | as `solve` (the reversed steps' too) |
| `solver.transform_degrees`, `cache_mb` | in a 3D run (the static problem's on lateral moduli too); in a 1D run whose config asks for lateral kernels (`kernels.lateral` true), taken and not used | as `solve` (the reversed steps' too); in a 1D run, the lateral kernels' grids |
| `solver.scheme` | quasi-static | as `solve` |
| `solver.step` | in a 3D run (the static problem's on lateral moduli too) | as `solve` |
| `checkpoint`, `restart` | quasi-static | checked as `solve` checks them, not used |
| `output` | a block, required: `surface`, `exchange`, `exchange_nodes`, `volume` | optional, checked as `solve` checks it, not used |
| `trajectory` | optional: `directory` records the run into the trajectory store there; `record_stride` its stride | optional: `record_stride`, the forward run's recording; `directory` sweeps from the store of the run there, or records into it (The trajectory block, below) |
| `kernels` | optional, checked as `kernel` checks it, not used | required: the measurements, the kernels and their directory (kernel, below) |

"In a 3D run" means a run on a model the `model` block makes 3D (solve
says when); a 1D run (the static problem on a radial Earth included)
refuses the members of the 3D step, but for the grids' two where the
config asks for lateral kernels.

### The discretisation block

The degree every command's run is truncated at, a block required in the
configs of `love-numbers`, `solve` and `kernel` alike:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `discretisation.lmax` | integer | required | the largest spherical harmonic degree: the band of every field and expansion of the run, the degree of its Gauss–Legendre grid and the degree its radial mesh is built for (Earth models, above); at least 1 in `solve` and `kernel`, at least 0 in `love-numbers` |

The block holds `lmax` alone, and refuses any other member. A top-level
`lmax` is refused, naming the block ("is not a top-level member: it
belongs in `discretisation.lmax`").

```json
"discretisation": {"lmax": 64}
```

### The model block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model.deck` | path | required | the deck (Earth models, above) |
| `model.format` | word | `"maxwell"` | `"maxwell"` (or `"gia"`), `"prony"`, `"burgers"`, `"mineos"` or `"elastic"` |
| `model.burgers` | list | absent | the quasi-static problem on a `"maxwell"` deck alone: regions of whole layers made Burgers bodies over their own viscosity (below); absent, the deck's Maxwell bodies |
| `model.moduli.file` | path | absent | a planetmodel model file whose `mu` and `kappa` fields give the lateral moduli, either alone (the other modulus then radial; a file holding neither is refused); absent, the radial model alone |
| `model.moduli.reference` | word | `"lateral_mean"` | `"lateral_mean"` (the perturbation about each field's lateral mean) or `"file_radial"` (about the file's own stored radial reference) |
| `model.viscosity.file` | path | absent | the quasi-static problem alone: a planetmodel model file of the 3D viscosity (3D viscosity models, above); absent, the deck's own viscosity column |
| `model.viscosity.reference` | word | `"geometric_mean"` | `"geometric_mean"` or `"file_radial"`: the radial reference (3D viscosity models, above) |

The moduli file holds absolute moduli (planetmodel's `mu` and `kappa`,
Pa), as planetmodel's conversion of a tomography model or
`io::WriteModuli` writes them; the commands take their RELATIVE lateral
perturbations, mu(x) / mu_ref(r) - 1, about the chosen reference, and
apply them to the deck's own radial moduli — so the file should be
built from the deck's radial model, and its lateral mean is not checked
against the deck. A fluid layer takes no shear perturbation (the fluid
form has no grid pipeline). In a viscoelastic run the shear perturbation
scales the unrelaxed shear modulus and every Maxwell branch's modulus
alike, the branch fractions and the relaxation times held
(`Viscoelastic3D::Options::moduli`).

`model.burgers` makes layers of a Maxwell deck Burgers bodies: a Maxwell
element of the layer's own viscosity (the deck's, the body's steady
viscosity) and unrelaxed shear modulus in series with a Kelvin element.
Each entry is a region:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `depth_km` | [top, bottom], km | required | the region's depths below the surface, 0 <= top < bottom; each end on a boundary of the deck's layers (to 1e-9 of the surface radius), every layer between holding a viscosity |
| `kelvin_viscosity` | number > 0, Pa s | required | the Kelvin element's viscosity, absolute (not a fraction of the steady one) |
| `kelvin_modulus_fraction` | number > 0 | required | the Kelvin element's shear modulus as a fraction of the layer's unrelaxed (Voigt) shear modulus, at every point |

```json
"model": {"deck": "prem8.postseismic.deck", "format": "maxwell",
          "burgers": [{"depth_km": [50, 217], "kelvin_viscosity": 1e18,
                       "kelvin_modulus_fraction": 1.0}]}
```

Each layer of a region is the body a burgers deck's Burgers layer of the
same numbers is (Earth models, above): the two give one run to rounding.
A region whose end cuts a layer, or that covers an elastic or a fluid
layer, is refused, naming the layer and its depths, and so are regions
that overlap, a region beyond the deck and one missing a member. A prony
or burgers deck states its rheology, and refuses the member; a
`"mineos"` or `"elastic"` deck has no viscosity to build on, and refuses
it too. Beside a `model.viscosity` file it is refused: the file replaces
the deck's viscosity, and over a lateral one the 3-D step scales every
branch's relaxation time alike (`exp(-delta)`), which would keep the
Kelvin element's viscosity in proportion to the steady one rather than
absolute: a Burgers body over a viscosity file is not built. Under
lateral moduli (`model.moduli`) a Burgers or prony body is perturbed as a
Maxwell body is, its branch fractions and relaxation times held. The
member enters the run's hash and, where given, the outputs' metadata
(`burgers`).

The viscosity matters only to a run through time: `model.viscosity` and
`model.burgers` are refused in the static problem. A `model.viscosity`
file beside a prony or burgers deck is refused: the deck states its
rheology in its own columns, and the 3-D step's lateral viscosity scales
every branch's relaxation time alike, which is not such a deck's body.
In the quasi-static problem the viscosity is the model file's when
`model.viscosity` is given (its radial reference the Maxwell body, the
deck's viscosity column, if any, unused), and the deck's own otherwise:
each layer holding a viscosity is a Maxwell body, or a Burgers body where
`model.burgers` names it, and a prony or burgers deck's layers are the
bodies it states (Earth models, above). A deck format with no viscosity
(`"elastic"`, `"mineos"`) is then refused at `model.format`, and a deck
with no viscosity or branches anywhere at `model.deck`.

A model file whose fields are stored as functions of the radius alone is
radial, and leaves a run 1D; a field stored on an angular grid varies
laterally, even where its values happen not to, and makes a run 3D: a
moduli file in either problem, a viscosity file through time (solve says
what that decides).

### The times block

The time nodes of the quasi-static problem, required there and refused
in the static one. The block takes one of two forms, chosen by the
members present. The DATED form gives dates before present, on the
calendar the files with a time axis share:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `times.start_kyr_bp` | number | required with `nodes` | the window's start, kyr BP |
| `times.end_kyr_bp` | number | 0 | the window's end, kyr BP; must be later than the start, so smaller |
| `times.nodes` | integer >= 2 | required with `start_kyr_bp` | the number of time nodes, both ends included, evenly spaced |
| `times.at_kyr_bp` | array of numbers | instead of the three above | the dates themselves, at least two, from the oldest (strictly decreasing); the first is the start |

The FORWARDS form counts on the run's own clock, in a unit it names; the
run then has no calendar of its own (the state, where there is one, is a
snapshot of its files at a date, The ocean block, below, and the terms'
times are on the clock, solve, The forcing block):

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `times.unit` | word | required | the unit of every time of the run: `"kyr"` (3.15576e10 s), `"yr"` (a Julian year, 3.15576e7 s) or `"day"` (86400 s) |
| `times.start` | number | 0 | the first node, in `unit` |
| `times.end` | number > `start` | required with `nodes` | the last node |
| `times.nodes` | integer >= 2 | required with `end` | the number of nodes, evenly spaced from `start` to `end`, both included |
| `times.at` | array of numbers | instead of `start`, `end` and `nodes` | the nodes themselves, at least two, strictly increasing; the first is the start |

In either form exactly one way of giving the nodes is taken: the window
and `nodes`, or the list (`start_kyr_bp` or `start` beside it is
refused). Evenly spaced nodes take one step length, the window over
(nodes - 1), for every step; listed nodes step by their differences, so
uneven steps are allowed, and listed nodes that coincide with evenly
spaced ones give the same run: to the bit on a run's own clock, whose
times are the nodes less the first in either form, and as dates to the
last bit of a step's length (a listed date's step is the difference of
two dates in seconds, an evenly spaced one a share of the window's span;
the two agree to the bit where that arithmetic is exact, as for the dates
2, 1.5, 1, 0.5 and 0 kyr BP). The run's hashes take the two forms of
dates apart, as two runs. Members of the two forms
together are refused, naming both, and so are `start_kyr`, `end_kyr` and
`at_kyr` (kyr without a unit), naming the forwards form's `start`, `end`
or `at` with its `unit`. A step's response moves with the last bit of its
length (a one-ulp change of a node moves the fields by up to about 1e-12
relative), so the same nodes stated in another unit, or as dates, give
the same run to about that.

`kernel` takes either form, evenly spaced or listed, and locates its
measurements' instants on the nodes: dates before present in the dated
form, times on the run's clock in the forwards one (kernel, below). In
every form the step is not limited by the shortest Maxwell time.

### The ocean block

`solve` and `kernel`, optional in both. Without it the body has no ocean
(solve, Without an ocean, below), and none of the members below applies.
It holds the state, what is on the body at one instant (The state and the
ice change, above): the sea level of the sea level equation, exactly one
of `ocean.sea_level` and `ocean.topography`, the ice, `ocean.ice`, and
the land held, `ocean.land`, each a file input of `file`, `format` and
`variable` alone; the instant, one for the whole block,
`ocean.date_kyr_bp` in the static problem and `ocean.at` in the
quasi-static one (with `ocean.date_kyr_bp` too in a run counting
forwards); the shorelines, in either problem; and, in the quasi-static
problem, the initial sea level's calibration. `calibration` and `at` are
refused in the static problem, which has no initial sea level to
calibrate, and `date_kyr_bp` in a dated run, whose state is at an end of
its window.

A run counting forwards on its own clock has no calendar: its state is
a snapshot, the content of the state's files at `ocean.date_kyr_bp`, put
at the window's start (`"at": "start"`). The date follows the static
problem's rule, required when any of the state's files has a time axis
and refused when none has one. `"at": "end"` is refused there: the state
at the window's end reads the ice at the start back along dated ice
histories (`thickness` terms) and calibrates the initial sea level
against the state's, and a run counting forwards has no such history.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ocean.sea_level` | file input | one of the two | a sea level, used as it stands (variable `sea_level` by default) |
| `ocean.topography` | file input | one of the two | a topography, converted with the state's ice, `ocean.ice` (variable `topography` by default) |
| `ocean.ice` | file input | required | the ice thickness on the body at the state's instant (variable `ice_thickness` by default) |
| `ocean.land` | file input | absent | a field that is one where the surface is held as land whatever the flotation and zero elsewhere (variable `land` by default). It is carried onto the run's grid and cut at one half. A static field holds the same land at every date; one on a time axis is read at every node's date under migrating shorelines in a dated run, at the state's instant (the end of the window `ocean.at` names) under frozen ones, and at `ocean.date_kyr_bp` in the static problem and a run counting forwards (The state and the ice change, above). Absent, the ocean function follows from flotation alone, and every depression below sea level is ocean |
| `ocean.date_kyr_bp` | number | absent | the static problem and a run counting forwards: the state's date, kyr BP, at which every member's file with a time axis is read; required when any of the state's files (the sea level's, the ice's, the land's) has a time axis, refused when none has one; refused in a dated run |
| `ocean.at` | word | required | the quasi-static problem: the end of the window the state belongs to, `"start"` (the initial state, its files read at `times.start_kyr_bp` where they have a time axis, or, counting forwards, at `ocean.date_kyr_bp`) or `"end"` (a dated run's alone: the state the run reaches at `times.end_kyr_bp`, against whose sea level the initial sea level is calibrated, its sea level's and ice's files reaching that date unless they hold their ends, `"outside_span": "hold"`; a land on a time axis is read there under frozen shorelines, and at every node's date under migrating ones); refused in the static problem |
| `ocean.shorelines` | word | `"frozen"` | `"frozen"`, the state's ocean function throughout (the linear equation); `"migrating"`, the ocean function follows flotation (the nonlinear equation; solve, The static problem and The run through time) |
| `ocean.calibration` | object | absent | the quasi-static problem with `"at": "end"`, the calibration's members, all optional (the calibration runs whenever the state is at the end, with these defaults when the block is absent); refused with `"at": "start"` |
| `ocean.calibration.method` | word | `"fixed_point"` | `"fixed_point"`: the classical update S0 <- S_f - xi(t_f; S0); `"gradient"`: steepest descent on the misfit with the adjoint gradient (one adjoint sweep per run); the same single step under frozen shorelines |
| `ocean.calibration.iterations` | integer >= 1 | 10 | the most runs under migrating shorelines; running out is an error |
| `ocean.calibration.rtol` | number > 0 | 1e-4 | convergence under migrating shorelines: the largest update below this times the largest initial sea level (1e-4 is of order a metre) |

A date or an `at` on a member of the state (`ocean.topography.date_kyr_bp`,
`ocean.ice.at`, ...) is refused, naming the block's; so are
`forcing.ice` (the state's ice is `ocean.ice`, a change of it a term of
`forcing.ice_change`), `forcing.sea_level`, `forcing.state` and
`ocean.calibrate_initial_sea_level`, the error naming the members that
hold what they would.

### Rotation and moments

`solve` and `kernel`; top-level members.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rotation` | word | `"full"` (`"none"` without gravity) | `"none"`, `"axial"` or `"full"` (Units and conventions, above); `"none"` alone without gravity |
| `moments` | word | `"input"` | `"input"`: the Earth's standard moments of inertia; `"hydrostatic"`: their difference set from the model's own hydrostatic fluid limit, which full rotational feedback needs when the standard moments sustain less oblateness than the model's tidal response; refused without gravity |

### The solver block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `solver.rtol` | number > 0 | 1e-9 | the relative tolerance of the sea level solves (and, for the kernels and the gradient calibration, of the adjoint solves); a sea level solve that misses it is an error. Without an ocean the solve closes in one pass and it has no effect |
| `solver.step_rtol` | number > 0 | 1e-9 | the relative tolerance of a 3D step's closure: each step's combined fixed point of the strain, the ocean load and the rotation (in `kernel`, the reversed sweep's steps too; under lateral moduli, the zero-length step of the start too, and the static problem's one zero-length step) |
| `solver.step_max_iterations` | integer >= 1 | 60 | its iteration budget; a step that runs out of it is an error |
| `solver.transform_degrees` | integers in [2, lmax] | absent | the degrees the lateral coupling's transforms may be cut to with depth. Each memory node takes the least of these (or lmax) that covers every degree its solution carries, so nothing the degree systems keep is dropped; only the grid that forms the lateral products is coarser at depth, which resolves the lateral structure's pattern there to about that degree. Absent: every node at lmax. Below about lmax 30 every mantle node needs lmax and nothing is cut |
| `solver.cache_mb` | integer >= 0 | 0 | the last-level cache every transform grid of the run blocks its work for, MiB per cache domain (32 on a server with 32 MiB per CCD); 0 takes GSHTrans's default (8 MiB). A server with large per-domain caches may gain at large lmax; a small cache loses by overstating it. The results are independent of it to rounding: it sets only how many fields a batched transform takes at a time, and the batches of another size are transformed under plans of their own |
| `solver.scheme` | word | `"exponential_trapezoid"` | the quasi-static problem alone: the time step, `"exponential_trapezoid"` (second order) or `"backward_euler"` (first) |
| `solver.step` | word | `"fixed_point"` | the quasi-static problem alone, but for the static problem on lateral moduli: how a 3D step's linear system is closed, `"fixed_point"` (the within-step iteration, Anderson-accelerated) or `"conjugate_gradients"` (preconditioned by the radial reference's solve; `Viscoelastic3D::Options::StepSolver`), the conjugate-gradient solves nested inside the step's combined fixed point, each one inexact, to the library's own inner tolerance (`Viscoelastic3D::Options::solve_rtol`) |

The members of the 3D step (`step`, `step_rtol`, `step_max_iterations`,
`transform_degrees`, `cache_mb`) are refused by a 1D run, and by the
static problem on a radial Earth, which takes
`rtol` alone (without a `model.moduli` file they are refused when the
config is read, with a radial one once the file is read); the static
problem on lateral moduli takes them all. A radial config asking for
lateral kernels (`kernels.lateral` true) takes the grids' two,
`transform_degrees` and `cache_mb`, in either problem: `kernel` sweeps
the lateral kernels on the grids they set, and `solve` leaves them
unused, as it does the `kernels` block, its run the run without them.
`scheme` is refused in the static problem.

### The output block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output.surface` | path | required | the surface file to write |
| `output.exchange` | path | absent | the surface coefficients as a cross-code exchange table; in the quasi-static problem one table per node, the path holding `{node}` (solve, Exchange tables and the volume file) |
| `output.exchange_nodes` | array of integers | every node | the quasi-static problem alone: the indices (from 0) of the nodes whose tables are written |
| `output.volume` | path | absent | the static problem alone: the response inside the body as a planetmodel model file (likewise) |
| `output.fields` | array of words | none | optional fields the surface file adds beside those it always holds: `"horizontal_displacement"` writes `displacement_east` and `displacement_north` (solve, Optional and smoothed fields) |
| `output.smoothing` | object | absent | `{"kind": "gaussian", "width_deg": s}`, both required: every field of the response written again smoothed over a Gaussian cap of width s degrees, as `<field>_smoothed` beside the raw one (likewise) |

In the quasi-static problem `output.exchange` must hold the literal
`{node}`, which is replaced by the node's index zero-padded to the width
of the last index (`00` to `40` for 41 nodes); in the static problem it
must not. `exchange_nodes` needs `output.exchange`, and an index past the
last node is refused. `output.volume` is refused in the quasi-static
problem and under seismic sources (the response inside the body is solved
from the static solve's loads and potentials, which do not hold the
sources' forces), and without gravity (it is solved by the systems with
gravity). On lateral moduli over several MPI ranks each rank forms
the response inside the body over its own radial segment, and the first
rank gathers the parts and writes the file.

`output.fields` is a list of words, each the name of a set of optional
fields; a word it does not know is refused, naming the words it takes
(`"horizontal_displacement"`, the one there is), and so is a word listed
twice. `output.smoothing` is an object of `kind` (`"gaussian"`, the one
built; any other is refused naming it) and `width_deg` (a positive
angle), both required. Neither member changes the run, its checkpoints,
its trajectory store or any field written without them; both apply in
either problem, under every forcing, with an ocean or without one and
without gravity.

`kernel` checks an `output` block as `solve` does and writes none of it;
its files go into `kernels.output` (kernel, below).

### The trajectory block

Where the forward run is recorded for the kernels, and how densely:
`solve` records its run into a trajectory store, and `kernel` sweeps from
that store without running the forward problem again, as many times as
there are measurements to take, in other processes, on other days. The
store is the library's (`io::TrajectoryStore`); its on-disk form is
[trajectory-store.md](trajectory-store.md).

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `trajectory.directory` | path | none | the trajectory store: `solve` records into it, `kernel` sweeps from it or records into it |
| `trajectory.record_stride` | integer >= 1 | 1 | through time: keep the forward state at every `record_stride`-th node (and the last); the sweeps rebuild the states in between |

A run without gravity (`"gravity": "none"`), which has no kernels, refuses
the block.

Recording every n-th of N nodes holds about N/n + n forward states during
a sweep, fewest near n = sqrt(N); in 3D each is large
([running.md](running.md), Memory), and with a directory the kept states
live on disk, the sweep reading each back as it reaches it. The static
problem has one node, recorded whole, and refuses `record_stride`; a run
under seismic sources is recorded at every node, each state with its
strain (its states are not rebuilt without the sources' forces), and
refuses a stride other than 1. Without a directory `solve` records nothing and
`kernel` records in memory at the stride, as without the block.

**solve.** With `trajectory.directory`, the run (static or through
time, 1D or 3D, over any number of ranks) is recorded into the store
there, created if absent, as it runs: its start (the calibrated initial
sea level's, where it calibrates), every node's surface fields, loads and
tides (and ocean function under migrating shorelines) and every kept
node's state, each rank's in its own `rank_RRRR`. The store is complete
once the run finishes; a run stopped (at the wall-time limit, or by a
fault) leaves it incomplete, its manifest saying so. The run's outputs
are those of the same run without the block.

- A directory that is absent, empty, or holds an incomplete store of this
  run is recorded into; the incomplete store's own files are removed
  first, and nothing else.
- A directory holding this run's complete store, recorded at the same
  stride over the same ranks, is left as it is: the run runs and writes
  its outputs, and the summary says so.
- Anything else is refused, nothing in the directory touched: another
  run's store, complete or not (another config, or changed input files);
  this run's store at another stride, or over another count of ranks or
  another partition of the mesh; files no store writes; a store of
  another format. The message says to give another directory, or to
  remove that one.
- A run that resumes from the checkpoint of node k (Checkpoints and
  restarts, below) continues the incomplete store its interrupted run
  left: the store must be this run's, recorded at the same stride over
  the same ranks, and hold whole every record the run wrote to node k
  (the start, the surface record of every node to k, every rank's state
  record of every kept node to k), which are kept; the records of later
  nodes, and the partial files of an interrupted write, are removed, and
  the run records on from node k + 1, the kept nodes counted from the
  run's first. The store it completes is the uninterrupted run's, file by
  file (bitwise within one process, which the tests check; across
  processes to rounding, as any two runs). A store short of a record to
  node k, or none at all, is refused, nothing in it removed (a store
  begun at the checkpoint would lack the nodes before it, which are not
  run again), and so is an incomplete store of another stride or layout:
  `"restart": false` starts the run over and records the store whole, and
  without `trajectory.directory` the run resumes without recording. A
  run stopped at the wall-time limit thus leaves its store incomplete to
  the node it stopped at, and the chain of jobs that completes the run
  completes the store; `kernel` records an incomplete store whole.

**kernel.** With `trajectory.directory`:

- a complete store of this run there (the same run, stride and ranks):
  the sweeps run from it, the forward run not run, its calibrated start
  read back rather than calibrated again (the summary says so);
- an absent or empty directory, or an incomplete store of this run: the
  forward run runs as without a directory, recording into the store, the
  sweeps fetching the kept states back from it, and the store is left in
  place for later sweeps (remove it once it has served);
- anything else is refused as by `solve`.

A store is swept over the ranks that recorded it: the 3D kernels about a
run under seismic sources, and the continuous sweep's tide kernels of a
3D run, are formed on one process (kernel, below), so their store is
recorded on one rank too (by `solve` on one, or by `kernel` itself).

The kernels and values from a store are bitwise those of the same
`kernel` run doing its forward run in memory at the same stride (the
tests hold this within one process; FFTW's measured plans are each
process's own, and where two processes' plans differ the last bits may
too).

The store belongs to its run by a hash of the members that decide the
forward run's numbers and of the contents of the files the config names:
the members `solve`'s checkpoints hash (Checkpoints and restarts, below),
and the seismic sources, the static problem's dates of its terms and the
file inputs' formats beside them. The hash covers what determines the
forward trajectory and nothing of the reverse sweep: the `kernels`,
`output`, `checkpoint` and `trajectory` blocks, `restart`,
`solver.cache_mb`, the file inputs' `outside_span` and the paths are left
out, and so is `solver.transform_degrees` on a radial model, where the
forward run has no 3D step and the member sets only the grids of the
sweep's lateral kernels (`kernels.lateral`); so one store serves every
kernel config of one run, at any `transform_degrees` of a radial run,
and a run directory can move. The stride and the ranks are checked
beside the hash, each with its own message.

**Solve once, kernel many.** One config, the run and its `trajectory`
block, with a `kernels` block per batch of measurements (or one config
edited between batches):

```sh
sl3d solve run.json        # the forward run, recorded into the store
sl3d kernel run.json       # sweeps from the store: no forward run
sl3d kernel run_more.json  # other measurements, the same store
```

`run_more.json` differs from `run.json` in its `kernels` block alone (its
measurements, kernels, method or output directory). Its sweeps take as
long as `run.json`'s; the forward run, and in 3D its calibration, are not
paid again.

**Size on disk.** With L = `lmax`, N nodes and stride n (k = N/n + 1
states kept, the last always), the store holds about

- 1D: k x 24 K E (L + 1)(L + 2) bytes of states, K the relaxing branches
  (one in a Maxwell deck) and E the radial mesh's points (its elements
  times their GLL points, which the mesh refines with L: about 160 for
  the prem8 deck at L = 8, 560 at L = 32), plus about N x 80 (L + 1)^2
  bytes of surface records. Under seismic sources every node is kept, each
  with its strain, the toroidal systems' state and strain and the sources'
  forces: about five and a half times the internal variables alone;
- 3D: k x 40 K G bytes of states over all ranks (each rank its share), G
  the grid values of the memory nodes, about their count times 2 (L +
  1)^2 (fewer with `solver.transform_degrees`), plus the same surface
  records; under seismic sources each state with its strain and the
  sources' forces, several times more.

At L = 8 on the prem8 deck a 1D state is 346 kB and a 3D one (lateral
viscosity) 2 MB; at L = 64 a 3D state is of order 100 MB, and a run of
81 nodes at stride 9 about 1 GB. Under seismic sources at L = 32
(examples/configs/kernel_postseismic.json, 1D, 8 nodes) each node's
record is 86 MB, 655 MiB in all. The summaries give each store's size.

## love-numbers

The per-degree Love numbers of the deck's elastic structure, degrees 0 to
`lmax`, in pyslfp's conventions (the generalised Love numbers of Al-Attar
et al. 2024).

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | path | required | the deck |
| `model_format` | word | `"maxwell"` | `"maxwell"` (or `"gia"`), `"prony"`, `"burgers"`, `"mineos"` or `"elastic"`: the deck's elastic values (a viscoelastic deck's unrelaxed ones) |
| `discretisation` | object | required | `lmax`, the largest degree, an integer >= 0 (The discretisation block, above) |
| `output` | path | required | the table to write |
| `conventional_output` | path | absent | where to write the conventional dimensionless numbers too (below); not `output`'s file |

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
response to a unit surface load (kg m^-2), whose sums are the load numbers
(h = h_u + h_phi, and likewise l and k); `v` is the response to a unit
tangential traction, and `t` the tidal response to a unit external
potential. h and l are surface displacements in metres and k the potential
perturbation in m^2 s^-2, per unit of the forcing, with the potential
negative near added mass. Degree 1 is in the centre-of-mass frame.

These are generalised numbers in SI, not the conventional dimensionless
ones. Farrell's (1972) load numbers, referred to the load's own potential
with the geodetic sign, and the tidal numbers referred to the external
potential, are

```
h' = (2l + 1) g / (4 pi G a) h      l' = (2l + 1) g / (4 pi G a) l
k' = -(2l + 1) / (4 pi G a) k - 1
h_tide = -g h_t    l_tide = -g l_t    k_tide = k_t
```

with a and g the header's `radius_m` and `surface_gravity_m_s2` and G =
6.6743e-11 m^3 kg^-1 s^-2 (so h'_2 is about -1 for the Earth and k'_2 about
-0.3). `conventional_output` writes them, one row per degree under the same
provenance header: `l h_load l_load k_load h_tide l_tide k_tide`, the tidal
numbers `nan` below degree 2. The library gives them as
`LoveNumbers::Conventional`.

Summary: `wrote <output> (degrees 0..<lmax>)`, or `wrote <output> and
<conventional_output> (degrees 0..<lmax>)`.

```json
{
  "model": "earth.deck",
  "discretisation": {"lmax": 256},
  "output": "love_numbers.txt"
}
```

```json
{
  "model": "prem8.elastic.deck",
  "model_format": "elastic",
  "discretisation": {"lmax": 1024},
  "output": "out/prem8_love_numbers.txt"
}
```

## solve

The sea level equation: the ocean coupled to the solid Earth, with
rotational feedback, on a state (the sea level and the ice at one
instant) read from file inputs, under a forcing of any mix of an ice
change, surface loads, a tide and seismic sources, with frozen or
migrating shorelines. The state is optional: without one the body has
no ocean, and the run is the response of the solid body, with its
rotational feedback as asked, to the loads, the tide and the sources
(Without an ocean, below); without gravity, a radial body's response
to surface loads, by their weight, and seismic sources (Without gravity,
below). The config is made of
blocks (The blocks, above): `model`, `times`, `forcing`, `ocean`,
`solver`, `checkpoint`, `trajectory`, `output` and `discretisation`,
with `problem`, `gravity`, `rotation`, `moments` and `restart` beside
them.

The problem decides what is solved:

- `"problem": "static"` is the solve at the state's date on the elastic
  Earth under the forcing: one solve of the sea level equation, the
  linear one on the state's shorelines or, under migrating shorelines,
  the nonlinear one, with the deck's elastic structure at `lmax`, or,
  under lateral moduli, on the laterally heterogeneous elastic Earth (The
  static problem, below). Under an ice change alone it is the elastic
  sea-level fingerprint of the change; under a fault alone, the
  co-seismic sea level change; without an ocean, the body's elastic
  response;
- `"problem": "quasi_static"` is the run through time under the forcing
  from the state at an end of the window: the viscoelastic Earth through
  the sea level equation over the nodes of its `times` block, dates
  before present (GIA over a deglaciation) or times on the run's own
  clock in a `unit` (a post-seismic relaxation over years, say) (The run
  through time, below), restartable from checkpoints; without an ocean,
  the body's viscoelastic response through time.

Both are runs of the library's evolutions (`Evolution`, `Evolution3D`):
the static problem is a run of one node, the solve of its start alone,
on an Earth with no relaxation (the deck's elastic structure, with no
Maxwell branch).

Whether the run is 1D or 3D is decided by the `model` block alone. A run
through time is 3D when the viscosity of `model.viscosity` or the moduli
of `model.moduli` vary laterally: `Evolution3D`, each step solving the
combined fixed point of the strain, the ocean load and the rotation
about the radial reference, over MPI ranks, with the radial reference
model's run written beside it; otherwise it is the radial model's
(`Evolution`), on the deck's viscosity column or the radial viscosity of
`model.viscosity`. The static problem is 3D when the moduli of
`model.moduli` vary laterally (The static problem on lateral moduli,
below), and radial otherwise, a radial moduli file leaving the deck's
moduli as they are; it has no use for a viscosity, and `model.viscosity`
is refused in the static problem.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `problem` | word | required | `"static"` (the solve at the state's date) or `"quasi_static"` (the run through time) (The problem, above) |
| `model` | object | required | the Earth model (The model block, above) |
| `discretisation` | object | required | `lmax`, the degree of the run and of its grid, an integer >= 1 (The discretisation block, above) |
| `times` | object | quasi-static: required | the nodes, dates before present or times on the run's own clock in its `unit` (The times block, above); refused in the static problem |
| `forcing` | object | required | the ice change, surface loads, the tide and seismic sources, any mix of them (below) |
| `ocean` | object | absent | the state, its instant, the shorelines and, in the quasi-static problem, the initial sea level's calibration (The ocean block, above); absent, the body has no ocean (Without an ocean, below) |
| `rotation` | word | `"full"` | the rotational feedback (Rotation and moments, above); without gravity `"none"`, its default then |
| `moments` | word | `"input"` | the moments of inertia (Rotation and moments, above) |
| `gravity` | word | `"full"` | `"full"`, the problem with self-gravitation, or `"none"`, the problem without gravity: a radial body under surface loads, acting by their weight, and seismic sources, with no ocean, tide or rotational feedback, the loads' degree-1 part removed (Without gravity, below) |
| `solver` | object | absent | the solves (The solver block, above): in the static problem, `rtol` alone on a radial Earth, and the 3D step's members too on lateral moduli |
| `checkpoint` | object | absent | the quasi-static problem alone: the checkpoints (Checkpoints and restarts, below) |
| `restart` | boolean | `true` | the quasi-static problem alone: whether to resume from a checkpoint (Checkpoints and restarts, below) |
| `trajectory` | object | absent | the trajectory store the run is recorded into (The trajectory block, above) |
| `output` | object | required | the files to write (The output block, above; Exchange tables and the volume file, below) |

### The forcing block

The forcing is what happens to the state (The state and the ice change,
above): an ice change, a direct load where the ice is grounded, surface
loads, a tide and seismic sources, in either problem and in any mix; the
state itself, the sea level and the ice at one instant, is the `ocean`
block's. At least one term of the four lists is needed: a `forcing`
holding none (the block empty, or each list empty) is refused. Without an
ice change the ice is the state's throughout; with an ocean and a fault
alone the sea level equation is still solved, the ocean's water
conserved. Without an ocean there is no ice to change, and
`forcing.ice_change` is refused (Without an ocean, below).

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `forcing.ice_change` | list | absent | the ice thickness change, the sum of its terms, each an object holding exactly one of the four members below |
| `forcing.ice_change[k].thickness` | file input | one of the four | a file of ice THICKNESS (variable `ice_thickness` by default) with a time axis: its change is its value less its own value at the state's instant (a file without a time axis, whose change would be zero, is refused). A dated ice history: refused in a run counting forwards |
| `forcing.ice_change[k].thickness.date_kyr_bp` | number | required in the static problem | the date the file is read at, kyr BP: the change is the file at this date less the file at the state's date (which the state must have, `ocean.date_kyr_bp`); refused in the quasi-static problem, which reads the file at each node's date |
| `forcing.ice_change[k].change` | file input | one of the four | a file holding a thickness CHANGE in metres (variable `ice_thickness_change` by default), taken as it stands. The static problem reads a file without a time axis as it is; a dated run reads a file with a time axis at each node's date and holds one without through the run; a run counting forwards holds a file without a time axis through the run and refuses one with (a dated history). Refused with the state at the window's end (`"at": "end"`), where the ice changes are thickness histories, `thickness` terms |
| `forcing.ice_change[k].change.date_kyr_bp` | number | absent | the static problem: the date the file is read at, kyr BP; required of a file with a time axis, refused for a file without one |
| `forcing.ice_change[k].cap_thickness` | object | one of the four | a Gaussian cap of `thickness_m` metres of ice at its centre (`latitude`, `longitude`, `width_deg`, `thickness_m`, negative to melt: a melting cap takes at each point no more than the ice the state holds). Refused with the state at the window's end |
| `forcing.ice_change[k].cap_fraction` | object | one of the four | a Gaussian cap of a fraction of the state's ice (`latitude`, `longitude`, `width_deg`, `fraction`, negative to melt). Refused with the state at the window's end |
| `forcing.ice_change[k].history_kyr_bp`, `forcing.ice_change[k].history` | list of pairs | absent | the quasi-static problem, beside a cap: the cap's amplitude history, as a surface load's (below), dated in a dated run and on the run's clock in one counting forwards; without one the cap is held through the run, and so is present at its start. Refused beside a `thickness` or a `change` term, and in the static problem |
| `forcing.load` | list | absent | surface loads: each a load term (one of `cap`, `disk`, `field`, `coefficients`, below) with the properties below; without gravity the loads summed act less their degree-1 part (Without gravity, below) |
| `forcing.load[k].water_budget` | true or false | required with an ocean; `false` without one | whether the load's mass is exchanged with the ocean: `true` for water held on land (hydrology), whose mass the ocean loses; `false` for any other mass (sediment), which moves no water. Without an ocean there is no water to exchange, and it has no effect |
| `forcing.load[k].moves_solid_surface` | true or false | required with an ocean; `false` without one | whether the load is a layer laid on the solid surface: the sea level, measured from the solid surface, then falls by the layer's thickness, the water it displaces going elsewhere, and under migrating shorelines the shorelines move with it. Without an ocean the relative sea level change falls by the layer's thickness, and nothing more; without gravity, which forms no sea level change, the layer acts by its weight alone, as any load |
| `forcing.load[k].density_kg_m3` | number > 0 | required of a load that moves the solid surface, refused otherwise | the layer's density: its thickness is its mass over this |
| `forcing.load[k].history_kyr_bp` | list of pairs | absent | a dated run: the term's amplitude history, pairs `[date_kyr_bp, factor]` from the oldest date (the dates strictly decreasing) |
| `forcing.load[k].history` | list of pairs | absent | a run counting forwards: the term's amplitude history, pairs `[time, factor]` on the run's clock in its unit, the times strictly increasing |
| `forcing.tide` | list | absent | the tide: coefficients of an external potential (below), summed |
| `forcing.seismic` | list | absent | seismic sources (below) |

The ice at a time is the state's ice plus the terms' changes summed
there. A change that takes more ice than the state holds is refused (The
state and the ice change, above). A cap's history scales its change at
each time: a `cap_fraction`'s is the fraction of the state's ice, a
melting `cap_thickness`'s the cap held to the state's ice, each times the
factor, so a factor above one may take more ice than there is, which is
refused. In the quasi-static problem a change present at the window's
start (a `change` file whose change is not zero there, a cap without a
history) acts at the first node: under frozen shorelines it is a load
there, the first node the response of the unrelaxed body to it; under
migrating shorelines the run solves its start, the nonlinear equation on
the unrelaxed body with the ice the state's plus the change there, so
that the first node's shorelines may differ from the state's.

**Load terms.** Each element of `forcing.load` is an object with exactly
one member naming the term's kind, whose value holds the term's members,
beside the load's properties above and, in the quasi-static problem, its
history. Each is a surface mass density in kg m^-2:

| Kind | Members | Surface mass density |
| --- | --- | --- |
| `cap` | `latitude`, `longitude`, `width_deg` (> 0), `amplitude_kg_m2` | A exp(-d^2 / (2 w^2)), with d the great-circle distance from the centre, w = `width_deg` and A = `amplitude_kg_m2`; no cut-off |
| `disk` | `latitude`, `longitude`, `radius_deg` (in (0, 180]), `amplitude_kg_m2` | A where d <= alpha (alpha = `radius_deg`), zero outside |
| `field` | a file input (File inputs, above): `file`, `format`, `variable` (default `"load"`); static: `date_kyr_bp`, or `date` with `unit`; counting forwards: `time_origin` (default 0) | the variable of the file |
| `coefficients` | `file` | the coefficients in a text table |

All members are required but a field's `format`, `variable`, dates and
time origin; latitudes lie in [-90, 90]. A field's `name` is refused,
naming `variable`.

- `cap` is sampled at the run's grid points and expanded by the grid's
  quadrature.
- `disk` is not sampled across its edge but expanded exactly to `lmax`.
  By the Funk–Hecke theorem its coefficients are

  ```
  f_lm = 2 pi A I_l conj(Y_lm(theta_0, phi_0))
  I_0 = 1 - cos(alpha)
  I_l = sin(alpha) P_l^1(cos(alpha)) / (l (l + 1))     (l >= 1)
  ```

  with I_l the integral of the Legendre polynomial P_l over
  [cos(alpha), 1], P_l^1 the associated Legendre function without the
  Condon–Shortley phase (by its upward recurrence), (theta_0, phi_0) the
  centre and Y_lm the harmonics of the coefficient tables below. The load
  is this truncated exact expansion, so it rings at the disk's edge as a
  band-limited step does; its mean is A (1 - cos(alpha)) / 2.
- `field` is read as every file input is (File inputs, above):
  resampled onto the run's Gauss–Legendre grid by the field's own lateral
  rule and converted to SI by its declared dimensions (a field declaring
  none is used as it stands, so it should hold kg m^-2), then expanded by
  the grid's quadrature. When its file is read is set out below (Times
  of the terms).
- `coefficients` names a text table. Lines starting with `#` and blank
  lines are skipped; every other line is a row `l m re im`: integers
  0 <= m <= l and the real and imaginary parts of the coefficient f_lm of
  the surface mass density, in kg m^-2. The harmonics are the exchange
  table's (Exchange tables and the volume file, below): the library's
  complex orthonormal harmonics with the Condon–Shortley phase,
  Y_lm(theta, phi) = sqrt((2l + 1) / (4 pi)) d^l_0m(theta) e^(i m phi),
  each squared modulus integrating to 1 over the unit sphere; the field is
  real, so the coefficients of m < 0 follow as
  f_(l,-m) = (-1)^m conj(f_lm) and the imaginary part at m = 0 must be
  zero. From the real orthonormal coefficients a_lm (cosine) and b_lm
  (sine) of pyshtools' `"ortho"` normalisation without the Condon–Shortley
  phase, f_l0 = a_l0 and f_lm = (-1)^m (a_lm - i b_lm) / sqrt(2) for
  m > 0. Degrees absent from the table are zero; a degree above `lmax`, a
  row given twice or a malformed row is refused, naming the line. For
  example, degrees 0 and 2 of a load (the m = 0 rows real):

  ```
  # a load's coefficients, kg m^-2
  0 0 3.5e3 0
  2 0 -1.2e3 0
  2 1 4.0e2 -2.5e2
  ```

The loads applied are the terms' coefficients, summed by their
properties and band-limited to `lmax`. The `load` field written is the
sum of each term's field at the grid points whose expansion is that
term's coefficients: a cap's and a field's values as sampled, a disk's
and a table's synthesised from their coefficients.

**Tide coefficients.** Each element of `forcing.tide` is one coefficient
of the external potential, in the harmonics of the coefficient tables;
coefficients of the same degree and order add. A tide is taken relative
to the centre of mass, so its degrees are 2 and up: a degree-0 potential
exerts no force, and a degree-1 one is a uniform field that moves the
whole body and deforms none of it.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `degree` | integer in [2, lmax] | required | the degree |
| `order` | integer in [0, degree] | 0 | the order (m >= 0: the potential is real) |
| `amplitude_m2_s2` | number | required | the coefficient's real part, m^2 s^-2 |
| `imaginary_m2_s2` | number | 0 | its imaginary part, zero at order 0 |
| `history_kyr_bp`, `history` | list of pairs | absent | the quasi-static problem alone: its amplitude history, as a load term's |

The tide is an external potential: the body answers it through its
tidal Love numbers, as it answers the centrifugal potential of the
rotation, and the potential moves the equipotentials, so it is in the
relative sea level change and not in the `potential` written, which is
the body's own perturbation. The ocean's load and the rotational
feedback answer the body's deformation under it as they answer any
other. Under migrating shorelines the tide enters the nonlinear sea
level equation as it enters the linear one: its potential moves the sea
surface, and so the flotation that places the shorelines, a tide large
against the depth of a shelf flooding or draining it; the fields mean
what they mean under frozen shorelines.

**Seismic sources.** Each element of `forcing.seismic` is a source inside
the body: an object holding exactly one of `point`, `fault` and
`fault_file`, and, in the quasi-static problem, optionally an onset and a
history. The source is
the stress glut itself, a moment tensor: in the static problem it acts
whole, the co-seismic response; through time it acts from its onset
(`onset_kyr_bp`, a date, in a dated run, `onset` on the run's clock in
one counting forwards; from the window's start when absent) and is held,
times its amplitude history where one is given, so the node at which it
first acts holds the co-seismic response and the nodes after it the
post-seismic relaxation. The onset should be a node of the run. A source
excites toroidal motion as well as spheroidal, and the surface file then
holds `toroidal_potential`.

A `point` is a moment tensor at a point:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `latitude`, `longitude` | number | required | the point, degrees |
| `depth_km` | number > 0 | required | its depth below the body's surface |
| `moment_tensor_nm` | object | one of the two | the tensor's components `rr`, `tt`, `pp`, `rt`, `rp`, `tp` in the spherical polar basis at the point (r up, theta south, phi east), N m |
| `strike_deg`, `dip_deg`, `rake_deg`, `moment_nm` | number | one of the two | a double couple instead: the fault plane and slip as Aki & Richards give them (strike clockwise from north with the fault dipping to its right, dip from the horizontal in [0, 90], rake of the hanging wall's slip in the plane from the strike direction) and the scalar moment, N m |

A `fault` is a flat rectangle of uniform slip, laid out as point sources
at the centres of equal patches, each with an equal share of the moment:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `latitude`, `longitude`, `depth_km` | number | required | the centroid |
| `strike_deg`, `dip_deg`, `rake_deg` | number | required | the plane and the slip at the centroid, as for a point |
| `length_km`, `width_km` | number > 0 | required | along strike and down dip |
| `moment_nm` | number | required | the total scalar moment, N m |
| `patches_along_strike`, `patches_down_dip` | integer >= 1 | 1 | the patches each way |

A patch whose centre lies above the surface is left out, its share of
the moment with it. One point source is band-limited by `lmax`; a fault
of several patches is the way to a smoother source.

A `fault_file` is a fault given by its slip: a planetmodel fault file
(`planetmodel.fault/1`, written by planetmodel's `write_fault`, or its
C++ `WriteFaultFile`; its format
page is planetmodel's `docs/source/formats/fault_file.rst`), one or more
patches of surface (planes and curved structured patches) with the slip of
the hanging wall on each, in fault components on a grid of its own,
static or on a time axis. It is laid out as point sources at the points
of a quadrature over every patch:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `file` | path | required | the fault file |
| `cells_along_strike`, `cells_down_dip` | integer >= 1 | required | the quadrature's cells on every patch: equal cells of the patch's parameter square, u along strike and v down dip |
| `points` | integer >= 1 | 1 | Gauss–Legendre points each way in a cell; 1 is the midpoint rule, a point at each cell's centre |
| `place_by` | `"depth"` or `"radius"` | required where the file's reference radius is not the model's surface radius | how the fault is placed in the model (below) |
| `time_origin` | number | 0 | a file on a time axis in a run counting forwards: the time of the run's clock, in its unit, at which the file's time zero falls. Refused in a dated run and on a static file |
| `outside_span` | `"refuse"` or `"hold"` | `"refuse"` | a file on a time axis: what a node's time outside the file's span does, as for a file input (File inputs, above). Refused on a static file |

The quadrature is planetmodel's (`Fault::SliceQuadrature`): each patch cut into
the cells given, with `points` by `points` Gauss–Legendre points in each, a
point's weight w the area it stands for; the points are taken patch by
patch, u slowest. At each point the file's slip s (the displacement of the
hanging wall relative to the footwall) and the patch's unit normal n (from
the footwall into the hanging wall) give the moment tensor
`M = w C : (s n)`, with C the model's elastic tensor there. The file holds
no rigidity, so the same file on another model gives another moment. C is
transversely isotropic about the radial direction (the deck's Love moduli
A, C, F, L, N; isotropic, A = C = lambda + 2 mu, F = lambda, L = N = mu),
and with e = (s n + n s) / 2 in the spherical polar basis at the point
(r up, theta south, phi east):

```
M_tt = w (A e_tt + (A - 2N) e_pp + F e_rr)
M_pp = w ((A - 2N) e_tt + A e_pp + F e_rr)
M_rr = w (F (e_tt + e_pp) + C e_rr)
M_rt = 2 w L e_rt,   M_rp = 2 w L e_rp,   M_tp = 2 w N e_tp
```

(isotropic: w (lambda (s.n) I + mu (s n + n s)); a slip in the fault's
surface, s.n = 0, a double couple of moment mu w |s|). The moduli are the
deck's at the point's radius, the layer above at a boundary, read from its
`kappa` and `mu`, its velocities (`vpv`, `vsv`, `vph`, `vsh`, `eta`), or
`vp` and `vs` as the radial solver reads them; where a moduli file varies
laterally (`model.moduli`), its relative perturbations s and b at the point
are added as the 3-D operator applies them, delta mu = s mu and
delta kappa = b kappa against the deck's Voigt-averaged mu and kappa there
(an isotropic change). A plane of the file is the rectangle a `fault`
lays out by the same strike, dip and rake, and its midpoint quadrature on
the `fault`'s patches puts its points at their centres: the file of a
plane whose uniform slip is a `fault`'s moment over mu L W is that `fault`,
field for field to rounding.

**Choosing the cells.** The quadrature integrates the moment density, the
model's moduli times the slip, over the fault, so the cells are chosen for
that integrand and not for `lmax`: fine enough to follow the slip (its
taper towards the edges above all) and the fault's change of depth. With
smooth moduli the midpoint rule converges at second order, and two or
three `points` a cell serve coarser cells. Where the fault crosses an
interface of the model the moduli jump, a point takes the moduli of its
own side, and the moment then converges only at first order in the cells
down dip, moving as points pass from one side of an interface to the
other. The megathrust of the examples (2 to 50 km deep through PREM's
crustal interfaces at 12 and 21.4 km and the 50 km base of its lid) shows
the size: its moment is 2.39e22 N m on 36 by 6 cells and on 72 by 12,
2.30e22 on 144 by 24, 2.22e22 on 36 by 6 with two points a cell and
2.08e22 on 18 by 3, with its potency (slip times area) the same to
0.3 % on all of them. A moment wanted to better than a few per cent on a
layered model takes many cells down dip, or a model with smooth moduli
through the fault's depths; doubling the cells and comparing is the test.
The cost grows with the number of distinct radii among the points.

**Placement.** The file's positions are latitude, longitude and radius
about its own reference sphere (`reference_radius`); a model's surface may
lie elsewhere (PREM without its ocean ends at 6368 km). Where the two radii
agree to 1e-9 the positions are taken as they are; otherwise `place_by`
is required: `"depth"` keeps each point's depth below the file's reference
sphere, below the model's surface (the point moved along its radius, its
normal, slip and weight the file's), and `"radius"` takes each point's
radius as stored. A point above the model's surface is refused, naming its
patch, place and depth (a `fault`'s patch above the surface is left out; a
file's never is), and so is a point where the model has no shear modulus.
A file in a Cartesian frame (a flat body) is refused: this body is a
sphere.

**Time.** A static file is a rupture: its slip acts as a `point`'s or a
`fault`'s moment does, from its onset and times its history where given.
A file whose slip is on a time axis (`node`, the slip at each stored time
the cumulative slip) carries its own time dependence: an onset or a
history beside it is refused, and so is such a file in the static problem,
whose source is one slip. In a run counting forwards it is read at
t - `time_origin` (the run's unit) at the clock's time t, as a `field`
load's file is; in a dated run it is read by date, its axis the
calendar's (the present at zero, the past negative), as a `field` load's
file is, and `time_origin` is refused. A node's time outside the file's
span is refused before the run, or, with `"outside_span": "hold"`, the
file's first or last slip is held there. A time within the tolerance the
program reads every file's time axis with (1e-9 of the larger of the
span's length and its ends' magnitudes: File inputs, above) of one of the
file's stored times is read as that time, under every rule, an end of the
span included. planetmodel's reading of the file (`Fault::Rows`) takes
that tolerance as its `time_rtol`, on the same scale, in place of its own
default (`TIME_RTOL`, 1e-12): a run's times are formed by arithmetic (its
clock converted to the file's unit, a date formed from the window's
start), so that a run counting forwards in kyr, its zero 120 kyr from the
file's, reaches a time stored at -999 yr as -998.99999999999511 yr, and
reads it as that stored time.
At each time the slip is read by the file's own
rule (`linear` between its times, `previous` from the latest not after
the time, `none` at its stored times alone, any other time then
refused): each slice of the slip is laid out as its own set of tensors,
on planetmodel's slice quadrature (`Fault::SliceQuadrature`: the points,
normals and weights formed once, each slice's slip at them), and the
moment at a point is the slices' tensors under the weights the reading
gives them (`linear`: one at the slice's own time, zero at the others and
linear between; `previous`: one from its time to the next), combined as
planetmodel combines the slices' slip (`FaultSliceQuadrature::SlipOf`):
the moment of the slip the file reads at that time, its size and
direction both.

**Cost.** Each quadrature point is a point source. Sources at one radius,
to 1e-12 (rounding, never a move of a source), are formed together, at
the cost of one: a plane's points pair across its middle along strike,
and so do those of a patch swept from a small-circle trench (a surface of
revolution about the circle's pole) at a quadrature symmetric about the
arc's middle; points whose radii differ by more, as a curved patch's along
strike otherwise do, are formed one by one. In a run through time the
sources are formed at every node.

**The build.** The term needs planetmodel's C++ library with the fault
file and its slice quadrature (`<planetmodel/faultfile.hpp>`, planetmodel
1.6.0 or later); a build of sl3d against an older planetmodel reads the
config but refuses the term when it runs, naming the planetmodel it was
built with (docs/building.md).

**What the seismic forcing resolves.** sl3d is a global code: a source's
field is carried to degree `lmax`, a wavelength of about 40 000 km /
`lmax` at the surface (400 km at lmax 96), while a source at depth d
puts structure into degrees up to about a / d (a the radius: some 600
for a source 10 km deep). Within a few source depths of a shallow fault
the point values of the displacement and the sea level are therefore not
converged at any degree a global run reaches, and they ring at the
truncation wavelength; the patches of a fault and the quadrature points of
a fault file, being point sources, are resolved one by one where `lmax`
passes a / (their spacing). Averages
over caps wider than the truncation wavelength are converged: the
measurements' caps, and the smoothed fields of `output.smoothing`
(Optional and smoothed fields, below), the 2° means the examples'
figures draw (`examples/configs/README.md`, "Convergence in lmax of the
seismic examples", has the numbers). The forcing serves the global and regional
response to an earthquake: the sea level change, the long-wavelength
displacement and potential, the post-seismic relaxation and their
kernels. Co-seismic deformation at kilometre scale about a finite fault
is the work of a regional or half-space code, not of this one.

**Times of the terms.** The terms' times take the form of the run's. In
the static problem and a DATED run they are dates, kyr before present,
as the window's are: a history is `history_kyr_bp`, pairs `[date_kyr_bp,
factor]` from the oldest (the dates strictly decreasing), and an onset
`onset_kyr_bp`. In a run COUNTING FORWARDS they are times on the run's
own clock in its unit, `times.unit`: a history is `history`, pairs
`[time, factor]` (the times strictly increasing), and an onset `onset`.
A history's factor at a time is the piecewise-LINEAR interpolant of its
factors between the pairs' times, held CONSTANT at the first factor
before the first time and at the last after the last; a single pair is a
constant factor, and a term without a history has the factor 1
throughout. A `field` load's file is read

- in the static problem: a file with a time axis at the term's
  `date_kyr_bp`, kyr before present (the file read at minus that on its
  axis), or at `date` with its `unit` (`"kyr"`, `"yr"` or `"day"`), a
  time on the file's OWN axis, read at +`date` x the unit, so `"date":
  -1500, "unit": "yr"` is `"date_kyr_bp": 1.5`; one of the two is
  required of a file with a time axis, and both together, or either on a
  file without one, are refused;
- in a dated run: at each node's date, as the ice change's files are, so
  a load file and an ice file on one calendar agree with nothing more
  said;
- in a run counting forwards: at t - `time_origin` (in the run's unit,
  0 by default) at the clock's time t, so with the default the file's
  axis is the clock's; a file whose axis counts from the present,
  negative in the past (an ice history, a dated run's output), is read
  from its date -D at the run's start by `"time_origin": D`.

A file with a time axis is read by its own rule (a `node` field with
linear time interpolation between its stored times) and expanded at
each time, every node's time held to its span before the run steps (a
time outside it refused, or, with `"outside_span": "hold"` on the
`field` input, its first or last field held there: File inputs, above);
a static file is the same pattern at every node, and a history on a
field term multiplies the field read. The forcing at a node is each
term's pattern at the node's time times its factor there: the forcing
enters a run through time at its nodes alone, and a history's or a field
file's variation between two nodes is not seen. A fault file whose slip is
on a time axis is read by the same rules, on the run's clock from its
`time_origin` or by date, its span held before the run (Seismic sources,
above).

The names of the other form are refused, naming this form's: the dated
names in a run counting forwards and the forwards names in a dated run
or the static problem; so are times in kyr without a unit
(`history_kyr`, `onset_kyr`, `time_origin_kyr`, a field's `date_kyr`),
naming the member to give, and the members of the other problem: a
term's date in a run through time (a file term's of the ice change, a
`field` load's), and a history or an onset in the static problem (on a
cap, a load, a tide coefficient or a source). A thickness term without a
date is refused in the static problem, as is a term naming no kind or
several in either. `forcing.ice`, `forcing.sea_level` and
`forcing.state` are refused, naming the members that hold the state's
ice and sea level and the ice change.

The loads, the tide and the sources are changes from the state: in the
static problem they act at its date, at their full amounts; in the
quasi-static problem a term present at the first node acts there, the
first node then the response of the unrelaxed body to it, and under
migrating shorelines its shorelines may already differ from the start's;
a term ramped up from zero has no response there. Each load is
band-limited to the run's degree. The ice is not such a load: it is in
the water budget, it does not move the solid surface (the solid surface
is the one under the ice) and it alone enters the flotation that places
the shorelines.

A run through time forced beside its ice is not calibrated (the state at
the window's end, `"at": "end"`, is refused), and a run under seismic
sources is not checkpointed. The output file's metadata lists the loads
(`load`), the tide (`tide`) and the sources (`seismic`), each term
described and joined by `; ` (a tide coefficient as `tide degree <l>
order <m>, <re> + <im> i m^2 s^-2`, with its history where it has one);
the files hold the loads applied too, the static problem's and a run
through time's at each node (The static problem, The run through time,
below).

For example, in a run through time from 21 kyr BP, a fifth of the ice
under a cap melted between 21 and 15 kyr BP and held after, from a
factor of zero at the window's start:

```json
"forcing": {"ice_change": [{"cap_fraction": {"latitude": 60, "longitude": 265,
                                             "width_deg": 15, "fraction": -0.2},
                            "history_kyr_bp": [[21, 0], [15, 1]]}]}
```

The same melt over the first fifty years of a run counting forwards in
years, `"times": {"unit": "yr", "at": [0, 10, 25, 50, 100]}`, is
`"history": [[0, 0], [50, 1]]` beside the cap.

### The static problem

The solve at the state's date on the elastic Earth under the forcing.
The state is the ice of `ocean.ice` and the sea level of
`ocean.sea_level` or `ocean.topography` (a topography converted with the
state's ice), each file with a time axis read at the state's date,
`ocean.date_kyr_bp`, and the land held, `ocean.land`; its ocean function
is flotation's. The ice change is the sum of the terms of
`forcing.ice_change`, each read at its own date: a `thickness` term's
file at its date less at the state's, a `change` term's file as it
stands. A cap is Gaussian in great-circle distance d from its centre,
weighted exp(-d^2 / (2 w^2)) with no cut-off: a `cap_fraction` change is
`fraction` times the weight times the state's ice thickness, so it acts
only where there is ice (and, `fraction` no less than -1, takes no more
than there is); a `cap_thickness` change is `thickness_m` times the
weight, and a negative one, whose tail reaches past the ice, takes at
each point no more than the ice the state holds there. The surface loads
are read at their own dates (a `field` file with a time axis at its
`date_kyr_bp`, or at its `date` in its `unit` on the file's own axis),
and the tide and the seismic sources act whole.

The solve is a run of one node of the library's evolution (`Evolution`;
`Evolution3D` under lateral moduli, below) on the deck's elastic
structure with no relaxation, the solve of its start alone on the
unrelaxed body:

- under frozen shorelines (`ocean.shorelines`, the default), the linear
  sea level equation on the state's shorelines: the ice change loads the
  Earth only where the state's ice is grounded, 917 kg m^-3 times the
  change times (1 - C), in the water budget, so a change of floating ice
  loads nothing; under an ice change alone this is the elastic sea-level
  fingerprint of the change;
- under migrating shorelines, the nonlinear sea level equation: the ice
  is the state's plus the change, and the shorelines follow the
  flotation of that ice on the sea level the solve reaches, so that ice
  grounding on the ocean displaces it and melting marine ice floods, and
  a layer laid on the solid surface moves them as the sea level does.

The surface loads join the solve by their properties, the tide through
the body's tidal Love numbers and its own potential in the sea level
change (under migrating shorelines, in the flotation too), and the
seismic sources force the solid body, its spheroidal and toroidal response,
within the same equation: with an ocean and a fault alone the sea level
equation is still solved, the ocean's water conserved (under frozen
shorelines the sea level change integrates to zero over the state's
ocean). The rotational feedback is `rotation`'s; the problem takes no
time step, so the full feedback is closed on the elastic response alone,
and the moments of inertia need not sustain the model's fluid limit
(`moments`). The solve's tolerance is `solver.rtol`; a solve that does
not converge is an error (under migrating shorelines a cell flipping
back and forth can keep the nonlinear solve above a tight tolerance:
loosen it).

The output is a surface file of static SI fields on the run's
Gauss–Legendre grid:

| Field | Unit | Meaning |
| --- | --- | --- |
| `ice_change` | m | with an ocean: the ice thickness change applied (before the grounded-ice mask; zero without an ice change) |
| `load` | kg m^-2 | with a surface load: every load applied, summed as given (a cap's or a field's values as sampled at the grid points, a disk's or a table's synthesised from its coefficients) |
| `load_in_water_budget` | kg m^-2 | with a surface load in the water budget: the sum of those loads applied, band-limited to `lmax` |
| `load_outside_water_budget` | kg m^-2 | with a surface load outside it: likewise |
| `solid_surface_change` | m | with a load that moves the solid surface: the height of the layers laid on it, each load over its density |
| `relative_sea_level_change` | m | the sea-level change |
| `displacement` | m | the radial surface displacement |
| `tangential_potential` | m | the potential V of the tangential displacement (the displacement its surface gradient) |
| `toroidal_potential` | m | under seismic sources, and on lateral moduli: the potential W of the toroidal displacement (minus the radial unit vector crossed with its surface gradient), in the frame of no net rotation of the surface, a rigid rotation of the body being undetermined |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |
| `ocean_function` | 1 | with an ocean: the state's ocean function |
| `ocean_function_final` | 1 | migrating shorelines alone: the ocean function the solve reaches |
| `displacement_east`, `displacement_north` | m | with `output.fields` `"horizontal_displacement"`: the tangential displacement's components to the east and to the north (Optional and smoothed fields, below) |
| `<field>_smoothed` | the field's | with `output.smoothing`: each field of the response smoothed (likewise) |

Its metadata record the config and deck paths (and the moduli file's,
where given), the state's inputs as read, `ice` (`<file>:<variable>
(<slice>)`) and `sea_level` (the same after its quantity, `topography`
or `sea level`), the slice being the state's date, `<d> kyr BP`, or
`static` for a file without a time axis (the land held follows the sea
level's, `, land <file>:<variable> (<slice>)`, its slice the date for a
field on a time axis and `held at every date` for a static one); the ice
change's terms,
`ice_change`, each described (`thickness <file>:<variable>, from the
state's date to <d> kyr BP`, `change <file>:<variable>[, at <d> kyr BP]`,
`cap at latitude <lat>, longitude <lon>, width <w> deg, <t> m of ice` or
`..., fraction <f> of the state's ice`) and joined by `; ` (absent
without terms); the loads (`load`), the tide (`tide`) and the sources
(`seismic`), each described as the quasi-static file's are (a `field`
term's with `, at <d> kyr BP`, or `, at <t> <unit>` on its own axis) and
joined by `; ` (absent without them);
the forcing's counts, `load_terms` and `tide_coefficients`, and, where
the load is exactly one cap or the tide exactly one coefficient, its
members (`load_latitude_deg`,
`load_longitude_deg`, `load_width_deg`, `load_amplitude_kg_m2`;
`tide_degree`, `tide_order`, `tide_amplitude_m2_s2`,
`tide_imaginary_m2_s2`); the `shorelines`, the `rotation`, and the
angular velocity perturbation as `omega_x_rad_s`, `omega_y_rad_s` and
`omega_z_rad_s` (rad s^-1); `lateral` (0 on a radial Earth) and the
solve's closure as a 3D one's: on a radial Earth `step_solver` `direct`
and `step_iterations`, `solve_iterations` and `step_residual` zero (on
lateral moduli, below, the closure's own). Without an ocean the state's
inputs, the shorelines and the fields of the state (`ice_change`,
`ocean_function`) are left out, and `ocean` is `none` (Without an ocean,
below).

### The static problem on lateral moduli

In the static problem, a `model.moduli` file whose moduli vary laterally
makes the solve 3D: the same problem, under the same forcing,
shorelines, rotation and moments, on the laterally heterogeneous elastic
Earth. Its unrelaxed solid varies laterally, so the solid's response is
no Love-number closure: the solve is the one-node run of the 3D
evolution (`Evolution3D`), the combined fixed point of the strain, the
ocean load and the rotation over a step of zero length about the radial
reference (docs/adjoints.md §13k, §13n), under migrating shorelines with
the nonlinear equation solved in each of its passes. It is what a 3D
run through time solves at its start; the solid is the deck's elastic
structure carrying the lateral moduli, and no viscosity enters. With the lateral part of the moduli
zero (a file stored laterally but uniform) it is the radial problem's
answer to the solves' tolerance.

The members are the radial problem's, and the 3D step's members of
the solver block apply (The solver block, above): `solver.step` chooses
how the zero-length step is closed, `"fixed_point"` (the default) or
`"conjugate_gradients"` (the sea level closure nested over
conjugate-gradient solves of the lateral elastic systems);
`solver.step_rtol` (1e-9) and `solver.step_max_iterations` (60) are its
tolerance and budget, and a closure that misses them is an error;
`solver.transform_degrees` and `solver.cache_mb` set the transforms;
`solver.rtol` is the sea level solves' tolerance, as in 1D. `scheme` is
refused: there is no time step. Over MPI ranks the solve is distributed
over radial segments like the 3D runs through time, the cuts balancing
the elements the moduli perturb (the static problem on a radial Earth is
the first rank's alone). The coupling carries every solid element where
the moduli are perturbed, so the cost grows with `lmax` as a 3D step's
does: the example config `examples/configs/sea_level_fingerprint_3d.json`
takes about 3 s and 0.9 GB at lmax 32 on four threads, and about 23 s
and 6 GB at lmax 64.

The output is the radial problem's surface file, its fields the 3D
answer's, with the radial model's answer under the same forcing beside
them, so that the lateral moduli's effect is a subtraction:

| Field | Unit | Meaning |
| --- | --- | --- |
| `relative_sea_level_change_radial_model` | m | the radial model's sea-level change under the same forcing |
| `displacement_radial_model` | m | its radial surface displacement |
| `potential_radial_model` | m^2 s^-2 | its potential perturbation at the surface |

Its metadata add `lateral` (1), the moduli file's reference
(`moduli_reference`: `lateral_mean` or `file_radial`), the step solver
(`step_solver`), the closure's passes (`step_iterations`), its
conjugate-gradient iterations (`solve_iterations`, 0 under the fixed
point) and its last residual (`step_residual`), and the radial model's
angular velocity perturbation (`omega_x_radial_model_rad_s`,
`omega_y_radial_model_rad_s`, `omega_z_radial_model_rad_s`).

### The run through time

The quasi-static problem: viscoelastic GIA under the forcing over the
window. With the state at the window's start (`"at": "start"`) the run
starts from it: the background is built from `ocean.ice` and the state's
sea level read at `times.start_kyr_bp` (a file without a time axis as it
stands), and the load at each time node is the grounded part of the ice
change there, each `thickness` term's file read at the node's date less
its value at the start, each `change` term's read at the node's date (or
held, without a time axis), each cap held or scaled by its history,
beside the surface loads, the tide and the sources there (The forcing
block, above). The outputs are changes from the window's start. The time
nodes are `times.nodes` evenly spaced dates from `start_kyr_bp` to
`end_kyr_bp`, or the dates of `at_kyr_bp`, stepped by `solver.scheme`.
Without an ocean the run is the body's under the loads, the tide and the
sources alone (Without an ocean, below).

A run counting forwards on its own clock (`times.unit`) is the same run
with no calendar: its state is the snapshot of its files at
`ocean.date_kyr_bp` (a state of files without a time axis needs none),
put at the window's start; its nodes, histories, onsets and a field's
time origin are times on its clock in its unit; its ice changes by caps,
held or scaled by their `history`, and by `change` files without a time
axis, held through the run. A post-seismic relaxation over decades, a
surface load cycled over years, or a tide over days are runs of this
kind; a deglaciation, whose ice history is dated, is not.

The `ocean` block changes the start and the equation, as pyslfp's
`run_ice_history` does:

- `ocean.shorelines: "migrating"` solves the nonlinear sea level equation
  at every node instead of the linear one: the ocean function follows the
  flotation condition as the ice and sea level change, so melting marine
  ice floods and grounding ice displaces the ocean. The ice thickness at
  each node is the background's plus the change since the start, and the
  land held at each node is `ocean.land`'s there (read at the node's date
  where it is on a time axis: The state and the ice change, above). Where a
  change is present at the start (a `change` file whose change is not
  zero there, a cap without a history), the run solves its start, the
  nonlinear equation on the unrelaxed body under the state's ice plus
  that change, and its first node is that solve; otherwise the first
  node is the state itself. In a 3D run the nonlinear solve runs in every
  pass of each step's combined fixed point. The kernels of such a run
  (`kernel`) linearise about the run's own shorelines, its start's
  included (each node's land held among what placed them, the land being
  data and not a parameter), and their `initial_sea_level_dual` is not
  zero, as it is under frozen shorelines.
- `"at": "end"` makes the state the one the run reaches at
  `times.end_kyr_bp` (usually the present day), and calibrates the
  initial sea level against its sea level (`ocean.calibration`): the
  standard device when the ice history carries thickness alone. The ice
  changes are then `thickness` terms, each read back from the end: the
  ice at the window's start is `ocean.ice` at the end plus each term's
  thickness at the start less its thickness at the end, so a state and a
  term naming one file give that file's ice at the start (to rounding),
  and the load at a node is each term's change since the start. The run
  then starts as pyslfp's calibration does: the ice is the run's at the
  start, and the starting guess for the sea level is the target itself (a
  topography converted with the state's ice), so the state's sea level
  and ice files are read at the window's end alone, which must lie
  within their time spans unless the inputs hold their ends
  (`"outside_span": "hold"`, the target then the files' last state).
  The land held, `ocean.land`, on a time axis, is read at the window's
  end under frozen shorelines, whose ocean function, the guess's, is the
  state's throughout, and at every node's date under migrating ones,
  each of the calibration's runs holding at each node that node's land
  and the start the first node's. Under frozen shorelines the
  problem is linear and one run gives the answer exactly, the guess's
  shorelines kept; under migrating shorelines the calibration iterates,
  by the classical update or by steepest descent with the adjoint
  gradient, each accelerated by Anderson mixing. The calibration runs
  before the run proper, which then costs one more run.

A sea level solve that misses `solver.rtol` is an error, and in 3D so is
a node whose combined fixed point does not converge within
`solver.step_max_iterations` to `solver.step_rtol` (under migrating
shorelines a cell flipping back and forth can keep the nonlinear solve,
or the 3D fixed point, above a tight tolerance: loosen it). A 1D run
refuses the solver members of the 3D step. Under lateral moduli the
start's instantaneous response is the combined fixed point of a
zero-length step (`Evolution3D`).

The run starts at the first node with no memory before it: the response
there is the unrelaxed one to the forcing at the first node, the static
problem's for the same forcing and state to the difference of their
discretisations, the radial stepper's unrelaxed systems carrying the
branch moduli at the memory points rather than at the nodes (about 1e-11
relative on the test decks under a load, 5e-10 under a point source's
sharper forcing; under lateral moduli to the solves' tolerances). Each
step then advances from one node to the next under the forcing AT the
next node, the scheme's end-time loads being that node's: a load that is
non-zero at the first node includes the elastic jump at the start (the
run is the response to a load switched on there), while a load ramped up
from zero has none.

The output is a surface file on the run's Gauss–Legendre grid with one
slice per time node, its time axis in seconds relative to the present
(negative in the past) in a dated run, and the run's own clock in
seconds (a node at t in the unit at t times the unit) in one counting
forwards. Its fields are SI node fields with linear time interpolation:

| Field | Unit | Meaning |
| --- | --- | --- |
| `relative_sea_level_change` | m | the sea-level change from the window's start |
| `displacement` | m | the radial surface displacement |
| `tangential_potential` | m | the potential V of the tangential displacement (the displacement its surface gradient) |
| `toroidal_potential` | m | under seismic sources, and in a 3D run: the potential W of the toroidal displacement (minus the radial unit vector crossed with its surface gradient), in the frame of no net rotation of the surface, as the static problem's; zero at a node that has none (a 3D run's start on the radial unrelaxed body) |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |
| `initial_sea_level` | m | with an ocean, static: the initial sea level the run started from (the sea level at a node is this plus `relative_sea_level_change` there) |
| `ocean_function` | 1 | with an ocean: under migrating shorelines the ocean function at each node; under frozen ones the state's, a static field, which the run holds at every node |
| `displacement_east`, `displacement_north` | m | with `output.fields` `"horizontal_displacement"`: the tangential displacement's components to the east and to the north at each node (Optional and smoothed fields, below) |
| `<field>_smoothed` | the field's | with `output.smoothing`: each field of the response smoothed at each node (likewise) |
| `load` | kg m^-2 | with a surface load: every load applied at each node, summed as given (each term's field times its factor there) |
| `load_in_water_budget`, `load_outside_water_budget`, `solid_surface_change` | kg m^-2, kg m^-2, m | with a load of each kind: the loads applied at each node by their properties, as the static problem's (The static problem, above) |
| `relative_sea_level_change_radial_model`, `displacement_radial_model`, `potential_radial_model` | m, m, m^2 s^-2 | a 3D run: the radial reference model's, under the same forcing (below) |

Its metadata record the shorelines (`shorelines`), where the initial sea
level came from (`initial_sea_level`: the state's sea level at the
window's start, or the calibration with its runs and misfits), the config
and deck paths (and the viscosity and moduli files', where given), the
state's inputs as read, `ice` (`<file>:<variable> (<slice>)`) and
`sea_level` (the same after its quantity), the slice the state's end of
the window, `the window's start, <d> kyr BP` or `the window's end, <d>
kyr BP`, or for a file without a time axis `static, at the window's
start` or `... end` (counting forwards, the state's date, `<d> kyr BP`,
or `static`), the land held following the sea level's, `, land
<file>:<variable> (<slice>)`, its slice `held at every date` for a
static field and for one on a time axis `each node's date` under
migrating shorelines, the state's end of the window under frozen ones;
the ice change's terms, `ice_change`, each described
(`thickness <file>:<variable>, from the state's instant to each node's
date`, `change <file>:<variable>, read by date` or, counting forwards,
`..., held through the run`) and joined by `; `
(absent without terms); the loads, the tide and the sources beside the
ice, where given (`load`, `tide`, `seismic`; a history given on the
run's clock described as `history in <unit> [[t, f], ...]`, an onset as
`from <t> <unit>`, a field's time origin as `time origin <t> <unit>`);
the `rotation`; the nodes, in a dated run their dates as `dates_kyr_bp`
(the dates given, listed or evenly spaced) and in one counting forwards
their times in its unit as `times` with the unit's name as `time_unit`
(`kyr`, `yr` or `day`; `time` says how the axis is formed); and the
angular velocity perturbation at each node as `omega_x_rad_s`,
`omega_y_rad_s` and `omega_z_rad_s` (rad s^-1); `lateral` (1 in a 3D
run), the forcing's counts and single terms as the static problem's
(above), the time step (`scheme`) and each node's closure:
`step_solver` (`"direct"` in 1D, `"fixed_point"` or
`"conjugate_gradients"` in 3D), and one per node `step_iterations` (the
passes of its combined fixed point), `solve_iterations` (their
conjugate-gradient iterations, zero under the fixed point) and
`step_residual` (its last residual), zero throughout in 1D and at the
first node of a run without lateral moduli; in a 3D run, the radial
reference model's angular velocity perturbation at each node,
`omega_x_radial_model_rad_s`, `omega_y_radial_model_rad_s` and
`omega_z_radial_model_rad_s`. Without an ocean the shorelines, the
initial sea level, the ocean function and the state's inputs are left
out, and `ocean` is `none`.

A 3D run writes the radial reference model beside it, as the static
problem on lateral moduli does: the radial Maxwell body the 3D
run is about (the deck's moduli, and the viscosity file's radial
reference or the deck's column), run through the same window by the 1D
evolution from its own world, the same state under the same forcing,
shorelines and rotation, with its own calibration where the run
calibrates. It is run on the first rank alone once the 3D run is done,
and is not checkpointed: a resumed run runs it whole. Its cost is a 1D
run's.

### Optional and smoothed fields

Beside the fields every surface file holds, `output.fields` adds
optional ones and `output.smoothing` smoothed copies of the response
(The output block, above), in the static problem's file and in a run
through time's, each field with the support, time interpolation and unit
of the field it is made from.

**The horizontal displacement.** The surface displacement is u = U r +
grad_1 V - r x grad_1 W: U the radial displacement (`displacement`), V
and W the potentials of the tangential part (`tangential_potential`,
`toroidal_potential`, W zero where the file has none), grad_1 the
gradient on the unit sphere and r the radial unit vector. With
`"fields": ["horizontal_displacement"]` the file holds the tangential
part's components along the geographic unit vectors at each grid point,
to the east (increasing longitude) as `displacement_east` and to the
north (increasing latitude) as `displacement_north`, in m: in colatitude
theta and longitude phi, east = (1 / sin theta) d_phi V - d_theta W and
north = -d_theta V - (1 / sin theta) d_phi W. They are synthesised from
the expansions of V and W to lmax through the spin-weighted harmonics of
upper index ±1 (the library's `DisplacementField`), exactly, not
differenced on the grid. The Gauss–Legendre grid has no point at a pole,
where east and north are undefined, so every point has both. W is in the
frame of no net rotation of the surface (a rigid rotation of the body
costs no strain and is undetermined), and so is the horizontal
displacement; a run without a toroidal part (a radial Earth under loads,
a tide or an ice change) moves the surface by grad_1 V alone, curl-free.
The metadata state the convention (`horizontal_displacement`).

**Smoothing.** With `"smoothing": {"kind": "gaussian", "width_deg": s}`,
each field of the response the file holds (`relative_sea_level_change`,
`displacement`, `tangential_potential`, `toroidal_potential`,
`potential`, and in a 3D run `relative_sea_level_change_radial_model`,
`displacement_radial_model` and `potential_radial_model`), and the east
and north components where asked, is written again smoothed, as
`<field>_smoothed`. The smoothing is the convolution over the sphere with
the Gaussian cap of the cap measurements (kernel, Measurements, below),

    w(d) = exp(-d^2 / (2 s^2)),  d the great-circle distance,

normalised to unit integral over the sphere and cut off nowhere, s its
standard deviation in angle: the weight falls to e^-1/2 at d = s, and its
full width at half maximum is 2 sqrt(2 ln 2) s, about 2.355 s. A smoothed
field's value at a point is the field's mean over the cap about that
point. By Funk and Hecke the convolution of a field of degree at most
lmax is its expansion with degree l weighted by

    lambda_l = int_0^pi w(d) P_l(cos d) sin d dd / int_0^pi w(d) sin d dd

(lambda_0 = 1), synthesised: one expansion and one synthesis a field a
node, exact, the integrals taken by Gauss–Legendre quadrature to
rounding (the library's `CapMeanWeights`). The east and north components
are smoothed as a vector: they are the components of the displacement of
the smoothed potentials, grad_1 of the smoothed V less r x grad_1 of the
smoothed W. The metadata record `smoothing_kind` (`gaussian`),
`smoothing_width_deg` (s as given), `smoothing_degree_weights` (lambda_l
for l = 0 to lmax) and the definition (`smoothing`).

A cap measurement of `kernel` of the same width (`relative_sea_level_cap`,
`displacement_cap`, `potential_cap`, and the series of a `decay_time`)
reads the same mean about its centre: its dual is the cap's band-limited
kernel, sum_l lambda_l (2l + 1) P_l(cos d) / (4 pi a^2) about the centre
with the same degree weights (the library's `CapMeanDual`), so a
measurement of a field is the field smoothed over the cap read at the
centre, to rounding, at every width: at the seismic examples' 2° and
lmax 96 as at 20° and lmax 32, where both are the cap means of a fine
quadrature of the band-limited fields to rounding. A
`displacement_vector_cap` measurement reads the displacement smoothed as
a vector the same way, at the centre along its direction: the upward part
the smoothed radial displacement, the east and north parts the centre's
components of the displacement of the smoothed potentials, which are
`displacement_east_smoothed` and `displacement_north_smoothed` there, to
rounding (the library's `CapDisplacementMeasurement`).

The fields of the state and of the forcing are not smoothed: the
initial sea level, the ocean functions (a smoothed ocean function is not
an ocean function), the ice change and the loads applied are the run's
inputs as applied, sampled at the grid's points, and not its response.
With an ocean, `relative_sea_level_change_smoothed` is the sea-level
change smoothed over the whole sphere, land included, as the field is
defined there: it is not a mean over the ocean's points alone.

Smoothing serves fields whose point values are not converged in lmax:
the response to a seismic source shallower than the truncation
wavelength (What the seismic forcing resolves, above), whose means over
caps wider than that wavelength are converged; the seismic example
configs smooth over 2°. A field resolved at lmax, a deglaciation's sea
level or a load's response, needs none.

### Exchange tables and the volume file

Beside the surface file, `output` takes an exchange table and a volume
file (The output block, above).

The exchange table holds the coefficients of the fields on the surface
file's grid, in the format of AdGIA's benchmarks:

- `#` provenance lines: what the table holds, `<elastic|viscoelastic>
  surface response from sl3dpp solve (<radial|laterally heterogeneous>
  <elastic|viscoelastic> model, <no ocean|shorelines frozen|shorelines
  migrating>, rotation <rotation>)` (`viscoelastic` through time) and the
  quantities (`, no gravity` follows the ocean's words in a run without
  gravity); the code's version and the date; `config <path> model
  <deck> moduli <file or none>`, through time with `viscosity <file or
  none>`; with an ocean, `state ice <ice>, <sea level>`, the state's
  inputs as read; and a line `forcing <term>` per term of the forcing,
  the ice change's first, each described as the metadata describe it (a
  `field` term's file and variable, with its date, time origin and
  history where set; a load's place in the water budget);
- named scalars, one per line as `name value`: `radius` (m),
  `surface_gravity` (m s^-2), `lmax`, `lateral`, `load_terms` and
  `tide_coefficients` (the surface loads and the tide's coefficients: an
  ice change has no scalars of its own); then, when the load is exactly
  one `cap`, `cap_latitude_deg`, `cap_longitude_deg`, `cap_width_deg` and
  `cap_amplitude_kg_m2`, and when the tide is exactly one coefficient,
  `tide_degree`, `tide_order` and `tide_amplitude_m2_s2` (two numbers,
  the real and imaginary parts), so that a table of one cap and one tide
  carries the same scalars whatever else it holds; through time, `node`
  (the node's index), then in a dated run `time_kyr` (kyr from the
  window's start, the terms' clock) and `date_kyr_bp` (the node's date),
  and in a run counting forwards `time_<unit>` (the node's time in the
  run's unit: `time_kyr`, `time_yr` or `time_day`);
- `columns l m U_re U_im Phi_re Phi_im`, and, with an ocean, `SL_re SL_im`
  after them, and one row per (l, m), m >= 0: the coefficients of the
  surface displacement (m), the potential (m^2 s^-2) and the relative sea
  level change (m) in the harmonics of the coefficient tables (The
  forcing block, above), SI, `%.17g`. Without gravity there is no
  potential, and the columns are `l m U_re U_im`. A reader that matches
  columns by name reads tables with and without an ocean, and with and
  without gravity, alike.

The volume file holds `radial_displacement` (m), `deviatoric_strain`
(sqrt(e:e), dimensionless) and, on lateral moduli, `mu_relative` and
`kappa_relative` (the perturbations applied) at the memory nodes of every
solid element but the one holding the centre, with the model's density
as the radial reference; planetmodel's `slices.cutaway` and
`plotting.cutaway` draw it (as `examples/python/plot_static_3d.py` does
for the library's example of the same problem). The response inside the
body is the elastic response to the total load the static solve's body
answered (the surface loads, the ice's grounded load and the ocean's)
and to the potential it answered (the tide, and the rotation's
centrifugal change, its degree-0 part the spherical mean of such a
change): by the radial reference's systems on a radial Earth, and on
lateral moduli by the operator's conjugate-gradient elastic solve (to
1e-12 within 200 iterations; one that misses them is an error). Asked
for it, the static solve records its loads, and the file costs one more
elastic solve.

### Without an ocean

A config without an `ocean` block has no state: the body has no ocean,
no ice and no instant of its own, and the run, in either problem, is the
library's evolution on a body without an ocean
(`GRDSolver::NoOceanState`, the ocean function zero everywhere) under
the forcing. Each node's sea level solve is then the response of the
solid body, with the rotational feedback of `rotation`, closed in one
pass (a 3D run's combined fixed point remains, over the lateral terms
and the rotation): there is no water to load the body or to conserve.

- The forcing is any mix of `forcing.load`, `forcing.tide` and
  `forcing.seismic`, as with an ocean. `forcing.ice_change` is refused:
  an ice change is a change of the state's ice, and its water has
  nowhere to go; give the ice's load as a `load` term (917 kg m^-3 times
  the thickness, as `examples/configs/no_ocean_ice_cap.json` does), or give
  the state.
- A load's `water_budget` and `moves_solid_surface` may be left out,
  each `false`. The water budget has no ocean to change, and has no
  effect; a layer on the solid surface (with its `density_kg_m3`, as
  with an ocean) lowers the relative sea level change by its thickness
  and does nothing more.
- What only the state has is absent with it: the shorelines, the
  calibration, the land held and the state's date. The static problem
  has no date; a `field` load term's `date_kyr_bp` still dates its own
  file. `rotation` and `moments` act as with an ocean, `"full"` by
  default, and a run through time under the full feedback is held to
  the model's fluid limit as one with an ocean is (Rotation and moments,
  above).
- Checkpoints work as with an ocean; the hash records that the body has
  none, so a run with an ocean is refused a run's without one, and the
  other way round.

With `"rotation": "none"` the run is the solid body's response to the
forcing alone: in the static problem its elastic response (on a radial
Earth the unrelaxed solves of the radial systems; on lateral moduli the
zero-length step's closure, The static problem on lateral moduli, above),
through time its viscoelastic response, the first node the unrelaxed
solve under the forcing there with no memory before it and each step
ending under the forcing at its end node. The examples'
`examples/configs/no_ocean*.json` are runs of this kind, counting forwards
on their own clock in kyr; those on lateral structure set
`solver.step_rtol` 1e-10 and `step_max_iterations` 200, tighter than the
defaults.

The fields are those with an ocean: `relative_sea_level_change` is the
change of an equipotential's height above the solid surface,
-(u + (phi + psi) / g) with psi the potential of the rotation's change
and of the tide, less the height of any layer, with no uniform shift;
`displacement`, `tangential_potential`, `toroidal_potential` (where the
run has one), `potential` and the loads applied (in a run through time
at each node). The fields and metadata of the state (`ice_change`,
`ocean_function`, `initial_sea_level`; `ice`, `sea_level`, `shorelines`)
are left out, and the metadata's `ocean` is `none`. The summary reads `no
ocean` where it would give the shorelines (Summary, below).

`examples/configs/no_ocean_load_cap.json` (a Hudson Bay cap loaded from
2 to 3 kyr on the run's clock, held to 7 kyr and removed by 7.5 kyr, on
the radial Maxwell deck) dated, its window the same 12 kyr in
quarter-kyr steps, ending at the present:

```json
{
  "problem": "quasi_static",
  "model": {"deck": "../data/prem8.gia.deck", "format": "maxwell"},
  "discretisation": {"lmax": 32},
  "times": {"start_kyr_bp": 12.0, "end_kyr_bp": 0.0, "nodes": 49},
  "forcing": {
    "load": [{"cap": {"latitude": 60.0, "longitude": -85.0, "width_deg": 5.0,
                      "amplitude_kg_m2": 1.834e6},
              "history_kyr_bp": [[10.0, 0.0], [9.0, 1.0], [5.0, 1.0], [4.5, 0.0]]}]
  },
  "rotation": "none",
  "output": {"surface": "out/sea_level_no_ocean/sea_level_no_ocean.nc"}
}
```

This is `examples/configs/sea_level_no_ocean.json`: its displacement,
potential and tangential potential are `no_ocean_load_cap.json`'s at
every node to 1e-11 of their largest values, the rounding of the two
clocks' node times, and its `load` to 2e-15. With a degree-2 tide beside
the cap, `"tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0,
"history_kyr_bp": [[12, 0], [10, 1]]}]` switches a potential on over the
first two kiloyears, and without `"rotation": "none"` the run takes the
full rotational feedback (Rotation and moments, above).

### Without gravity

`"gravity": "none"` is the problem without gravity: the solid body's
response to surface loads and seismic sources with no self-gravitation,
no potential and no buoyancy, the systems of the radial operator taking
the displacement alone (docs/design.md §3a). Much post-seismic work is
done so; its solves are quicker, and reach higher degrees. The default,
`"full"`, is the problem with gravity, every other section's; giving it
changes nothing.

- **What it takes.** A radial model (a deck, with a radial viscosity or
  moduli file if any) in either problem: the static problem is the
  elastic response, a run through time the viscoelastic one, on the
  run's own clock or dated, under `forcing.load` and `forcing.seismic`,
  either or both.
- **Loads.** A load is a mass per unit area sigma, as with gravity, in
  every form (`cap`, `disk`, `field`, `coefficients`, with histories).
  Leaving gravity out drops the self-gravitation of the deformation and
  the potential, not the weight of what sits on the surface: a load acts
  as the normal traction g sigma pressing on the surface, g the deck's
  reference surface gravity (the exchange tables' `surface_gravity`,
  and `love-numbers`' `surface_gravity_m_s2`), so the static response to a load of degree l is
  the non-gravitating traction numbers times g sigma, g h_u sigma and g
  l_u sigma. At degree 0 the load is a uniform pressure, and the body
  compresses. A load's `water_budget` has no ocean to change and no
  effect, as without an ocean with gravity; a layer that moves the solid
  surface (`moves_solid_surface`, with its `density_kg_m3`) acts by its
  weight alone, its thickness entering only the relative sea level
  change, which is not formed (it is written, `solid_surface_change`, as
  without an ocean).
- **Degree 1.** A load with a degree-1 part exerts a net force on the
  body, which nothing balances without gravity (with gravity the load's
  own attraction of the body does), and has no equilibrium. The run
  removes it: the loads applied, every term summed, act at every node
  less their degree-1 part, whatever its size, as a finite-element solver
  of the problem would project out the rigid translation. Physical loads
  balance, and theirs is the grid's error: a pair of antipodal caps or
  disks of the same amplitude and history has none, but caps are sampled
  on the grid, and a pair narrower than the grid's spacing aliases one
  (at lmax 32, a pair of 3° caps 1e-10 of its size, of 2° 5e-6, of 1°
  7e-3), which is removed as every part is: the run answers the pair, to
  the odd degrees above 1 that the sampling aliases too (6e-7 and 3e-4 of
  the response at 3° and 2°), which exert no net force. A load that does
  not balance, a cap or a disk alone, a coefficient table with l = 1 rows,
  is answered as the load less its degree-1 part (if you specify a net
  force, you get the response to the rest). The part is measured as the norm of the loads' degree-1
  coefficients over that of all of them; where it is above 1e-12 at some
  node (or, static, at the problem's instant) the run says so in one line
  on standard error, `sl3d: forcing.load: the loads' degree-1 part (... of
  their size at ...) ...`, with its largest share and that node, and
  runs on. The response is in the frame of the centre of mass of the
  solved region, as every response without gravity is (The body, below).
  The surface file's loads (`load`, `load_in_water_budget`,
  `load_outside_water_budget`) are the loads as given, their degree-1
  part included.
- **What it refuses**, each naming its member: the `ocean` block (there
  is no sea level without gravity), `forcing.tide` (an external
  potential) and `forcing.ice_change` (it needs an ocean); a `rotation`
  other than `"none"`, which is the
  default without gravity (the rotational feedback reads the inertia's
  change from the degree-2 potential, which is absent);
  `"moments": "hydrostatic"`; a model file that varies laterally (a 3D
  run without gravity is not built: `model.moduli.file` or
  `model.viscosity.file`); the `trajectory` and `kernels` blocks, and
  `kernel` itself, at `gravity` (no kernels); `output.volume` (the
  response inside the body is solved by the systems with gravity); and,
  as under any seismic sources, `checkpoint`. A run under loads alone is
  checkpointed and resumed as any other.
- **The body.** A fluid core has no degree-l pressure without gravity
  above degree zero: there the systems stop at the core-mantle boundary,
  the mantle's base traction-free, and the core (with an inner core
  below it) is not solved; at degree zero the core keeps its bulk
  modulus. A deck with more than one fluid region is refused. Degree 1
  is kept, in the frame of the centre of mass of the solved region (the
  density-weighted mean displacement zero), the frame the potential sets
  with gravity.
- **The outputs.** The surface file holds `displacement` and
  `tangential_potential`, `toroidal_potential` under seismic sources, and
  under loads the loads applied as without an ocean with gravity (`load`,
  `load_in_water_budget`, `load_outside_water_budget`,
  `solid_surface_change`, each where a term has it); there is no
  `potential` and no `relative_sea_level_change` (an equipotential's
  height has no meaning without gravity). Its metadata add `gravity`,
  `none` (absent with gravity, as `ocean` is with an ocean), beside
  `ocean`, `none`, `rotation`, `none`, and the loads (`load`) as with
  gravity. An exchange table holds the displacement's coefficients alone
  (`columns l m U_re U_im`), its heading saying `no ocean, no gravity`
  and its scalars and `forcing` lines those of any run. The summary
  reads `no ocean, no gravity` where it would give the shorelines
  (Summary, below).
- **The hash.** A run without gravity is another run to the hashes of
  the checkpoints and the trajectory store; `"gravity": "full"`, or no
  member, hashes as every run with gravity does.

`examples/configs/no_gravity_postseismic.json` is a run of this kind:
the thrust of `sea_level_postseismic.json` under the synthetic South
America's west coast (centred at 22° S, 74.5° W and 25 km deep, 600 by
150 km, dipping 18°, 2e22 N m as 12 by 3 point sources) on the
post-seismic Earth (`prem8.postseismic.deck`: a 50 km elastic lid over
a 1e19 Pa s asthenosphere, a 1e20 Pa s upper mantle and a 1e22 Pa s
lower mantle) at lmax 96, its nodes the 23 of `sea_level_postseismic.json`
on the run's clock in years, with exchange tables at the first and last:

```json
{
  "problem": "quasi_static",
  "model": {"deck": "../data/prem8.postseismic.deck", "format": "maxwell"},
  "discretisation": {"lmax": 96},
  "times": {"unit": "yr", "at": [0.0, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5, 10.0, 13.0, 17.0, 22.0, 30.0,
                                 40.0, 55.0, 75.0, 100.0, 140.0, 200.0, 300.0]},
  "forcing": {
    "seismic": [{"fault": {"latitude": -22.0, "longitude": 285.5, "depth_km": 25.0,
                           "strike_deg": 0.0, "dip_deg": 18.0, "rake_deg": 90.0,
                           "length_km": 600.0, "width_km": 150.0,
                           "moment_nm": 2.0e22,
                           "patches_along_strike": 12, "patches_down_dip": 3},
                 "onset": 0.0}]
  },
  "gravity": "none",
  "rotation": "none",
  "output": {"surface": "out/no_gravity_postseismic/no_gravity_postseismic.nc",
             "exchange": "out/no_gravity_postseismic/no_gravity_postseismic_{node}.txt",
             "exchange_nodes": [0, 22],
             "fields": ["horizontal_displacement"],
             "smoothing": {"kind": "gaussian", "width_deg": 2.0}}
}
```

`examples/configs/no_ocean_postseismic.json` is the same run with
gravity: no `gravity` member (`"full"`), no ocean, `"rotation": "none"`.
Averaged over a Gaussian cap of 2° over the fault (the fields of a
fault this shallow are not converged in lmax point by point, their 2°
means are), the uplift is 0.153 m at the event without gravity and
0.147 m with it, 0.356 m at a century against 0.296 m, and 0.297 m at
300 years against 0.200 m: without the self-gravitation and the
buoyancy of the deformation only the elastic lid holds the relaxing
surface down. The horizontal motion hardly differs (up dip over the
fault, 0.851 m at the event either way and 0.700 m against 0.701 m at
300 years). About 35 s on four threads against 47 s with gravity.

`examples/configs/no_gravity_load.json` is a run under loads: the Hudson
Bay cap of `no_ocean_load_cap.json` and its antipode, the same cap at
60° S, 95° E, both loaded from 2 to 3 kyr on the run's clock, held to
7 kyr and removed by 7.5 kyr, on the radial Maxwell deck at lmax 32, with
exchange tables at 3, 7 and 12 kyr:

```json
{
  "problem": "quasi_static",
  "model": {"deck": "../data/prem8.gia.deck", "format": "maxwell"},
  "discretisation": {"lmax": 32},
  "times": {"unit": "kyr", "start": 0.0, "end": 12.0, "nodes": 49},
  "forcing": {
    "load": [{"cap": {"latitude": 60.0, "longitude": -85.0, "width_deg": 5.0,
                      "amplitude_kg_m2": 1.834e6},
              "history": [[2.0, 0.0], [3.0, 1.0], [7.0, 1.0], [7.5, 0.0]]},
             {"cap": {"latitude": -60.0, "longitude": 95.0, "width_deg": 5.0,
                      "amplitude_kg_m2": 1.834e6},
              "history": [[2.0, 0.0], [3.0, 1.0], [7.0, 1.0], [7.5, 0.0]]}]
  },
  "gravity": "none",
  "output": {"surface": "out/no_gravity_load/no_gravity_load.nc",
             "exchange": "out/no_gravity_load/no_gravity_load_{node}.txt",
             "exchange_nodes": [12, 28, 48]}
}
```

The pair's degree-1 part is 4e-16 of its size. Under each cap the
surface sinks 203 m once the load is on (3 kyr) and 1018 m by 7 kyr,
against 125 m and 327 m for the same run with gravity (`"gravity":
"full"`, `"rotation": "none"`): without buoyancy nothing holds the load
up but the lithosphere, and the mantle flows on. Once the load is gone the
surface springs back by its elastic part alone and stays down, 792 m at
12 kyr against 60 m with gravity. About 0.7 s on four threads against
1.1 s with gravity.

### Checkpoints and restarts

A run through time takes a `checkpoint` block and `restart`; both are
refused in the static problem.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `checkpoint.directory` | path | required in `checkpoint` | where the checkpoints go |
| `checkpoint.every` | integer >= 1 | required in `checkpoint` | the checkpoint stride, in time nodes |
| `checkpoint.max_walltime_min` | number | 0 | stop cleanly once a checkpoint is written after this many minutes; 0 for no limit |
| `restart` | boolean | `true` | whether to resume from a checkpoint found in `checkpoint.directory`; `false` starts over, replacing the checkpoints there |

Without a `checkpoint` block nothing is written until the run finishes.
With one, the run writes a checkpoint after every `checkpoint.every`-th
node it steps and after the last node. Each is a file `node_NNNNNN.ckpt`
in `checkpoint.directory`, named by its node, holding the run's state
there and the histories so far (the response's fields, the angular
velocity and the step's closure at each node; the loads applied and a
3D run's radial reference model are formed when the output is written).
It is written to a temporary file and
renamed into place, so an interrupted write leaves no partial checkpoint,
and only the newest two are kept.

When a run starts and finds a checkpoint in the directory, it resumes from
the newest one, with the histories up to that node restored. A resumed run
reproduces an uninterrupted one: bitwise within one process (the tests),
and to rounding as any two runs of one config (Running sl3d, above).
Rerunning a
finished run whose last checkpoint is at the final node just writes the
output again.

A checkpoint is bound to its run: it records a hash of the members that
decide the computation (everything but `checkpoint`, `restart`,
`solver.cache_mb`, `output` and the file inputs' `outside_span`, which
a run that runs computes alike either way, and on a radial model
`solver.transform_degrees`, which only a 3D run's steps use; the inputs'
variables, the sea level's quantity, the state's end of the window, the ice change's
terms, the surface loads and the tide's coefficients, every number whole,
and
whether the body has an ocean among them, and a run's having no gravity
(the member enters the hash only as `"none"`, so that a config with
gravity hashes the same with or without it); the
nodes by their form: evenly spaced dates by their window and count,
listed dates and times counted forwards by the list and, forwards, the
unit and the state's date, so that the same nodes in another form or
another unit are another run, their times possibly other in the last
bit) and of
the contents of the files the config names (the deck, the state's sea
level and ice files, each ice change term's file, and the land, the
surface loads' and the viscosity and moduli files where given). A run is
refused, with an error naming the
checkpoint, rather than resuming another run's state when any of these
differ: another time grid, tolerance or model, or the same config over an
input file that has changed since. The output, the checkpoint stride, the
wall-time limit and the transforms' cache may all change between a run
and its resumption, and the paths may change as long as the files'
contents do not, so a run directory can be moved. A checkpoint of a run
of the other dimension (a 1D run's against a 3D run's) is refused too,
and so is one of another format of this program's checkpoints (which
holds other histories), naming the file: `"restart": false` starts over.
The files are raw binary, for the machine that wrote them.

A calibrating run also keeps its calibrated initial state in the directory
(`initial_state.bin`, bound to the run by the same hash), and a resumption
reads it back instead of calibrating again.

A run that records a trajectory store (`trajectory.directory`, The
trajectory block, above) continues the store on resumption, from the
checkpoint's node, its records to that node the interrupted run's: a
chain of jobs completes the store with the run. A store complete
already is left as it is, and does not stop a resumption.

With `restart` set to `false` the run starts over: it first removes the
checkpoints it finds in the directory (and any kept initial state), then
writes its own. Only this program's checkpoint files (`node_NNNNNN.ckpt`,
and the `.node_NNNNNN.partial` it renames) are read, pruned or removed;
anything else in the directory is left alone. Leave `restart` at `true` in
a chain of jobs that is meant to resume.

`checkpoint.max_walltime_min` serves batch systems with a time limit.
After writing a checkpoint, a run that has been stepping for longer than
the limit and has nodes still to go stops cleanly, with the summary
`<meaning>; <the inputs as read>; checkpointed node <n> of <last> to
<directory> at the wall-time limit[, replacing <k> earlier
checkpoint(s)]; rerun the same config to resume` (with `"restart":
false`, `; set "restart": true (the default) to resume` instead), the
trajectory store's note ending it where there is one (Summary, below).
It exits with status 0 and writes no output yet, so a script can tell a
stopped run from a finished one by the summary or by whether
`output.surface` exists; rerunning the same config resumes it, so a
chain of jobs running one config completes the run. The clock starts
with the time stepping (a calibration's runs among it), after the model
and files are read.

### Summary

What the config was taken to mean, the inputs as read, then the files:

- the static problem: `<model>, elastic, static, shorelines
  <frozen|migrating>, rotation <rotation>[, moments hydrostatic]; ice
  <ice>, <sea level>[; ice change <change>][; <forcing>][; <field>...];
  wrote <surface>[ and <exchange>][ and <volume>]` (without an ocean,
  `<model>, elastic, static, no ocean, rotation <rotation>; <forcing>[;
  <field>...]; wrote ...`), ending on lateral moduli in
  the 3D closure's parenthesis, `(<n> fixed-point iterations, residual
  <r>)`, or under `"conjugate_gradients"` `(<n> conjugate-gradient
  iterations in <k> passes, residual <r>)`;
- a run through time: `<model>, viscoelastic, <n> nodes <start> to <end>
  kyr BP, shorelines <frozen|migrating>[, initial sea level calibrated],
  rotation <rotation>[, moments hydrostatic]; ice <ice>, <sea level>[;
  ice change <change>][; <forcing>][; <field>...]; wrote <surface>[ and
  <k> exchange table[s]]` (counting forwards, `<n> nodes <start> to
  <end> <unit> on the run's clock`), ending in a 3D run in the steps'
  closures summed, `(<i> fixed-point iterations over <s> steps)` or `(<i>
  conjugate-gradient iterations over <s> steps)`, then in `, resumed`
  when the run resumed from a checkpoint, and in `, replacing <k> earlier
  checkpoint(s)` when a run starting over removed some (with `"restart":
  false`, or over MPI ranks that had no checkpoint in common);
- with `trajectory.directory`, either summary ends in `; recorded the
  trajectory to the store <directory> (<size>)` (a resumed run's in `,
  continuing it from node <k>`, the checkpoint's node), or, where the
  directory holds the run's complete store already, `; the trajectory
  store <directory> holds this run complete (<size>), left as it is`; a
  run stopped at the wall-time limit ends in `; the trajectory store
  <directory> left incomplete at node <k>, the resumption recording on
  from there`;
- without an ocean, the state's words are left out: `<model>, elastic,
  no ocean, rotation <rotation>[, moments hydrostatic]; <forcing>[;
  <field>...]; wrote <surface>[ and <exchange>][ and <volume>]` in the
  static problem (ending as above on lateral moduli), and `<model>,
  viscoelastic, <n> nodes <start> to <end> kyr BP, no ocean, rotation
  <rotation>[, moments hydrostatic]; <forcing>[; <field>...]; wrote
  <surface>[ and <k> exchange table[s]]` through time (counting forwards
  as above; ending as above);
- without gravity, `, no gravity` follows `no ocean`: `<model>, elastic,
  static, no ocean, no gravity, rotation none; <forcing>[; <field>...];
  wrote <surface>[ and <exchange>]` and `<model>, viscoelastic, <n> nodes
  <start> to <end> <unit> on the run's clock, no ocean, no gravity,
  rotation none; <forcing>[; <field>...]; wrote <surface>[ and <k>
  exchange table[s]]` (dated, `<n> nodes <start> to <end> kyr BP`), the
  forcing loads and seismic sources.

`<ice>` and `<sea level>` are the metadata's `ice` and `sea_level`
(above): the state's file, variable and slice read. `<change>` is the
ice change's terms as the metadata's `ice_change` describes them, joined
by ` + ` (a run without terms leaves it out). `<forcing>` is what forces
the run beside its ice, `load (<n> term[s])`, `tide (<n>
coefficient[s])`, `seismic (<n> source[s])` or several joined by ` + `
(left out without any). Each `field` load follows as read, `field
<file>:<variable> (<slice>)`, the slice `static` for a file without a
time axis, `<d> kyr BP` or `<t> <unit>` for the static problem's date,
and through time `each node's date`, `the run's clock` or `time origin
<t> <unit>`. `<model>` is `1-D`, `1-D (the viscosity file is radial)`,
`1-D (the moduli file is radial)`, `1-D (the viscosity and moduli files
are radial)`, `3-D (lateral viscosity)`, `3-D (lateral moduli)` or `3-D
(lateral viscosity, lateral moduli)`. The example configs
`examples/configs/sea_level_fingerprint.json`,
`examples/configs/sea_level_fingerprint_3d.json`,
`examples/configs/sea_level_static_migrating.json`,
`examples/configs/sea_level_static_coseismic.json`,
`examples/configs/sea_level_3d_calibrated.json`,
`examples/configs/sea_level_postseismic.json`,
`examples/configs/no_gravity_postseismic.json`,
`examples/configs/no_gravity_load.json`,
`examples/configs/sea_level_no_ocean.json`,
`examples/configs/no_ocean.json` and
`examples/configs/no_ocean_viscoelastic.json` print

```
sl3d solve: 1-D, elastic, static, shorelines frozen, rotation full; ice ../data/synthetic_earth.nc:ice_thickness (0 kyr BP), topography ../data/synthetic_earth.nc:topography (0 kyr BP); ice change cap at latitude 72, longitude 318, width 8 deg, fraction -0.1 of the state's ice; wrote out/sea_level_fingerprint/sea_level_fingerprint.nc in 0.35 s
sl3d solve: 3-D (lateral moduli), elastic, static, shorelines frozen, rotation full; ice ../data/synthetic_earth.nc:ice_thickness (0 kyr BP), topography ../data/synthetic_earth.nc:topography (0 kyr BP); ice change cap at latitude 72, longitude 318, width 8 deg, fraction -0.1 of the state's ice; wrote out/sea_level_fingerprint_3d/sea_level_fingerprint_3d.nc (6 fixed-point iterations, residual 9.88e-10) in 3.24 s
sl3d solve: 1-D, elastic, static, shorelines migrating, rotation full; ice ../data/synthetic_earth.nc:ice_thickness (21 kyr BP), topography ../data/synthetic_earth.nc:topography (21 kyr BP); ice change thickness ../data/synthetic_earth.nc:ice_thickness, from the state's date to 0 kyr BP; wrote out/sea_level_static_migrating/sea_level_static_migrating.nc in 0.36 s
sl3d solve: 1-D, elastic, static, shorelines frozen, rotation full; ice ../data/synthetic_earth.nc:ice_thickness (0 kyr BP), topography ../data/synthetic_earth.nc:topography (0 kyr BP); seismic (1 source); wrote out/sea_level_static_coseismic/sea_level_static_coseismic.nc in 2.53 s
sl3d solve: 3-D (lateral viscosity), viscoelastic, 43 nodes 21 to 0 kyr BP, shorelines migrating, initial sea level calibrated, rotation full, moments hydrostatic; ice ../data/synthetic_earth.nc:ice_thickness (the window's end, 0 kyr BP), topography ../data/synthetic_earth.nc:topography (the window's end, 0 kyr BP); ice change thickness ../data/synthetic_earth.nc:ice_thickness, from the state's instant to each node's date; wrote out/sea_level_3d_calibrated/sea_level_3d_calibrated.nc (352 fixed-point iterations over 42 steps) in 13.98 s
sl3d solve: 1-D, viscoelastic, 23 nodes 0 to 300 yr on the run's clock, shorelines frozen, rotation full, moments hydrostatic; ice ../data/synthetic_earth.nc:ice_thickness (0 kyr BP), topography ../data/synthetic_earth.nc:topography (0 kyr BP); seismic (1 source); wrote out/sea_level_postseismic/sea_level_postseismic.nc in 45.84 s
sl3d solve: 1-D, viscoelastic, 23 nodes 0 to 300 yr on the run's clock, no ocean, no gravity, rotation none; seismic (1 source); wrote out/no_gravity_postseismic/no_gravity_postseismic.nc and 2 exchange tables in 35.05 s
sl3d solve: 1-D, viscoelastic, 49 nodes 0 to 12 kyr on the run's clock, no ocean, no gravity, rotation none; load (2 terms); wrote out/no_gravity_load/no_gravity_load.nc and 3 exchange tables in 0.67 s
sl3d solve: 1-D, viscoelastic, 49 nodes 12 to 0 kyr BP, no ocean, rotation none; load (1 term); wrote out/sea_level_no_ocean/sea_level_no_ocean.nc in 1.06 s
sl3d solve: 3-D (lateral moduli), elastic, static, no ocean, rotation none; load (1 term) + tide (1 coefficient); wrote out/no_ocean/no_ocean.nc and out/no_ocean/no_ocean.txt and out/no_ocean/no_ocean_volume.nc (8 fixed-point iterations, residual 4.51e-11) in 1.45 s
sl3d solve: 3-D (lateral viscosity, lateral moduli), viscoelastic, 25 nodes 0 to 12 kyr on the run's clock, no ocean, rotation none; load (1 term) + tide (1 coefficient); wrote out/no_ocean_viscoelastic/no_ocean_viscoelastic.nc and 4 exchange tables (302 fixed-point iterations over 24 steps) in 4.39 s
```

### Examples

```json
{
  "problem": "static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 128},
  "forcing": {
    "ice_change": [{"thickness": {"file": "ice7g.nc", "date_kyr_bp": 0}}]
  },
  "ocean": {"date_kyr_bp": 26, "shorelines": "migrating",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"},
            "land": {"file": "ice7g.nc"}},
  "output": {"surface": "deglaciation.nc"}
}
```

The elastic response to the whole deglaciation on the state at the last
glacial maximum, its shorelines migrating: the state is ICE-7G at 26 ka
(its ice, its topography and the land it holds), the change is its ice
thickness at the present less its thickness at 26 ka, and so the ice at
the solve is ICE-7G's present ice. The shorelines follow flotation of
that ice on the sea level the solve reaches: the ocean floods where the
ice that is gone was grounded below sea level, and `deglaciation.nc`
holds the ocean function the solve reaches (`ocean_function_final`)
beside the state's. On the examples' PREM-like deck (`prem8`) at lmax 128 the ocean
covers 67.4% of the surface at 26 ka and 72.3% at the solve, and rises
by 120 m on average over the 26 ka ocean, in half a second on four
threads. Without `shorelines` (frozen) the same config is the linear
elastic fingerprint of the change on the 26 ka shorelines; the same
state takes a change from a file of one's own as
`"forcing": {"ice_change": [{"change": {"file": "my_melt.nc"}}]}`.

```json
{
  "problem": "static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 128},
  "forcing": {
    "seismic": [{"fault": {"latitude": 38.3, "longitude": 142.4, "depth_km": 25,
                           "strike_deg": 200, "dip_deg": 12, "rake_deg": 90,
                           "length_km": 400, "width_km": 150, "moment_nm": 4.0e22,
                           "patches_along_strike": 8, "patches_down_dip": 3}}]
  },
  "ocean": {"date_kyr_bp": 0,
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"},
            "land": {"file": "ice7g.nc"}},
  "output": {"surface": "coseismic.nc"}
}
```

The co-seismic sea level change of a thrust off north-east Japan (a 400
by 150 km plane of scalar moment 4e22 N m, laid out as 8 by 3 point
sources) on the present day's state, ICE-7G at 0 ka: no ice change, the
fault alone. The fault lifts the sea floor above it and lets the coast
behind it sink, the body's potential changes with it, and the ocean's
water, conserved, settles on the deformed body under the rotation's
feedback: `coseismic.nc` holds the sea level change, the displacement,
the tangential and toroidal potentials (the horizontal motion) and the
potential. On the same deck at lmax 128 the sea floor rises
by up to 1.5 m and the sea level over it falls by as much, in about 4 s
and 2.2 GB on four threads.

```json
{
  "problem": "static",
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "moduli": {"file": "s40rts_moduli.nc"}},
  "discretisation": {"lmax": 64},
  "forcing": {
    "ice_change": [{"cap_fraction": {"latitude": 72, "longitude": 318,
                                     "width_deg": 8, "fraction": -0.1}}]
  },
  "ocean": {"date_kyr_bp": 0,
            "sea_level": {"file": "present_day.nc", "variable": "rsl0"},
            "ice": {"file": "ice7g.nc"}},
  "rotation": "full",
  "solver": {"step": "conjugate_gradients", "step_rtol": 1e-9},
  "output": {"surface": "fingerprint_3d.nc"}
}
```

The fingerprint of melting a tenth of Greenland's ice on the lateral
moduli of `s40rts_moduli.nc` (a planetmodel model file of `mu` and
`kappa`), its zero-length step closed by the sea level closure nested
over conjugate-gradient solves, with the radial model's fingerprint
beside it in `fingerprint_3d.nc`. Its sea level is the static variable
`rsl0` of `present_day.nc`, used as it stands, the same at every date;
the state's date reads ICE-7G's ice at the present.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 64},
  "times": {"start_kyr_bp": 26.0, "nodes": 53},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "output": {"surface": "histories.nc"}
}
```

A radial run through time from ICE-7G's state at 26 kyr BP under its ice
history. A thickness-only history under migrating shorelines, its
initial sea level calibrated against the present-day sea level,
checkpointed every four nodes:

```json
{
  "problem": "quasi_static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 64},
  "times": {"start_kyr_bp": 21.0, "end_kyr_bp": 0.0, "nodes": 43},
  "forcing": {
    "ice_change": [{"thickness": {"file": "my_ice_history.nc",
                                  "variable": "thk"}}]
  },
  "ocean": {"at": "end", "shorelines": "migrating",
            "sea_level": {"file": "present_day.nc", "variable": "rsl0"},
            "ice": {"file": "my_ice_history.nc", "variable": "thk"},
            "calibration": {"method": "fixed_point", "iterations": 12}},
  "checkpoint": {"directory": "checkpoints", "every": 4},
  "output": {"surface": "histories.nc"}
}
```

Here the state is the present day's: with `"at": "end"` its files are
read at the window's end alone, so the sea level's file need only hold
the present day (a file without a time axis is the present day's), and
the ice at the window's start is the history's own.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "viscosity": {"file": "s40rts_viscosity.nc",
                          "reference": "geometric_mean"}},
  "discretisation": {"lmax": 128},
  "times": {"start_kyr_bp": 26.0, "end_kyr_bp": 0.0, "nodes": 105},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "rotation": "full",
  "moments": "hydrostatic",
  "solver": {"scheme": "exponential_trapezoid", "rtol": 1e-9,
             "step": "fixed_point", "step_rtol": 1e-8,
             "step_max_iterations": 80, "transform_degrees": [32, 64, 96]},
  "checkpoint": {"directory": "checkpoints", "every": 4,
                 "max_walltime_min": 1380},
  "restart": true,
  "output": {"surface": "out/histories_3d.nc"}
}
```

The first three are static problems, radial, radial and 3D, and the
next two radial runs through time; the last is 3D, under the lateral
viscosity of `s40rts_viscosity.nc`.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "../data/prem8.postseismic.deck", "format": "maxwell"},
  "discretisation": {"lmax": 96},
  "times": {"unit": "yr", "at": [0.0, 0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5, 10.0, 13.0, 17.0, 22.0, 30.0,
                                 40.0, 55.0, 75.0, 100.0, 140.0, 200.0, 300.0]},
  "forcing": {
    "seismic": [{"fault": {"latitude": -22.0, "longitude": 285.5, "depth_km": 25.0,
                           "strike_deg": 0.0, "dip_deg": 18.0, "rake_deg": 90.0,
                           "length_km": 600.0, "width_km": 150.0,
                           "moment_nm": 2.0e22,
                           "patches_along_strike": 12, "patches_down_dip": 3},
                 "onset": 0.0}]
  },
  "ocean": {"at": "start", "date_kyr_bp": 0.0,
            "topography": {"file": "../data/synthetic_earth.nc"},
            "ice": {"file": "../data/synthetic_earth.nc"}},
  "rotation": "full",
  "moments": "hydrostatic",
  "output": {"surface": "out/sea_level_postseismic/sea_level_postseismic.nc",
             "fields": ["horizontal_displacement"],
             "smoothing": {"kind": "gaussian", "width_deg": 2.0}}
}
```

A run counting forwards in years, `examples/configs/sea_level_postseismic.json`:
the thrust of `sea_level_static_coseismic.json` off the synthetic South
America, acting from the run's start (`onset` 0), and the ocean's answer
to the mantle's relaxation over three centuries at uneven nodes, dense
early and each step after the first year a quarter to a third of the
time elapsed, on the post-seismic Earth `prem8.postseismic.deck` (an
elastic lid 50 km thick over Maxwell layers of 1e19 Pa s to 217 km
depth, 1e20 Pa s to 667 km and 1e22 Pa s below, their Maxwell times
4.7 to 4.8, 26 to 43 and 1080 to 2050 years). The state is the
synthetic Earth's present day, a snapshot of its files at `date_kyr_bp`
0; the time axis of the output is the run's clock in seconds, its nodes
in years in the metadata (`times`, `time_unit` `yr`). The first node is
the co-seismic answer, the static example's to 1.6 mm (the deck's
viscous layers isotropic where the elastic deck's are transversely
isotropic, and the hydrostatic moments against the standard ones).
Averaged over a Gaussian cap of 2° over the fault (the field's point
values are not converged in lmax for a fault this shallow, its 2°
means are), the uplift of 0.150 m at the event grows by 6 mm in the
first year and to 0.319 m at a century as the asthenosphere and the
upper mantle relax, and falls back to 0.233 m by 300 years. The file
holds those means at every node, each field of the response smoothed
over 2° (`output.smoothing`), and the horizontal displacement's east and
north components, raw and smoothed (`output.fields`). About 46 s and
4.0 GB on four threads.

Without an ocean and without rotational feedback, the response of the
solid body:

```json
{
  "problem": "static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 128},
  "forcing": {
    "load": [{"disk": {"latitude": 64, "longitude": 20, "radius_deg": 6,
                       "amplitude_kg_m2": 1.0e5}}]
  },
  "rotation": "none",
  "output": {"surface": "out/disk.nc"}
}
```

```json
{
  "problem": "static",
  "model":   {"deck": "prem.deck", "format": "elastic",
              "moduli": {"file": "s40rts_moduli.nc", "reference": "lateral_mean"}},
  "discretisation": {"lmax": 64},
  "forcing": {
    "load": [
      {"cap":  {"latitude": 60, "longitude": -85, "width_deg": 10, "amplitude_kg_m2": 1.8e6}},
      {"disk": {"latitude": -75, "longitude": 0, "radius_deg": 8, "amplitude_kg_m2": -9.0e5}},
      {"field": {"file": "load.nc", "variable": "sigma"}},
      {"field": {"file": "load_history.nc", "date": 10, "unit": "kyr"}},
      {"coefficients": {"file": "load_lm.txt"}}
    ],
    "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0, "imaginary_m2_s2": 0.0}]
  },
  "rotation": "none",
  "solver":  {"step_rtol": 1e-10, "step_max_iterations": 200, "transform_degrees": [16, 32], "cache_mb": 0},
  "output":  {"surface": "out/response.nc", "exchange": "out/response.txt", "volume": "out/response_volume.nc"}
}
```

The first is a 1D run (the deck's radial model); the second solves the
3D problem under the lateral moduli of `s40rts_moduli.nc`, its loads
including the variable `sigma` of the static file `load.nc` and the
variable `load` of `load_history.nc`, a file with a time axis, at 10 kyr
on that axis.

```json
{
  "problem": "quasi_static",
  "model": {
    "deck": "prem_maxwell.deck", "format": "maxwell",
    "viscosity": {"file": "visc3d.nc", "reference": "geometric_mean"},
    "moduli":    {"file": "s40rts_moduli.nc"}
  },
  "discretisation": {"lmax": 64},
  "times": {"unit": "kyr", "start": 0, "end": 20, "nodes": 41},
  "forcing": {
    "load": [
      {"cap": {"latitude": 60, "longitude": -85, "width_deg": 10, "amplitude_kg_m2": 1.8e6},
       "history": [[0, 0.0], [10, 1.0], [10.5, 0.0]]},
      {"field": {"file": "load_history.nc"}}
    ],
    "tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0,
              "history": [[0, 1.0]]}]
  },
  "rotation": "none",
  "solver": {"scheme": "exponential_trapezoid", "step": "fixed_point",
             "step_rtol": 1e-10, "step_max_iterations": 200,
             "transform_degrees": [16, 32], "cache_mb": 0},
  "output": {"surface": "out/relax.nc", "exchange": "out/relax_{node}.txt"}
}
```

A viscoelastic 3D run under a lateral viscosity and lateral moduli: a cap
loaded linearly over 10 kyr and removed over the next 0.5 kyr, a load
history read from a surface file on the run's clock (a file spanning the
run's 20 kyr: one that ends earlier is refused at the nodes past its end
unless the input holds its last field, `"outside_span": "hold"`), and a
constant tide, over 20 kyr in half-kyr steps, with a table per node
(`out/relax_00.txt` to `out/relax_40.txt`). Without `viscosity` and
`moduli` (and the solver members of the 3D step) the same config runs
the deck's radial Maxwell model in 1D.

## kernel

Adjoint sensitivity kernels of configured measurements of the problems
`solve` solves. A kernel config is a `solve` config with a `kernels`
block (The blocks, above): the forward run is `solve`'s run of the same
blocks, built by the same pieces, the static problem (a run of one node
on the elastic Earth) or the quasi-static one, with an ocean or without
one, under any mix of an ice change, surface loads, a tide and seismic
sources, with frozen or migrating shorelines, its times dated or on its
own clock, evenly spaced or listed, and 1D or 3D as the `model` block
makes it (the static problem on lateral moduli in 3D; through time a
lateral viscosity, lateral moduli or both). Its states are recorded (in
memory, or in the trajectory store of `trajectory.directory`, from which
a later run sweeps without the forward run: The trajectory block, above),
and then, for each measurement, one reverse (adjoint) sweep of the library's
derivative classes gives the kernels the block names, by the continuous
adjoint or by the exact discrete transpose of the time-stepped map
(docs/adjoints.md). In the static problem the two methods agree to
rounding, there being no step to commute; through time they differ by
the commute error, which falls as the steps are refined.

With the state at the window's start the initial sea level is the
state's; with it at the end (`"at": "end"`) it is calibrated first, as
`solve` calibrates it, and the kernels hold the calibrated initial sea
level fixed.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| the shared blocks | | | as for `solve` (The blocks, above); `output`, `checkpoint` and `restart` checked as `solve` checks them and not used |
| `kernels.method` | word | `"continuous"` | `"continuous"` or `"discrete"` (the exact transpose of the time-stepped map) |
| `kernels.parameters` | array of words | required | the kernels written, at least one, each where the run gives it (below) |
| `kernels.lateral` | boolean | false on a radial model, true on one with lateral structure | the kernels in the viscosity and the moduli as fields over the Earth, whatever the background (Lateral kernels, below) |
| `kernels.measurements` | array | required | the measurements, at least one (below) |
| `kernels.output` | path | required | the directory the kernel files go into |

The members of the block given at the top level (`adjoint`,
`measurements`, `parameters`, `method`) are refused, naming the block's
member, and so is a top-level `output` that is a path: the top-level
`output` is `solve`'s block. The forward run's recording is the
top-level `trajectory` block's (The trajectory block, above), which
`solve` shares: `kernels.record_stride` and `kernels.trajectory` are
refused, naming `trajectory.record_stride` and `trajectory.directory`. A
run without gravity (`"gravity": "none"`) has no kernels: `kernel` refuses
it at `gravity`, and `solve` a `kernels` block beside it.

### The kernels

Each word of `kernels.parameters` names a set of kernels, written per
measurement into `kernels.output` as files whose names begin with the
measurement's name. The member is required, and its absence is refused,
the error listing the words the run takes; a word the run cannot give is
refused when the config is read, saying why. The kernels in the sources
are `"moment"`'s: the forcing's word for them, `"seismic"`, is not a
kernel's, and is refused here, the error naming `"moment"`.

| Word | The kernels | Taken | Files |
| --- | --- | --- | --- |
| `"ice"` | in the ice history: the ice thickness at each node, its rates, the state's ice and the initial sea level | with an ocean | `<name>_ice_kernels.nc` |
| `"load"` | in a surface load at each node: in the water budget, outside it, and the height of a layer on the solid surface | always; without an ocean the one load kernel | `<name>_load_kernels.nc` |
| `"tide"` | in the coefficients of a tidal potential at each node | always: with an ocean or without, under frozen or migrating shorelines (about a migrating run, its own shorelines held) | `<name>_tide_kernels.txt` |
| `"moment"` | in the point moment tensors of the run's seismic sources at each node, at every point source, every patch of a fault and every quadrature point of a fault file | with `forcing.seismic` | `<name>_moment_kernels.txt` |
| `"viscosity"` | in the viscosity | through time | 1D `<name>_viscosity_kernel.txt` (and `<name>_viscosity_kernels.nc` with `kernels.lateral`), 3D `<name>_viscosity_kernels.nc` |
| `"moduli"` | in the bulk modulus kappa and the unrelaxed shear modulus mu | always | 1D `<name>_moduli_kernel.txt` (and `<name>_moduli_kernels.nc` with `kernels.lateral`), 3D `<name>_moduli_kernels.nc` |

The kernels in the surface fields (ice, load) are densities per square
metre of the surface of the run's sphere (its radius, `radius_m`, in each
file's metadata), and dJ = integral of K delta dS, summed over the nodes
(intervals, instants) for the fields that have them. The kernels in the
Earth's parameters (viscosity, moduli) are densities per unit volume
(docs/adjoints.md, "The kernels' convention"), and dJ = integral of K
delta dV. Every kernel is SI: the measurement's unit (`measurement_unit`,
or the `J in` line of a table) per unit of the perturbation and per
square or cubic metre. Each file states the measurement, its type and
unit, and the pairing, with the points and weights the pairing needs
where the run has them.

**The ice and the state.** The ice at a node is the state's ice plus the
ice change there (The state and the ice change, above). With the state
at the window's start, and in the static problem, the state is held:
`thickness_dual` at every node, the first included, is the derivative
with respect to the ice there with the state's ice held, which is the
derivative with respect to the ice change at that node; an ice change
present at the first node is taken (the library's thickness view moves
the background's ice with the first node's, and its first node less its
baseline is this one; docs/adjoints.md §20). With the state at the
window's end, the run starts from the ice at the window's start, and the
first node's `thickness_dual` moves that start with it, the ice at the
other nodes held. The surface file's `ice_views` says which
(`state_held` or `run_start_moves`). Every field holds the initial sea
level, and `initial_sea_level_dual` holds the ice. The fields:

| Field | Support | Meaning |
| --- | --- | --- |
| `thickness_dual` | node (static in the static problem) | dJ per metre of grounded ice thickness at each node, as above |
| `rate_dual` | interval | through time: the quelled view, dJ = sum_k integral of K_k (I_{k+1} - I_k) dS (= sum_k h_k integral of K_k (dI/dt)_k dS), the first node's ice held |
| `instantaneous_dual` | instant | through time: the delta amplitude at each measurement instant of the thickness kernel's distributional view, per metre of ice |
| `thickness_baseline_dual` | static | dJ per metre of the state's ice (with the state at the end, of the ice the run starts from), the ice at every node held; with the state at the end no config moves it alone (the run starts from the first node's ice, whose `thickness_dual` folds it in) |
| `initial_sea_level_dual` | static | dJ per metre of the initial sea level, the ice held; zero under frozen shorelines |

With the state held, the ice change at the first node with the later
nodes' ice held pairs with `thickness_dual` there, and a change of the
state's ice with the ice change held (the ice at every node moving with
it) pairs with `thickness_baseline_dual` plus the sum of `thickness_dual`
over the nodes. With the state at the end, such a change moves the ice
at every node and the run's start alike, and so leaves the run's loads
(the ice's changes from its start) as they were: it pairs with the sum
of `thickness_dual` over the nodes, which is zero, and acts through the
calibration's target alone (below).

A calibrated run's initial sea level (`"at": "end"`) is not an input: it
follows the run's ice and its target, the state's sea level at the end,
and solve re-calibrates it for every run. A change of a calibrated run's
ice therefore changes J by the thickness kernels paired with the change
of the ice plus `initial_sea_level_dual` paired with the change of the
calibrated initial sea level (solve's `initial_sea_level`), and a change
of its target by the second term alone. Under frozen shorelines that
dual is zero, and the thickness kernels alone are the derivative of the
re-calibrated run; under migrating ones it lives on the cells whose
shorelines moved, and the second term is part of the derivative.

**The load.** `<name>_load_kernels.nc` holds, at each node (static
fields in the static problem):

| Field | Meaning |
| --- | --- |
| `load_dual` | dJ per kg m^-2 of a surface load in the water budget (without an ocean, of any surface load) |
| `load_outside_water_budget_dual` | with an ocean: dJ per kg m^-2 of a surface load outside the water budget |
| `solid_surface_change_dual` | with an ocean: dJ per metre of the height of a layer laid on the solid surface, its mass a load beside it |

A layer of density rho whose mass is outside the water budget has the
kernel rho `load_outside_water_budget_dual` + `solid_surface_change_dual`
in its height; one in the budget, rho `load_dual` +
`solid_surface_change_dual`.

**The tide.** `<name>_tide_kernels.txt` is a table of rows `node <time>
l m dJ_damplitude dJ_dimaginary` for every degree 2 to `lmax` and order
0 to l at every node: the derivatives of J with respect to the real and
imaginary parts of the tidal potential's coefficient there, the members
`amplitude_m2_s2` and `imaginary_m2_s2` of a `forcing.tide` term (the
orthonormal complex coefficient, Condon-Shortley phase, m >= 0, of a
real potential), in J's unit per m^2 s^-2. A term's coefficient at a
node is its value times its history's factor there, so the derivative
with respect to a term's member is the sum over the nodes of the factor
times the row's entry. The imaginary part is zero at m = 0, and so is its
column. A tide has no degree 0 or 1 (Tide coefficients, above), so the
table has no rows there. `<time>` is the node's date (`date_kyr_bp`) in a
dated run, its time on the run's clock (`time_<unit>`) in one counting
forwards, and 0 (`instant`) in the static problem; the moment table's
likewise.

**The seismic sources.** `<name>_moment_kernels.txt` lists the run's
source points under its header (`# point p term patch latitude_deg
longitude_deg depth_m`, a point source one patch, a fault's patches along
strike then down dip, a fault file's quadrature points patch by patch, u
slowest, numbered in the `patch` column), then rows `node <time> point
dJ_dMrr dJ_dMtt dJ_dMpp dJ_dMrt dJ_dMrp dJ_dMtp`: the derivatives of J
with respect to the members of the point's moment tensor at the node, in
the basis of `moment_tensor_nm` (r up, theta south, phi east), an
off-diagonal member setting the symmetric pair, in J's unit per N m. A
term's moment at a node is its tensor times its factor there (zero before
its onset; a fault file's on a time axis, its slices as its rule reads
them), so the derivative with respect to a point source's member is the
sum over the nodes of the factor times the entry. Kernels in a fault
file's slip are not written: they follow from these, the moment at a
quadrature point being w C : (s n) (Seismic sources, above), so that the
derivative of J with respect to the slip there is w (C : K) n, K the
point's moment kernel (with its off-diagonal entries halved, each setting
a pair), gathered onto the slip's own grid through the rule that reads
the slip at the point.

**The viscosity.** The kernels are a branch's, each in the branch's own
viscosity, its modulus times its relaxation time (mu_b tau_b; a Maxwell
body's one branch, the viscosity itself), the branch's modulus held. A
Maxwell deck has one branch; a Burgers body (a burgers deck's layer, a
region of `model.burgers`) two, `K_branch_0` its slower term and
`K_branch_1` its faster, the Maxwell layers' kernel in `K_branch_0`; a
prony deck's are its branches in order. The kernels with respect to the
parameters a burgers or prony deck states (the steady viscosity, the
Kelvin element's viscosity and modulus fraction; the strengths and the
times) are not built: they are the branches' kernels through the
derivatives of the branches in those parameters, which no command forms.
Every branch's relaxation time raised alike (a Burgers body's two
viscosities scaled together, its strengths held) pairs with the sum over
the branches, as `uniform_log_viscosity_pairing` (below) gives it. In
1D, `<name>_viscosity_kernel.txt`: columns `r_m`, `K_branch_<b>` per
branch and `W_m3`, the kernel in eta_b in SI, dJ = integral of K
delta eta_b dV = sum over the rows of W_m3 K delta eta_b, eta_b in Pa s; the rows are the radial mesh's nodes, element by element (a
shared element edge twice, once per element), `W_m3` the node's volume
weight (4 pi r^2 times its radial quadrature weight). In 3D,
`<name>_viscosity_kernels.nc`: the log-viscosity kernels as a planetmodel
model file, a field `K_ln_eta_<b>` per branch, dJ = integral of K
delta(ln eta) dV, written on the angular grid of the run's 3D coupling at
the memory nodes, one part per contiguous run of elements, with the
model's density alongside. Its metadata holds two pairings, in J's unit:
`lateral_log_viscosity_pairing`, the kernels paired with the run's own
lateral log-viscosity perturbation (dJ per unit scale of it, as the
operator samples it), and `uniform_log_viscosity_pairing`, the kernels
paired with one (dJ per unit of ln eta raised alike wherever the
coupling carries memory nodes).

**The moduli.** The kernels in kappa and in the unrelaxed shear modulus
mu, with respect to their absolute perturbations in Pa (the branch
moduli fixed fractions of the unrelaxed one, the relaxation times held;
a relative perturbation delta ln mu pairs as K_mu mu). In 1D,
`<name>_moduli_kernel.txt`: rows at the operator's memory points, element
by element, columns `r_m W_m3 kappa_Pa mu_Pa K_kappa K_mu`, dJ = sum over
the rows of W_m3 (K_kappa delta kappa + K_mu delta mu), with the model's
moduli at each point. In 3D, `<name>_moduli_kernels.nc`: fields
`K_kappa` and `K_mu` as the viscosity's, at the coupling's memory nodes,
which are the viscous elements and those where the lateral moduli are
not zero (in the static problem, these alone): a perturbation elsewhere
is not covered. Its metadata `lateral_moduli_pairing` is the kernels
paired with the run's own lateral moduli perturbation, dJ per unit scale
of it.

**Lateral kernels.** The kernels in lateral structure — the
log-viscosity and moduli kernels as fields — are the starting point of a
3D inversion, and about a radial Earth they need no 3D run: the forward
and adjoint problems separate by degree, and the radial sweeps expand
their fields on the angular grids and multiply them pointwise
(docs/adjoints.md §21). `"lateral": true` on a radial model writes
`<name>_viscosity_kernels.nc` and `<name>_moduli_kernels.nc` in the 3D
form above (the words of `kernels.parameters` choose which) beside the
radial `.txt` profiles, which are the fields' means over the sphere and
are, line for line, those of the run without the member. They are the
kernels a 3D run of the same model with no lateral structure gives, by
the same method. Their nodes are the memory points of the viscous
elements when the viscosity's kernels alone are written, and of every
solid element above the one holding the centre when the moduli's are
(the viscosity kernel is then zero in the elastic elements); the files'
pairings with the run's own lateral structure are zero, it having none,
and `uniform_log_viscosity_pairing` is the radial kernel's pairing with a
uniform change of ln eta over the viscous mantle. The grids' members of
the `solver` block, `transform_degrees` (the transforms cut by depth as a
3D run's are) and `cache_mb`, are taken by such a run, through time or
in the static problem (whose fields are the moduli's, on the band's grid
at every node), and refused by a radial run otherwise; `solve` takes them
on the same config and leaves them unused. They are the sweep's alone, out
of the run's hash, so a run recorded into the trajectory store once
serves the lateral kernels at any `transform_degrees`. On a model with lateral
structure the kernels are fields whatever the member says: `"lateral":
false` there is refused, naming the model's structure. `"lateral": true`
naming neither `"viscosity"` nor `"moduli"` is refused. The files of a
run swept from its trajectory store are those of the run in memory, to
the bit.

**Over MPI ranks.** A 1D run is the first rank's alone, its lateral
kernels too. A 3D run is
distributed over radial segments, its 3D kernels gathered to rank 0 and
written over the whole mesh, except what the sweeps form on one process,
refused on several ranks naming the member: the kernels about a run
under seismic sources (`forcing.seismic`), and the tide kernels of the
continuous sweep (`"tide"` with `"method": "continuous"`; the discrete
sweep's are distributed).

### Measurements

Each element of `kernels.measurements` is an object with a `type` and a
`name`. The `name` (by default `measurement_<k>`, with k the element's
position from 0) is the stem of the measurement's files, so names must be
distinct plain file names: letters, digits, `_`, `-` and `.`, not
starting with `.`. A measurement at one instant reads the outputs at a
node of the run: in a dated run at `date_kyr_bp` (default 0, the
present), in a run counting forwards at `time` on the run's clock in its
unit (default the window's last node), each located on the node list (to
1e-9 of the window's span); in the static problem at its one instant,
which takes neither member. The member of the other form is refused,
naming this form's. Every measurement is held to the run when the config
is read, before anything runs: an instant that is not a node, a
coefficient that does not exist at `lmax`, a decay window holding fewer
than four nodes, or a measurement of the sea level without an ocean is
refused then, naming the member.

| `type` | Members | Value (SI) |
| --- | --- | --- |
| `relative_sea_level_cap` | `latitude`, `longitude`, `width_deg`, the instant | the mean sea-level change over the Gaussian cap (the field smoothed over it, read at its centre: `output.smoothing`'s), m |
| `relative_sea_level_point` | `latitude`, `longitude`, the instant | the sea-level change at the point, m |
| `displacement_cap` | `latitude`, `longitude`, `width_deg`, the instant | the mean radial displacement over the cap, m |
| `displacement_point` | `latitude`, `longitude`, the instant | the radial displacement at the point, m |
| `displacement_vector_cap` | `latitude`, `longitude`, `width_deg`, `direction`, the instant | the displacement smoothed over the cap as a vector field (`output.smoothing`'s), read at its centre along the direction, m |
| `displacement_vector_point` | `latitude`, `longitude`, `direction`, the instant | the displacement at the point along the direction, m |
| `potential_cap` | `latitude`, `longitude`, `width_deg`, the instant | the mean potential perturbation over the cap, m^2 s^-2 |
| `potential_point` | `latitude`, `longitude`, the instant | the potential perturbation at the point, m^2 s^-2 |
| `stokes_coefficient` | `degree`, `order`, the instant | a real orthonormal coefficient of the potential perturbation, m^2 s^-2 |
| `rotation` | `component`, the instant | a component of the angular velocity perturbation, rad s^-1 |
| `decay_time` | `latitude`, `longitude`, `width_deg`, the window, `value_uncertainty_m` | the relaxation time of a site's sea-level series, s |

- The instant is `date_kyr_bp` or `time` as above. The sea-level
  measurements and `decay_time` need an ocean.
- `latitude`, `longitude` (degrees) and `width_deg` (degrees, > 0) are
  required where listed.
- A cap is Gaussian in great-circle distance d from its centre, weighted
  exp(-d^2 / (2 w^2)) with w = `width_deg` and normalised to unit area
  integral, so it reads a weighted mean: the mean `solve`'s
  `output.smoothing` of that width writes about every point (solve,
  Optional and smoothed fields), exactly at every width, the dual being
  the cap's kernel on the band `lmax` (its degree weights synthesised
  about the centre). A point value is the band-limited value at `lmax`
  (the grid's reproducing kernel); a narrow cap is often the steadier
  datum. The band resolves a cap whose width is at least half the
  truncation wavelength, w >= 180°/lmax (its degree weights below 1% by
  `lmax`); a narrower cap is read just as exactly, but what it reads is the
  truncated field under a cap the band cuts off, and the example configs
  keep to the rule (examples/configs/README.md).
- `displacement_vector_cap`, `displacement_vector_point`: `direction` is an
  object of `east`, `north` and `up` (each optional, default 0, not all
  zero), the components along the east, north and upward unit vectors at
  the cap's centre, or at the point; it need not be a unit vector (a line
  of sight is one). A cap's displacement is smoothed as a vector, through
  its potentials U, V and W, each weighted by the cap's degree weights as
  `output.smoothing` weights them, and read at the centre: not the mean
  over the cap of the components in each point's own frame, which turn
  across the cap. The tangential displacement is
  read through the potentials V and W of u = U r + grad_1 V - r x grad_1
  W, in the frame of no net rotation of the surface, as `solve`'s
  `toroidal_potential` is.
- `stokes_coefficient`: `degree` l and `order` m are required integers,
  0 <= l <= `lmax` and |m| <= l, with m >= 0 the cosine term and m < 0 the
  sine term of |m| (the pyshtools convention); the value is the
  coefficient of the potential perturbation in the orthonormal real
  harmonics (each squared integrates to 1 over the unit sphere), so a
  GRACE-style geodetic coefficient differs from it by a normalisation
  constant alone.
- `rotation`: `component` is `"x"`, `"y"` or `"z"`; x and y drive polar
  motion, z the length of day.
- `decay_time`, a run through time's: the cap-mean sea-level change at
  every node of the window is fitted by a e^(-t / tau) + b, following
  Al-Attar & Tromp (2014, Appendix C), and the datum is tau. In a dated
  run the window is `window_start_kyr_bp` (required) to
  `window_end_kyr_bp` (default 0), with `time_uncertainty_kyr`; in a run
  counting forwards it is `window_start` (required) to `window_end`
  (default the window's last node), in the run's unit, with
  `time_uncertainty` in that unit. Both ends must be nodes, the start the
  earlier, and the window must hold at least four nodes, the fit's
  minimum. The timing uncertainty (>= 0, required) and
  `value_uncertainty_m` (> 0, required) are the one-sigma timing and
  series errors; only their ratio shapes the fit, and a zero timing
  uncertainty holds the times at the nodes. The window is held to the
  nodes when the config is read, the uncertainties only by the fit, which
  runs on the forward run's series when the measurement's turn comes
  (the measurements listed before it swept and written by then). The fit
  refuses a `value_uncertainty_m` that is not positive and a negative
  timing uncertainty; a timing uncertainty not smaller than the window's
  span (in the uncertainty's own unit) and a `value_uncertainty_m` not smaller
  than the series' range (its largest less its smallest cap mean over the
  window), which leave the fit meaningless (the message asks whether the
  unit is right); and a series that constrains no decay (a fitted rate not
  positive, or below 1e-2 of the window's inverse span). Each is the
  run's error (exit status 1), not naming the member.

### Output

In `kernels.output`:

- `measurements.txt`: under a header of `#` lines (program and version,
  creation time, config, the state's inputs as read, `ice` and
  `sea_level`, or `ocean none`, and the forcing's terms, `ice_change`,
  `load`, `tide` and `seismic`, as `solve`'s metadata give them), one row
  per measurement: `name type value_si`, the forward value in the SI unit
  of the table above; a `decay_time` row adds `tau amplitude_m offset_m`
  from the fit, tau in kyr in a dated run and in the run's unit in one
  counting forwards (the header names it).
- The kernel files of each measurement (The kernels, above). The surface
  files are on the run's Gauss–Legendre grid; through time their time
  axis is every node, in seconds relative to the present (negative in
  the past) in a dated run and on the run's own clock in one counting
  forwards, with the nodes listed (`dates_kyr_bp`, or `times` with
  `time_unit`) as `solve`'s surface file lists them, and the instants of
  `instantaneous_dual` are the measurement's own nodes.

Summary: `<what the config was taken to mean>; <the inputs as read>;
wrote <n> measurement(s) of kernels to <output> (<parameters>; <method>
adjoint)` (`; the kernels in the Earth as fields` before the
closing parenthesis with `kernels.lateral` true), the first two as in
`solve`'s summary, then, with
`trajectory.directory`, `; the forward run recorded to the trajectory
store <directory> (<size>)` or `; the forward run read back from the
trajectory store <directory> (<size>)`.

### Examples

```json
{
  "problem": "quasi_static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 32},
  "times": {"start_kyr_bp": 20.0, "nodes": 41},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "kernels": {
    "parameters": ["ice", "viscosity"],
    "measurements": [
      {"type": "relative_sea_level_point", "name": "fremantle",
       "latitude": -32.05, "longitude": 115.74}
    ],
    "output": "kernels"
  }
}
```

The ice and viscosity kernels of a sea level at Fremantle at the
present, on a radial Earth.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "earth.deck"},
  "discretisation": {"lmax": 64},
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 41},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "rotation": "full",
  "moments": "hydrostatic",
  "output": {"surface": "out/run.nc"},
  "trajectory": {"directory": "out/trajectory", "record_stride": 6},
  "kernels": {
    "method": "discrete",
    "parameters": ["ice", "viscosity"],
    "measurements": [
      {"type": "relative_sea_level_cap", "name": "barbados", "latitude": 13.1,
       "longitude": -59.6, "width_deg": 3.0, "date_kyr_bp": 6.0},
      {"type": "displacement_point", "name": "uplift_hudson",
       "latitude": 60.0, "longitude": -85.0},
      {"type": "displacement_vector_point", "name": "east_hudson",
       "latitude": 55.0, "longitude": -80.0, "direction": {"east": 1}},
      {"type": "potential_cap", "name": "geoid_canada", "latitude": 60.0,
       "longitude": -90.0, "width_deg": 10.0},
      {"type": "stokes_coefficient", "name": "c20", "degree": 2, "order": 0},
      {"type": "rotation", "name": "lod", "component": "z"},
      {"type": "decay_time", "name": "rebound", "latitude": 60.0,
       "longitude": -85.0, "width_deg": 3.0, "window_start_kyr_bp": 8.0,
       "window_end_kyr_bp": 0.0, "time_uncertainty_kyr": 0.0,
       "value_uncertainty_m": 0.5}
    ],
    "output": "out/kernels_1d"
  }
}
```

One config for both commands: `sl3d solve` writes `out/run.nc`,
recording every sixth node's state into `out/trajectory`, and leaves the
`kernels` block alone; `sl3d kernel` then writes the kernels by the
discrete adjoint from that store, without running the forward problem
(run alone, it records the store itself).

```json
{
  "problem": "quasi_static",
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "viscosity": {"file": "s40rts_viscosity.nc"}},
  "discretisation": {"lmax": 64},
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 81},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "solver": {"step_rtol": 1e-9, "step_max_iterations": 60},
  "trajectory": {"directory": "/scratch/trajectory", "record_stride": 9},
  "kernels": {
    "parameters": ["ice", "viscosity"],
    "measurements": [
      {"type": "relative_sea_level_cap", "name": "barbados", "latitude": 13.1,
       "longitude": -59.6, "width_deg": 3.0, "date_kyr_bp": 10.0}
    ],
    "output": "out/kernels_3d"
  }
}
```

A 3D run under a lateral viscosity, its recording on disk: the first
`sl3d kernel` runs the forward problem and records it, every ninth state
to `/scratch/trajectory` (each rank's to its own `rank_RRRR`), and a
second config of other measurements sweeps from the same store.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "prem8.gia.deck", "format": "maxwell"},
  "discretisation": {"lmax": 64},
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 81},
  "forcing": {"ice_change": [{"thickness": {"file": "ice7g.nc"}}]},
  "ocean": {"at": "start",
            "topography": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "rotation": "full",
  "moments": "hydrostatic",
  "solver": {"transform_degrees": [16, 32]},
  "kernels": {
    "lateral": true,
    "parameters": ["viscosity", "moduli"],
    "measurements": [
      {"type": "relative_sea_level_cap", "name": "barbados", "latitude": 13.1,
       "longitude": -59.6, "width_deg": 3.0, "date_kyr_bp": 10.0}
    ],
    "output": "out/kernels_lateral"
  }
}
```

The same measurement's kernels in the lateral structure about the radial
Earth, the starting point of a 3D inversion: the radial run and its
sweep, the fields expanded on the grids, the transforms of the deep
nodes, where the high degrees have no dofs, cut to degree 16 or 32.

```json
{
  "problem": "static",
  "model": {"deck": "prem8.elastic.deck", "format": "elastic"},
  "discretisation": {"lmax": 32},
  "forcing": {"load": [{"disk": {"latitude": 60, "longitude": 275,
                                 "radius_deg": 8, "amplitude_kg_m2": 1.0e5}}]},
  "rotation": "none",
  "kernels": {
    "parameters": ["load", "moduli"],
    "measurements": [
      {"type": "displacement_cap", "name": "uplift", "latitude": 60,
       "longitude": 275, "width_deg": 6}
    ],
    "output": "out/static_moduli"
  }
}
```

The static kernels of an uplift under a load on a body without an ocean,
in the load and in the moduli.

```json
{
  "problem": "quasi_static",
  "model": {"deck": "prem8.postseismic.deck"},
  "discretisation": {"lmax": 32},
  "times": {"unit": "yr", "at": [0, 2, 5, 10, 20, 40, 70, 100]},
  "forcing": {"seismic": [{"fault": {"latitude": -22.0, "longitude": 285.5,
      "depth_km": 25.0, "strike_deg": 0.0, "dip_deg": 18.0, "rake_deg": 90.0,
      "length_km": 600.0, "width_km": 150.0, "moment_nm": 2.0e22,
      "patches_along_strike": 6, "patches_down_dip": 2}}]},
  "rotation": "none",
  "kernels": {
    "method": "discrete",
    "parameters": ["moment", "viscosity"],
    "measurements": [
      {"type": "displacement_vector_point", "name": "gps_east",
       "latitude": -23.0, "longitude": 290.0, "direction": {"east": 1},
       "time": 20}
    ],
    "output": "out/postseismic"
  }
}
```

The moment and viscosity kernels of a GPS station's eastward
displacement 20 years after a thrust, on a body without an ocean and on
the run's own clock in years.
