# The sl3d program: configuration reference

`sl3d` runs the library's common capabilities from a JSON config, without
writing C++. Each run is one command and one config:

```
sl3d <command> <config.json>
```

| Command | What it computes | Writes |
| --- | --- | --- |
| `love-numbers` | the SI Love numbers of a radial model | a plain-text table |
| `response` | the response to a prescribed surface load and tide, no ocean: elastic, or viscoelastic through time with a `times` block; 1D or 3D as the model's moduli (and viscosity) make it | a surface file (of histories, with `times`), and optionally coefficient tables and, elastic, a model file |
| `sea-level` | the sea level equation under an ice load on a present-day state: the elastic fingerprint of an ice change, or with a `times` block GIA through time under an ice history; 1D or 3D as the model's moduli (and, through time, viscosity) make it | a surface file (of histories, with `times`) |
| `kernel1d` | adjoint sensitivity kernels of configured measurements of a `sea-level` run through time, on a radial model | a directory of kernel files |
| `kernel` | their 3D counterpart, on a laterally varying viscosity | a directory of kernel files |

## Running sl3d

- `sl3d --help` prints the usage; `sl3d --version` prints the versions of
  sl3dpp and of the planetmodel it was built with, as
  `sl3d (sl3dpp <version>, planetmodel <version>)`.
- A run prints one summary line to standard error, as
  `sl3d <command>: <summary> in <seconds> s`; the summary of each command
  is given with it below.
- Exit status: 0 when the command finished (including a `sea-level` run
  that stopped at its wall-time limit, below), 1 when it failed, with the error
  on standard error as `sl3d <command>: <message>`, and 2 for a usage
  error (a wrong number of arguments or an unknown command).
- The library runs threaded with OpenMP on every core by default; set
  `OMP_NUM_THREADS` to choose the thread count. One-thread runs are bitwise
  reproducible; threaded runs agree to round-off.

### Over MPI ranks

Built with the CMake option `SL3DPP_WITH_MPI`, `response`, `sea-level`
and `kernel` run under `mpiexec` over several ranks, the same config on
every rank:

```
OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close \
  mpiexec -n 4 -bind-to core:32 sl3d sea-level sea_level_3d.json
```

(the binding flags are MPICH's; another MPI names them otherwise). The
run is split over radius, one contiguous radial segment per rank, the cuts
balancing the transform work ([radial-partitioning.md](radial-partitioning.md) §7 and §9):
each rank holds and steps its own segment's internal variables, with its
threads inside, and every rank reads the inputs and solves the sea level
equation, the first rank's solution serving all (a `response` run has no
sea level equation: every rank builds the forcing, the first rank's
coefficients serving all). A 1D run of these commands (the `sea-level`
fingerprint on a radial Earth, a radial `sea-level` run through time, a
1D viscoelastic `response`) is the first rank's alone, as a run of one process, its
checkpoints in `checkpoint.directory` itself; the other ranks return at
once. One rank per socket (or per node) with threads inside is the
intended shape. Rank 0 writes every output and prints the summary, with
`on <n> ranks` appended. A failure on any rank ends the whole job, the
message naming the rank. `love-numbers` and `kernel1d` refuse several
ranks (exit status 2).

The results agree with a one-process run to round-off of the radially
substructured solves (about 1e-12 relative), not bitwise. Each rank's
checkpoints go to its own subdirectory, `rank_RRRR` of
`checkpoint.directory`, holding its own segment's state (rank 0's holds
the histories too), and a recorded trajectory spills to `rank_RRRR` of
`trajectory.directory` likewise. A run resumes from the newest node whose
checkpoint every rank has, and over the same number of ranks only: the
rank count is part of the checkpoints' hash, so a resumption over another
count is refused like another run's.

## Configs

A config is one JSON object, read strictly by planetmodel's JSON reader:

- A member that the command does not know is refused, so a misspelt member
  is an error rather than silently ignored. A missing required member, a
  value of the wrong type, a non-integer where an integer is wanted, and a
  word outside a member's choices are refused too. The error names the
  config file as given, the member's path within it and the fault, for
  example `run.json: forcing.ice.width_deg: is not a positive angle`; a JSON
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
- Dates in configs are in kiloyears before present (`_kyr_bp` members:
  the `times` of `sea-level` and the kernel commands, and their
  measurements' dates), a kiloyear being 1000 Julian years,
  3.15576e10 s. A date of D kyr BP is the time -D kyr: the time axes of the
  files sl3d writes are in seconds relative to the present, negative in
  the past. `response`'s `times` (`_kyr` members) count forwards instead,
  on the run's own clock, and its file's time axis is that clock in
  seconds (response, below).
- Latitudes and longitudes are in degrees, latitude north positive and
  longitude east positive; cap widths and disk radii are in degrees.
- `rotation` chooses the rotational feedback: `"none"`, `"axial"` (the
  length-of-day change alone) or `"full"` (polar wander as well); the
  default is `"full"` wherever the member is accepted.

### Earth models

`model.deck` names a deck: a table of knots, radius first, in which a
repeated radius marks a layer boundary. `model.format` says which kind
(`love-numbers` takes the two as `model` and `model_format`):

| `model.format` | Header lines | Columns after the radius |
| --- | --- | --- |
| `"maxwell"` (the default; `"gia"` names the same format) | 2 | `rho vpv vsv vph vsh eta viscosity`, planetmodel's Maxwell deck; viscosity `nan` in elastic and fluid layers |
| `"mineos"` | 3 | `rho vpv vsv qkappa qmu vph vsh eta`, or `rho vp vs qkappa qmu` for an isotropic model |
| `"elastic"` | 0 | `rho vpv vsv vph vsh eta`, the six-column elastic table of the test and example decks |

All values are SI. Blank lines and lines starting with `#` in the table are
skipped. The elastic structure always comes from the deck. A run through
time (a `times` block: `response`, `sea-level`, `kernel1d`, `kernel`)
takes the viscosity from the model file of `model.viscosity` when given,
ignoring any viscosity column, and from the deck's `viscosity` column
otherwise, which then needs a deck that has one (`"maxwell"`): each layer
holding a viscosity is a Maxwell body, and a deck with no viscosity at all
is refused. An elastic `response` and the `sea-level` fingerprint use no
viscosity.

The radial mesh is built for the run's `lmax` from the deck alone; there
are no mesh members.

### Present-day states and ice histories

The ice comes in as planetmodel surface files (`planetmodel.surface/1`):
named scalar fields on one angular grid, static or on a time axis.
pyslfp's `export_ice_history` (`from pyslfp.ice import export_ice_history`)
writes one from ICE-5G, ICE-6G or ICE-7G, holding the `ice_thickness` and
`topography` histories in metres at the model's stored dates.

- A present-day state (`forcing.state.file`) must hold the fields
  `ice_thickness` and `topography`; an ice history (`forcing.ice.file`)
  must hold `ice_thickness`. The two may be the same file. A `field` load
  term of `response` is read from a surface file the same way, at each
  time node of a viscoelastic run when the file has a time axis.
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
  one, its shorelines fixed at the state's, unless a run through time
  lets them migrate (`ocean.shorelines`).

### 3D viscosity models

`model.viscosity` names a planetmodel model file (`planetmodel.model/1`)
with a field named `viscosity`, as planetmodel's tomography conversion
writes, for a run through time of `response`, `sea-level`, `kernel1d` or
`kernel`. The viscosity is split as
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

`response`, `sea-level`, `kernel1d` and `kernel` read their configs as
blocks, JSON objects each read by one piece of code for every command that
takes it: `model` (the deck, its format, the lateral moduli and the 3D
viscosity), `times` (the time nodes of a run through time), `forcing`
(what drives the run), `ocean` (the shorelines and the initial sea
level's calibration), `solver` (the solves' tolerances, budgets and
transforms, and the time step), `checkpoint` (a run's checkpoints) and
`output` (the files written), with `lmax`, `rotation`, `moments` and
`restart` beside them. A member a command does not take is refused, and
so is a block's member given at the top level, the error naming the block
it belongs in (a top-level `state`, for one, belongs in `forcing.state`).

Which command takes which:

| Block or member | `response` | `sea-level` | `kernel1d`, `kernel` |
| --- | --- | --- | --- |
| `model.deck`, `model.format` | yes | yes | yes |
| `model.viscosity` | with `times` | with `times` | `kernel1d`: a radial one; `kernel`: required, varying laterally |
| `model.moduli` | yes | yes | no |
| `lmax` | yes | yes | yes |
| `times` | the forwards form; absent, elastic | the before-present form; absent, the fingerprint | the before-present form, required |
| `forcing` | `load` and `tide` | `state` and `ice` | `state` and `ice` (an ice history) |
| `ocean` | no | with `times` | yes |
| `rotation`, `moments` | no | yes | yes |
| `solver.rtol` | no | yes | yes |
| `solver.step_rtol`, `step_max_iterations`, `transform_degrees`, `cache_mb` | yes | in a 3D run (the fingerprint's on lateral moduli too) | `kernel` |
| `solver.scheme` | with `times` | with `times` | yes |
| `solver.step` | with `times` | in a 3D run (the fingerprint's on lateral moduli too) | `kernel` |
| `checkpoint`, `restart` | no | with `times` | no |
| `output` | a block: `surface`, `exchange`, `exchange_nodes`, `volume` | a block: `surface` | a path: the directory of the kernel files |

The kernel commands also take members of their own (kernel1d and kernel,
below). "In a 3D run" means a run on a model the `model` block makes 3D
(each command says when); a 1D run of `sea-level` (its fingerprint
included) and `kernel1d` refuse the members of the 3D step.

### The model block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model.deck` | path | required | the deck (Earth models, above) |
| `model.format` | word | `"maxwell"` | `"maxwell"` (or `"gia"`), `"mineos"` or `"elastic"` |
| `model.moduli.file` | path | absent | a planetmodel model file whose `mu` and `kappa` fields give the lateral moduli; absent, the radial model alone |
| `model.moduli.reference` | word | `"lateral_mean"` | `"lateral_mean"` (the perturbation about each field's lateral mean) or `"file_radial"` (about the file's own stored radial reference) |
| `model.viscosity.file` | path | absent | with `times` alone: a planetmodel model file of the 3D viscosity (3D viscosity models, above); absent, the deck's own viscosity column |
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

The viscosity matters only to a run through time: `model.viscosity` is
refused without `times`. With `times`, the viscosity is the model file's
when `model.viscosity` is given (its radial reference the Maxwell body,
the deck's viscosity column, if any, unused), and the deck's own column
otherwise: each layer holding a viscosity is a Maxwell body. A deck
format with no viscosity column (`"elastic"`, `"mineos"`) is then refused
at `model.format`, and a `"maxwell"` deck with no viscosity anywhere at
`model.deck`.

A model file whose fields are stored as functions of the radius alone is
radial, and leaves a run 1D; a field stored on an angular grid varies
laterally, even where its values happen not to, and makes a run through
time 3D. Each command's section says what that decides.

### The times block

The time nodes of a run through time, in one of two forms. `response`
takes the FORWARDS form, in kyr on the run's own clock:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `times.start_kyr` | number | 0 | the first node |
| `times.end_kyr` | number > `start_kyr` | required with `nodes` | the last node |
| `times.nodes` | integer >= 2 | required with `end_kyr` | the number of nodes, evenly spaced from `start_kyr` to `end_kyr`, both included |
| `times.at_kyr` | array of numbers | instead of the three above | the nodes themselves, at least two, strictly increasing; the first is the start |

Exactly one form is given: `end_kyr` and `nodes` (with `start_kyr` or
its default), or `at_kyr` alone (`start_kyr` beside it is refused). Evenly
spaced nodes take one step length, (end - start) / (nodes - 1), for every
step; listed nodes step by their differences, so uneven steps are
allowed. Listed nodes that coincide with evenly spaced ones give the same
run to the bit.

`sea-level`, `kernel1d` and `kernel` take the BEFORE-PRESENT form, a
window of dates:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `times.start_kyr_bp` | number | required | the window's start, kyr BP |
| `times.end_kyr_bp` | number | 0 | the window's end, kyr BP; must be later than the start, so smaller |
| `times.nodes` | integer >= 2 | required | the number of time nodes, both ends included, evenly spaced |

Each command refuses the other form's members, naming the form it takes.
In either form the step is not limited by the shortest Maxwell time.

### The ocean block

`sea-level`'s run through time and the kernel commands; refused without
`times`, the fingerprint solving the linear equation on the state's own
shorelines.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ocean.shorelines` | word | `"frozen"` | `"frozen"`: the start's ocean function throughout (the linear equation); `"migrating"`: the ocean function follows flotation (sea-level, The run through time) |
| `ocean.calibrate_initial_sea_level` | object | absent | calibrate the initial sea level against `forcing.state.file` at the window's end (sea-level, The run through time); its members, all optional, follow |
| `ocean.calibrate_initial_sea_level.method` | word | `"fixed_point"` | `"fixed_point"`: the classical update S0 <- S_f - xi(t_f; S0); `"gradient"`: steepest descent on the misfit with the adjoint gradient (one adjoint sweep per run); the same single step under frozen shorelines |
| `ocean.calibrate_initial_sea_level.iterations` | integer >= 1 | 10 | the most runs under migrating shorelines; running out is an error |
| `ocean.calibrate_initial_sea_level.rtol` | number > 0 | 1e-4 | convergence under migrating shorelines: the largest update below this times the largest initial sea level (1e-4 is of order a metre) |

### Rotation and moments

`sea-level` and the kernel commands; top-level members.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rotation` | word | `"full"` | `"none"`, `"axial"` or `"full"` (Units and conventions, above) |
| `moments` | word | `"input"` | `"input"`: the Earth's standard moments of inertia; `"hydrostatic"`: their difference set from the model's own hydrostatic fluid limit, which full rotational feedback needs when the standard moments sustain less oblateness than the model's tidal response |

### The solver block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `solver.rtol` | number > 0 | 1e-9 | `sea-level` and the kernel commands: the relative tolerance of the sea level solves (and, for the kernels and the gradient calibration, of the adjoint solves); a sea level solve that misses it is an error. `response` has no sea level equation and refuses it |
| `solver.step_rtol` | number > 0 | 1e-10 in `response`, 1e-9 in the others | the relative tolerance of a 3D step's closure. `sea-level` and `kernel`: each step's combined fixed point of the strain, the ocean load and the rotation (in `kernel`, the reversed sweep's steps too; under lateral moduli, the zero-length step of the start too, and the fingerprint's one zero-length step). `response`: each step's closure (the fixed point's relative change of the strain, or the conjugate gradients' relative residual) and the unrelaxed solve under lateral moduli (an elastic response's one solve, a viscoelastic run's first node) |
| `solver.step_max_iterations` | integer >= 1 | 200 in `response`, 60 in the others | its iteration budget; a step or solve that runs out of it is an error |
| `solver.transform_degrees` | integers in [2, lmax] | absent | the degrees the lateral coupling's transforms may be cut to with depth. Each memory node takes the least of these (or lmax) that covers every degree its solution carries, so nothing the degree systems keep is dropped; only the grid that forms the lateral products is coarser at depth, which resolves the lateral structure's pattern there to about that degree. Absent: every node at lmax. Below about lmax 30 every mantle node needs lmax and nothing is cut |
| `solver.cache_mb` | integer >= 0 | 0 | the last-level cache every transform grid of the run blocks its work for, MiB per cache domain (32 on a server with 32 MiB per CCD); 0 takes GSHTrans's default (8 MiB). A server with large per-domain caches may gain at large lmax; a small cache loses by overstating it. The results are bitwise independent of it: it sets only how many fields a batched transform takes at a time |
| `solver.scheme` | word | `"exponential_trapezoid"` | with `times` alone: the time step, `"exponential_trapezoid"` (second order) or `"backward_euler"` (first) |
| `solver.step` | word | `"fixed_point"` | with `times` alone, but for `sea-level`'s fingerprint on lateral moduli: how a 3D step's linear system is closed, `"fixed_point"` (the within-step iteration, Anderson-accelerated) or `"conjugate_gradients"` (preconditioned by the radial reference's solve; `Viscoelastic3D::Options::StepSolver`). In `sea-level` and `kernel` the conjugate-gradient solves are nested inside the step's combined fixed point, each one inexact, to the library's own inner tolerance (`Viscoelastic3D::Options::solve_rtol`); in `response` they are the step's closure, to `step_rtol` |

The members of the 3D step (`step`, `step_rtol`, `step_max_iterations`,
`transform_degrees`, `cache_mb`) are refused by a 1D run of `sea-level`
and by `kernel1d`, and by `sea-level`'s fingerprint on a radial Earth,
which takes `rtol` alone (without a `model.moduli` file they are refused
when the config is read, with a radial one once the file is read);
`sea-level`'s fingerprint on lateral moduli takes them all. `scheme` is
refused by every fingerprint. `response` takes them in 1D too, where its
steps are direct solves and they have no effect, and takes all but
`scheme` and `step` without `times` (its 3D elastic solve under lateral
moduli).

### The output block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output.surface` | path | required | the surface file to write |
| `output.exchange` | path | absent | `response` alone: the surface coefficients as a cross-code exchange table (response, Outputs) |
| `output.exchange_nodes` | array of integers | every node | `response` alone (response, Outputs) |
| `output.volume` | path | absent | `response` alone (response, Outputs) |

The kernel commands' `output` is a path, the directory their files go
into (kernel1d and kernel, below).

## love-numbers

The per-degree Love numbers of the deck's elastic structure, degrees 0 to
`lmax`, in pyslfp's conventions (the generalised Love numbers of Al-Attar
et al. 2024).

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | path | required | the deck |
| `model_format` | word | `"maxwell"` | `"maxwell"` (or `"gia"`), `"mineos"` or `"elastic"` |
| `lmax` | integer >= 0 | required | the largest degree |
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

## response

The planet's response to a prescribed surface load and external tidal
potential, with no ocean: no sea level equation is solved, and there is
no rotational feedback. The config is made of blocks (The blocks, above):
`model`, `times`, `forcing`, `solver` and `output`, with `lmax` beside
them.

The `times` block decides whether the response is elastic or
viscoelastic:

- without `times` it is the instantaneous elastic response;
- with `times` it is the viscoelastic response, stepped through the
  block's time nodes under the forcing at each node (The viscoelastic
  run, below).

Whether the run is 1D or 3D is decided by the `model` block alone:

- elastic: with no `moduli`, or with a moduli file whose perturbations do
  not vary laterally, the run is the radial model's, by direct per-degree
  solves; with laterally varying moduli it is the 3D static problem
  (docs/adjoints.md §13j), closed by preconditioned conjugate gradients
  on the radial systems, over MPI ranks like a 3D `sea-level` run, and
  the radial model's response to the same forcing is written beside it;
- viscoelastic: the run is 3D when the viscosity file's viscosity
  (`model.viscosity`) or the moduli file's moduli vary laterally, by the
  3D stepper (`Viscoelastic3D`) over MPI ranks like a 3D `sea-level`
  run, with the radial reference model's response written beside it;
  otherwise it is
  the radial model's, by its own stepper per degree, every order a column
  (`ViscoelasticOperator`), with no 3D machinery.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | object | required | the Earth model (The model block, above) |
| `lmax` | integer >= 1 | required | the degree of the solve and of its grid |
| `times` | object | absent | the time nodes of a viscoelastic run, the forwards form (The times block, above); absent, the elastic response |
| `forcing` | object | required | the load and the tide (below) |
| `solver` | object | absent | the 3D solves' tolerance, budget and transforms, and the time step (The solver block, above): `step_rtol`, `step_max_iterations`, `transform_degrees`, `cache_mb`, and with `times` `scheme` and `step` |
| `output` | object | required | the files to write (below) |

### The forcing block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `forcing.load` | array | absent | the load terms, summed (below) |
| `forcing.tide` | array | absent | the tide coefficients, summed (below) |

At least one load term or tide coefficient is needed. Without `times`
every term is a fixed pattern; with `times` a term may vary in time
(Forcing through time, below).

**Load terms.** Each element of `forcing.load` is an object with exactly
one member naming the term's kind, whose value holds the term's members,
and, with `times`, optionally `history_kyr` beside it. Each is a surface
mass density in kg m^-2:

| Kind | Members | Surface mass density |
| --- | --- | --- |
| `cap` | `latitude`, `longitude`, `width_deg` (> 0), `amplitude_kg_m2` | A exp(-d^2 / (2 w^2)), with d the great-circle distance from the centre, w = `width_deg` and A = `amplitude_kg_m2`; no cut-off |
| `disk` | `latitude`, `longitude`, `radius_deg` (in (0, 180]), `amplitude_kg_m2` | A where d <= alpha (alpha = `radius_deg`), zero outside |
| `field` | `file`, `name` (default `"load"`), `time_origin_kyr` (with `times`, default 0) | the field `name` of a planetmodel surface file |
| `coefficients` | `file` | the coefficients in a text table |

All members are required but `name` and `time_origin_kyr`; latitudes lie
in [-90, 90].

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
- `field` is read as present-day states and ice histories are (above):
  resampled onto the run's Gauss–Legendre grid by the field's own lateral
  rule and converted to SI by its declared dimensions (a field declaring
  none is used as it stands, so it should hold kg m^-2), then expanded by
  the grid's quadrature. A file with a time axis is refused without
  `times`: a load that varies in time needs a viscoelastic run.
- `coefficients` names a text table. Lines starting with `#` and blank
  lines are skipped; every other line is a row `l m re im`: integers
  0 <= m <= l and the real and imaginary parts of the coefficient f_lm of
  the surface mass density, in kg m^-2. The harmonics are the exchange
  table's (Outputs, below): the library's complex orthonormal harmonics
  with the Condon–Shortley phase,
  Y_lm(theta, phi) = sqrt((2l + 1) / (4 pi)) d^l_0m(theta) e^(i m phi),
  each squared modulus integrating to 1 over the unit sphere; the field is
  real, so the coefficients of m < 0 follow as
  f_(l,-m) = (-1)^m conj(f_lm) and the imaginary part at m = 0 must be
  zero. From the real orthonormal coefficients a_lm (cosine) and b_lm
  (sine) of pyshtools' `"ortho"` normalisation without the Condon–Shortley
  phase, f_l0 = a_l0 and f_lm = (-1)^m (a_lm - i b_lm) / sqrt(2) for
  m > 0. Degrees absent from the table are zero; a degree above `lmax`, a
  row given twice or a malformed row is refused, naming the line.

The load applied is the sum of the terms' coefficients. The `load` field
written is the sum of each term's field at the grid points whose
expansion is that term's coefficients: a cap's and a field's values as
sampled, a disk's and a table's synthesised from their coefficients.

**Tide coefficients.** Each element of `forcing.tide` is one coefficient
of the external potential, in the harmonics of the coefficient tables;
coefficients of the same degree and order add.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `degree` | integer in [2, lmax] | required | the degree |
| `order` | integer in [0, degree] | 0 | the order (m >= 0: the potential is real) |
| `amplitude_m2_s2` | number | required | the coefficient's real part, m^2 s^-2 |
| `imaginary_m2_s2` | number | 0 | its imaginary part, zero at order 0 |
| `history_kyr` | array | absent | with `times` alone: its amplitude history (below) |

**Forcing through time.** With `times`, the forcing at a node is each
term's pattern at the node's time t times the term's factor there, summed
as above:

- `history_kyr`, on a load term or a tide coefficient, is a list of
  `[time_kyr, factor]` pairs, at least one, its times strictly increasing
  on the run's clock. The factor at t is the piecewise-LINEAR interpolant
  of the factors between the pairs' times, held CONSTANT at the first
  factor before the first time and at the last factor after the last; a
  single pair is a constant factor. A term without a history has the
  factor 1 throughout. A history is refused without `times`.
- A `field` term whose file has a time axis is read at each node by the
  file's own time rule, as `sea-level` reads an ice history at its nodes: a
  `node` field with linear time interpolation (what pyslfp and this
  program write) between its stored times, its first or last field held
  outside the file's span, without a warning; then expanded. The file's
  time axis is in seconds (Units and conventions, above), and the run's
  time t kyr reads the file at (t - `time_origin_kyr`) x 3.15576e10 s:
  with the default origin 0, t kyr on the run's clock is t kyr on the
  file's axis, so a `response` surface file (whose axis is the run's
  clock) reads back as it was written; a file whose axis counts from the
  present, negative in the past (an ice history, a `sea-level` output),
  is read from its date -D kyr at the run's start by
  `"time_origin_kyr": D` with `start_kyr` 0. A static file is the same
  pattern at every node. A field term may also carry `history_kyr`: its
  factor multiplies the field read. `time_origin_kyr` is refused without
  `times`.

### The output files

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `output.surface` | path | required | the surface file to write |
| `output.exchange` | path | absent | the surface coefficients as a cross-code exchange table; with `times`, one table per node, the path holding `{node}` |
| `output.exchange_nodes` | array of integers | every node | with `times` alone: the indices (from 0) of the nodes whose tables are written |
| `output.volume` | path | absent | without `times` alone: the response inside the body as a planetmodel model file |

With `times`, `output.exchange` must hold the literal `{node}`, which is
replaced by the node's index zero-padded to the width of the last index
(`00` to `40` for 41 nodes); without `times` it must not.
`exchange_nodes` needs `output.exchange`, and an index past the last node
is refused. `output.volume` is refused with `times`: the volume file is
written for the elastic response alone.

### The viscoelastic run

The run starts at the first node with no memory: the response there is
the instantaneous elastic one to the forcing at the first node (the
unrelaxed solve; under lateral moduli the conjugate-gradient solve of
the elastic run). Each step then advances from one node to the next
under the forcing AT the next node, the scheme's end-time loads being
that node's: the forcing enters at the nodes alone, and a history's or a
field file's variation between two nodes is not seen. This is
`sea-level`'s convention: a load that is non-zero at the first
node includes the elastic jump at the start (the run is the response to
a load switched on there), while a load ramped up from zero has none.

- 1D: the radial model's stepper, every degree with all its orders as
  columns, each step one direct solve per degree.
- 3D: `Viscoelastic3D`, each step closed by the fixed point or by
  conjugate gradients (`solver.step`) to `solver.step_rtol` within
  `solver.step_max_iterations`; a step that does not converge is an
  error. Over MPI ranks it is distributed over radial segments as a 3D
  `sea-level` run is, every rank reading the inputs and the first rank's
  forcing coefficients serving all. The radial reference model (the deck's radial moduli and
  the radial reference viscosity: the viscosity file's reference, or the
  deck's column) is stepped beside it by the 1D stepper under the same
  forcing, on the first rank. A 1D run on several ranks is the first
  rank's alone.

The first node's response is the elastic run's for the same forcing to
rounding (about 1e-11 relative on the test decks; under lateral moduli to
the conjugate gradients' tolerance), the radial stepper's unrelaxed
systems carrying the branch moduli at the memory points rather than at
the nodes.

### Outputs

**Without `times`.** The surface file, on the run's Gauss–Legendre grid,
holds `load` (kg m^-2, with a load term: the field above), `displacement`
(the radial displacement, m), `tangential_potential` (V, m: the
tangential displacement is its surface gradient), `potential`
(m^2 s^-2) and, under lateral moduli, `displacement_radial_model` and
`potential_radial_model`, the radial model's under the same forcing. Its
metadata record the config, deck and moduli paths, `lateral` (1 for the
3D solve), `load_terms` and `tide_coefficients` (the counts), `forcing`
(each term described, `; ` between them) and `solve_iterations` and
`solve_residual` (0 in 1D); when the load is exactly one `cap`, also
`load_latitude_deg`, `load_longitude_deg`, `load_width_deg` and
`load_amplitude_kg_m2`, and when the tide is exactly one coefficient,
`tide_degree`, `tide_order`, `tide_amplitude_m2_s2` and
`tide_imaginary_m2_s2`.

The exchange table is the format of AdGIA's benchmarks:

- `#` provenance lines: what the table holds, the code's version and the
  date, the config, deck and moduli paths (with `times`, the viscosity
  file's too), and a line `# forcing ...` describing each load term and
  tide coefficient (with its time origin and history where set);
- named scalars, one per line as `name value`: `radius` (m),
  `surface_gravity` (m s^-2), `lmax`, `lateral`, `load_terms` and
  `tide_coefficients`; then, when the load is exactly one `cap`,
  `cap_latitude_deg`, `cap_longitude_deg`, `cap_width_deg` and
  `cap_amplitude_kg_m2`, and when the tide is exactly one coefficient,
  `tide_degree`, `tide_order` and `tide_amplitude_m2_s2` (two numbers,
  the real and imaginary parts), so that a table of one cap and one tide
  carries the same scalars whatever else it holds;
- `columns l m U_re U_im Phi_re Phi_im` and one row per (l, m), m >= 0:
  the coefficients of the surface displacement (m) and potential
  (m^2 s^-2) in the harmonics of the coefficient tables above, SI,
  `%.17g`.

The volume file holds `radial_displacement` (m), `deviatoric_strain`
(sqrt(e:e), dimensionless) and, under lateral moduli, `mu_relative` and
`kappa_relative` (the perturbations applied) at the memory nodes of every
solid element but the one holding the centre, with the model's density
as the radial reference; planetmodel's `slices.cutaway` and
`plotting.cutaway` draw it (as `examples/python/plot_static_3d.py` does
for the library's example of the same problem).

**With `times`.** The surface file has one slice per node: its fields are
SI node fields with linear time interpolation, `load` (with a load term:
the load applied at each node, each term's field times its factor),
`displacement`, `tangential_potential` and `potential` and, in a 3D run,
`displacement_radial_model` and `potential_radial_model`, the radial
reference model's under the same forcing. Its time axis is the run's
clock in seconds: a node at t kyr is at t x 3.15576e10 s, so the axis
starts at `start_kyr` (zero by default). Its metadata record what the
elastic file's do, with `viscosity` (the viscosity file's path, where
given) and, in place of `solve_iterations` and `solve_residual`,
`dates_kyr` (the nodes, kyr), `time` (the axis's meaning), `scheme`,
`step_solver` (`"fixed_point"` or `"conjugate_gradients"` in 3D,
`"direct"` in 1D), and `step_iterations` and `step_residual`, one per
node (the first node's those of its conjugate-gradient solve under
lateral moduli, zero otherwise; zero throughout in 1D).

Each exchange table is the elastic one's format for one node, with two
named scalars more, `node` (the node's index) and `time_kyr` (its time),
before `columns`.

**Summary.** What the config was taken to mean, then the files. Elastic:
`<model>, elastic, <forcing>; wrote <surface> [and <exchange>] [and
<volume>]`, ending under lateral moduli in `(<n> conjugate-gradient
iterations, residual <r>)`; `<model>` is `3-D (lateral moduli)`, `1-D`,
or `1-D (the moduli file is radial)`. Viscoelastic: `<model>,
viscoelastic, <n> nodes <start> to <end> kyr, <forcing>; wrote <surface>
[and <k> exchange tables]`, ending in 3D in `(<i> fixed-point iterations
over <s> steps)` or `(<i> conjugate-gradient iterations over <s> steps)`,
the iterations of the steps summed; `<model>` is `3-D (lateral
viscosity)`, `3-D (lateral moduli)`, `3-D (lateral viscosity, lateral
moduli)`, `1-D`, or `1-D` followed by `(the viscosity file is radial)`,
`(the moduli file is radial)` or `(the viscosity and moduli files are
radial)`. `<forcing>` is `load (<n> term[s])`, `tide (<n>
coefficient[s])`, or both joined by ` + `. The example configs
`examples/configs/response.json` and
`examples/configs/response_viscoelastic.json` print

```
sl3d response: 3-D (lateral moduli), elastic, load (1 term) + tide (1 coefficient); wrote out/response/response.nc and out/response/response.txt and out/response/response_volume.nc (5 conjugate-gradient iterations, residual 3.52e-11) in 1.29 s
sl3d response: 3-D (lateral viscosity, lateral moduli), viscoelastic, 25 nodes 0 to 12 kyr, load (1 term) + tide (1 coefficient); wrote out/response_viscoelastic/response_viscoelastic.nc and 4 exchange tables (302 fixed-point iterations over 24 steps) in 4.49 s
```

### Examples

```json
{
  "model": {"deck": "earth.deck"},
  "lmax": 128,
  "forcing": {
    "load": [{"disk": {"latitude": 64, "longitude": 20, "radius_deg": 6,
                       "amplitude_kg_m2": 1.0e5}}]
  },
  "output": {"surface": "out/disk.nc"}
}
```

```json
{
  "model":   {"deck": "prem.deck", "format": "elastic",
              "moduli": {"file": "s40rts_moduli.nc", "reference": "lateral_mean"}},
  "lmax":    64,
  "forcing": {
    "load": [
      {"cap":  {"latitude": 60, "longitude": -85, "width_deg": 10, "amplitude_kg_m2": 1.8e6}},
      {"disk": {"latitude": -75, "longitude": 0, "radius_deg": 8, "amplitude_kg_m2": -9.0e5}},
      {"field": {"file": "load.nc", "name": "load"}},
      {"coefficients": {"file": "load_lm.txt"}}
    ],
    "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0, "imaginary_m2_s2": 0.0}]
  },
  "solver":  {"step_rtol": 1e-10, "step_max_iterations": 200, "transform_degrees": [16, 32], "cache_mb": 0},
  "output":  {"surface": "out/response.nc", "exchange": "out/response.txt", "volume": "out/response_volume.nc"}
}
```

The first is a 1D run (the deck's radial model); the second solves the
3D problem under the lateral moduli of `s40rts_moduli.nc`.

```json
{
  "model": {
    "deck": "prem_maxwell.deck", "format": "maxwell",
    "viscosity": {"file": "visc3d.nc", "reference": "geometric_mean"},
    "moduli":    {"file": "s40rts_moduli.nc"}
  },
  "lmax":  64,
  "times": {"start_kyr": 0, "end_kyr": 20, "nodes": 41},
  "forcing": {
    "load": [
      {"cap": {"latitude": 60, "longitude": -85, "width_deg": 10, "amplitude_kg_m2": 1.8e6},
       "history_kyr": [[0, 0.0], [10, 1.0], [10.5, 0.0]]},
      {"field": {"file": "load_history.nc", "name": "load"}}
    ],
    "tide": [{"degree": 2, "order": 0, "amplitude_m2_s2": 1.0,
              "history_kyr": [[0, 1.0]]}]
  },
  "solver": {"scheme": "exponential_trapezoid", "step": "fixed_point",
             "step_rtol": 1e-10, "step_max_iterations": 200,
             "transform_degrees": [16, 32], "cache_mb": 0},
  "output": {"surface": "out/relax.nc", "exchange": "out/relax_{node}.txt"}
}
```

A viscoelastic 3D run under a lateral viscosity and lateral moduli: a cap
loaded linearly over 10 kyr and removed over the next 0.5 kyr, a load
history read from a surface file on the run's clock, and a constant
tide, over 20 kyr in half-kyr steps, with a table per node
(`out/relax_00.txt` to `out/relax_40.txt`). Without `viscosity` and
`moduli` the same config runs the deck's radial Maxwell model in 1D.

A coefficient table, degrees 0 and 2 of a load (the m = 0 rows real):

```
# a load's coefficients, kg m^-2
0 0 3.5e3 0
2 0 -1.2e3 0
2 1 4.0e2 -2.5e2
```

## sea-level

The sea level equation: the ocean coupled to the solid Earth under an ice
load, on a state read from planetmodel surface files, with rotational
feedback. The config is made of blocks (The blocks, above): `model`,
`times`, `forcing`, `ocean`, `solver`, `checkpoint` and `output`, with
`lmax`, `rotation`, `moments` and `restart` beside them.

The `times` block decides what is solved:

- without `times` it is the elastic sea-level fingerprint of an ice change
  applied to a present-day state: one solve of the linear sea level
  equation on the state's shorelines, with the Love numbers of the deck's
  elastic structure at `lmax`, or, under lateral moduli, on the laterally
  heterogeneous elastic Earth (The fingerprint, below);
- with `times` it is GIA through time under an ice history: the
  viscoelastic Earth through the sea level equation over a window of
  dates (The run through time, below), restartable from checkpoints.

Whether the run is 1D or 3D is decided by the `model` block alone. A run
through time is 3D when the viscosity of `model.viscosity` or the moduli
of `model.moduli` vary laterally: `Evolution3D`, each step
solving the combined fixed point of the strain, the ocean load and the
rotation about the radial reference, over MPI ranks; otherwise it is the
radial model's (`Evolution`), on the deck's viscosity column or
the radial viscosity of `model.viscosity`. The fingerprint is 3D when the
moduli of `model.moduli` vary laterally (The fingerprint on lateral
moduli, below), and radial otherwise, a radial moduli file leaving the
deck's moduli as they are; it has no use for a viscosity, and
`model.viscosity` is refused without `times`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | object | required | the Earth model (The model block, above) |
| `lmax` | integer >= 1 | required | the degree of the run and of its grid |
| `times` | object | absent | the window, dates before present (The times block, above); absent, the fingerprint |
| `forcing` | object | required | the state and the ice (below) |
| `ocean` | object | absent | with `times` alone: the shorelines and the initial sea level's calibration (The ocean block, above) |
| `rotation` | word | `"full"` | the rotational feedback (Rotation and moments, above) |
| `moments` | word | `"input"` | the moments of inertia (Rotation and moments, above) |
| `solver` | object | absent | the solves (The solver block, above): without `times`, `rtol` alone on a radial Earth, and the 3D step's members too on lateral moduli |
| `checkpoint` | object | absent | with `times` alone: the checkpoints (Checkpoints and restarts, below) |
| `restart` | boolean | `true` | with `times` alone: whether to resume from a checkpoint (Checkpoints and restarts, below) |
| `output` | object | required | `output.surface`, the surface file to write (the block's other members are `response`'s) |

### The forcing block

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `forcing.state.file` | path | required | a surface file holding `ice_thickness` and `topography` (Present-day states and ice histories, above) |
| `forcing.state.date_kyr_bp` | number | 0 | without `times` alone: the date at which the fingerprint reads the state, kyr BP |
| `forcing.ice` | object | required | the ice: with `times` an ice history, without `times` a change (below) |

With `times`, `forcing.ice.file` (a path, required) names a surface file
holding `ice_thickness`, which may be the state's file. Without `times`,
`forcing.ice` is the change applied to the state:

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `forcing.ice.type` | word | required | `"cap_fraction"` or `"cap_thickness"` |
| `forcing.ice.latitude` | number | required | the cap's centre, degrees |
| `forcing.ice.longitude` | number | required | the cap's centre, degrees |
| `forcing.ice.width_deg` | number > 0 | required | the cap's width w, degrees |
| `forcing.ice.fraction` | number | required for `cap_fraction` | the fraction of the local ice thickness added (negative to melt) |
| `forcing.ice.thickness_m` | number | required for `cap_thickness` | the thickness added at the cap's centre, metres (negative to melt) |

The members of the other kind of run are refused, naming what this run
takes: a dated state or an ice change with `times`, an ice history
without. The ice is the block's one surface load.

### The fingerprint

Without `times`: the elastic fingerprint of the ice change on the state
read from `forcing.state.file` at `forcing.state.date_kyr_bp`. The cap is
Gaussian in great-circle distance d from its centre, weighted
exp(-d^2 / (2 w^2)) with no cut-off. A `cap_fraction` change is
`fraction` times the weight times the state's ice thickness, so it acts
only where there is ice; a `cap_thickness` change is `thickness_m` times
the weight. Either loads the Earth only where the ice is grounded. Giving
the amount member of the other type is refused. The solve's tolerance is
`solver.rtol`; a solve that does not converge is an error.

The output is a surface file of static SI fields on the run's
Gauss–Legendre grid:

| Field | Unit | Meaning |
| --- | --- | --- |
| `ice_change` | m | the ice thickness change applied (before the grounded-ice mask) |
| `relative_sea_level_change` | m | the sea-level change |
| `displacement` | m | the radial surface displacement |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |
| `ocean_function` | 1 | the ocean function of the state |

Its metadata record the config, deck and state paths (and the moduli
file's, where given), `date_kyr_bp`, the `rotation`, the angular velocity
perturbation as `omega_x_rad_s`, `omega_y_rad_s` and `omega_z_rad_s`
(rad s^-1), and the number of sea level iterations (`sle_iterations`).

### The fingerprint on lateral moduli

Without `times`, a `model.moduli` file whose moduli vary laterally makes
the fingerprint 3D: the same problem, the linear sea level equation on
the state's own shorelines under the same ice change, rotation and
moments, on the laterally heterogeneous elastic Earth. Its unrelaxed
solid varies laterally, so the solid's response is no Love-number
closure: the fingerprint is the instantaneous response of the 3D sea
level problem (`Evolution3D::Instantaneous`), the combined fixed
point of the strain, the ocean load and the rotation over a step of zero
length about the radial reference (docs/adjoints.md §13k, §13n). It is
what a 3D run through time solves at its start, here under the ice
change; the solid is the deck's elastic structure carrying the lateral
moduli, as in `response`'s elastic 3D solve, and no viscosity enters.
With the lateral part of the moduli zero (a file stored laterally but
uniform) it is the radial fingerprint to the solves' tolerance.

The members are the radial fingerprint's, and the 3D step's members of
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
the elements the moduli perturb (a fingerprint on a radial Earth is the
first rank's alone). The coupling carries every solid element where the
moduli are perturbed, so the cost grows with `lmax` as a 3D step's does:
the example config `examples/configs/sea_level_fingerprint_3d.json`
takes about 3 s and 0.9 GB at lmax 32 on four threads, and about 23 s
and 6 GB at lmax 64.

The output is the radial fingerprint's surface file, its fields the 3D
fingerprint's, with the radial model's fingerprint of the same change
beside them, so that the lateral moduli's effect is a subtraction:

| Field | Unit | Meaning |
| --- | --- | --- |
| `relative_sea_level_change_radial_model` | m | the sea-level change of the radial model's fingerprint |
| `displacement_radial_model` | m | its radial surface displacement |
| `potential_radial_model` | m^2 s^-2 | its potential perturbation at the surface |

Its metadata add `lateral` (1), the moduli file's reference
(`moduli_reference`: `lateral_mean` or `file_radial`), the step solver
(`step_solver`), the closure's passes (`step_iterations`), its
conjugate-gradient iterations (`solve_iterations`, 0 under the fixed
point) and its last residual (`step_residual`), and the radial model's
angular velocity perturbation (`omega_x_radial_model_rad_s`,
`omega_y_radial_model_rad_s`, `omega_z_radial_model_rad_s`); there is no
`sle_iterations`.

### The run through time

With `times`: viscoelastic GIA under the ice history over the window. The
run starts from the state at the window's start: the background is built
from `forcing.state.file` read at `times.start_kyr_bp`, and the load at
each time node is the change of grounded ice thickness in
`forcing.ice.file` from its value at the start. The outputs are changes
from the window's start. The time nodes are `times.nodes` evenly spaced
dates from `start_kyr_bp` to `end_kyr_bp`, stepped by `solver.scheme`.

The `ocean` block changes the start and the equation, as pyslfp's
`run_ice_history` does:

- `ocean.shorelines: "migrating"` solves the nonlinear sea level equation
  at every node instead of the linear one: the ocean function follows the
  flotation condition as the ice and sea level change, so melting marine
  ice floods and grounding ice displaces the ocean. The ice thickness at
  each node is the background's plus the history's change since the
  start. In a 3D run the nonlinear solve runs in every pass of each
  step's combined fixed point. The kernels of `kernel1d` and `kernel` then
  linearise about the run's own shorelines, and their
  `initial_sea_level_dual` is no longer zero.
- `ocean.calibrate_initial_sea_level` replaces the initial sea level read
  from `forcing.state.file` at the start by the one whose run reaches the
  sea level `forcing.state.file` gives at `times.end_kyr_bp` (usually the
  present day): the standard device when the ice history carries
  thickness alone. The run then starts as pyslfp's calibration does: the
  ice is the ice history's at the start, and the starting guess for the
  sea level is the target itself, so `forcing.state.file` is read at the
  window's end alone, which must lie within the file's time span. Under
  frozen shorelines the problem is linear and one run gives the answer
  exactly, the guess's shorelines kept; under migrating shorelines the
  calibration iterates, by the classical update or by steepest descent
  with the adjoint gradient, each accelerated by Anderson mixing. The
  calibration runs before the run proper, which then costs one more run.

A sea level solve that misses `solver.rtol` is an error, and in 3D so is
a node whose combined fixed point does not converge within
`solver.step_max_iterations` to `solver.step_rtol` (under migrating
shorelines a cell flipping back and forth can keep the nonlinear solve,
or the 3D fixed point, above a tight tolerance: loosen it). A 1D run
refuses the solver members of the 3D step. Under lateral moduli the
start's instantaneous response is the combined fixed point of a
zero-length step (`Evolution3D`).

The output is a surface file on the run's Gauss–Legendre grid with one
slice per time node, its time axis in seconds relative to the present
(negative in the past). Its fields are SI node fields with linear time
interpolation:

| Field | Unit | Meaning |
| --- | --- | --- |
| `relative_sea_level_change` | m | the sea-level change from the window's start |
| `displacement` | m | the radial surface displacement |
| `potential` | m^2 s^-2 | the gravitational potential perturbation at the surface |
| `initial_sea_level` | m | static: the initial sea level the run started from (the sea level at a node is this plus `relative_sea_level_change` there) |
| `ocean_function` | 1 | migrating shorelines alone: the ocean function at each node |

Its metadata record the shorelines (`shorelines`), where the initial sea
level came from (`initial_sea_level`: the state file at the start, or the
calibration with its runs and misfits), the config, deck and ice-history
paths (and the viscosity and moduli files', where given), the node dates
as `dates_kyr_bp`, and the angular velocity perturbation at each node as
`omega_x_rad_s`, `omega_y_rad_s` and `omega_z_rad_s` (rad s^-1).

### Checkpoints and restarts

A run through time takes a `checkpoint` block and `restart`; both are
refused without `times`.

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
there and the histories so far. It is written to a temporary file and
renamed into place, so an interrupted write leaves no partial checkpoint,
and only the newest two are kept.

When a run starts and finds a checkpoint in the directory, it resumes from
the newest one, with the histories up to that node restored. A resumed run
reproduces an uninterrupted one (bitwise, on one thread). Rerunning a
finished run whose last checkpoint is at the final node just writes the
output again.

A checkpoint is bound to its run: it records a hash of the members that
decide the computation (everything but `checkpoint`, `restart`,
`solver.cache_mb` and `output`) and of the contents of the files the
config names (the deck, the state, the ice history and the viscosity and
moduli files where given). A run is refused, with an error naming the
checkpoint, rather than resuming another run's state when any of these
differ: another time grid, tolerance or model, or the same config over an
input file that has changed since. The output, the checkpoint stride, the
wall-time limit and the transforms' cache may all change between a run
and its resumption, and the paths may change as long as the files'
contents do not, so a run directory can be moved. A checkpoint of a run
of the other dimension (a 1D run's against a 3D run's) is refused too.
The files are raw binary, for the machine that wrote them.

A calibrating run also keeps its calibrated initial state in the directory
(`initial_state.bin`, bound to the run by the same hash), and a resumption
reads it back instead of calibrating again.

With `restart` set to `false` the run starts over: it first removes the
checkpoints it finds in the directory (and any kept initial state), then
writes its own. Only this program's checkpoint files (`node_NNNNNN.ckpt`,
and the `.node_NNNNNN.partial` it renames) are read, pruned or removed;
anything else in the directory is left alone. Leave `restart` at `true` in
a chain of jobs that is meant to resume.

`checkpoint.max_walltime_min` serves batch systems with a time limit.
After writing a checkpoint, a run that has been stepping for longer than
the limit and has nodes still to go stops cleanly, with the summary
`<meaning>; checkpointed node <n> of <last> to <directory> at the
wall-time limit; rerun the same config to resume`. It exits with status 0
and writes no output yet, so a script can tell a stopped run from a
finished one by the summary or by whether `output.surface` exists;
rerunning the same config resumes it, so a chain of jobs running one
config completes the run. The clock starts with the time stepping, after
the model and files are read.

### Summary

What the config was taken to mean, then the file:

- the fingerprint: `<model>, elastic (fingerprint), rotation
  <rotation>[, moments hydrostatic]; wrote <surface> (<n> SLE
  iterations)`; on lateral moduli the parenthesis is the 3D closure's,
  `(<n> fixed-point iterations, residual <r>)`, or under
  `"conjugate_gradients"` `(<n> conjugate-gradient iterations in <k>
  passes, residual <r>)`;
- a run through time: `<model>, viscoelastic, <n> nodes <start> to <end>
  kyr BP, shorelines <frozen|migrating>[, initial sea level calibrated],
  rotation <rotation>[, moments hydrostatic]; wrote <surface>`, ending in
  `, resumed` when the run resumed from a checkpoint, and in
  `, replacing <k> earlier checkpoint(s)` when a run starting over
  removed some.

`<model>` is `1-D`, `1-D (the viscosity file is radial)`, `1-D (the
moduli file is radial)`, `1-D (the viscosity and moduli files are
radial)`, `3-D (lateral viscosity)`, `3-D (lateral moduli)` or `3-D
(lateral viscosity, lateral moduli)`. The example configs
`examples/configs/sea_level_fingerprint.json`,
`examples/configs/sea_level_fingerprint_3d.json` and
`examples/configs/sea_level_3d_calibrated.json` print

```
sl3d sea-level: 1-D, elastic (fingerprint), rotation full; wrote out/sea_level_fingerprint/sea_level_fingerprint.nc (9 SLE iterations) in 0.29 s
sl3d sea-level: 3-D (lateral moduli), elastic (fingerprint), rotation full; wrote out/sea_level_fingerprint_3d/sea_level_fingerprint_3d.nc (6 fixed-point iterations, residual 9.88e-10) in 3.38 s
sl3d sea-level: 3-D (lateral viscosity), viscoelastic, 43 nodes 21 to 0 kyr BP, shorelines migrating, initial sea level calibrated, rotation full, moments hydrostatic; wrote out/sea_level_3d_calibrated/sea_level_3d_calibrated.nc in 28.34 s
```

### Examples

```json
{
  "model": {"deck": "earth.deck"},
  "lmax": 128,
  "forcing": {
    "state": {"file": "ice7g.nc"},
    "ice": {"type": "cap_thickness", "latitude": -75, "longitude": 250,
            "width_deg": 8, "thickness_m": -10}
  },
  "output": {"surface": "fingerprint.nc"}
}
```

The fingerprint of thinning West Antarctic ice by 10 m at its centre.

```json
{
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "moduli": {"file": "s40rts_moduli.nc"}},
  "lmax": 64,
  "forcing": {
    "state": {"file": "ice7g.nc"},
    "ice": {"type": "cap_fraction", "latitude": 72, "longitude": 318,
            "width_deg": 8, "fraction": -0.1}
  },
  "rotation": "full",
  "solver": {"step": "conjugate_gradients", "step_rtol": 1e-9},
  "output": {"surface": "fingerprint_3d.nc"}
}
```

The fingerprint of melting a tenth of Greenland's ice on the lateral
moduli of `s40rts_moduli.nc` (a planetmodel model file of `mu` and
`kappa`), its zero-length step closed by the sea level closure nested
over conjugate-gradient solves, with the radial model's fingerprint
beside it in `fingerprint_3d.nc`.

```json
{
  "model": {"deck": "earth.deck"},
  "lmax": 64,
  "times": {"start_kyr_bp": 26.0, "nodes": 53},
  "forcing": {"state": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "output": {"surface": "histories.nc"}
}
```

A thickness-only history under migrating shorelines, its initial sea level
calibrated against the present-day state, checkpointed every four nodes:

```json
{
  "model": {"deck": "earth.deck"},
  "lmax": 64,
  "times": {"start_kyr_bp": 26.0, "nodes": 53},
  "forcing": {"state": {"file": "present_day.nc"},
              "ice": {"file": "ice_history.nc"}},
  "ocean": {"shorelines": "migrating",
            "calibrate_initial_sea_level": {"method": "fixed_point",
                                            "iterations": 12}},
  "checkpoint": {"directory": "checkpoints", "every": 4},
  "output": {"surface": "histories.nc"}
}
```

Here `forcing.state.file` need only hold the present day: with the
calibration it is read at the window's end alone.

```json
{
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "viscosity": {"file": "s40rts_viscosity.nc",
                          "reference": "geometric_mean"}},
  "lmax": 128,
  "times": {"start_kyr_bp": 26.0, "end_kyr_bp": 0.0, "nodes": 105},
  "forcing": {"state": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
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

The first two are fingerprints, radial and 3D, and the next two radial
runs through time; the last is 3D, under the lateral viscosity of
`s40rts_viscosity.nc`.

## kernel1d and kernel

Adjoint sensitivity kernels of configured measurements of a run through
time. The forward run is `sea-level`'s run through time of the same
blocks (The blocks, above): `model`, `lmax`, `times` (required, dates
before present), `forcing` (`state` and an ice history), `ocean`,
`rotation`, `moments` and `solver`, each meaning what it means there.
Then, for each measurement, one reverse (adjoint) sweep gives its
sensitivity to the ice history and to the viscosity, by the continuous
adjoint or the exact discrete transpose.

`kernel1d` runs a radial model: the deck's viscosity column, or a
`model.viscosity` file whose viscosity is radial; a laterally varying one
is refused (use `kernel`), and so are the solver members of the 3D step.
`kernel` runs a laterally varying viscosity: `model.viscosity` is
required, and a radial one is refused (use `kernel1d`). Neither takes
`model.moduli`, `checkpoint` or `restart`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| the shared blocks | | | as for `sea-level`'s run through time (The blocks, above) |
| `record_stride` | integer >= 1 | 1 | record the forward state at every `record_stride`-th node (and the last); the sweep rebuilds the states in between |
| `adjoint` | word | `"continuous"` | `"continuous"` or `"discrete"` (the exact transpose of the time-stepped map; in 3D distributed like the continuous sweep) |
| `trajectory.directory` | path | none | `kernel` alone: spill the recorded states to this directory instead of holding them in memory |
| `measurements` | array | required | the measurements, at least one (below) |
| `output` | path | required | the directory the kernel files go into |

The kernel commands have no checkpoints. Recording every n-th of N nodes
holds about N/n + n forward states during the sweep, fewest near n = sqrt(N);
in 3D each is large (see the README's notes on memory). With
`trajectory.directory`, `kernel` writes each recorded state to a file
`node_NNNNNN.traj` there as the forward run passes it, and the sweep reads
them back one at a time: the lever for large `lmax`. The files are raw
binary; the run removes the ones it wrote once its sweeps are done (the
summary says how many), leaving anything else in the directory alone, and
a run that fails leaves them in place. `kernel1d` refuses a
`trajectory` section, its trajectory fitting in memory.

### Measurements

Each element of `measurements` is an object with a `type` and a `name`.
The `name` (by default `measurement_<k>`, with k the element's position
from 0) is the stem of the measurement's files, so names must be distinct
plain file names: letters, digits, `_`, `-` and `.`, not starting with `.`
(anything else is refused when the config is read). A measurement taken on one date reads the outputs
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
  SI, a density per unit volume as the 3D kernels are: dJ = integral of
  K delta eta dV = 4 pi integral of K(r) delta eta(r) r^2 dr, eta in
  Pa s (neither the r^2 nor the 4 pi is in K). The
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
  "model": {"deck": "earth.deck"},
  "lmax": 32,
  "times": {"start_kyr_bp": 20.0, "nodes": 41},
  "forcing": {"state": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "measurements": [
    {"type": "sea_level_point", "name": "fremantle",
     "latitude": -32.05, "longitude": 115.74}
  ],
  "output": "kernels"
}
```

```json
{
  "model": {"deck": "earth.deck"},
  "lmax": 64,
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 41},
  "forcing": {"state": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
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
  "model": {"deck": "prem8.elastic.deck", "format": "elastic",
            "viscosity": {"file": "s40rts_viscosity.nc"}},
  "lmax": 64,
  "times": {"start_kyr_bp": 20.0, "end_kyr_bp": 0.0, "nodes": 81},
  "forcing": {"state": {"file": "ice7g.nc"}, "ice": {"file": "ice7g.nc"}},
  "solver": {"step_rtol": 1e-9, "step_max_iterations": 60},
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
