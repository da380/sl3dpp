# sl3dpp examples

Nine small drivers, one per capability of the library, each paired with a
Python plotting script, and an example config for every command of the
`sl3d` program:

| Driver | What it shows | Default | Plot |
| --- | --- | --- | --- |
| `ExampleLoveNumbers` | load and tidal Love numbers of the PREM-like deck, in the conventional (Farrell) normalisation | lmax 256 | `plot_love_numbers.py` |
| `ExampleFingerprint` | the elastic sea-level fingerprint of a 10% Greenland melt: one solve of the generalised sea level equation with full rotational feedback, SI in and out | lmax 256 | `plot_fingerprint.py` |
| `ExampleEvolution` | 1D viscoelastic GIA: a West Antarctic melt ramp then hold, the elastic jump and the viscous creep after it, polar motion | lmax 192 | `plot_evolution.py` |
| `ExampleLateralViscosity` | 3D viscosity: an analytic lateral pattern of one decade against its radial twin, and the cost of the within-step iteration | lmax 48 | `plot_lateral.py` |
| `ExampleModelViscosity` | the route from real tomography: a 3D viscosity read from a planetmodel model file (`io::ModelViscosity`), run through the whole deglaciation against the file's own radial reference | lmax 32 | `plot_model_viscosity.py` |
| `ExampleIceKernels` | 1D adjoint kernels of a tide gauge near the melting Greenland ice: rate kernels of the ice history at four intervals, radial viscosity kernels by the continuous and the discrete adjoint; the forward run checkpointed | lmax 128 | `plot_kernels.py` |
| `ExampleViscosityKernels3D` | 3D adjoint kernels by the discrete adjoint (the library's default): the log-viscosity kernels of a sea-level and a decay-time measurement at Hudson Bay, written as a planetmodel model file (`io::WriteKernels`); the forward run checkpointed | lmax 24 | `plot_kernels_3d.py` |
| `ExampleStatic3D` | 3D elasticity: the static response of the synthetic 3D elastic Earth (`data/synthetic_moduli.nc`, lateral moduli read by `io::ModelModuli`) to a cap load over Hudson Bay, solved exactly by conjugate gradients on the radial systems, against the radial model's; the response inside the body as a model file for planetmodel's cutaways | lmax 32 | `plot_static_3d.py` |
| `ExampleSeismic` | a seismic source: a thrust off the synthetic South America as point sources on a planar fault (`PlanarFaultSources`), its co-seismic response and 300 years of relaxation with an ocean on the post-seismic Earth (`data/prem8.postseismic.deck`: a 50 km elastic lid over a 1e19 Pa s asthenosphere) and on its twin with a Burgers asthenosphere (`Rheology::Burgers`), the horizontal motion from the tangential and toroidal potentials (`DisplacementField`); and a tide gauge's kernels in the patches' moment tensors (`AdjointSolver::Options::moment_points`) predicting the change a turn of the rake from 90 to 120 degrees makes, checked against a second run (to 1e-13); maps and series as 3° Gaussian means, the degree weights of the cap measurements' cap (`CapMeanWeights`) | lmax 64 | `plot_seismic.py` |

The defaults are sized for a laptop: each run takes from a fraction of a
second to a minute or two on 8 threads, `ExampleSeismic` (runs through
time and an adjoint sweep, its fault's 36 point sources forcing every
step) and `ExampleViscosityKernels3D` (two discrete 3D sweeps after its
forward run) the longest, and `ExampleIceKernels` takes the most memory.
The
configs, one per `sl3d` command, are described in
[configs/README.md](configs/README.md).

Every driver takes `[--lmax N] [--state FILE]`: the degree (the defaults are
sized for a laptop; CI runs each at a small one) and the Earth's state.
The drivers' measurements over Gaussian caps take caps their default
degree resolves, a width at least half the truncation wavelength
360°/lmax (configs/README.md, where the kernel configs are described):
6° at lmax 32 (`ExampleModelViscosity`), 8° at lmax 24
(`ExampleViscosityKernels3D`), 3° at lmax 64 (`ExampleSeismic`) and
2.9° at lmax 128 (`ExampleIceKernels`); at a much lower `--lmax` the band
cuts them off.

Each driver writes into its own `output/library/<example>/`, and its
plotting script draws `figures/library/<example>.png`, where `<example>` is
`love_numbers`, `fingerprint`, `evolution`, `lateral_viscosity`,
`model_viscosity`, `ice_kernels`, `viscosity_kernels_3d`, `static_3d` or
`seismic`.
The table is `<example>.txt`; `ExampleViscosityKernels3D` also writes the
kernels as a model file, `viscosity_kernels.nc`, and `ExampleStatic3D` the
response inside the body, `static_3d_volume.nc`.

## The inputs

Every input is in `build/examples/data/`.

- **The synthetic 3D elastic Earth**, `synthetic_moduli.nc`, which the
  build writes too (`MakeSyntheticModuli`): a planetmodel model file of
  the PREM-like deck's density and moduli with a lateral perturbation of
  the upper mantle — a stiff craton under the Canadian shield, a soft
  ridge under the eastern Pacific — in the form planetmodel's conversion
  of a tomography model writes, for `ExampleStatic3D` and `no_ocean.json`.
- **The synthetic perturbations of the viscosity**,
  `synthetic_weak_zone.nc` and `synthetic_weak_patch.nc`, which the build
  writes too (`MakeSyntheticPerturbation`): planetmodel model files on the
  PREM-like deck's body whose field `ln_viscosity` holds a CHANGE of the
  natural logarithm of the viscosity (a name planetmodel's vocabulary does
  not hold, written with its spec: a dimensionless scalar and its
  meaning) — a weak zone of the upper mantle, a factor of ten at its peak
  at 250 km depth, under the whole Earth (a function of the radius alone)
  and under a cap about Hudson Bay (varying laterally) — the form a
  direction in the viscosity is given to `sl3d tangent` in, for
  `tangent1d.json` and `tangent3d.json`.
- **The synthetic Earth**, `synthetic_earth.nc`, which the build writes
  (`MakeSyntheticEarth`): a planetmodel surface file of `ice_thickness` and
  `topography` on a 1° grid over a toy glacial cycle — smooth continents on
  a 4 km deep ocean, Laurentide and Fennoscandian sheets melting away from
  21 ka and Greenland and Antarctica shrinking to their present size. It
  needs no data, so the examples run anywhere, CI included. The drivers use
  it by default.
- **ICE-7G**, the real thing, through pyslfp: `get_ice7g.py` writes
  `data/ice7g.nc` in the same form (below), and `--state data/ice7g.nc`
  runs any driver on it; `configs/*_ice7g.json` are the configs that use
  it.
- **The toy planet**, `planet.gia.deck` and `planet.elastic.deck`, which
  `make_planet_deck.py` writes (below), for the configs
  `no_ocean_tide_planet.json` and `no_ocean_tide_planet_relax.json`.
- The PREM-like deck `prem8.elastic.deck` and, for the 3D model-file
  example and configs, `viscosity_log.nc`, a planetmodel model file of a
  3D viscosity on the same body (both copied from `tests/data`), and
  `prem8.gia.deck`, the same deck in planetmodel's GIA format with a
  1e21 Pa s Maxwell mantle, for the configs' radial runs through time
  (copied from `examples/data`).
- **The post-seismic Earth**, `prem8.postseismic.deck` (copied from
  `examples/data`; `make_postseismic_deck.py` writes it): PREM without
  its ocean, sampled as `prem8.elastic.deck` is, with an elastic lid 50
  km thick over a Maxwell asthenosphere of 1e19 Pa s to 217 km depth, an
  upper mantle of 1e20 Pa s to 667 km and a lower mantle of 1e22 Pa s,
  their Maxwell times 4.7 to 4.8, 26 to 43 and 1080 to 2050 years
  (configs/README.md), for `ExampleSeismic` and the post-seismic
  configs.

A state's sea level is built from its topography and ice thickness by
`SeaLevelFromTopography` (pyslfp's construction), as the `sl3d` program
does.

A generated input is replaced only where its content changed: the build's
generators (`KeepUnchanged.hpp`) and the `make_*.py` scripts
(`keep_unchanged` in `inputs.py`) write beside the target and keep the
target, its bytes untouched, where the two hold the same dimensions,
variables, values and attributes but for the creation stamp `created`
(the same bytes, for a deck or a table). A run's hash covers its inputs'
bytes, so the checkpoints and trajectory stores of a run stay its own
across rebuilds that leave its inputs as they were, and a generator whose
output does change rewrites it.

## Making inputs with planetmodel and pyslfp

The `make_*.py` scripts write inputs for the `sl3d` program into `data/`,
each run from `build/examples` as `poetry run python make_<...>.py`, each
saying in its docstring (and `--help`) what it writes and how a config
uses it; `inputs.py` holds what they share.

| Script | Writes (by default) | Read by |
| --- | --- | --- |
| `make_earth_deck.py` | `earth_custom.gia.deck`, `earth_custom.elastic.deck`: PREM without its ocean, an elastic lithosphere (100 km), Maxwell upper and lower mantles (5e20 and 5e21 Pa s), isotropic where they relax | `model.deck` |
| `make_planet_deck.py` | `planet.gia.deck`, `planet.elastic.deck`: a three-layer toy Mars (fluid core, Maxwell mantle, elastic lid); prints its mass and surface gravity | `model.deck`; `no_ocean_tide_planet*.json` |
| `make_postseismic_deck.py` | `prem8.postseismic.deck`: PREM without its ocean, an elastic lid (50 km) over Maxwell asthenosphere, upper and lower mantles (1e19, 1e20 and 1e22 Pa s), isotropic where they relax; prints the layers' Maxwell times. Its defaults write the deck `examples/data` holds | `model.deck`; the post-seismic configs, `ExampleSeismic` |
| `make_lateral_model.py` | `<deck>_<field>_<kind>.nc` (`s40rts` in place of the `tomography` kind, as `earth_custom_viscosity_s40rts.nc`): a 3D viscosity or lateral moduli on a deck's own body, an analytic pattern, S40RTS or a Gaussian random field; from S40RTS the viscosity by sl3d's conversion chain (optionally renormalised to the deck's mean, with a lid), the moduli from the shear speed and a scaled P speed (`--vp-scaling`) | `model.viscosity.file`, `model.moduli.file` |
| `make_ice_history.py` | `ice7g.nc` (ICE-5G/6G/7G through pyslfp), or a hand-made disk ice sheet from numpy arrays, `synthetic_disk.nc` and its present day, `synthetic_disk_present.nc` | `ocean.ice` and `ocean.topography` (the state), a `thickness` term of `forcing.ice_change` (the history) |
| `make_load_field.py` | `load_disk.nc`, `load_cap.nc` or `load_greenland.nc`, a surface mass density, static or (`--history`) on the run's clock | a `field` load term (`forcing.load`) |
| `make_load_coefficients.py` | `<file>_lm.txt`: a load field's coefficient table, in the exchange table's harmonics | a `coefficients` load term (`forcing.load`) |
| `make_megathrust.py` | `megathrust.nc`: a planetmodel fault file, a curved megathrust off the synthetic South America swept from a small-circle trench, a thrust tapered from 12 m. It needs planetmodel's fault module (after 1.5.1; until a release carries it, `PYTHONPATH=<planetmodel checkout>/src`), and without it says so and writes nothing | a `fault_file` seismic term (`forcing.seismic`); `sea_level_static_megathrust.json` |
| `make_tide_history.py` | `configs/no_ocean_tide_oscillating.json`: a config of a sinusoidal degree-2 tide at a chosen period on a body without an ocean | `sl3d solve` |

A data set a script needs that is not on disk yet (S40RTS for
planetmodel, ICE-xG for pyslfp) is downloaded only with `--download`;
without it the script says what it would fetch and stops (`get_ice7g.py`
gives `--download` itself).

`timestep_study.py` is not an input script: it writes the time-step
study's configs into `configs/` and the decks they need into `data/`,
runs each through `sl3d` and prints the errors of the forward runs and of
both adjoints' kernels against a truth taken from forward runs alone
(configs/README.md, The time-step study); `--short` is the form
`configs/` holds and `run_all.sh` runs.

## Running the sl3d program on the examples' inputs

Every command of the `sl3d` program has a config in `build/examples/configs`,
set up to run on the inputs above, so the program can be tried without
writing a config from scratch:

```sh
cd build/examples/configs
../../apps/sl3d solve sea_level_fingerprint.json   # one config
./run_all.sh                                       # every config in turn
```

Each config's outputs land in `configs/out/<config>/` (`<config>` is the
file name without `.json`), and `poetry run python plot_app.py`, from
`build/examples`, draws each config's figures into
`figures/sl3d/<config>/`, named for what they show:

| Config | Figure |
| --- | --- |
| `love_numbers` | `love_numbers.png` |
| `sea_level_fingerprint`, `sea_level_fingerprint_ice7g` | `fingerprint.png` |
| `sea_level_fingerprint_3d` | `fingerprint.png`: the fingerprint on the synthetic 3D elastic Earth and its 3D minus radial difference |
| `sea_level_1d` | `sea_level.png`: the 1D run and `sea_level_3d`'s together |
| `sea_level_1d_ice7g` | `sea_level.png` |
| `sea_level_1d_calibrated`, `sea_level_3d_calibrated`, `sea_level_1d_ice7g_calibrated` | `calibration.png` |
| `kernel1d` | `kernels.png`: its ice kernels and radial viscosity kernels |
| `kernel` | `viscosity_kernels.png` |
| `no_ocean`, `no_ocean_viscoelastic`, `no_ocean_1d_disk` | `displacement.png` |
| `no_ocean_1d_tide` | `love_numbers.png` |
| `no_ocean_tide_3d` | `body_tide.png`: the radial model's tidal displacement, the 3D minus radial difference and the difference's spectrum by degree |
| `no_ocean_tide_oscillating` | `oscillating.png`: the potential and the response through the run, and the Love numbers' amplitudes and lags at its period and those of `no_ocean_tide_oscillating_short` and `_long` |
| `no_ocean_tide_planet` | `body_tide.png`: the toy planet's tidal displacement and potential, with its h2 and k2 |
| `no_ocean_tide_planet_relax` | `love_numbers.png`: the toy planet's h2 and k2 through time, beside the Earth's |
| `sea_level_fingerprint_ice_cap` (with `no_ocean_ice_cap`) | `ocean.png`: one cap of ice laid on bare land without an ocean and with one, the displacement each way, its difference (the ocean's load) and the sea-level change |
| `no_ocean_cap_weak` (with `no_ocean_cap_strong`) | `relaxation.png`: the lateral viscosity with the two caps, and the displacement at each cap's centre through time, 3D and radial model |
| `no_ocean_load_field` (with `no_ocean_load_cap`, `no_ocean_load_coefficients`) | `agreement.png`: the forcing on the run's clock and the file's axis, the three runs' displacement, and their agreement |
| `timestep_study` (the `timestep_*` configs) | `accuracy.png`: the errors against the step of the sea level by both schemes and of the measurements' derivatives by both adjoints, in the viscosity (relative, and in metres beside the values') and along a change of the ice, against the truth from forward runs; `layers.png` (the whole study): each adjoint's error layer by layer, and the continuous adjoint's against the step over the shortest relaxation time with the viscosity scaled; `convergence.png`: the change at each halving of the step beside the error, and the error by the cap's width |
| `sea_level_fingerprint_3d_ice7g` | `fingerprint.png`: the fingerprint on PREM with S40RTS's moduli, over the ocean and near Greenland, and its 3D minus radial difference |
| `sea_level_3d_ice7g` (with `sea_level_1d_ice7g_custom`) | `sea_level.png`: the sea-level change since 26 ka in 3D and 3D minus radial, and relative sea-level histories at three sites, 3D and radial |
| `sea_level_static_coseismic` | `coseismic.png`: maps centred on the fault of the vertical displacement, the horizontal displacement (its size, and arrows) and the relative sea-level change over the ocean, 2° Gaussian means (the fields the run writes smoothed, `output.smoothing`), the fault's surface projection drawn, and all of them along a profile across the fault, with the vertical displacement's point values, whose ripples are the truncation |
| `sea_level_static_megathrust` | `coseismic.png`: as `sea_level_static_coseismic`'s, the fault's outline and trench drawn from its file (with planetmodel's fault module), the profile through the middle of the fault across its strike there |
| `sea_level_postseismic`, `sea_level_postseismic_burgers`, `no_ocean_postseismic`, `no_gravity_postseismic`, `no_ocean_seismic` | `postseismic.png`: maps centred on the source of the vertical displacement at the event and of its change since then at two later nodes (near twice the shortest Maxwell time, and the last), of the sea-level change (without an ocean, the horizontal motion) since the event at the last, and the vertical displacement and the sea level (without an ocean, the horizontal displacement up dip) through time at three places, all 2° Gaussian means (the fields the run writes smoothed), the co-seismic step included, on a time axis linear about the event and logarithmic after it with the deck's Maxwell times marked |
| `kernel_postseismic` | `kernels.png`: each measurement's kernels in the moment tensors of the fault's patches (the six components at each patch, held from the onset), their contraction with the fault's own mechanism (the derivative per unit of each patch's slip, summing to the measurement with the fault's moment), and the radial viscosity kernels |
| `no_gravity_postseismic` against `no_ocean_postseismic` | `with_and_without_gravity/postseismic.png`: the vertical and horizontal displacement through time at three places without gravity and with it on one panel, and the vertical displacement across the fault at the event and the last node, 2° Gaussian means |
| `no_gravity_load` against `no_ocean_load_cap` | `with_and_without_gravity/load.png`: the displacement at the cap's centre through time without gravity (the cap and its antipode) and with it (the cap alone), against distance from the cap at three times, and the map without gravity |
| `sea_level_tide_migrating` | `sea_level.png`: the relative sea-level change since 21 ka at four dates over each date's own ocean, the shorelines the run moved, and the tide's equilibrium surface |

`sea_level_3d`'s run is drawn against the 1D one in
`figures/sl3d/sea_level_1d/sea_level.png` (in `sea_level_3d/` when it
alone has run), so it has no folder of its own. The static problem's
`sea_level_static_migrating` (the synthetic deglaciation under migrating
shorelines) has no figure, nor has `sea_level_no_ocean`
(`no_ocean_load_cap`'s run, its times dates before present); their
surface files hold the fields `examples/configs/README.md` names.
The three runs of the thrust off South America through time,
`sea_level_postseismic`, `no_ocean_postseismic` and
`no_gravity_postseismic`, share the fault, the Earth and the nodes: they
differ by the ocean and by gravity alone.

The comparisons of a run with gravity and its twin without are drawn in
`figures/sl3d/with_and_without_gravity/`, not in either run's folder.

The seismic configs run at lmax 96 and ask for every field of the
response smoothed over a Gaussian cap of 2° (`output.smoothing`: the
field averaged over a cap of 2° about each point, written beside it) and
for the horizontal displacement's east and north components
(`output.fields`); their figures draw the smoothed fields as the files
hold them, saying so: a fault a few kilometres deep laid out as
point sources 50 km apart is not resolved at any degree these runs
reach, its point values ringing about the source at the truncation and
moving by a third of their size from one lmax to the next, while the 2°
means are converged to a few tenths of a per cent
([configs/README.md](configs/README.md), Convergence in lmax of the
seismic examples). The co-seismic figure's profile draws the point
values beside the means, for their ripples. The thrust off the synthetic
South America lifts the sea floor by up to 0.16 m and moves it 0.87 m
towards the trench in 2° means, the sea level falling by the uplift over
the ocean. On the post-seismic Earth the asthenosphere then relaxes over
years to decades and the upper mantle over decades to centuries: over
the fault the uplift grows over the first century and falls back after
it, the curve bending on the logarithmic axis at the layers' Maxwell
times. Without gravity it grows the more. `ExampleSeismic`'s Burgers
asthenosphere relaxes a part of the way within months, where the Maxwell
one has barely begun, the two Earths then relaxing alike; the program
runs the same twin from a config, `sea_level_postseismic_burgers`
(`model.burgers`). The moment kernels contracted with the fault's own
mechanism and moment give back each measurement's value.

The tide figures show three things. On the synthetic 3D Earth the lunar
body tide is the radial model's to 2e-4: a bulge of 0.21 m, which the
lateral moduli change by 0.04 mm at most, the 3D model putting 7.0e-5 of
its rms into degrees other than 2, all of it the lateral coupling. On the
radial viscoelastic Earth an oscillating tide's response lags the
potential, most at periods between the elastic and fluid limits, its
amplitude rising from the
elastic Love numbers towards the fluid ones as the period grows. The toy
planet's elastic Love numbers are h2 = 0.339 and k2 = 0.182, and a held
tide relaxes several times more slowly on it than on the Earth.

**The configs in `build/examples/configs` are yours to edit.** The build copies
a config there when its copy is missing, and again when its source has
changed and its copy has not been edited (an edited copy is kept, the
configure step saying so), so change the degree, the dates, the ice
change or the measurements and run again; delete a file (or the whole
directory) and reconfigure to get the original back. [configs/README.md](configs/README.md) says what each config does, and every
member is documented in the configuration reference
([docs/sl3d-configs.md](../docs/sl3d-configs.md)). One thing to know:
`sea_level_1d.json` writes checkpoints into `out/sea_level_1d/checkpoints`,
and a run refuses another run's checkpoints, so after changing it set
`"restart": false` (or delete that directory) for the first rerun.

## Outputs generated in the build

`cmake --build build -t sl3dpp_example_outputs` runs every driver at the
small degree CI uses (seconds each) from `build/examples`, each writing
into its `output/library/<example>/` with what it prints as
`<example>.log` there, and with `-DSL3DPP_RUN_EXAMPLES=ON` the build does
so by itself; `-t sl3dpp_example_figures` then draws every library
figure into `figures/library/` through the poetry environment there
(`poetry install` in `build/examples` first). The full-size runs are the
drivers' own defaults.

## Everything runs in the build tree

At configure time CMake copies the inputs into `build/examples/data`, the
configs into `build/examples/configs` (above), and the Python scripts and
the poetry file into `build/examples`, and the build writes the synthetic
Earths into `data/`. Everything is run from `build/examples` (the configs
from `configs/`), and every path is relative to it:

```
build/examples/
  Example*, Make*         the drivers and the synthetic-Earth writers
  *.py, pyproject.toml    the plotting scripts and plotutil.py, the loading
                          and mapping they share; get_ice7g.py, the make_*.py
                          input scripts and timestep_study.py; the poetry file
  data/                   every input: the decks, viscosity_log.nc,
                          synthetic_earth.nc, synthetic_moduli.nc,
                          synthetic_weak_zone.nc, synthetic_weak_patch.nc,
                          ice7g.nc
                          and what the make_*.py scripts write
  output/library/<example>/   a driver's outputs and its log
  figures/library/            a driver's figure, <example>.png
  figures/sl3d/<config>/      a config's figures
  configs/                the sl3d configs and run_all.sh
  configs/out/<config>/   a config's outputs
```

The poetry environment and a downloaded ICE-7G file land there too. The
source tree is never written to.

## Setup and run

```sh
# 1. Build (from the repository root; the examples are on by default).
cmake -B build && cmake --build build -j

# 2. Run the drivers from the build tree.
cd build/examples
./ExampleLoveNumbers                      # writes output/library/love_numbers/
./ExampleFingerprint
./ExampleEvolution
./ExampleLateralViscosity
./ExampleModelViscosity
./ExampleIceKernels
./ExampleViscosityKernels3D
./ExampleStatic3D
./ExampleSeismic

# 3. One-off: create the Python environment HERE (not in the source).
poetry install --no-root

# 4. Plot (figures land in figures/library/).
poetry run python plot_love_numbers.py    # and plot_fingerprint.py, ...

# 5. The sl3d configs (outputs in configs/out/<config>/), and their
#    figures (in figures/sl3d/<config>/).
(cd configs && ./run_all.sh)
poetry run python plot_app.py
```

## ICE-7G

```sh
cd build/examples                         # from the repository root
poetry run python get_ice7g.py            # writes data/ice7g.nc
./ExampleFingerprint --state data/ice7g.nc    # any driver
(cd configs && ./run_all.sh)              # every config, the *_ice7g.json ones among them
```

With ICE-7G there, the custom PREM deck and its S40RTS files make the
3D configs on the real Earth (configs/README.md, PREM with S40RTS on
ICE-7G):

```sh
poetry run python make_earth_deck.py      # data/earth_custom.{gia,elastic}.deck
poetry run python make_lateral_model.py --field moduli --kind tomography \
    --lmax 64                             # data/earth_custom_moduli_s40rts.nc
poetry run python make_lateral_model.py --field viscosity --kind tomography \
    --renormalise                         # data/earth_custom_viscosity_s40rts.nc
(cd configs && ./run_all.sh)              # runs them too (minutes)
```

and the load given as a file and as a table (configs/README.md,
`no_ocean_load_field.json`) needs

```sh
poetry run python make_load_field.py --shape cap --lmax 32 --width-deg 5
poetry run python make_load_field.py --shape cap --lmax 32 --width-deg 5 \
    --history 0:0 1:1 5:1 5.5:0           # data/load_cap_history.nc
poetry run python make_load_coefficients.py data/load_cap.nc
```

`get_ice7g.py` is `make_ice_history.py --model ICE7G --download`, and
uses pyslfp's `export_ice_history`: ICE-7G's stored slices
from 26 ka to the present (every kiloyear to 21 ka, then every half
kiloyear), `ice_thickness` and `topography` on the dataset's grid, in
metres. The first run downloads pyslfp's ICE-7G archive from Zenodo (about
24 MB) into pyslfp's data directory (`~/.pyslfp_data`, or `$PYSLFP_DATA`);
later runs reuse it. `--version ICE6G` or `ICE5G` writes those models
instead (`data/ice6g.nc`, `data/ice5g.nc`).

Notes:

- The drivers run **threaded by default** (OpenMP), on every core unless
  told otherwise: set `OMP_NUM_THREADS=N` to choose, or 1 for a serial run.
  For the 3D drivers, `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` also asks for
  huge pages for their large tensors ([docs/running.md](../docs/running.md),
  Memory).
- The Python environment holds pyslfp (for ICE-7G), planetmodel (to read
  the `sl3d` outputs and the synthetic Earth) and matplotlib with cartopy
  for the maps. Maps of a run on the synthetic Earth draw its own
  coastline; maps of a real Earth draw the real one (cartopy downloads the
  coastline data on first use). Without cartopy the scripts fall back to
  plain latitude–longitude maps.
- `poetry install --no-root` installs into poetry's cache, or set
  `POETRY_VIRTUALENVS_IN_PROJECT=1` first to keep the environment in
  `build/examples/.venv`. Either way the source tree stays clean; wiping
  `build/` removes everything the examples ever produced.
- The drivers are demonstrations, not benchmarks; the benchmark drivers
  live in `benchmarks/`.
