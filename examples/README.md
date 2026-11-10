# sl3dpp examples

Eight small drivers, one per capability of the library, each paired with a
Python plotting script, and an example config for every command of the
`sl3d` program:

| Driver | What it shows | Default | Plot |
| --- | --- | --- | --- |
| `ExampleLoveNumbers` | load and tidal Love numbers of the PREM-like deck, in the conventional (Farrell) normalisation | lmax 256, 0.1 s | `plot_love_numbers.py` |
| `ExampleFingerprint` | the elastic sea-level fingerprint of a 10% Greenland melt: one solve of the generalised sea level equation with full rotational feedback, SI in and out | lmax 256, 0.8 s | `plot_fingerprint.py` |
| `ExampleEvolution` | 1D viscoelastic GIA: a West Antarctic melt ramp then hold, the elastic jump and the viscous creep after it, polar motion | lmax 192, 43 s | `plot_evolution.py` |
| `ExampleLateralViscosity` | 3D viscosity: an analytic lateral pattern of one decade against its radial twin, and the cost of the within-step iteration | lmax 48, 48 s | `plot_lateral.py` |
| `ExampleModelViscosity` | the route from real tomography: a 3D viscosity read from a planetmodel model file (`io::ModelViscosity`), run through the whole deglaciation against the file's own radial reference | lmax 32, 20 s | `plot_model_viscosity.py` |
| `ExampleIceKernels` | 1D adjoint kernels of a tide gauge near the melting Greenland ice: rate kernels of the ice history at four intervals, radial viscosity kernels by the continuous and the discrete adjoint; the forward run checkpointed | lmax 128, 36 s | `plot_kernels.py` |
| `ExampleViscosityKernels3D` | 3D adjoint kernels: the log-viscosity kernels of a sea-level and a decay-time measurement at Hudson Bay, written as a planetmodel model file (`io::WriteKernels`); the forward run checkpointed | lmax 24, 31 s | `plot_kernels_3d.py` |
| `ExampleStatic3D` | 3D elasticity: the static response of the synthetic 3D elastic Earth (`data/synthetic_moduli.nc`, lateral moduli read by `io::ModelModuli`) to a cap load over Hudson Bay, solved exactly by conjugate gradients on the radial systems, against the radial model's; the response inside the body as a model file for planetmodel's cutaways | lmax 32, 5 s | `plot_static_3d.py` |

The defaults keep each run under a minute on 8 threads, the times shown;
the largest peak memory is 11 GB, for `ExampleIceKernels`. The
configs, one per `sl3d` command, are described in
[configs/README.md](configs/README.md).

Every driver takes `[--lmax N] [--state FILE]`: the degree (the defaults are
sized for a laptop; CI runs each at a small one) and the Earth's state.

Each driver writes into its own `output/library/<example>/`, and its
plotting script draws `figures/library/<example>.png`, where `<example>` is
`love_numbers`, `fingerprint`, `evolution`, `lateral_viscosity`,
`model_viscosity`, `ice_kernels`, `viscosity_kernels_3d` or `static_3d`.
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
  of a tomography model writes, for `ExampleStatic3D` and `response.json`.
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
  `response_tide_planet.json` and `response_tide_planet_relax.json`.
- The PREM-like deck `prem8.elastic.deck` and, for the 3D model-file
  example and configs, `viscosity_log.nc`, a planetmodel model file of a
  3D viscosity on the same body (both copied from `tests/data`), and
  `prem8.gia.deck`, the same deck in planetmodel's GIA format with a
  1e21 Pa s Maxwell mantle, for the configs' radial runs through time
  (copied from `examples/data`).

A state's sea level is built from its topography and ice thickness by
`SeaLevelFromTopography` (pyslfp's construction), as the `sl3d` program
does.

## Making inputs with planetmodel and pyslfp

The `make_*.py` scripts write inputs for the `sl3d` program into `data/`,
each run from `build/examples` as `poetry run python make_<...>.py`, each
saying in its docstring (and `--help`) what it writes and how a config
uses it; `inputs.py` holds what they share.

| Script | Writes (by default) | Read by |
| --- | --- | --- |
| `make_earth_deck.py` | `earth_custom.gia.deck`, `earth_custom.elastic.deck`: PREM without its ocean, an elastic lithosphere (100 km), Maxwell upper and lower mantles (5e20 and 5e21 Pa s), isotropic where they relax | `model.deck` |
| `make_planet_deck.py` | `planet.gia.deck`, `planet.elastic.deck`: a three-layer toy Mars (fluid core, Maxwell mantle, elastic lid); prints its mass and surface gravity | `model.deck`; `response_tide_planet*.json` |
| `make_lateral_model.py` | `<deck>_<field>_<kind>.nc`: a 3D viscosity or lateral moduli on a deck's own body, an analytic pattern, S40RTS or a Gaussian random field; from S40RTS the viscosity by sl3d's conversion chain (optionally renormalised to the deck's mean, with a lid), the moduli from the shear speed and a scaled P speed (`--vp-scaling`) | `model.viscosity.file`, `model.moduli.file` |
| `make_ice_history.py` | `ice7g.nc` (ICE-5G/6G/7G through pyslfp), or a hand-made disk ice sheet from numpy arrays, `synthetic_disk.nc` and its present day, `synthetic_disk_present.nc` | `forcing.ice.file`, `forcing.state.file` |
| `make_load_field.py` | `load_disk.nc`, `load_cap.nc` or `load_greenland.nc`, a surface mass density, static or (`--history`) on the run's clock | a `field` load term of `response` |
| `make_load_coefficients.py` | `<file>_lm.txt`: a load field's coefficient table, in the exchange table's harmonics | a `coefficients` load term of `response` |
| `make_tide_history.py` | `configs/response_tide_oscillating.json`: a `response` config of a sinusoidal degree-2 tide at a chosen period | `sl3d response` |

A data set a script needs that is not on disk yet (S40RTS for
planetmodel, ICE-xG for pyslfp) is downloaded only with `--download`;
without it the script says what it would fetch and stops (`get_ice7g.py`
gives `--download` itself).

`timestep_study.py` is not an input script: it writes the time-step
study's configs into `configs/`, runs each through `sl3d` and prints the
errors against the converged reference (configs/README.md, The synthetic
Earth); `--short` is the family `configs/` holds and `run_all.sh` runs.

## Running the sl3d program on the examples' inputs

Every command of the `sl3d` program has a config in `build/examples/configs`,
set up to run on the inputs above, so the program can be tried without
writing a config from scratch:

```sh
cd build/examples/configs
../../apps/sl3d sea-level sea_level_fingerprint.json   # one command
./run_all.sh                                           # every config in turn
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
| `kernel1d` | `kernels.png` |
| `kernel` | `viscosity_kernels.png` |
| `response`, `response_viscoelastic`, `response1d_disk` | `displacement.png` |
| `response1d_tide` | `love_numbers.png` |
| `response_tide_3d` | `body_tide.png`: the radial model's tidal displacement, the 3D minus radial difference and the difference's spectrum by degree |
| `response_tide_oscillating` | `oscillating.png`: the potential and the response through the run, and the Love numbers' amplitudes and lags at its period and those of `response_tide_oscillating_short` and `_long` |
| `response_tide_planet` | `body_tide.png`: the toy planet's tidal displacement and potential, with its h2 and k2 |
| `response_tide_planet_relax` | `love_numbers.png`: the toy planet's h2 and k2 through time, beside the Earth's |
| `sea_level_fingerprint_melt` (with `response_melt`) | `ocean.png`: one melt through `response` and through `sea-level`, the displacement each way, its difference (the ocean's load) and the sea-level change |
| `response_cap_weak` (with `response_cap_strong`) | `relaxation.png`: the lateral viscosity with the two caps, and the displacement at each cap's centre through time, 3D and radial model |
| `response_load_field` (with `response_load_cap`, `response_load_coefficients`) | `agreement.png`: the forcing on the run's clock and the file's axis, the three runs' displacement, and their agreement |
| `timestep_study` (the `timestep_*` configs) | `accuracy.png`: the error against the step of the sea level by both schemes and of the viscosity kernels by both adjoints, and the kernels at 1 kyr steps |
| `sea_level_fingerprint_3d_ice7g` | `fingerprint.png`: the fingerprint on PREM with S40RTS's moduli, over the ocean and near Greenland, and its 3D minus radial difference |
| `sea_level_3d_ice7g` (with `sea_level_1d_ice7g_custom`) | `sea_level.png`: the sea-level change since 26 ka in 3D and 3D minus radial, and relative sea-level histories at three sites, 3D and radial |

`sea_level_3d`'s run is drawn against the 1D one in
`figures/sl3d/sea_level_1d/sea_level.png` (in `sea_level_3d/` when it
alone has run), so it has no folder of its own.

The tide figures show three things. On the synthetic 3D Earth the lunar
body tide is the radial model's to 2e-4: a bulge of 0.21 m, which the
lateral moduli change by 0.04 mm at most, the 3D model putting 7.0e-5 of
its rms into degrees other than 2, all of it the lateral coupling. On the
radial viscoelastic Earth an oscillating tide's response lags the
potential, most at periods between the elastic and fluid limits (24° at
1 kyr, 4° at 0.1 kyr, 12° at 10 kyr), its amplitude rising from the
elastic Love numbers towards the fluid ones as the period grows. The toy
planet's elastic Love numbers are h2 = 0.339 and k2 = 0.182, and a held
tide relaxes several times more slowly on it than on the Earth.

**The configs in `build/examples/configs` are yours to edit.** The build copies
them there only when they are missing and never overwrites them, so change
the degree, the dates, the ice change or the measurements and run again;
delete a file (or the whole directory) and reconfigure to get the original
back. [configs/README.md](configs/README.md) says what each config does, and every
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
configs into `build/examples/configs` (once), and the Python scripts and
the poetry file into `build/examples`, and the build writes the synthetic
Earths into `data/`. Everything is run from `build/examples` (the configs
from `configs/`), and every path is relative to it:

```
build/examples/
  Example*, Make*         the drivers and the synthetic-Earth writers
  *.py, pyproject.toml    the plotting scripts, get_ice7g.py, the make_*.py
                          input scripts, the poetry file
  data/                   every input: the decks, viscosity_log.nc,
                          synthetic_earth.nc, synthetic_moduli.nc, ice7g.nc
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
(cd configs && ./run_all.sh)              # now runs the *_ice7g.json configs too
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
`response_load_field.json`) needs

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
  huge pages for their large tensors (see the top-level README, Running).
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
