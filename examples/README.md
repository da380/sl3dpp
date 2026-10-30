# sl3dpp examples

Seven small drivers, one per capability of the library, each paired with a
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

The defaults keep each run under a minute on 8 threads, the times shown;
the largest peak memory is 11 GB, for `ExampleIceKernels`. The
configs, one per `sl3d` command, are described in
[configs/README.md](configs/README.md).

Every driver takes `[--lmax N] [--state FILE]`: the degree (the defaults are
sized for a laptop; CI runs each at a small one) and the Earth's state.

## The inputs

- **The synthetic Earth**, `synthetic_earth.nc`, which the build writes
  (`MakeSyntheticEarth`): a planetmodel surface file of `ice_thickness` and
  `topography` on a 1° grid over a toy glacial cycle — smooth continents on
  a 4 km deep ocean, Laurentide and Fennoscandian sheets melting away from
  21 ka and Greenland and Antarctica shrinking to their present size. It
  needs no data, so the examples run anywhere, CI included. The drivers use
  it by default.
- **ICE-7G**, the real thing, through pyslfp: `get_ice7g.py` writes
  `ice7g.nc` in the same form (below), and `--state ice7g.nc` runs any
  driver on it; `configs/*_ice7g.json` are the configs that use it.
- The PREM-like deck `prem8.elastic.deck` and, for the 3D model-file
  example and configs, `viscosity_log.nc`, a planetmodel model file of a
  3D viscosity on the same body (both copied from `tests/data`).

A state's sea level is built from its topography and ice thickness by
`SeaLevelFromTopography` (pyslfp's construction), as the `sl3d` program
does.

## Running the sl3d program on the examples' inputs

Every command of the `sl3d` program has a config in `build/examples/configs`,
set up to run on the inputs above, so the program can be tried without
writing a config from scratch:

```sh
cd build/examples/configs
../../apps/sl3d fingerprint fingerprint.json   # one command
./run_all.sh                                   # every config in turn
```

Outputs land in `configs/out/`, and `poetry run python plot_app.py`, from
`build/examples`, draws a figure per command from them.

**The configs in `build/examples/configs` are yours to edit.** The build copies
them there only when they are missing and never overwrites them, so change
the degree, the dates, the ice change or the measurements and run again;
delete a file (or the whole directory) and reconfigure to get the original
back. [configs/README.md](configs/README.md) says what each config does, and every
member is documented in the configuration reference
([docs/sl3d-configs.md](../docs/sl3d-configs.md)). One thing to know:
`evolve1d.json` writes checkpoints into `out/`, and a run refuses another
run's checkpoints, so after changing it set `"restart": false` (or delete
`out/`) for the first rerun.

## Everything runs in the build tree

At configure time CMake copies the inputs, the configs (into
`build/examples/configs`, once), the Python scripts and the poetry file into
`build/examples`, and the build writes the synthetic Earth there. The
drivers read and write in the current directory: run them from
`build/examples`, and every output — data files, figures, the poetry
environment, a downloaded ICE-7G file — lands there. The source tree is
never written to.

## Setup and run

```sh
# 1. Build (from the repository root; the examples are on by default).
cmake -B build && cmake --build build -j

# 2. Run the drivers from the build tree.
cd build/examples
./ExampleLoveNumbers
./ExampleFingerprint
./ExampleEvolution
./ExampleLateralViscosity
./ExampleModelViscosity
./ExampleIceKernels
./ExampleViscosityKernels3D

# 3. One-off: create the Python environment HERE (not in the source).
poetry install --no-root

# 4. Plot (figures land beside the data).
poetry run python plot_love_numbers.py    # and plot_fingerprint.py, ...

# 5. The sl3d configs, and their figures.
(cd configs && ./run_all.sh)
poetry run python plot_app.py
```

## ICE-7G

```sh
cd build/examples                         # from the repository root
poetry run python get_ice7g.py            # writes ice7g.nc
./ExampleFingerprint --state ice7g.nc     # any driver
(cd configs && ./run_all.sh)                  # now runs the *_ice7g.json configs too
```

`get_ice7g.py` uses pyslfp's `export_ice_history`: ICE-7G's stored slices
from 26 ka to the present (every kiloyear to 21 ka, then every half
kiloyear), `ice_thickness` and `topography` on the dataset's grid, in
metres. The first run downloads pyslfp's ICE-7G archive from Zenodo (about
24 MB) into pyslfp's data directory (`~/.pyslfp_data`, or `$PYSLFP_DATA`);
later runs reuse it. `--version ICE6G` or `ICE5G` writes those models
instead.

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
