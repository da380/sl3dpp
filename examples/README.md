# sl3dpp examples

Four small drivers, one per major capability, each paired with a Python
plotting script:

| Driver | Feature | Plot |
| --- | --- | --- |
| `ExampleFingerprint` | elastic sea-level fingerprint (stage 2): a 10% Greenland melt on the real present-day Earth, lmax 256, one SLE solve, full rotation, SI in/out | `plot_fingerprint.py` |
| `ExampleEvolution` | 1D viscoelastic GIA (stage 3): a West Antarctic melt ramp then hold, lmax 128, elastic jump + viscous creep, polar motion | `plot_evolution.py` |
| `ExampleLateralViscosity` | 3D lateral viscosity (stage 4): a one-decade lateral pattern vs the radial twin, within-step iteration cost (kept small and analytic by design) | `plot_lateral.py` |
| `ExampleIceKernels` | adjoint kernels (stage 5): a Greenland melt observed by a Fremantle tide gauge — ice-history duals, the quelled rate kernel, radial viscosity kernels by the continuous and discrete adjoints | `plot_kernels.py` |

Inputs: the PREM-like test deck (copied from `tests/data`), and the
present-day Earth state (ICE-7G ice and topography), which the build
**extracts from a pyslfp checkout automatically** — see below. The 3D
example uses pyslfp's analytic state and needs no data at all.

## Everything runs in the build tree

At configure time CMake copies the deck, the plotting scripts and the
poetry file into `build/examples/`, and — when a pyslfp checkout is
found (a sibling `~/dev/pyslfp` by default; override with
`-DSL3DPP_PYSLFP_DIR=...`) — the build runs
`scripts/make_example_earth_state.py` there to extract the present-day
ICE-7G Earth state as spherical-harmonic coefficients
(`earth_state_l256.txt`, written straight into `build/examples`).
The drivers read and write **in the current directory**: run them from
`build/examples`, and every output — data files, figures, the poetry
lock and environment — lands there. The source tree is never written
to; don't run the scripts from `examples/plots/`.

## Setup and run

```sh
# 1. Build (from the repository root; examples are on by default).
cmake -B build && cmake --build build -j

# 2. Run the drivers from the build tree.
cd build/examples
./ExampleFingerprint        # ~a second (lmax 256)
./ExampleEvolution          # ~a minute (lmax 128, 31 steps)
./ExampleLateralViscosity   # ~a minute (lmax 16, the 3D fixed point)
./ExampleIceKernels         # ~seconds (lmax 32, two reverse sweeps)

# 3. One-off: create the plotting environment HERE (not in the source).
poetry install --no-root

# 4. Plot (figures land beside the data).
poetry run python plot_fingerprint.py
poetry run python plot_evolution.py
poetry run python plot_lateral.py
poetry run python plot_kernels.py
```

Notes:

- **cartopy** provides the map projections and coastlines; its first use
  downloads the coastline data (one-off, needs network). If cartopy is
  not importable the scripts fall back to plain lat/lon maps, so any
  environment with numpy + matplotlib works in a pinch:
  `python3 plot_fingerprint.py` with system packages is fine.
- `poetry install --no-root` resolves and installs into poetry's own
  cache (or set `POETRY_VIRTUALENVS_IN_PROJECT=1` first to keep the venv
  in `build/examples/.venv`). Either way the source tree stays clean;
  wiping `build/` removes everything the examples ever produced.
- The drivers are sized for a laptop (lmax 16–64). They are
  demonstrations, not benchmarks — the benchmark kit lives in
  `benchmarks/` and `scripts/`.
