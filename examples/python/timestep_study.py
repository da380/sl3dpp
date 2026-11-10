"""The time-stepping accuracy of the sl3d program's runs through time: one
fixed problem run at several step sizes, the sea level by both time
schemes and the viscosity kernels by both adjoints, each held against a
converged reference.

The problem: the synthetic deglaciation (data/synthetic_earth.nc) from
16 ka to 6 ka on the radial viscoelastic Earth of data/prem8.gia.deck (a
1e21 Pa s Maxwell mantle, its Maxwell times, viscosity over shear
modulus, from 0.11 to 0.43 kyr), at lmax 16, without rotational feedback
(so that the viscoelastic memory is all the time step acts on), and two
measurements of the sea level averaged over a 4 degree cap at Hudson Bay:
at 10 ka, mid-run, and at 6 ka, the end. Every step size divides the
10 kyr window into a whole number of steps that puts a node at 10 ka, so
the mid-run measurement is at the same date in every run.

Each step size N (the number of steps over the window) is four configs,
written into configs/ beside the other example configs, each writing into
out/<config>/ as they do:

    timestep_et_n<N>.json, timestep_be_n<N>.json      sl3d sea-level, the
        exponential trapezoid (second order) and backward Euler (first)
    timestep_discrete_n<N>.json, timestep_continuous_n<N>.json
        sl3d kernel1d, the exponential trapezoid, the discrete adjoint (the
        exact gradient of the time-stepped model) and the continuous one

and the reference is the finest step's timestep_et_n<N>.json and
timestep_discrete_n<N>.json. The study runs N = 5, 10, 20, 40, 80 (steps
of 2 to 1/8 kyr) against N = 320 (1/32 kyr), with the continuous kernel at
N = 320 too, to show the two adjoints' agreement there; --short runs
N = 10, 20, 40 against N = 160, the configs that examples/configs holds
and run_all.sh runs. Run from build/examples:

    poetry run python timestep_study.py [--short] [--write-only]

It writes the configs, runs each through ../../apps/sl3d from configs/
(SL3D names another program; OMP_NUM_THREADS, 4 unless set, its threads),
and prints the errors: for the sea level, the relative L2 norm over the
sphere of the difference from the reference at each measurement date; for
the kernels, of the difference of the radial viscosity kernel from the
reference's, in the volume-weighted L2 norm (K is a density per unit
volume, dJ = 4 pi int K(r) delta eta(r) r^2 dr, so the norm is
(int K^2 r^2 dr)^(1/2)); and for the measurements' values. plot_app.py
draws them, from whatever of these runs exist, into
figures/sl3d/timestep_study/accuracy.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

CONFIGS = Path("configs")
OUT = CONFIGS / "out"
START, END, MID = 16.0, 6.0, 10.0  # kyr BP: the window and the mid-run date
STEPS = [5, 10, 20, 40, 80]  # steps over the window
REFERENCE = 320
SHORT_STEPS, SHORT_REFERENCE = [10, 20, 40], 160
SITE = {"latitude": 60, "longitude": 275, "width_deg": 4}  # Hudson Bay
MEASUREMENTS = {f"hudson_bay_{date:g}ka": date for date in (MID, END)}
SCHEMES = {"et": "exponential_trapezoid", "be": "backward_euler"}
ADJOINTS = ("discrete", "continuous")
NAME = re.compile(r"timestep_(et|be|discrete|continuous)_n(\d+)$")


def step_kyr(steps: int) -> float:
    """The step length of a run of `steps` steps over the window."""
    return (START - END) / steps


def config(kind: str, steps: int) -> dict:
    """The config of one run: kind "et" or "be" (sea-level, by that
    scheme) or "discrete" or "continuous" (kernel1d, by that adjoint)."""
    name = f"timestep_{kind}_n{steps}"
    if (START - MID) * steps % (START - END):
        raise ValueError(f"{steps} steps put no node at {MID:g} ka")
    out = {
        "model": {"deck": "../data/prem8.gia.deck", "format": "maxwell"},
        "lmax": 16,
        "times": {"start_kyr_bp": START, "end_kyr_bp": END,
                  "nodes": steps + 1},
        "forcing": {"state": {"file": "../data/synthetic_earth.nc"},
                    "ice": {"file": "../data/synthetic_earth.nc"}},
        "rotation": "none",
        "solver": {"scheme": SCHEMES.get(kind, "exponential_trapezoid")},
    }
    if kind in SCHEMES:
        out["output"] = {"surface": f"out/{name}/{name}.nc"}
    else:
        out["adjoint"] = kind
        out["measurements"] = [
            {"type": "sea_level_cap", "name": measurement, **SITE,
             "date_kyr_bp": date}
            for measurement, date in MEASUREMENTS.items()]
        out["output"] = f"out/{name}"
    return out


def members(value: dict) -> list[str]:
    """A JSON object's members, each as `"key": value` text."""
    return [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()]


def write(kind: str, steps: int) -> Path:
    """Writes a run's config into configs/, in the examples' layout: a
    block that would run past 78 columns has a member to a line, and a
    measurement its type and name on one line and the rest on the next."""
    c = config(kind, steps)
    path = CONFIGS / f"timestep_{kind}_n{steps}.json"
    lines = ["{"]
    items = list(c.items())
    for k, (key, value) in enumerate(items):
        comma = "," if k < len(items) - 1 else ""
        text = f'  "{key}": {json.dumps(value)}{comma}'
        if key == "measurements":
            rows = []
            for m in value:
                parts = members(m)
                rows.append("    {" + ", ".join(parts[:2]) + ",\n     "
                            + ", ".join(parts[2:]) + "}")
            text = f'  "{key}": [\n' + ",\n".join(rows) + f"\n  ]{comma}"
        elif isinstance(value, dict) and len(text) > 78:
            text = (f'  "{key}": {{\n    ' + ",\n    ".join(members(value))
                    + f"\n  }}{comma}")
        lines.append(text)
    path.write_text("\n".join(lines + ["}"]) + "\n")
    return path


def family(short: bool) -> list[tuple[str, int]]:
    """The runs of the study, (kind, steps): every kind at each step, and
    the reference's sea-level and discrete-kernel runs."""
    steps, reference = ((SHORT_STEPS, SHORT_REFERENCE) if short
                        else (STEPS, REFERENCE))
    runs = [(kind, n) for n in steps
            for kind in ("et", "be", "discrete", "continuous")]
    runs += [("et", reference), ("discrete", reference)]
    if not short:  # the two adjoints' agreement at the reference step
        runs.append(("continuous", reference))
    return runs


def compared(reference: int, present) -> list[int]:
    """The step counts held against a reference: the family's own (the
    study's or the short one's) where the reference is one of theirs, else
    every coarser run present."""
    family_steps = {REFERENCE: STEPS, SHORT_REFERENCE: SHORT_STEPS}
    return [n for n in family_steps.get(reference, sorted(present))
            if n in present and n < reference]


# ------------------------------------------------------------ the errors
def runs_present() -> dict[str, dict[int, Path]]:
    """The study's runs whose outputs exist, by kind and step count."""
    found: dict[str, dict[int, Path]] = {}
    for directory in sorted(OUT.glob("timestep_*")):
        match = NAME.match(directory.name)
        if not match:
            continue
        kind, steps = match.group(1), int(match.group(2))
        done = ((directory / f"{directory.name}.nc").exists() if kind in SCHEMES
                else (directory / "measurements.txt").exists())
        if done:
            found.setdefault(kind, {})[steps] = directory
    return found


def sea_level_at(directory: Path):
    """A sea-level run's sea-level change at the measurement dates, and the
    grid's area weights."""
    from planetmodel.surfacefile import read_surface_file

    file = read_surface_file(directory / f"{directory.name}.nc")
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    values = np.asarray(file.variables["sea_level_change"].values)
    weights = np.asarray(file.grid.weights)[:, None] * np.ones(
        len(file.grid.longitudes))[None, :]
    picks = {}
    for measurement, date in MEASUREMENTS.items():
        k = int(np.argmin(np.abs(dates - date)))
        if abs(dates[k] - date) > 1e-9:
            raise ValueError(f"{directory.name} has no node at {date:g} ka")
        picks[measurement] = values[k]
    return picks, weights


def kernel(directory: Path, measurement: str):
    """A kernel1d run's radial viscosity kernel: radius and K(r)."""
    data = np.loadtxt(directory / f"{measurement}_viscosity_kernel.txt")
    return data[:, 0], data[:, 1]


def values(directory: Path) -> dict[str, float]:
    """A kernel1d run's measured values, by name."""
    out = {}
    for line in (directory / "measurements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            words = line.split()
            out[words[0]] = float(words[2])
    return out


def errors():
    """The relative errors of every run present against the finest run of
    its kind, as {quantity: {series: (steps, errors)}} with the quantities
    "sea_level" (series "<scheme> <measurement>"), "kernel" ("<adjoint>
    <measurement>") and "value" (each scheme's measured values), and the
    reference step counts. None when no reference has run."""
    found = runs_present()
    if "et" not in found or "discrete" not in found:
        return None
    ref_fields = max(found["et"])
    ref_kernels = max(found["discrete"])
    reference, weights = sea_level_at(found["et"][ref_fields])
    ref_k = {m: kernel(found["discrete"][ref_kernels], m) for m in MEASUREMENTS}
    ref_v = values(found["discrete"][ref_kernels])

    def l2(x, w):
        return np.sqrt(np.sum(w * x**2))

    out = {"sea_level": {}, "kernel": {}, "value": {}}
    for scheme in SCHEMES:
        present = found.get(scheme, {})
        for steps in compared(ref_fields, present):
            fields, _ = sea_level_at(present[steps])
            for m in MEASUREMENTS:
                e = l2(fields[m] - reference[m], weights) / l2(reference[m],
                                                              weights)
                out["sea_level"].setdefault(f"{scheme} {m}", ([], []))
                out["sea_level"][f"{scheme} {m}"][0].append(steps)
                out["sea_level"][f"{scheme} {m}"][1].append(e)
    for adjoint in ADJOINTS:
        present = found.get(adjoint, {})
        for steps in compared(ref_kernels, present):
            directory = present[steps]
            for m in MEASUREMENTS:
                r, k = kernel(directory, m)
                r0, k0 = ref_k[m]
                if r.shape != r0.shape or np.max(np.abs(r - r0)) > 1e-6:
                    raise ValueError(f"{directory.name}: another radial mesh")
                e = np.sqrt(np.trapezoid((k - k0) ** 2 * r**2, r)
                            / np.trapezoid(k0**2 * r**2, r))
                out["kernel"].setdefault(f"{adjoint} {m}", ([], []))
                out["kernel"][f"{adjoint} {m}"][0].append(steps)
                out["kernel"][f"{adjoint} {m}"][1].append(e)
            if adjoint == "discrete":
                v = values(directory)
                for m in MEASUREMENTS:
                    out["value"].setdefault(f"et {m}", ([], []))
                    out["value"][f"et {m}"][0].append(steps)
                    out["value"][f"et {m}"][1].append(
                        abs(v[m] - ref_v[m]) / abs(ref_v[m]))
    agreement = {}  # the continuous kernel against the discrete at its step
    if ref_kernels in found.get("continuous", {}):
        for m in MEASUREMENTS:
            r, k = kernel(found["continuous"][ref_kernels], m)
            r0, k0 = ref_k[m]
            agreement[m] = np.sqrt(np.trapezoid((k - k0) ** 2 * r**2, r)
                                   / np.trapezoid(k0**2 * r**2, r))
    out["agreement"] = agreement
    return out, (ref_fields, ref_kernels)


def maxwell_times_kyr(deck=Path("data/prem8.gia.deck")) -> tuple[float, float]:
    """The shortest and longest Maxwell time (viscosity over shear modulus)
    of a Maxwell deck's viscous layers, kyr."""
    data = np.genfromtxt(deck, skip_header=2)
    rho, vsv, vsh, eta = data[:, 1], data[:, 3], data[:, 5], data[:, 7]
    viscous = np.isfinite(eta) & (vsv > 0.0)
    mu = rho * (2.0 * vsv**2 + vsh**2) / 3.0
    tau = eta[viscous] / mu[viscous] / 3.15576e10
    return float(tau.min()), float(tau.max())


def report() -> None:
    """Prints the errors by step, and the observed orders."""
    result = errors()
    if result is None:
        print("no reference runs yet")
        return
    table, (ref_fields, ref_kernels) = result
    print(f"against the sea level at {step_kyr(ref_fields):g} kyr steps and "
          f"the discrete kernel at {step_kyr(ref_kernels):g} kyr steps")
    agreement = table.pop("agreement")
    for quantity, series in table.items():
        for label, (steps, errs) in series.items():
            dt = [step_kyr(n) for n in steps]
            orders = [np.log(errs[k] / errs[k + 1]) / np.log(dt[k] / dt[k + 1])
                      for k in range(len(errs) - 1)]
            print(f"{quantity:9s} {label:32s} "
                  + "  ".join(f"{d:g}: {e:.2e}" for d, e in zip(dt, errs))
                  + "   orders " + " ".join(f"{p:.2f}" for p in orders))
    for m, e in agreement.items():
        print(f"at {step_kyr(ref_kernels):g} kyr steps the continuous kernel of "
              f"{m} is the discrete one's to {e:.1e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--short", action="store_true",
                        help="the CI family: N = 10, 20, 40 against 160")
    parser.add_argument("--write-only", action="store_true",
                        help="write the configs and stop")
    args = parser.parse_args()
    sl3d = os.environ.get("SL3D", "../../apps/sl3d")
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS",
                                                           "4")}
    for kind, steps in family(args.short):
        path = write(kind, steps)
        print(f"wrote {path}")
        if args.write_only:
            continue
        command = "sea-level" if kind in SCHEMES else "kernel1d"
        subprocess.run([sl3d, command, path.name], cwd=CONFIGS, env=env,
                       check=True)
    if not args.write_only:
        report()


if __name__ == "__main__":
    main()
