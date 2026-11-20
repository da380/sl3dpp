"""The time-step study of the sl3d program: the kernels of its two
adjoints, the discrete and the continuous, held against the derivative of
the problem continuous in time, which forward runs alone give.

What it shows. (1) Both kernels converge at second order in the step to
the same limit, the derivative of the problem continuous in time: their
directional derivatives, in ln(viscosity) uniformly and layer by layer
and along a change of the ice, fall by four at each halving of the step,
and each adjoint's Richardson limit is the truth to within the truth's
uncertainty. (2) At a given step their errors differ, for a reason. The
discrete kernel is the exact gradient of the stepped forward run (its
pairings are that run's own central differences, to the differences'
truncation), so it carries the forward run's error and nothing else; it
reads larger against the derivative than the value's error reads against
the value because the viscosity's part of a value is a small part of it,
and in the measurement's own unit, metres, the two errors are of a size.
The continuous adjoint is a second problem: each measurement forces it
impulsively, after which it relaxes on the model's shortest relaxation
time, which the smoothly forced forward run never has to resolve, and its
kernel is a time quadrature of the forward and adjoint fields, each
discretised on its own. Its error is set by the step over the shortest
relaxation time (sl3d kernel states the ratio), is largest in the layers
that relax fastest, and follows the ratio when the viscosity is scaled.

The problem: the synthetic deglaciation (data/synthetic_earth.nc, its ice
history a slice each kiloyear) from 16 to 6 ka on the radial viscoelastic
Earth of data/prem8.gia.deck (a 1e21 Pa s Maxwell mantle in six layers,
its Maxwell times 0.11 kyr at the base of the mantle to 0.43 kyr at its
top), at lmax 16, with frozen shorelines and no rotational feedback, and
the sea level averaged over a 12 degree cap at Hudson Bay measured at
10 ka (mid-run) and 6 ka (the end). N is the number of steps over the
10 kyr window; every N run gives steps of 1 kyr or less dividing a
kiloyear, so that each run has a node at every knot of the ice history
and at both measurement dates, and every run steps the same forcing.

The truth, from forward runs (sl3d solve) alone: the measured values,
read from the sea level the runs write (its mean over the cap, as sl3d
kernel reads it: the field's expansion weighted by the cap's degree
weights, at the centre); their derivatives in ln(viscosity), uniformly
and in each viscous layer, as central differences of runs with the
viscosity multiplied by exp(+eps) and exp(-eps), eps = 1e-3; and their
derivative along an ice change (a 12 degree Gaussian cap of 100 m of ice
at Hudson Bay, its amplitude a hat from 13 to 11 ka peaking at 12 ka) as
the run under that ice alone, the problem being linear in the ice under
frozen shorelines. Each is Richardson-extrapolated in the step from its
two finest step counts (the finer plus a third of their difference), its
uncertainty the change from the extrapolation of the next coarser pair
(with two step counts only, the extrapolation's own correction). The
difference step lies in the flat region: at N = 10, central differences
at eps = 1e-1 to 1e-6 are held against the discrete kernel, that run's
exact gradient.

The runs, each a config written into configs/ beside the other example
configs and writing into out/<config>/ as they do, are named
timestep_[<deck>_]<kind>_n<N>:

    et, be           sl3d solve by the exponential trapezoid (second
                     order) and by backward Euler (first)
    discrete,        sl3d kernel (the exponential trapezoid): the ice and
    continuous       viscosity kernels of the two measurements by each
                     adjoint
    sdirk2,          the full form alone: sl3d solve by Sdirk2 (second
    sdirk2_discrete, order, damped, two solves a step), and sl3d kernel's
    sdirk2_continuous
                     kernels under it by each adjoint: the discrete one
                     the exact transpose of the two-stage step, the
                     continuous one stepped by the same stages; at the
                     coarsest step sdirk2_up, sdirk2_down and sdirk2_pulse,
                     the differences and the ice change under Sdirk2, which
                     the discrete kernel is held against
    up, down         sl3d solve with ln eta raised and lowered by eps in
                     every viscous layer; layer<l>_up, layer<l>_down in
                     the deck's layer l alone (sl3d's numbering, from the
                     centre); eps<k>_up, eps<k>_down by 10^-k
    pulse            sl3d solve under the ice change alone
    widths           sl3d kernel (discrete) of the sea level at 6 ka over
                     caps of the widths WIDTHS
    x4_..., x0.25_...
                     the kernels and the uniform differences on the deck
                     with its viscosity, and every Maxwell time, scaled by
                     4 and by 1/4

and the decks they need are written into data/ as prem8_<deck>_<kind>
.gia.deck (prem8_up.gia.deck, prem8_x4.gia.deck, ...). The study runs
N = 10 to 320 (1 to 1/32 kyr), its truth from N = 80, 160 and 320; on the
scaled decks the kernels at N = 10 to 80, their truths from N = 80, 160
and 320 (x4) and 320, 640 and 1280 (x0.25); Sdirk2's forward runs and
both its kernels at N = 10 to 320, against the same truth, beside the
trapezoid's. --short runs N = 10, 20 and
40 with the uniform and the ice derivatives, its truth from N = 80 and
160: the configs examples/configs holds and run_all.sh runs. Run from
build/examples:

    poetry run python timestep_study.py [--short] [--jobs J]
                                        [--write-only | --report-only]

It writes the decks and the configs, runs each config through
../../apps/sl3d from configs/ (SL3D names another program;
OMP_NUM_THREADS, 4 unless set, its threads; --jobs J runs J at once), and
prints the truth and every error against it. On four threads the short
form takes about 10 s; the study, 142 runs, about nine and a half minutes
on two threads with the runs one at a time. plot_app.py draws its figures, from whatever of these runs
exist, into figures/sl3d/timestep_study/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

CONFIGS = Path("configs")
OUT = CONFIGS / "out"
DATA = Path("data")
DECK = DATA / "prem8.gia.deck"
ICE = "../data/synthetic_earth.nc"
KYR = 1.0e3 * 365.25 * 86400.0  # s: sl3d's kiloyear
START, END = 16.0, 6.0  # kyr BP: the window
DATES = (10.0, 6.0)  # kyr BP: the measurement dates
SITE = {"latitude": 60, "longitude": 275, "width_deg": 12}  # Hudson Bay
MEASUREMENTS = {f"hudson_bay_{date:g}ka": date for date in DATES}
SCHEMES = {"et": "exponential_trapezoid", "be": "backward_euler"}
ADJOINTS = ("discrete", "continuous")
# the full form's Sdirk2 series: the forward run and the kernels by each
# adjoint; and, at the coarsest step, the runs its discrete kernel is held
# against as the stepped run's own gradient
DAMPED = {"sdirk2": "sdirk2", "sdirk2_discrete": "sdirk2",
          "sdirk2_continuous": "sdirk2"}
DAMPED_CHECKS = {"sdirk2_up": "sdirk2", "sdirk2_down": "sdirk2",
                 "sdirk2_pulse": "sdirk2"}
EPS = 3  # the difference step, 10^-EPS in ln eta
EPS_SWEEP = (1, 2, 3, 4, 5, 6)  # at N = EPS_SWEEP_STEPS
EPS_SWEEP_STEPS = 10
PULSE = {"cap_thickness": {"latitude": 60, "longitude": 275, "width_deg": 12,
                           "thickness_m": 100},
         "history_kyr_bp": [[13.0, 0.0], [12.0, 1.0], [11.0, 0.0]]}
WIDTHS = (12, 16, 20, 24)  # degrees: caps lmax 16 resolves (>= 11.25)
SCALES = {"x4": 4.0, "x0.25": 0.25}


@dataclass(frozen=True)
class Form:
    """A form of the study: the step counts of the runs compared, the
    truth's, and what else it runs."""
    steps: tuple[int, ...]
    truth: tuple[int, ...]
    layers: bool  # the per-layer differences
    extras: bool  # the difference-step sweep, the widths, the scaled decks
    scaled_steps: tuple[int, ...] = ()
    scaled_truth: dict = field(default_factory=dict)


FULL = Form(steps=(10, 20, 40, 80, 160, 320), truth=(80, 160, 320),
            layers=True, extras=True, scaled_steps=(10, 20, 40, 80),
            scaled_truth={"x4": (80, 160, 320), "x0.25": (320, 640, 1280)})
SHORT = Form(steps=(10, 20, 40), truth=(80, 160), layers=False, extras=False)


def step_kyr(steps: int) -> float:
    """The step of a run of `steps` steps over the window, kyr."""
    return (START - END) / steps


def name(kind: str, steps: int, deck: str | None = None) -> str:
    """A run's config name."""
    return f"timestep_{deck + '_' if deck else ''}{kind}_n{steps}"


# ------------------------------------------------------------ the decks
@dataclass(frozen=True)
class Layer:
    """A layer of a deck: its index from the centre (sl3d's numbering),
    radii (m), viscosity (Pa s, nan where elastic or fluid) and shortest
    Maxwell time over its knots (kyr, the Voigt shear modulus)."""
    index: int
    bottom: float
    top: float
    viscosity: float
    tau_kyr: float


def deck_rows(path: Path):
    """A Maxwell deck's header lines, rows (as text) and each row's layer
    index (a repeated radius starts a layer)."""
    lines = path.read_text().splitlines()
    head, body = lines[:2], [line for line in lines[2:] if line.strip()]
    layer, previous, index = 0, None, []
    for line in body:
        r = float(line.split()[0])
        if previous is not None and r == previous:
            layer += 1
        index.append(layer)
        previous = r
    return head, body, index


@lru_cache(maxsize=None)
def deck_layers(path: Path = DECK) -> tuple[Layer, ...]:
    """A Maxwell deck's layers."""
    _, body, index = deck_rows(path)
    data = np.array([[float(w) for w in line.split()] for line in body])
    r, rho, vpv, vsv, vph, vsh, eta, viscosity = data.T
    big_a, big_c = rho * vph**2, rho * vpv**2
    big_l, big_n = rho * vsv**2, rho * vsh**2
    big_f = eta * (big_a - 2.0 * big_l)
    mu = (big_a + big_c - 2.0 * big_f + 5.0 * big_n + 6.0 * big_l) / 15.0
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = viscosity / mu / KYR
    index = np.array(index)
    out = []
    for k in range(index.max() + 1):
        rows = index == k
        v = viscosity[rows]
        if np.all(np.isfinite(v)) and np.ptp(v) > 0.0:
            raise ValueError(f"{path}: layer {k} has more than one viscosity")
        out.append(Layer(k, float(r[rows].min()), float(r[rows].max()),
                         float(v[0]), float(np.min(tau[rows]))))
    return tuple(out)


def viscous_layers(path: Path = DECK) -> list[Layer]:
    """The layers of a deck that relax, from the centre."""
    return [layer for layer in deck_layers(path)
            if np.isfinite(layer.viscosity)]


def regions(form: Form) -> list[str]:
    """The regions of ln eta the form differentiates in."""
    out = ["uniform"]
    if form.layers:
        out += [f"layer{layer.index}" for layer in viscous_layers()]
    return out


def deck_name(tag: str) -> Path:
    """The deck of a tag: DECK for none, else data/prem8_<tag>.gia.deck."""
    return DECK if not tag else DATA / f"prem8_{tag}.gia.deck"


def write_deck(tag: str, scale: float = 1.0, eps: int | None = None,
               region: str = "uniform", sign: int = 1) -> Path:
    """Writes DECK with its viscosity times `scale`, and times
    exp(sign 10^-eps) in a region (`uniform`, or `layer<l>`), into
    data/prem8_<tag>.gia.deck: the viscosity column rewritten, the rest as
    the deck has it."""
    head, body, index = deck_rows(DECK)
    layer = None if region == "uniform" else int(region.removeprefix("layer"))
    factor = 1.0 if eps is None else float(np.exp(sign * 10.0**-eps))
    rows = []
    for line, k in zip(body, index):
        words = line.split()
        f = scale * (factor if layer is None or k == layer else 1.0)
        if words[7] != "nan" and f != 1.0:
            words[7] = repr(float(words[7]) * f)
            line = line[:line.rstrip().rfind(" ") + 1] + words[7]
        rows.append(line)
    what = [] if scale == 1.0 else [f"viscosity times {scale:g}"]
    if eps is not None:
        what.append(f"ln viscosity {'+' if sign > 0 else '-'}1e-{eps} in "
                    + ("every viscous layer" if layer is None
                       else f"layer {layer}"))
    path = deck_name(tag)
    path.write_text("\n".join([head[0] + "; " + ", ".join(what)
                               + " (timestep_study.py)", head[1]] + rows)
                    + "\n")
    return path


def fd_kind(region: str, sign: int, eps: int = EPS) -> str:
    """The kind of a difference run: `up` or `down`, after its region
    where that is a layer (`layer<l>_up`) and its difference step where
    that is not EPS (`eps<k>_up`)."""
    return ("" if region == "uniform" else f"{region}_") + (
        "" if eps == EPS else f"eps{eps}_") + ("up" if sign > 0 else "down")


def fd_deck(tag: str | None, region: str, sign: int, eps: int = EPS) -> str:
    """The tag of a difference run's deck: its kind, after the scaled
    deck's tag."""
    return f"{tag + '_' if tag else ''}{fd_kind(region, sign, eps)}"


# ----------------------------------------------------------- the configs
def kernel_kind(kind: str) -> bool:
    """Whether a kind's runs are sl3d kernel's."""
    return kind in ADJOINTS or kind in ("widths", "sdirk2_discrete",
                                        "sdirk2_continuous")


def config(run: str, steps: int, kind: str, deck: Path = DECK) -> dict:
    """The config of a run: a solve (kind "et", "be", "sdirk2", "pulse"
    or a difference's, fd_kind, and Sdirk2's own of DAMPED_CHECKS) or a
    kernel run ("discrete", "continuous", "sdirk2_discrete",
    "sdirk2_continuous", "widths"), on a deck."""
    out = {
        "problem": "quasi_static",
        "model": {"deck": f"../{deck.as_posix()}", "format": "maxwell"},
        "discretisation": {"lmax": 16},
        "times": {"start_kyr_bp": START, "end_kyr_bp": END,
                  "nodes": steps + 1},
        "forcing": {"ice_change": ([PULSE] if kind.endswith("pulse") else
                                   [{"thickness": {"file": ICE}}])},
        "ocean": {"at": "start", "topography": {"file": ICE},
                  "ice": {"file": ICE}},
        "rotation": "none",
        "solver": {"scheme": {**SCHEMES, **DAMPED, **DAMPED_CHECKS}.get(
            kind, "exponential_trapezoid")},
    }
    if kernel_kind(kind):
        if kind == "widths":
            measurements = [
                {"type": "relative_sea_level_cap", "name": f"width_{w}",
                 **SITE, "width_deg": w, "date_kyr_bp": END} for w in WIDTHS]
        else:
            measurements = [
                {"type": "relative_sea_level_cap", "name": measurement,
                 **SITE, "date_kyr_bp": date}
                for measurement, date in MEASUREMENTS.items()]
        out["kernels"] = {
            "method": ("continuous" if kind.endswith("continuous")
                       else "discrete"),
            "parameters": (["ice", "viscosity"] if deck == DECK
                           and kind != "widths" else ["viscosity"]),
            "measurements": measurements,
            "output": f"out/{run}"}
    else:
        out["output"] = {"surface": f"out/{run}/{run}.nc"}
    return out


WIDTH = 78


def inline(value, col: int) -> str:
    """A JSON value starting at column `col`, on one line where it fits,
    else its members filled into lines aligned after its bracket."""
    text = json.dumps(value)
    if (col + len(text) + 2 <= WIDTH or not isinstance(value, (dict, list))
            or not value):
        return text
    if isinstance(value, dict):
        pieces = [(json.dumps(k) + ": ", v) for k, v in value.items()]
        out, close = "{", "}"
    else:
        pieces = [("", v) for v in value]
        out, close = "[", "]"
    at = col + 1
    for k, (prefix, v) in enumerate(pieces):
        sep = ", " if k else ""
        if k and at + len(sep) + len(prefix + json.dumps(v)) + 2 > WIDTH:
            out, at, sep = out + ",\n" + " " * (col + 1), col + 1, ""
        piece = prefix + inline(v, at + len(sep) + len(prefix))
        out += sep + piece
        tail = piece.rsplit("\n", 1)
        at = at + len(sep) + len(piece) if len(tail) == 1 else len(tail[1])
    return out + close


def text(value: dict) -> str:
    """A config in the examples' layout: a top-level member to a line, a
    block that would run past the line's end a member to a line, and in
    the kernels block a measurement's type and name on one line and the
    rest on the next."""
    lines = []
    items = list(value.items())
    for k, (key, member) in enumerate(items):
        comma = "," if k < len(items) - 1 else ""
        head = f'  "{key}": '
        if key == "kernels":
            rows = []
            for m in member["measurements"]:
                parts = [f"{json.dumps(a)}: {json.dumps(b)}"
                         for a, b in m.items()]
                rows.append("      {" + ", ".join(parts[:2]) + ",\n       "
                            + ", ".join(parts[2:]) + "}")
            body = [f'    "{a}": {json.dumps(b)}' for a, b in member.items()
                    if a != "measurements"]
            body.insert(2, '    "measurements": [\n' + ",\n".join(rows)
                        + "\n    ]")
            lines.append(head + "{\n" + ",\n".join(body) + f"\n  }}{comma}")
        elif len(head) + len(json.dumps(member)) + len(comma) <= WIDTH:
            lines.append(head + json.dumps(member) + comma)
        else:
            body = [f'    "{a}": ' + inline(b, 6 + len(a))
                    for a, b in member.items()]
            lines.append(head + "{\n" + ",\n".join(body) + f"\n  }}{comma}")
    return "{\n" + "\n".join(lines) + "\n}\n"


@dataclass(frozen=True)
class Run:
    """One run of the study: its config's name, the sl3d command and the
    config."""
    name: str
    command: str
    config: dict


def runs(form: Form) -> list[Run]:
    """The runs of a form of the study, writing the decks they need."""
    out: dict[str, Run] = {}

    def add(kind: str, steps: int, deck_tag: str | None = None,
            deck: Path = DECK) -> None:
        run = name(kind, steps, deck_tag)
        command = "kernel" if kernel_kind(kind) else "solve"
        out.setdefault(run, Run(run, command, config(run, steps, kind, deck)))

    def differences(levels, region: str, eps: int = EPS,
                    tag: str | None = None, scale: float = 1.0) -> None:
        for sign in (1, -1):
            kind = fd_kind(region, sign, eps)
            deck = write_deck(fd_deck(tag, region, sign, eps), scale, eps,
                              region, sign)
            for n in levels:
                add(kind, n, tag, deck)

    for n in form.steps:
        for kind in ("et", "be", *ADJOINTS):
            add(kind, n)
    for n in form.truth:
        add("et", n)
    for n in sorted(set(form.steps) | set(form.truth)):
        add("pulse", n)
    differences(sorted(set(form.truth) | {form.steps[0]}
                       | (set(form.steps) if form.extras else set())),
                "uniform")
    for region in regions(form)[1:]:
        differences(form.truth, region)
    if form.extras:
        for eps in EPS_SWEEP:
            if eps != EPS:
                differences((EPS_SWEEP_STEPS,), "uniform", eps)
        for n in form.steps:
            if n <= 160:
                add("widths", n)
            for kind in DAMPED:
                add(kind, n)
        # Sdirk2's own differences and ice change at the coarsest step
        add("sdirk2_up", form.steps[0], None,
            deck_name(fd_deck(None, "uniform", 1)))
        add("sdirk2_down", form.steps[0], None,
            deck_name(fd_deck(None, "uniform", -1)))
        add("sdirk2_pulse", form.steps[0])
        for tag, scale in SCALES.items():
            deck = write_deck(tag, scale)
            for n in form.scaled_steps:
                for kind in ADJOINTS:
                    add(kind, n, tag, deck)
            differences(form.scaled_truth[tag], "uniform", tag=tag,
                        scale=scale)
    return list(out.values())


def write(run: Run) -> Path:
    """Writes a run's config into configs/."""
    path = CONFIGS / f"{run.name}.json"
    path.write_text(text(run.config))
    return path


def execute(runs_: list[Run], jobs: int) -> None:
    """Runs each config through sl3d, `jobs` at once, saying each one's
    time; stops at the first failure."""
    sl3d = os.environ.get("SL3D", "../../apps/sl3d")
    env = {**os.environ,
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}

    def one(run: Run) -> str:
        begin = time.perf_counter()
        done = subprocess.run([sl3d, run.command, f"{run.name}.json"],
                              cwd=CONFIGS, env=env, capture_output=True,
                              text=True)
        if done.returncode:
            raise RuntimeError(f"{run.name}: sl3d {run.command} failed\n"
                               + done.stdout + done.stderr)
        return f"ran {run.name} ({time.perf_counter() - begin:.1f} s)"

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for line in pool.map(one, runs_):
            print(line, flush=True)


# ------------------------------------------------------- reading the runs
def run_dir(run: str) -> Path:
    """A run's output directory."""
    return OUT / run


@lru_cache(maxsize=None)
def surface(run: str):
    """A solve run's dates (kyr BP), sea-level change by node, angular grid
    and area weights (summing to 4 pi)."""
    from planetmodel.surfacefile import read_surface_file

    file = read_surface_file(run_dir(run) / f"{run}.nc")
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    values = np.asarray(file.variables["relative_sea_level_change"].values)
    grid = file.grid
    weights = (np.asarray(grid.weights)[:, None]
               * np.full(len(grid.longitudes), 2.0 * np.pi
                         / len(grid.longitudes))[None, :])
    return dates, values, grid, weights


@lru_cache(maxsize=None)
def cap_weights(width_deg: float, lmax: int) -> np.ndarray:
    """The degree weights lambda_l of a Gaussian cap of standard deviation
    `width_deg` (docs/sl3d-configs.md, Smoothing), by Gauss-Legendre
    quadrature in the distance."""
    x, w = np.polynomial.legendre.leggauss(400)
    d = 0.5 * np.pi * (x + 1.0)
    s = np.deg2rad(width_deg)
    weight = 0.5 * np.pi * w * np.exp(-d**2 / (2.0 * s**2)) * np.sin(d)
    legendre = np.polynomial.legendre.legvander(np.cos(d), lmax)
    return (legendre * weight[:, None]).sum(axis=0) / weight.sum()


def node(dates: np.ndarray, date: float) -> int:
    """The index of the node at a date."""
    k = int(np.argmin(np.abs(dates - date)))
    if abs(dates[k] - date) > 1e-9:
        raise ValueError(f"no node at {date:g} ka")
    return k


def cap_mean(run: str, date: float, width_deg: float = SITE["width_deg"]
             ) -> float:
    """A solve run's sea-level change at a date averaged over a Gaussian
    cap about the site, as sl3d kernel's relative_sea_level_cap reads it:
    the field smoothed over the cap (its expansion weighted by the cap's
    degree weights), at the centre."""
    from planetmodel.harmonics import analyse_grid, real_harmonics

    dates, values, grid, _ = surface(run)
    coefficients = analyse_grid(values[node(dates, date)], grid)
    lmax = coefficients.shape[1] - 1
    basis = real_harmonics(lmax, np.deg2rad(90.0 - SITE["latitude"]),
                           np.deg2rad(SITE["longitude"]))
    return float(np.sum(coefficients * basis
                        * cap_weights(width_deg, lmax)[None, :, None]))


def value(run: str, measurement: str) -> float:
    """A solve run's value of a measurement, m."""
    return cap_mean(run, MEASUREMENTS[measurement])


def stated_values(run: str) -> dict[str, float]:
    """A kernel run's measured values (measurements.txt), by name."""
    out = {}
    for line in (run_dir(run) / "measurements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            words = line.split()
            out[words[0]] = float(words[2])
    return out


def stated_ratio(run: str) -> tuple[float, float] | None:
    """A continuous kernel run's statement of its longest step over the
    model's shortest relaxation time, and that time (kyr), from its
    measurements.txt."""
    path = run_dir(run) / "measurements.txt"
    for line in path.read_text().splitlines():
        found = re.search(r"its longest step (\S+) times the shortest "
                          r"relaxation time \((\S+) kyr", line)
        if found:
            return float(found.group(1)), float(found.group(2))
    return None


@lru_cache(maxsize=None)
def viscosity_kernel(run: str, measurement: str):
    """A kernel run's radial viscosity kernel: the rows' radii (m), K
    (per unit volume) and volume weights W (m^3), and each row's deck
    layer (rows listed element by element, so its element's)."""
    data = np.loadtxt(run_dir(run) / f"{measurement}_viscosity_kernel.txt")
    r, k, w = data[:, 0], data[:, 1], data[:, 2]
    starts = np.flatnonzero(np.r_[True, np.diff(r) == 0.0])
    layer = np.empty(r.size, dtype=int)
    bounds = [(lay.bottom, lay.top, lay.index) for lay in deck_layers()]
    for a, b in zip(starts, np.r_[starts[1:], r.size]):
        middle = 0.5 * (r[a] + r[b - 1])
        layer[a:b] = next(i for lo, hi, i in bounds if lo <= middle <= hi)
    return r, k, w, layer


def viscosity_pairings(run: str, measurement: str, deck: Path = DECK
                       ) -> dict[str, float]:
    """A kernel run's derivatives of a measurement in ln eta: uniform, and
    in each viscous layer, sum W K eta over the region's rows (m per unit
    of ln eta)."""
    _, k, w, layer = viscosity_kernel(run, measurement)
    eta = {lay.index: lay.viscosity for lay in viscous_layers(deck)}
    density = np.array([w_ * k_ * eta.get(l_, 0.0)
                        for w_, k_, l_ in zip(w, k, layer)])
    out = {f"layer{i}": float(density[layer == i].sum()) for i in eta}
    out["uniform"] = float(density.sum())
    return out


def pulse_field(latitudes, longitudes, dates) -> np.ndarray:
    """The ice direction's thickness at each node and grid point, m."""
    cap = PULSE["cap_thickness"]
    lat, lon = np.deg2rad(latitudes)[:, None], np.deg2rad(longitudes)[None, :]
    lat0, lon0 = np.deg2rad(cap["latitude"]), np.deg2rad(cap["longitude"])
    cosd = (np.sin(lat) * np.sin(lat0)
            + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0))
    d = np.arccos(np.clip(cosd, -1.0, 1.0))
    width = np.deg2rad(cap["width_deg"])
    shape = cap["thickness_m"] * np.exp(-d**2 / (2.0 * width**2))
    history = np.array(PULSE["history_kyr_bp"])[::-1]  # increasing dates
    amplitude = np.interp(dates, history[:, 0], history[:, 1])
    return amplitude[:, None, None] * shape[None, :, :]


@lru_cache(maxsize=None)
def ice_pairing(run: str, measurement: str) -> float:
    """A kernel run's derivative of a measurement along the ice direction:
    sum over the nodes of the thickness kernel's integral against the
    direction's thickness there, m."""
    import netCDF4

    with netCDF4.Dataset(run_dir(run) / f"{measurement}_ice_kernels.nc") as f:
        kernel = np.asarray(f["thickness_dual"][:])
        weights = np.asarray(f["colatitude_weights"][:])
        latitudes = np.asarray(f["latitude"][:])
        longitudes = np.asarray(f["longitude"][:])
        dates = -np.asarray(f["time"][:]) / KYR
        radius = float(f.getncattr("meta_radius_m"))
    area = radius**2 * weights[:, None] * 2.0 * np.pi / len(longitudes)
    return float(np.sum(kernel * pulse_field(latitudes, longitudes, dates)
                        * area[None, :, :]))


def exists(run: str) -> bool:
    """Whether a run's output is there."""
    directory = run_dir(run)
    return ((directory / f"{run}.nc").exists()
            or (directory / "measurements.txt").exists())


# --------------------------------------------------------------- results
def richardson(coarse, fine):
    """The second-order extrapolation from a step and its half."""
    return fine + (fine - coarse) / 3.0


def extrapolated(series: dict[int, float], levels) -> tuple[float, float]:
    """The truth from a quantity at the truth levels' step counts: the
    Richardson extrapolation of the finest two, and its uncertainty, the
    change from that of the next coarser two (with two levels, its own
    correction)."""
    have = [n for n in levels if n in series]
    if len(have) < 2:
        raise LookupError("the truth's runs are missing")
    best = richardson(series[have[-2]], series[have[-1]])
    if len(have) >= 3:
        error = abs(best - richardson(series[have[-3]], series[have[-2]]))
    else:
        error = abs(series[have[-1]] - series[have[-2]]) / 3.0
    return best, error


def truth_present(form: Form) -> bool:
    """Whether a form's truth has run: the solves, the uniform
    differences and the ice pulse at its truth's step counts."""
    return all(exists(name(kind, n)) for n in form.truth
               for kind in ("et", "pulse", fd_kind("uniform", 1),
                            fd_kind("uniform", -1)))


def form_present() -> Form:
    """The fullest form whose truth has run."""
    return FULL if truth_present(FULL) else SHORT


def central(eps: int, region: str, steps: int, measurement: str,
            tag: str | None = None, width_deg: float = SITE["width_deg"]
            ) -> float | None:
    """The central difference of the forward runs in ln eta, m per unit,
    where both runs exist."""
    up = name(fd_kind(region, 1, eps), steps, tag)
    down = name(fd_kind(region, -1, eps), steps, tag)
    if not (exists(up) and exists(down)):
        return None
    date = MEASUREMENTS[measurement]
    return ((cap_mean(up, date, width_deg) - cap_mean(down, date, width_deg))
            / (2.0 * 10.0**-eps))


def present(kind: str, steps, tag: str | None = None) -> list[int]:
    """The step counts among `steps` at which a kind's run exists."""
    return [n for n in steps if exists(name(kind, n, tag))]


def results(form: Form | None = None) -> dict | None:
    """Everything the study measures, from the runs present, or None when
    the truth's runs are missing:

        truth[m]               value, its field (by node date), the
                               derivatives (uniform, layers, pulse), each
                               (best, uncertainty)
        steps                  the step counts compared
        ratio[n]               h / tau_min as sl3d kernel states it
        tau_min                the shortest relaxation time, kyr
        value[m][n]            et's value
        field[scheme][m][n]    the sea level's relative L2 error
        derivative[adjoint][m][region|"pulse"][n]   the pairing
        exact[m][region|"pulse"][n]   discrete pairing over the forward
                                      difference at the same step, less 1
        eps[m][k]              forward difference at 10^-k over the
                               discrete pairing at N = EPS_SWEEP_STEPS,
                               less 1
        layers                 the viscous layers
        limits[adjoint][m][region]   the pairings' Richardson limit from
                               the finest two steps, its uncertainty
                               from the next coarser pair
        widths[w][n]           the widths study: the discrete pairing's
                               error at 6 ka, and its truth
        scaled[tag]            the scaled decks: ratio[n], error[adjoint][m][n]
        damped                 the full form's Sdirk2 series (empty
                               without its runs): value[m][n],
                               field[m][n] and the continuous kernels'
                               derivative[m][region|"pulse"][n]
    """
    form = form or form_present()
    if not truth_present(form):
        return None
    out: dict = {"form": form, "steps": [
        n for n in form.steps
        if all(exists(name(kind, n)) for kind in ("et", *ADJOINTS))]}
    out["layers"] = viscous_layers()
    region_list = ["uniform"] + [f"layer{lay.index}" for lay in out["layers"]]
    truth: dict = {}
    for m, date in MEASUREMENTS.items():
        t: dict = {}
        t["value"] = extrapolated({n: value(name("et", n), m)
                                   for n in form.truth}, form.truth)
        fields = {}
        for n in form.truth:
            dates, values, _, _ = surface(name("et", n))
            fields[n] = values[node(dates, date)]
        t["field"] = richardson(fields[form.truth[-2]], fields[form.truth[-1]])
        for region in region_list:
            series = {n: central(EPS, region, n, m) for n in form.truth}
            series = {n: v for n, v in series.items() if v is not None}
            if len(series) >= 2:
                t[region] = extrapolated(series, form.truth)
        t["pulse"] = extrapolated({n: value(name("pulse", n), m)
                                   for n in form.truth}, form.truth)
        truth[m] = t
    out["truth"] = truth
    _, _, _, weights = surface(name("et", form.truth[-1]))

    def l2(x):
        return float(np.sqrt(np.sum(weights * x**2)))

    out["value"] = {m: {n: value(name("et", n), m) for n in out["steps"]}
                    for m in MEASUREMENTS}
    out["value_be"] = {m: {n: value(name("be", n), m) for n in out["steps"]
                           if exists(name("be", n))} for m in MEASUREMENTS}
    out["field"] = {}
    for scheme in SCHEMES:
        out["field"][scheme] = {}
        for m, date in MEASUREMENTS.items():
            row = {}
            for n in out["steps"]:
                if exists(name(scheme, n)):
                    dates, values, _, _ = surface(name(scheme, n))
                    row[n] = (l2(values[node(dates, date)] - truth[m]["field"])
                              / l2(truth[m]["field"]))
            out["field"][scheme][m] = row
    out["derivative"] = {}
    for adjoint in ADJOINTS:
        out["derivative"][adjoint] = {}
        for m in MEASUREMENTS:
            row: dict = {region: {} for region in region_list + ["pulse"]}
            for n in out["steps"]:
                pairs = viscosity_pairings(name(adjoint, n), m)
                for region in region_list:
                    row[region][n] = pairs[region]
                row["pulse"][n] = ice_pairing(name(adjoint, n), m)
            out["derivative"][adjoint][m] = row
    out["ratio"], out["tau_min"] = {}, None
    for n in out["steps"]:
        stated = stated_ratio(name("continuous", n))
        if stated:
            out["ratio"][n], out["tau_min"] = stated
    out["exact"] = {}
    for m in MEASUREMENTS:
        row = {region: {} for region in region_list + ["pulse"]}
        for n in out["steps"]:
            d = out["derivative"]["discrete"][m]
            for region in region_list:
                fd = central(EPS, region, n, m)
                if fd is not None:
                    row[region][n] = d[region][n] / fd - 1.0
            if exists(name("pulse", n)):
                row["pulse"][n] = (d["pulse"][n] / value(name("pulse", n), m)
                                   - 1.0)
        out["exact"][m] = row
    out["eps"] = {}
    n0 = EPS_SWEEP_STEPS
    for m in MEASUREMENTS:
        row = {}
        if n0 in out["steps"]:
            exact = out["derivative"]["discrete"][m]["uniform"][n0]
            for k in EPS_SWEEP:
                fd = central(k, "uniform", n0, m)
                if fd is not None:
                    row[k] = fd / exact - 1.0
        out["eps"][m] = row
    out["limits"] = {}
    for adjoint in ADJOINTS:
        out["limits"][adjoint] = {}
        for m in MEASUREMENTS:
            levels = out["steps"][-3:]
            out["limits"][adjoint][m] = {
                region: extrapolated(series, levels)
                for region, series in out["derivative"][adjoint][m].items()
                if len(levels) >= 2}
    out["widths"] = widths_study(form)
    out["scaled"] = scaled_study(form)
    out["damped"] = damped_study(out, region_list, l2)
    return out


def damped_study(result: dict, region_list: list[str], l2) -> dict:
    """The Sdirk2 series at the steps compared where its runs are present:
    the values, the sea level's relative L2 error, and each adjoint's
    kernels' pairings; and, at the coarsest step, the discrete kernel's
    pairings against the stepped run's own derivatives (`gradient`: the
    central difference of Sdirk2 runs in ln eta and the run under the ice
    change alone)."""
    truth = result["truth"]
    steps = [n for n in result["steps"]
             if all(exists(name(kind, n)) for kind in DAMPED)]
    if not steps:
        return {}
    out: dict = {"steps": steps, "value": {}, "field": {}, "derivative": {},
                 "discrete": {}, "gradient": {}}
    for m, date in MEASUREMENTS.items():
        out["value"][m] = {n: value(name("sdirk2", n), m) for n in steps}
        out["field"][m] = {}
        for n in steps:
            dates, values, _, _ = surface(name("sdirk2", n))
            out["field"][m][n] = (l2(values[node(dates, date)]
                                     - truth[m]["field"])
                                  / l2(truth[m]["field"]))
        for key, kind in (("derivative", "sdirk2_continuous"),
                          ("discrete", "sdirk2_discrete")):
            row: dict = {region: {} for region in region_list + ["pulse"]}
            for n in steps:
                run = name(kind, n)
                pairs = viscosity_pairings(run, m)
                for region in region_list:
                    row[region][n] = pairs[region]
                row["pulse"][n] = ice_pairing(run, m)
            out[key][m] = row
        n = steps[0]
        up, down, pulse = (name(kind, n) for kind in DAMPED_CHECKS)
        if all(exists(run) for run in (up, down, pulse)):
            out["gradient"][m] = {
                "uniform": (out["discrete"][m]["uniform"][n],
                            (cap_mean(up, date) - cap_mean(down, date))
                            / (2.0 * 10.0**-EPS)),
                "pulse": (out["discrete"][m]["pulse"][n],
                          cap_mean(pulse, date))}
    return out


def widths_study(form: Form) -> dict:
    """The widths study: at each cap width, the truth of the 6 ka cap
    mean's uniform derivative (from the uniform differences, read at that
    width) and the discrete kernel's pairing at each step."""
    out = {}
    steps = present("widths", [n for n in form.steps if n <= 160])
    if not steps or not form.extras:
        return out
    for w in WIDTHS:
        series = {n: central(EPS, "uniform", n, f"hudson_bay_{END:g}ka",
                             width_deg=w) for n in form.truth}
        series = {n: v for n, v in series.items() if v is not None}
        if len(series) < 2:
            continue
        best, error = extrapolated(series, form.truth)
        pairs = {n: viscosity_pairings(name("widths", n), f"width_{w}")
                 ["uniform"] for n in steps}
        out[w] = {"truth": (best, error), "pairing": pairs}
    return out


def scaled_study(form: Form) -> dict:
    """The decks with the viscosity scaled: each one's truth of the
    uniform derivatives and the kernels' pairings, with the stated
    ratios."""
    out = {}
    for tag, scale in SCALES.items():
        deck = deck_name(tag)
        levels = form.scaled_truth.get(tag, ())
        if not deck.exists() or not levels:
            continue
        entry: dict = {"scale": scale, "truth": {}, "derivative": {},
                       "ratio": {}}
        complete = True
        for m in MEASUREMENTS:
            series = {n: central(EPS, "uniform", n, m, tag) for n in levels}
            series = {n: v for n, v in series.items() if v is not None}
            if len(series) < 2:
                complete = False
                break
            entry["truth"][m] = extrapolated(series, levels)
            for adjoint in ADJOINTS:
                entry["derivative"].setdefault(adjoint, {})[m] = {
                    n: viscosity_pairings(name(adjoint, n, tag), m,
                                          deck)["uniform"]
                    for n in present(adjoint, form.scaled_steps, tag)}
        if not complete:
            continue
        for n in present("continuous", form.scaled_steps, tag):
            stated = stated_ratio(name("continuous", n, tag))
            if stated:
                entry["ratio"][n], entry["tau_min"] = stated
        out[tag] = entry
    return out


# ---------------------------------------------------------------- report
def orders(steps, errors) -> str:
    """The observed orders between successive steps."""
    return " ".join(f"{np.log(abs(a / b)) / np.log(2):.2f}"
                    for a, b in zip(errors, errors[1:]) if a and b)


def report_truth(result: dict) -> None:
    """Prints the truth, its uncertainty and the difference step."""
    truth = result["truth"]
    levels = ", ".join(map(str, result["form"].truth))
    print(f"\nTHE TRUTH, from forward runs at N = {levels} (Richardson; "
          "its uncertainty in brackets)")
    for m in MEASUREMENTS:
        t = truth[m]
        (j, dj), (u, du), (q, dq) = t["value"], t["uniform"], t["pulse"]
        print(f"  {m}: J {j:.6f} m [{dj:.0e}]; dJ/d ln eta {u:.6f} m "
              f"[{du:.0e}, {du / abs(u):.0e} of it], {abs(u / j):.1%} of J; "
              f"along the ice pulse {q:.6f} m [{dq:.0e}]")
        layered = [lay for lay in result["layers"]
                   if f"layer{lay.index}" in t]
        if layered:
            parts = [f"{lay.index} ({lay.tau_kyr:.3f} kyr) "
                     f"{t[f'layer{lay.index}'][0]:.5f} "
                     f"[{t[f'layer{lay.index}'][1]:.0e}]" for lay in layered]
            total = sum(t[f"layer{lay.index}"][0] for lay in layered)
            print("    by layer (shortest Maxwell time), m: "
                  + "; ".join(parts) + f"; their sum less the uniform "
                  f"{total - u:.1e}")
    if any(len(row) > 1 for row in result["eps"].values()):
        print(f"\nTHE DIFFERENCE STEP at N = {EPS_SWEEP_STEPS}: the central "
              "difference at eps over the discrete pairing, less 1")
        for m, row in result["eps"].items():
            print(f"  {m}: " + "  ".join(f"1e-{k}: {e:+.1e}"
                                         for k, e in row.items()))
    print("\nTHE DISCRETE KERNEL IS THE STEPPED RUN'S GRADIENT: its pairing "
          f"over the central difference (eps 1e-{EPS}) of runs at the same "
          "N, less 1")
    for m, row in result["exact"].items():
        for region, series in row.items():
            if series:
                print(f"  {m} {region:8s} " + "  ".join(
                    f"N={n}: {e:+.1e}" for n, e in series.items()))


def report_errors(result: dict) -> None:
    """Prints the errors against the truth by step, their orders and
    the successive changes."""
    truth, steps = result["truth"], result["steps"]
    print("\nERRORS AGAINST THE TRUTH, relative (in metres in brackets); "
          f"h / tau_min as sl3d kernel states it, tau_min {result['tau_min']}"
          " kyr")
    for m in MEASUREMENTS:
        t = truth[m]
        value = {n: result["value"][m][n] - t["value"][0] for n in steps}
        be = {n: result["value_be"][m][n] - t["value"][0] for n in steps}
        error = {a: {key: {n: result["derivative"][a][m][key][n]
                           - t[key][0] for n in steps}
                     for key in ("uniform", "pulse")} for a in ADJOINTS}
        print(f"  {m}\n    {'h kyr':>7} {'h/tau':>5} | {'value, et':>18} "
              f"{'be':>9} | {'field et':>8} {'be':>8} | "
              f"{'dJ/d ln eta, discrete':>21} {'continuous':>19} | "
              f"{'ice disc':>9} {'cont':>9}")
        for n in steps:
            d, c = error["discrete"], error["continuous"]
            u, q = t["uniform"][0], t["pulse"][0]
            ratio = result["ratio"].get(n, np.nan)
            print(f"    {step_kyr(n):7.5g} {ratio:5.2g} | "
                  f"{value[n] / t['value'][0]:+9.2e} ({value[n]:+.3f}) "
                  f"{be[n] / t['value'][0]:+9.2e} | "
                  f"{result['field']['et'][m][n]:8.2e} "
                  f"{result['field']['be'][m][n]:8.2e} | "
                  f"{d['uniform'][n] / u:+9.2e} ({d['uniform'][n]:+.3f}) "
                  f"{c['uniform'][n] / u:+9.2e} ({c['uniform'][n]:+.3f}) | "
                  f"{d['pulse'][n] / q:+9.2e} {c['pulse'][n] / q:+9.2e}")
        series = [("value et", value)] + [
            (f"{key} {a}", error[a][key]) for key in ("uniform", "pulse")
            for a in ADJOINTS]
        print("    observed orders: " + "; ".join(
            f"{label} {orders(steps, [e[n] for n in steps])}"
            for label, e in series))
        for adjoint in ADJOINTS:
            p = result["derivative"][adjoint][m]["uniform"]
            changes = [abs(p[a] - p[b]) for a, b in zip(steps, steps[1:])]
            print(f"    {adjoint}: the uniform pairing's change at each "
                  "halving, m, and each change over the next: "
                  + "  ".join(f"{a}->{b} {c:.2e}" for a, b, c
                              in zip(steps, steps[1:], changes))
                  + " | " + " ".join(f"{x / y:.2f}" for x, y
                                     in zip(changes, changes[1:])))


def report_layers(result: dict) -> None:
    """Prints the per-layer errors, the limits, the widths and the scaled
    decks."""
    truth, steps = result["truth"], result["steps"]
    first = next(iter(MEASUREMENTS))
    layered = [lay for lay in result["layers"]
               if f"layer{lay.index}" in truth[first]]
    if layered:
        print("\nBY LAYER: the pairing's relative error against the truth")
        for m in MEASUREMENTS:
            print(f"  {m:>23}, layer " + " ".join(
                f"{lay.index:>9d}" for lay in layered))
            print(f"  {'shortest Maxwell time, kyr':>30} " + " ".join(
                f"{lay.tau_kyr:9.3f}" for lay in layered))
            for adjoint in ADJOINTS:
                row = result["derivative"][adjoint][m]
                for n in steps:
                    errors = [row[f"layer{lay.index}"][n]
                              / truth[m][f"layer{lay.index}"][0] - 1.0
                              for lay in layered]
                    print(f"  {adjoint:>21} h={step_kyr(n):<7.5g} "
                          + " ".join(f"{e:+9.2e}" for e in errors))
    levels = " and ".join(map(str, steps[-2:]))
    print(f"\nTHE LIMITS: the Richardson extrapolation of each adjoint's "
          f"pairings from N = {levels}, relative to the truth (in brackets "
          "the difference over the two's uncertainties summed)")
    for adjoint in ADJOINTS:
        for m in MEASUREMENTS:
            parts = []
            for region, (lim, error) in result["limits"][adjoint][m].items():
                if region in truth[m]:
                    best, uncertainty = truth[m][region]
                    share = (lim - best) / (error + uncertainty)
                    parts.append(f"{region} {lim / best - 1:+.1e} "
                                 f"({share:+.2f})")
            print(f"  {adjoint:10s} {m}: " + "  ".join(parts))
    if result["widths"]:
        print(f"\nTHE CAP'S WIDTH: the discrete pairing's relative error at "
              f"{END:g} ka against each width's truth")
        for w, entry in result["widths"].items():
            best = entry["truth"][0]
            print(f"  {w:3d} deg (dJ/d ln eta {best:.4f} m): " + "  ".join(
                f"h={step_kyr(n):g} {p / best - 1:+.2e}"
                for n, p in entry["pairing"].items()))
    for entry in result["scaled"].values():
        print(f"\nTHE VISCOSITY TIMES {entry['scale']:g} (tau_min "
              f"{entry.get('tau_min')} kyr): the uniform pairing's relative "
              "error")
        for m in MEASUREMENTS:
            best, uncertainty = entry["truth"][m]
            print(f"  {m} (dJ/d ln eta {best:.5f} m [{uncertainty:.0e}])")
            discrete = entry["derivative"]["discrete"][m]
            continuous = entry["derivative"]["continuous"][m]
            for n in discrete:
                print(f"    h={step_kyr(n):<6.4g} "
                      f"h/tau={entry['ratio'].get(n, np.nan):<5.3g} "
                      f"discrete {discrete[n] / best - 1:+.2e}  "
                      f"continuous {continuous[n] / best - 1:+.2e}")


def report_damped(result: dict) -> None:
    """Prints the Sdirk2 series' errors against the truth beside the
    trapezoid's at the same steps."""
    damped = result["damped"]
    if not damped:
        return
    truth, steps = result["truth"], damped["steps"]
    first = next(iter(MEASUREMENTS))
    layered = [lay for lay in result["layers"]
               if f"layer{lay.index}" in truth[first]]
    print("\nSDIRK2 (solver.scheme \"sdirk2\"): relative errors against the "
          "truth, the exponential trapezoid's at the same step after the "
          "slash")
    for m in MEASUREMENTS:
        t = truth[m]
        d = damped["derivative"][m]
        e = damped["discrete"][m]
        c = result["derivative"]["continuous"][m]
        x = result["derivative"]["discrete"][m]
        print(f"  {m}\n    {'h kyr':>7} {'h/tau':>5} | {'value':>19} | "
              f"{'field':>17} | {'dJ/d ln eta, discrete':>19} | "
              f"{'continuous':>19} | {'ice, discrete':>19} | "
              f"{'continuous':>19}")
        for n in steps:
            ratio = result["ratio"].get(n, np.nan)
            print(f"    {step_kyr(n):7.5g} {ratio:5.2g} | "
                  f"{damped['value'][m][n] / t['value'][0] - 1:+9.2e}/"
                  f"{result['value'][m][n] / t['value'][0] - 1:+9.2e} | "
                  f"{damped['field'][m][n]:8.2e}/"
                  f"{result['field']['et'][m][n]:8.2e} | "
                  f"{e['uniform'][n] / t['uniform'][0] - 1:+9.2e}/"
                  f"{x['uniform'][n] / t['uniform'][0] - 1:+9.2e} | "
                  f"{d['uniform'][n] / t['uniform'][0] - 1:+9.2e}/"
                  f"{c['uniform'][n] / t['uniform'][0] - 1:+9.2e} | "
                  f"{e['pulse'][n] / t['pulse'][0] - 1:+9.2e}/"
                  f"{x['pulse'][n] / t['pulse'][0] - 1:+9.2e} | "
                  f"{d['pulse'][n] / t['pulse'][0] - 1:+9.2e}/"
                  f"{c['pulse'][n] / t['pulse'][0] - 1:+9.2e}")
        if m in damped["gradient"]:
            g = damped["gradient"][m]
            print(f"    the discrete kernel against the stepped run's own "
                  f"derivative at h = {step_kyr(steps[0]):g}: dJ/d ln eta "
                  f"{g['uniform'][0] / g['uniform'][1] - 1:+.1e}, ice "
                  f"{g['pulse'][0] / g['pulse'][1] - 1:+.1e}")
        series = [("value", {n: damped["value"][m][n] - t["value"][0]
                             for n in steps}),
                  ("dJ/d ln eta, discrete",
                   {n: e["uniform"][n] - t["uniform"][0] for n in steps}),
                  ("continuous", {n: d["uniform"][n] - t["uniform"][0]
                                  for n in steps}),
                  ("ice, discrete", {n: e["pulse"][n] - t["pulse"][0]
                                     for n in steps}),
                  ("continuous", {n: d["pulse"][n] - t["pulse"][0]
                                  for n in steps})]
        print("    observed orders: " + "; ".join(
            f"{label} {orders(steps, [e[n] for n in steps])}"
            for label, e in series))
        for label, rows in (("discrete", e), ("continuous", d)):
            if not layered:
                continue
            print(f"    by layer, {label}, " + " ".join(
                f"{lay.index:>9d}" for lay in layered))
            for n in steps:
                errors = [rows[f"layer{lay.index}"][n]
                          / t[f"layer{lay.index}"][0] - 1.0 for lay in layered]
                print(f"    {'h=' + format(step_kyr(n), '.5g'):>24} "
                      + " ".join(f"{err:+9.2e}" for err in errors))


def report(result: dict) -> None:
    """Prints the truth and the errors against it."""
    report_truth(result)
    report_errors(result)
    report_layers(result)
    report_damped(result)
    stated = max(abs(stated_values(name("discrete", n))[m]
                     / result["value"][m][n] - 1.0)
                 for n in result["steps"] for m in MEASUREMENTS)
    print(f"\nthe kernel runs' values (measurements.txt) are the solve "
          f"runs' cap means to {stated:.0e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--short", action="store_true",
                        help="the short form: N = 10, 20, 40 against "
                             "a truth from 80 and 160")
    parser.add_argument("--jobs", type=int, default=1,
                        help="runs at once (default 1)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-only", action="store_true",
                       help="write the decks and configs and stop")
    group.add_argument("--report-only", action="store_true",
                       help="print the report from the runs present")
    args = parser.parse_args()
    form = SHORT if args.short else FULL
    if not args.report_only:
        todo = runs(form)
        for run in todo:
            write(run)
        print(f"wrote {len(todo)} configs into {CONFIGS}/")
        if args.write_only:
            return
        execute(todo, args.jobs)
    result = results(form)
    if result is None:
        print("the truth's runs are missing")
        return
    report(result)


if __name__ == "__main__":
    main()
