"""Stage-11 WP2 fixtures: planetmodel 3D model files and their readings.

The TestViscoelastic3D body (prem8 elastic deck, 1e21 Pa s on the solid
layers within [3.4799e6, 6.1511e6] m) written as model files the io/
adapter reads back:

  stage11_viscosity_log.nc     the happy path: the radial viscosity with
                               an analytic zero-mean log perturbation
                               delta = A g(r) sin^2(theta) cos(2 phi),
                               g(r) = (r / R)^2 (what conversion writes);
  stage11_viscosity_mean.nc    the same with a nonzero lateral mean,
                               delta = (C0 + A sin^2 cos 2phi) g(r), which
                               splits the two reference choices;
  stage11_viscosity_full3d.nc  the viscosity stored fully 3D (no
                               perturbation): eta = ETA exp(delta);
  stage11_viscosity_radial.nc  the radial viscosity alone.

Beside each laterally varying file, a probe table of the PYTHON's own
reading at deterministic off-grid points (SI): the C++ reading must
match it to rounding, both ends interpolating the same stored numbers
by the same stated rules.

Run from the planetmodel checkout:

    cd ~/dev/planetmodel && poetry run python \
        ~/dev/sl3dpp/scripts/make_stage11_fixtures.py

Output (committed): tests/data/stage11_viscosity_*.nc and
tests/data/stage11_probes_*.txt
"""
from __future__ import annotations

import pathlib

import numpy as np
from planetmodel import (AnalyticField, DeckFormat, Geometry, Model,
                        RadialMesh, constant_field, deck_layers,
                        gauss_legendre, perturbed, read_deck)
from planetmodel.modelfile import read_model, write_model

DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"
ELASTIC = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta"),
                     name="elastic")
FIELDS = ["rho", "vpv", "vph", "vsv", "vsh", "eta", "viscosity"]
# The prem8 deck's surface radius: the EarthModel's length scale, so the
# C++ tests' g(r_nd) = r_nd^2 is exactly this file's profile.
R = 6.368e6
ETA = 1e21
A = 0.4
C0 = 0.3
NPROBES = 40


def g(r):
    rn = np.asarray(r) / R
    return rn * rn


def delta_zero_mean(r, theta, phi):
    return A * g(r) * np.sin(theta) ** 2 * np.cos(2.0 * phi)


def delta_with_mean(r, theta, phi):
    return (C0 + A * np.sin(theta) ** 2 * np.cos(2.0 * phi)) * g(r)


def base_model() -> tuple[Model, list[int]]:
    """The TestViscoelastic3D body: prem8, 1e21 Pa s on the mantle."""
    deck = read_deck(DATA / "prem8.elastic.deck", ELASTIC)
    skeleton, layers = deck_layers(deck, kind="cubic")
    model = Model(Geometry(skeleton), layers)
    chosen = []
    b = skeleton.boundaries
    for layer in model.layers:
        i = layer.index
        fluid = np.all(np.asarray(layer["vsv"](
            np.linspace(*layer.interval, 3))) == 0.0)
        if not fluid and b[i] >= 3.4799e6 and b[i + 1] <= 6.1511e6:
            model = model.with_field(i, "viscosity", constant_field(
                layer.interval, ETA, name="viscosity"))
            chosen.append(i)
    return model, chosen


def probes(model: Model, chosen: list[int], seed: int) -> np.ndarray:
    """Deterministic off-grid points inside the viscous layers."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(NPROBES):
        i = chosen[rng.randint(len(chosen))]
        lo, hi = model.layer(i).interval
        margin = 0.01 * (hi - lo)
        r = lo + margin + (hi - lo - 2.0 * margin) * rng.uniform()
        theta = 0.1 + (np.pi - 0.2) * rng.uniform()
        phi = 2.0 * np.pi * rng.uniform()
        rows.append((r, theta, phi))
    return np.asarray(rows)


def dump(path: pathlib.Path, back: Model, chosen: list[int],
         points: np.ndarray, source: str) -> None:
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# stage-11 WP2 probes of {source}: the Python's own "
                  "reading, SI\n")
        out.write(f"# ETA {ETA:.17e} A {A:.17e} C0 {C0:.17e} R {R:.17e}\n")
        out.write("# r theta phi eta\n")
        for r, theta, phi in points:
            i = next(i for i in chosen
                     if back.layer(i).interval[0] <= r <= back.layer(i).interval[1])
            eta = float(np.asarray(back.layer(i)["viscosity"](r, theta, phi)))
            out.write(f"{r:.17e} {theta:.17e} {phi:.17e} {eta:.17e}\n")


def miniapp(out: pathlib.Path) -> None:
    """The viscosity3d mini-app's pipeline on the prem8 body: S40RTS on
    the viscous layers, the sl3d conversion, the viscosity written as a
    log perturbation of the radial one at band 96 on 50 km elements,
    bicubic — the file WP2's done line names, kept out of the suite for
    its size (~40 MB; the gate skips without it)."""
    from planetmodel import datasets
    from planetmodel.conversion import building, parameters

    model, chosen = base_model()
    model = datasets.transplanted(model, "S40RTS", remove_mean=True,
                                  layers=chosen)
    dt = building.temperature_anomaly(
        building.velocity_anomaly(model, layers=chosen),
        scaling=parameters.SL3D_DENSITY_SCALING,
        expansivity=parameters.SL3D_EXPANSIVITY)
    model = building.with_viscosity(model, dt, parameters.SL3D_VISCOSITY)
    lateral = {n: ("polynomial", "cubic") for n in ("vsv", "vsh", "viscosity")}
    write_model(out, model, grid=gauss_legendre(96),
                radial=RadialMesh(model.geometry, drmax=50e3, ngll=5),
                fields=FIELDS, interpolation=lateral)
    dump(out.with_suffix(".probes.txt"), read_model(out), chosen,
         probes(model, chosen, seed=19), out.name)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--miniapp":
        miniapp(pathlib.Path(sys.argv[2]))
        return
    model, chosen = base_model()
    span = (model.layer(chosen[0]).interval[0],
            model.layer(chosen[-1]).interval[1])
    print("viscous layers:", chosen, "on", span)
    radial = RadialMesh(model.geometry, drmax=400e3, ngll=5)
    rules = {"viscosity": ("polynomial", "cubic")}

    # The happy path: a zero-mean log perturbation of the radial model.
    log_model = perturbed(
        model, "viscosity", AnalyticField(span, delta_zero_mean),
        kind="log", layers=chosen)
    path = DATA / "stage11_viscosity_log.nc"
    write_model(path, log_model, grid=gauss_legendre(32), radial=radial,
                fields=FIELDS, interpolation=rules)
    dump(DATA / "stage11_probes_log.txt", read_model(path), chosen,
         probes(model, chosen, seed=11), path.name)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")

    # A perturbation of nonzero lateral mean: the reference choices split.
    mean_model = perturbed(
        model, "viscosity", AnalyticField(span, delta_with_mean),
        kind="log", layers=chosen)
    path = DATA / "stage11_viscosity_mean.nc"
    write_model(path, mean_model, grid=gauss_legendre(16),
                radial=RadialMesh(model.geometry, drmax=800e3, ngll=5),
                fields=FIELDS, interpolation=rules)
    dump(DATA / "stage11_probes_mean.txt", read_model(path), chosen,
         probes(model, chosen, seed=13), path.name)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")

    # The viscosity stored fully 3D, the same body as the happy path.
    full = model
    for i in chosen:
        full = full.with_field(i, "viscosity", AnalyticField(
            full.layer(i).interval,
            lambda r, theta, phi: ETA * np.exp(delta_zero_mean(r, theta, phi)),
            name="viscosity"), replace=True)
    path = DATA / "stage11_viscosity_full3d.nc"
    write_model(path, full, grid=gauss_legendre(32), radial=radial,
                fields=FIELDS, interpolation=rules)
    dump(DATA / "stage11_probes_full3d.txt", read_model(path), chosen,
         probes(model, chosen, seed=17), path.name)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")

    # The radial viscosity alone.
    path = DATA / "stage11_viscosity_radial.nc"
    write_model(path, model, grid=gauss_legendre(8),
                radial=RadialMesh(model.geometry, drmax=800e3, ngll=5),
                fields=FIELDS)
    print(f"wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
