"""The migrating-shoreline references: pyslfp's forward run with migrating
shorelines, the gradient of its final sea level misfit, and its calibration
of the initial sea level.

The stepper fixture's Maxwell body (as make_evolution_fixture.py) under
pyslfp's marine-grounded cap: a 5 km cap at 60N 85W (radius 20 degrees, a
cos^2 profile) on a 3000 m ocean with a 200 m degree-one tilt, melting
linearly from 10 ka to nothing at present, so that melting floods the
cap's footprint and flotation cells flip. The ice and the initial sea level
are band-limited and written here as coefficients, so the C++ synthesises
the same grid values. tests/TestInitialSeaLevel.cpp holds the C++ to it.

Run in this directory's environment (pyproject.toml):

    cd scripts && poetry install --no-root
    poetry run python make_migrating_fixture.py [--data-dir DIR]

Output (committed): tests/data/pyslfp_migrating.txt
"""
from __future__ import annotations

import numpy as np
from planetmodel import Geometry, Model, RadialMesh, deck_layers, read_deck, units
from planetmodel.deck import DeckFormat
from pyshtools import SHGrid
from pyslfp.core import EarthModel, EarthState
from pyslfp.ice import BaseIceModel
from pyslfp.love_numbers.material import Material
from pyslfp.love_numbers.table import LoveNumbers
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator

from fixture_tools import data_dir, provenance

LMAX = 16
ETA = 1e21
START = 10.0  # ka
NODES = 6
THICKNESS = 5000.0  # m
DEPTH = 3000.0  # m
TILT = 200.0  # m
TARGET_TILT = 50.0  # m, the misfit's target off the reached sea level
RTOL = 1e-10
ELASTIC = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta"), name="elastic")
VISCOSITY = units.MASS * units.LENGTH**-1 * units.TIME**-1


def build(data):
    deck = read_deck(data / "prem8.elastic.deck", ELASTIC)
    skeleton, layers = deck_layers(deck, kind="cubic")
    model_si = Model(Geometry(skeleton), layers)
    table = LoveNumbers.from_model(
        model_si, LMAX, mesh=RadialMesh(model_si, lmax=LMAX, ngll=5))
    earth = EarthModel(LMAX, love_numbers=table, grid="GLQ", extend=False)
    # pyslfp refuses mixed unit frames: build the radial side in the
    # earth model's scales, the viscosity among it.
    model = model_si.converted(earth.parameters.scales)
    mesh = RadialMesh(model, lmax=LMAX, ngll=5)
    material = Material(mesh, model)
    b = skeleton.boundaries
    viscosity = {}
    for i, layer in enumerate(model_si.layers):
        fluid = np.all(np.asarray(layer["vsv"](np.linspace(*layer.interval, 3)))
                       == 0.0)
        if not fluid and b[i] >= 3.4799e6 and b[i + 1] <= 6.1511e6:
            viscosity[i] = ETA / earth.parameters.scales.factor(VISCOSITY)
    op = ViscoelasticOperator(Rheology.maxwell(material, viscosity), LMAX)
    return earth, op


def band_limited(earth, values):
    grid = SHGrid.from_array(values, grid=earth.grid)
    return earth.expand_coefficient(earth.expand_field(grid))


def main() -> None:
    data = data_dir(__doc__)
    earth, op = build(data)
    metre = 1.0 / earth.parameters.length_scale
    theta = np.radians(90.0 - earth.lats())[:, None] * np.ones(earth.lons().size)
    phi = np.ones(earth.lats().size)[:, None] * np.radians(earth.lons())[None, :]
    theta0, phi0, radius = np.radians(30.0), np.radians(275.0), np.radians(20.0)
    c = (np.cos(theta) * np.cos(theta0)
         + np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0))
    d = np.arccos(np.clip(c, -1.0, 1.0))
    taper = np.where(d < radius, np.cos(0.5 * np.pi * d / radius) ** 2, 0.0)
    ice0 = band_limited(earth, THICKNESS * metre * taper)
    sea0 = band_limited(
        earth, (DEPTH + TILT * np.sin(theta) * np.cos(phi)) * metre)

    class MeltingCap(BaseIceModel):
        def get_ice_thickness(self, date, lmax, /, *, grid="DH", sampling=1,
                              extend=True):
            fraction = np.clip(date / START, 0.0, 1.0)
            return SHGrid.from_array(fraction * ice0.data, grid=earth.grid)

    ice_model = MeltingCap(length_scale=earth.parameters.length_scale)
    dates = np.linspace(START, 0.0, NODES)
    truth = EarthState(ice0, sea0, earth, exclude_caspian=False)
    evolution = SeaLevelEvolution(op, truth)
    snapshots, _ = evolution.run_ice_history(
        dates, ice_model, shorelines="migrating", rtol=RTOL)
    flipped = int(np.sum(snapshots[-1].ocean_function.data
                         != truth.ocean_function.data))
    final = SHGrid.from_array(
        sea0.data + snapshots[-1].sea_level_change.data, grid=earth.grid)

    # The misfit's gradient against a target off the reached sea level.
    target = SHGrid.from_array(
        final.data + TARGET_TILT * metre * np.cos(theta), grid=earth.grid)
    misfit = evolution.final_sea_level_misfit(
        dates, ice_model, target, shorelines="migrating", rtol=RTOL)

    # The calibration from the final sea level as the guess.
    histories = {}
    for method in ("fixed_point", "gradient"):
        guess = EarthState(ice0, final, earth, exclude_caspian=False)
        calibration = SeaLevelEvolution(op, guess).calibrate_initial_sea_level(
            dates, ice_model, final, shorelines="migrating", method=method,
            iterations=12, rtol=1e-9, solver_rtol=RTOL)
        error = np.max(np.abs(calibration.state.sea_level.data - sea0.data))
        print(f"{method}: {len(calibration.misfits)} runs, converged "
              f"{calibration.converged}, error {error / np.max(sea0.data):.2e}")
        histories[method] = [m / metre for m in calibration.misfits]

    length = 1.0 / metre
    frequency = 1.0 / earth.parameters.scales.factor(units.TIME)
    area = earth.parameters.length_scale ** 2

    def rows(coeffs):
        out = []
        for l in range(LMAX + 1):
            for m in range(l + 1):
                out.append(f"{l} {m} {coeffs[0, l, m]:.17g} "
                           f"{coeffs[1, l, m]:.17g}")
        return out

    lines = [
        "# migrating-shoreline references, generated by "
        "scripts/make_migrating_fixture.py",
        f"# {provenance('pyslfp', 'planetmodel')} (GLQ grid); the stepper "
        "fixture's Maxwell body under a melting marine cap",
        f"# {flipped} grid cells flip between the first and last nodes",
        "# blocks: 'ice0' and 'sea0' (the initial fields), then per node "
        "'node n t_si omega_x omega_y omega_z ocean_area_si' and the sea "
        "level change, then 'ocean n' and the ocean function, then "
        "'gradient' (dJ/dS0 of the misfit, J = 1/2 int r^2 dS against the "
        "reached sea level plus 50 m cos(theta)), all as coefficient rows "
        "'l m a b' (pyshtools 4pi real, SI metres); last 'misfits <method>' "
        "with the calibration's rms misfit per run (m)",
    ]
    lines.append("ice0")
    lines += rows(earth.expand_field(ice0).coeffs * length)
    lines.append("sea0")
    lines += rows(earth.expand_field(sea0).coeffs * length)
    for node, snap in enumerate(snapshots):
        t_si = (START - snap.date) * 1e3 * 365.25 * 86400.0
        omega = snap.angular_velocity_change * frequency
        ocean_area = earth.integrate(snap.ocean_function) * area
        lines.append(f"node {node} {t_si:.17g} "
                     + " ".join(f"{x:.17g}" for x in omega)
                     + f" {ocean_area:.17g}")
        lines += rows(earth.expand_field(snap.sea_level_change).coeffs * length)
        lines.append(f"ocean {node}")
        lines += rows(earth.expand_field(snap.ocean_function).coeffs)
    lines.append("gradient")
    lines += rows(earth.expand_field(misfit.gradient).coeffs * length)
    for method, history in histories.items():
        lines.append(f"misfits {method} "
                     + " ".join(f"{m:.17g}" for m in history))
    (data / "pyslfp_migrating.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote pyslfp_migrating.txt ({flipped} cells flip)")


if __name__ == "__main__":
    main()
