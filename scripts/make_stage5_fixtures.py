"""Stage-5 adjoint fixture: pyslfp's ice-history and itau kernels.

The stage-3 setting (analytic test state, PREM-deck Maxwell body,
lmax 16, GLQ grid — which coincides with sl3dpp's GSHTrans grid, so the
comparison isolates implementation error) with a full measurement set:
a sea-level dual at the middle node and displacement + potential +
angular-velocity duals at the last. Both adjoint methods are dumped —
the C++ port's algebra is identical formulation by formulation, so each
gates its own counterpart tightly.

Per method block: the load-dual coefficients per node (SI, the kernel
against the direct load paired by the SI surface integral) and the
branch-0 itau kernel (SI, nodal over the radial mesh, paired by the
plain nodal sum) from the recorded linear-melt forward trajectory.

Run from the pyslfp checkout:

    cd ~/dev/pyslfp && poetry run python \
        ~/dev/sl3dpp/scripts/make_stage5_fixtures.py

Output (committed): tests/data/stage5_kernels.txt
"""
from __future__ import annotations

import datetime
import pathlib

import numpy as np
from planetmodel import (Geometry, Model, RadialMesh, deck_layers, read_deck,
                         units)
from planetmodel.deck import DeckFormat
from pyshtools import SHGrid
from pyslfp.core import EarthModel, EarthState
from pyslfp.ice import AnalyticalIceModel
from pyslfp.love_numbers.material import Material
from pyslfp.love_numbers.table import LoveNumbers
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator
from pyslfp.viscoelastic.evolution import KYR, Measurement

DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"
LMAX = 16
# The resolved regime (h = dt itau < ~1.4 everywhere): in the stiff
# regime the CONTINUOUS multiplier integral amplifies like exp(sum h)
# at the stiffest nodes (measured 16 Oct 2026: ~1e31 at eta 1e21 with
# 2 kyr steps, every measurement type; the discrete kernel stays O(1)),
# so a stiff cross-code comparison only compares amplified round-off.
ETA = 1e22
T_KYR = 10.0
NODES = 6
MELT_FRACTION = 0.1
ELASTIC = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta"),
                     name="elastic")
SURFACE_DENSITY = units.DENSITY * units.LENGTH
POTENTIAL = units.LENGTH**2 * units.TIME**-2


def cap(theta, phi, colat0, lon0, width):
    c = (np.cos(theta) * np.cos(colat0)
         + np.sin(theta) * np.sin(colat0) * np.cos(phi - lon0))
    return np.exp(-0.5 * (np.arccos(np.clip(c, -1.0, 1.0)) / width) ** 2)


def main() -> None:
    deck = read_deck(DATA / "prem8.elastic.deck", ELASTIC)
    skeleton, layers = deck_layers(deck, kind="cubic")
    model = Model(Geometry(skeleton), layers)
    mesh = RadialMesh(model, lmax=LMAX, ngll=5)
    material = Material(mesh, model)
    b = skeleton.boundaries
    viscosity = {}
    for i, layer in enumerate(model.layers):
        fluid = np.all(np.asarray(layer["vsv"](np.linspace(*layer.interval, 3)))
                       == 0.0)
        if not fluid and b[i] >= 3.4799e6 and b[i + 1] <= 6.1511e6:
            viscosity[i] = ETA
    rheology = Rheology.maxwell(material, viscosity)
    op = ViscoelasticOperator(rheology, LMAX)
    table = LoveNumbers.from_model(model, LMAX, mesh=mesh)
    earth = EarthModel(LMAX, love_numbers=table, grid="GLQ", extend=False)
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    ice, sl = ice_model.get_ice_thickness_and_sea_level(
        0.0, LMAX, grid=earth.grid_name, sampling=earth.sampling,
        extend=earth.extend)
    state = EarthState(ice, sl, earth, exclude_caspian=False)
    evolution = SeaLevelEvolution(op, state)
    scales = earth.parameters.scales

    # The measurement duals: the Cap formulas of the C++ gate, sampled on
    # the coinciding grid (dimensionless).
    zeros = state.sea_level  # any grid of the right shape
    theta = np.deg2rad(90.0 - zeros.lats())[:, None]
    phi = np.deg2rad(zeros.lons())[None, :]

    def grid_of(amplitude, colat0, lon0, width):
        return SHGrid.from_array(
            amplitude * cap(theta, phi, colat0, lon0, width),
            grid=earth.grid_name, copy=False)

    dates = np.linspace(T_KYR, 0.0, NODES)
    # The potential and angular-velocity duals are SI-ANCHORED: pyslfp's
    # EarthModel scales (T ≈ 805 s, g_nd ≈ 1) differ from
    # planetmodel/sl3dpp geophysical (T ≈ 1648 s, G_nd = 1), and the
    # non-dimensional potential scales as T^2 (omega as 1/T), so a plain
    # shared dual would define DIFFERENT objectives (a factor 4 pi / 3 on
    # phi; see plan.md decision log, 16 Oct 2026). Each code multiplies
    # the dual by ITS OWN factor against the shared SI anchor, making the
    # objective code-invariant. Lengths need no anchor (same L).
    PHI_ANCHOR = 1.0e7   # m^2 s^-2
    OMEGA_ANCHOR = 1.0e-4  # s^-1
    pot_conv = scales.factor(POTENTIAL) / PHI_ANCHOR
    om_conv = (1.0 / scales.factor(units.TIME)) / OMEGA_ANCHOR
    measurement = {
        float(dates[3]): grid_of(1.0, 0.9, 1.2, 0.5),
        float(dates[NODES - 1]): Measurement(
            displacement=grid_of(0.7, 2.4, 0.3, 0.6),
            gravity_potential=grid_of(0.4 * pot_conv, 1.4, 3.3, 0.7),
            angular_velocity=om_conv * np.array([0.2, -0.4, 0.8]),
        ),
    }

    # The linear-melt forward run supplies the recorded trajectory the
    # itau kernels pair with (the stage-3 history on the coarser grid).
    kyr = KYR / scales.factor(units.TIME)
    times = (dates[0] - dates) * kyr

    def ice_load(t):
        fraction = -MELT_FRACTION * (t / times[-1])
        change = fraction * state.ice_thickness
        return state.direct_load_from_ice_thickness_change(change)

    snapshots, _ = evolution.run(times, ice_load, rotational_feedbacks=True,
                                 rtol=1e-12, record=True)

    # SI conversions: the load dual pairs by the SI surface integral with
    # J = int g xi dS in SI; the itau kernel pairs with nodal d(itau) in
    # SI inverse seconds by the plain nodal sum.
    length = scales.factor(units.LENGTH)
    sigma = scales.factor(SURFACE_DENSITY)
    t_mat = op.material.scales.factor(units.TIME)
    load_si = length / sigma
    itau_si = length**3 * t_mat
    r_nodal = op.material.mesh.r * op.material.scales.factor(units.LENGTH)

    lines = [
        "# stage-5 adjoint kernel reference, generated by "
        "scripts/make_stage5_fixtures.py",
        f"# pyslfp (GLQ grid), {datetime.date.today().isoformat()}; the "
        f"stage-3 setting with eta {ETA:.0e} (the RESOLVED regime — see "
        "the script on the stiff continuous-kernel amplification); "
        "sea-level dual at node 3, displacement/potential/"
        "angular-velocity duals at node 5; linear-melt trajectory for "
        "the itau kernels",
        "# blocks: 'method <discrete|continuous>'; per node 'node n t_si' "
        "then load-dual coefficient rows 'l m a b' (SI, m^3/kg: pairs "
        "with a surface-density perturbation by the SI surface "
        "integral); then 'itau <count>' and nodal rows 'r_si value_si' "
        "(pairs with d(itau) in 1/s by the plain nodal sum)",
        "# the potential and angular-velocity duals are SI-anchored "
        f"(PHI_ANCHOR {PHI_ANCHOR:.0e} m^2 s^-2, OMEGA_ANCHOR "
        f"{OMEGA_ANCHOR:.0e} s^-1): each code multiplies its dual by its "
        "own factor(POTENTIAL)/PHI_ANCHOR etc., making the objective "
        f"code-invariant; scales here: length {length:.17g} m, time "
        f"{scales.factor(units.TIME):.17g} s",
    ]
    # The continuous block carries NO itau kernel: pyslfp's continuous
    # itau accumulation produces ~1e31-scale values in this layered
    # Maxwell setting even for a single step (16 Oct 2026, under
    # investigation on the pyslfp side); sl3dpp's own continuous kernel
    # is gated internally against its exact discrete instead.
    for name, method, with_itau in (
            ("discrete", evolution.adjoint_ice_history, True),
            ("continuous", evolution.continuous_adjoint_ice_history, False)):
        kernels = method(dates, measurement, internal_states=snapshots,
                         rtol=1e-12)
        lines.append(f"method {name}")
        for n in range(dates.size):
            t_si = float(times[n] * scales.factor(units.TIME))
            lines.append(f"node {n} {t_si:.17g}")
            coeffs = earth.expand_field(kernels.load_duals[n]).coeffs * load_si
            for l in range(LMAX + 1):
                for m in range(l + 1):
                    lines.append(f"{l} {m} {coeffs[0, l, m]:.17g} "
                                 f"{coeffs[1, l, m]:.17g}")
        if not with_itau:
            lines.append("itau 0")
            continue
        kernel = kernels.itau[0] * itau_si
        flat_r = r_nodal.reshape(-1)
        flat_k = kernel.reshape(-1)
        lines.append(f"itau {flat_k.size}")
        for r, v in zip(flat_r, flat_k):
            lines.append(f"{r:.17g} {v:.17g}")
    (DATA / "stage5_kernels.txt").write_text("\n".join(lines) + "\n")
    print("wrote stage5_kernels.txt")


if __name__ == "__main__":
    main()
