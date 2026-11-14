"""Writes a fault file (planetmodel.fault/1) for a `fault_file` seismic term
of an sl3d config: a curved megathrust off the synthetic South America's
west coast, its slip a tapered thrust, static (a rupture).

The interface is planetmodel's swept patch (StructuredPatch.swept): from a
trench that is an arc of the small circle 13.5 degrees from (20 S, 60 W),
inland, between the azimuths 252 and 288 degrees about it (about 940 km of
trench near 74 W, from 24 S to 16 S, concave towards the continent), the
surface
descends towards the pole at a dip of 10 degrees at the trench steepening
to 25 degrees at 50 km, its base; a bicubic patch through 9 by 6 control
positions. The slip is a thrust (rake 90, the hanging wall up dip) of
12 m at the patch's middle tapered to nothing at every edge,
12 cos^2(pi u) sin^2(pi (v + 1/2)) m in the patch's parameters, bicubic
on 13 by 9 nodes. The positions are on the Earth's mean sphere, 6371 km:
the post-seismic deck ends at 6368 km (PREM without its ocean), so the
config places the fault by depth (`"place_by": "depth"`). The file holds no
moment: sl3d forms it from the deck's moduli at each quadrature point.

The fault module is planetmodel's after 1.5.1; this environment's
planetmodel (pyproject.toml) lacks it until a release carries it, and the
script then says so and writes nothing. Run from build/examples:

    poetry run python make_megathrust.py [--out FILE]

or, against a checkout of planetmodel with the module,

    PYTHONPATH=<planetmodel checkout>/src poetry run python make_megathrust.py

It writes data/megathrust.nc (replaced only where its content changed),
which sea_level_static_megathrust.json reads; configs/run_all.sh skips that
config while the file is absent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inputs import DATA, keep_unchanged, wrote

try:
    from planetmodel.fault import (Fault, GeographicFrame, Slip, SmallCircle,
                                   StructuredPatch)
    from planetmodel.faultfile import write_fault
except ImportError:  # a planetmodel without the fault file
    Fault = None

EARTH = 6371e3  # m, the reference sphere's radius
POLE = (-20.0, 300.0)  # the trench's small circle's pole, inland
RADIUS_DEG = 13.5
AZIMUTHS = (252.0, 288.0)  # from the south end to the north, about the pole
BOTTOM = 50e3  # m
PEAK_SLIP = 12.0  # m


def dip(z):
    """The dip in degrees at depth z (m): 10 at the trench, 25 at the base."""
    return 10.0 + 15.0 * np.asarray(z) / BOTTOM


def megathrust() -> "Fault":
    frame = GeographicFrame(EARTH)
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, 13), np.linspace(-0.5, 0.5, 9),
                       indexing="ij")
    taper = PEAK_SLIP * np.cos(np.pi * u) ** 2 * np.sin(np.pi * (v + 0.5)) ** 2
    patch = StructuredPatch.swept(
        frame, SmallCircle(POLE, RADIUS_DEG, *AZIMUTHS), dip=dip, bottom=BOTTOM,
        n_u=9, n_v=6, slip=Slip.from_rake(90.0, taper, rule="cubic"))
    return Fault([patch])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DATA / "megathrust.nc")
    args = parser.parse_args()
    if Fault is None:
        print("make_megathrust.py: this planetmodel has no fault module "
              "(planetmodel.fault and planetmodel.faultfile, the fault file "
              "planetmodel.fault/1: planetmodel after 1.5.1). Install one, or "
              "run with PYTHONPATH=<planetmodel checkout>/src; "
              f"{args.out} is not written.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fault = megathrust()
    keep_unchanged(args.out, lambda target: write_fault(
        target, fault, source=Path(__file__).name,
        names=["megathrust"],
        title="a curved megathrust off the synthetic South America, a "
              "tapered thrust of 12 m at its middle",
        meta={"trench_pole": f"{POLE[0]:g} {POLE[1]:g}",
              "trench_radius_deg": RADIUS_DEG}))
    wrote(args.out, f"a megathrust swept from a {RADIUS_DEG:g} degree small "
                    f"circle to {BOTTOM / 1e3:g} km, {PEAK_SLIP:g} m of "
                    f"thrust at its middle")


if __name__ == "__main__":
    main()
