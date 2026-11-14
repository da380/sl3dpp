"""Fault files (planetmodel.fault/1) for the fault-file term of sl3d.

The faults the tests lay over the homogeneous test ball (radius 6371 km,
rho 5500 kg m^-3, vs 5500 m/s, so mu = 1.66375e11 Pa), written by
planetmodel's own writer:

  fault_plane.nc    the twin of a planar `fault` term: a plane centred at
                    40 S, 150 E, 30 km deep, strike 30, dip 15, 600 by
                    200 km, a uniform thrust (rake 90) of the slip that
                    gives it a moment of 1e22 N m on the ball,
                    M0 / (mu L W); static;
  fault_growing.nc  that plane, its slip on a time axis: cumulative slip 0
                    at -2000 yr and the twin's at 0 yr, linear between;
  fault_turning.nc  that plane, two slices whose slip turns: a thrust of
                    the twin's slip at 0 yr, a strike-slip of twice it at
                    10 yr, linear between;
  fault_curved.nc   a curved megathrust swept from a small-circle trench
                    (6 degrees from 30 S, 160 E, azimuths 250 to 290) at a
                    dip steepening from 12 degrees at the trench to 50 km,
                    a bicubic patch of 7 by 5 control positions, its thrust
                    tapered to nothing at every edge (5 m at its middle),
                    bicubic on 9 by 7 nodes; static.

The fault module is planetmodel's after 1.5.1; until a release carries it,
run against a checkout (the scripts' environment otherwise, pyproject.toml):

    cd scripts && PYTHONPATH=<planetmodel checkout>/src \\
        poetry run python make_fault_fixtures.py [--data-dir DIR]

Output (committed): tests/data/fault_*.nc, each naming this script and the
planetmodel that wrote it (its commit, from a checkout) in its `source`
and `meta_script` attributes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from planetmodel import Scales, __version__
from planetmodel.fault import (Fault, GeographicFrame, Plane, Slip,
                               SmallCircle, StructuredPatch)
from planetmodel.faultfile import write_fault

from fixture_tools import data_dir

RADIUS = 6371e3  # the ball's
MU = 5500.0 * 5500.0**2  # its shear modulus, Pa
YEAR = 3.15576e7  # s
EARTH = GeographicFrame(RADIUS)

# The twin plane, as the tests' `fault` term gives it.
CENTROID = (-40.0, 150.0, RADIUS - 30e3)
STRIKE, DIP, LENGTH, WIDTH = 30.0, 15.0, 600e3, 200e3
MOMENT = 1e22
SLIP = MOMENT / (MU * LENGTH * WIDTH)


def plane(slip: Slip) -> Plane:
    return Plane(EARTH, CENTROID, strike=STRIKE, dip=DIP, length=LENGTH,
                 width=WIDTH, slip=slip)


def planetmodel_commit() -> str:
    """The commit of the planetmodel checkout imported, or "" for an
    installed release."""
    import planetmodel
    import subprocess
    try:
        return subprocess.run(
            ["git", "-C", str(Path(planetmodel.__file__).parent), "rev-parse",
             "--short", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    out = data_dir(__doc__)
    commit = planetmodel_commit()
    source = (f"planetmodel {__version__}" + (f" ({commit})" if commit else "")
              + ", sl3dpp scripts/make_fault_fixtures.py")
    meta = {"script": "scripts/make_fault_fixtures.py"}

    def write(name: str, fault: Fault, title: str) -> None:
        path = write_fault(out / name, fault, source=source, title=title,
                           meta=meta)
        print("wrote", path)

    write("fault_plane.nc", Fault([plane(Slip.from_rake(90.0, SLIP))]),
          "the twin of a planar fault of 1e22 N m on the test ball")
    write("fault_growing.nc",
          Fault([plane(Slip.from_rake(90.0, np.array([0.0, SLIP])[:, None, None]))],
                support="node", time_interpolation="linear",
                time=[-2000.0, 0.0], scales=Scales(time=YEAR)),
          "the twin plane, its slip growing linearly from -2000 to 0 yr")
    write("fault_turning.nc",
          Fault([plane(Slip.from_rake(np.array([90.0, 0.0])[:, None, None],
                                      np.array([SLIP, 2.0 * SLIP])[:, None, None]))],
                support="node", time_interpolation="linear", time=[0.0, 10.0],
                scales=Scales(time=YEAR)),
          "the twin plane, a thrust at 0 yr turning to a strike-slip by 10 yr")

    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.5, 0.5, 7),
                       indexing="ij")
    taper = 5.0 * np.cos(np.pi * U) ** 2 * np.cos(np.pi * V) ** 2
    curved = StructuredPatch.swept(
        EARTH, SmallCircle((-30.0, 160.0), 6.0, 250.0, 290.0),
        dip=lambda z: 12.0 + z / 4e3, bottom=50e3, n_u=7, n_v=5,
        slip=Slip.from_rake(90.0, taper, rule="cubic"))
    write("fault_curved.nc", Fault([curved]),
          "a curved megathrust with a tapered thrust")


if __name__ == "__main__":
    main()
