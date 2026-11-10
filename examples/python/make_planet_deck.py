"""Writes data/planet.gia.deck and data/planet.elastic.deck: a simple
planet other than the Earth for the sl3d program, built with planetmodel,
and prints its mass and surface gravity.

The default is a three-layer "toy Mars": a fluid iron core, a Maxwell
mantle and an elastic lid, each layer of uniform density and seismic
velocities (a sketch with Mars's radius and roughly its mass, not a model
of Mars). Every number is an option: the radius, the core's radius, the
lid's thickness, and each layer's density and velocities (core, mantle,
lid), and the mantle's viscosity.

Two decks are written, the same planet in sl3d's two formats:
<stem>.gia.deck, planetmodel's Maxwell (GIA) deck, its viscosity `nan` in
the core and the lid ("format": "maxwell"), and <stem>.elastic.deck, the
elastic structure alone ("format": "elastic"). Run from build/examples:

    poetry run python make_planet_deck.py [--radius-km 3390]
        [--core-km 1830] [--lid-km 150] [--density 6200 3500 3000]
        [--vp 5000 8000 7000] [--vs 0 4500 4000] [--viscosity 1e21]
        [--out data/planet]

The tide configs response_tide_planet.json (the elastic tidal response
and the planet's Love numbers) and response_tide_planet_relax.json (a
held tide relaxing on the Maxwell mantle) read these decks; run_all.sh
runs them once data/planet.gia.deck exists.
"""
from __future__ import annotations

import argparse
import math

import planetmodel as pm
from planetmodel import LayeredIsotropicElastic

from inputs import DATA, with_viscosity, write_decks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--radius-km", type=float, default=3390.0,
                        help="the planet's radius (default 3390 km)")
    parser.add_argument("--core-km", type=float, default=1830.0,
                        help="the core's radius (default 1830 km)")
    parser.add_argument("--lid-km", type=float, default=150.0,
                        help="the elastic lid's thickness (default 150 km)")
    parser.add_argument("--density", type=float, nargs=3,
                        default=[6200.0, 3500.0, 3000.0],
                        metavar=("CORE", "MANTLE", "LID"),
                        help="kg m^-3 (default 6200 3500 3000)")
    parser.add_argument("--vp", type=float, nargs=3,
                        default=[5000.0, 8000.0, 7000.0],
                        metavar=("CORE", "MANTLE", "LID"),
                        help="P-wave speeds, m s^-1 (default 5000 8000 7000)")
    parser.add_argument("--vs", type=float, nargs=3,
                        default=[0.0, 4500.0, 4000.0],
                        metavar=("CORE", "MANTLE", "LID"),
                        help="S-wave speeds, m s^-1, 0 for a fluid (default "
                             "0 4500 4000)")
    parser.add_argument("--viscosity", type=float, default=1e21,
                        help="the mantle's viscosity, Pa s (default 1e21)")
    parser.add_argument("--out", default=str(DATA / "planet"),
                        help="the decks' common stem (default data/planet)")
    args = parser.parse_args()

    radius, core = 1.0e3 * args.radius_km, 1.0e3 * args.core_km
    lid = radius - 1.0e3 * args.lid_km
    if not 0.0 < core < lid < radius:
        raise SystemExit("the core must lie below the lid's base")
    planet = LayeredIsotropicElastic(
        [0.0, core, lid, radius], rho=args.density, vp=args.vp, vs=args.vs,
        layer_names=["core", "mantle", "lid"],
        interface_names=["cmb", "lid_base", "surface"])
    planet = with_viscosity(planet, ["mantle"], args.viscosity).named(
        f"a three-layer planet of radius {args.radius_km:g} km: fluid core "
        f"to {args.core_km:g} km, a {args.viscosity:g} Pa s mantle, a "
        f"{args.lid_km:g} km elastic lid")

    mass = pm.mass(planet)
    gravity = pm.gravity(planet, radius)
    mu = args.density[1] * args.vs[1] ** 2
    print(planet.name)
    print(f"  mass {mass:.4g} kg, surface gravity {gravity:.4g} m s^-2, "
          f"mean density {mass / (4.0 / 3.0 * math.pi * radius**3):.0f}"
          f" kg m^-3")
    print(f"  the mantle's Maxwell time, viscosity / shear modulus: "
          f"{args.viscosity / mu / 3.15576e10:.3g} kyr")
    write_decks(planet, args.out)


if __name__ == "__main__":
    main()
