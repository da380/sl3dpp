"""Writes data/earth_custom.gia.deck and data/earth_custom.elastic.deck: a
radial Earth for the sl3d program, built with planetmodel from PREM.

The Earth is PREM without its ocean (the solid surface at 6368 km), with
an elastic lithosphere of the chosen thickness, a Maxwell upper mantle
from its base down to the 670 km discontinuity and a Maxwell lower mantle
from there to the core-mantle boundary, each of one viscosity; the core is
fluid or elastic as PREM has it. A boundary is inserted at the
lithosphere's base, and the mantle layers that relax are made isotropic
(their moduli the Voigt averages of PREM's): sl3d refuses a transversely
isotropic Maxwell layer, and PREM's is transversely isotropic between 24
and 220 km depth. The lithosphere keeps PREM's own description.

Two decks are written, the same structure in sl3d's two formats:
<stem>.gia.deck, planetmodel's Maxwell (GIA) deck, viscosity `nan` in the
elastic and fluid layers, for a run through time ("format": "maxwell");
and <stem>.elastic.deck, the elastic structure alone ("format":
"elastic"). Run from build/examples:

    poetry run python make_earth_deck.py [--lithosphere-km 100]
        [--upper-mantle-viscosity 5e20] [--lower-mantle-viscosity 5e21]
        [--out data/earth_custom]

then, from build/examples/configs, any config's model block takes them,
{"deck": "../data/earth_custom.gia.deck", "format": "maxwell"} or the
elastic deck with "format": "elastic"; the deck carries the viscosity of a
radial run through time. make_lateral_model.py builds 3D viscosity and
moduli files on it (by default on this deck).
"""
from __future__ import annotations

import argparse

from planetmodel import PREM

from inputs import DATA, with_viscosity, write_decks

CMB = 3480.0e3  # m: PREM's core-mantle boundary
D670 = 5701.0e3  # m: PREM's 670 km discontinuity


def earth(lithosphere_km: float, upper: float, lower: float):
    """PREM without its ocean, the lithosphere elastic, the mantle below
    Maxwell with the two viscosities (Pa s) above and below 670 km."""
    model = PREM(ocean=False).elastic()
    surface = model.skeleton.boundaries[-1]
    base = surface - 1.0e3 * lithosphere_km
    if not D670 < base < surface:
        raise SystemExit("the lithosphere must be thinner than "
                         f"{(surface - D670) / 1e3:.0f} km")
    if base not in model.skeleton.boundaries:
        model = model.refined([base], names=["lithosphere_base"])
        i = model.skeleton.locate(base - 1.0).layer
        model = model.renamed(layers={i: "asthenosphere",
                                      i + 1: "lithospheric_mantle"})

    def between(lo, hi):
        return [lay.index for lay in model.layers
                if lay.interval[0] >= lo - 1.0 and lay.interval[1] <= hi + 1.0]

    model = with_viscosity(model, between(D670, base), upper)
    model = with_viscosity(model, between(CMB, D670), lower)
    return model.named(f"PREM without its ocean, a {lithosphere_km:g} km "
                       f"elastic lithosphere, {upper:g} Pa s to 670 km, "
                       f"{lower:g} Pa s below")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lithosphere-km", type=float, default=100.0,
                        help="the elastic lithosphere's thickness (default "
                             "100 km)")
    parser.add_argument("--upper-mantle-viscosity", type=float, default=5e20,
                        help="Pa s, from the lithosphere's base to 670 km "
                             "(default 5e20)")
    parser.add_argument("--lower-mantle-viscosity", type=float, default=5e21,
                        help="Pa s, from 670 km to the core (default 5e21)")
    parser.add_argument("--out", default=str(DATA / "earth_custom"),
                        help="the decks' common stem (default "
                             "data/earth_custom)")
    args = parser.parse_args()
    model = earth(args.lithosphere_km, args.upper_mantle_viscosity,
                  args.lower_mantle_viscosity)
    print(model.name)
    for lay in model.layers:
        eta = (f"{lay['viscosity'](lay.interval[1]):.3g} Pa s"
               if "viscosity" in lay.names else
               "fluid" if model.is_fluid(lay.index) else "elastic")
        print(f"  {lay.interval[0] / 1e3:8.1f} - {lay.interval[1] / 1e3:6.1f}"
              f" km  {lay.name or '-':24s} {eta}")
    write_decks(model, args.out)


if __name__ == "__main__":
    main()
