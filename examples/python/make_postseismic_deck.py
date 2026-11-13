"""Writes data/prem8.postseismic.deck: the post-seismic examples' radial
Earth for the sl3d program and ExampleSeismic, built with planetmodel from
PREM, and prints each layer's Maxwell times.

The Earth is PREM without its ocean (the solid surface at 6368 km),
sampled as tests/data/prem8.elastic.deck is (8 knots per layer), with a
thin elastic lid over a weak asthenosphere: the lid elastic from the
surface to its base (a boundary inserted there, inside PREM's lid), a
Maxwell asthenosphere from the lid's base down to PREM's 220 km
discontinuity (6151 km), a Maxwell upper mantle from there to the 670 km
discontinuity and a Maxwell lower mantle from there to the core-mantle
boundary, each of one viscosity; the core is fluid or elastic as PREM has
it. The layers that relax are made isotropic, their moduli the Voigt
averages of PREM's (sl3d refuses a transversely isotropic Maxwell layer,
and PREM's is transversely isotropic between 24 and 220 km depth); the
lid keeps PREM's own description. The deck is in planetmodel's Maxwell
(GIA) format ("format": "maxwell"), viscosity `nan` in the elastic and
fluid layers. Run from build/examples:

    poetry run python make_postseismic_deck.py [--lid-km 50]
        [--asthenosphere-viscosity 1e19] [--upper-mantle-viscosity 1e20]
        [--lower-mantle-viscosity 1e22] [--out data/prem8.postseismic.deck]

The defaults write the deck examples/data holds, which the build copies
into data/; other values write a variant (give --out to keep the
original). The Maxwell time of a layer is eta / mu, mu its shear modulus,
which varies through the layer: the range is printed.
"""
from __future__ import annotations

import argparse

import numpy as np
from planetmodel import MAXWELL, PREM, sample_deck, write_deck

from inputs import DATA, maxwell_times, with_viscosity, wrote

KNOTS_PER_LAYER = 8  # as tests/data/prem8.elastic.deck
CMB = 3480.0e3  # m: PREM's core-mantle boundary
D670 = 5701.0e3  # m: PREM's 670 km discontinuity
D220 = 6151.0e3  # m: PREM's 220 km discontinuity


def earth(lid_km: float, asthenosphere: float, upper: float, lower: float):
    """PREM without its ocean, an elastic lid `lid_km` thick, and Maxwell
    layers below it of the three viscosities (Pa s): the asthenosphere to
    PREM's 220 km discontinuity, the upper mantle to 670 km and the lower
    mantle to the core."""
    model = PREM(ocean=False).elastic()
    surface = model.skeleton.boundaries[-1]
    base = surface - 1.0e3 * lid_km
    if not D220 < base < surface:
        raise SystemExit("the lid must be thinner than "
                         f"{(surface - D220) / 1e3:.0f} km")
    if base not in model.skeleton.boundaries:
        model = model.refined([base], names=["lid_base"])
        i = model.skeleton.locate(base - 1.0).layer
        model = model.renamed(layers={i: "asthenosphere_lid",
                                      i + 1: "lid_mantle"})

    def between(lo, hi):
        return [lay.index for lay in model.layers
                if lay.interval[0] >= lo - 1.0 and lay.interval[1] <= hi + 1.0]

    model = with_viscosity(model, between(D220, base), asthenosphere)
    model = with_viscosity(model, between(D670, D220), upper)
    model = with_viscosity(model, between(CMB, D670), lower)
    return model.named(
        f"prem8 post-seismic deck: PREM without its ocean, 8 knots per "
        f"layer, an elastic lid {lid_km:g} km thick, Maxwell layers "
        f"{asthenosphere:g} Pa s to 220 km, {upper:g} Pa s to 670 km, "
        f"{lower:g} Pa s to the core (isotropic, Voigt), nan elsewhere")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lid-km", type=float, default=50.0,
                        help="the elastic lid's thickness (default 50 km)")
    parser.add_argument("--asthenosphere-viscosity", type=float, default=1e19,
                        help="Pa s, from the lid's base to 220 km (default "
                             "1e19)")
    parser.add_argument("--upper-mantle-viscosity", type=float, default=1e20,
                        help="Pa s, from 220 km to 670 km (default 1e20)")
    parser.add_argument("--lower-mantle-viscosity", type=float, default=1e22,
                        help="Pa s, from 670 km to the core (default 1e22)")
    parser.add_argument("--out", default=str(DATA / "prem8.postseismic.deck"),
                        help="the deck (default data/prem8.postseismic.deck)")
    args = parser.parse_args()
    model = earth(args.lid_km, args.asthenosphere_viscosity,
                  args.upper_mantle_viscosity, args.lower_mantle_viscosity)
    knots = [np.linspace(*lay.interval, KNOTS_PER_LAYER)
             for lay in model.layers]
    deck = sample_deck(model, MAXWELL.columns, knots=knots)
    surface = model.skeleton.boundaries[-1]
    print(model.name)
    for lay in model.layers:
        eta = (f"{lay['viscosity'](lay.interval[1]):.3g} Pa s"
               if "viscosity" in lay.names else
               "fluid" if model.is_fluid(lay.index) else "elastic")
        print(f"  {lay.interval[0] / 1e3:8.1f} - {lay.interval[1] / 1e3:6.1f}"
              f" km  {lay.name or '-':24s} {eta}")
    path = wrote(write_deck(args.out, deck, MAXWELL))
    print("Maxwell times eta / mu, from the deck as written (depths below "
          f"the surface at {surface / 1e3:g} km):")
    for top, bottom, eta, shortest, longest in maxwell_times(path):
        print(f"  {top:6.1f} - {bottom:6.1f} km  {eta:8.3g} Pa s  "
              f"{shortest:8.4g} - {longest:8.4g} yr")


if __name__ == "__main__":
    main()
