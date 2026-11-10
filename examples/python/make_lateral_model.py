"""Writes a 3D model file for the sl3d program: a laterally varying
viscosity (`model.viscosity.file`) or lateral moduli (`model.moduli.file`)
on the body of a given deck, built with planetmodel from that deck, so that
its radial ends and layers are the deck's, as sl3d requires.

--field viscosity perturbs the deck's own viscosity (a Maxwell deck) on its
Maxwell layers, as a log perturbation; --field moduli perturbs the shear
and bulk moduli, relatively, on the solid layers above the core (the bulk
modulus by four tenths of the shear's), `mu` absent in fluid layers.
--kind chooses the pattern:

- analytic: a smooth degree-2 and degree-3 pattern in a radial envelope
  vanishing at both ends of the layers, peaking at --amplitude (decades of
  viscosity, default 1; shear modulus, default 0.05);
- tomography: S40RTS (Ritsema et al. 2011), an Earth model, transplanted
  onto the deck's layers to degree --lmax, its mean removed; the viscosity
  through sl3d's conversion chain (planetmodel's
  conversion.building.sl3d_viscosity), the moduli from the perturbed shear
  speed. planetmodel caches it in ~/.cache/planetmodel (or
  $PLANETMODEL_DATA); if it is not there the script says what it would
  download and stops, unless --download is given;
- random: a zero-mean Gaussian random field (planetmodel's
  randomfield.SphericalGRF, Matern) with correlation lengths
  --correlation-km (radial and lateral, default 400 and 800 km), standard
  deviation --amplitude (default 0.3 decades, or 0.02) and seed --seed.

Run from build/examples, after make_earth_deck.py for the default deck:

    poetry run python make_lateral_model.py --field viscosity|moduli
        --kind analytic|tomography|random [--deck data/earth_custom.gia.deck]
        [--lmax 32] [--amplitude A] [--correlation-km 400 800] [--seed 1]
        [--download] [--out FILE]

It writes data/<deck>_<field>_<kind>.nc, e.g.
data/earth_custom_viscosity_s40rts.nc, on a Gauss-Legendre grid of degree
--lmax. A config's model block takes it beside the deck:
{"deck": "../data/earth_custom.gia.deck", "viscosity": {"file":
"../data/earth_custom_viscosity_s40rts.nc"}} for a run through time, or
"moduli": {"file": ...} (docs/sl3d-configs.md, The model block).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from planetmodel import (SCALAR, AnalyticField, RadialMesh, gauss_legendre,
                         perturbed)
from planetmodel.modelfile import write_model

from inputs import DATA, read_deck_model, wrote

MOHO = 6346.6e3  # m: PREM's Moho, where S40RTS ends


def pattern(theta, phi):
    """A smooth degree-2 and degree-3 lateral pattern, largest magnitude 1."""
    p = (np.sin(theta) ** 2 * np.cos(2.0 * phi)
         + 0.8 * np.sin(theta) * (5.0 * np.cos(theta) ** 2 - 1.0)
         * np.cos(phi - 1.0))
    return p / 1.69316  # its largest magnitude on the sphere


def envelope(r, lo, hi):
    """sin^2 in radius: zero at lo and hi, one halfway."""
    return np.sin(np.pi * np.clip((r - lo) / (hi - lo), 0.0, 1.0)) ** 2


def tomography(deck, field, layers, fluid, lmax, download):
    """S40RTS on the deck's layers: the model and the layers it covers."""
    from planetmodel import datasets
    from planetmodel.conversion import building

    try:
        datasets.fetch("S40RTS", download=download)
    except Exception:  # not cached, and no download allowed
        url = datasets.REGISTRY["S40RTS"].sources[0].url
        raise SystemExit(
            "S40RTS is not in planetmodel's cache. Getting it "
            f"downloads {url} (about 0.4 MB) into "
            f"{datasets.cache_directory()}; run again with --download to do "
            "so.") from None
    if field == "viscosity":
        model = datasets.transplanted(deck, "S40RTS", remove_mean=True,
                                      layers=layers, lmax=lmax)
        return building.sl3d_viscosity(model, layers=layers), layers
    layers = [i for i in layers if deck.layer(i).interval[1] <= MOHO + 1.0]
    model = datasets.transplanted(deck, "S40RTS", remove_mean=True,
                                  layers=layers, lmax=lmax)
    return model.isotropic().without_field("mu", layers=fluid), layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--field", required=True,
                        choices=["viscosity", "moduli"])
    parser.add_argument("--kind", required=True,
                        choices=["analytic", "tomography", "random"])
    parser.add_argument("--deck", type=Path,
                        default=DATA / "earth_custom.gia.deck")
    parser.add_argument("--lmax", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--correlation-km", type=float, nargs=2,
                        default=[400.0, 800.0], metavar=("RADIAL", "LATERAL"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.deck.exists():
        raise SystemExit(f"no deck {args.deck} (make_earth_deck.py writes "
                         "data/earth_custom.gia.deck)")
    deck = read_deck_model(args.deck)
    viscosity = args.field == "viscosity"
    fluid = [lay.index for lay in deck.layers if deck.is_fluid(lay.index)]
    if viscosity:
        layers = list(deck.layers_with("viscosity"))
        if not layers:
            raise SystemExit(f"{args.deck} has no viscosity: a viscosity "
                             "file needs a Maxwell deck")
        model, fields = deck, ["rho", "vpv", "vsv", "vph", "vsh", "eta",
                               "viscosity"]
        scale = math.log(10.0) * (args.amplitude or
                                  (1.0 if args.kind == "analytic" else 0.3))
    else:  # every layer holds kappa and mu, mu absent in fluids
        top = max([deck.layer(i).interval[1] for i in fluid], default=0.0)
        layers = [lay.index for lay in deck.layers
                  if lay.interval[0] >= top and lay.index not in fluid]
        model = deck.isotropic().without_field("mu", layers=fluid)
        fields = ["rho", "kappa", "mu"]
        scale = args.amplitude or (0.05 if args.kind == "analytic" else 0.02)
    span = (deck.layer(layers[0]).interval[0],
            deck.layer(layers[-1]).interval[1])

    if args.kind == "tomography":
        if abs(deck.skeleton.boundaries[-1] - 6371.0e3) > 20.0e3:
            raise SystemExit("S40RTS is an Earth model, and this deck is not "
                             "the Earth's size")
        model, layers = tomography(deck, args.field, layers, fluid, args.lmax,
                                   args.download)
    else:
        if args.kind == "analytic":
            def make(name, factor):  # the pattern times factor
                return AnalyticField(span, lambda r, t, p: factor * scale
                                     * envelope(r, *span) * pattern(t, p),
                                     name=name)
        else:
            from planetmodel.randomfield import SphericalGRF

            radial, lateral = (1.0e3 * x for x in args.correlation_km)
            grf = SphericalGRF(*span, 1.5, radial, lam_h=lateral, sigma=scale,
                               lmax=args.lmax)
            sample = grf.sample(rng=np.random.default_rng(args.seed))

            def make(name, factor):
                return grf.to_field(factor * sample, character=SCALAR,
                                    name=name)
        if viscosity:
            model = perturbed(model, "viscosity", make("viscosity", 1.0),
                              kind="log", layers=layers)
        else:
            for name, factor in (("mu", 1.0), ("kappa", 0.4)):
                model = perturbed(model, name, make(name, factor),
                                  kind="relative", layers=layers)

    varying = ["viscosity", "vsv", "vsh"] if viscosity else ["mu", "kappa"]
    tag = "s40rts" if args.kind == "tomography" else args.kind
    out = args.out or DATA / f"{args.deck.name.split('.')[0]}_{args.field}_{tag}.nc"
    write_model(out, model, grid=gauss_legendre(args.lmax),
                radial=RadialMesh(deck.geometry, drmax=100.0e3, ngll=5),
                fields=fields,
                interpolation={n: ("polynomial", "cubic") for n in varying},
                meta={"deck": str(args.deck), "field": args.field,
                      "kind": args.kind}, source=Path(__file__).name)
    top = deck.layer(layers[-1]).interval[1]
    wrote(out, f"{args.field}, {args.kind}, lmax {args.lmax}, on "
               f"{len(layers)} layers from {span[0] / 1e3:.0f} to "
               f"{top / 1e3:.0f} km, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
