"""Writes a 3D model file for the sl3d program: a laterally varying
viscosity (`model.viscosity.file`) or lateral moduli (`model.moduli.file`)
on the body of a given deck, built with planetmodel from that deck, so that
its radial ends and layers are the deck's, as sl3d requires.

--field viscosity perturbs the deck's own viscosity (a Maxwell deck) on its
Maxwell layers, as a log perturbation; --field moduli perturbs the shear
and bulk moduli, relatively, on the solid layers above the core, `mu`
absent in fluid layers. Either way the file stores each perturbed field as
the deck's radial field and a perturbation of it, so that both choices of
a config's `reference` (`file_radial`, or the lateral or geometric mean)
apply. --kind chooses the pattern:

- analytic: a smooth degree-2 and degree-3 pattern in a radial envelope
  vanishing at both ends of the layers, peaking at --amplitude (decades of
  viscosity, default 1; shear modulus, default 0.05, the bulk modulus by
  four tenths of the shear's);
- tomography: S40RTS (Ritsema et al. 2011), an Earth model, transplanted
  onto the deck's layers to degree --lmax, its mean removed. For the
  viscosity, the shear-speed anomaly dlnVs goes through sl3d's conversion
  chain (planetmodel's conversion.building: sl3d's density scaling and
  expansivity to a temperature anomaly, then the exponential law,
  eta = eta_0 exp(-eps dT), eps 0.01 per K by --law sl3d or Milne et
  al.'s 0.04 by --law milne). --renormalise puts each radius's lateral
  mean viscosity back to the deck's (Latychev et al. 2005), so that the
  deck's radial viscosity is the 3D model's arithmetic mean; without it
  the deck's is the geometric mean. --lid ETA (Pa s) thickens the
  lithosphere where the mantle beneath it stays stiff: above the depth,
  within 400 km of the surface, at which the viscosity first falls to
  ETA, the viscosity is raised to 1e30 Pa s, elastic on any time scale
  of GIA. For the moduli, the shear speed takes dlnVs and the P speed
  dlnVp = dlnVs / R, R = --vp-scaling (default 2; values of about 1.5
  to 3 are used, and R is often taken to grow with depth: here it is one
  constant), at fixed density (sl3d takes no lateral density), and the
  moduli follow from the perturbed speeds, the Voigt averages of the
  deck's description where it is transversely isotropic: so
  dln mu = (1 + dlnVs)^2 - 1, about 2 dlnVs, and
  delta kappa = rho [Vp^2 ((1 + dlnVs / R)^2 - 1)
                     - 4/3 Vs^2 ((1 + dlnVs)^2 - 1)],
  which in the mantle (Vp / Vs about 1.8) at R = 2 is about a third of
  dlnVs times kappa, of the sign of dln mu: a slow region is softer in
  both moduli. S40RTS ends at the Moho, so the crust is not perturbed.
  planetmodel caches S40RTS in ~/.cache/planetmodel (or
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
        [--vp-scaling 2] [--law sl3d|milne] [--renormalise] [--lid ETA]
        [--download] [--out FILE]

It writes data/<deck>_<field>_<kind>.nc, e.g.
data/earth_custom_viscosity_s40rts.nc, on a Gauss-Legendre grid of degree
--lmax, and prints the lateral contrast it holds: for a viscosity, ln of
the viscosity over the deck's radial one (its smallest and largest values
and its rms), for moduli, the rms of each relative perturbation and the
correlation between the two. A config's model block takes it beside the
deck: {"deck": "../data/earth_custom.gia.deck", "viscosity": {"file":
"../data/earth_custom_viscosity_s40rts.nc"}} for a run through time, or
"moduli": {"file": ...} (docs/sl3d-configs.md, The model block).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from planetmodel import (SCALAR, AnalyticField, PerturbedField, RadialMesh,
                         equiangular, gauss_legendre, perturbed)
from planetmodel.modelfile import read_model_file, write_model

from inputs import DATA, read_deck_model, wrote

MOHO = 6346.6e3  # m: PREM's Moho, where S40RTS ends
LID_DEPTH = 400.0e3  # m: how deep below the surface a lid's base is sought
LID_VISCOSITY = 1.0e30  # Pa s: the lid, elastic on GIA's time scales


def pattern(theta, phi):
    """A smooth degree-2 and degree-3 lateral pattern, largest magnitude 1."""
    p = (np.sin(theta) ** 2 * np.cos(2.0 * phi)
         + 0.8 * np.sin(theta) * (5.0 * np.cos(theta) ** 2 - 1.0)
         * np.cos(phi - 1.0))
    return p / 1.69316  # its largest magnitude on the sphere


def envelope(r, lo, hi):
    """sin^2 in radius: zero at lo and hi, one halfway."""
    return np.sin(np.pi * np.clip((r - lo) / (hi - lo), 0.0, 1.0)) ** 2


def s40rts(deck, layers, lmax, download):
    """The deck with S40RTS's relative shear-speed perturbation, its mean
    removed, on the layers given (vs lands on vsv and vsh)."""
    from planetmodel import datasets

    try:
        datasets.fetch("S40RTS", download=download)
    except Exception:  # not cached, and no download allowed
        url = datasets.REGISTRY["S40RTS"].sources[0].url
        raise SystemExit(
            "S40RTS is not in planetmodel's cache. Getting it "
            f"downloads {url} (about 0.4 MB) into "
            f"{datasets.cache_directory()}; run again with --download to do "
            "so.") from None
    return datasets.transplanted(deck, "S40RTS", remove_mean=True,
                                 layers=layers, lmax=lmax)


def renormalised(model, layers, grid, radial):
    """The model with each layer's log viscosity perturbation d less
    ln <exp d>, the lateral mean by the grid's quadrature, so that the
    viscosity's lateral mean is the radial reference at every radius:
    the mean is taken at the radial layout's nodes, where the file holds
    the field, and is linear in radius between them."""
    weights = np.asarray(grid.weights)[:, None] * np.ones(grid.nphi)[None, :]
    weights /= weights.sum()
    theta = np.asarray(grid.colatitudes)[None, :, None]
    phi = np.asarray(grid.longitudes)[None, None, :]
    nodes = np.unique(np.asarray(radial.r).ravel())
    for i in layers:
        field = model.layer(i)["viscosity"]
        d = field.perturbation
        lo, hi = field.interval
        r = nodes[(nodes >= lo) & (nodes <= hi)]
        values = d(r[:, None, None], theta, phi)
        log_mean = np.log(np.sum(weights * np.exp(values), axis=(1, 2)))
        less = AnalyticField(
            (lo, hi), lambda r, t, p, d=d, at=r, m=log_mean:
            d(r, t, p) - np.interp(r, at, m), name="viscosity")
        model = model.with_field(i, "viscosity", PerturbedField(
            field.reference, less, kind="log", name="viscosity"),
            replace=True)
    return model


def tomography_viscosity(deck, layers, lmax, download, law, renormalise,
                         lid, radial):
    """S40RTS's viscosity on the deck's Maxwell layers by sl3d's chain,
    renormalised (on the writer's radial layout) and lidded as asked."""
    from planetmodel.conversion import building, lithosphere, parameters

    laws = {"sl3d": parameters.SL3D_VISCOSITY,
            "milne": parameters.MILNE_2018_VISCOSITY}
    model = s40rts(deck, layers, lmax, download)
    dt = building.temperature_anomaly(
        building.velocity_anomaly(model, layers=layers),
        scaling=parameters.SL3D_DENSITY_SCALING,
        expansivity=parameters.SL3D_EXPANSIVITY)
    model = building.with_viscosity(model, dt, laws[law])
    if renormalise:
        model = renormalised(model, layers, gauss_legendre(lmax), radial)
    if lid is not None:
        surface = deck.skeleton.boundaries[-1]
        upper = [i for i in layers
                 if deck.layer(i).interval[0] >= surface - LID_DEPTH - 1.0]
        base = lithosphere.viscosity_threshold(model, lid, layers=upper,
                                               grid=equiangular(90, 180))
        depth = (surface - np.asarray(base.function.values)) / 1e3
        print(f"the lid's base, where the viscosity falls to {lid:g} Pa s: "
              f"from {depth.min():.0f} to {depth.max():.0f} km deep")
        model = lithosphere.with_lid(model, base, viscosity=LID_VISCOSITY)
    return model


def tomography_moduli(deck, reference, layers, lmax, download, vp_scaling):
    """S40RTS's moduli on the deck's mantle layers below the Moho: the
    shear speed by dlnVs, the P speed by dlnVs / vp_scaling, at fixed
    density; each modulus stored as the reference's radial field and its
    relative perturbation. Returns the model and the layers perturbed."""
    layers = [i for i in layers if deck.layer(i).interval[1] <= MOHO + 1.0]
    source = s40rts(deck, layers, lmax, download)
    for i in layers:  # dlnVp = dlnVs / R, on both P speeds
        dlnvs = source.layer(i)["vsv"].perturbation
        for name in ("vpv", "vph"):
            source = perturbed(source, name, dlnvs / vp_scaling,
                               kind="relative", layers=[i])
    moduli = source.isotropic()  # the Voigt averages of the perturbed speeds
    out = reference
    for i in layers:
        for name in ("kappa", "mu"):
            radial = reference.layer(i)[name]
            relative = moduli.layer(i)[name] / radial - 1.0
            out = out.with_field(i, name, PerturbedField(radial, relative,
                                                         kind="relative",
                                                         name=name),
                                 replace=True)
    return out.with_citation(*source.citation), layers


def contrast(path, field: str) -> str:
    """The lateral contrast a written file holds, from the file itself:
    for a viscosity ln(eta / eta_ref) about the stored radial reference,
    its smallest and largest values and its rms (area-weighted, over the
    nodes it is perturbed on); for moduli, each relative perturbation's
    rms and their correlation."""
    part = read_model_file(path).parts[0]
    weights = np.asarray(part.grid.weights)[:, None] * np.ones(
        len(part.grid.longitudes))[None, :]
    weights /= weights.sum()

    def perturbation(name):
        values = np.asarray(part.variables[f"{name}_perturbation"].values)
        return values[np.isfinite(values[:, 0, 0])]

    def rms(x):
        return float(np.sqrt(np.mean(np.sum(weights * x**2, axis=(1, 2)))))

    if field == "viscosity":
        d = perturbation("viscosity")  # ln(eta / eta_ref), a log perturbation
        return (f"ln(eta / eta_ref) from {d.min():.2f} to {d.max():.2f}, "
                f"rms {rms(d):.2f}")
    mu, kappa = perturbation("mu"), perturbation("kappa")
    correlation = (np.sum(weights * mu * kappa)
                   / np.sqrt(np.sum(weights * mu**2) * np.sum(weights * kappa**2)))
    return (f"dln mu rms {rms(mu):.4f}, dln kappa rms {rms(kappa):.4f}, "
            f"their correlation {correlation:.2f}")


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
    parser.add_argument("--vp-scaling", type=float, default=2.0, metavar="R",
                        help="tomography moduli: dlnVs / dlnVp (default 2)")
    parser.add_argument("--law", choices=["sl3d", "milne"], default="sl3d",
                        help="tomography viscosity: the exponential law's "
                             "eps, 0.01 (sl3d) or 0.04 (milne) per K")
    parser.add_argument("--renormalise", action="store_true",
                        help="tomography viscosity: each radius's mean "
                             "viscosity the deck's")
    parser.add_argument("--lid", type=float, default=None, metavar="ETA",
                        help="tomography viscosity: a 1e30 Pa s lid above "
                             "where the viscosity falls to ETA (Pa s)")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.deck.exists():
        raise SystemExit(f"no deck {args.deck} (make_earth_deck.py writes "
                         "data/earth_custom.gia.deck)")
    if args.vp_scaling <= 0.0:
        raise SystemExit("--vp-scaling is a positive ratio dlnVs / dlnVp")
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
    radial = RadialMesh(deck.geometry, drmax=100.0e3, ngll=5)

    what = args.kind
    if args.kind == "tomography":
        if abs(deck.skeleton.boundaries[-1] - 6371.0e3) > 20.0e3:
            raise SystemExit("S40RTS is an Earth model, and this deck is not "
                             "the Earth's size")
        if viscosity:
            model = tomography_viscosity(deck, layers, args.lmax,
                                         args.download, args.law,
                                         args.renormalise, args.lid, radial)
            what = (f"S40RTS, {args.law} law"
                    + (", renormalised" if args.renormalise else "")
                    + (f", lid at {args.lid:g} Pa s" if args.lid else ""))
        else:
            model, layers = tomography_moduli(deck, model, layers, args.lmax,
                                              args.download, args.vp_scaling)
            what = f"S40RTS, dlnVs / dlnVp = {args.vp_scaling:g}"
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
    meta = {"deck": str(args.deck), "field": args.field, "kind": args.kind}
    if args.kind == "tomography":
        meta.update({"vp_scaling": args.vp_scaling} if not viscosity else
                    {"law": args.law, "renormalised": int(args.renormalise),
                     "lid_threshold_pa_s": args.lid or 0.0})
    write_model(out, model, grid=gauss_legendre(args.lmax),
                radial=radial, fields=fields,
                interpolation={n: ("polynomial", "cubic") for n in varying},
                meta=meta, source=Path(__file__).name)
    top = deck.layer(layers[-1]).interval[1]
    wrote(out, f"{args.field}, {what}, lmax {args.lmax}, on "
               f"{len(layers)} layers from {span[0] / 1e3:.0f} to "
               f"{top / 1e3:.0f} km, {out.stat().st_size / 1e6:.1f} MB")
    print(f"  {contrast(out, args.field)}")


if __name__ == "__main__":
    main()
