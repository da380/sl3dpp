"""Writes a surface mass density (kg m^-2) as a planetmodel surface file,
for a `field` load term of the sl3d program's `response` command: static,
or with --history on a time axis.

--shape disk (the default): a disk of uniform surface density, --radius-deg
wide at (--lat, --lon), --amplitude kg m^-2 (default 1.834e6, about 2 km of
ice), sampled on a 1-degree latitude-longitude grid. It is a sampled step,
not band-limited: sl3d's own `disk` load term is the exact expansion, and
this one shows the route from an array to a load file.

--shape cap: a Gaussian cap, A exp(-d^2 / (2 w^2)) with d the distance
from (--lat, --lon), w = --width-deg (default 5) and A = --amplitude,
sampled at the nodes of the Gauss-Legendre grid of degree --lmax, which is
the run's own grid when --lmax is the run's: the run then reads the field
at its own nodes, and the term is sl3d's own `cap` load term of the same
members, to rounding (the configs response_load_field.json and
response_load_cap.json show it).

--shape greenland: the direct load of melting --fraction (default 0.1) of
the Greenland ice sheet of ICE-7G today, pyslfp's EarthState.greenland_load
(the ice density times the thickness change where the ice is grounded), on
pyslfp's Driscoll-Healy grid of degree --lmax: a mass loss, so negative.
It needs pyslfp's ICE-7G data (~/.pyslfp_data); if they are not there the
script says what it would download and stops, unless --download is given.

--history [T:F ...] multiplies the pattern by the factor F at the time T,
in kiloyears on the response run's clock (default 0:0 1:1 5:1 5.5:0: the
load grows over the first kiloyear, is held to 5 kyr and removed by 5.5),
and writes a node field linear in time. The file's time axis is in
kiloyears: the run's time t kyr reads it at t - time_origin_kyr (the
term's member, default 0), so with the default the file's time 0 is the
run's start; outside the file's span its first or last field is held. A
file whose axis counts from the present instead, negative in the past (an
ice history, a sea-level output), is read from its date -D kyr at the
run's start by "time_origin_kyr": D.

Run from build/examples:

    poetry run python make_load_field.py [--shape disk|cap|greenland]
        [--lat 60] [--lon -85] [--radius-deg 8] [--width-deg 5]
        [--amplitude 1.834e6] [--fraction 0.1] [--lmax 128] [--download]
        [--history [T:F ...]] [--out FILE]

It writes data/load_<shape>.nc, or data/load_<shape>_history.nc with
--history. A `response` config's forcing then takes it as a load term,
{"field": {"file": "../data/load_disk.nc", "name": "load"}}, elastic (no
`times`) for a static file and viscoelastic (with `times`) for one with a
time axis, which is refused without `times`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from planetmodel import gauss_legendre

from inputs import (DATA, distance_deg, grid_from_latlon, need_pyslfp_data,
                    write_load, wrote)


def disk(lat0: float, lon0: float, radius: float, amplitude: float):
    """A sampled disk on a 1-degree grid: the grid and the values on it."""
    lat = np.arange(89.5, -90.0, -1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    lats, lons = np.meshgrid(lat, lon, indexing="ij")
    load = np.where(distance_deg(lats, lons, lat0, lon0) <= radius,
                    amplitude, 0.0)
    grid, orient = grid_from_latlon(lat, lon)
    return grid, orient(load)


def cap(lat0: float, lon0: float, width: float, amplitude: float,
        lmax: int):
    """A Gaussian cap sampled on the Gauss-Legendre grid of degree lmax:
    the grid and the values on it."""
    grid = gauss_legendre(lmax)
    lats = 90.0 - np.rad2deg(np.asarray(grid.colatitudes))[:, None]
    lons = np.rad2deg(np.asarray(grid.longitudes))[None, :]
    d = distance_deg(lats, lons, lat0, lon0)
    return grid, amplitude * np.exp(-0.5 * (d / width) ** 2)


def greenland(fraction: float, lmax: int, download: bool):
    """pyslfp's load of melting a fraction of Greenland's ice, in SI, on its
    Driscoll-Healy grid: the grid and the values on it."""
    need_pyslfp_data("ICE7G", download)
    import pyslfp
    from planetmodel import Dimensions
    from planetmodel.sampling import driscoll_healy

    state = pyslfp.EarthState.from_defaults(lmax=lmax)
    to_si = state.model.parameters.scales.factor(Dimensions(mass=1, length=-2))
    load = state.greenland_load(fraction=fraction) * to_si
    values = np.array(load.data[:-1, :-1])  # no south pole row, no 360 column
    values[0, :] = values[0, :].mean()  # the north pole: one value
    return driscoll_healy(load.lmax, sampling=load.sampling), values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shape", choices=["disk", "cap", "greenland"],
                        default="disk")
    parser.add_argument("--lat", type=float, default=60.0)
    parser.add_argument("--lon", type=float, default=-85.0)
    parser.add_argument("--radius-deg", type=float, default=8.0)
    parser.add_argument("--width-deg", type=float, default=5.0,
                        help="the cap's width w, degrees")
    parser.add_argument("--amplitude", type=float, default=1.834e6,
                        help="the disk's surface density, or the cap's at "
                             "its centre, kg m^-2")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="the fraction of Greenland's ice melted")
    parser.add_argument("--lmax", type=int, default=128,
                        help="the degree of the cap's Gauss-Legendre "
                             "grid, or of pyslfp's grid (greenland)")
    parser.add_argument("--download", action="store_true",
                        help="let pyslfp download ICE-7G (greenland)")
    parser.add_argument("--history", nargs="*", metavar="T:F", default=None,
                        help="factors F at times T (kyr on the run's clock)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.shape == "disk":
        grid, load = disk(args.lat, args.lon, args.radius_deg, args.amplitude)
        what = (f"a {args.radius_deg:g} degree disk of {args.amplitude:g} "
                f"kg m^-2 at ({args.lat:g}, {args.lon:g})")
    elif args.shape == "cap":
        grid, load = cap(args.lat, args.lon, args.width_deg, args.amplitude,
                         args.lmax)
        what = (f"a Gaussian cap of width {args.width_deg:g} degrees and "
                f"{args.amplitude:g} kg m^-2 at ({args.lat:g}, {args.lon:g}),"
                f" on the Gauss-Legendre grid of degree {args.lmax}")
    else:
        grid, load = greenland(args.fraction, args.lmax, args.download)
        what = f"{args.fraction:g} of Greenland's ice melted (ICE-7G)"
    stem = f"load_{args.shape}" + ("_history" if args.history is not None
                                   else "")
    out = args.out or DATA / f"{stem}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.history is None:
        write_load(out, grid, load, title=what)
        wrote(out, f"{what}, static")
        return
    pairs = [p.split(":") for p in (args.history or
                                    ["0:0", "1:1", "5:1", "5.5:0"])]
    times = np.array([float(t) for t, _ in pairs])
    factors = np.array([float(f) for _, f in pairs])
    write_load(out, grid, factors[:, None, None] * load[None],
               times_kyr=times, title=what)
    wrote(out, f"{what}, at {times.size} times from {times[0]:g} to "
               f"{times[-1]:g} kyr on the run's clock")


if __name__ == "__main__":
    main()
