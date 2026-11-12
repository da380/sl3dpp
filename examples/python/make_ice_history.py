"""Writes an ice history and its present-day state as planetmodel surface
files for the sl3d program: from ICE-5G, ICE-6G or ICE-7G through pyslfp,
or a hand-made one from numpy arrays.

--model ICE7G (or ICE6G, ICE5G): pyslfp's export_ice_history writes the
model's stored slices (for ICE-7G every kiloyear from 26 to 21 ka, then
every half kiloyear to the present), `ice_thickness` and `topography` in
metres on the data set's grid, to data/ice7g.nc (data/ice6g.nc,
data/ice5g.nc). The file holds the present day too, so it is both the ice
history (`forcing.ice`) and the sea level, as topography
(`ocean.topography`), of a `sea-level` config: configs/*_ice7g.json read
data/ice7g.nc. pyslfp keeps
the data in ~/.pyslfp_data (or $PYSLFP_DATA); if they are not there the
script says what it would download and stops, unless --download is given.

--synthetic: a disk-shaped ice sheet on a round continent in a 4 km deep
ocean, built as numpy arrays on a 1-degree latitude-longitude grid
(latitude ascending, longitude from -180) and turned into the files sl3d
reads. The sheet, a parabolic dome of --radius-deg and --peak-m at
(--lat, --lon), grows linearly from nothing at 26 ka to its maximum at
21 ka and melts linearly away by 10 ka; there is a slice every kiloyear.
The topography is the surface height, the ice included, as in the ICE-xG
files: the bedrock height plus the ice. Two files are written:
data/synthetic_disk.nc, the history (both fields, on a time axis in
kiloyears relative to the present, negative in the past), and
data/synthetic_disk_present.nc, the present-day state as static fields (a
state of one date is written static, not on a time axis of one node).
Run from build/examples:

    poetry run python make_ice_history.py --model ICE7G [--download]
        [--out data/ice7g.nc]
    poetry run python make_ice_history.py --synthetic [--lat 60]
        [--lon -85] [--radius-deg 15] [--peak-m 3000]
        [--out data/synthetic_disk.nc]

then a `sea-level` config through time ("problem": "quasi_static", with
"times": {"start_kyr_bp": 26, "nodes": 27}) takes the file for both its ice
history and its topography, "forcing": {"ice": {"file":
"../data/synthetic_disk.nc"}} and "ocean": {"topography": {"file":
"../data/synthetic_disk.nc", "at": "start"}}, and a fingerprint ("problem":
"static") takes the present-day file for both, "ice": {"file":
"../data/synthetic_disk_present.nc", ...} and "ocean": {"topography":
{"file": "../data/synthetic_disk_present.nc"}}, with no date: its fields
are static.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inputs import (DATA, distance_deg, grid_from_latlon, need_pyslfp_data,
                    write_ice_history, write_state, wrote)


def from_ice_model(version: str, out: Path, download: bool) -> None:
    """A stored ICE-NG model through pyslfp."""
    need_pyslfp_data(version, download)
    from pyslfp.ice import export_ice_history, standard_dates

    dates = standard_dates(version)
    written = export_ice_history(out, version=version)
    wrote(written, f"{version}, {len(dates)} dates from {max(dates):g} ka to "
                   f"the present, {written.stat().st_size / 1e6:.1f} MB")


def synthetic(out: Path, lat0: float, lon0: float, radius: float,
              peak: float) -> None:
    """The hand-made history: arrays (time, lat, lon) to surface files."""
    lat = np.arange(-89.5, 90.0, 1.0)  # cell centres, south to north
    lon = np.arange(-180.0, 180.0, 1.0) + 0.5
    lats, lons = np.meshgrid(lat, lon, indexing="ij")
    continent = distance_deg(lats, lons, lat0, lon0)
    bedrock = -4000.0 + 4500.0 * np.exp(-((continent / 30.0) ** 6))
    dome = np.sqrt(np.clip(1.0 - (distance_deg(lats, lons, lat0, lon0)
                                  / radius) ** 2, 0.0, None))
    dome *= bedrock > 0.0  # grounded on land alone

    dates = np.arange(26.0, -0.5, -1.0)  # kyr BP, a slice every kiloyear
    size = np.interp(-dates, [-26.0, -21.0, -10.0], [0.0, 1.0, 0.0])
    ice = peak * size[:, None, None] * dome[None]  # (time, lat, lon)
    topography = np.where(bedrock > 0.0, bedrock + ice, bedrock)

    grid, orient = grid_from_latlon(lat, lon)
    write_ice_history(out, grid, dates, orient(ice), orient(topography),
                      title=f"a disk ice sheet at ({lat0:g}, {lon0:g}): grows "
                            "26-21 ka, melts 21-10 ka")
    wrote(out, f"{dates.size} dates from 26 ka to the present, "
               f"{ice.max():.0f} m at most")
    present = out.with_name(out.stem + "_present.nc")
    write_state(present, grid, orient(ice[-1]), orient(topography[-1]),
                title="the disk ice sheet's present day")
    wrote(present, "the present-day state, static")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", choices=["ICE7G", "ICE6G", "ICE5G"],
                       help="a stored ice model through pyslfp")
    which.add_argument("--synthetic", action="store_true",
                       help="the hand-made disk ice sheet")
    parser.add_argument("--download", action="store_true",
                        help="let pyslfp download the model's data")
    parser.add_argument("--lat", type=float, default=60.0)
    parser.add_argument("--lon", type=float, default=-85.0)
    parser.add_argument("--radius-deg", type=float, default=15.0)
    parser.add_argument("--peak-m", type=float, default=3000.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="the file to write (default data/ice7g.nc, "
                             "data/ice6g.nc, data/ice5g.nc or "
                             "data/synthetic_disk.nc)")
    args = parser.parse_args(argv)
    if args.model:
        out = args.out or DATA / f"{args.model.lower()}.nc"
    else:
        out = args.out or DATA / "synthetic_disk.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.model:
        from_ice_model(args.model, out, args.download)
    else:
        synthetic(out, args.lat, args.lon, args.radius_deg, args.peak_m)


if __name__ == "__main__":
    main()
