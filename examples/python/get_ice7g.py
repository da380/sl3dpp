"""Writes data/ice7g.nc: the ICE-7G deglaciation history as a planetmodel
surface file, for the examples and the sl3d configs.

This is make_ice_history.py --model ICE7G --download: pyslfp's
export_ice_history reads ICE-7G's stored time slices (every kiloyear from
26 to 21 ka, then every half kiloyear to the present, 48 in all) and
writes the `ice_thickness` and `topography` histories on the dataset's own
grid, in metres, at the stored dates alone. That is the form the
synthetic Earth takes too, so anything run on data/synthetic_earth.nc
runs on data/ice7g.nc:

    ExampleFingerprint --state data/ice7g.nc
    sl3d sea-level configs/sea_level_fingerprint_ice7g.json

Run from build/examples, in the examples' environment:

    poetry install --no-root
    poetry run python get_ice7g.py [--version ICE7G|ICE6G|ICE5G] [--out FILE]

The first run downloads pyslfp's ICE-7G archive from Zenodo (about 24 MB)
into pyslfp's data directory (~/.pyslfp_data, or $PYSLFP_DATA); later runs
reuse it. The file is written into data/ under the current directory (the
examples' inputs), never the source tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import make_ice_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", default="ICE7G",
                        choices=["ICE7G", "ICE6G", "ICE5G"],
                        help="the ice model (default ICE7G)")
    parser.add_argument("--out", type=Path, default=None,
                        help="the file to write (default data/ice7g.nc, or "
                             "data/ice6g.nc / data/ice5g.nc for the other "
                             "versions)")
    args = parser.parse_args()
    make_ice_history.main(["--model", args.version, "--download"]
                          + (["--out", str(args.out)] if args.out else []))


if __name__ == "__main__":
    main()
