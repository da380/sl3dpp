"""Shared pieces of the make_*.py scripts, which write inputs for the sl3d
program with planetmodel and pyslfp: the two radial deck formats sl3d
reads, a deck read back as a planetmodel model, an angular grid from
latitude and longitude arrays, the surface files of ice histories,
present-day states and loads, and the check that a pyslfp data set is on
disk before anything is downloaded.

Every path is relative to the current directory, build/examples, whose
data/ holds the examples' inputs (docs/sl3d-configs.md describes the
files sl3d reads).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from planetmodel import (MAXWELL, AngularGrid, DeckFormat, Dimensions,
                         MaxwellDeckModel, constant_field, equiangular,
                         read_deck, sample_deck, write_deck)
from planetmodel.surfacefile import SurfaceVariable, write_surface
from planetmodel.units import LENGTH, Scales

DATA = Path("data")
KYR = 1.0e3 * 365.25 * 86400.0  # s: sl3d's (and pyslfp's) kiloyear
SURFACE_DENSITY = Dimensions(mass=1, length=-2)  # kg m^-2

# sl3d's "elastic" deck: r rho vpv vsv vph vsh eta, SI, no header lines.
ELASTIC = DeckFormat(("rho", "vpv", "vsv", "vph", "vsh", "eta"),
                     name="elastic", header_lines=0)


def wrote(path, what: str = "") -> Path:
    """Says that a file was written."""
    print(f"wrote {path}" + (f" ({what})" if what else ""))
    return Path(path)


# ------------------------------------------------------------------ decks
def with_viscosity(model, layers, viscosity: float):
    """The model with a constant viscosity (Pa s) on the layers given, each
    made isotropic first, its moduli replaced by their Voigt averages: sl3d
    makes every layer holding a viscosity a Maxwell body, and refuses one
    that is transversely isotropic."""
    iso = model.isotropic()
    for i in layers:
        for name in ("kappa", "mu"):
            model = model.with_field(i, name, iso.layer(i)[name], replace=True)
        model = model.with_field(i, "viscosity",
                                 constant_field(model.layer(i).interval,
                                                viscosity, name="viscosity"))
    return model


def write_decks(model, stem) -> tuple[Path, Path]:
    """The model as sl3d's two radial decks: <stem>.gia.deck, planetmodel's
    Maxwell (GIA) format with the viscosity `nan` in elastic and fluid
    layers ("format": "maxwell"), and <stem>.elastic.deck, the elastic
    structure alone ("format": "elastic")."""
    gia = write_deck(f"{stem}.gia.deck", sample_deck(model, MAXWELL.columns),
                     MAXWELL)
    elastic = write_deck(f"{stem}.elastic.deck",
                         sample_deck(model, ELASTIC.columns), ELASTIC)
    return wrote(gia), wrote(elastic)


def read_deck_model(path):
    """A deck sl3d reads, as a planetmodel model, its format from its name:
    *.elastic.deck the six columns with no header, anything else the
    Maxwell format (two header lines, a remark after the column names
    allowed, as data/prem8.gia.deck has)."""
    if Path(path).name.endswith(".elastic.deck"):
        return MaxwellDeckModel(read_deck(path, ELASTIC), kinds={})
    return MaxwellDeckModel(read_deck(path, MAXWELL.columns, header_lines=2))


# ------------------------------------------------------- surface files
def distance_deg(lat, lon, lat0: float, lon0: float):
    """Great-circle distance in degrees from (lat0, lon0), all in degrees."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    lat0, lon0 = np.deg2rad(lat0), np.deg2rad(lon0)
    c = (np.sin(lat) * np.sin(lat0)
         + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0))
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def grid_from_latlon(lat_deg, lon_deg):
    """The planetmodel grid of a regular latitude-longitude grid given by
    its two axes in degrees, in any order (latitude ascending or
    descending, longitude in [-180, 180) or [0, 360)), and a function that
    reorders an (..., nlat, nlon) array onto it (colatitude increasing,
    longitude increasing from its smallest value in [0, 360))."""
    lat, lon = np.asarray(lat_deg, float), np.asarray(lon_deg, float)
    flip = lat[0] < lat[-1]
    wrapped = np.mod(lon, 360.0)
    roll = int(np.argmin(wrapped))
    theta = np.deg2rad(90.0 - (lat[::-1] if flip else lat))
    phi = np.deg2rad(np.roll(wrapped, -roll))
    grid = equiangular(lat.size, lon.size, phi0=float(phi[0]))
    if not (np.allclose(grid.colatitudes, theta, atol=1e-12)
            and np.allclose(grid.longitudes, phi, atol=1e-12)):
        grid = AngularGrid(theta, phi)  # not cell-centred: a custom grid

    def orient(a):
        a = np.asarray(a, float)
        return np.roll(a[..., ::-1, :] if flip else a, -roll, axis=-1)

    return grid, orient


def write_ice_history(path, grid, dates_kyr_bp, ice_thickness, topography,
                      *, title: str):
    """An ice history sl3d reads as `forcing.ice.file` (and, holding the
    topography, as `forcing.state.file`): node fields in metres, linear
    in time, on a time axis in kiloyears relative to the present (negative
    in the past) as pyslfp's export_ice_history writes. The arrays are
    (ntime, nlat, nlon) on the grid, the dates in kyr BP in any order, at
    least two of them."""
    dates = np.asarray(dates_kyr_bp, float)
    order = np.argsort(-dates)  # oldest first: the time axis increases
    node = dict(support="node", time_interpolation="linear",
                dimensions=LENGTH)
    fields = {
        "ice_thickness": SurfaceVariable(np.asarray(ice_thickness)[order],
                                         meaning="ice thickness", **node),
        "topography": SurfaceVariable(
            np.asarray(topography)[order],
            meaning="surface height above sea level, the ice included",
            **node)}
    return write_surface(Path(path), grid, fields=fields, title=title,
                         source=Path(__file__).name, time=-dates[order],
                         scales=Scales(time=KYR),
                         meta={"time": "kiloyears relative to the present, "
                                       "negative in the past"})


def write_state(path, grid, ice_thickness, topography, *, title: str):
    """A present-day state sl3d reads as `forcing.state.file`: static
    `ice_thickness` and `topography`, metres, (nlat, nlon) on the grid."""
    fields = {
        "ice_thickness": SurfaceVariable(ice_thickness, dimensions=LENGTH,
                                         meaning="ice thickness"),
        "topography": SurfaceVariable(
            topography, dimensions=LENGTH,
            meaning="surface height above sea level, the ice included")}
    return write_surface(Path(path), grid, fields=fields, title=title,
                         source=Path(__file__).name)


def write_load(path, grid, load, *, times_kyr=None, title: str):
    """A surface mass density (kg m^-2) for a `field` load term of `sl3d
    response`, named `load`: static, (nlat, nlon), or with `times_kyr` a
    node field linear in time, (ntime, nlat, nlon), on a time axis in
    kiloyears."""
    if times_kyr is None:
        variable = SurfaceVariable(load, dimensions=SURFACE_DENSITY,
                                   meaning="surface mass density")
        return write_surface(Path(path), grid, fields={"load": variable},
                             title=title, source=Path(__file__).name)
    variable = SurfaceVariable(load, support="node",
                               time_interpolation="linear",
                               dimensions=SURFACE_DENSITY,
                               meaning="surface mass density")
    return write_surface(Path(path), grid, fields={"load": variable},
                         title=title, source=Path(__file__).name,
                         time=np.asarray(times_kyr, float),
                         scales=Scales(time=KYR),
                         meta={"time": "kiloyears on the run's clock"})


# --------------------------------------------------------- data on disk
# The unpacked sizes of pyslfp's ice data sets, for the message below.
PYSLFP_SIZES = {"ICE7G": "about 24 MB to download, 37 MB unpacked",
                "ICE6G": "72 MB unpacked", "ICE5G": "30 MB unpacked"}


def need_pyslfp_data(key: str, download: bool) -> None:
    """Stops, saying what would be downloaded and where, unless pyslfp's
    data set `key` (ICE7G, ICE6G, ICE5G) is already on disk or `download`
    allows pyslfp to fetch it."""
    from pyslfp.data.config import DATADIR
    from pyslfp.data.downloader import DATASET_URLS, FOLDER_MAP

    folder = DATADIR / FOLDER_MAP[key]
    if download or (folder.is_dir() and any(folder.iterdir())):
        return
    raise SystemExit(
        f"pyslfp's {key} data are not in {folder}. Getting them downloads "
        f"{DATASET_URLS[key]} ({PYSLFP_SIZES.get(key, 'tens of MB')}) "
        f"into pyslfp's data directory, {DATADIR}; run again with "
        "--download to do so.")
