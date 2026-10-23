"""Shared loading and mapping for the sl3dpp example plots.

Reads the examples' text format (see examples/Common.hpp): a 'grid'
header with the colatitudes and longitudes, then named 'field', 'series'
and 'scalar' blocks.

A driver records the Earth's state it ran on as a '# state <file>'
comment; world_map then draws the coastline that belongs to the data: the
real one for a real Earth, the synthetic Earth's own for outputs computed
on it (real coastlines would mislead over its toy continents).

Maps use cartopy (Robinson projection, coastlines) when it is importable
— the poetry environment provides it — and fall back to a plain
lat/lon pcolormesh otherwise, so the scripts run anywhere matplotlib
does. Everything is read from and written to the CURRENT directory: run
these from build/examples, never from the source tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cartopy.crs as ccrs

    HAVE_CARTOPY = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_CARTOPY = False


class Example:
    """One example's outputs: fields, series and scalars by name."""

    def __init__(self, path: str):
        self.fields: dict[str, np.ndarray] = {}
        self.series: dict[str, np.ndarray] = {}
        self.scalars: dict[str, float] = {}
        self.state: str | None = None  # the Earth's state the run used
        self.lats = None
        self.lons = None
        with open(path) as f:
            lines = iter(f.readlines())
        nlat = nlon = 0
        for line in lines:
            line = line.strip()
            if line.startswith("# state "):
                self.state = line[len("# state "):].strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "grid":
                nlat, nlon = int(parts[1]), int(parts[2])
                colat = np.array(next(lines).split(), dtype=float)
                lon = np.array(next(lines).split(), dtype=float)
                self.lats = 90.0 - np.rad2deg(colat)
                self.lons = np.rad2deg(lon)
            elif parts[0] == "field":
                rows = [next(lines).split() for _ in range(nlat)]
                self.fields[parts[1]] = np.array(rows, dtype=float)
            elif parts[0] == "series":
                n = int(parts[2])
                header = next(lines)
                columns = header.lstrip("# ").split()
                rows = [next(lines).split() for _ in range(n)]
                data = np.array(rows, dtype=float)
                self.series[parts[1]] = data
                self.series[parts[1] + ":columns"] = columns
            elif parts[0] == "scalar":
                self.scalars[parts[1]] = float(parts[2])

    @classmethod
    def from_fields(cls, colatitudes, longitudes, fields: dict) -> "Example":
        """An Example from arrays rather than a text file: colatitudes and
        longitudes in radians, and fields as (nlat, nlon) arrays by name —
        as read from the planetmodel files the sl3d program writes."""
        example = cls.__new__(cls)
        example.fields = {k: np.asarray(v, dtype=float)
                          for k, v in fields.items()}
        example.series, example.scalars = {}, {}
        example.lats = 90.0 - np.rad2deg(np.asarray(colatitudes, dtype=float))
        example.lons = np.rad2deg(np.asarray(longitudes, dtype=float))
        return example

    def column(self, series: str, name: str) -> np.ndarray:
        """One named column of a series."""
        columns = self.series[series + ":columns"]
        return self.series[series][:, columns.index(name)]


def map_axes(fig, rows, cols, index, central_longitude=0.0,
             coastlines=True):
    """An axes ready for a world map: cartopy Robinson when available,
    centred on the prime meridian by default (pyslfp's convention), with
    the real coastlines unless told otherwise (a synthetic Earth draws its
    own)."""
    if HAVE_CARTOPY:
        ax = fig.add_subplot(
            rows, cols, index,
            projection=ccrs.Robinson(central_longitude=central_longitude))
        ax.set_global()
        if coastlines:
            try:  # coastlines need a one-off download; stay quiet offline
                ax.coastlines(linewidth=0.4, color="0.3")
            except Exception:
                pass
        return ax
    return fig.add_subplot(rows, cols, index)


def is_synthetic(state) -> bool:
    """Whether a state file is the build's synthetic Earth (the default
    when an output names none)."""
    return state is None or Path(state).name == "synthetic_earth.nc"


def world_map(fig, rows, cols, index, state=None, central_longitude=0.0):
    """A map axes with the coastline that belongs to the data: for the
    synthetic Earth, its own present-day coastline (where its topography
    crosses sea level, read from synthetic_earth.nc in the current
    directory); for any other state, the real one. `state` is a state file
    name, or an Example, whose '# state' comment is used."""
    if isinstance(state, Example):
        state = state.state
    if not is_synthetic(state):
        return map_axes(fig, rows, cols, index, central_longitude)
    ax = map_axes(fig, rows, cols, index, central_longitude, coastlines=False)
    try:
        from planetmodel.surfacefile import read_surface_file

        file = read_surface_file(Path(state or "synthetic_earth.nc"))
        topography = np.asarray(file.variables["topography"].values)[-1]
        lats = 90.0 - np.rad2deg(np.asarray(file.grid.colatitudes))
        lons = np.rad2deg(np.asarray(file.grid.longitudes))
        kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        ax.contour(lons, lats, topography, levels=[0.0], colors="0.3",
                   linewidths=0.5, **kwargs)
    except Exception:  # no planetmodel or no file: a map without a coast
        pass
    return ax


def plot_field(ax, example: Example, name: str, *, cmap="RdBu_r",
               symmetric=True, title=None):
    """The field as a filled map; returns the mesh for a colorbar."""
    field = example.fields[name]
    lons, lats = example.lons, example.lats
    # close the longitude seam for a tidy global plot
    field = np.column_stack([field, field[:, 0]])
    lons = np.append(lons, lons[0] + 360.0)
    kwargs = {"cmap": cmap, "shading": "auto"}
    if symmetric:
        top = np.max(np.abs(field))
        kwargs["vmin"], kwargs["vmax"] = -top, top
    if HAVE_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    mesh = ax.pcolormesh(lons, lats, field, **kwargs)
    if not HAVE_CARTOPY:
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    ax.set_title(title if title is not None else name, fontsize=10)
    return mesh


def mark_sites(ax, latlons, colors=None):
    """Points at (lat, lon) degrees on a map made by map_axes."""
    kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    for k, (lat, lon) in enumerate(latlons):
        color = colors[k] if colors is not None else "k"
        ax.plot(lon, lat, "o", markersize=5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                **kwargs)
