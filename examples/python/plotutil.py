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
does. Every path is relative to the CURRENT directory, which is
build/examples (never the source tree): the inputs are in data/, a
library example's outputs in output/library/<example>/, and the figures
go to figures/library/ and figures/sl3d/<config>/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cartopy.crs as ccrs

    HAVE_CARTOPY = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_CARTOPY = False


DATA = Path("data")
SYNTHETIC_EARTH = DATA / "synthetic_earth.nc"


def library_output(example: str, name: str) -> Path:
    """A file a library example wrote: output/library/<example>/<name>."""
    return Path("output") / "library" / example / name


def library_figure(name: str) -> Path:
    """A library example's figure: figures/library/<name>.png."""
    return Path("figures") / "library" / f"{name}.png"


def save_figure(fig, path: Path, dpi: int = 180) -> None:
    """Writes a figure, creating its directory, and says so."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    print(f"wrote {path}")


class Example:
    """One example's outputs: fields, series and scalars by name."""

    def __init__(self, path: str | Path):
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
    centred on the prime meridian by default, with the real coastlines
    unless told otherwise (a synthetic Earth draws its own)."""
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


def world_map(fig, rows, cols, index, state=None, central_longitude=0.0,
              extent=None):
    """A map axes with the coastline that belongs to the data: for the
    synthetic Earth, its own present-day coastline (where its topography
    crosses sea level, read from data/synthetic_earth.nc); for any other
    state, the real one. `state` is a state file name, or an Example, whose
    '# state' comment is used. `extent`, as
    (west, east, south, north) in degrees, shows a region instead of the
    globe."""
    if isinstance(state, Example):
        state = state.state
    if not is_synthetic(state):
        ax = map_axes(fig, rows, cols, index, central_longitude)
        _show_region(ax, extent)
        return ax
    ax = map_axes(fig, rows, cols, index, central_longitude, coastlines=False)
    _show_region(ax, extent)
    try:
        from planetmodel.surfacefile import read_surface_file

        file = read_surface_file(Path(state) if state else SYNTHETIC_EARTH)
        topography = np.asarray(file.variables["topography"].values)[-1]
        lats = 90.0 - np.rad2deg(np.asarray(file.grid.colatitudes))
        lons = np.rad2deg(np.asarray(file.grid.longitudes))
        kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        ax.contour(lons, lats, topography, levels=[0.0], colors="0.3",
                   linewidths=0.5, **kwargs)
    except Exception:  # no planetmodel or no file: a map without a coast
        pass
    return ax


def _show_region(ax, extent):
    """Narrows a map axes to a region, given as (west, east, south, north)
    in degrees; the globe when None."""
    if extent is None:
        return
    if HAVE_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(extent[0] % 360.0, extent[1] % 360.0)
        ax.set_ylim(extent[2], extent[3])


def plot_field(ax, example: Example, name: str, *, cmap="RdBu_r",
               symmetric=True, title=None, limit=None, shading="auto"):
    """The field as a filled map; returns the mesh for a colorbar. A
    symmetric scale runs to the field's largest magnitude, or to `limit`
    when given (one scale across several panels). A field over a region
    (longitudes not spanning the globe) is drawn as it is. `shading` is
    pcolormesh's: "gouraud" interpolates between the points, for a
    regional map of a coarse grid."""
    field = example.fields[name]
    lons, lats = example.lons, example.lats
    spacing = abs(lons[1] - lons[0]) if lons.size > 1 else 360.0
    if lons[-1] - lons[0] + 1.5 * spacing >= 360.0:
        # close the longitude seam for a tidy global plot
        field = np.column_stack([field, field[:, 0]])
        lons = np.append(lons, lons[0] + 360.0)
    kwargs = {"cmap": cmap, "shading": shading}
    if symmetric:
        top = limit if limit is not None else np.nanmax(np.abs(field))
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


def destination(lat: float, lon: float, azimuth: float,
                distance_km: float) -> tuple[float, float]:
    """The point a great-circle distance from (lat, lon) along an azimuth
    (degrees clockwise from north) on a sphere of the Earth's mean radius,
    the other way for a negative distance; its longitude in [0, 360)."""
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    a, d = np.deg2rad(azimuth), distance_km / 6371.0
    phi2 = np.arcsin(np.sin(phi) * np.cos(d)
                     + np.cos(phi) * np.sin(d) * np.cos(a))
    lam2 = lam + np.arctan2(np.sin(a) * np.sin(d) * np.cos(phi),
                            np.cos(d) - np.sin(phi) * np.sin(phi2))
    return float(np.rad2deg(phi2)), float(np.rad2deg(lam2)) % 360.0


def mark_fault(ax, lat: float, lon: float, strike: float, dip: float,
               length_km: float, width_km: float) -> None:
    """A planar fault on a map made by map_axes, given by its centroid,
    strike, dip and size (degrees, km): the surface projection of its
    plane, its up-dip edge heavy (a fault dips to the right of its
    strike)."""
    half_length = 0.5 * length_km
    half_across = 0.5 * width_km * np.cos(np.deg2rad(dip))
    corners = []
    for along, across in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
        middle = destination(lat, lon, strike, along * half_length)
        corners.append(destination(*middle, strike + 90.0, across * half_across))
    lats, lons = np.array(corners).T
    kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    ax.plot(lons, lats, color="k", lw=0.9, zorder=6, **kwargs)
    ax.plot(lons[:2], lats[:2], color="k", lw=2.6, zorder=6, **kwargs)


def arrows(ax, example: Example, extent, east: str = "east",
           north: str = "north", spacing: float = 1.6) -> None:
    """A horizontal displacement, the example's fields `east` and `north`
    (m), as arrows about `spacing` degrees apart over a map region (west,
    east, south, north in degrees), the largest a third of an inch long,
    those under a twentieth of it left out, with a key."""
    west, east_edge, south, north_edge = extent
    lats, lons = example.lats, example.lons
    step = max(1, int(round(spacing / abs(lats[1] - lats[0]))))
    rows = np.where((lats >= south) & (lats <= north_edge))[0][::step]
    cols = np.where(((lons - west) % 360.0) <= (east_edge - west))[0][::step]
    u = example.fields[east][np.ix_(rows, cols)]
    v = example.fields[north][np.ix_(rows, cols)]
    size = np.hypot(u, v)
    largest = float(np.max(size))
    shown = size >= 0.05 * largest
    at_lons, at_lats = np.meshgrid(lons[cols], lats[rows])
    key = float(f"{largest / 2.0:.0e}")  # a round length near half the largest
    kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    quiver = ax.quiver(at_lons[shown], at_lats[shown], u[shown], v[shown],
                       color="k", scale=3.0 * largest, scale_units="inches",
                       width=0.004, zorder=5, **kwargs)
    ax.quiverkey(quiver, 0.08, 0.05, key, f"{key:g} m", labelpos="E",
                 coordinates="axes", fontproperties={"size": 8})


# --------------------------------------------------- relaxation through time
def relaxation_axis(ax, times) -> None:
    """A time axis for a relaxation from a co-seismic step at times[0]:
    linear from a little before the event to the first node after it, and
    logarithmic beyond (symlog), so that a decay over decades bends where a
    linear axis would draw it as a line; ticks as plain numbers."""
    import matplotlib.ticker

    times = np.asarray(times, float)
    first = float(np.min(times[times > times[0]]) - times[0])
    ax.set_xscale("symlog", linthresh=first, linscale=0.5)
    ax.set_xlim(times[0] - 0.7 * first, times[-1] * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x:g}"))


def stepped(times, values):
    """A node series drawn from zero before the event, on relaxation_axis:
    the co-seismic step at times[0] a vertical rise."""
    times = np.asarray(times, float)
    first = float(np.min(times[times > times[0]]) - times[0])
    return (np.r_[times[0] - 0.5 * first, times[0], times],
            np.r_[0.0, 0.0, values])


def mark_relaxation_times(ax, bands, label: bool = True) -> list[str]:
    """Marks on a time axis each band of relaxation times, (key,
    description, shortest, longest, style): style "band" a grey band over
    [shortest, longest], "line" a dashed line at their geometric middle;
    with `label`, each key above the axes at its band's middle. Returns
    "key: description" for every band, those beyond the axis saying so."""
    left, right = ax.get_xlim()
    keys = []
    for key, text, shortest, longest, style in bands:
        if shortest > right:
            keys.append(f"{key}: {text} (beyond the axis)")
            continue
        keys.append(f"{key}: {text}")
        middle = np.sqrt(shortest * longest)
        if style == "band":
            ax.axvspan(shortest, min(longest, right), color="0.86", lw=0,
                       zorder=0)
        else:
            ax.axvline(middle, color="0.55", lw=0.8, ls=(0, (4, 3)),
                       zorder=0)
        if label:
            ax.text(middle, 1.01, key, ha="center", va="bottom", fontsize=7,
                    color="0.3", transform=ax.get_xaxis_transform())
    return keys
