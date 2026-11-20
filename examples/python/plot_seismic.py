"""The seismic example: run ExampleSeismic first, then
`poetry run python plot_seismic.py` from build/examples. Reads
output/library/seismic/seismic.txt and writes figures/library/seismic.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import (Example, arrows, library_figure, library_output,
                      mark_fault, mark_relaxation_times, mark_sites,
                      plot_field, relaxation_axis, save_figure, stepped,
                      world_map)

ex = Example(library_output("seismic", "seismic.txt"))
s = ex.scalars
lat0, lon0 = s["centroid_lat"], s["centroid_lon"]
half = 11.0
across = half / np.cos(np.deg2rad(lat0))
extent = (lon0 - across, lon0 + across, lat0 - half, lat0 + half)
places = ex.series["places"]
colors = ["C0", "C1", "C2"]

# The maps' region, with a margin the projected corners reach: drawn as a
# region, the grid's points shaded between.
rows = np.abs(ex.lats - lat0) <= half + 4.0
cols = np.abs((ex.lons - lon0 + 180.0) % 360.0 - 180.0) <= across + 5.0
fields = {name: values[np.ix_(rows, cols)] for name, values in ex.fields.items()}
ocean = fields["ocean"] > 0.5
fields["horizontal_event_m"] = np.hypot(fields["east_event_m"],
                                        fields["north_event_m"])
for name in ("sea_level_event_m", "sea_level_change_m"):
    fields[name] = np.where(ocean, fields[name], np.nan)
region = Example.from_fields(np.deg2rad(90.0 - ex.lats[rows]),
                             np.deg2rad(ex.lons[cols]), fields)
region.state = ex.state


def region_limit(name):
    """The largest magnitude of a field over the maps' region."""
    inside = ((np.abs(region.lats - lat0) <= half)[:, None]
              & (np.abs((region.lons - lon0 + 180.0) % 360.0 - 180.0)
                 <= across)[None, :])
    return float(np.nanmax(np.abs(np.where(inside, region.fields[name],
                                           np.nan))))


fig = plt.figure(figsize=(16, 11))
spec = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.3])
maps = (("up_event_m", "RdBu_r", True, "vertical displacement at the event (m)"),
        ("horizontal_event_m", "YlOrRd", False,
         "horizontal displacement at the event (m)"),
        ("sea_level_event_m", "RdBu_r", True,
         "relative sea-level change at the event,\nover the ocean (m)"),
        ("up_change_m", "RdBu_r", True,
         f"vertical displacement from the event to {s['last_yr']:g}\nyears, "
         "the Maxwell Earth (m)"))
for k, (name, cmap, symmetric, title) in enumerate(maps):
    ax = world_map(fig, 2, 3, (1, 2, 4, 5)[k], state=ex,
                   central_longitude=lon0, extent=extent)
    ax.set_subplotspec(spec[k // 2, k % 2])
    mesh = plot_field(ax, region, name, cmap=cmap, symmetric=symmetric,
                      title=title,
                      limit=region_limit(name) if symmetric else None,
                      shading="gouraud")
    if not symmetric:
        mesh.set_clim(0.0, region_limit(name))
        arrows(ax, region, extent, east="east_event_m", north="north_event_m")
    mark_fault(ax, lat0, lon0, s["strike_deg"], s["dip_deg"], s["length_km"],
               s["width_km"])
    if name == "sea_level_event_m":
        mark_sites(ax, places, colors)
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="m")

# the three places through time on the Maxwell Earth and its Burgers twin,
# each a 2 degree cap mean, from zero before the event, on a time axis
# linear about the event and logarithmic after it: over the fault and the
# gauge above, the far field below
t = ex.column("history", "t_yr")
def years(t):
    """A time in years to two figures, without an exponent."""
    return f"{t:.2g}" if t < 100.0 else f"{t:.0f}"


bands = [(f"M{k + 1}", f"{top:.0f}–{bottom:.0f} km, "
          + f"{eta:.0e}".replace("e+", "e") + " Pa s, Maxwell times "
          f"{years(shortest)}–{years(longest)} yr", shortest, longest, "band")
         for k, (top, bottom, eta, shortest, longest)
         in enumerate(ex.series["layers"])]
bands += [(f"B{k + 1}", f"the Burgers asthenosphere's term {k + 1}, "
           f"{low:.2f} μ at {years(short)}–{years(long)} yr", short, long,
           "line")
          for k, (low, _, short, long) in enumerate(ex.series["burgers"])]
cell = spec[0, 2].subgridspec(2, 1, hspace=0.06, height_ratios=[1.7, 1.0])
top = fig.add_subplot(cell[0])
bottom = fig.add_subplot(cell[1], sharex=top)
labels = ["over the fault", "about the tide gauge", "the far field, west"]
for ax, chosen in ((top, (("fault", 0), ("gauge", 1))),
                   (bottom, (("far", 2),))):
    for where, k in chosen:
        for prefix, ls in (("", "-"), ("burgers_", "--")):
            values = ex.column("history", f"{prefix}up_{where}_m")
            ax.plot(*stepped(t, values), color=colors[k], ls=ls, marker=".",
                    ms=3, markevery=slice(2, None))
    relaxation_axis(ax, t)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_ylabel("uplift (m)")
keys = mark_relaxation_times(top, bands)
mark_relaxation_times(bottom, bands, label=False)
top.legend([plt.Line2D([], [], color=c) for c in colors]
           + [plt.Line2D([], [], color="0.3", ls=ls) for ls in ("-", "--")],
           labels + ["Maxwell", "Burgers asthenosphere"], fontsize=7, ncol=2,
           loc="upper left")
top.tick_params(labelbottom=False)
top.set_title("uplift through time", fontsize=10, pad=14)
bottom.set_xlabel(f"years since the event (linear to {t[1]:g}, "
                  "logarithmic after)\n" + "\n".join(keys), fontsize=7)

# each patch's kernel contracted with its own mechanism
patches = ex.series["patches"]
along = int(s["patches_along_strike"])
down = int(s["patches_down_dip"])
slip = patches[:, 3].reshape(along, down)
depth = patches[:, 2].reshape(along, down)
ax = fig.add_subplot(spec[1, 2])
width = 0.8 / down
for k in range(down):
    ax.bar(np.arange(along) + (k - 0.5 * (down - 1)) * width, slip[:, k],
           width, color=f"C{k + 3}",
           label=f"row {k + 1} down dip, {depth[:, k].min():.0f} to "
                 f"{depth[:, k].max():.0f} km")
ax.axhline(0.0, color="0.6", lw=0.5)
ax.set_xticks(range(along), [f"{k + 1}" for k in range(along)], fontsize=8)
ax.set_xlabel("patch along strike, south to north")
ax.set_ylabel("mm per 10$^{21}$ N m of the patch's slip")
ax.set_title(
    f"the gauge's mean sea level at {s['measured_yr']:g} years "
    f"({s['gauge_m'] * 1e3:.1f} mm):\nits kernel per unit of each patch's "
    "slip", fontsize=10)
ax.legend(fontsize=7, loc="lower right")

fig.suptitle("a thrust off the synthetic South America (600 by 150 km, dipping "
             f"18° east, 2e22 N m as {along} by {down} point sources) with an "
             f"ocean, lmax {s['lmax']:g}:\nprem8.postseismic.deck, a 50 km "
             "elastic lid over a Maxwell mantle, and its twin with a Burgers "
             "asthenosphere (a Kelvin element of μ_K = "
             f"{s['kelvin_modulus_fraction']:g} μ, η_K = "
             f"{s['kelvin_viscosity_fraction']:g} η)\n"
             "maps centred on the fault, its up-dip edge heavy; the places "
             "of the time series marked (blue: over the fault, orange: the "
             "tide gauge, green: the far field)\nmaps and series "
             f"{s['cap_width_deg']:g}° Gaussian means (the fields averaged "
             f"over a cap of {s['cap_width_deg']:g}° about each point; their "
             "point values ring at this lmax)\nthe rake turned from "
             f"{s['rake_deg']:g} to {s['other_rake_deg']:g}°: the gauge's "
             f"change {s['change_by_kernels_m'] * 1e3:.4f} mm by its moment "
             "kernels, a second run agreeing to "
             f"{s['relative_difference']:.0e}", fontsize=11)
fig.tight_layout()
save_figure(fig, library_figure("seismic"))
