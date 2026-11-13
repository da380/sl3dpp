"""The seismic example: run ExampleSeismic first, then
`poetry run python plot_seismic.py` from build/examples. Reads
output/library/seismic/seismic.txt and writes figures/library/seismic.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import (Example, arrows, library_figure, library_output,
                      mark_fault, mark_sites, plot_field, save_figure,
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


fig = plt.figure(figsize=(15, 9.5))
maps = (("up_event_m", "RdBu_r", True, "vertical displacement at the event (m)"),
        ("horizontal_event_m", "YlOrRd", False,
         "horizontal displacement at the event (m)"),
        ("sea_level_event_m", "RdBu_r", True,
         "relative sea-level change at the event,\nover the ocean (m)"),
        ("up_change_m", "RdBu_r", True,
         f"vertical displacement from the event\nto {s['last_yr']:g} years (m)"))
for k, (name, cmap, symmetric, title) in enumerate(maps):
    ax = world_map(fig, 2, 3, (1, 2, 4, 5)[k], state=ex,
                   central_longitude=lon0, extent=extent)
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

# the three places through time, from zero before the event: over the
# fault above, the gauge and the far field below
t = ex.column("history", "t_yr")
lead = np.r_[-0.04 * t[-1], 0.0, t]
cell = fig.add_gridspec(2, 3)[0, 2].subgridspec(2, 1, hspace=0.08)
top = fig.add_subplot(cell[0])
bottom = fig.add_subplot(cell[1], sharex=top)
labels = ["over the fault", "the tide gauge's 2° mean", "the far field, west"]
for ax, chosen in ((top, (("fault", 0),)),
                   (bottom, (("gauge", 1), ("far", 2)))):
    for where, k in chosen:
        for quantity, ls in (("up", "-"), ("sea", "--")):
            values = ex.column("history", f"{quantity}_{where}_m")
            ax.plot(lead, np.r_[0.0, 0.0, values], color=colors[k], ls=ls)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_ylabel("m")
top.legend([plt.Line2D([], [], color=c) for c in colors]
           + [plt.Line2D([], [], color="0.3", ls=ls) for ls in ("-", "--")],
           labels + ["uplift", "relative sea level"], fontsize=7, ncol=2,
           loc="center right")
top.tick_params(labelbottom=False)
top.set_title("uplift (solid) and relative sea level (dashed)", fontsize=10)
bottom.set_xlabel("years since the event")

# each patch's kernel contracted with its own mechanism
patches = ex.series["patches"]
along = int(s["patches_along_strike"])
down = int(s["patches_down_dip"])
slip = patches[:, 3].reshape(along, down)
depth = patches[:, 2].reshape(along, down)
ax = fig.add_subplot(2, 3, 6)
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
             f"18° east, 2e22 N m as {along} by {down} point sources) on prem8 "
             f"with a 1e21 Pa s mantle and an ocean, lmax {s['lmax']:g}\n"
             "maps centred on the fault, its up-dip edge heavy; the places "
             "of the time series marked (blue: over the fault, orange: the "
             "tide gauge, green: the far field)\nthe rake turned from "
             f"{s['rake_deg']:g} to {s['other_rake_deg']:g}°: the gauge's "
             f"change {s['change_by_kernels_m'] * 1e3:.4f} mm by its moment "
             "kernels, a second run agreeing to "
             f"{s['relative_difference']:.0e}", fontsize=11)
fig.tight_layout()
save_figure(fig, library_figure("seismic"))
