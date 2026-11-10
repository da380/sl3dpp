"""The 3D viscosity-kernel example: run ExampleViscosityKernels3D first,
then `poetry run python plot_kernels_3d.py` from build/examples. Reads
output/library/viscosity_kernels_3d/viscosity_kernels_3d.txt and writes
figures/library/viscosity_kernels_3d.png.

The full kernels are in viscosity_kernels.nc beside the text, a
planetmodel model file (planetmodel's Python reads, draws and integrates
it); this script draws the depth slices and radial profiles the example
also writes as text."""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import (Example, library_figure, library_output, mark_sites,
                      plot_field, save_figure, world_map)

ex = Example(library_output("viscosity_kernels_3d",
                            "viscosity_kernels_3d.txt"))
site = [(ex.scalars["site_lat"], ex.scalars["site_lon"])]

rows = (("K_ln_eta_sea_level", "sea-level change", "m per km$^3$"),
        ("K_ln_eta_decay_time", "decay time", "yr per km$^3$"))
depths = sorted({int(m.group(1)) for name in ex.fields
                 if (m := re.search(r"_(\d+)km$", name))})

fig = plt.figure(figsize=(15, 6.5))
cols = len(depths) + 1
for row, (stem, label, unit) in enumerate(rows):
    for col, depth in enumerate(depths):
        ax = world_map(fig, 2, cols, row * cols + col + 1,
                      central_longitude=-90.0, state=ex)
        mesh = plot_field(ax, ex, f"{stem}_{depth}km",
                          title=f"{label}: {depth} km depth\n({unit})")
        mark_sites(ax, site, ["C3"])
        # exponents in the tick labels, not an offset above the bar, which
        # would collide with the title
        fig.colorbar(mesh, ax=ax, shrink=0.7, format="%.1e")

# The radial profiles: the angular integral, dJ/dr per km of radius — what
# a uniform change of ln(viscosity) at each radius does to each datum.
r = ex.column("radial", "r_km")
for row, (name, label, unit) in enumerate((
        ("sea_level_per_km", "sea-level change", "m per km"),
        ("decay_time_per_km", "decay time", "yr per km"))):
    ax = fig.add_subplot(2, cols, (row + 1) * cols)
    ax.plot(ex.column("radial", name), r, ".", markersize=2)
    ax.set_xlabel(f"$dJ/dr$ ({unit})")
    ax.set_ylabel("radius (km)")
    ax.set_title(f"{label}: radial profile", fontsize=10)

fig.suptitle(
    "3D log-viscosity kernels of two observations at the marked site "
    f"(sea-level change since 21 ka {ex.scalars['sea_level_change_m']:.0f} m, "
    f"decay time {ex.scalars['decay_time_kyr']:.2f} kyr)")
fig.tight_layout()
save_figure(fig, library_figure("viscosity_kernels_3d"))
