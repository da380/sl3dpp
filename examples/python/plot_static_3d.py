"""The 3D static response: the synthetic lateral shear modulus and the
radial displacement under a Hudson Bay cap load as planetmodel cutaways
of the body, and the surface displacement with its departure from the
radial model's as maps: run ExampleStatic3D first, then
`poetry run python plot_static_3d.py` from build/examples. Reads
data/synthetic_moduli.nc, and static_3d.txt and static_3d_volume.nc from
output/library/static_3d/; writes figures/library/static_3d.png."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import slices
from planetmodel.modelfile import read_model
from planetmodel.plotting import cutaway

from plotutil import (DATA, Example, library_figure, library_output,
                      mark_sites, plot_field, save_figure, world_map)

ex = Example(library_output("static_3d", "static_3d.txt"))
site = [(ex.scalars["site_lat"], ex.scalars["site_lon"])]

moduli = read_model(DATA / "synthetic_moduli.nc")
response = read_model(library_output("static_3d", "static_3d_volume.nc"))
cmb = 3480.0e3
# a wedge over the Americas, the northern hemisphere cut, the core left
# standing: the craton under Hudson Bay on the faces and the surface
wedge = dict(longitudes=(np.deg2rad(230.0), np.deg2rad(320.0)),
             colatitudes=(0.0, 0.5 * np.pi), inner=cmb, n=700)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(2, 2, 1)
cut = slices.cutaway(moduli, "mu", relative_to="mean", departure="percent",
                     **wedge)
cutaway(ax, cut, scale=1e-3, inner="0.75", colorbar=True, diverging=True)
ax.set_title("the synthetic shear modulus, % departure from the mean",
             fontsize=10)

ax = fig.add_subplot(2, 2, 2)
cut = slices.cutaway(response, "radial_displacement", **wedge)
cutaway(ax, cut, scale=1e-3, inner="0.75", colorbar=True, diverging=True)
ax.set_title("radial displacement under the Hudson Bay cap (m)", fontsize=10)

extent = (-140.0, -30.0, 25.0, 85.0)
ax = world_map(fig, 2, 2, 3, central_longitude=-85.0, state=ex, extent=extent)
mesh = plot_field(ax, ex, "displacement_m",
                  title="surface displacement, 3D model (m)")
mark_sites(ax, site, ["C3"])
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 4, central_longitude=-85.0, state=ex, extent=extent)
mesh = plot_field(ax, ex, "displacement_difference_m",
                  title="3D minus radial model (m)")
mark_sites(ax, site, ["C3"])
fig.colorbar(mesh, ax=ax, shrink=0.7)

fig.suptitle(
    "the static response of a laterally heterogeneous elastic Earth to a cap "
    f"load ({int(ex.scalars['cg_iterations'])} conjugate-gradient iterations)")
fig.tight_layout()
save_figure(fig, library_figure("static_3d"))
