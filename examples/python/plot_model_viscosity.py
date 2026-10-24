"""The model-file viscosity example: run ExampleModelViscosity first, then
`poetry run python plot_model_viscosity.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import Example, world_map, mark_sites, plot_field

ex = Example("model_viscosity.txt")
sites = [(ex.scalars["site0_lat"], ex.scalars["site0_lon"]),
         (ex.scalars["site1_lat"], ex.scalars["site1_lon"])]
colors = ["C0", "C1"]
fig = plt.figure(figsize=(12, 7.5))

ax = world_map(fig, 2, 3, 1, state=ex)
mesh = plot_field(ax, ex, "delta_1000km", cmap="PuOr",
                  title=r"$\delta\ln\eta$ at 1000 km depth (model file)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 3, 2, state=ex)
mesh = plot_field(ax, ex, "ice_21ka_m", cmap="Blues", symmetric=False,
                  title="ice thickness at 21 ka (m)")
mark_sites(ax, sites, colors)
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 3, 3, state=ex)
mesh = plot_field(ax, ex, "slc_lateral_m",
                  title="sea-level change since 21 ka, 3D mantle (m)")
mark_sites(ax, sites, colors)
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 3, 4, state=ex)
mesh = plot_field(ax, ex, "slc_difference_m",
                  title="3D minus radial (m)")
mark_sites(ax, sites, colors)
fig.colorbar(mesh, ax=ax, shrink=0.7)

# The sea-level histories at the two sites: the 3D run against the radial
# twin. Time runs from 21 ka (left) to the present (right).
ax = fig.add_subplot(2, 3, 5)
t = ex.column("sites", "t_kyr_bp")
for (name, label), color in zip((("hudson", "Hudson Bay"),
                                 ("norway", "southern Norway")), colors):
    ax.plot(t, ex.column("sites", name + "_lateral_m"), "-", color=color,
            label=label + ", 3D")
    ax.plot(t, ex.column("sites", name + "_radial_m"), "--", color=color,
            label=label + ", radial")
ax.invert_xaxis()
ax.set_xlabel("kyr before present")
ax.set_ylabel("sea-level change since 21 ka (m)")
ax.legend(fontsize=7)
ax.set_title("site histories", fontsize=10)

ax = fig.add_subplot(2, 3, 6)
ax.bar(t[1:], ex.column("sites", "coupled_iterations")[1:], width=0.6)
ax.invert_xaxis()
ax.set_xlabel("kyr before present")
ax.set_ylabel("within-step iterations")
ax.set_title("the lateral fixed point's cost per step", fontsize=10)

fig.suptitle("A 3D viscosity from a planetmodel model file, under a "
             "deglaciation from 21 ka")
fig.tight_layout()
fig.savefig("model_viscosity.png", dpi=180)
print("wrote model_viscosity.png")
