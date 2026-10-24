"""The adjoint-kernel example: run ExampleIceKernels first, then
`poetry run python plot_kernels.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import Example, world_map, mark_sites, plot_field

ex = Example("kernels.txt")
fig = plt.figure(figsize=(12, 7))

gauge = [(ex.scalars["gauge_lat"], ex.scalars["gauge_lon"])]

# The measurement is J = int g xi dS at the final node: g is the
# Gaussian averaging function centred on the gauge, normalised to unit
# area integral, so J is the cap-weighted MEAN sea-level change there
# (in metres). Shown peak-normalised; its 1/area units carry no insight.
ex.fields["averaging_function"] = (
    ex.fields["site_dual"] / ex.fields["site_dual"].max())
ax = world_map(fig, 2, 3, 1, state=ex)
mesh = plot_field(ax, ex, "averaging_function", cmap="Greens",
                  symmetric=False,
                  title="averaging function $g$ of the measurement\n"
                        r"$J = \int g\,\xi\, dS$ (peak-normalised)")
mark_sites(ax, gauge, ["C3"])
fig.colorbar(mesh, ax=ax, shrink=0.7)

# The rate kernels: the observation's response to ice lost during an
# interval of the history (interval k from node k - 1 to node k). The
# kernel is the response to ice ADDED, so it is negated here: positive
# means the gauge's sea level rises when ice is lost there. One colour
# scale for all four, so that their sizes compare. The gauge is in the near
# field, where the solid Earth's response (the elastic fall at once, the
# viscous one over the millennia after) outweighs the eustatic share, so the
# kernels change with the interval.
step = ex.scalars["interval_kyr"]
history = ex.scalars["history_kyr"]
names = []
for k in (1, 4, 7, 10):
    name = f"response_to_loss_{k}"
    ex.fields[name] = -ex.fields[f"rate_kernel_{k}"]
    names.append((k, name))
limit = max(np.max(np.abs(ex.fields[name])) for _, name in names)
# The sensitivity sits on the ice nearest the gauge: a regional view.
region = (280.0, 355.0, 55.0, 85.0)  # west, east, south, north
for i, (k, name) in enumerate(names):
    before = history - (k - 1) * step, history - k * step
    ax = world_map(fig, 2, 3, 2 + i, state=ex, central_longitude=-40.0,
                   extent=region)
    mesh = plot_field(ax, ex, name, limit=limit,
                      title=f"ice lost {before[0]:g}–{before[1]:g} kyr before "
                            "the observation\n(m of sea level per m of ice, "
                            "per m$^2$)")
    mark_sites(ax, gauge, ["C3"])
    fig.colorbar(mesh, ax=ax, shrink=0.7)

# The radial viscosity kernels: the continuous adjoint (the preferred
# path) against the exact discrete transpose — their agreement is the
# resolution diagnostic.
ax = fig.add_subplot(2, 3, 6)
for tag, style in (("continuous", "-"), ("discrete", "--")):
    r = ex.column("viscosity_kernel_" + tag, "r_km")
    k = ex.column("viscosity_kernel_" + tag, "dJ_deta")
    order = np.argsort(r)
    ax.plot(r[order], k[order], style, label=tag)
ax.set_xlabel("radius (km)")
ax.set_ylabel(r"$dJ/d\eta(r)$ (m per Pa s per m)")
ax.legend(fontsize=8)
ax.set_title("radial viscosity kernels", fontsize=10)

fig.suptitle("Adjoint kernels of a single sea-level observation")
fig.tight_layout()
fig.savefig("kernels.png", dpi=180)
print("wrote kernels.png")
