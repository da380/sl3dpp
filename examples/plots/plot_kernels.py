"""The adjoint-kernel example: run ExampleIceKernels first, then
`poetry run python plot_kernels.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import Example, map_axes, mark_sites, plot_field

ex = Example("kernels.txt")
fig = plt.figure(figsize=(12, 7))

gauge = [(ex.scalars["gauge_lat"], ex.scalars["gauge_lon"])]

# The measurement is J = int g xi dS at the final node: g is the
# Gaussian averaging function centred on the gauge, normalised to unit
# area integral, so J is the cap-weighted MEAN sea-level change there
# (in metres). Shown peak-normalised; its 1/area units carry no insight.
ex.fields["averaging_function"] = (
    ex.fields["site_dual"] / ex.fields["site_dual"].max())
ax = map_axes(fig, 2, 3, 1)
mesh = plot_field(ax, ex, "averaging_function", cmap="Greens",
                  symmetric=False,
                  title="averaging function $g$ of the measurement\n"
                        r"$J = \int g\,\xi\, dS$ (peak-normalised)")
mark_sites(ax, gauge, ["C3"])
fig.colorbar(mesh, ax=ax, shrink=0.7)

for i, (name, title) in enumerate((
        ("thickness_dual_2kyr", "ice kernel at 2 kyr"),
        ("thickness_dual_6kyr", "ice kernel at 6 kyr"),
        ("thickness_dual_10kyr", "ice kernel at 10 kyr"),
        ("rate_kernel_first_interval", "quelled rate kernel"))):
    ax = map_axes(fig, 2, 3, 2 + i)
    mesh = plot_field(ax, ex, name, title=title)
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
