"""The 3D lateral-viscosity example: run ExampleLateralViscosity first,
then `poetry run python plot_lateral.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import Example, world_map, plot_field

ex = Example("lateral.txt")
fig = plt.figure(figsize=(11, 7))

ax = world_map(fig, 2, 2, 1, state=ex)
mesh = plot_field(ax, ex, "delta_log_viscosity", cmap="PuOr",
                  title=r"lateral $\delta\ln\eta$ (one decade peak to peak)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 2, state=ex)
mesh = plot_field(ax, ex, "slc_radial_m",
                  title="sea level change, radial mantle (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 3, state=ex)
mesh = plot_field(ax, ex, "slc_difference_m",
                  title="lateral minus radial (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = fig.add_subplot(2, 2, 4)
t = ex.column("iterations", "t_kyr")
ax.bar(t[1:], ex.column("iterations", "coupled_iterations")[1:], width=0.6)
ax.set_xlabel("time (kyr)")
ax.set_ylabel("within-step iterations")
ax.set_title("the lateral fixed point's cost per step", fontsize=10)

fig.suptitle("3D lateral viscosity against the radial reference (10 kyr)")
fig.tight_layout()
fig.savefig("lateral.png", dpi=180)
print("wrote lateral.png")
