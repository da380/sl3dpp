"""The viscoelastic GIA example: run ExampleEvolution first, then
`poetry run python plot_evolution.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import Example, map_axes, mark_sites, plot_field

ex = Example("evolution.txt")
fig = plt.figure(figsize=(11, 7))

# The site histories: elastic jump during the melt, viscous creep after.
ax = fig.add_subplot(2, 2, 1)
t = ex.column("history", "t_kyr")
for name, label in (("slc_near_m", "near field"),
                    ("slc_mid_m", "mid field"),
                    ("slc_far_m", "far field")):
    ax.plot(t, ex.column("history", name), label=label)
ax.axvline(ex.scalars["melt_end_kyr"], color="0.6", linestyle=":",
           label="melting stops")
ax.set_xlabel("time (kyr)")
ax.set_ylabel("sea level change (m)")
ax.legend(fontsize=8)
ax.set_title("site histories", fontsize=10)

ax = fig.add_subplot(2, 2, 2)
ax.plot(t, 1.0e12 * ex.column("history", "omega_x"), label=r"$\omega_x$")
ax.plot(t, 1.0e12 * ex.column("history", "omega_y"), label=r"$\omega_y$")
ax.plot(t, 1.0e12 * ex.column("history", "omega_z"), label=r"$\omega_z$")
ax.set_xlabel("time (kyr)")
ax.set_ylabel(r"$\Delta\omega$ (10$^{-12}$ rad s$^{-1}$)")
ax.legend(fontsize=8)
ax.set_title("rotational response", fontsize=10)

sites = ex.series["sites"]
site_colors = ["C0", "C1", "C2"]  # match the history lines

ax = map_axes(fig, 2, 2, 3)
mesh = plot_field(ax, ex, "slc_end_of_melt_m",
                  title="sea level change at 5 kyr (m)")
mark_sites(ax, sites, site_colors)
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = map_axes(fig, 2, 2, 4)
mesh = plot_field(ax, ex, "slc_final_m",
                  title="sea level change at 15 kyr (m)")
mark_sites(ax, sites, site_colors)
fig.colorbar(mesh, ax=ax, shrink=0.7)

fig.suptitle("1D viscoelastic GIA: West Antarctic melt over 5 kyr, relaxation to 15 kyr")
fig.tight_layout()
fig.savefig("evolution.png", dpi=180)
print("wrote evolution.png")
