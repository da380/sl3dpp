"""Maps of the elastic fingerprint example: run ExampleFingerprint first,
then `poetry run python plot_fingerprint.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import Example, map_axes, plot_field

ex = Example("fingerprint.txt")
fig = plt.figure(figsize=(11, 7))

ax = map_axes(fig, 2, 2, 1)
mesh = plot_field(ax, ex, "ice_change_m", cmap="Blues_r", symmetric=False,
                  title="ice thickness change (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = map_axes(fig, 2, 2, 2)
mesh = plot_field(ax, ex, "sea_level_change_m",
                  title="sea level change (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = map_axes(fig, 2, 2, 3)
mesh = plot_field(ax, ex, "displacement_m", title="vertical displacement (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = map_axes(fig, 2, 2, 4)
mesh = plot_field(ax, ex, "potential_m2s2",
                  title="potential perturbation (m$^2$ s$^{-2}$)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

fig.suptitle("Elastic sea-level fingerprint: 10% Greenland melt (ICE-7G, lmax 256)")
fig.tight_layout()
fig.savefig("fingerprint.png", dpi=180)
print("wrote fingerprint.png")
