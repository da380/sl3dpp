"""Maps of the elastic fingerprint example: run ExampleFingerprint first,
then `poetry run python plot_fingerprint.py` from build/examples. Reads
output/library/fingerprint/fingerprint.txt and writes
figures/library/fingerprint.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotutil import (Example, library_figure, library_output, plot_field,
                      save_figure, world_map)

ex = Example(library_output("fingerprint", "fingerprint.txt"))
fig = plt.figure(figsize=(11, 7))

ax = world_map(fig, 2, 2, 1, state=ex)
mesh = plot_field(ax, ex, "ice_change_m", cmap="Blues_r", symmetric=False,
                  title="ice thickness change (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 2, state=ex)
mesh = plot_field(ax, ex, "sea_level_change_m",
                  title="sea level change (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 3, state=ex)
mesh = plot_field(ax, ex, "displacement_m", title="vertical displacement (m)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

ax = world_map(fig, 2, 2, 4, state=ex)
mesh = plot_field(ax, ex, "potential_m2s2",
                  title="potential perturbation (m$^2$ s$^{-2}$)")
fig.colorbar(mesh, ax=ax, shrink=0.7)

fig.suptitle("Elastic sea-level fingerprint: 10% Greenland melt")
fig.tight_layout()
save_figure(fig, library_figure("fingerprint"))
