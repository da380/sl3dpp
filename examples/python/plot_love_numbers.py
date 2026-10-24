"""The Love-number example: run ExampleLoveNumbers first, then
`poetry run python plot_love_numbers.py` from build/examples."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotutil import Example

ex = Example("love_numbers.txt")
l = ex.column("love", "l")
h = ex.column("love", "h_load")
ll = ex.column("love", "l_load")
k = ex.column("love", "k_load")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# The load numbers, scaled as Farrell (1972) plots them: h', l l' and l k'
# each tend to a constant at large degree, set by the elastic structure
# near the surface. Degree 0 and 1 (the centre-of-mass frame) are left out.
ax = axes[0]
mask = l >= 2
for values, label in ((h, r"$h'_l$"), (l * ll, r"$l\,l'_l$"),
                      (l * k, r"$l\,k'_l$")):
    ax.semilogx(l[mask], values[mask], label=label)
ax.axhline(0.0, color="0.7", linewidth=0.6)
ax.set_xlabel("degree $l$")
ax.set_title("load Love numbers (Farrell's scaling)", fontsize=10)
ax.legend(fontsize=9)

# The tidal numbers, defined from degree 2.
ax = axes[1]
tidal = l >= 2
for name, label in (("h_tide", "$h_l$"), ("l_tide", "$l_l$"),
                    ("k_tide", "$k_l$")):
    ax.semilogx(l[tidal], ex.column("love", name)[tidal], label=label)
ax.axhline(0.0, color="0.7", linewidth=0.6)
ax.set_xlabel("degree $l$")
ax.set_title("tidal Love numbers", fontsize=10)
ax.legend(fontsize=9)

two = np.argmax(l == 2)
fig.suptitle(
    "Elastic Love numbers of the PREM-like deck "
    f"(degree 2: $h'$ = {h[two]:.3f}, $k'$ = {k[two]:.3f}, "
    f"$k$ = {ex.column('love', 'k_tide')[two]:.3f})")
fig.tight_layout()
fig.savefig("love_numbers.png", dpi=180)
print("wrote love_numbers.png")
