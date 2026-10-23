"""The Python's side of the kernel-file round trip.

Reads a kernel model file written by sl3dpp's io/KernelWriter — the
TestKernelWriter smooth kernel, written when SL3DPP_KERNEL_FILE_OUT
is set:

    SL3DPP_KERNEL_FILE_OUT=/tmp/kernel.nc build/tests/TestKernelWriter

— integrates the kernel against the test perturbation
p = 0.3 + sin^2(theta) cos(2 phi) - 0.5 cos(theta) by its own
quadrature (Gauss-Legendre per file element in r, the file grid's
weights in angle), and compares with the in-memory pairing the C++
stored as meta_pairing. With --plot, draws the kernel at mid-mantle.
Run in this directory's environment (pyproject.toml):

    cd scripts && poetry install --no-root
    poetry run python check_kernel_file.py /tmp/kernel.nc [--plot out.png]
"""
from __future__ import annotations

import sys

import netCDF4 as nc
import numpy as np
from planetmodel.modelfile import read_model


def perturbation(theta, phi):
    return (0.3 + np.sin(theta) ** 2 * np.cos(2.0 * phi)
            - 0.5 * np.cos(theta))


def main() -> int:
    path = sys.argv[1]
    plot = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--plot" else None

    ds = nc.Dataset(path)
    pairing = float(ds.getncattr("meta_pairing"))
    length = float(ds.getncattr("meta_length_scale"))
    back = read_model(path)

    total = 0.0
    kernel_layers = [lay for lay in back.layers if "log_viscosity_kernel" in lay]
    for g in ds.groups:
        if not g.startswith("part_"):
            continue
        part = ds.groups[g]
        if "log_viscosity_kernel" not in part.variables:
            continue
        theta = np.deg2rad(90.0 - np.array(part.variables["latitude"][:]))
        phi = np.deg2rad(np.array(part.variables["longitude"][:]))
        w = np.array(part.variables["colatitude_weights"][:])
        T, P = np.meshgrid(theta, phi, indexing="ij")
        p = perturbation(T, P)
        angular = 2.0 * np.pi / len(phi) * w[:, None] * p
        bounds = np.array(part.variables["element_bounds"][:]).reshape(-1, 2)
        x, gw = np.polynomial.legendre.leggauss(12)
        for lo, hi in bounds:
            lay = next(l for l in kernel_layers
                       if l.interval[0] <= 0.5 * (lo + hi) <= l.interval[1])
            field = lay["log_viscosity_kernel"]
            r = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x
            jac = 0.5 * (hi - lo)
            for rk, wk in zip(r, gw):
                vals = np.asarray(field(np.full(T.shape, rk), T, P))
                total += jac * wk * rk * rk * float(np.sum(vals * angular))

    # The objective is dimensionless, so the SI volume integral IS the
    # non-dimensional pairing; length_scale stays in the meta for a
    # dimensioned objective's reader.
    del length
    rel = abs(total - pairing) / abs(pairing)
    print(f"integral {total:.15e}  meta_pairing {pairing:.15e}  rel {rel:.2e}")

    if plot is not None:
        lay = kernel_layers[len(kernel_layers) // 2]
        field = lay["log_viscosity_kernel"]
        part = next(ds.groups[g] for g in ds.groups
                    if "log_viscosity_kernel" in ds.groups[g].variables)
        theta = np.deg2rad(90.0 - np.array(part.variables["latitude"][:]))
        phi = np.deg2rad(np.array(part.variables["longitude"][:]))
        T, P = np.meshgrid(theta, phi, indexing="ij")
        r = 0.5 * sum(lay.interval)
        vals = np.asarray(field(np.full(T.shape, r), T, P))
        try:
            from planetmodel.plotting import sphere_map
            fig = sphere_map(np.rad2deg(np.pi / 2 - theta), np.rad2deg(phi),
                             vals, title=f"log-viscosity kernel at r = {r:.3g} m")
            fig.savefig(plot, dpi=150, bbox_inches="tight")
        except Exception:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            m = ax.pcolormesh(np.rad2deg(phi), 90.0 - np.rad2deg(theta), vals,
                              cmap="RdBu_r")
            fig.colorbar(m, ax=ax, label="kernel (SI)")
            ax.set_title(f"log-viscosity kernel at r = {r:.3g} m")
            fig.savefig(plot, dpi=150, bbox_inches="tight")
        print(f"drew {plot}")

    return 0 if rel < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
