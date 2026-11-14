"""Figures of the sl3d example configs' outputs: run configs/run_all.sh first
(and, for the ICE-7G figures, get_ice7g.py, which makes run_all.sh run the
ICE-7G configs too), then `poetry run python plot_app.py` from
build/examples. Each config's outputs, in configs/out/<config>/, are read
with planetmodel's own readers, and its figures are written to
figures/sl3d/<config>/, those comparing a run with gravity and its twin
without to figures/sl3d/with_and_without_gravity/:

    love_numbers/love_numbers.png
    sea_level_fingerprint/fingerprint.png
    sea_level_fingerprint_3d/fingerprint.png
    sea_level_1d/sea_level.png        (sea_level_1d against sea_level_3d)
    sea_level_1d_calibrated/calibration.png
    sea_level_3d_calibrated/calibration.png
    kernel1d/kernels.png
    kernel/viscosity_kernels.png
    kernel1d_lateral/viscosity_kernels.png
    no_ocean/displacement.png
    no_ocean_viscoelastic/displacement.png
    no_ocean_1d_disk/displacement.png
    no_ocean_1d_tide/love_numbers.png
    no_ocean_tide_3d/body_tide.png
    no_ocean_tide_oscillating/oscillating.png   (with the _short and _long
                                                 runs' amplitudes and lags)
    sea_level_fingerprint_ice_cap/ocean.png     (with no_ocean_ice_cap)
    no_ocean_cap_weak/relaxation.png            (with no_ocean_cap_strong)
    timestep_study/accuracy.png                 (the timestep_* runs)
    sea_level_static_coseismic/coseismic.png
    sea_level_static_megathrust/coseismic.png   (with planetmodel's fault
                                                 module, after 1.5.1)
    sea_level_postseismic/postseismic.png
    sea_level_postseismic_burgers/postseismic.png
    no_ocean_postseismic/postseismic.png
    no_gravity_postseismic/postseismic.png
    no_ocean_seismic/postseismic.png
    kernel_postseismic/kernels.png
    with_and_without_gravity/postseismic.png    (no_gravity_postseismic
                                                 against no_ocean_postseismic)
    with_and_without_gravity/load.png           (no_gravity_load against
                                                 no_ocean_load_cap)
    sea_level_tide_migrating/sea_level.png

and, where the ICE-7G runs exist, sea_level_fingerprint_ice7g/fingerprint.png,
sea_level_1d_ice7g/sea_level.png and
sea_level_1d_ice7g_calibrated/calibration.png; where those on the custom
PREM deck with S40RTS exist too, sea_level_fingerprint_3d_ice7g/fingerprint.png
and sea_level_3d_ice7g/sea_level.png (with sea_level_1d_ice7g_custom); where
the toy planet's runs exist (make_planet_deck.py),
no_ocean_tide_planet/body_tide.png and
no_ocean_tide_planet_relax/love_numbers.png; and where the load from a file
and from a table have run, no_ocean_load_field/agreement.png (with
no_ocean_load_cap and no_ocean_load_coefficients). A missing output skips
its figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import equiangular
from planetmodel.harmonics import (analyse_grid, synthesise,
                                  synthesise_gradient, synthesise_grid)
from planetmodel.modelfile import read_model_file
from planetmodel.surfacefile import read_surface_file

from inputs import distance_deg, maxwell_times
from plotutil import (DATA, HAVE_CARTOPY, Example, arrows, destination,
                      map_axes, mark_fault, mark_relaxation_times, mark_sites,
                      plot_field, relaxation_axis, save_figure, stepped,
                      world_map)

if HAVE_CARTOPY:
    import cartopy.crs as ccrs


CONFIGS = Path("configs")  # the configs themselves
OUT = CONFIGS / "out"  # a config's outputs are in OUT / <config>
FIGURES = Path("figures/sl3d")  # its figures in FIGURES / <config>
ICE7G = str(DATA / "ice7g.nc")
KYR = 1.0e3 * 365.25 * 86400.0  # s
G = 6.6743e-11  # m^3 kg^-1 s^-2

# The synthetic Earth's places the configs measure (latitude, longitude).
HUDSON_BAY = (60.0, 275.0)
FENNOSCANDIA = (64.0, 22.0)
FAR_FIELD = (-17.5, 210.5)


def surface(path: Path, fields, index=None) -> Example:
    """Fields of a surface file as an Example for plotutil's maps: static
    fields as they are, node fields at the given time index."""
    file = read_surface_file(path)
    grid = file.grid
    arrays = {}
    for name in fields:
        values = np.asarray(file.variables[name].values)
        arrays[name] = values if index is None else values[index]
    return Example.from_fields(grid.colatitudes, grid.longitudes, arrays)


def save(fig, config: str, name: str) -> None:
    """Writes a config's figure, figures/sl3d/<config>/<name>.png, and
    closes it."""
    save_figure(fig, FIGURES / config / f"{name}.png", dpi=150)
    plt.close(fig)


def nearest(grid, lat: float, lon: float) -> tuple[int, int]:
    """The grid point nearest to a place."""
    i = int(np.argmin(np.abs(np.asarray(grid.colatitudes)
                             - np.deg2rad(90.0 - lat))))
    j = int(np.argmin(np.abs(np.asarray(grid.longitudes)
                             - np.deg2rad(lon % 360.0))))
    return i, j


def love_numbers() -> None:
    path = OUT / "love_numbers" / "love_numbers.txt"
    if not path.exists():
        return
    header = {}
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            parts = line[1:].split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    data = np.array(rows)
    names = header["l"]
    column = {name: data[:, k + 1] for k, name in enumerate(names)}
    n = data[:, 0]
    a = float(header["radius_m"][0])
    g = float(header["surface_gravity_m_s2"][0])
    # The table's numbers are per unit surface density, with the physical
    # potential (which includes the load's own); the classical load Love
    # numbers divide by the load's potential 4 pi G a sigma / (2n + 1) and
    # drop its own term, the tidal ones are per unit potential.
    f = (2 * n + 1) / (4 * np.pi * G * a)
    h_load = (column["h_u"] + column["h_phi"]) * g * f
    l_load = (column["l_u"] + column["l_phi"]) * g * f
    k_load = -(column["k_u"] + column["k_phi"]) * f - 1.0
    keep = n >= 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(n[keep], h_load[keep], label="$h'_n$")
    ax.semilogx(n[keep], n[keep] * l_load[keep], label="$n\\,l'_n$")
    ax.semilogx(n[keep], n[keep] * k_load[keep], label="$n\\,k'_n$")
    ax.set_xlabel("degree n")
    ax.set_title("load Love numbers")
    ax.axhline(0.0, color="0.7", linewidth=0.6)
    ax.legend()
    ax = axes[1]
    ax.semilogx(n[keep], -column["h_t"][keep] * g, label="$h_n$")
    ax.semilogx(n[keep], -column["l_t"][keep] * g, label="$l_n$")
    ax.semilogx(n[keep], column["k_t"][keep], label="$k_n$")
    ax.set_xlabel("degree n")
    ax.set_title("tidal Love numbers")
    ax.axhline(0.0, color="0.7", linewidth=0.6)
    ax.legend()
    fig.suptitle("sl3d love-numbers: the prem8 deck, elastic")
    fig.tight_layout()
    save(fig, "love_numbers", "love_numbers")


def sea_level_fingerprint(name: str, title: str, synthetic: bool) -> None:
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    ex = surface(path, ["ice_change", "relative_sea_level_change", "displacement",
                        "potential"])
    fig = plt.figure(figsize=(11, 7))
    panels = [
        ("ice_change", "ice thickness change (m)", "Blues_r", False),
        ("relative_sea_level_change", "sea level change (m)", "RdBu_r", True),
        ("displacement", "vertical displacement (m)", "RdBu_r", True),
        ("potential", "potential perturbation (m$^2$ s$^{-2}$)", "RdBu_r",
         True),
    ]
    for k, (field, label, cmap, symmetric) in enumerate(panels):
        ax = world_map(fig, 2, 2, k + 1,
                       state=None if synthetic else ICE7G)
        mesh = plot_field(ax, ex, field, cmap=cmap, symmetric=symmetric,
                          title=label)
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, name, "fingerprint")


def sea_level_fingerprint_3d() -> None:
    """The fingerprint on the synthetic 3D elastic Earth: the sea level
    change and the vertical displacement, each beside its 3D minus radial
    difference, the lateral moduli's part of the fingerprint (the radial
    model's fingerprint of the same melt is in the same file)."""
    name = "sea_level_fingerprint_3d"
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    fields = {}
    for field in ("relative_sea_level_change", "displacement"):
        values = np.asarray(file.variables[field].values)
        radial = np.asarray(file.variables[f"{field}_radial_model"].values)
        fields[field] = values
        fields[f"{field}_difference"] = 1e2 * (values - radial)  # cm
    ex = Example.from_fields(grid.colatitudes, grid.longitudes, fields)
    melt = (72.0, 318.0)  # the melted cap
    structure = [(58.0, -95.0), (-15.0, -110.0)]  # the craton, the ridge

    fig = plt.figure(figsize=(12, 7))
    panels = (
        ("relative_sea_level_change", "sea level change, 3D model (m)"),
        ("relative_sea_level_change_difference", "sea level, 3D minus radial model "
         "(cm)"),
        ("displacement", "vertical displacement, 3D model (m)"),
        ("displacement_difference", "displacement, 3D minus radial model "
         "(cm)"),
    )
    # the differences over the North Atlantic, where the lateral part lies
    extent = (-130.0, 20.0, 35.0, 88.0)
    for k, (field, title) in enumerate(panels):
        difference = field.endswith("_difference")
        ax = world_map(fig, 2, 2, k + 1,
                       central_longitude=-55.0 if difference else 0.0,
                       extent=extent if difference else None)
        mesh = plot_field(ax, ex, field, title=title)
        mark_sites(ax, [melt] + structure, colors=["k", "C0", "C3"])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    fig.suptitle("sl3d solve, the fingerprint on the synthetic 3D elastic "
                 "Earth: 10% of the synthetic Greenland melted\n(black: the "
                 "melt; blue: the stiff craton; red: the soft ridge)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, name, "fingerprint")


def histories(path: Path):
    """A dated run's grid, node dates (kyr BP) and sea-level change."""
    file = read_surface_file(path)
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    sea_level = np.asarray(file.variables["relative_sea_level_change"].values)
    return file.grid, dates, sea_level


def sea_level() -> None:
    """The 1D and 3D deglaciations side by side, in the folder of the first
    that ran (sea_level_1d's)."""
    runs = [(name, OUT / name / f"{name}.nc")
            for name in ("sea_level_1d", "sea_level_3d")]
    runs = [(name, path) for name, path in runs if path.exists()]
    if not runs:
        return
    sites = [("Hudson Bay", HUDSON_BAY), ("Fennoscandia", FENNOSCANDIA),
             ("far field", FAR_FIELD)]
    fig = plt.figure(figsize=(12, 7.5))
    for k, (name, path) in enumerate(runs):
        grid, dates, sea_level = histories(path)
        ex = Example.from_fields(grid.colatitudes, grid.longitudes,
                                 {"relative_sea_level_change": sea_level[-1]})
        ax = world_map(fig, 2, 2, k + 1)
        mesh = plot_field(ax, ex, "relative_sea_level_change",
                          title=f"{name}: sea level change, "
                                f"{dates[0]:g} ka to the present (m)")
        mark_sites(ax, [site for _, site in sites], colors=["C0", "C1", "C2"])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 1, 2)
    for k, (label, site) in enumerate(sites):
        for name, path in runs:
            grid, dates, sea_level = histories(path)
            i, j = nearest(grid, *site)
            ax.plot(dates, sea_level[:, i, j], color=f"C{k}",
                    linestyle="-" if name == "sea_level_1d" else "--",
                    label=f"{label} ({name})")
    ax.invert_xaxis()
    ax.set_xlabel("kyr before present")
    ax.set_ylabel("sea level change since the start (m)")
    ax.axhline(0.0, color="0.7", linewidth=0.6)
    ax.legend(fontsize=8, ncol=3)
    fig.suptitle("sl3d solve, 1D (solid) and 3D viscosity (dashed): the "
                 "synthetic deglaciation")
    fig.tight_layout()
    save(fig, runs[0][0], "sea_level")


def sea_level_calibrated(name: str, reference: str, state: str | None,
                         title: str) -> None:
    """A migrating-shoreline run whose initial sea level was calibrated
    against the present day, beside the uncalibrated run from the state
    file's own initial sea level: what the calibration changed at the
    start, and where the ocean moved over the history."""
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    calibrated = np.asarray(file.variables["initial_sea_level"].values)
    ocean = np.asarray(file.variables["ocean_function"].values)
    sea_level = np.asarray(file.variables["relative_sea_level_change"].values)
    fields = {"flooded": ocean[-1] - ocean[0],
              "relative_sea_level_change": sea_level[-1]}
    reference_path = OUT / reference / f"{reference}.nc"
    if reference_path.exists():
        held = read_surface_file(reference_path).variables
        if "initial_sea_level" in held:
            fields["change"] = calibrated - np.asarray(
                held["initial_sea_level"].values)
    ex = Example.from_fields(grid.colatitudes, grid.longitudes, fields)
    fig = plt.figure(figsize=(12, 7.5))
    if "change" in fields:
        ax = world_map(fig, 2, 2, 1, state=state)
        mesh = plot_field(ax, ex, "change",
                          title="calibrated initial sea level minus the "
                                f"state file's at {dates[0]:g} ka (m)")
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 2, state=state)
    mesh = plot_field(ax, ex, "flooded",
                      title=f"ocean function, present minus {dates[0]:g} ka")
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 3, state=state)
    mesh = plot_field(ax, ex, "relative_sea_level_change",
                      title=f"sea level change, {dates[0]:g} ka to the "
                            "present (m)")
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    ax.text(0.0, 0.9, "initial sea level: " + file.meta["initial_sea_level"],
            wrap=True, fontsize=9, va="top")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, name, "calibration")


def sea_level_ice7g() -> None:
    path = OUT / "sea_level_1d_ice7g" / "sea_level_1d_ice7g.nc"
    if not path.exists():
        return
    grid, dates, sea_level = histories(path)
    sites = [("Hudson Bay", (60.0, 275.0)), ("Gulf of Bothnia", (64.0, 21.0)),
             ("Barbados", (13.1, 300.4))]
    fig = plt.figure(figsize=(12, 7.5))
    for k, index in enumerate([dates.size // 2, dates.size - 1]):
        ex = Example.from_fields(grid.colatitudes, grid.longitudes,
                                 {"relative_sea_level_change": sea_level[index]})
        ax = world_map(fig, 2, 2, k + 1, state=ICE7G)
        mesh = plot_field(ax, ex, "relative_sea_level_change",
                          title=f"sea level change, {dates[0]:g} to "
                                f"{dates[index]:g} ka (m)")
        mark_sites(ax, [site for _, site in sites], colors=["C0", "C1", "C2"])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 1, 2)
    for k, (label, site) in enumerate(sites):
        i, j = nearest(grid, *site)
        ax.plot(dates, sea_level[:, i, j], color=f"C{k}", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("kyr before present")
    ax.set_ylabel(f"sea level change since {dates[0]:g} ka (m)")
    ax.axhline(0.0, color="0.7", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.suptitle("sl3d solve (1D) on ICE-7G")
    fig.tight_layout()
    save(fig, "sea_level_1d_ice7g", "sea_level")


def measurements(directory: Path) -> dict[str, float]:
    """A kernel run's forward values, by measurement name."""
    values = {}
    for line in (directory / "measurements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            values[parts[0]] = float(parts[2])
    return values


def kernels1d() -> None:
    directory = OUT / "kernel1d"
    if not directory.exists():
        return
    values = measurements(directory)
    name = "hudson_bay_sea_level"
    file = read_surface_file(directory / f"{name}_ice_kernels.nc")
    dates = -np.asarray(file.time) / KYR
    dual = np.asarray(file.variables["thickness_dual"].values)
    fig = plt.figure(figsize=(12, 7.5))
    picks = [0, dates.size // 2, dates.size - 2]
    for k, index in enumerate(picks):
        ex = Example.from_fields(file.grid.colatitudes, file.grid.longitudes,
                                 {"dual": dual[index]})
        ax = world_map(fig, 2, 3, k + 1)
        mesh = plot_field(ax, ex, "dual",
                          title=f"ice-thickness kernel at {dates[index]:g} "
                                "ka\n(m of sea level per m of ice, per m$^2$)")
        mark_sites(ax, [HUDSON_BAY])
        fig.colorbar(mesh, ax=ax, shrink=0.6, orientation="horizontal")
    ax = fig.add_subplot(2, 1, 2)
    for path in sorted(directory.glob("*_viscosity_kernel.txt")):
        stem = path.name.removesuffix("_viscosity_kernel.txt")
        data = np.loadtxt(path)
        radius, kernel = data[:, 0], data[:, 1]
        scale = np.max(np.abs(kernel)) or 1.0
        ax.plot(kernel / scale, (radius[-1] - radius) / 1e3,
                label=f"{stem} (J = {values[stem]:.3g})")
    ax.set_ylim(3000.0, 0.0)  # the viscous mantle lies above the core
    ax.set_xlabel("viscosity kernel K(r) per unit volume "
                  "(dJ = $\\int$ K d$\\eta$ dV), each scaled to its largest value")
    ax.set_ylabel("depth (km)")
    ax.axvline(0.0, color="0.7", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.suptitle("sl3d kernel (1D): kernels of the synthetic deglaciation's "
                 "measurements")
    fig.tight_layout()
    save(fig, "kernel1d", "kernels")


def kernels3d() -> None:
    directory = OUT / "kernel"
    if not directory.exists():
        return
    values = measurements(directory)
    fig = plt.figure(figsize=(12, 7.5))
    stems = ["hudson_bay_sea_level", "hudson_bay_decay"]
    for row, stem in enumerate(stems):
        model = read_model_file(directory / f"{stem}_viscosity_kernels.nc")
        part = next(p for p in model.parts if "K_ln_eta_0" in p.variables)
        surface_radius = max(float(np.max(p.mesh.r)) for p in model.parts)
        kernel = np.asarray(part.variables["K_ln_eta_0"].values)
        radius = np.asarray(part.mesh.r).ravel()
        # the depth at which the kernel is largest, mapped
        index = int(np.argmax(np.max(np.abs(kernel), axis=(1, 2))))
        ex = Example.from_fields(part.grid.colatitudes, part.grid.longitudes,
                                 {"kernel": kernel[index]})
        ax = world_map(fig, 2, 2, 2 * row + 1)
        mesh = plot_field(ax, ex, "kernel",
                          title=f"{stem}: log-viscosity kernel at "
                                f"{(surface_radius - radius[index]) / 1e3:.0f} km "
                                "depth")
        mark_sites(ax, [HUDSON_BAY])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        ax = fig.add_subplot(2, 2, 2 * row + 2)
        rms = np.sqrt(np.mean(kernel**2, axis=(1, 2)))
        mean = np.mean(kernel, axis=(1, 2))
        ax.plot(rms, radius / 1e3, label="lateral rms")
        ax.plot(mean, radius / 1e3, label="lateral mean")
        ax.set_xlabel("dJ / d(ln $\\eta$) per m$^3$")
        ax.set_ylabel("radius (km)")
        ax.set_title(f"{stem} (J = {values[stem]:.3g})", fontsize=10)
        ax.axvline(0.0, color="0.7", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("sl3d kernel: 3D log-viscosity kernels")
    fig.tight_layout()
    save(fig, "kernel", "viscosity_kernels")


def kernels1d_lateral() -> None:
    """The lateral kernels about the radial example Earth (kernels.lateral):
    each measurement's log-viscosity kernel as a field over the viscous
    mantle — a depth slice where it is largest, a section along the meridian
    through Hudson Bay, and its lateral mean and rms against radius, the
    mean being the radial kernel's."""
    directory = OUT / "kernel1d_lateral"
    if not directory.exists():
        return
    values = measurements(directory)
    fig = plt.figure(figsize=(16, 7.5))
    stems = ["hudson_bay_sea_level", "hudson_bay_decay"]
    for row, stem in enumerate(stems):
        model = read_model_file(directory / f"{stem}_viscosity_kernels.nc")
        # the kernels' nodes span the whole solid (the moduli's are among
        # the parameters): the viscosity's part is the mantle's, where it
        # is not zero
        part = max((p for p in model.parts if "K_ln_eta_0" in p.variables),
                   key=lambda p: float(np.max(np.abs(
                       p.variables["K_ln_eta_0"].values))))
        surface_radius = max(float(np.max(p.mesh.r)) for p in model.parts)
        kernel = np.asarray(part.variables["K_ln_eta_0"].values)
        radius = np.asarray(part.mesh.r).ravel()
        depth = (surface_radius - radius) / 1e3
        scale = float(np.max(np.abs(kernel))) or 1.0
        # the depth slice where the kernel is largest
        index = int(np.argmax(np.max(np.abs(kernel), axis=(1, 2))))
        ex = Example.from_fields(part.grid.colatitudes, part.grid.longitudes,
                                 {"kernel": kernel[index] / scale})
        ax = world_map(fig, 2, 3, 3 * row + 1)
        mesh = plot_field(ax, ex, "kernel",
                          title=f"{stem}\nat {depth[index]:.0f} km depth, "
                                "over its largest")
        mark_sites(ax, [HUDSON_BAY])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        # the section along the meridian through Hudson Bay, from the north
        # pole to 30 degrees south
        colatitudes = np.rad2deg(np.asarray(part.grid.colatitudes))
        longitudes = np.rad2deg(np.asarray(part.grid.longitudes))
        east = int(np.argmin(np.abs(longitudes - HUDSON_BAY[1])))
        span = colatitudes <= 120.0
        section = kernel[:, span, east] / scale
        ax = fig.add_subplot(2, 3, 3 * row + 2)
        limit = float(np.max(np.abs(section))) or 1.0
        mesh = ax.pcolormesh(90.0 - colatitudes[span], depth, section,
                             cmap="RdBu_r", vmin=-limit, vmax=limit,
                             shading="gouraud")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.axvline(HUDSON_BAY[0], color="k", linewidth=0.6, linestyle=":")
        ax.set_xlabel(f"latitude along {HUDSON_BAY[1]:g}$^\\circ$E "
                      "(degrees)")
        ax.set_ylabel("depth (km)")
        ax.set_title(f"{stem}: section through Hudson Bay (dotted)",
                     fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        ax = fig.add_subplot(2, 3, 3 * row + 3)
        rms = np.sqrt(np.mean(kernel**2, axis=(1, 2)))
        mean = np.mean(kernel, axis=(1, 2))
        ax.plot(rms, radius / 1e3, label="lateral rms")
        ax.plot(mean, radius / 1e3, label="lateral mean (the radial kernel)")
        ax.set_xlabel("dJ / d(ln $\\eta$) per m$^3$")
        ax.set_ylabel("radius (km)")
        ax.set_title(f"{stem} (J = {values[stem]:.3g})", fontsize=10)
        ax.axvline(0.0, color="0.7", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("sl3d kernel (1D, lateral): log-viscosity kernels as fields "
                 "about the radial Earth, by the radial sweeps")
    fig.tight_layout()
    save(fig, "kernel1d_lateral", "viscosity_kernels")


def no_ocean() -> None:
    path = OUT / "no_ocean" / "no_ocean.nc"
    if not path.exists():
        return
    ex = surface(path, ["displacement", "displacement_radial_model", "potential"])
    ex.fields["difference"] = (ex.fields["displacement"]
                               - ex.fields["displacement_radial_model"])
    fig = plt.figure(figsize=(12, 4))
    extent = (-140.0, -30.0, 25.0, 85.0)
    for k, (name, title) in enumerate((
            ("displacement", "surface displacement, 3D model (m)"),
            ("difference", "3D minus radial model (m)"))):
        ax = world_map(fig, 1, 2, k + 1, central_longitude=-85.0, extent=extent)
        mesh = plot_field(ax, ex, name, title=title)
        mark_sites(ax, [HUDSON_BAY])
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    fig.suptitle("sl3d solve, no ocean: a cap load over Hudson Bay on the "
                 "synthetic 3D elastic Earth")
    fig.tight_layout()
    save(fig, "no_ocean", "displacement")


def no_ocean_viscoelastic() -> None:
    """The cap load grown and removed on the synthetic 3D viscoelastic Earth:
    the displacement under full load, just after the load has gone and at the
    end, the lateral structure's part of the last, and the histories at the
    cap's centre of the load, the displacement and the radial model's."""
    path = OUT / "no_ocean_viscoelastic" / "no_ocean_viscoelastic.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    times = np.asarray(file.time) / KYR
    load = np.asarray(file.variables["load"].values)
    disp = np.asarray(file.variables["displacement"].values)
    radial = np.asarray(file.variables["displacement_radial_model"].values)
    i, j = nearest(grid, *HUDSON_BAY)
    full = int(np.argmax(load[:, i, j]))
    after = full + int(np.argmax(load[full:, i, j] == 0.0))
    last = len(times) - 1

    fig = plt.figure(figsize=(14, 8))
    extent = (-140.0, -30.0, 25.0, 85.0)
    panels = (
        (disp[full], f"displacement at {times[full]:g} kyr, full load (m)"),
        (disp[after], f"at {times[after]:g} kyr, the load just removed (m)"),
        (disp[last], f"at {times[last]:g} kyr (m)"),
        (disp[last] - radial[last],
         f"3D minus radial model at {times[last]:g} kyr (m)"),
    )
    for k, (values, title) in enumerate(panels):
        ex = Example.from_fields(grid.colatitudes, grid.longitudes,
                                 {"field": values})
        ax = world_map(fig, 2, 3, (1, 2, 4, 5)[k], central_longitude=-85.0,
                       extent=extent)
        mesh = plot_field(ax, ex, "field", title=title)
        mark_sites(ax, [HUDSON_BAY])
        fig.colorbar(mesh, ax=ax, shrink=0.7)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(times, load[:, i, j] / 1.0e6, color="0.3")
    ax.set_ylabel("load at the cap's centre (10$^6$ kg m$^{-2}$)")
    ax.set_xlabel("time (kyr)")
    ax.set_title("the forcing")
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(times, disp[:, i, j], label="3D model")
    ax.plot(times, radial[:, i, j], "--", label="radial model")
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_ylabel("displacement at the cap's centre (m)")
    ax.set_xlabel("time (kyr)")
    ax.set_title("the response")
    ax.legend()
    fig.suptitle("sl3d solve, no ocean: a cap load grown and removed over "
                 "Hudson Bay on the synthetic 3D viscoelastic Earth")
    fig.tight_layout()
    save(fig, "no_ocean_viscoelastic", "displacement")


def no_ocean_1d_disk() -> None:
    """A disk load held and then removed on the radial viscoelastic Earth:
    the displacement against distance from the disk's centre at a few times,
    the peripheral bulge growing under the load and the uplift after it."""
    path = OUT / "no_ocean_1d_disk" / "no_ocean_1d_disk.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    times = np.asarray(file.time) / KYR
    disp = np.asarray(file.variables["displacement"].values)
    theta = np.asarray(grid.colatitudes)[:, None]
    phi = np.asarray(grid.longitudes)[None, :]
    theta0 = np.deg2rad(90.0 - HUDSON_BAY[0])
    phi0 = np.deg2rad(HUDSON_BAY[1])
    cosine = (np.cos(theta) * np.cos(theta0)
              + np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0))
    distance = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))).ravel()
    order = np.argsort(distance)
    near = distance[order] <= 40.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, title, chosen in (
            (axes[0], "under the load", (0.0, 1.0, 5.0, 10.0)),
            (axes[1], "after its removal", (10.5, 12.0, 15.0, 20.0))):
        for t in chosen:
            n = int(np.argmin(np.abs(times - t)))
            ax.plot(distance[order][near], disp[n].ravel()[order][near],
                    label=f"{times[n]:g} kyr")
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.axvline(8.0, color="0.7", lw=0.5, ls=":")
        ax.set_xlabel("distance from the disk's centre (degrees)")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("radial displacement (m)")
    fig.suptitle("sl3d solve, no ocean: a disk load of 8 degrees held for "
                 "10 kyr on the radial viscoelastic Earth")
    fig.tight_layout()
    save(fig, "no_ocean_1d_disk", "displacement")


def exchange_table(path: Path):
    """An exchange table: its named scalars (each a list of words) and its
    rows by (l, m), each the complex coefficients (U, Phi)."""
    scalars, rows = {}, {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        words = line.split()
        if words[0].isdigit():
            u_re, u_im, phi_re, phi_im = (float(w) for w in words[2:6])
            rows[(int(words[0]), int(words[1]))] = (complex(u_re, u_im),
                                                    complex(phi_re, phi_im))
        else:
            scalars[words[0]] = words[1:]
    return scalars, rows


def held_tide(name: str):
    """A held degree-2 tide's Love numbers through time, from the exchange
    tables of a run's nodes: the times (kyr), h_2 and k_2. None when the
    run has not been made."""
    tables = sorted((OUT / name).glob(f"{name}_*.txt"))
    if not tables:
        return None
    times, h2, k2 = [], [], []
    for table in tables:
        scalars, rows = exchange_table(table)
        gravity = float(scalars["surface_gravity"][0])
        amplitude = float(scalars["tide_amplitude_m2_s2"][0])
        times.append(float(scalars["time_kyr"][0]))
        # the potential is the physical one, negative where it attracts, so
        # the conventional h_2 takes the opposite sign of g U / psi
        h2.append(-gravity * rows[(2, 0)][0].real / amplitude)
        k2.append(rows[(2, 0)][1].real / amplitude)
    return np.asarray(times), np.asarray(h2), np.asarray(k2)


def no_ocean_1d_tide() -> None:
    """A degree-2 tidal potential switched on and held: the tidal Love
    numbers of the radial viscoelastic Earth through time, from the elastic
    values at the start towards the fluid limit, read from the exchange
    tables of the nodes."""
    held = held_tide("no_ocean_1d_tide")
    if held is None:
        return
    times, h2, k2 = held

    fig, ax = plt.subplots(figsize=(7, 4.5))
    shown = np.where(times > 0.0, times, 0.5 * times[1])
    ax.semilogx(shown, h2, "o-", ms=3, label="$h_2$")
    ax.semilogx(shown, k2, "o-", ms=3, label="$k_2$")
    ax.set_xlabel("time since the potential was applied (kyr; the first "
                  "point is the elastic response)")
    ax.set_ylabel("degree-2 tidal Love number")
    ax.legend()
    ax.set_title("sl3d solve, no ocean: a held degree-2 tide on the radial "
                 "viscoelastic Earth")
    fig.tight_layout()
    save(fig, "no_ocean_1d_tide", "love_numbers")


def tide_coefficients(name: str) -> dict[tuple[int, int], complex]:
    """A config's tide coefficients by (degree, order), from its JSON."""
    config = json.loads((CONFIGS / f"{name}.json").read_text())
    coefficients = {}
    for term in config["forcing"]["tide"]:
        key = (term["degree"], term.get("order", 0))
        coefficients[key] = coefficients.get(key, 0.0) + complex(
            term["amplitude_m2_s2"], term.get("imaginary_m2_s2", 0.0))
    return coefficients


def degree_rms(coefficients) -> np.ndarray:
    """Each degree's part of a field's rms over the sphere, from its real
    orthonormal coefficients (planetmodel.harmonics' layout)."""
    return np.sqrt(np.sum(coefficients**2, axis=(0, 2)) / (4.0 * np.pi))


def peak(grid, values) -> tuple[float, float]:
    """The (latitude, longitude) in degrees where a field is largest."""
    i, j = np.unravel_index(np.argmax(values), values.shape)
    return (90.0 - np.rad2deg(grid.colatitudes[i]),
            np.rad2deg(grid.longitudes[j]))


def no_ocean_tide_3d() -> None:
    """The lunar body tide on the synthetic 3D elastic Earth: the radial
    model's displacement, the lateral structure's part of the 3D model's
    (3D minus radial) and the spectrum of that part by degree. The radial
    model answers a degree-2 potential in degree 2 alone; everything the
    3D model puts into the other degrees is the lateral coupling."""
    name = "no_ocean_tide_3d"
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    disp = np.asarray(file.variables["displacement"].values)
    radial = np.asarray(file.variables["displacement_radial_model"].values)
    difference = disp - radial
    ex = Example.from_fields(grid.colatitudes, grid.longitudes,
                             {"radial": radial, "difference": 1e3 * difference})
    moon = peak(grid, radial)  # the sub-lunar point
    structure = [(58.0, -95.0), (-15.0, -110.0)]  # the craton, the ridge

    fig = plt.figure(figsize=(12, 7))
    ax = world_map(fig, 2, 2, 1)
    mesh = plot_field(ax, ex, "radial",
                      title="radial model: radial displacement (m)")
    mark_sites(ax, [moon], colors=["gold"])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 2)
    mesh = plot_field(ax, ex, "difference",
                      title="3D minus radial model (mm)")
    mark_sites(ax, [moon] + structure, colors=["gold", "C0", "C3"])
    fig.colorbar(mesh, ax=ax, shrink=0.7)

    whole = degree_rms(analyse_grid(disp, grid)) * 1e3
    part = degree_rms(analyse_grid(difference, grid)) * 1e3
    degrees = np.arange(whole.size)
    others = degrees != 2
    outside = np.sqrt(np.sum(whole[others] ** 2) / np.sum(whole**2))
    ax = fig.add_subplot(2, 1, 2)
    ax.semilogy(degrees, np.where(whole > 0.0, whole, np.nan), "o", ms=4,
                color="0.3", label="3D model's displacement")
    ax.semilogy(degrees, np.where(part > 0.0, part, np.nan), "-",
                color="C3", label="3D minus radial model")
    ax.set_xlabel("degree")
    ax.set_ylabel("rms of the degree's part (mm)")
    ax.set_xlim(-0.5, degrees[-1] + 0.5)
    ax.text(0.75 * degrees[-1], 1e-4, "the floor beyond degree 15 is the "
            "moduli file's linear lateral\ninterpolation onto the coupling's "
            "grid, not its structure", ha="center", fontsize=8, color="0.3")
    ax.set_title("by degree: the 3D model's response outside degree 2 is "
                 f"{outside:.1e} of its rms, all of it the lateral coupling",
                 fontsize=10)
    ax.legend()
    fig.suptitle("sl3d solve, no ocean: the lunar body tide on the synthetic "
                 "3D elastic Earth (gold: the sub-lunar point; blue: the stiff "
                 "craton; red: the soft ridge)", fontsize=11)
    fig.tight_layout()
    save(fig, name, "body_tide")


def no_ocean_tide_planet() -> None:
    """The Sun's tide on the toy planet of make_planet_deck.py, elastic and
    radial: the displacement and the potential perturbation, with the
    planet's degree-2 Love numbers read off the exchange table, from each
    order's coefficients over the potential's."""
    name = "no_ocean_tide_planet"
    path = OUT / name / f"{name}.nc"
    table = OUT / name / f"{name}.txt"
    if not (path.exists() and table.exists()):
        return
    scalars, rows = exchange_table(table)
    gravity = float(scalars["surface_gravity"][0])
    radius = float(scalars["radius"][0])
    tide = tide_coefficients(name)
    h2 = [(-gravity * rows[key][0] / psi).real for key, psi in tide.items()]
    k2 = [(rows[key][1] / psi).real for key, psi in tide.items()]
    file = read_surface_file(path)
    grid = file.grid
    ex = Example.from_fields(
        grid.colatitudes, grid.longitudes,
        {"displacement": 1e3 * np.asarray(file.variables["displacement"].values),
         "potential": np.asarray(file.variables["potential"].values)})
    sun = peak(grid, ex.fields["displacement"])

    fig = plt.figure(figsize=(12, 4.2))
    for k, (field, title) in enumerate((
            ("displacement", "radial displacement (mm)"),
            ("potential", "potential perturbation (m$^2$ s$^{-2}$)"))):
        ax = map_axes(fig, 1, 2, k + 1, coastlines=False)
        mesh = plot_field(ax, ex, field, title=title)
        mark_sites(ax, [sun], colors=["gold"])
        fig.colorbar(mesh, ax=ax, shrink=0.75)
    spread = max(np.ptp(h2) / np.mean(h2), np.ptp(k2) / np.mean(k2))
    fig.suptitle(
        "sl3d solve, no ocean: the Sun's tide on the toy planet (radius "
        f"{radius / 1e3:.0f} km, g = {gravity:.2f} m s$^{{-2}}$; gold: the "
        "sub-solar point)\n"
        f"degree-2 tidal Love numbers h$_2$ = {np.mean(h2):.4f}, "
        f"k$_2$ = {np.mean(k2):.4f} (the three orders agree to "
        f"{spread:.0e})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "body_tide")


def no_ocean_tide_planet_relax() -> None:
    """A held degree-2 tide on the toy planet's Maxwell mantle: its tidal
    Love numbers through time, beside the Earth's (no_ocean_1d_tide) where
    that has run."""
    planet = held_tide("no_ocean_tide_planet_relax")
    if planet is None:
        return
    earth = held_tide("no_ocean_1d_tide")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, held, style in (("toy planet", planet, "o-"),
                               ("Earth (prem8)", earth, "--")):
        if held is None:
            continue
        times, h2, k2 = held
        shown = np.where(times > 0.0, times, 0.5 * times[1])
        ax.semilogx(shown, h2, style, ms=3, color="C0",
                    label=f"$h_2$, {label}")
        ax.semilogx(shown, k2, style, ms=3, color="C1",
                    label=f"$k_2$, {label}")
    ax.set_xlabel("time since the potential was applied (kyr; the first "
                  "point is the elastic response)")
    ax.set_ylabel("degree-2 tidal Love number")
    ax.legend(fontsize=9)
    ax.set_title("sl3d solve, no ocean: a held degree-2 tide on the toy "
                 "planet's Maxwell mantle", fontsize=11)
    fig.tight_layout()
    save(fig, "no_ocean_tide_planet_relax", "love_numbers")


def oscillation(name: str):
    """An oscillating tide's run: its times (kyr), the potential over its
    amplitude, h_2(t) and k_2(t) (the degree-2 displacement and potential
    coefficients over the amplitude, h with the conventional sign), the
    period, and the amplitude and lag (degrees) of h_2 and of k_2 over the
    last cycle. None when the run has not been made."""
    path = OUT / name / f"{name}.nc"
    tables = sorted((OUT / name).glob(f"{name}_*.txt"))
    if not (path.exists() and tables):
        return None
    tide = json.loads((CONFIGS / f"{name}.json").read_text())["forcing"]["tide"][0]
    history = np.asarray(tide["history"])
    rising = (history[:-1, 1] < 0.0) & (history[1:, 1] >= 0.0)
    period = history[1:, 0][rising][0] - history[0, 0]  # sin(2 pi t / P)
    gravity = float(exchange_table(tables[0])[0]["surface_gravity"][0])
    file = read_surface_file(path)
    times = np.asarray(file.time) / KYR
    amplitude = tide["amplitude_m2_s2"]
    u20 = analyse_grid(np.asarray(file.variables["displacement"].values),
                       file.grid)[0, 2, 0]
    phi20 = analyse_grid(np.asarray(file.variables["potential"].values),
                         file.grid)[0, 2, 0]
    h2, k2 = -gravity * u20 / amplitude, phi20 / amplitude
    # over the last cycle: a sin(wt) + b cos(wt) + c = A sin(wt - lag) + c
    last = times >= times[-1] - period - 1e-9
    w = 2.0 * np.pi / period
    basis = np.column_stack([np.sin(w * times[last]), np.cos(w * times[last]),
                             np.ones(last.sum())])
    fits = []
    for series in (h2, k2):
        (a, b, _), *_ = np.linalg.lstsq(basis, series[last], rcond=None)
        fits.append((np.hypot(a, b), np.rad2deg(np.arctan2(-b, a))))
    forcing = np.interp(times, history[:, 0], history[:, 1])
    return times, forcing, h2, k2, period, fits


def no_ocean_tide_oscillating() -> None:
    """An oscillating degree-2 tide on the radial viscoelastic Earth: the
    potential and the response through the run at one period, the response
    lagging; and the Love numbers' amplitudes and lags at the three periods
    of the oscillating configs that have run."""
    name = "no_ocean_tide_oscillating"
    run = oscillation(name)
    if run is None:
        return
    times, forcing, h2, k2, period, ((h_abs, h_lag), (k_abs, k_lag)) = run
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(times, forcing, color="0.3", label="the potential over its "
                                               "amplitude")
    ax.plot(times, h2, color="C0",
            label="$h_2(t) = -g\\,U_{20}(t)\\,/$ amplitude")
    ax.plot(times, k2, color="C1", label="$k_2(t) = \\Phi_{20}(t)\\,/$ "
                                         "amplitude")
    crest = times[-1] - 0.75 * period  # the potential's last crest
    lagged = crest + k_lag / 360.0 * period  # k_2's
    for at, color in ((crest, "0.3"), (lagged, "C1")):
        ax.axvline(at, color=color, lw=0.8, ls=":")
    ax.annotate("", xy=(lagged, 1.12), xytext=(crest, 1.12),
                arrowprops={"arrowstyle": "->", "color": "C1"})
    ax.text(lagged + 0.01 * times[-1], 1.12, f"{k_lag:.0f}$^\\circ$",
            color="C1", va="center", fontsize=9)
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_ylim(-1.55, 1.25)
    ax.set_xlabel("time (kyr)")
    ax.set_ylabel("over the potential's amplitude")
    ax.set_title(f"period {period:g} kyr; over the last cycle |h$_2$| = "
                 f"{h_abs:.3f}, lagging {h_lag:.1f}$^\\circ$, and |k$_2$| = "
                 f"{k_abs:.3f}, lagging {k_lag:.1f}$^\\circ$", fontsize=10)
    ax.legend(fontsize=9, loc="lower center", ncol=3)

    runs = [oscillation(n) for n in (f"{name}_short", name, f"{name}_long")]
    runs = [run for run in runs if run is not None]
    periods = np.array([run[4] for run in runs])
    fits = np.array([run[5] for run in runs])  # (run, h or k, abs or lag)
    held = held_tide("no_ocean_1d_tide")
    for column, ylabel in ((0, "amplitude"), (1, "lag behind the potential "
                                                 "(degrees)")):
        ax = fig.add_subplot(2, 2, 3 + column)
        for k, label in enumerate(("$h_2$", "$k_2$")):
            ax.semilogx(periods, fits[:, k, column], "o-", color=f"C{k}",
                        label=label)
            if column == 0 and held is not None:
                ends = held[1 + k][[0, -1]]
                for end, style in zip(ends, ("--", ":")):
                    ax.axhline(end, color=f"C{k}", lw=0.8, ls=style)
        ax.set_xlabel("period (kyr)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        if column == 0:
            ax.set_title("the Love numbers' amplitudes (dashed: elastic, "
                         "dotted: held 500 kyr)", fontsize=10)
        else:
            ax.set_title("their lags", fontsize=10)
    fig.suptitle("sl3d solve, no ocean: an oscillating degree-2 tide on the "
                 "radial viscoelastic Earth (prem8, a 1e21 Pa s mantle)")
    fig.tight_layout()
    save(fig, name, "oscillating")


def at_point(values, grid, lat: float, lon: float) -> np.ndarray:
    """A field's value at a place, from its expansion on the grid: values
    (..., nlat, nlon) give (...)."""
    coefficients = analyse_grid(np.asarray(values), grid)
    return synthesise(coefficients, np.deg2rad(90.0 - lat),
                      np.deg2rad(lon % 360.0))


def refined(grid, fields: dict, nlat: int = 360) -> Example:
    """Band-limited fields on the run's grid resampled onto an
    equiangular grid of nlat x 2 nlat by their expansions, for smooth maps
    of a run at a modest degree."""
    fine = equiangular(nlat, 2 * nlat)
    return Example.from_fields(fine.colatitudes, fine.longitudes, {
        name: synthesise_grid(analyse_grid(values, grid), fine)
        for name, values in fields.items()})


def cover_land(ax) -> None:
    """Grey land over a cartopy map, so that a sea-level map shows the
    ocean alone (Natural Earth's 110 m land, cartopy's cached data)."""
    if not HAVE_CARTOPY:
        return
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m"),
                   facecolor="0.85", edgecolor="none", zorder=2)


def sea_level_fingerprint_3d_ice7g() -> None:
    """The fingerprint of 10% of ICE-7G's Greenland ice melted on PREM with
    S40RTS's lateral moduli: the 3D fingerprint's sea level over the global
    ocean and around the North Atlantic, and the 3D minus radial difference
    over each, with the real coastlines."""
    name = "sea_level_fingerprint_3d_ice7g"
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    sea = np.asarray(file.variables["relative_sea_level_change"].values)
    radial = np.asarray(file.variables["relative_sea_level_change_radial_model"].values)
    ex = refined(grid, {"sea": sea, "difference": 1e3 * (sea - radial)})
    ocean = np.asarray(file.variables["ocean_function"].values) > 0.5
    far = float(np.percentile(np.abs(1e3 * (sea - radial))[ocean], 99.0))
    melt = (72.0, 320.0)
    near = (-75.0, 15.0, 45.0, 85.0)  # the North Atlantic
    fig = plt.figure(figsize=(12, 7.5))
    panels = (  # field, region, colour limit, land covered, title
        ("sea", None, 1.0, True,
         "sea level change, 3D model, over the ocean (m)"),
        ("sea", near, float(np.max(np.abs(sea[ocean]))), True,
         "near field, over the ocean (m)"),
        ("difference", None, far, True,
         "3D minus radial model, over the ocean (mm)"),
        ("difference", near, None, False,
         "3D minus radial model, near field, land included (mm)"),
    )
    for k, (field, extent, limit, covered, title) in enumerate(panels):
        ax = world_map(fig, 2, 2, k + 1, state=ICE7G,
                       central_longitude=-30.0 if extent else 0.0,
                       extent=extent)
        mesh = plot_field(ax, ex, field, title=title, limit=limit)
        if covered:
            cover_land(ax)
        mark_sites(ax, [melt])
        fig.colorbar(mesh, ax=ax, shrink=0.7,
                     extend="both" if limit is not None else "neither")
    fig.suptitle("sl3d solve, the fingerprint of 10% of ICE-7G's Greenland "
                 "ice melted, on PREM (no ocean) with S40RTS's lateral moduli "
                 f"(lmax {grid.lmax})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "fingerprint")


def site_histories(path: Path, sites):
    """A dated run's dates (kyr BP) and each site's relative sea level,
    the sea level at each date less the present's, by site name."""
    file = read_surface_file(path)
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    sea = np.asarray(file.variables["relative_sea_level_change"].values)
    out = {}
    for label, (lat, lon) in sites:
        series = at_point(sea, file.grid, lat, lon)
        out[label] = series - series[-1]
    return dates, out


def sea_level_3d_ice7g() -> None:
    """GIA through time on ICE-7G on PREM (no ocean, a 100 km lithosphere,
    two mantle viscosities) with S40RTS's lateral viscosity and moduli: the
    sea-level change since 26 ka in 3D and 3D minus the deck's radial run
    (land included: the largest differences lie under the former ice), and
    the relative sea-level histories at three sites in both, with their
    difference."""
    name, radial_name = "sea_level_3d_ice7g", "sea_level_1d_ice7g_custom"
    path = OUT / name / f"{name}.nc"
    radial_path = OUT / radial_name / f"{radial_name}.nc"
    if not (path.exists() and radial_path.exists()):
        return
    file = read_surface_file(path)
    grid = file.grid
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    sea = np.asarray(file.variables["relative_sea_level_change"].values)[-1]
    radial = read_surface_file(radial_path)
    sea_1d = np.asarray(radial.variables["relative_sea_level_change"].values)[-1]
    ex = refined(grid, {"sea": sea, "difference": sea - sea_1d})
    sites = [("Hudson Bay", (60.0, 275.0)), ("Gulf of Bothnia", (64.0, 21.0)),
             ("Barbados", (13.1, 300.4))]

    fig = plt.figure(figsize=(13, 9))
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    for k, (field, title) in enumerate((
            ("sea", f"sea level change, {dates[0]:g} ka to the present, "
                    "3D model (m)"),
            ("difference", "3D minus radial model (m)"))):
        ax = fig.add_subplot(grid_spec[0, k], projection=ccrs.Robinson()) \
            if HAVE_CARTOPY else fig.add_subplot(grid_spec[0, k])
        if HAVE_CARTOPY:
            ax.set_global()
            ax.coastlines(linewidth=0.4, color="0.3")
        mesh = plot_field(ax, ex, field, title=title)
        mark_sites(ax, [site for _, site in sites], colors=["C0", "C1", "C2"])
        fig.colorbar(mesh, ax=ax, shrink=0.75, orientation="horizontal",
                     pad=0.04)
    _, three = site_histories(path, sites)
    _, one = site_histories(radial_path, sites)
    ax = fig.add_subplot(grid_spec[1, 0])
    ax_d = fig.add_subplot(grid_spec[1, 1])
    for k, (label, _) in enumerate(sites):
        ax.plot(dates, three[label], color=f"C{k}", label=f"{label}, 3D")
        ax.plot(dates, one[label], color=f"C{k}", ls="--",
                label=f"{label}, radial")
        ax_d.plot(dates, three[label] - one[label], color=f"C{k}", label=label)
    for a in (ax, ax_d):
        a.invert_xaxis()
        a.axhline(0.0, color="0.7", lw=0.5)
        a.set_xlabel("kyr before present")
    ax.set_ylabel("relative sea level (m, present = 0)")
    ax.set_title("relative sea level, 3D (solid) and radial model (dashed)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax_d.set_ylabel("3D minus radial model (m)")
    ax_d.set_title("the lateral structure's part", fontsize=10)
    ax_d.legend(fontsize=8)
    fig.suptitle("sl3d solve on ICE-7G: PREM (no ocean) with S40RTS's "
                 "lateral viscosity and moduli, against its radial model "
                 f"(lmax {grid.lmax})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "sea_level")


def ocean_load() -> None:
    """What the ocean adds: one cap of ice laid on the synthetic Earth's
    bare land, without an ocean (no_ocean_ice_cap: the ice's load alone) and
    with one (sea_level_fingerprint_ice_cap: the fingerprint, the water drawn
    from the ocean): the displacement of each, their difference (the
    ocean's load), and the sea-level change."""
    name, alone = "sea_level_fingerprint_ice_cap", "no_ocean_ice_cap"
    path, alone_path = OUT / name / f"{name}.nc", OUT / alone / f"{alone}.nc"
    if not (path.exists() and alone_path.exists()):
        return
    file = read_surface_file(path)
    grid = file.grid
    with_ocean = np.asarray(file.variables["displacement"].values)
    load_alone = np.asarray(read_surface_file(alone_path)
                            .variables["displacement"].values)
    ocean = np.asarray(file.variables["ocean_function"].values) > 0.5
    sea = np.asarray(file.variables["relative_sea_level_change"].values)
    ex = Example.from_fields(grid.colatitudes, grid.longitudes, {
        "alone": load_alone,
        "difference": 1e2 * (with_ocean - load_alone),
        "sea": np.where(ocean, sea, np.nan)})
    cap = (50.0, 260.0)
    lats = 90.0 - np.rad2deg(np.asarray(grid.colatitudes))[:, None]
    lons = np.rad2deg(np.asarray(grid.longitudes))[None, :]
    distance = np.broadcast_to(distance_deg(lats, lons, *cap), sea.shape)

    fig = plt.figure(figsize=(12, 7.5))
    ax = world_map(fig, 2, 2, 1, central_longitude=-100.0,
                   extent=(-150.0, -50.0, 15.0, 85.0))
    mesh = plot_field(ax, ex, "alone", title="displacement under the cap "
                      "(no ocean, the load alone; m)")
    mark_sites(ax, [cap])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 2, central_longitude=-100.0)
    mesh = plot_field(ax, ex, "difference", title="displacement, with the "
                      "ocean minus without: the ocean's load (cm)")
    mark_sites(ax, [cap])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 3, central_longitude=-100.0)
    mesh = plot_field(ax, ex, "sea", cmap="viridis", symmetric=False,
                      title="sea level change over the ocean (m)")
    mark_sites(ax, [cap])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 2, 4)
    far = distance > 15.0
    order = np.argsort(distance[far])
    ax.plot(distance[far][order], 1e2 * load_alone[far][order], color="0.2",
            lw=1.5, label="the load alone (no ocean)")
    for where, color, label in ((ocean, "C0", "with the ocean, under it"),
                                (~ocean, "C5", "with the ocean, on land")):
        pick = far & where
        ax.plot(distance[pick], 1e2 * with_ocean[pick], ".", ms=1.0,
                color=color, alpha=0.4, label=label)
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_xlim(15.0, 180.0)
    ax.set_xlabel("distance from the cap's centre (degrees)")
    ax.set_ylabel("displacement (cm)")
    ax.set_title("away from the cap: the load alone, and with the ocean",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=8)
    fig.suptitle("sl3d solve: 1 km of ice (a 3 degree cap) laid on the "
                 "synthetic Earth, elastic, without rotation:\nwithout an "
                 "ocean (the ice's load alone) against with one (the "
                 "fingerprint, the water drawn from the ocean)", fontsize=11)
    fig.tight_layout()
    save(fig, name, "ocean")



def no_ocean_cap_by_location() -> None:
    """The same cap loaded and removed at two places of the synthetic 3D
    viscoelastic Earth, over weak and over stiff mantle: the viscosity
    file's lateral pattern with the two caps, and the displacement at each
    cap's centre through time, 3D and radial model."""
    runs = {"weak": "no_ocean_cap_weak", "strong": "no_ocean_cap_strong"}
    paths = {k: OUT / n / f"{n}.nc" for k, n in runs.items()}
    if not all(p.exists() for p in paths.values()):
        return
    centres = {}
    for key, name in runs.items():
        config = json.loads((CONFIGS / f"{name}.json").read_text())
        cap = config["forcing"]["load"][0]["cap"]
        centres[key] = (cap["latitude"], cap["longitude"])
    model = read_model_file(DATA / "viscosity_log.nc").parts[0]
    radius = np.asarray(model.mesh.r).ravel()
    log_eta = np.asarray(model.variables["viscosity_perturbation"].values)
    depth = 400.0e3
    k = int(np.argmin(np.abs(radius.max() - radius - depth)
                      + 1e9 * ~np.isfinite(log_eta[:, 0, 0])))
    ex = Example.from_fields(model.grid.colatitudes, model.grid.longitudes,
                             {"log_eta": log_eta[k]})

    fig = plt.figure(figsize=(13, 4.5))
    ax = map_axes(fig, 1, 3, 1, coastlines=False)
    mesh = plot_field(ax, ex, "log_eta",
                      title=f"ln($\\eta/\\eta_{{ref}}$) at "
                            f"{(radius.max() - radius[k]) / 1e3:.0f} km depth")
    mark_sites(ax, [centres["weak"], centres["strong"]], colors=["C0", "C3"])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    colors = {"weak": "C0", "strong": "C3"}
    ax_d = fig.add_subplot(1, 3, 2)
    ax_r = fig.add_subplot(1, 3, 3)
    for key, path in paths.items():
        file = read_surface_file(path)
        times = np.asarray(file.time) / KYR
        lat, lon = centres[key]
        disp = at_point(file.variables["displacement"].values, file.grid,
                        lat, lon)
        radial = at_point(file.variables["displacement_radial_model"].values,
                          file.grid, lat, lon)
        local = at_point(log_eta[k], model.grid, lat, lon)
        ax_d.plot(times, disp, color=colors[key],
                  label=f"3D, over {key} mantle ({lat:g}, {lon:g})")
        if key == "strong":  # the radial model's is the same at both
            ax_d.plot(times, radial, color="k", ls="--", lw=0.9,
                      label="radial model, at either")
        ax_r.plot(times, disp - radial, color=colors[key],
                  label=f"{key}: ln($\\eta/\\eta_{{ref}}$) = {local:+.2f} at "
                        f"{(radius.max() - radius[k]) / 1e3:.0f} km")
    ax_d.axhline(0.0, color="0.7", lw=0.5)
    ax_d.set_xlabel("time (kyr)")
    ax_d.set_ylabel("displacement at the cap's centre (m)")
    ax_d.set_title("loaded at 0, removed over 6 to 6.5 kyr", fontsize=10)
    ax_d.legend(fontsize=7)
    ax_r.axhline(0.0, color="0.7", lw=0.5)
    ax_r.set_xlabel("time (kyr)")
    ax_r.set_ylabel("3D minus radial model (m)")
    ax_r.set_title("the lateral structure's part", fontsize=10)
    ax_r.legend(fontsize=8)
    fig.suptitle("sl3d solve, no ocean: the same cap load over weak and over "
                 "stiff mantle on the synthetic 3D viscoelastic Earth "
                 "(viscosity_log.nc)", fontsize=11)
    fig.tight_layout()
    save(fig, "no_ocean_cap_weak", "relaxation")


def no_ocean_load() -> None:
    """One load history given three ways: an analytic cap with a history
    (no_ocean_load_cap), the same cap sampled into a surface file on a time
    axis read with a time origin (no_ocean_load_field), and its expansion
    as a coefficient table with the cap's history (no_ocean_load_coefficients):
    the forcing on the run's clock and on the file's axis, the displacement
    at the cap's centre through time, and how far each run departs from the
    analytic one anywhere on the grid."""
    names = ["no_ocean_load_cap", "no_ocean_load_field",
             "no_ocean_load_coefficients"]
    paths = {n: OUT / n / f"{n}.nc" for n in names}
    if not all(p.exists() for p in paths.values()):
        return
    term = json.loads((CONFIGS / "no_ocean_load_cap.json").read_text())[
        "forcing"]["load"][0]
    centre = (term["cap"]["latitude"], term["cap"]["longitude"])
    history = np.asarray(term["history"])
    field = json.loads((CONFIGS / "no_ocean_load_field.json").read_text())[
        "forcing"]["load"][0]["field"]
    origin = field.get("time_origin", 0.0)
    stored = read_surface_file(CONFIGS / field["file"])
    # the file's own axis, in its own time scale
    stored_times = np.asarray(stored.time) * stored.scales.time / KYR
    stored_load = at_point(
        stored.variables[field.get("variable", "load")].values, stored.grid,
        *centre)
    files = {n: read_surface_file(p) for n, p in paths.items()}
    times = np.asarray(files[names[0]].time) / KYR
    reference = np.asarray(files[names[0]].variables["displacement"].values)
    styles = {"no_ocean_load_cap": ("C0", "-", "cap, history"),
              "no_ocean_load_field": ("C1", "--",
                                      f"field file, time_origin {origin:g}"),
              "no_ocean_load_coefficients": ("C2", ":",
                                             "coefficient table, history")}

    fig, (ax_f, ax, ax_e) = plt.subplots(1, 3, figsize=(15, 4.4))
    factor = np.interp(times, history[:, 0], history[:, 1])
    ax_f.plot(times, factor, color="C0", lw=2.0, label="history of the "
              "cap and the table (held beyond its ends)")
    ax_f.plot(stored_times + origin, stored_load / stored_load.max(), "o",
              color="C1", label="the file's stored times, read at "
                                f"t - {origin:g} kyr")
    ax_f.set_xlim(times[0], times[-1])
    ax_f.set_ylim(-0.1, 1.35)
    ax_f.set_xlabel("time on the run's clock, t (kyr)")
    ax_f.set_ylabel("load over its largest, at the cap's centre")
    ax_f.set_title("the forcing", fontsize=10)
    ax_f.legend(fontsize=8, loc="upper right")
    top = ax_f.secondary_xaxis("top", functions=(lambda t: t - origin,
                                                 lambda u: u + origin))
    top.set_xlabel("time on the file's axis (kyr)", fontsize=8, color="C1")
    for n, file in files.items():
        color, ls, label = styles[n]
        disp = at_point(file.variables["displacement"].values, file.grid,
                        *centre)
        ax.plot(times, disp, color=color, ls=ls, lw=2.0 if ls == "-" else 1.5,
                label=label)
        if n != names[0]:
            values = np.asarray(file.variables["displacement"].values)
            departure = (np.max(np.abs(values - reference), axis=(1, 2))
                         / np.max(np.abs(reference)))
            ax_e.semilogy(times, np.where(departure > 0, departure, np.nan),
                          color=color, label=label)
    ax.set_xlabel("time on the run's clock (kyr)")
    ax.set_ylabel("displacement at the cap's centre (m)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("the response, the same three ways", fontsize=10)
    ax_e.set_xlabel("time on the run's clock (kyr)")
    ax_e.set_ylabel("largest |difference| from the cap run, over its largest")
    ax_e.set_title("agreement, everywhere on the grid", fontsize=10)
    ax_e.legend(fontsize=8)
    fig.suptitle("sl3d solve, no ocean: a load history from a file and from a "
                 "coefficient table, on the radial viscoelastic Earth "
                 "(prem8, 1e21 Pa s)", fontsize=11)
    fig.tight_layout()
    save(fig, "no_ocean_load_field", "agreement")


def timestep_study() -> None:
    """The time-stepping study of timestep_study.py: the error against the
    step of the sea level at the two measurement dates by both schemes, and
    of the viscosity kernels by both adjoints, each against its converged
    reference, with first- and second-order guides and the step in
    Maxwell times; and the kernels themselves at a 1 kyr step."""
    import timestep_study as study

    result = study.errors()
    if result is None:
        return
    table, (ref_fields, ref_kernels) = result
    tau_min, tau_max = study.maxwell_times_kyr()
    fig = plt.figure(figsize=(14, 4.8))
    styles = {"et": ("C0", "exponential trapezoid"),
              "be": ("C3", "backward Euler"),
              "discrete": ("C0", "discrete adjoint"),
              "continuous": ("C1", "continuous adjoint")}
    marks = {f"hudson_bay_{study.MID:g}ka": ("o", "-",
                                            f"{study.MID:g} ka (mid-run)"),
             f"hudson_bay_{study.END:g}ka": ("s", "--",
                                            f"{study.END:g} ka (the end)")}
    panels = (("sea_level", f"(a) the sea level, against "
                            f"{study.step_kyr(ref_fields):g} kyr steps",
               "relative L2 error over the sphere"),
              ("kernel", f"(b) the viscosity kernel, against the discrete "
                         f"one at {study.step_kyr(ref_kernels):g} kyr",
               "relative error, (∫ δK$^2$ r$^2$ dr)$^{1/2}$"))
    for k, (quantity, title, ylabel) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1)
        low = np.inf
        for label, (steps, errs) in table[quantity].items():
            kind, measurement = label.split()
            color, name = styles[kind]
            marker, ls, when = marks[measurement]
            dt = np.array([study.step_kyr(n) for n in steps])
            ax.loglog(dt, errs, color=color, ls=ls, marker=marker, ms=4,
                      label=f"{name}, {when}")
            low = min(low, np.min(errs))
        dt = np.array([study.step_kyr(n) for n in study.STEPS])
        for order, anchor in ((1, 0.6), (2, 0.6)):
            guide = anchor * low * (dt / dt[-1]) ** order * 4
            ax.loglog(dt, guide, color="0.6", lw=0.8, ls=":")
            ax.text(dt[0], guide[0], f" order {order}", fontsize=8,
                    color="0.4", va="center")
        ax.set_xlabel("step (kyr)")
        ax.set_xticks(dt, [f"{d:g}" if d >= 1 else f"1/{1 / d:g}" for d in dt])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="major", color="0.9", lw=0.5)
        top = ax.secondary_xaxis(
            "top", functions=(lambda x: x / tau_min, lambda x: x * tau_min))
        top.set_xticks(dt / tau_min, [f"{d / tau_min:.2g}" for d in dt])
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.set_xlabel(f"step over the shortest Maxwell time "
                       f"({tau_min:.2f} kyr)", fontsize=8)
    # the kernels at a 1 kyr step against the reference
    ax = fig.add_subplot(1, 3, 3)
    measurement = f"hudson_bay_{study.END:g}ka"
    found = study.runs_present()
    one_kyr = int(round((study.START - study.END) / 1.0))
    r0, k0 = study.kernel(found["discrete"][ref_kernels], measurement)
    depth = (r0.max() - r0) / 1e3
    scale = np.max(np.abs(k0))
    ax.plot(k0 / scale, depth, color="k", lw=2.0,
            label=f"reference ({study.step_kyr(ref_kernels):g} kyr)")
    for adjoint in study.ADJOINTS:
        if one_kyr in found.get(adjoint, {}):
            _, kk = study.kernel(found[adjoint][one_kyr], measurement)
            ax.plot(kk / scale, depth, color=styles[adjoint][0],
                    label=f"{styles[adjoint][1]}, 1 kyr steps")
    ax.set_ylim(2900.0, 0.0)
    ax.axvline(0.0, color="0.7", lw=0.5)
    ax.set_xlabel("viscosity kernel K(r), over the reference's largest")
    ax.set_ylabel("depth (km)")
    ax.set_title(f"(c) the kernel of the sea level at {study.END:g} ka",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("sl3d: time-step accuracy on the radial viscoelastic Earth "
                 "(prem8, 1e21 Pa s; Maxwell times "
                 f"{tau_min:.2f} to {tau_max:.2f} kyr), the synthetic "
                 "deglaciation 16 to 6 ka, Hudson Bay", fontsize=11)
    fig.tight_layout()
    save(fig, "timestep_study", "accuracy")


# -- seismic sources, and the problem without gravity ------------------------

# A unit of each run clock's times, in seconds (the surface files' meta
# `time_unit`).
TIME_UNITS = {"day": 86400.0, "yr": KYR / 1.0e3, "kyr": KYR}


def seismic_source(name: str):
    """A config's first seismic term: its kind, "fault" or "point", and its
    members (latitude, longitude, depth_km, strike_deg, dip_deg, ...)."""
    term = json.loads((CONFIGS / f"{name}.json").read_text())["forcing"][
        "seismic"][0]
    kind = "fault" if "fault" in term else "point"
    return kind, term[kind]


def bearing(lat1, lon1, lat2, lon2):
    """The azimuth (degrees clockwise from north) of the great circle from
    the first points to the second, all in degrees."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    return np.rad2deg(np.arctan2(
        np.sin(dl) * np.cos(phi2),
        np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dl)))


def mark_source(ax, kind: str, source: dict) -> None:
    """A seismic source on a map: a fault's surface projection, its up-dip
    edge heavy, or a point source's epicentre, a star."""
    if kind == "fault":
        mark_fault(ax, source["latitude"], source["longitude"],
                   source["strike_deg"], source["dip_deg"],
                   source["length_km"], source["width_km"])
    else:
        kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        ax.plot(source["longitude"], source["latitude"], "*", ms=14,
                mfc="gold", mec="k", mew=0.8, zorder=6, **kwargs)


def source_extent(source: dict, half_width: float = 11.0) -> tuple:
    """A map region (west, east, south, north) centred on a source, about
    `half_width` degrees of arc to each side."""
    lat, lon = source["latitude"], source["longitude"]
    across = half_width / np.cos(np.deg2rad(lat))
    return (lon - across, lon + across, lat - half_width, lat + half_width)


def source_sites(source: dict):
    """Three places about a source, named: over its centroid, 250 km down
    dip (beyond a shallow thrust's deep end, towards its coast) and 1000 km
    up dip (the far field)."""
    lat, lon = source["latitude"], source["longitude"]
    dip = source.get("strike_deg", 0.0) + 90.0
    return [("over the source", (lat, lon)),
            ("250 km down dip", destination(lat, lon, dip, 250.0)),
            ("1000 km up dip", destination(lat, lon, dip, -1000.0))]


def in_region(ex: Example, name: str, extent) -> np.ndarray:
    """A field's values inside a map region."""
    west, east, south, north = extent
    rows = (ex.lats >= south) & (ex.lats <= north)
    cols = ((ex.lons - west) % 360.0) <= (east - west)
    return ex.fields[name][np.ix_(rows, cols)]


def region_limit(ex: Example, names, extent) -> float:
    """The largest magnitude of some fields inside a map region: one
    colour scale across their panels."""
    return max(float(np.nanmax(np.abs(in_region(ex, n, extent))))
               for n in names)


def tangential(V, W, grid, theta, phi):
    """The tangential surface displacement, east and north, at points
    (colatitudes and longitudes in radians, broadcast together), from a
    run's tangential and toroidal potentials (m) on its grid, u_t = grad_1
    V - r x grad_1 W (W None: none), by their expansions' gradients. Fields
    with a node axis first give that axis first."""
    v_theta, v_phi = synthesise_gradient(analyse_grid(np.asarray(V), grid),
                                         theta, phi)
    s = np.sin(theta)
    u_theta, u_phi = v_theta, v_phi / s
    if W is not None:
        w_theta, w_phi = synthesise_gradient(
            analyse_grid(np.asarray(W), grid), theta, phi)
        u_theta = u_theta + w_phi / s
        u_phi = u_phi - w_theta
    return u_phi, -u_theta


def ocean_of(file):
    """A run's ocean on its grid: its ocean function (the state's under
    frozen shorelines, a migrating run's at its first node); None without
    an ocean."""
    if "ocean_function" not in file.variables:
        return None
    values = np.asarray(file.variables["ocean_function"].values)
    return (values if values.ndim == 2 else values[0]) > 0.5


def stored(file, name: str, smoothed: bool = True) -> np.ndarray:
    """A field of a seismic run's file as stored: the smoothed one sl3d
    writes beside it (`<name>_smoothed`, the config's `output.smoothing`)
    or, with `smoothed` false, the field itself."""
    if not smoothed:
        return np.asarray(file.variables[name].values)
    key = f"{name}_smoothed"
    if key not in file.variables:
        raise KeyError(f"{key}: the seismic figures draw the smoothed fields, "
                       "which the config's `output.smoothing` asks for")
    return np.asarray(file.variables[key].values)


def smoothing_width(file) -> float:
    """The width (degrees) of the Gaussian cap a run's fields are smoothed
    over, its file's `smoothing_width_deg`."""
    return float(np.ravel(file.meta["smoothing_width_deg"])[0])


def means(width: float) -> str:
    """The figures' words for the smoothed fields."""
    return (f"{width:g}° Gaussian means (the fields averaged over a cap of "
            f"{width:g}° about each point: sl3d's output.smoothing)")


def check_components(file) -> float:
    """The largest difference, over the largest size, of a seismic run's
    east and north components as its file holds them (`displacement_east`,
    `displacement_north`, raw and smoothed, at its first and last nodes)
    from their synthesis here from the file's potentials; printed, and an
    error past 1e-10."""
    grid = file.grid
    theta, phi = np.meshgrid(np.asarray(grid.colatitudes),
                             np.asarray(grid.longitudes), indexing="ij")
    worst = 0.0
    for smoothed in (False, True):
        V = stored(file, "tangential_potential", smoothed)
        W = (stored(file, "toroidal_potential", smoothed)
             if "toroidal_potential" in file.variables else None)
        east = stored(file, "displacement_east", smoothed)
        north = stored(file, "displacement_north", smoothed)
        picks = [0, -1] if V.ndim == 3 else [None]
        for n in picks:
            at = (lambda x: x) if n is None else (lambda x: x[n])
            e, nn = tangential(at(V), None if W is None else at(W), grid,
                               theta, phi)
            size = max(np.max(np.abs(at(east))), np.max(np.abs(at(north))))
            worst = max(worst, np.max(np.abs(e - at(east))) / size,
                        np.max(np.abs(nn - at(north))) / size)
    print(f"  the file's east and north components against their synthesis "
          f"from its potentials: {worst:.1e} of their size")
    if worst > 1e-10:
        raise RuntimeError("the file's east and north components are not its "
                           "potentials' displacement")
    return worst


def grid_motion(file, index=None, since=None) -> Example:
    """A seismic run's smoothed horizontal displacement on its own grid, as
    its file holds it (`displacement_east_smoothed`,
    `displacement_north_smoothed`): `east` and `north` at a node (index
    None: a static file), less those at node `since` where given, m."""
    grid = file.grid

    def node(name):
        values = stored(file, name)
        if index is None:
            return values
        return values[index] - (values[since] if since is not None else 0.0)

    return Example.from_fields(grid.colatitudes, grid.longitudes, {
        "east": node("displacement_east"), "north": node("displacement_north")})


def motion(file, extent, index=None, since=None,
           spacing: float = 0.1) -> Example:
    """A seismic run's smoothed surface motion over a map region (west,
    east, south, north) at a node (index None: a static file), less that
    at node `since` where given, on a lattice `spacing` degrees apart by
    the expansions of the file's smoothed fields (stored): the vertical
    displacement `up`, the horizontal `east`, `north` (the displacement of
    the smoothed potentials) and its size `horizontal`, and, with an ocean,
    the relative sea-level change `sea` over the ocean (NaN on land), all
    in m."""
    grid = file.grid

    def node(name):
        values = stored(file, name)
        if index is None:
            return values
        return values[index] - (values[since] if since is not None else 0.0)

    # the region with a margin, which a projected map's corners reach
    west, east, south, north = extent
    margin = 4.0 / np.cos(np.deg2rad(0.5 * (south + north)))
    lats = np.arange(min(north + 4.0, 89.0), max(south - 4.0, -89.0),
                     -spacing)
    lons = np.arange(west - margin, east + margin, spacing) % 360.0
    colatitudes, longitudes = np.deg2rad(90.0 - lats), np.deg2rad(lons)
    theta, phi = np.meshgrid(colatitudes, longitudes, indexing="ij")

    def at(values):
        return synthesise(analyse_grid(values, grid), theta, phi)

    toroidal = (node("toroidal_potential")
                if "toroidal_potential" in file.variables else None)
    east_, north_ = tangential(node("tangential_potential"), toroidal, grid,
                               theta, phi)
    fields = {"up": at(node("displacement")), "east": east_,
              "north": north_, "horizontal": np.hypot(east_, north_)}
    ocean = ocean_of(file)
    if ocean is not None and "relative_sea_level_change" in file.variables:
        wet = at(ocean.astype(float)) > 0.5
        fields["sea"] = np.where(wet, at(node("relative_sea_level_change")),
                                 np.nan)
    return Example.from_fields(colatitudes, longitudes, fields)


def regional_map(fig, rows, cols, index, source, extent):
    """A map axes over a region about a source, with the synthetic Earth's
    coastline and the meridians and parallels every 5 degrees."""
    ax = world_map(fig, rows, cols, index,
                   central_longitude=source["longitude"], extent=extent)
    if HAVE_CARTOPY:
        ax.gridlines(xlocs=np.arange(-180.0, 181.0, 5.0),
                     ylocs=np.arange(-90.0, 91.0, 5.0), linewidth=0.3,
                     color="0.5", alpha=0.6)
    return ax


def profile(file, source: dict, index=None, reach_km: float = 900.0,
            count: int = 241, smoothed: bool = True):
    """The motion along the great circle through a source across its
    strike, from `reach_km` up dip to `reach_km` down dip, at a node (index
    None: a static file), by the expansions of the file's smoothed fields
    or, with `smoothed` false, of its raw ones, whose point values these
    are (stored): the distances (km, positive down dip), the vertical
    displacement, the horizontal displacement up dip (towards a thrust's
    trench) and along strike, and the relative sea-level change where the
    ocean is (NaN elsewhere), m."""
    grid = file.grid

    def node(name):
        values = stored(file, name, smoothed)
        return values if index is None else values[index]

    dip = source.get("strike_deg", 0.0) + 90.0
    distance = np.linspace(-reach_km, reach_km, count)
    lat, lon = np.array([destination(source["latitude"], source["longitude"],
                                     dip, d) for d in distance]).T
    theta, phi = np.deg2rad(90.0 - lat), np.deg2rad(lon)
    up = synthesise(analyse_grid(node("displacement"), grid), theta, phi)
    toroidal = (node("toroidal_potential")
                if "toroidal_potential" in file.variables else None)
    east, north = tangential(node("tangential_potential"), toroidal, grid,
                             theta, phi)
    # the profile's own direction at each point, down dip
    ahead = np.r_[1:count, count - 1]
    behind = np.r_[0, 0:count - 1]
    azimuth = np.deg2rad(bearing(lat[behind], lon[behind], lat[ahead],
                                 lon[ahead]))
    down_dip = east * np.sin(azimuth) + north * np.cos(azimuth)
    along_strike = east * np.cos(azimuth) - north * np.sin(azimuth)
    sea = np.full(count, np.nan)
    ocean = ocean_of(file)
    if ocean is not None and "relative_sea_level_change" in file.variables:
        values = synthesise(analyse_grid(node("relative_sea_level_change"),
                                         grid), theta, phi)
        wet = np.array([ocean[nearest(grid, a, b)] for a, b in zip(lat, lon)])
        sea = np.where(wet, values, np.nan)
    return distance, up, -down_dip, along_strike, sea


def node_times(file):
    """A run's node times in its own clock's unit, and the unit."""
    unit = str(file.meta.get("time_unit", "kyr"))
    return np.asarray(file.time) / TIME_UNITS[unit], unit


# The seismic figures draw the fields the runs write smoothed over a
# Gaussian cap (the configs' `output.smoothing`, 2°). The fields of a fault
# a few kilometres deep, laid out as point sources 50 km apart, are not
# converged in lmax at any degree these runs reach: the truncation rings
# about the source, and finer degrees resolve the point sources one by
# one. Their means over 2° caps are converged by lmax 96 (to a few tenths
# of a per cent of lmax 128; configs/README.md).


class Drawn:
    """How a co-seismic figure draws its source: the place its maps are
    centred on and its profile crosses (`source`: latitude, longitude and
    the strike there, degrees), what marks it on a map (`mark`), the
    profile's stretch over the fault (km, up dip negative), and the
    words for both."""

    def __init__(self, source: dict, mark, span: tuple[float, float],
                 drawn: str, through: str) -> None:
        self.source, self.mark, self.span = source, mark, span
        self.drawn, self.through = drawn, through


def planar_drawn(name: str) -> Drawn:
    """A config's planar fault or point source as its co-seismic figure
    draws it: the fault's surface projection, its up-dip edge heavy."""
    kind, source = seismic_source(name)
    half = (0.5 * source["width_km"] * np.cos(np.deg2rad(source["dip_deg"]))
            if kind == "fault" else 0.0)
    return Drawn(source, lambda ax: mark_source(ax, kind, source),
                 (-half, half),
                 "the fault's surface projection drawn, its up-dip edge heavy",
                 "its centroid")


def fault_file_drawn(name: str) -> Drawn | None:
    """A config's fault file (its first seismic term's `fault_file`) as its
    co-seismic figure draws it, read with planetmodel's reader: every
    patch's outline on the surface, its trench (the up-dip edge, v = -1/2)
    heavy, the maps centred on the first patch's middle and the profile
    crossing it along the strike there. None without planetmodel's fault
    module (planetmodel after 1.5.1) or the file."""
    try:
        from planetmodel.faultfile import read_fault
    except ImportError:
        return None
    term = json.loads((CONFIGS / f"{name}.json").read_text())["forcing"][
        "seismic"][0]["fault_file"]
    path = CONFIGS / term["file"]
    if not path.exists():
        return None
    fault = read_fault(path)
    edge = np.linspace(-0.5, 0.5, 101)
    low, high = np.full_like(edge, -0.5), np.full_like(edge, 0.5)
    outlines = []
    for patch in fault.patches:
        u = np.concatenate([edge, high, edge[::-1], low])
        v = np.concatenate([low, edge, high, edge[::-1]])
        outlines.append((patch.position(u, v), patch.position(edge, low)))
    first = fault.patches[0]
    middle = first.position(0.0, 0.0)
    basis = fault.frame.basis(middle)
    strike = first.fault_basis(0.0, 0.0)[:, 0]
    source = {"latitude": float(middle[0]), "longitude": float(middle[1]),
              "strike_deg": float(np.rad2deg(np.arctan2(strike @ basis[:, 0],
                                                        strike @ basis[:, 1])))}

    def across(position) -> float:
        """The signed distance (km) of a point's surface place from the
        middle's, down dip positive."""
        a, b = np.deg2rad(middle[:2]), np.deg2rad(position[:2])
        d = 2.0 * np.arcsin(np.sqrt(
            np.sin(0.5 * (b[0] - a[0])) ** 2
            + np.cos(a[0]) * np.cos(b[0]) * np.sin(0.5 * (b[1] - a[1])) ** 2))
        return 6371.0 * float(d)

    span = (-across(first.position(0.0, -0.5)), across(first.position(0.0, 0.5)))

    def mark(ax) -> None:
        kwargs = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        for outline, trench in outlines:
            ax.plot(outline[:, 1], outline[:, 0], color="k", lw=0.9, zorder=6,
                    **kwargs)
            ax.plot(trench[:, 1], trench[:, 0], color="k", lw=2.6, zorder=6,
                    **kwargs)

    return Drawn(source, mark, span,
                 "the fault's outline drawn from its file, its trench heavy",
                 "its middle")


def coseismic(name: str, title: str, drawn: Drawn | None = None) -> None:
    """A fault's co-seismic response, the static problem: maps centred on
    the source of the vertical displacement, of the horizontal
    displacement (its size, and arrows) and, with an ocean, of the relative
    sea-level change over the ocean, each a 2° Gaussian mean, the fault
    drawn (`drawn`: a planar fault's surface projection, its up-dip edge
    heavy, by default); and every one along a profile across the fault
    through its centroid (a fault file's middle), the 2° means and the
    vertical displacement's point values, whose ripples are the
    truncation."""
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    drawn = drawn or planar_drawn(name)
    source = drawn.source
    file = read_surface_file(path)
    check_components(file)
    width = smoothing_width(file)
    extent = source_extent(source)
    ex = motion(file, extent)
    lmax = file.grid.lmax

    fig = plt.figure(figsize=(13, 9.5))
    panels = [("up", "RdBu_r", True, "vertical displacement (m)"),
              ("horizontal", "YlOrRd", False,
               "horizontal displacement, its size (m) and direction")]
    if "sea" in ex.fields:
        panels.append(("sea", "RdBu_r", True,
                       "relative sea-level change over the ocean (m)"))
    for k, (field, cmap, symmetric, label) in enumerate(panels):
        ax = regional_map(fig, 2, 2, k + 1, source, extent)
        limit = region_limit(ex, [field], extent) if symmetric else None
        mesh = plot_field(ax, ex, field, cmap=cmap, symmetric=symmetric,
                          title=label, limit=limit)
        if not symmetric:
            mesh.set_clim(0.0, region_limit(ex, [field], extent))
            arrows(ax, grid_motion(file), extent)
        drawn.mark(ax)
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="m")

    distance, up, updip, strike, sea = profile(file, source)
    _, up_point, _, _, _ = profile(file, source, smoothed=False)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(distance, up_point, color="C3", lw=0.7, alpha=0.6,
            label="vertical, point values (not converged in lmax)")
    ax.plot(distance, up, color="C3", lw=2.0, label="vertical")
    ax.plot(distance, updip, color="C0", lw=2.0, label="horizontal, up dip")
    ax.plot(distance, strike, color="C0", ls=":",
            label="horizontal, along strike")
    if np.any(np.isfinite(sea)):
        ax.plot(distance, sea, color="C2", lw=2.0, alpha=0.7,
                label="relative sea level, over the ocean")
    ax.axvspan(*drawn.span, color="0.85", zorder=0,
               label="the fault's surface projection")
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel(f"distance across the fault through {drawn.through} (km, "
                  f"positive down dip, to azimuth "
                  f"{(source['strike_deg'] + 90.0) % 360.0:.0f}°)")
    ax.set_ylabel("m")
    ax.set_title(f"along a profile across the fault: {width:g}° means "
                 f"(thick), and the\npoint values, whose ripples, 360/{lmax} "
                 f"degrees apart, are the truncation", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle(f"{title}, lmax {lmax}\nmaps centred on the fault, "
                 f"meridians and parallels 5° apart, {drawn.drawn};\nmaps "
                 f"{means(width)}",
                 fontsize=10)
    fig.tight_layout()
    save(fig, name, "coseismic")


def deck_of(name: str) -> Path:
    """A config's deck, its path as the config resolves it."""
    model = json.loads((CONFIGS / f"{name}.json").read_text())["model"]
    return CONFIGS / model["deck"]


def relaxation_bands(name: str):
    """The Maxwell times of a config's deck, its viscous layers of one
    viscosity together (inputs.maxwell_times), as mark_relaxation_times's
    bands keyed M1, M2, ... from the surface down."""
    return [(f"M{k + 1}", f"{top:.0f}–{bottom:.0f} km, "
             f"{eta:.0e} Pa s".replace("e+", "e") + ", Maxwell times "
             f"{years_text(shortest)}–{years_text(longest)} yr",
             shortest, longest, "band")
            for k, (top, bottom, eta, shortest, longest)
            in enumerate(maxwell_times(deck_of(name)))]


def years_text(t: float) -> str:
    """A time in years to two figures, without an exponent."""
    return f"{t:.2g}" if t < 100.0 else f"{t:.0f}"


def site_series(file, source: dict, sites):
    """A seismic run's series at places, each the smoothed field there (a
    mean over the file's Gaussian cap: stored), by its expansion: per place
    the vertical displacement and, with an ocean, the relative sea-level
    change, or without one the horizontal displacement up dip, the
    displacement of the smoothed potentials there; m."""
    grid = file.grid
    wet = ("relative_sea_level_change" in file.variables
           and ocean_of(file) is not None)
    W = (stored(file, "toroidal_potential")
         if "toroidal_potential" in file.variables else None)
    cU = analyse_grid(stored(file, "displacement"), grid)
    cS = (analyse_grid(stored(file, "relative_sea_level_change"), grid)
          if wet else None)
    cV = analyse_grid(stored(file, "tangential_potential"), grid)
    cW = analyse_grid(W, grid) if W is not None else None
    dip = np.deg2rad(source.get("strike_deg", 0.0) + 90.0)
    out = []
    for _, (lat, lon) in sites:
        theta, phi = np.deg2rad(90.0 - lat), np.deg2rad(lon % 360.0)
        vertical = synthesise(cU, theta, phi)
        if wet:
            other = synthesise(cS, theta, phi)
        else:
            v_theta, v_phi = synthesise_gradient(cV, theta, phi)
            u_theta, u_phi = v_theta, v_phi / np.sin(theta)
            if cW is not None:
                w_theta, w_phi = synthesise_gradient(cW, theta, phi)
                u_theta = u_theta + w_phi / np.sin(theta)
                u_phi = u_phi - w_theta
            east, north = u_phi, -u_theta
            other = -(east * np.sin(dip) + north * np.cos(dip))
        out.append((vertical, other))
    return out, wet


def time_series_panel(ax, times, curves, bands, ylabel: str,
                      label_bands: bool = True) -> list[str]:
    """Node series from zero before the event on relaxation_axis, each
    curve (values, color, linestyle, label), and the relaxation times'
    bands marked; returns the bands' keys."""
    for values, color, ls, label in curves:
        ax.plot(*stepped(times, values), color=color, ls=ls, marker="o",
                ms=2.5, markevery=slice(2, None), label=label)
    relaxation_axis(ax, times)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_ylabel(ylabel)
    return mark_relaxation_times(ax, bands, label=label_bands)


def postseismic(name: str, title: str) -> None:
    """A source's co-seismic and post-seismic response, a run through
    time: maps centred on the source of the vertical displacement at the
    event and of its change since the event at two later nodes (about
    twice the deck's shortest Maxwell time, and the last); with an ocean,
    of the relative sea-level change since the event at the last node
    over the ocean, and without one of the horizontal displacement's
    change; and the vertical displacement and the sea level (without an
    ocean, the horizontal displacement up dip) through time at three
    places, 2° cap means, from zero before the event on a time axis
    linear about the event and logarithmic after it, the deck's Maxwell
    times marked."""
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    kind, source = seismic_source(name)
    file = read_surface_file(path)
    times, unit = node_times(file)
    bands = relaxation_bands(name)
    last = len(times) - 1
    middle = int(np.argmin(np.abs(np.log(np.maximum(times, 1e-30))
                                  - np.log(2.0 * bands[0][2]))))
    check_components(file)
    width = smoothing_width(file)
    extent = source_extent(source)
    event = motion(file, extent, 0)
    later = {n: motion(file, extent, n, since=0) for n in (middle, last)}

    fig = plt.figure(figsize=(16, 10))
    spec = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25])
    ax = regional_map(fig, 2, 3, 1, source, extent)
    ax.set_subplotspec(spec[0, 0])
    mesh = plot_field(ax, event, "up", title="vertical displacement at the "
                      f"event, {times[0]:g} {unit} (m)",
                      limit=region_limit(event, ["up"], extent))
    mark_source(ax, kind, source)
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="m")
    for k, n in enumerate((middle, last)):
        ax = regional_map(fig, 2, 3, 2 + k, source, extent)
        ax.set_subplotspec(spec[0, 1] if k == 0 else spec[1, 0])
        mesh = plot_field(ax, later[n], "up",
                          limit=region_limit(later[n], ["up"], extent),
                          title=f"its change from the event to {times[n]:g} "
                                f"{unit} (m)")
        mark_source(ax, kind, source)
        fig.colorbar(mesh, ax=ax, shrink=0.8, label="m")
    ax = regional_map(fig, 2, 3, 5, source, extent)
    ax.set_subplotspec(spec[1, 1])
    sites = source_sites(source)
    series, wet = site_series(file, source, sites)
    if wet:
        mesh = plot_field(ax, later[last], "sea",
                          limit=region_limit(later[last], ["sea"], extent),
                          title="sea-level change since the event, at "
                                f"{times[last]:g} {unit} (m)")
    else:
        mesh = plot_field(ax, later[last], "horizontal", cmap="YlOrRd",
                          symmetric=False,
                          title="horizontal motion since the event, at "
                                f"{times[last]:g} {unit} (m)")
        mesh.set_clim(0.0, region_limit(later[last], ["horizontal"], extent))
        arrows(ax, grid_motion(file, last, since=0), extent)
    mark_source(ax, kind, source)
    mark_sites(ax, [site for _, site in sites], colors=["C0", "C1", "C2"])
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="m")

    cell = spec[:, 2].subgridspec(2, 1, hspace=0.12)
    ax_u = fig.add_subplot(cell[0])
    ax_s = fig.add_subplot(cell[1], sharex=ax_u)
    keys = time_series_panel(
        ax_u, times, [(up, f"C{k}", "-", f"{label} ({lat:.1f}, {lon:.1f})")
                      for k, ((label, (lat, lon)), (up, _))
                      in enumerate(zip(sites, series))],
        bands, "vertical displacement (m)")
    time_series_panel(
        ax_s, times, [(other, f"C{k}", "-", label)
                      for k, ((label, _), (_, other))
                      in enumerate(zip(sites, series))],
        bands, "relative sea level (m)" if wet else
        "horizontal displacement up dip (m)", label_bands=False)
    ax_u.set_title("at three places: the co-seismic step and the "
                   "relaxation", fontsize=10, pad=14)
    ax_u.legend(fontsize=8, loc="best")
    ax_u.tick_params(labelbottom=False)
    ax_s.set_title("relative sea level there" if wet else
                   "the horizontal displacement up dip there", fontsize=10)
    ax_s.set_xlabel(f"time on the run's clock ({unit}; the event at "
                    f"{times[0]:g}; linear to {times[1]:g}, logarithmic "
                    "after)\n" + "\n".join(keys), fontsize=8)
    fig.suptitle(f"{title}, lmax {file.grid.lmax}\nmaps centred on the "
                 "source, meridians and parallels 5° apart, the places of the "
                 f"time series marked; maps and series {means(width)}",
                 fontsize=10)
    fig.tight_layout()
    save(fig, name, "postseismic")


# The figures comparing a run with gravity and its twin without, in a
# folder of their own rather than either run's.
GRAVITY_PAIRS = "with_and_without_gravity"


def postseismic_gravity() -> None:
    """The same fault on the same Earth without an ocean, with gravity
    (no_ocean_postseismic) and without it (no_gravity_postseismic): the
    vertical and horizontal displacement through time at three places
    and the vertical displacement across the fault at the event and at
    the last node, both runs on each panel, all 2° Gaussian means."""
    names = {"with": "no_ocean_postseismic", "without": "no_gravity_postseismic"}
    paths = {k: OUT / n / f"{n}.nc" for k, n in names.items()}
    if not all(p.exists() for p in paths.values()):
        return
    kind, source = seismic_source(names["without"])
    files = {k: read_surface_file(p) for k, p in paths.items()}
    widths = {smoothing_width(f) for f in files.values()}
    if len(widths) != 1:
        raise ValueError(f"the pair is smoothed over caps of {widths} degrees")
    width = widths.pop()
    times, unit = node_times(files["without"])
    bands = relaxation_bands(names["without"])
    sites = source_sites(source)
    series = {k: site_series(f, source, sites)[0] for k, f in files.items()}
    styles = {"with": "--", "without": "-"}

    fig = plt.figure(figsize=(16, 6.6))
    spec = fig.add_gridspec(1, 3)
    keys = []
    for col, (part, ylabel) in enumerate(
            ((0, "vertical displacement (m)"),
             (1, "horizontal displacement up dip (m)"))):
        ax = fig.add_subplot(spec[0, col])
        curves = [(series[key][k][part], f"C{k}", styles[key], None)
                  for k in range(len(sites)) for key in ("without", "with")]
        keys = time_series_panel(ax, times, curves, bands, ylabel)
        ax.set_xlabel(f"years since the event (linear to {times[1]:g}, "
                      "logarithmic after)", fontsize=8)
        ax.set_title("vertical" if part == 0 else "horizontal up dip",
                     fontsize=10, pad=14)
        if part == 0:
            ax.legend([plt.Line2D([], [], color=f"C{k}")
                       for k in range(len(sites))]
                      + [plt.Line2D([], [], color="0.3", ls=styles[key])
                         for key in ("without", "with")],
                      [label for label, _ in sites]
                      + ["without gravity", "with gravity"], fontsize=8,
                      loc="best")

    ax = fig.add_subplot(spec[0, 2])
    last = len(times) - 1
    for n, color in ((0, "0.4"), (last, "C3")):
        for key, file in files.items():
            distance, up, _, _, _ = profile(file, source, index=n)
            ax.plot(distance, up, color=color, ls=styles[key],
                    label=f"{times[n]:g} {unit}, {key} gravity")
    half = 0.5 * source["width_km"] * np.cos(np.deg2rad(source["dip_deg"]))
    ax.axvspan(-half, half, color="0.88", zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel("distance across the fault through its centroid (km, "
                  "positive down dip;\nits surface projection shaded)",
                  fontsize=8)
    ax.set_ylabel("vertical displacement (m)")
    ax.set_title(f"across the fault, at the event and at {times[last]:g} "
                 f"{unit}", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("sl3d solve, without an ocean: the thrust of "
                 "sea_level_postseismic with gravity (no_ocean_postseismic) "
                 "and without it (no_gravity_postseismic), on "
                 f"{deck_of(names['without']).name}, lmax "
                 f"{files['with'].grid.lmax}\nseries and profiles "
                 f"{means(width)}\n"
                 + "; ".join(keys), fontsize=10)
    fig.tight_layout()
    save(fig, GRAVITY_PAIRS, "postseismic")


def double_couple(strike: float, dip: float, rake: float) -> np.ndarray:
    """The moment tensor of a slip of unit moment, in the spherical polar
    basis (r up, theta south, phi east) at the point where its strike, dip
    and rake are given (degrees; Aki & Richards 2002, box 4.4), 3 x 3."""
    s, d, r = np.deg2rad([strike, dip, rake])
    mxx = -(np.sin(d) * np.cos(r) * np.sin(2 * s)
            + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    mxy = (np.sin(d) * np.cos(r) * np.cos(2 * s)
           + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
    mxz = -(np.cos(d) * np.cos(r) * np.cos(s)
            + np.cos(2 * d) * np.sin(r) * np.sin(s))
    myy = (np.sin(d) * np.cos(r) * np.sin(2 * s)
           - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
    myz = -(np.cos(d) * np.cos(r) * np.sin(s)
            - np.cos(2 * d) * np.sin(r) * np.cos(s))
    mzz = np.sin(2 * d) * np.sin(r)
    # (x north, y east, z down) to (r, theta, phi) = (-z, -x, y)
    return np.array([[mzz, mxz, -myz], [mxz, mxx, -mxy], [-myz, -mxy, myy]])


def spherical_basis(lat: float, lon: float) -> np.ndarray:
    """The unit vectors r, theta, phi at a place as the columns of a 3 x 3
    Cartesian matrix."""
    t, p = np.deg2rad(90.0 - lat), np.deg2rad(lon)
    return np.array([[np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), -np.sin(p)],
                     [np.sin(t) * np.sin(p), np.cos(t) * np.sin(p), np.cos(p)],
                     [np.cos(t), -np.sin(t), 0.0]])


def moment_kernel_table(path: Path):
    """A moment-kernel table of sl3d kernel: its points' (latitude,
    longitude, depth in m), and each point's kernel summed over the nodes,
    the derivative of J with respect to a change of its moment tensor held
    from the onset, components (rr, tt, pp, rt, rp, tp) in J's unit per N m
    (an off-diagonal member setting its pair)."""
    points, rows = [], []
    for line in path.read_text().splitlines():
        words = line.split()
        if line.startswith("# point ") and words[2].isdigit():
            points.append([float(w) for w in words[5:8]])
        elif words and not line.startswith("#"):
            rows.append([float(w) for w in words])
    points, rows = np.array(points), np.array(rows)
    held = np.zeros((len(points), 6))
    for row in rows:
        held[int(row[2])] += row[3:9]
    return points, held


def kernel_postseismic() -> None:
    """The kernels of kernel_postseismic's three measurements about the
    fault-driven run: in the moment tensors of the fault's patches (each
    held from the onset: the per-node rows summed), the six components at
    each patch along strike, shallow row and deep; the patches' kernels
    contracted with the fault's own mechanism, the measurement's
    derivative with respect to each patch's slip; and the radial
    viscosity kernels, as kernel1d's figure draws them."""
    name = "kernel_postseismic"
    directory = OUT / name
    if not (directory / "measurements.txt").exists():
        return
    values = measurements(directory)
    _, fault = seismic_source(name)
    down = fault.get("patches_down_dip", 1)
    along = fault.get("patches_along_strike", 1)
    mechanism = double_couple(fault["strike_deg"], fault["dip_deg"],
                              fault["rake_deg"])
    at_centroid = spherical_basis(fault["latitude"], fault["longitude"])
    in_space = at_centroid @ mechanism @ at_centroid.T
    labels = ["rr", "θθ", "φφ", "rθ", "rφ", "θφ"]
    per = 1.0e21  # N m
    share = fault["moment_nm"] / (along * down)

    fig = plt.figure(figsize=(15, 10))
    spec = fig.add_gridspec(len(values), 3, width_ratios=[1.5, 1.0, 0.9])
    for row, (stem, value) in enumerate(values.items()):
        points, held = moment_kernel_table(directory / f"{stem}_moment_kernels.txt")
        # the patches as (row down dip, place along strike), the patch
        # index running down dip fastest
        ordered = held.reshape(along, down, 6).transpose(1, 0, 2)
        table = 1.0e3 * per * np.concatenate(list(ordered), axis=0).T
        ax = fig.add_subplot(spec[row, 0])
        top = np.max(np.abs(table))
        image = ax.imshow(table, cmap="RdBu_r", vmin=-top, vmax=top,
                          aspect="auto")
        for k in range(1, down):
            ax.axvline(k * along - 0.5, color="k", lw=1.2)
        depths = points[:, 2].reshape(along, down) / 1.0e3
        ax.set_xticks(range(along * down),
                      [f"{k + 1}" for _ in range(down) for k in range(along)],
                      fontsize=7)
        ax.set_yticks(range(6), labels)
        ax.set_xlabel("patch along strike (south to north), in each row down "
                      "dip: " + ", ".join(
                          f"{depths[:, k].min():.0f}–{depths[:, k].max():.0f} km"
                          for k in range(down)), fontsize=8)
        ax.set_title(f"{stem} (J = {value:.4g} m): the moment kernels, held "
                     "from the onset", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.9,
                     label="mm per 10$^{21}$ N m")
        # each patch's kernel contracted with the fault's own mechanism, in
        # the patch's basis
        slip = np.empty(len(points))
        for p, (lat, lon, _) in enumerate(points):
            basis = spherical_basis(lat, lon)
            m = basis.T @ in_space @ basis
            slip[p] = held[p] @ np.array([m[0, 0], m[1, 1], m[2, 2],
                                          m[0, 1], m[0, 2], m[1, 2]])
        ax = fig.add_subplot(spec[row, 1])
        width = 0.8 / down
        for k in range(down):
            ax.bar(np.arange(along) + (k - 0.5 * (down - 1)) * width,
                   1.0e3 * per * slip.reshape(along, down)[:, k], width,
                   color=f"C{k}",
                   label=f"row {k + 1} down dip, "
                         f"{depths[:, k].min():.0f}–{depths[:, k].max():.0f} km")
        ax.axhline(0.0, color="0.6", lw=0.5)
        ax.set_xticks(range(along), [f"{k + 1}" for k in range(along)])
        ax.set_xlabel("patch along strike (south to north)", fontsize=8)
        ax.set_ylabel("mm per 10$^{21}$ N m of slip")
        ax.set_title("per unit of each patch's slip; with the fault's own "
                     f"moment, {share * slip.sum():.4g} m", fontsize=10)
        ax.legend(fontsize=7)

    ax = fig.add_subplot(spec[:, 2])
    deepest = 300.0  # km: the depth the kernels reach, at least this
    for stem, value in values.items():
        data = np.loadtxt(directory / f"{stem}_viscosity_kernel.txt")
        radius, kernel = data[:, 0], data[:, 1]
        scale = np.max(np.abs(kernel)) or 1.0
        depth = (radius[-1] - radius) / 1e3
        ax.plot(kernel / scale, depth, label=f"{stem} (J = {value:.3g} m)")
        deepest = max(deepest, depth[np.abs(kernel) > 0.01 * scale].max())
    for top, bottom, *_ in maxwell_times(deck_of(name)):
        ax.axhline(bottom, color="0.75", lw=0.6, ls=":")
    ax.set_ylim(min(1.3 * deepest, 2900.0), 0.0)
    ax.axvline(0.0, color="0.7", linewidth=0.6)
    ax.set_xlabel("viscosity kernel K(r) per unit volume\n(dJ = $\\int$ K "
                  "d$\\eta$ dV), each scaled to its largest value")
    ax.set_ylabel("depth (km)")
    ax.set_title("the radial viscosity kernels\n(dotted: the bases of the "
                 "deck's viscous layers)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("sl3d kernel: a GPS station's eastward and upward motion 20 "
                 "years after a thrust off the synthetic South America, and a "
                 "tide gauge's sea level a century after,\nby the discrete "
                 f"adjoint, in the moment tensors of the fault's {along} by "
                 f"{down} patches and in the viscosity (lmax 32)", fontsize=11)
    fig.tight_layout()
    save(fig, name, "kernels")


def no_gravity_load() -> None:
    """Surface loads with and without gravity: no_gravity_load (a 5 degree
    cap of 2 km of ice's weight over Hudson Bay and its antipode, the
    problem without gravity, the load pressing by its weight alone) against
    no_ocean_load_cap (the cap alone, with gravity), the same history on
    the same radial Maxwell Earth: the displacement at the cap's centre
    through time, the displacement against distance from it at three
    times, and the map without gravity."""
    names = {"without": "no_gravity_load", "with": "no_ocean_load_cap"}
    paths = {k: OUT / n / f"{n}.nc" for k, n in names.items()}
    if not all(p.exists() for p in paths.values()):
        return
    terms = json.loads((CONFIGS / f"{names['without']}.json").read_text())[
        "forcing"]["load"]
    centres = [(t["cap"]["latitude"], t["cap"]["longitude"]) for t in terms]
    centre = centres[0]
    history = np.asarray(terms[0]["history"])
    files = {k: read_surface_file(p) for k, p in paths.items()}
    times = np.asarray(files["without"].time) / KYR
    grid = files["without"].grid
    disp = {k: np.asarray(f.variables["displacement"].values)
            for k, f in files.items()}
    styles = {"without": ("-", "without gravity (the cap and its antipode)"),
              "with": ("--", "with gravity (the cap alone)")}

    fig = plt.figure(figsize=(15.5, 5.0))
    spec = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    ax = fig.add_subplot(spec[0, 0])
    for key, values in disp.items():
        ls, label = styles[key]
        ax.plot(times, at_point(values, grid, *centre), color="k", ls=ls,
                label=label)
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel("time (kyr)")
    ax.set_ylabel("displacement at the cap's centre (m)")
    load = ax.twinx()
    load.fill_between(times, np.interp(times, history[:, 0], history[:, 1]),
                      color="0.88", zorder=0)
    load.set_ylim(0.0, 4.0)
    load.set_yticks([0.0, 1.0], ["0", "full"])
    load.set_ylabel("the load (shaded)", color="0.5")
    ax.set_zorder(load.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title("loaded over 2 to 3 kyr, held to 7, removed by 7.5",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")

    ax = fig.add_subplot(spec[0, 1])
    lats = 90.0 - np.rad2deg(np.asarray(grid.colatitudes))[:, None]
    lons = np.rad2deg(np.asarray(grid.longitudes))[None, :]
    distance = np.broadcast_to(distance_deg(lats, lons, *centre),
                               disp["without"].shape[1:]).ravel()
    order = np.argsort(distance)
    near = distance[order] <= 30.0
    chosen = ((3.0, "C0", "loaded, 3 kyr"), (7.0, "C3", "held, 7 kyr"),
              (12.0, "C2", "removed, 12 kyr"))
    for t, color, _ in chosen:
        n = int(np.argmin(np.abs(times - t)))
        for key, values in disp.items():
            ax.plot(distance[order][near], values[n].ravel()[order][near],
                    color=color, ls=styles[key][0])
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.axvline(5.0, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("distance from the cap's centre (degrees; the cap's width "
                  "dotted)")
    ax.set_ylabel("radial displacement (m)")
    ax.set_title("against distance: solid without gravity, dashed with",
                 fontsize=10)
    ax.legend([plt.Line2D([], [], color=c) for _, c, _ in chosen],
              [label for _, _, label in chosen], fontsize=8, loc="lower right")

    n = int(np.argmin(np.abs(times - 7.0)))
    ex = refined(grid, {"field": disp["without"][n]}, nlat=180)
    ax = map_axes(fig, 1, 3, 3, central_longitude=centre[1] + 90.0)
    ax.set_subplotspec(spec[0, 2])
    mesh = plot_field(ax, ex, "field", title="without gravity at "
                      f"{times[n]:g} kyr, the loads on (m)")
    mark_sites(ax, centres, colors=["C3", "C3"])
    fig.colorbar(mesh, ax=ax, shrink=0.8, orientation="horizontal",
                 pad=0.05, label="m")
    fig.suptitle("sl3d solve, the problem without gravity: a cap load and "
                 "its antipode, acting by their weight, against the cap with "
                 "gravity, on the radial Maxwell Earth (prem8, 1e21 Pa s, "
                 "lmax 32)", fontsize=11)
    fig.tight_layout()
    save(fig, GRAVITY_PAIRS, "load")


def sea_level_tide_migrating() -> None:
    """The synthetic deglaciation under migrating shorelines with a
    degree-2 tide switched on over the window: the relative sea-level
    change since 21 ka at four dates over each date's own ocean, the
    shorelines the run moved, and the tide's equilibrium surface."""
    name = "sea_level_tide_migrating"
    path = OUT / name / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    sea = np.asarray(file.variables["relative_sea_level_change"].values)
    ocean = np.asarray(file.variables["ocean_function"].values) > 0.5
    picks = [int(np.argmin(np.abs(dates - d))) for d in (15.0, 10.0, 5.0, 0.0)]
    fine = equiangular(180, 360)
    maps = {}
    for n in picks:
        wet = synthesise_grid(analyse_grid(ocean[n].astype(float), grid),
                              fine) > 0.5
        values = synthesise_grid(analyse_grid(sea[n], grid), fine)
        maps[f"sea{n}"] = np.where(wet, values, np.nan)
    ex = Example.from_fields(fine.colatitudes, fine.longitudes, maps)
    limit = float(np.nanpercentile(np.abs(np.concatenate(
        [v.ravel() for v in maps.values()])), 99.0))

    fig = plt.figure(figsize=(15, 9))
    for k, n in enumerate(picks):
        ax = world_map(fig, 2, 3, (1, 2, 4, 5)[k])
        mesh = plot_field(ax, ex, f"sea{n}", limit=limit,
                          title=f"sea-level change, 21 to {dates[n]:g} ka, "
                                "over its ocean (m)")
        fig.colorbar(mesh, ax=ax, shrink=0.75, extend="both", label="m",
                     orientation="horizontal", pad=0.04)

    ax = world_map(fig, 2, 3, 3)
    moved = ocean[-1].astype(float) - ocean[0].astype(float)
    shore = Example.from_fields(grid.colatitudes, grid.longitudes,
                                {"moved": np.where(moved != 0.0, moved,
                                                   np.nan)})
    plot_field(ax, shore, "moved",
               cmap=matplotlib.colors.ListedColormap(["C3", "C0"]), limit=1.0,
               title=f"the shorelines moved, 21 ka to the present (its lmax "
                     f"{grid.lmax} grid)")
    ax.legend([matplotlib.patches.Patch(color="C0"),
               matplotlib.patches.Patch(color="C3")],
              [f"sea flooded ({int(np.sum(moved > 0))} points)",
               f"land emerged ({int(np.sum(moved < 0))} points)"],
              fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncol=2)

    tide = json.loads((CONFIGS / f"{name}.json").read_text())["forcing"][
        "tide"][0]
    history = np.asarray(tide["history_kyr_bp"])
    latitude = np.linspace(-90.0, 90.0, 181)
    x = np.sin(np.deg2rad(latitude))
    # the orthonormal degree-2 zonal harmonic, sqrt(5 / 4 pi) P_2
    y20 = np.sqrt(5.0 / (4.0 * np.pi)) * 0.5 * (3.0 * x**2 - 1.0)
    gravity = 9.82
    ax = fig.add_subplot(2, 3, 6)
    for when in (21.0, 10.0, 0.0):
        factor = np.interp(when, history[::-1, 0], history[::-1, 1])
        ax.plot(latitude, -factor * tide["amplitude_m2_s2"] * y20 / gravity,
                label=f"{when:g} ka")
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.set_xlabel("latitude (degrees)")
    ax.set_ylabel("$-\\psi / g$ (m)")
    ax.set_title("the tide's equilibrium surface, degree 2, order 0, "
                 f"switched on\nto {tide['amplitude_m2_s2']:g} m$^2$ "
                 "s$^{-2}$ by the present", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("sl3d solve: the synthetic deglaciation under migrating "
                 "shorelines with a degree-2 tide switched on over the window "
                 f"(radial Maxwell Earth, lmax {grid.lmax})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "sea_level")


if __name__ == "__main__":
    love_numbers()
    sea_level_fingerprint("sea_level_fingerprint",
                          "sl3d solve, the fingerprint: 10% of the "
                          "synthetic Greenland melted", synthetic=True)
    sea_level_fingerprint_3d()
    sea_level_fingerprint("sea_level_fingerprint_ice7g",
                          "sl3d solve, the fingerprint: 10% of the "
                          "ICE-7G Greenland ice melted", synthetic=False)
    sea_level()
    sea_level_calibrated("sea_level_1d_calibrated", "sea_level_1d", None,
                         "sl3d solve (1D), migrating shorelines, initial "
                         "sea level calibrated against the present day")
    sea_level_calibrated("sea_level_1d_ice7g_calibrated", "sea_level_1d_ice7g",
                         ICE7G,
                         "sl3d solve (1D) on ICE-7G, migrating shorelines, "
                         "initial sea level calibrated against the present "
                         "day")
    sea_level_calibrated("sea_level_3d_calibrated", "sea_level_3d", None,
                         "sl3d solve (3D), migrating shorelines, initial "
                         "sea level calibrated against the present day")
    sea_level_ice7g()
    kernels1d()
    kernels3d()
    kernels1d_lateral()
    no_ocean()
    no_ocean_viscoelastic()
    no_ocean_1d_disk()
    no_ocean_1d_tide()
    no_ocean_tide_3d()
    no_ocean_tide_planet()
    no_ocean_tide_planet_relax()
    no_ocean_tide_oscillating()
    sea_level_fingerprint_3d_ice7g()
    sea_level_3d_ice7g()
    ocean_load()
    no_ocean_cap_by_location()
    no_ocean_load()
    timestep_study()
    coseismic("sea_level_static_coseismic",
              "sl3d solve, the static problem: the co-seismic response of a "
              "thrust off the synthetic South America\n(600 by 150 km, "
              "dipping 18° east, 2e22 N m as 12 by 3 point sources), with an "
              "ocean and full rotational feedback, on the elastic prem8 deck")
    megathrust = fault_file_drawn("sea_level_static_megathrust")
    if megathrust is not None:
        coseismic("sea_level_static_megathrust",
                  "sl3d solve, the static problem: the co-seismic response of "
                  "a curved megathrust off the synthetic South America (a "
                  "fault file: 940 km of trench,\na thrust tapered from 12 m, "
                  "at 36 by 6 quadrature points), with an ocean and full "
                  "rotational feedback, on prem8.postseismic.deck",
                  megathrust)
    postseismic("sea_level_postseismic",
                "sl3d solve: that thrust's co-seismic and post-seismic "
                "response with an ocean and full rotational feedback, on "
                "prem8.postseismic.deck (a 50 km elastic lid over a Maxwell "
                "mantle)")
    postseismic("sea_level_postseismic_burgers",
                "sl3d solve: that thrust's response on prem8.postseismic.deck "
                "with a Burgers asthenosphere (`model.burgers`: a Kelvin "
                "element of the unrelaxed modulus and 1e18 Pa s from 50 to "
                "217 km)")
    postseismic("no_ocean_postseismic",
                "sl3d solve, no ocean: that thrust's response without an "
                "ocean or a rotational feedback, on prem8.postseismic.deck")
    postseismic("no_gravity_postseismic",
                "sl3d solve, the problem without gravity: that thrust, no "
                "ocean, on prem8.postseismic.deck")
    postseismic_gravity()
    postseismic("no_ocean_seismic",
                "sl3d solve, no ocean: a thrust off north-east Japan (400 by "
                "150 km, dipping 12° to the west-north-west, 4e22 N m as 8 by "
                "3 point sources),\non prem8.postseismic.deck (a 50 km "
                "elastic lid over a Maxwell mantle)")
    kernel_postseismic()
    no_gravity_load()
    sea_level_tide_migrating()
