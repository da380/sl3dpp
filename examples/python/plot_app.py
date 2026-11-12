"""Figures of the sl3d example configs' outputs: run configs/run_all.sh first
(and, for the ICE-7G figures, get_ice7g.py, which makes run_all.sh run the
ICE-7G configs too), then `poetry run python plot_app.py` from
build/examples. Each config's outputs, in configs/out/<config>/, are read
with planetmodel's own readers, and its figures are written to
figures/sl3d/<config>/:

    love_numbers/love_numbers.png
    sea_level_fingerprint/fingerprint.png
    sea_level_fingerprint_3d/fingerprint.png
    sea_level_1d/sea_level.png        (sea_level_1d against sea_level_3d)
    sea_level_1d_calibrated/calibration.png
    sea_level_3d_calibrated/calibration.png
    kernel1d/kernels.png
    kernel/viscosity_kernels.png
    response/displacement.png
    response_viscoelastic/displacement.png
    response1d_disk/displacement.png
    response1d_tide/love_numbers.png
    response_tide_3d/body_tide.png
    response_tide_oscillating/oscillating.png   (with the _short and _long
                                                 runs' amplitudes and lags)
    sea_level_fingerprint_melt/ocean.png        (with response_melt)
    response_cap_weak/relaxation.png            (with response_cap_strong)
    timestep_study/accuracy.png                 (the timestep_* runs)

and, where the ICE-7G runs exist, sea_level_fingerprint_ice7g/fingerprint.png,
sea_level_1d_ice7g/sea_level.png and
sea_level_1d_ice7g_calibrated/calibration.png; where those on the custom
PREM deck with S40RTS exist too, sea_level_fingerprint_3d_ice7g/fingerprint.png
and sea_level_3d_ice7g/sea_level.png (with sea_level_1d_ice7g_custom); where
the toy planet's runs exist (make_planet_deck.py),
response_tide_planet/body_tide.png and
response_tide_planet_relax/love_numbers.png; and where the load from a file
and from a table have run, response_load_field/agreement.png (with
response_load_cap and response_load_coefficients). A missing output skips
its figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import equiangular
from planetmodel.harmonics import analyse_grid, synthesise, synthesise_grid
from planetmodel.modelfile import read_model_file
from planetmodel.surfacefile import read_surface_file

from inputs import distance_deg
from plotutil import (DATA, HAVE_CARTOPY, Example, map_axes, mark_sites,
                      plot_field, save_figure, world_map)

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
    """Writes a config's figure, figures/sl3d/<config>/<name>.png."""
    save_figure(fig, FIGURES / config / f"{name}.png", dpi=150)


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
    fig.suptitle("sl3d sea-level, the fingerprint on the synthetic 3D elastic "
                 "Earth: 10% of the synthetic Greenland melted\n(black: the "
                 "melt; blue: the stiff craton; red: the soft ridge)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, name, "fingerprint")


def histories(path: Path):
    """A sea-level run's grid, node dates (kyr BP) and sea-level change."""
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
    fig.suptitle("sl3d sea-level, 1D (solid) and 3D viscosity (dashed): the "
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
    fig.suptitle("sl3d sea-level (1D) on ICE-7G")
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
    fig.suptitle("sl3d kernel1d: kernels of the synthetic deglaciation's "
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


def response() -> None:
    path = OUT / "response" / "response.nc"
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
    fig.suptitle("sl3d response: a cap load over Hudson Bay on the synthetic 3D "
                 "elastic Earth")
    fig.tight_layout()
    save(fig, "response", "displacement")


def response_viscoelastic() -> None:
    """The cap load grown and removed on the synthetic 3D viscoelastic Earth:
    the displacement under full load, just after the load has gone and at the
    end, the lateral structure's part of the last, and the histories at the
    cap's centre of the load, the displacement and the radial model's."""
    path = OUT / "response_viscoelastic" / "response_viscoelastic.nc"
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
    fig.suptitle("sl3d response: a cap load grown and removed over Hudson Bay "
                 "on the synthetic 3D viscoelastic Earth")
    fig.tight_layout()
    save(fig, "response_viscoelastic", "displacement")


def response1d_disk() -> None:
    """A disk load held and then removed on the radial viscoelastic Earth:
    the displacement against distance from the disk's centre at a few times,
    the peripheral bulge growing under the load and the uplift after it."""
    path = OUT / "response1d_disk" / "response1d_disk.nc"
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
    fig.suptitle("sl3d response: a disk load of 8 degrees held for 10 kyr on "
                 "the radial viscoelastic Earth")
    fig.tight_layout()
    save(fig, "response1d_disk", "displacement")


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


def response1d_tide() -> None:
    """A degree-2 tidal potential switched on and held: the tidal Love
    numbers of the radial viscoelastic Earth through time, from the elastic
    values at the start towards the fluid limit, read from the exchange
    tables of the nodes."""
    held = held_tide("response1d_tide")
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
    ax.set_title("sl3d response: a held degree-2 tide on the radial "
                 "viscoelastic Earth")
    fig.tight_layout()
    save(fig, "response1d_tide", "love_numbers")


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


def response_tide_3d() -> None:
    """The lunar body tide on the synthetic 3D elastic Earth: the radial
    model's displacement, the lateral structure's part of the 3D model's
    (3D minus radial) and the spectrum of that part by degree. The radial
    model answers a degree-2 potential in degree 2 alone; everything the
    3D model puts into the other degrees is the lateral coupling."""
    name = "response_tide_3d"
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
    fig.suptitle("sl3d response: the lunar body tide on the synthetic 3D "
                 "elastic Earth (gold: the sub-lunar point; blue: the stiff "
                 "craton; red: the soft ridge)", fontsize=11)
    fig.tight_layout()
    save(fig, name, "body_tide")


def response_tide_planet() -> None:
    """The Sun's tide on the toy planet of make_planet_deck.py, elastic and
    radial: the displacement and the potential perturbation, with the
    planet's degree-2 Love numbers read off the exchange table, from each
    order's coefficients over the potential's."""
    name = "response_tide_planet"
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
        "sl3d response: the Sun's tide on the toy planet (radius "
        f"{radius / 1e3:.0f} km, g = {gravity:.2f} m s$^{{-2}}$; gold: the "
        "sub-solar point)\n"
        f"degree-2 tidal Love numbers h$_2$ = {np.mean(h2):.4f}, "
        f"k$_2$ = {np.mean(k2):.4f} (the three orders agree to "
        f"{spread:.0e})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "body_tide")


def response_tide_planet_relax() -> None:
    """A held degree-2 tide on the toy planet's Maxwell mantle: its tidal
    Love numbers through time, beside the Earth's (response1d_tide) where
    that has run."""
    planet = held_tide("response_tide_planet_relax")
    if planet is None:
        return
    earth = held_tide("response1d_tide")
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
    ax.set_title("sl3d response: a held degree-2 tide on the toy planet's "
                 "Maxwell mantle", fontsize=11)
    fig.tight_layout()
    save(fig, "response_tide_planet_relax", "love_numbers")


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
    history = np.asarray(tide["history_kyr"])
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


def response_tide_oscillating() -> None:
    """An oscillating degree-2 tide on the radial viscoelastic Earth: the
    potential and the response through the run at one period, the response
    lagging; and the Love numbers' amplitudes and lags at the three periods
    of the oscillating configs that have run."""
    name = "response_tide_oscillating"
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
    held = held_tide("response1d_tide")
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
    fig.suptitle("sl3d response: an oscillating degree-2 tide on the radial "
                 "viscoelastic Earth (prem8, a 1e21 Pa s mantle)")
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
    fig.suptitle("sl3d sea-level, the fingerprint of 10% of ICE-7G's Greenland "
                 "ice melted, on PREM (no ocean) with S40RTS's lateral moduli "
                 f"(lmax {grid.lmax})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "fingerprint")


def site_histories(path: Path, sites):
    """A sea-level run's dates (kyr BP) and each site's relative sea level,
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
    fig.suptitle("sl3d sea-level on ICE-7G: PREM (no ocean) with S40RTS's "
                 "lateral viscosity and moduli, against its radial model "
                 f"(lmax {grid.lmax})", fontsize=11)
    fig.tight_layout()
    save(fig, name, "sea_level")


def ocean_load() -> None:
    """What the ocean adds: one ice cap melted on the synthetic Earth,
    through `response` (the ice's load alone) and through `sea-level` (the
    fingerprint, the meltwater spread over the ocean): the displacement of
    each, their difference (the ocean's load), and the sea-level change."""
    name, alone = "sea_level_fingerprint_melt", "response_melt"
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
    mesh = plot_field(ax, ex, "alone", title="displacement under the melt "
                      "(response, the load alone; m)")
    mark_sites(ax, [cap])
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = world_map(fig, 2, 2, 2, central_longitude=-100.0)
    mesh = plot_field(ax, ex, "difference", title="displacement, sea-level "
                      "minus response: the ocean's load (cm)")
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
            lw=1.5, label="the load alone (response)")
    for where, color, label in ((ocean, "C0", "with the ocean, under it"),
                                (~ocean, "C5", "with the ocean, on land")):
        pick = far & where
        ax.plot(distance[pick], 1e2 * with_ocean[pick], ".", ms=1.0,
                color=color, alpha=0.4, label=label)
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_xlim(15.0, 180.0)
    ax.set_xlabel("distance from the cap's centre (degrees)")
    ax.set_ylabel("displacement (cm)")
    ax.set_title("away from the melt: the load alone, and with the ocean",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=8)
    fig.suptitle("sl3d: 1 km of ice (a 3 degree cap) melted on the synthetic "
                 "Earth, elastic, without rotation:\n`response` (the ice's "
                 "load alone) against `sea-level` (the fingerprint, the "
                 "meltwater in the ocean)", fontsize=11)
    fig.tight_layout()
    save(fig, name, "ocean")



def response_cap_by_location() -> None:
    """The same cap loaded and removed at two places of the synthetic 3D
    viscoelastic Earth, over weak and over stiff mantle: the viscosity
    file's lateral pattern with the two caps, and the displacement at each
    cap's centre through time, 3D and radial model."""
    runs = {"weak": "response_cap_weak", "strong": "response_cap_strong"}
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
    fig.suptitle("sl3d response: the same cap load over weak and over stiff "
                 "mantle on the synthetic 3D viscoelastic Earth "
                 "(viscosity_log.nc)", fontsize=11)
    fig.tight_layout()
    save(fig, "response_cap_weak", "relaxation")


def response_load() -> None:
    """One load history given three ways: an analytic cap with a history
    (response_load_cap), the same cap sampled into a surface file on a time
    axis read with a time origin (response_load_field), and its expansion
    as a coefficient table with the cap's history (response_load_coefficients):
    the forcing on the run's clock and on the file's axis, the displacement
    at the cap's centre through time, and how far each run departs from the
    analytic one anywhere on the grid."""
    names = ["response_load_cap", "response_load_field",
             "response_load_coefficients"]
    paths = {n: OUT / n / f"{n}.nc" for n in names}
    if not all(p.exists() for p in paths.values()):
        return
    term = json.loads((CONFIGS / "response_load_cap.json").read_text())[
        "forcing"]["load"][0]
    centre = (term["cap"]["latitude"], term["cap"]["longitude"])
    history = np.asarray(term["history_kyr"])
    field = json.loads((CONFIGS / "response_load_field.json").read_text())[
        "forcing"]["load"][0]["field"]
    origin = field.get("time_origin_kyr", 0.0)
    stored = read_surface_file(CONFIGS / field["file"])
    # the file's own axis, in its own time scale
    stored_times = np.asarray(stored.time) * stored.scales.time / KYR
    stored_load = at_point(
        stored.variables[field.get("variable", "load")].values, stored.grid,
        *centre)
    files = {n: read_surface_file(p) for n, p in paths.items()}
    times = np.asarray(files[names[0]].time) / KYR
    reference = np.asarray(files[names[0]].variables["displacement"].values)
    styles = {"response_load_cap": ("C0", "-", "cap, history_kyr"),
              "response_load_field": ("C1", "--",
                                      f"field file, time_origin_kyr {origin:g}"),
              "response_load_coefficients": ("C2", ":",
                                             "coefficient table, history_kyr")}

    fig, (ax_f, ax, ax_e) = plt.subplots(1, 3, figsize=(15, 4.4))
    factor = np.interp(times, history[:, 0], history[:, 1])
    ax_f.plot(times, factor, color="C0", lw=2.0, label="history_kyr of the "
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
    fig.suptitle("sl3d response: a load history from a file and from a "
                 "coefficient table, on the radial viscoelastic Earth "
                 "(prem8, 1e21 Pa s)", fontsize=11)
    fig.tight_layout()
    save(fig, "response_load_field", "agreement")


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


if __name__ == "__main__":
    love_numbers()
    sea_level_fingerprint("sea_level_fingerprint",
                          "sl3d sea-level, the fingerprint: 10% of the "
                          "synthetic Greenland melted", synthetic=True)
    sea_level_fingerprint_3d()
    sea_level_fingerprint("sea_level_fingerprint_ice7g",
                          "sl3d sea-level, the fingerprint: 10% of the "
                          "ICE-7G Greenland ice melted", synthetic=False)
    sea_level()
    sea_level_calibrated("sea_level_1d_calibrated", "sea_level_1d", None,
                         "sl3d sea-level (1D), migrating shorelines, initial "
                         "sea level calibrated against the present day")
    sea_level_calibrated("sea_level_1d_ice7g_calibrated", "sea_level_1d_ice7g",
                         ICE7G,
                         "sl3d sea-level (1D) on ICE-7G, migrating shorelines, "
                         "initial sea level calibrated against the present "
                         "day")
    sea_level_calibrated("sea_level_3d_calibrated", "sea_level_3d", None,
                         "sl3d sea-level (3D), migrating shorelines, initial "
                         "sea level calibrated against the present day")
    sea_level_ice7g()
    kernels1d()
    kernels3d()
    response()
    response_viscoelastic()
    response1d_disk()
    response1d_tide()
    response_tide_3d()
    response_tide_planet()
    response_tide_planet_relax()
    response_tide_oscillating()
    sea_level_fingerprint_3d_ice7g()
    sea_level_3d_ice7g()
    ocean_load()
    response_cap_by_location()
    response_load()
    timestep_study()
