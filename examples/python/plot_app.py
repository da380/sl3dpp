"""Figures of the sl3d example configs' outputs: run configs/run_all.sh first
(and, for the ICE-7G figures, get_ice7g.py, which makes run_all.sh run the
ICE-7G configs too), then `poetry run python plot_app.py` from
build/examples. The outputs in configs/out/ are read with planetmodel's own
readers, and figures of each command are written beside this script:
app_love_numbers.png, app_sea_level_fingerprint.png, app_sea_level.png,
app_sea_level_1d_calibrated.png, app_sea_level_3d_calibrated.png,
app_kernels1d.png, app_kernels3d.png and app_response.png,
app_response_viscoelastic.png, app_response1d_disk.png and
app_response1d_tide.png, plus app_sea_level_fingerprint_ice7g.png,
app_sea_level_1d_ice7g.png and app_sea_level_1d_ice7g_calibrated.png
where the ICE-7G runs exist. A missing output skips its figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from planetmodel.modelfile import read_model_file
from planetmodel.surfacefile import read_surface_file

from plotutil import Example, mark_sites, plot_field, world_map


OUT = Path("configs/out")
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


def nearest(grid, lat: float, lon: float) -> tuple[int, int]:
    """The grid point nearest to a place."""
    i = int(np.argmin(np.abs(np.asarray(grid.colatitudes)
                             - np.deg2rad(90.0 - lat))))
    j = int(np.argmin(np.abs(np.asarray(grid.longitudes)
                             - np.deg2rad(lon % 360.0))))
    return i, j


def love_numbers() -> None:
    path = OUT / "love_numbers.txt"
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
    fig.savefig("app_love_numbers.png", dpi=150)
    print("wrote app_love_numbers.png")


def sea_level_fingerprint(name: str, title: str, synthetic: bool) -> None:
    path = OUT / f"{name}.nc"
    if not path.exists():
        return
    ex = surface(path, ["ice_change", "sea_level_change", "displacement",
                        "potential"])
    fig = plt.figure(figsize=(11, 7))
    panels = [
        ("ice_change", "ice thickness change (m)", "Blues_r", False),
        ("sea_level_change", "sea level change (m)", "RdBu_r", True),
        ("displacement", "vertical displacement (m)", "RdBu_r", True),
        ("potential", "potential perturbation (m$^2$ s$^{-2}$)", "RdBu_r",
         True),
    ]
    for k, (field, label, cmap, symmetric) in enumerate(panels):
        ax = world_map(fig, 2, 2, k + 1,
                       state=None if synthetic else "ice7g.nc")
        mesh = plot_field(ax, ex, field, cmap=cmap, symmetric=symmetric,
                          title=label)
        fig.colorbar(mesh, ax=ax, shrink=0.7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"app_{name}.png", dpi=150)
    print(f"wrote app_{name}.png")


def histories(path: Path):
    """A sea-level run's grid, node dates (kyr BP) and sea-level change."""
    file = read_surface_file(path)
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    sea_level = np.asarray(file.variables["sea_level_change"].values)
    return file.grid, dates, sea_level


def sea_level() -> None:
    runs = [(name, OUT / f"{name}.nc")
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
                                 {"sea_level_change": sea_level[-1]})
        ax = world_map(fig, 2, 2, k + 1)
        mesh = plot_field(ax, ex, "sea_level_change",
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
    fig.savefig("app_sea_level.png", dpi=150)
    print("wrote app_sea_level.png")


def sea_level_calibrated(name: str, reference: str, state: str | None,
                         title: str) -> None:
    """A migrating-shoreline run whose initial sea level was calibrated
    against the present day, beside the uncalibrated run from the state
    file's own initial sea level: what the calibration changed at the
    start, and where the ocean moved over the history."""
    path = OUT / f"{name}.nc"
    if not path.exists():
        return
    file = read_surface_file(path)
    grid = file.grid
    dates = np.asarray(file.meta["dates_kyr_bp"], dtype=float)
    calibrated = np.asarray(file.variables["initial_sea_level"].values)
    ocean = np.asarray(file.variables["ocean_function"].values)
    sea_level = np.asarray(file.variables["sea_level_change"].values)
    fields = {"flooded": ocean[-1] - ocean[0],
              "sea_level_change": sea_level[-1]}
    reference_path = OUT / f"{reference}.nc"
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
    mesh = plot_field(ax, ex, "sea_level_change",
                      title=f"sea level change, {dates[0]:g} ka to the "
                            "present (m)")
    fig.colorbar(mesh, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    ax.text(0.0, 0.9, "initial sea level: " + file.meta["initial_sea_level"],
            wrap=True, fontsize=9, va="top")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"app_{name}.png", dpi=150)
    print(f"wrote app_{name}.png")


def sea_level_ice7g() -> None:
    path = OUT / "sea_level_1d_ice7g.nc"
    if not path.exists():
        return
    grid, dates, sea_level = histories(path)
    sites = [("Hudson Bay", (60.0, 275.0)), ("Gulf of Bothnia", (64.0, 21.0)),
             ("Barbados", (13.1, 300.4))]
    fig = plt.figure(figsize=(12, 7.5))
    for k, index in enumerate([dates.size // 2, dates.size - 1]):
        ex = Example.from_fields(grid.colatitudes, grid.longitudes,
                                 {"sea_level_change": sea_level[index]})
        ax = world_map(fig, 2, 2, k + 1, state="ice7g.nc")
        mesh = plot_field(ax, ex, "sea_level_change",
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
    fig.savefig("app_sea_level_1d_ice7g.png", dpi=150)
    print("wrote app_sea_level_1d_ice7g.png")


def measurements(directory: Path) -> dict[str, float]:
    """A kernel run's forward values, by measurement name."""
    values = {}
    for line in (directory / "measurements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            values[parts[0]] = float(parts[2])
    return values


def kernels1d() -> None:
    directory = OUT / "kernels1d"
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
    fig.savefig("app_kernels1d.png", dpi=150)
    print("wrote app_kernels1d.png")


def kernels3d() -> None:
    directory = OUT / "kernels3d"
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
    fig.savefig("app_kernels3d.png", dpi=150)
    print("wrote app_kernels3d.png")


def response() -> None:
    path = OUT / "response.nc"
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
    fig.savefig("app_response.png", dpi=150)
    print("wrote app_response.png")


def response_viscoelastic() -> None:
    """The cap load grown and removed on the synthetic 3D viscoelastic Earth:
    the displacement under full load, just after the load has gone and at the
    end, the lateral structure's part of the last, and the histories at the
    cap's centre of the load, the displacement and the radial model's."""
    path = OUT / "response_viscoelastic.nc"
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
    fig.savefig("app_response_viscoelastic.png", dpi=150)
    print("wrote app_response_viscoelastic.png")


def response1d_disk() -> None:
    """A disk load held and then removed on the radial viscoelastic Earth:
    the displacement against distance from the disk's centre at a few times,
    the peripheral bulge growing under the load and the uplift after it."""
    path = OUT / "response1d_disk.nc"
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
    fig.savefig("app_response1d_disk.png", dpi=150)
    print("wrote app_response1d_disk.png")


def response1d_tide() -> None:
    """A degree-2 tidal potential switched on and held: the tidal Love
    numbers of the radial viscoelastic Earth through time, from the elastic
    values at the start towards the fluid limit, read from the exchange
    tables of the nodes."""
    tables = sorted(OUT.glob("response1d_tide_*.txt"))
    if not tables:
        return
    times, h2, k2 = [], [], []
    for table in tables:
        scalars, rows = {}, {}
        for line in table.read_text().splitlines():
            if line.startswith("#"):
                continue
            words = line.split()
            if words[0].isdigit():
                rows[(int(words[0]), int(words[1]))] = [float(w)
                                                        for w in words[2:]]
            else:
                scalars[words[0]] = words[1:]
        gravity = float(scalars["surface_gravity"][0])
        amplitude = float(scalars["tide_amplitude_m2_s2"][0])
        times.append(float(scalars["time_kyr"][0]))
        # the potential is the physical one, negative where it attracts, so
        # the conventional h_2 takes the opposite sign of g U / psi
        h2.append(-gravity * rows[(2, 0)][0] / amplitude)
        k2.append(rows[(2, 0)][2] / amplitude)
    times = np.asarray(times)

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
    fig.savefig("app_response1d_tide.png", dpi=150)
    print("wrote app_response1d_tide.png")


if __name__ == "__main__":
    love_numbers()
    sea_level_fingerprint("sea_level_fingerprint",
                          "sl3d sea-level, the fingerprint: 10% of the "
                          "synthetic Greenland melted", synthetic=True)
    sea_level_fingerprint("sea_level_fingerprint_ice7g",
                          "sl3d sea-level, the fingerprint: 10% of the "
                          "ICE-7G Greenland ice melted", synthetic=False)
    sea_level()
    sea_level_calibrated("sea_level_1d_calibrated", "sea_level_1d", None,
                         "sl3d sea-level (1D), migrating shorelines, initial "
                         "sea level calibrated against the present day")
    sea_level_calibrated("sea_level_1d_ice7g_calibrated", "sea_level_1d_ice7g",
                         "ice7g.nc",
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
