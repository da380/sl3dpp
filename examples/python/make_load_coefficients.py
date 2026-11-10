"""Writes a load's coefficient table for a `coefficients` load term of the
sl3d program's `response` command, from a surface mass density held as a
field of a planetmodel surface file.

The field (`--name`, default `load`, in kg m^-2) is read on its own grid
(any grid of a kind planetmodel's harmonics transform: Gauss-Legendre,
Driscoll-Healy, equiangular) and analysed to degree --lmax, which must not
exceed the grid's band or the run's lmax. Its real orthonormal
coefficients c (planetmodel.harmonics: pyshtools' "ortho" normalisation,
no Condon-Shortley phase, c[0] the cosine and c[1] the sine harmonics) are
turned into the coefficients of the table's harmonics, the exchange
table's complex orthonormal ones with the Condon-Shortley phase
(docs/sl3d-configs.md, response, The forcing block):

    f_l0 = c[0, l, 0],    f_lm = (-1)^m (c[0, l, m] - i c[1, l, m]) / sqrt 2,

one row `l m re im` per degree and order m >= 0, under `#` lines saying
what the table holds and where it came from. A field on a time axis is
read at one of its stored times, --index (from 0).

This is a LOAD's table. An exchange table that sl3d writes holds the
response, the coefficients of the surface displacement U and potential
Phi, in the same harmonics; those are not a load, and feeding them back as
one means nothing. A load field written by an sl3d run (the `load` field
of a `response` surface file, at a node) or by make_load_field.py is.

Run from build/examples, e.g. after make_load_field.py --shape cap
--lmax 32 has written data/load_cap.nc on the run's own grid:

    poetry run python make_load_coefficients.py data/load_cap.nc
        [--name load] [--index N] [--lmax 32] [--out data/load_cap_lm.txt]

It writes data/<stem>_lm.txt by default. A `response` config's forcing
takes it as {"coefficients": {"file": "../data/load_cap_lm.txt"}}; the
table of a field on the run's own Gauss-Legendre grid, analysed to the
run's lmax, is the run's own expansion of that field, so the
`coefficients` term reproduces the `field` term (and the `cap` term the
field samples) to rounding: response_load_coefficients.json against
response_load_field.json and response_load_cap.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from planetmodel.harmonics import analyse_grid
from planetmodel.surfacefile import read_surface_file

from inputs import DATA, wrote


def table_coefficients(values, grid, lmax: int) -> np.ndarray:
    """The complex coefficients f[l, m] (m >= 0) of a field on a grid, in
    the exchange table's harmonics, to degree lmax."""
    c = analyse_grid(values, grid, lmax=lmax)
    m = np.arange(lmax + 1)[None, :]
    f = (-1.0) ** m * (c[0] - 1j * c[1]) / np.sqrt(2.0)
    f[:, 0] = c[0, :, 0]
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path,
                        help="a planetmodel surface file holding the load")
    parser.add_argument("--name", default="load",
                        help="the field, kg m^-2 (default load)")
    parser.add_argument("--index", type=int, default=None,
                        help="the stored time to read, for a field on a "
                             "time axis")
    parser.add_argument("--lmax", type=int, default=None,
                        help="the largest degree (default the grid's band)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    file = read_surface_file(args.file)
    if args.name not in file.variables:
        raise SystemExit(f"{args.file} holds no field {args.name!r} (it holds "
                         f"{', '.join(sorted(file.variables))})")
    values = np.asarray(file.variables[args.name].values, dtype=float)
    where = ""
    if values.ndim == 3:
        if args.index is None:
            raise SystemExit(f"{args.name} is on a time axis of {len(values)} "
                             "times: --index chooses one")
        values = values[args.index]
        where = f" at stored time {args.index}"
    elif args.index is not None:
        raise SystemExit(f"{args.name} is static: --index is for a field on "
                         "a time axis")
    grid = file.grid
    lmax = grid.lmax if args.lmax is None else args.lmax
    if not 0 <= lmax <= grid.lmax:
        raise SystemExit(f"--lmax must lie in 0..{grid.lmax}, the grid's band")
    f = table_coefficients(values, grid, lmax)

    out = args.out or DATA / f"{args.file.stem}_lm.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# a load's coefficients, kg m^-2: l m re im, in the exchange "
        "table's harmonics",
        "# (complex orthonormal, with the Condon-Shortley phase), m >= 0",
        f"# from the field {args.name} of {args.file}{where}, degrees 0 to "
        f"{lmax}",
        f"# written by {Path(__file__).name}",
    ]
    for l in range(lmax + 1):
        for m in range(l + 1):
            lines.append(f"{l} {m} {f[l, m].real:.17g} "
                         f"{0.0 if m == 0 else f[l, m].imag:.17g}")
    out.write_text("\n".join(lines) + "\n")
    wrote(out, f"degrees 0 to {lmax}, {(lmax + 1) * (lmax + 2) // 2} rows")


if __name__ == "__main__":
    main()
