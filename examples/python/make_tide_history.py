"""Writes an sl3d `response` config of an oscillating degree-2 tide on a
radial viscoelastic Earth: the tidal potential's (2, 0) coefficient times
sin(2 pi t / P), from t = 0 over a whole number of periods P.

The run's time nodes are evenly spaced, --points to a period, and the
amplitude history (`history_kyr`) has a point at every node: the forcing
enters a viscoelastic run at its nodes alone (docs/sl3d-configs.md, The
viscoelastic run), so these are the sinusoid's values wherever the run
sees it. The tide starts from zero, so there is no elastic jump at the
start; the transient that the start excites dies away over the first
cycles, and the last cycles show the periodic response, its amplitude and
its lag behind the potential: the Love numbers at the forcing's frequency.
The potential's amplitude, 1 m^2 s^-2 by default, makes the response's
coefficients those Love numbers in SI (as in response1d_tide.json).

The configs response_tide_oscillating_short.json,
response_tide_oscillating.json and response_tide_oscillating_long.json
are this script's output at periods of 0.1, 1 and 10 kyr on
data/prem8.gia.deck, whose Maxwell mantle (1e21 Pa s) has Maxwell times
of 0.11 to 0.43 kyr. Run from build/examples:

    poetry run python make_tide_history.py [--period-kyr 1] [--cycles 6]
        [--points 24] [--deck ../data/prem8.gia.deck] [--lmax 4]
        [--name response_tide_oscillating] [--out FILE | --out -]

It writes configs/<name>.json (`--out -` prints it instead). Run from
configs/, the config writes out/<name>/<name>.nc, the surface response at
every node, and one exchange table, out/<name>/<name>_<node>.txt, at the
last cycle's crest of the potential (its header carries the surface
gravity, which turns the displacement into h2):

    cd configs && ../../apps/sl3d response <name>.json

The three example configs are, from build/examples (each overwriting the
copy in configs/):

    poetry run python make_tide_history.py --period-kyr 0.1 \
        --name response_tide_oscillating_short
    poetry run python make_tide_history.py --period-kyr 1
    poetry run python make_tide_history.py --period-kyr 10 \
        --name response_tide_oscillating_long
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def config(name: str, period: float, cycles: int, points: int, deck: str,
           lmax: int, amplitude: float) -> str:
    """The config as JSON text, in the layout of the example configs."""
    nodes = cycles * points + 1
    crest = (cycles - 1) * points + points // 4  # the last crest's node
    pairs = []
    for k in range(nodes):
        factor = round(math.sin(2.0 * math.pi * k / points), 6) + 0.0
        pairs.append(f"[{float(round(period * k / points, 9))!r}, "
                     f"{factor!r}]")
    rows = [", ".join(pairs[i:i + 4]) for i in range(0, nodes, 4)]
    history = (",\n" + " " * 30).join(rows)
    return f"""{{
  "model": {{"deck": {json.dumps(deck)}, "format": "maxwell"}},
  "lmax": {lmax},
  "times": {{"start_kyr": 0.0, "end_kyr": {float(cycles * period)!r}, "nodes": {nodes}}},
  "forcing": {{
    "tide": [{{"degree": 2, "order": 0, "amplitude_m2_s2": {float(amplitude)!r},
              "history_kyr": [{history}]}}]
  }},
  "output": {{"surface": "out/{name}/{name}.nc",
             "exchange": "out/{name}/{name}_{{node}}.txt",
             "exchange_nodes": [{crest}]}}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--period-kyr", type=float, default=1.0,
                        help="the tide's period (default 1 kyr)")
    parser.add_argument("--cycles", type=int, default=6,
                        help="whole periods run (default 6)")
    parser.add_argument("--points", type=int, default=24,
                        help="time nodes per period (default 24)")
    parser.add_argument("--deck", default="../data/prem8.gia.deck",
                        help="the Maxwell deck, as the config names it "
                             "(default ../data/prem8.gia.deck)")
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="the potential's amplitude, m^2 s^-2 (default 1)")
    parser.add_argument("--name", default="response_tide_oscillating",
                        help="the config's name (default "
                             "response_tide_oscillating)")
    parser.add_argument("--out", default=None,
                        help="the file to write (default configs/<name>.json; "
                             "- prints it)")
    args = parser.parse_args()
    if args.period_kyr <= 0.0 or args.cycles < 1 or args.points < 4:
        raise SystemExit("needs a positive period, a cycle or more and at "
                         "least four points to a period")
    text = config(args.name, args.period_kyr, args.cycles, args.points,
                  args.deck, args.lmax, args.amplitude)
    if args.out == "-":
        print(text, end="")
        return
    out = Path(args.out or Path("configs") / f"{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} (period {args.period_kyr:g} kyr, {args.cycles} "
          f"cycles, {args.cycles * args.points + 1} nodes)")


if __name__ == "__main__":
    main()
