#!/usr/bin/env bash
# Runs every example config through sl3d, from build/examples/configs (where the
# build copies this directory), and stops at the first failure. The
# synthetic Earth configs always run; the ICE-7G ones run only once
# ../data/ice7g.nc exists (python get_ice7g.py, from build/examples, writes
# it), and the toy planet's only once ../data/planet.gia.deck and
# ../data/planet.elastic.deck exist (python make_planet_deck.py writes
# them). CI calls this.
#
#   cd build/examples/configs && ./run_all.sh
#
# SL3D names the program (default ../../apps/sl3d); OMP_NUM_THREADS sets
# its threads. Each config's outputs land in out/<config>/.
set -euo pipefail
cd "$(dirname "$0")"
sl3d=${SL3D:-../../apps/sl3d}
if [[ ! -x $sl3d ]]; then
  echo "run_all.sh: no sl3d program at $sl3d (build it, or set SL3D)" >&2
  exit 2
fi

run() {  # run <command> <config>
  "$sl3d" "$1" "$2"
}

run love-numbers love_numbers.json
run response response.json
run response response_viscoelastic.json
run response response1d_disk.json
run response response1d_tide.json
run response response_tide_3d.json
run response response_tide_oscillating_short.json
run response response_tide_oscillating.json
run response response_tide_oscillating_long.json
run sea-level sea_level_fingerprint.json
run sea-level sea_level_1d.json
run sea-level sea_level_1d_calibrated.json
run sea-level sea_level_3d.json
run sea-level sea_level_3d_calibrated.json
run kernel1d kernel1d.json
run kernel kernel.json

if [[ -f ../data/ice7g.nc ]]; then
  run sea-level sea_level_fingerprint_ice7g.json
  run sea-level sea_level_1d_ice7g.json
  run sea-level sea_level_1d_ice7g_calibrated.json
else
  echo "run_all.sh: no ../data/ice7g.nc, so the ICE-7G configs are skipped" \
       "(python get_ice7g.py in build/examples writes it)"
fi

if [[ -f ../data/planet.gia.deck && -f ../data/planet.elastic.deck ]]; then
  run response response_tide_planet.json
  run response response_tide_planet_relax.json
else
  echo "run_all.sh: no ../data/planet.gia.deck, so the toy planet's configs" \
       "are skipped (python make_planet_deck.py in build/examples writes it)"
fi
