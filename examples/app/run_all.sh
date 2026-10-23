#!/usr/bin/env bash
# Runs every example config through sl3d, from build/examples/app (where the
# build copies this directory), and stops at the first failure. The
# synthetic Earth configs always run; the ICE-7G ones run only once
# ../ice7g.nc exists (python get_ice7g.py, from build/examples, writes it).
# CI calls this.
#
#   cd build/examples/app && ./run_all.sh
#
# SL3D names the program (default ../../apps/sl3d); OMP_NUM_THREADS sets
# its threads. Outputs land in out/.
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
run fingerprint fingerprint.json
run evolve1d evolve1d.json
run evolve evolve.json
run kernel1d kernel1d.json
run kernel kernel.json

if [[ -f ../ice7g.nc ]]; then
  run fingerprint fingerprint_ice7g.json
  run evolve1d evolve1d_ice7g.json
else
  echo "run_all.sh: no ../ice7g.nc, so the ICE-7G configs are skipped" \
       "(python get_ice7g.py in build/examples writes it)"
fi
