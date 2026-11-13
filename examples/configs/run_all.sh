#!/usr/bin/env bash
# Runs every example config through sl3d, from build/examples/configs (where the
# build copies this directory), and stops at the first failure. The
# synthetic Earth configs always run, the time-step study's short family
# among them (python timestep_study.py, from build/examples, runs the whole
# study); the ICE-7G ones run only once ../data/ice7g.nc exists (python
# get_ice7g.py, from build/examples, writes it), those on the custom PREM
# deck with S40RTS only once that deck and its lateral files exist too
# (make_earth_deck.py and make_lateral_model.py write them), the toy
# planet's only once ../data/planet.gia.deck and ../data/planet.elastic.deck
# exist (python make_planet_deck.py writes them), and the load given as a
# file and as a coefficient table only once make_load_field.py and
# make_load_coefficients.py have written them. CI calls this.
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

all_exist() {  # all_exist <file>...: whether every file is there
  local f
  for f in "$@"; do [[ -f $f ]] || return 1; done
}

run love-numbers love_numbers.json
run solve no_ocean.json
run solve no_ocean_viscoelastic.json
run solve no_ocean_seismic.json
run solve no_ocean_1d_disk.json
run solve no_ocean_1d_tide.json
run solve no_ocean_tide_3d.json
run solve no_ocean_tide_oscillating_short.json
run solve no_ocean_tide_oscillating.json
run solve no_ocean_tide_oscillating_long.json
run solve sea_level_fingerprint.json
run solve sea_level_fingerprint_3d.json
run solve sea_level_static_migrating.json
run solve sea_level_static_coseismic.json
run solve sea_level_postseismic.json
run solve postseismic_no_gravity.json
# the checkpoints of a former run_all.sh go first: the build may have
# written the run's inputs since, which makes them another run's
rm -rf out/sea_level_1d/checkpoints
run solve sea_level_1d.json
run solve sea_level_1d_calibrated.json
run solve sea_level_3d.json
run solve sea_level_3d_calibrated.json
run kernel kernel1d.json
run kernel kernel.json
run kernel kernel_static_moduli.json
# solve records the post-seismic run into its trajectory store, and kernel
# sweeps from the store without running it again (the store of a former
# run_all.sh, whose inputs the build may have written since, goes first)
rm -rf out/kernel_postseismic/trajectory
run solve kernel_postseismic.json
run kernel kernel_postseismic.json
run solve sea_level_fingerprint_ice_cap.json
run solve no_ocean_ice_cap.json
run solve no_ocean_cap_weak.json
run solve no_ocean_cap_strong.json
run solve no_ocean_load_cap.json
run solve sea_level_no_ocean.json
for n in 10 20 40; do  # the time-step study's short family
  run solve timestep_et_n$n.json
  run solve timestep_be_n$n.json
  run kernel timestep_discrete_n$n.json
  run kernel timestep_continuous_n$n.json
done
run solve timestep_et_n160.json
run kernel timestep_discrete_n160.json

if [[ -f ../data/ice7g.nc ]]; then
  run solve sea_level_fingerprint_ice7g.json
  run solve sea_level_1d_ice7g.json
  run solve sea_level_1d_ice7g_calibrated.json
else
  echo "run_all.sh: no ../data/ice7g.nc, so the ICE-7G configs are skipped" \
       "(python get_ice7g.py in build/examples writes it)"
fi

custom=(../data/ice7g.nc ../data/earth_custom.gia.deck
        ../data/earth_custom.elastic.deck ../data/earth_custom_moduli_s40rts.nc
        ../data/earth_custom_viscosity_s40rts.nc)
if all_exist "${custom[@]}"; then  # the 3D run through time takes minutes
  run solve sea_level_fingerprint_3d_ice7g.json
  run solve sea_level_1d_ice7g_custom.json
  run solve sea_level_3d_ice7g.json
else
  echo "run_all.sh: the custom PREM deck or its S40RTS files are missing, so" \
       "the 3D ICE-7G configs are skipped (make_earth_deck.py and" \
       "make_lateral_model.py in build/examples write them; README.md)"
fi

if [[ -f ../data/planet.gia.deck && -f ../data/planet.elastic.deck ]]; then
  run solve no_ocean_tide_planet.json
  run solve no_ocean_tide_planet_relax.json
else
  echo "run_all.sh: no ../data/planet.gia.deck, so the toy planet's configs" \
       "are skipped (python make_planet_deck.py in build/examples writes it)"
fi

if all_exist ../data/load_cap_history.nc ../data/load_cap_lm.txt; then
  run solve no_ocean_load_field.json
  run solve no_ocean_load_coefficients.json
else
  echo "run_all.sh: no ../data/load_cap_history.nc or ../data/load_cap_lm.txt," \
       "so the load from a file and from a table are skipped" \
       "(make_load_field.py and make_load_coefficients.py write them)"
fi
