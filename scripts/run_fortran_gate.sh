#!/usr/bin/env bash
# The cross-code benchmark against the Fortran sl3d
# (docs/dev/lateral-coupling.md §5b): builds the original Fortran code and this
# repo side by side, runs visco_test at a chosen lmax, replays the same
# setup through sl3dpp, and prints the coefficient-space comparison.
# Intended for a server; lmax 32 takes minutes, lmax 128 (the code's
# original scale) considerably longer on the Fortran side.
#
# Usage:
#   SL3D_DIR=/path/to/sl3d ./scripts/run_fortran_gate.sh [lmax] [threads]
#
# Requirements: the sl3d checkout must carry the benchmark-output driver
# patch to visco_test (an lmax argument and `visccoefs.out`); gfortran,
# a C++23 compiler, CMake, FFTW3, LAPACK/BLAS, netCDF-C.

set -euo pipefail

LMAX="${1:-32}"
THREADS="${2:-$(nproc)}"
# Huge pages for the large 3D tensors where they are on request only (the
# top-level README, Running); harmless elsewhere. An existing setting wins.
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.malloc.hugetlb=1}"
if [ -z "${SL3D_DIR:-}" ]; then
  echo "set SL3D_DIR to a checkout of da380/sl3d (the Fortran code)" >&2
  exit 2
fi
SL3DPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$SL3DPP_DIR/benchmarks/runs/fortran_gate_lmax${LMAX}}"

echo "== sl3d:   $SL3D_DIR"
echo "== sl3dpp: $SL3DPP_DIR"
echo "== lmax $LMAX, $THREADS threads, workdir $WORK"
echo "== GLIBC_TUNABLES=$GLIBC_TUNABLES"
mkdir -p "$WORK"

# --- the Fortran side -------------------------------------------------------
cmake -S "$SL3D_DIR" -B "$SL3D_DIR/build-gate" -DCMAKE_BUILD_TYPE=Release \
      > "$WORK/sl3d_configure.log" 2>&1
cmake --build "$SL3D_DIR/build-gate" -j "$THREADS" --target sl3d_visco_test \
      > "$WORK/sl3d_build.log" 2>&1
VISCO_TEST="$(find "$SL3D_DIR/build-gate" -name visco_test -type f | head -1)"
DECK="$SL3D_DIR/data/prem.simp.vm1"
[ -x "$VISCO_TEST" ] || { echo "visco_test not built"; exit 1; }

FORTRAN_OUT="$WORK/visccoefs_lmax${LMAX}.txt"
if [ ! -s "$FORTRAN_OUT" ]; then
  echo "== running Fortran visco_test (lmax $LMAX)..."
  ( cd "$WORK" && OMP_NUM_THREADS="$THREADS" \
      "$VISCO_TEST" "$DECK" "$LMAX" > fortran_run.log 2>&1 )
  mv "$WORK/visccoefs.out" "$FORTRAN_OUT"
else
  echo "== reusing $FORTRAN_OUT"
fi

# --- the sl3dpp side --------------------------------------------------------
cmake -S "$SL3DPP_DIR" -B "$SL3DPP_DIR/build-gate" \
      -DCMAKE_BUILD_TYPE=Release -DSL3DPP_BUILD_BENCHMARKS=ON \
      -DSL3DPP_BUILD_TESTS=OFF > "$WORK/sl3dpp_configure.log" 2>&1
cmake --build "$SL3DPP_DIR/build-gate" -j "$THREADS" --target fortran_gate \
      > "$WORK/sl3dpp_build.log" 2>&1

echo "== running sl3dpp fortran_gate (lmax $LMAX)..."
OMP_NUM_THREADS="$THREADS" \
  "$SL3DPP_DIR/build-gate/benchmarks/fortran_gate" \
  "$DECK" "$FORTRAN_OUT" "$LMAX" "$THREADS" | tee "$WORK/comparison.txt"

echo "== done; results in $WORK"
