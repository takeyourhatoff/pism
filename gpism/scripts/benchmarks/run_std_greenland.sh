#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="${GPISM_BUILD_DIR:-$ROOT_DIR/gpism/build-cuda-mpi}"
BIN="$BUILD_DIR/gpism"
INPUT_DEFAULT="$ROOT_DIR/examples/std-greenland/pism_Greenland_5km_v1.1.nc"
INPUT="${PISM_STD_GREENLAND_INPUT:-$INPUT_DEFAULT}"
OVERRIDE_DEFAULT="$ROOT_DIR/gpism/scripts/benchmarks/std_greenland_override.cfg"
OVERRIDE="${GPISM_STD_GREENLAND_OVERRIDE:-$OVERRIDE_DEFAULT}"
OUT="${GPISM_STD_GREENLAND_OUT:-/tmp/gpism_std_greenland_out.nc}"
YEARS="${GPISM_STD_GREENLAND_YEARS:-0.05}"

if [[ ! -x "$BIN" ]]; then
  echo "gpism binary not found at $BIN" >&2
  echo "Set GPISM_BUILD_DIR or build gpism first." >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "std-greenland input not found at $INPUT" >&2
  echo "Run examples/std-greenland/preprocess.sh or set PISM_STD_GREENLAND_INPUT." >&2
  exit 1
fi

if [[ ! -f "$OVERRIDE" ]]; then
  echo "override config not found at $OVERRIDE" >&2
  exit 1
fi

MPI_CMD=()
if [[ -n "${GPISM_NP:-}" ]]; then
  MPI_CMD=(mpirun -np "$GPISM_NP")
fi

set -x
"${MPI_CMD[@]}" "$BIN" \
  -i "$INPUT" \
  -o "$OUT" \
  -y "$YEARS" \
  -config_override "$OVERRIDE"
