#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="${GPISM_BUILD_DIR:-$ROOT_DIR/gpism/build-cuda-mpi}"
BIN="$BUILD_DIR/gpism-timestep-gpu-smoke"

if [[ ! -x "$BIN" ]]; then
  echo "gpism-timestep-gpu-smoke not found at $BIN" >&2
  echo "Set GPISM_BUILD_DIR or build gpism tests first." >&2
  exit 1
fi

set -x
"$BIN"
