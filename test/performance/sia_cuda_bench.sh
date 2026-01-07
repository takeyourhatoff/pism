#!/bin/bash

set -euo pipefail

PISM_PATH="${1:-/home/ec2-user/pism-build}"
MPIEXEC="${2:-/usr/lib64/openmpi/bin/mpiexec}"
CONFIG_FILE="${3:-${PISM_PATH}/pism_config.nc}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-}"
NO_OUTPUT="${NO_OUTPUT:-1}"
FLOW_LAW="${FLOW_LAW:-}"
STEPS="${STEPS:-10}"
REPORT_ERRORS="${REPORT_ERRORS:-1}"
FULL_UPDATE="${FULL_UPDATE:-1}"

GRID_LIST="${GRID_LIST:-"101x101x21 201x201x41 401x401x61"}"
CUDA_BLOCKS="${CUDA_BLOCKS:-"16x16 32x8 8x32"}"
REPS="${REPS:-1}"
RUN_STRICT="${RUN_STRICT:-0}"

GPU_AWARE_MPI="${GPU_AWARE_MPI:-1}"
GPU_MPI_OPTS="-use_gpu_aware_mpi ${GPU_AWARE_MPI}"

if [[ ! -x "${PISM_PATH}/pism_siafd_test" ]]; then
  echo "Missing ${PISM_PATH}/pism_siafd_test"
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file ${CONFIG_FILE}"
  exit 1
fi

if [[ ! -e /dev/nvidia0 && ! -d /proc/driver/nvidia/gpus ]]; then
  echo "No CUDA device detected; GPU runs will be skipped."
  HAVE_GPU=0
else
  HAVE_GPU=1
fi

tmpdir=$(mktemp -d --tmpdir pism-bench-XXXX)
trap 'rm -rf "$tmpdir"' EXIT
cd "$tmpdir"

run_case() {
  local label="$1"
  local extra_opts="$2"
  local out="$3"
  local grid="$4"

  set +e
  /usr/bin/time -f "${label} elapsed_sec=%e" \
    ${MPIEXEC} -n 1 \
    "${PISM_PATH}/pism_siafd_test" \
    -config "${CONFIG_FILE}" \
    ${OUTPUT_FORMAT:+-o_format "${OUTPUT_FORMAT}"} \
    ${FLOW_LAW:+-stress_balance.sia.flow_law "${FLOW_LAW}"} \
    ${grid} \
    ${GPU_MPI_OPTS} \
    -steps "${STEPS}" \
    -o "${out}" \
    $( [[ "${NO_OUTPUT}" -eq 1 ]] && echo "-no_output" ) \
    $( [[ "${REPORT_ERRORS}" -eq 0 ]] && echo "-no_report" ) \
    $( [[ "${FULL_UPDATE}" -eq 0 ]] && echo "-no_full_update" ) \
    ${extra_opts} > /dev/null
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "${label} FAILED exit=${status}"
    if [[ "${RUN_STRICT}" -eq 1 ]]; then
      exit "${status}"
    fi
  fi
}

for grid in ${GRID_LIST}; do
  IFS=x read -r Mx My Mz <<< "${grid}"
  GRID_OPTS="-Mx ${Mx} -My ${My} -Mz ${Mz}"

  if [[ "${SKIP_CPU:-0}" -ne 1 ]]; then
    for rep in $(seq 1 "${REPS}"); do
      run_case "cpu grid=${grid} rep=${rep}" "" "cpu_${grid}_${rep}.nc" "${GRID_OPTS}"
    done
  fi

  if [[ "${HAVE_GPU}" -eq 1 ]]; then
    for block in ${CUDA_BLOCKS}; do
      IFS=x read -r bx by <<< "${block}"
      export PISM_SIAFD_CUDA_BLOCK_X="${bx}"
      export PISM_SIAFD_CUDA_BLOCK_Y="${by}"

      for rep in $(seq 1 "${REPS}"); do
        run_case "gpu grid=${grid} block=${block} rep=${rep}" \
          "-vec_type cuda -dm_vec_type cuda" \
          "gpu_${grid}_${block}_${rep}.nc" \
          "${GRID_OPTS}"
      done
    done
  fi
done
