#!/bin/bash

set -e
set -x

PISM_PATH=$1
MPIEXEC=$2
PISM_SOURCE_DIR=$3

temp_dir=$(mktemp -d --tmpdir pism-test-XXXX)
trap 'rm -rf "$temp_dir"' EXIT
cd $temp_dir

if [[ ! -e /dev/nvidia0 && ! -d /proc/driver/nvidia/gpus ]]; then
  echo "No CUDA device detected; skipping."
  exit 0
fi

GRID="-Mx 11 -My 11 -Mz 9"
GPU_MPI_OPTS="-use_gpu_aware_mpi 0"

$MPIEXEC -n 1 $PISM_PATH/pism_siafd_test $GRID $GPU_MPI_OPTS -o cpu.nc
$MPIEXEC -n 1 $PISM_PATH/pism_siafd_test $GRID $GPU_MPI_OPTS -o cuda.nc -dm_vec_type cuda

$PISM_PATH/pism_nccmp -t 5e-3 -x -v time,timestamp cpu.nc cuda.nc
