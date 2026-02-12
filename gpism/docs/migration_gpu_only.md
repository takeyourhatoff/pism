# Migration Guide: GPU-Only gpism

This release removes CPU execution mode and legacy runtime fallbacks.

## Build changes

- Removed build options:
  - `GPISM_ENABLE_CUDA`
  - `GPISM_ENABLE_NETCDF`
- Required dependencies at configure time:
  - CUDA toolkit (`nvcc`)
  - NetCDF
- Kept optional build option:
  - `GPISM_ENABLE_MPI`

## Removed runtime config keys

The following keys are removed. Override files containing them now fail fast.

- `device.enabled`
- `device.halo_mode`
- `ssa.force_host_convergence`
- `ssa.max_speed`

## Retained runtime keys

- `device.enforce_hotloop_residency`
- `device.require_cuda_aware_mpi`
- `ssa.precond_precision`
- `ssa.gmres.residual_check_interval`
- `io.output.device_ring_depth`
- `io.output.async_stage_from_device`

## Script changes

- `gpism/scripts/compare_pism_gpism.sh`
  - Strict parity workflow only.
- `gpism/scripts/benchmark_pism_gpism.sh`
  - Throughput workflow (`MIN_WALL`) against original PISM.

## MPI behavior changes

- Halo exchange is device-buffer only.
- Multi-rank runs require CUDA-aware MPI support when
  `device.require_cuda_aware_mpi=1`.

## Public API cleanups

- Removed:
  - `set_deterministic_reductions(bool)`
  - `deterministic_reductions_enabled()`
- Removed files:
  - `include/gpism/device_policy.h`
  - `src/device_policy.cpp`
