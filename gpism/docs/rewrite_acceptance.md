# GPU-Only Rewrite Acceptance

This document defines merge gates for the GPU-only rewrite.

## Build and startup gates

- Configure must fail without CUDA toolkit.
- Configure must fail without NetCDF.
- Multi-rank startup must fail fast when device-buffer MPI exchange cannot be
  used and `device.require_cuda_aware_mpi=1`.

## Correctness gate

Run:

```bash
COMPARE_STRICT=1 COMPARE_TOL_RMS=0.02 COMPARE_TOL_MAX=0.05 \
YEARS=0.05 GPISM_DT=0.05 MIN_WALL=0 \
gpism/scripts/compare_pism_gpism.sh
```

Pass condition:

- `u_ssa` and `v_ssa` both satisfy:
  - relative RMS <= `0.02`
  - relative max <= `0.05`

## Device residency gate

- Timestep hot loop runs with no unexpected syncs when
  `device.enforce_hotloop_residency=1`.
- Output path may stage asynchronously from device but must not force blocking
  compute-stream synchronization.

## Throughput gate

Run:

```bash
MIN_WALL=120 gpism/scripts/benchmark_pism_gpism.sh
```

Pass condition:

- gpism throughput regression <= 5% versus locked baseline artifact on the same
  instance class.
