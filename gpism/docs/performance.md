# Performance Notes (GPU-first)

## Environment

- Host: single NVIDIA A10G (CUDA 13.1, nvcc `/usr/local/cuda-13.1/bin/nvcc`)
- Build: `GPISM_ENABLE_CUDA=ON`, `GPISM_ENABLE_MPI=ON`, `GPISM_ENABLE_NETCDF=ON`
- Input: `/home/ec2-user/pism/examples/std-greenland/pism_Greenland_5km_v1.1.nc`
- Run length: `-y 0.05` (one output)
- Config override (excerpt):

```
ssa.mg.enabled=1
ssa.mg.smoother=jacobi
ssa.mg.pre_iters=3
ssa.mg.post_iters=3
ssa.mg.coarse_iters=10
ssa.gmres_max_iter=60
ssa.max_picard=1
time.dt=0.05
time.output_interval=0.05
io.async_output=1
```

## Fused SSA apply+residual (MG `compute_residual`)

Goal: reduce memory traffic by fusing the SSA apply and residual computation.
Result: **reverted** (no clear win).

### Kernel mix (Nsight Systems `cuda_gpu_kern_sum`)

- **Before (a58e129a6)**:
  - `apply_region_kernel`: 48.0%
  - `residual_kernel`: 0.8%
- **After (current)**:
  - `apply_region_kernel`: 42.1%
  - `apply_residual_region_kernel`: 6.5%

### Wall time (single rank, same config)

- **Before:** 0.98 s
- **After:** 1.00 s

Result: no clear win; within noise (slightly slower). Keep monitoring before
expanding the fusion to other hot paths.

### Memcopy counts (Nsight Systems `cuda_gpu_mem_time_sum`)

- **Before:** D2H 556, H2D 551
- **After:**  D2H 558, H2D 551

No material change in transfer counts.

## Async/double-buffered output

Single-rank runs now stage output into a reusable host-side double buffer and
write asynchronously. Multi-rank runs remain synchronous (NetCDF parallel).

## Scaling (single GPU proxy)

**Strong scaling (same full dataset, single GPU):**

- 1 MPI rank: 1.00 s
- 2 MPI ranks (same GPU, `GPISM_CUDA_AWARE_MPI=0`): 3.60 s

**Weak scaling proxy:**

- 1 rank on synthetic half-size input (150×561): 0.59 s
- 2 ranks on full input (301×561, same GPU): 3.60 s

### Notes / limitations

- This host has **one GPU**, so multi-GPU scaling is not yet measured.
- Two MPI ranks share a single GPU, which is expected to be slower; treat these
  numbers as **MPI overhead indicators**, not true multi-GPU scaling.
- `GPISM_CUDA_AWARE_MPI=0` was required to avoid hangs in this environment.
