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

## Repeatable benchmarks

Scripts live in `gpism/scripts/benchmarks/`:

- `run_std_greenland.sh` (std-greenland, fixed override config)
- `run_gpu_smoke.sh` (device-resident hot loop smoke runtime)

See `gpism/scripts/benchmarks/README.md` for required inputs and environment
variables.

## Nsight Systems profile (std-greenland, single step)

Command (single rank):

```
nsys profile --stats=true -t cuda,osrt,nvtx -o /tmp/gpism_stdgreenland_nsys \
  gpism/build-cuda-mpi/gpism \
  -i /home/ec2-user/pism/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
  -o /tmp/gpism_stdgreenland_out_nsys.nc \
  -y 0.05 \
  -config_override gpism/scripts/benchmarks/std_greenland_override.cfg
```

Wall time (same args, no profiler): **1.00 s** (`/usr/bin/time -p`).

### CUDA summary highlights (`cuda_api_gpu_sum`)

- CUDA API time dominated by `cudaHostAlloc`/`cudaFreeHost` (staging buffers).
- GPU kernel time is small in this short run (total kernel time ~6.6 ms).

Top kernels (`cuda_gpu_kern_sum`):

- `apply_region_kernel`: 43.7%
- `jacobi_update_kernel`: 14.7%
- `dot_stag_kernel`: 12.5%
- `dot_stag_batch_kernel`: 8.4%
- `diff_norm1_stag_kernel`: 4.2%

This run is **not yet kernel-dominated** at the wall-clock level; longer runs
or reduced host allocations are required to satisfy the M10 DoD.

## Nsight Systems profile (std-greenland, longer run)

Command (single rank, reduced output frequency):

```
nsys profile --force-overwrite true --stats=true -t cuda,osrt,nvtx \
  -o /tmp/gpism_stdgreenland_long_nsys \
  gpism/build-cuda-mpi/gpism \
  -i /home/ec2-user/pism/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
  -o /tmp/gpism_stdgreenland_long_out_nsys.nc \
  -y 0.5 \
  -config_override /tmp/gpism_std_greenland_long.cfg
```

Wall time (same args, no profiler): **1.57 s** (`/usr/bin/time -p`).

### CUDA summary highlights (`cuda_api_sum`)

- CUDA API time dominated by `cudaMemcpy` (~375 ms total), then
  `cudaHostAlloc` (~258 ms) and `cudaFreeHost` (~128 ms).
- Total kernel time ~**151 ms** (from `cuda_gpu_kern_sum`).

Top kernels (`cuda_gpu_kern_sum`):

- `apply_region_kernel`: 43.8%
- `jacobi_update_kernel`: 14.7%
- `dot_stag_kernel`: 12.5%
- `dot_stag_batch_kernel`: 8.4%
- `diff_norm1_stag_kernel`: 4.3%

Even with reduced output, this run is **still not kernel-dominated**; host
allocations and memcopies remain the largest contributors.

## Nsight Systems profile (std-greenland, 4-year run, single rank)

Config: `-y 4.0` with output interval `4.0` (`/tmp/gpism_std_greenland_4yr.cfg`).

Wall time (no profiler): **10.11 s** (`/usr/bin/time -p`), improved from
**26.39 s** prior to skipping halo exchange on single-rank MPI runs.

### CUDA summary highlights (`cuda_api_gpu_sum`)

- `cudaMemcpy`: **6.49 s** (5469 calls)
- `dot_stag_batch_kernel`: **5.60 s** (1819 calls)
- `apply_kernel`: **0.92 s** (93780 calls)
- `jacobi_update_kernel`: **0.75 s** (79192 calls)
- `dot_stag_kernel`: **0.66 s** (2084 calls)

Memcopy volume dropped dramatically (from ~47 GB total to ~17 MB total):

- D2H: **8.4 MB** over 4229 calls
- H2D: **8.6 MB** over 3644 calls

Kernels now account for a large share of runtime, but `cudaMemcpy` still shows
up prominently in the CUDA API breakdown (likely sync points around scalar
reductions and GMRES orthogonalization).

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

## Fused thickness update (flux + update)

Goal: reduce temporaries by computing fluxes on the fly during the thickness
update. Result: **reverted** (no measurable improvement; within noise).

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
- CUDA-aware MPI is auto-detected when MPI provides `MPIX_Query_cuda_support`.
  Set `GPISM_CUDA_AWARE_MPI=1` to force enable, or `GPISM_CUDA_AWARE_MPI=0` to
  force host staging. `GPISM_CUDA_AWARE_MPI=0` was required to avoid hangs in
  this environment.
