# Performance Protocol

## Goal

Measure gpism GPU throughput against the original PISM binary while preserving
strict SSA parity.

## Locked benchmark protocol

- Baseline comparator: original `pism` binary.
- Same input, physics family, and year horizon for both runs.
- Wall-clock target: `MIN_WALL=120` (auto-scales years until threshold).
- Benchmark script:
  - `gpism/scripts/benchmark_pism_gpism.sh`
  - For stable long-wall comparisons, cap PISM timestep:
    - `PISM_MAX_DT=${GPISM_DT}`
- Parity script (separate from benchmark):
  - `gpism/scripts/compare_pism_gpism.sh`

## Baseline artifacts (locked references)

- Strict parity artifact:
  - `/home/ec2-user/work/runs/strict_short_20260211_222659`
- Throughput artifact:
  - `/home/ec2-user/work/runs/bench_nowallhours_20260211_222851`

## Acceptance thresholds

- Correctness:
  - `COMPARE_STRICT=1`
  - `COMPARE_TOL_RMS=0.02`
  - `COMPARE_TOL_MAX=0.05`
  - Pass required for `u_ssa` and `v_ssa`.
- Throughput:
  - Non-regression target: no worse than 5% versus the locked gpism baseline on
    the same instance class and toolchain.

## Example commands

Parity:

```bash
COMPARE_STRICT=1 COMPARE_TOL_RMS=0.02 COMPARE_TOL_MAX=0.05 \
YEARS=0.05 GPISM_DT=0.05 \
gpism/scripts/compare_pism_gpism.sh
```

Benchmark:

```bash
MIN_WALL=120 YEARS=0.05 GPISM_DT=0.05 \
gpism/scripts/benchmark_pism_gpism.sh
```

## Phase 1 run log (2026-02-12)

Hardware/profile target:
- AWS instance `i-056a5fcf119ae85ad` (Tesla T4).
- Build dir: `/home/ec2-user/work/build/gpism-codex-opt`
- Source dir: `/home/ec2-user/work/src/pism-gpism/gpism`

### Commands used

Build:

```bash
cmake --build /home/ec2-user/work/build/gpism-codex-opt -j
```

Targeted tests:

```bash
ctest --test-dir /home/ec2-user/work/build/gpism-codex-opt --output-on-failure \
  -R "gmres|multigrid|ssa-operator|halo-exchange|timestep-device-residency|timestep-io"
```

Strict parity gate:

```bash
GPISM_BIN=/home/ec2-user/work/build/gpism-codex-opt/gpism \
PISM_BIN=/home/ec2-user/pism-build-cudaaware-baseline/pism \
PISM_CONFIG=/home/ec2-user/pism-build-cudaaware-baseline/pism_config.nc \
INPUT=/home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
COMPARE_STRICT=1 COMPARE_TOL_RMS=0.02 COMPARE_TOL_MAX=0.05 \
YEARS=0.05 GPISM_DT=0.05 MIN_WALL=0 \
/home/ec2-user/work/src/pism-gpism/gpism/scripts/compare_pism_gpism.sh
```

Short benchmark gate:

```bash
GPISM_BIN=/home/ec2-user/work/build/gpism-codex-opt/gpism \
PISM_BIN=/home/ec2-user/pism-build-cudaaware-baseline/pism \
PISM_CONFIG=/home/ec2-user/pism-build-cudaaware-baseline/pism_config.nc \
INPUT=/home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
MIN_WALL=20 YEARS=0.05 GPISM_DT=0.05 \
/home/ec2-user/work/src/pism-gpism/gpism/scripts/benchmark_pism_gpism.sh
```

Nsight Systems (short pass):

```bash
nsys profile --force-overwrite true --sample=none --trace=cuda \
  -o /home/ec2-user/work/runs/nsys_phase1b_20260212_114333/gpism_nsys \
  /home/ec2-user/work/build/gpism-codex-opt/gpism \
  -i /home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
  -o /home/ec2-user/work/runs/nsys_phase1b_20260212_114333/gpism_out.nc \
  -y 0.4
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  /home/ec2-user/work/runs/nsys_phase1b_20260212_114333/gpism_nsys.nsys-rep \
  > /home/ec2-user/work/runs/nsys_phase1b_20260212_114333/nsys_stats.txt
```

### Artifacts

- Strict parity: `/tmp/gpism_pism_compare_20260212_114059`
- Short benchmark: `/tmp/gpism_pism_bench_20260212_114151`
- Nsight short profile:
  - Baseline: `/home/ec2-user/work/runs/nsys_codex_now_20260212_112109/nsys_stats.txt`
  - Phase 1: `/home/ec2-user/work/runs/nsys_phase1b_20260212_114333/nsys_stats.txt`

### Measured results

- Targeted ctest subset: pass (13/13).
- Strict parity: pass (`u_ssa`/`v_ssa` within strict tolerances).
- Short benchmark (`MIN_WALL=20`):
  - `gpism`: `0.59869` years/sec
  - `pism`: `0.00112057` years/sec
  - speedup (`gpism/pism`): `534.273x`

- Corrected long benchmark (`MIN_WALL=120`, `PISM_MAX_DT=0.05`):
  - artifact: `/tmp/gpism_pism_bench_20260212_120941`
  - `gpism`: `0.584141` years/sec
  - `pism`: `0.0411629` years/sec
  - speedup (`gpism/pism`): `14.191x`

Notes:
- A prior long benchmark without `PISM_MAX_DT` let PISM take very large timesteps
  at long horizons, producing non-comparable throughput numbers.

### Profile delta (baseline -> phase 1)

- CUDA API:
  - `cudaStreamSynchronize`: dominant at `68.0%` in baseline, removed from top calls in phase 1 path.
  - `cudaMemcpy`: `12.5%` (`3278` calls) -> `79.3%` (`4911` calls), now dominant due batched blocking scalar reads.
  - `cudaLaunchKernel` calls: `413,262` -> `365,534` (~`11.5%` lower).
- Kernel summary:
  - `jacobi_fused_kernel` instances: `170,088` -> `149,720` (~`12.0%` lower).

## Phase 1C micro-optimization log (2026-02-12)

Change:
- GMRES GPU orthogonalization now uses the fused norm term emitted by
  `orthogonalize_stag_cuda`; removed the extra `dot(z, z)` launch in the Arnoldi
  inner loop.

Commands used (AWS `i-056a5fcf119ae85ad`, Tesla T4):

```bash
cmake --build /home/ec2-user/work/build/gpism-codex-opt -j

ctest --test-dir /home/ec2-user/work/build/gpism-codex-opt --output-on-failure \
  -R "gmres|multigrid|ssa-operator|halo-exchange|timestep-device-residency|timestep-io"

GPISM_BIN=/home/ec2-user/work/build/gpism-codex-opt/gpism \
PISM_BIN=/home/ec2-user/pism-build-cudaaware-baseline/pism \
PISM_CONFIG=/home/ec2-user/pism-build-cudaaware-baseline/pism_config.nc \
INPUT=/home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
COMPARE_STRICT=1 COMPARE_TOL_RMS=0.02 COMPARE_TOL_MAX=0.05 \
YEARS=0.05 GPISM_DT=0.05 MIN_WALL=0 \
/home/ec2-user/work/src/pism-gpism/gpism/scripts/compare_pism_gpism.sh

GPISM_BIN=/home/ec2-user/work/build/gpism-codex-opt/gpism \
PISM_BIN=/home/ec2-user/pism-build-cudaaware-baseline/pism \
PISM_CONFIG=/home/ec2-user/pism-build-cudaaware-baseline/pism_config.nc \
INPUT=/home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
MIN_WALL=20 YEARS=0.05 GPISM_DT=0.05 PISM_MAX_DT=0.05 \
/home/ec2-user/work/src/pism-gpism/gpism/scripts/benchmark_pism_gpism.sh

GPISM_BIN=/home/ec2-user/work/build/gpism-codex-opt/gpism \
PISM_BIN=/home/ec2-user/pism-build-cudaaware-baseline/pism \
PISM_CONFIG=/home/ec2-user/pism-build-cudaaware-baseline/pism_config.nc \
INPUT=/home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
MIN_WALL=120 YEARS=0.05 GPISM_DT=0.05 PISM_MAX_DT=0.05 \
/home/ec2-user/work/src/pism-gpism/gpism/scripts/benchmark_pism_gpism.sh

nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt \
  -o /home/ec2-user/work/runs/nsys_phase1d_y04_default_20260212_130057/gpism_nsys \
  /home/ec2-user/work/build/gpism-codex-opt/gpism \
  -i /home/ec2-user/pism-gpu-5/examples/std-greenland/pism_Greenland_5km_v1.1.nc \
  -o /home/ec2-user/work/runs/nsys_phase1d_y04_default_20260212_130057/gpism_out.nc \
  -y 0.4
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  /home/ec2-user/work/runs/nsys_phase1d_y04_default_20260212_130057/gpism_nsys.nsys-rep \
  > /home/ec2-user/work/runs/nsys_phase1d_y04_default_20260212_130057/nsys_stats.txt
```

Artifacts:
- Strict parity: `/tmp/gpism_pism_compare_20260212_124012`
- Short benchmark: `/tmp/gpism_pism_bench_20260212_124105`
- Long benchmark: `/tmp/gpism_pism_bench_20260212_124331`
- Nsight (default `-y 0.4`):
  - previous reference: `/home/ec2-user/work/runs/nsys_phase1b_20260212_114333/nsys_stats.txt`
  - current: `/home/ec2-user/work/runs/nsys_phase1d_y04_default_20260212_130057/nsys_stats.txt`

Measured results:
- Targeted ctest subset: pass (13/13).
- Strict parity: pass (`u_ssa`/`v_ssa` within strict tolerances).
- Short benchmark (`MIN_WALL=20`, `PISM_MAX_DT=0.05`):
  - `gpism`: `0.582878` years/sec
  - `pism`: `0.00112259` years/sec
  - speedup (`gpism/pism`): `519.226x`
- Long benchmark (`MIN_WALL=120`, `PISM_MAX_DT=0.05`):
  - `gpism`: `0.569934` years/sec
  - `pism`: `0.0410783` years/sec
  - speedup (`gpism/pism`): `13.8743x`
  - vs prior locked long-run gpism (`0.584141` y/s): `-2.43%` (within the
    current 5% non-regression threshold).

Profile delta (phase1b -> phase1d, same default `-y 0.4` command family):
- CUDA API:
  - `cudaLaunchKernel` calls: `365,534` -> `362,237` (`-0.9%`).
  - `cudaMemcpy` calls: unchanged at `4,911` (dominant API time remains transfer-bound).
- GMRES-related kernels:
  - `dot_stag_kernel` instances: `5,543` -> `2,246` (`-59.5%`).
  - `dot_stag_batch_kernel` instances: `3,297` -> `3,297` (unchanged).
  - `orthogonalize_stag_kernel` instances: `3,297` -> `3,297` (unchanged).
