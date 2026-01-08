# Repository Guidelines

## Project Structure & Module Organization
PISM is a C++/C codebase organized by domain modules under `src/` (for example,
`src/stressbalance`, `src/energy`, `src/hydrology`). Executables and library targets
are defined via CMake. Tests and utilities live in `test/` and `tests/`, with regression
scripts under `test/regression/` and performance scripts under `test/performance/`.
Documentation sources are in `doc/`, and user-facing examples are in `examples/`.

## Build, Test, and Development Commands
- Configure and build (out-of-source is standard):
  `cmake -S . -B /path/to/build` then `cmake --build /path/to/build -j$(nproc)`.
- Run a specific test with CTest:
  `ctest -R <regex> -V` from the build directory.
- Run a regression script directly:
  `test/regression/<script>.sh /path/to/build <mpiexec> <repo-root>`.
- Example executable (after build):
  `/path/to/build/pism -help` for options and diagnostics.

## Coding Style & Naming Conventions
Follow existing style in each module and rely on the repo format config:
`.clang-format` and `.clang-tidy` are present at the repo root. Prefer descriptive
snake_case for file names and CMake targets, and keep class and type names consistent
with nearby code. Avoid reformatting unrelated sections.

## Testing Guidelines
Tests are driven by CTest plus shell-based regression scripts in `test/regression/`.
Name new tests after the component and behavior (for example, `sia_*`, `ssa_*`).
Add or update regression tests when changing numerical kernels. If results are expected
to change, document tolerances or reference changes in the test script.

## GPU Acceleration Notes (Current Work)
We are actively adding CUDA acceleration for SIA finite-difference stencils in
`src/stressbalance/sia/`. Key points:
- Build toggle: `Pism_USE_CUDA_SIA` enables CUDA kernels alongside CPU loops.
- PETSc CUDA vectors: use `-dm_vec_type cuda` to allocate DMDA arrays on device.
  (The test harness sets `-use_gpu_aware_mpi 0` to avoid MPI/CUDA interop issues.)
- Performance script: `test/performance/sia_cuda_bench.sh` supports `NO_OUTPUT=1`,
  `FLOW_LAW=pb|arr|arrwarm`, `GRID_LIST=...`, and `CUDA_BLOCKS=...`.
  Block sizes can be tuned via `PISM_SIAFD_CUDA_BLOCK_X/Y`.
- Regression: `test/regression/sia_cuda_smallgrid.sh` compares CPU vs GPU with a small
  tolerance; adjust only if numerical changes are expected and justified.
- Recent learnings: attempting async ghost updates by calling multiple
  `DMLocalToLocalBegin` operations on the same DM caused PETSc errors (outstanding
  operation). Keep geometry ghost updates synchronous unless PETSc constraints are
  addressed. Fused Mahaffy gradient + diffusivity and a single pass over both stencil
  orientations are working and validated against CPU within tolerance.

## Agent Notes (Keep Updated)
- Record important learnings and failed approaches here so they survive context
  compactions (for example, async ghost updates for multiple geometry arrays failed
  with PETSc outstanding operation errors).
- Tried fusing bed smoother `theta`/`thk_smooth` into the CUDA diffusivity kernel
  (computed per-thread instead of separate kernels). On A10G with
  `-Mx 801 -My 801 -Mz 81 -steps 200`, runtime stayed ~6.7s (no meaningful gain),
  so the change was reverted.
- Implemented a combined-orientation loop in the CUDA diffusivity kernel for the
  `no_full_update` path (single vertical loop for both orientations). A10G
  `-Mx 801 -My 801 -Mz 81 -steps 200` runs hovered ~6.6s (slight improvement vs
  ~6.7s baseline; some run-to-run variance).
- Reduced atomics (single max/add per thread) and skipped work when surface
  gradients are exactly zero; performance stayed within noise (~6.6–6.7s), so
  gains are marginal.
- Fused full-update CUDA diffusivity to handle both orientations in one vertical
  loop (reuses enthalpy loads; minor single-rank improvement ~11.75s vs 11.87s,
  still within noise on A10G). Also introduced geometry ghost-update masks so
  only required fields (surface/thickness/bed/mask) exchange ghosts per path;
  expected to help multi-rank runs more than single-rank.
- A10G bench (pism_siafd_test, 801x801x81, steps=10, 1 rank): CPU 39.48s, GPU 13.31s
  with non-CUDA-aware MPI (`-use_gpu_aware_mpi 0`), GPU 11.87s with CUDA-aware MPI
  (`pism-build-cudaaware`, OpenMPI CUDA, block 16x16). Re-check `-steps` scaling if needed.
- Nsight Systems shows large `cudaMallocHost`/`cudaFreeHost` and `VecScatter` time
  for non-GPU-aware MPI staging.
- CUDA-aware OpenMPI is installed at `/home/ec2-user/local/openmpi-cuda` and is
  used by `pism-build-cudaaware`; after moving `update_ghosts()` calls outside
  AccessScope (Geometry/Array helpers/siafd_test), `-use_gpu_aware_mpi 1` runs on
  801x801x81 and trims ~11% vs non-aware MPI.
- Reused the I fields computed in the GPU diffusivity kernel for
  `compute_3d_horizontal_velocity` (skip re-integrating delta); default path now
  uses I when valid, `PISM_SIAFD_CUDA_USE_I=0` forces delta. A10G 801x801x81
  runs: steps=50 improved 19.49s -> 18.87s (~3.2%), steps=10 improved
  11.28s -> 11.14s (~1.2%).
- Tried fusing vertical velocity + strain heating into a single CUDA kernel; it
  was slightly slower (e.g., 19.21s vs 18.87s with I reuse) and has been removed.

## Commit & Pull Request Guidelines
Commit messages in this repo are short, imperative, and scoped (for example, “Fix typo”,
“Build system fix …”). For contributions:
- Work in a feature branch.
- Update `CHANGES.rst` for bug fixes or user-facing changes.
- Include a concise PR description and link relevant issues if available.
Documentation and tests are strongly encouraged.

## Configuration Tips
PISM relies on MPI and PETSc; many runs are configured via command-line options and
`pism_config.nc`. Check `doc/` and `pism -help` for parameter names before adding new
defaults.
