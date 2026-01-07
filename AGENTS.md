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
