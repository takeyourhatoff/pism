# gpism Project Plan (tickable checklist)

## Status legend

* `[ ]` not started
* `[x]` done
* `[-]` blocked / waiting (add a note)
* `[~]` in progress (optional convention)

---

## Plan discipline (do this every time)

* Do not start work unless there is an explicit task for it in this plan.
* If new work is needed, add a task (or sub-task) first, then execute it.
* When a task is finished, immediately update its checkbox and the progress log.

## Milestone M0 — Repo + build + dev workflow (scaffold)

**Goal:** You can compile/run a “hello timestep” executable on CPU + GPU backend.

### M0 Tasks

* [x] Create repo structure (scaffolded under `gpism/`)

  * [x] `src/`, `include/`, `tests/`, `docs/`, `cmake/`, `examples/`
  * [x] Add `PROJECT_PLAN.md` (this doc) and `CONTRIBUTING.md`
* [x] Choose implementation stack (record decisions in `docs/decisions.md`)

  * [x] GPU model: CUDA/HIP/SYCL, or portability layer (e.g., Kokkos)
  * [x] MPI library requirements (CUDA-aware MPI desired)
  * [x] NetCDF stack choice (netcdf-c + HDF5; optional PnetCDF later)
* [x] Set up build system (CMake recommended)

  * [x] CPU-only build works
  * [x] GPU build works (one backend first)
  * [x] `gpism --version` prints build info (backend, precision, MPI, NetCDF)
* [x] Add formatting + linting + basic static analysis

  * [x] clang-format config (or equivalent)

### M0 Definition of Done

* [x] `gpism` builds locally
* [x] You can run `gpism --help` and `gpism --version`

---

## Milestone M1 — Core data model: grid, fields, halos, MPI↔GPU mapping

**Goal:** Device-resident fields with halo exchange across MPI ranks.

### M1 Tasks

* [x] Implement `Context`

  * [x] MPI init / finalize
  * [x] rank, size
  * [x] neighbor ranks
  * [x] device selection policy (rank→GPU mapping)
* [x] Implement `Grid2D`

  * [x] global sizes `Mx, My`, spacings `dx, dy`
  * [x] local patch extents (owned region) per rank
  * [x] ghost width `gw` parameter
* [x] Implement device-friendly `Field2D<T>`

  * [x] allocation includes halos
  * [x] indexing helpers for `(i,j)` with ghost offsets
  * [x] device pointer access + host staging buffer (pinned if available)
* [x] Implement device-friendly `FieldStag2D<T>` (staggered, 2 components)

  * [x] layout for `u-face` and `v-face` components
  * [x] halo storage + component-aware access
* [x] Implement halo exchange

  * [x] pack/unpack kernels (host buffers)
  * [x] MPI Isend/Irecv
  * [x] correctness test for 1-rank and multi-rank cases
* [x] Add a “field smoke test”

  * [x] fill field with a known pattern on each rank
  * [x] exchange halos
  * [x] verify halo values match neighbor interior

### M1 Definition of Done

* [x] Multi-rank run successfully exchanges halos for `Field2D` and `FieldStag2D`
* [x] No host roundtrips inside the halo exchange except the MPI buffers (device buffers supported if CUDA-aware MPI is enabled)

---

## Milestone M2 — PISM-like CLI + config + logging (compat shell v0)

**Goal:** `gpism` feels like a PISM executable: options, config override, provenance.

### M2 Tasks

* [x] Implement CLI parsing

  * [x] `-i`, `-o`, `-y`/`-time` style run length
  * [x] `-config`, `-config_override`
  * [x] gpism-only options prefixed `-gpism_*`
* [x] Implement config system

  * [x] load defaults from `gpism_config.nc` (or embedded defaults)
  * [x] apply `-config` replacement
  * [x] apply `-config_override` (partial overrides)
  * [x] provide `get<T>(key)` accessors with type checks
* [x] Implement logging + run metadata

  * [x] per-rank logging with rank 0 summary
  * [x] write build + git hash into output metadata when available
* [x] Implement “dry run” mode

  * [x] `gpism -dry_run` prints resolved config and exits

### M2 Definition of Done

* [x] `gpism -dry_run -config X -config_override Y` produces a deterministic, readable resolved config summary
* [x] Unknown options produce helpful error messages (or warnings if you choose permissive mode)

---

## Milestone M3 — NetCDF I/O: restart-in + output-out (compat v0)

**Goal:** Read a restart-like file and write gpism output in a PISM-friendly layout.

### M3 Tasks

* [x] Implement NetCDF reader

  * [x] read grid metadata (Mx/My from dimensions)
  * [x] read core fields: `thk`, `topg`, `tauc` (at minimum)
  * [x] read SSA Dirichlet BC fields (`u_bc`, `v_bc`, `vel_bc_mask`)
  * [x] read dx/dy or coordinate variables
  * [x] handle missing optional fields with defaults
* [x] Implement NetCDF writer

  * [x] define dimensions (include `time`)
  * [x] write core fields
  * [x] write SSA Dirichlet BC fields (`u_bc`, `v_bc`, `vel_bc_mask`)
  * [x] write metadata (units/history)
  * [x] add BC metadata (long_name, flag values/meanings for `vel_bc_mask`)
  * [x] enforce dimension order convention consistently (time,y,x)
  * [x] enable NetCDF4/parallel output when library support is available
  * [x] verify parallel IO stack on this machine (MPI + HDF5 parallel + netcdf-c parallel)
* [x] Implement restart semantics (minimum viable)

  * [x] read “state at time t” (last record or `io.time_index`)
  * [x] write restart file at end of run
* [x] Add I/O tests

  * [x] write then read round-trip test for each field type
  * [x] multi-rank output correctness (each rank writes its slab)

### M3 Definition of Done

* [x] `gpism -i input.nc -o out.nc` produces a readable NetCDF with expected variables
* [x] round-trip tests pass (single-rank and MPI)

---

## Milestone M4 — SSA operator (matrix-free) + RHS assembly

**Goal:** Compute `A(U)` and `b` for SSA on GPU with correct BC handling.

### M4 Tasks

* [x] Define SSA discretization spec (write it down in `docs/ssa_discretization.md`)

  * [x] variable locations (cell-centered vs staggered faces)
  * [x] coefficient interpolation rules
  * [x] boundary condition treatment (v0: Dirichlet mask/value)
* [x] Implement geometry diagnostics kernels

  * [x] compute `usurf = topg + thk` (or flotation-aware later)
  * [x] surface slopes `∂h/∂x, ∂h/∂y`
* [x] Implement viscosity update kernel (v0: isothermal)

  * [x] strain rate invariants on staggered grid
  * [x] effective viscosity `nu` and `nuH`
* [x] Implement basal resistance term

  * [x] v0: simple pseudo-plastic or linear drag using `tauc`
* [x] Implement RHS assembly kernel `b`
* [x] Implement matrix-free apply: `y = A(x)`

  * [x] one kernel per component or fused kernel (start simple)
  * [x] uses halos correctly
  * [x] include basic u/v coupling term (shear contribution)
* [x] Implement GPU kernels for SSA operator components

  * [x] basal drag (beta)
  * [x] RHS assembly
  * [x] matrix-free apply
* [x] Implement Dirichlet BCs

  * [x] support `vel_bc_mask`, `vel_bc_values` (or gpism equivalents)
  * [x] enforce in `A.apply` and in residual computation
* [x] Operator verification tests (non-physics)

  * [x] “zero solution” sanity test with zero slopes + no forcing
  * [x] symmetry-ish checks on simple constant coefficient cases (as applicable)
  * [ ] manufactured solution test (optional but recommended)
  * [x] CUDA smoke test exercises device path (when GPISM_ENABLE_CUDA=ON)
  * [x] CPU/GPU parity check on a tiny grid (tolerances)
  * [x] CUDA+MPI operator smoke (A.apply + b + residual finite)

### M4 Definition of Done

* [x] `A.apply()` and `b` run on GPU, multi-rank, without NaNs
* [x] Residual norms behave sensibly for trivial setups

---

## Milestone M5 — Linear solver: GMRES/FGMRES on GPU

**Goal:** Solve `A(U)=b` robustly on GPU without assembling a sparse matrix.

### M5 Tasks

* [~] Implement vector operations on device

  * [x] axpy, scal, dot, norm2, norm1
  * [ ] reductions are deterministic option (optional)
  * [x] validate with small device-side unit tests
* [x] Implement GMRES (start with restart GMRES(m))

  * [x] device-side orthogonalization (modified Gram-Schmidt)
  * [x] host-side small Hessenberg solve (acceptable v0) OR device-side small QR
* [x] Implement preconditioner interface

  * [x] Identity preconditioner first
* [x] Add solver test harness

  * [x] solve a known linear system from a frozen coefficient operator
  * [x] measure iterations, residual curves
  * [x] SSA operator smoke solve (few GMRES iterations, residual drops)
* [x] Add non-trivial SSA GMRES regression (nuH>0, known solution)
* [x] Validate GMRES convergence on simple SSA cases under MPI decomposition
* [x] Add MPI decomposition stability test for GMRES iterations (1 vs 2 ranks)

### M5 Definition of Done

* [x] SSA linear solve converges for simple cases (flat bed, uniform thk, mild slopes)
* [x] Iteration counts are stable under MPI decomposition changes (roughly)

---

## Milestone M6 — Geometric Multigrid (GMG) preconditioner (GPU-native)

**Goal:** Add a V-cycle preconditioner that gives grid-independent-ish convergence.

### M6 Tasks

* [x] Implement multigrid level builder

  * [x] coarsening rules for grid + halos
  * [x] allocate per-level fields `U_l, R_l, nuH_l` etc.
* [x] Implement restriction/prolongation for staggered fields

  * [x] u-face restriction
  * [x] v-face restriction
  * [x] prolongation back to fine
* [x] Implement smoother

  * [x] Jacobi smoother first (easy)
  * [x] Chebyshev-Jacobi smoother next (recommended for GPU)
* [x] Implement residual computation per level
* [x] Implement V-cycle

  * [x] pre-smooth
  * [x] restrict residual
  * [x] coarse solve (few GMRES/Jacobi iterations)
  * [x] prolongate correction
  * [x] post-smooth
* [x] Integrate as preconditioner in FGMRES
* [x] Multigrid tests

  * [x] constant coefficient cases
  * [x] variable coefficient cases (nuH varies)
  * [x] scaling test: iterations vs resolution
  * [x] GMRES vs GMRES+MG iteration comparison smoke test
  * [x] MG refinement sanity check (16x16 vs 32x32)

### M6 Definition of Done

* [x] FGMRES+MG converges in dramatically fewer iterations than GMRES alone
* [x] Iterations grow slowly (or not at all) with grid refinement on representative cases

---

## Milestone M7 — SSA nonlinearity: Picard loop + convergence criteria

**Goal:** Full SSA solve with viscosity updated iteratively until converged.

### M7 Tasks

* [x] Implement Picard outer loop

  * [x] initialize U (from file if available; else zeros)
  * [x] update `nuH(U)` each iteration
  * [x] solve linear system for U
  * [x] wire `vel_bc_mask/u_bc/v_bc` into solver state (config-gated)
* [x] Implement Picard convergence checks

  * [x] L1 norm change of `nuH` (recommended)
  * [x] optional U-change norm
  * [x] max iterations + fallback
* [x] Add stabilization options

  * [x] under-relaxation of U or nuH
  * [x] viscosity floor/ceiling
* [x] Add MPI halo exchange for SSA solver fields (nuH, vel, rhs, beta)
* [x] Add Picard convergence test on non-trivial geometry (single-rank)
* [x] Add MPI Picard convergence test (1 vs 2 ranks)
* [x] Regression tests: consistent results across GPU/CPU (tolerance-based)

### M7 Definition of Done

* [x] SSA solve converges robustly on a set of non-trivial geometries
* [x] Results are stable across MPI decomposition and GPU backend (within tolerance)

---

## Milestone M8 — Time stepping + thickness evolution (minimal dynamics loop)

**Goal:** Run multiple timesteps and evolve thickness (`thk`) using computed velocities + SMB.

### M8 Tasks

* [x] Implement time manager

  * [x] `t0`, `dt`, `t_end`, output intervals
* [x] Implement simple forcing module (v0)

  * [x] constant surface mass balance (SMB) or read from file
* [x] Implement thickness update (transport)

  * [x] compute face fluxes `H_face * U_face`
  * [x] conservative divergence update
  * [x] positivity preservation (no negative thickness)
* [x] Implement mask update (minimal)

  * [x] define ice-free vs ice-covered
  * [x] grounding/floating can be deferred (but leave hooks)
* [x] Output at intervals

  * [x] thk, usurf, velocities, basic diagnostics
  * [x] append multiple time records to a single NetCDF output file
* [x] Add “end-to-end” test case

  * [x] 10–50 steps, writes outputs, no blow-up
  * [x] SSA-only loop (no thickness evolution) as a first end-to-end test

### M8 Definition of Done

* [x] `gpism` can run a short simulation end-to-end and write multiple time records
* [x] thickness remains non-negative and physically plausible

---

## Milestone M9 — Thermodynamics (optional for v0.1; critical for realism)

**Goal:** Add enthalpy/temperature evolution and temperature-dependent viscosity.

### M9 Tasks

* [ ] Decide thermodynamics model scope (v0)

  * [x] enthalpy diffusion/advection (batched tridiagonal)
* [x] Implement 3D field storage (`Field3D` column-contiguous)
* [x] Implement batched per-column vertical diffusion solve

  * [x] tridiagonal assembly kernel
  * [x] batched tridiagonal solve kernel
* [x] Couple temperature/enthalpy to viscosity
* [x] Add thermodynamics validation cases (column tests)
* [x] Add thermodynamics scale smoke test + host-transfer audit (GPU)
* [x] Wire thermodynamics update into timestep loop (config + init + diffusion step)

### M9 Definition of Done

* [x] Thermodynamics step runs at scale on GPU without excessive host transfers
* [x] viscosity changes impact SSA in expected ways

---

## Milestone M10 — Performance + scaling + profiling

**Goal:** Ensure gpism actually “makes full use of GPUs” and scales multi-GPU.

### M10 Tasks

* [ ] Add profiling hooks

  * [x] kernel timing (CUDA events / ROCm / SYCL profiling)
  * [x] MPI timing for halo exchange
* [ ] Reduce host-device traffic audit

  * [x] confirm timestep loop is device-resident
  * [x] no hidden syncs in reductions
  * [x] remove unconditional `cudaDeviceSynchronize()` calls from CUDA ops (use error checks + sync only at host reads)
  * [x] move timestep kernels (geometry, thickness, velocity) and SSA Picard norms/relax to device; stage to host only for IO
* [ ] Overlap comm/compute

  * [ ] interior compute while halos exchange
  * [ ] boundary compute after halos arrive
* [ ] Memory bandwidth optimization

  * [ ] fuse kernels where it matters (operator apply + coefficient loads)
  * [ ] minimize temporaries
* [ ] Mixed precision option (optional but high value)

  * [ ] keep solution in FP64, smoothers in FP32 (or configurable)
* [ ] Scaling benchmarks

  * [ ] 1 GPU strong scaling
  * [ ] multi-GPU weak scaling
  * [ ] write benchmark report in `docs/performance.md`

### M10 Definition of Done

* [ ] Clear GPU utilization in profiler (kernels dominate wall time)
* [ ] Multi-GPU runs show expected scaling trends
* [ ] A benchmark suite exists and is repeatable

---

## Milestone M11 — PISM compatibility expansion (targeted)

**Goal:** Support the most common PISM workflows users actually run.

### M11 Tasks (pick in priority order)

* [ ] Implement `-bootstrap` pathway (create initial fields from minimal inputs)
* [ ] Support reading/writing SSA initial guess fields (`ubar_ssa`, `vbar_ssa`)
* [ ] Implement additional common diagnostics

  * [ ] `velbar`, `uvelsurf`, `vvelsurf`, flux diagnostics
* [ ] Expand forcing modules

  * [ ] SMB from file
  * [ ] simple surface temperature forcing
* [ ] Add calving-front traction BC (separate milestone if big)
* [ ] Improve grounding line / flotation logic (if needed for your target cases)

### M11 Definition of Done

* [ ] A real PISM workflow you care about can be replicated using gpism with minimal script changes

---

## Milestone M12 — Documentation + release + user onboarding

**Goal:** Someone else can install gpism, run a case, and understand differences vs PISM.

### M12 Tasks

* [ ] Write “Getting Started”

  * [ ] build instructions (CPU/GPU)
  * [ ] required libraries
* [ ] Write “PISM compatibility” guide

  * [ ] supported flags and variables
  * [ ] unsupported flags (explicit list)
  * [ ] known numerical differences
* [ ] Provide example runs in `examples/`

  * [ ] SSA-only demo
  * [ ] thickness evolution demo
* [ ] Add versioning + changelog
* [ ] Tag a `v0.1.0` release

### M12 Definition of Done

* [ ] New user can run an example and get expected output with minimal help

---

# Backlog (future milestones / optional major features)

* [ ] SSA+SIA hybrid coupling
* [ ] Calving + front evolution
* [ ] Subglacial hydrology coupling
* [ ] Bed deformation / GIA coupling
* [ ] Adjoint/inversion workflows
* [ ] More I/O formats and post-processing integrations

---

# Progress log template

**Week of YYYY-MM-DD**

* ✅ Completed:

  * -
* 🔧 In progress:

  * -
* 🧱 Blocked:

  * -
* 🎯 Next:

  * -

**Week of 2026-01-11**

* ✅ Completed:

  * Scaffolded `gpism/` directories and starter docs (`PROJECT_PLAN.md`, `CONTRIBUTING.md`, `docs/decisions.md`).
  * Added a minimal gpism CMake build and `gpism --version` stub.
  * Added CMake cache options for backend/precision with optional MPI/NetCDF detection.
  * Added clang-format target and optional clang-tidy integration in gpism CMake.
  * Added gpism `Context` (MPI-aware) and `Grid2D` decomposition stubs.
  * Added `Field2D`/`FieldStag2D` scaffolding and a halo exchange implementation with a smoke test executable.
  * Added env-driven device selection policy in `gpism::Context`.
  * Installed cmake and built gpism locally (including `gpism --version`).
  * Ran gpism field smoke test (single-rank) via CTest.
  * Completed CUDA-enabled gpism build on Linux GPU host.
  * Ran gpism field smoke tests in MPI mode (2 ranks) and single-rank.
  * Expanded smoke test to validate staggered field halo exchange.
  * Added host staging buffers + optional CUDA device allocations in `Field2D`.
  * Added `Context::neighbors_2d` helper for rank neighbor lookup.
  * Selected SSA solver Strategy A (native matrix-free multigrid + Krylov).
  * Added optional CUDA-aware MPI halo exchange path (device buffers).
  * Implemented CLI/config/dry-run scaffold for M2.
  * Added minimal NetCDF restart read/write path for `thk`, `topg`, `tauc`.
  * Verified `gpism -i input.nc -o out.nc` writes expected NetCDF variables.
  * Added NetCDF round-trip CTest (io_smoke).
  * Wrote SSA discretization spec for M4.
  * Completed remaining M3 subtasks (dx/dy, metadata, time dimension, multi-rank IO).
  * Verified NetCDF IO smoke tests with MPI.
  * Added geometry diagnostics (usurf + surface slopes) with CUDA-aware path.
  * Implemented isothermal SSA viscosity update (strain-rate invariants + nuH).
  * Built MPI-enabled HDF5 + NetCDF-C with parallel NetCDF4 support.
  * Enabled NetCDF4 parallel output path in `NetcdfIO`.
  * Added gpism build/version/git metadata to NetCDF outputs.
  * Implemented SSA basal drag, RHS assembly, and matrix-free apply (CPU v0).
  * Added SSA operator smoke test (zero forcing/zero velocity).
  * Added simple SSA operator coupling term and symmetry-ish check.
  * Added NetCDF I/O for SSA Dirichlet BC fields (`u_bc`, `v_bc`, `vel_bc_mask`).
  * Added CUDA kernels for SSA operator apply/RHS/basal drag (device path).
  * Added `vel_bc_mask` metadata (long_name + flag values/meanings) in NetCDF output.
  * Added CUDA SSA operator smoke test with CPU/GPU parity checks.
  * Added device-capable linear algebra ops (axpy/scal/dot/norms) with a smoke test.
  * Added restart GMRES with identity preconditioner and a basic solver smoke test.
  * Extended GMRES harness with residual tracking and SSA operator smoke solve.
  * Added non-trivial SSA GMRES regression with nuH>0 and known solution.
  * Added multigrid level builder with a sizing smoke test.
  * Added staggered restriction/prolongation with transfer smoke test.
  * Added host-side Jacobi smoother with a smoke test.
  * Added Chebyshev-Jacobi smoother with a diagonal smoke test.
  * Added residual computation + V-cycle scaffold with smoke coverage.
  * Added multigrid preconditioner wrapper with a smoke test.
  * Added multigrid constant/variable coefficient residual reduction tests.
  * Added multigrid scaling smoke test (8x8 vs 16x16).
  * Added SSA Picard solver scaffold with convergence checks and a smoke test.
  * Added Picard stabilization options (relaxation + nuH bounds).
  * Added non-trivial SSA Picard smoke test (non-zero slopes).
  * Verified CPU build (NetCDF disabled) and ran `ctest` for core smoke tests.
  * Verified CUDA build (NetCDF disabled) and ran `ctest` including CUDA smoke tests.
  * Verified parallel NetCDF stack (HDF5-par + netcdf-c-par) and MPI `gpism-io-smoke-mpi` test.
  * Verified CUDA+MPI SSA operator smoke with finite residual norms.
  * Verified GMRES MPI smoke (1 vs 2 ranks) with stable iteration counts.
  * Verified MG preconditioner reduces GMRES iterations and scales gently with refinement.
  * Added MPI halo exchanges for SSA solver fields and halo-aware SSA apply.
  * Added global MPI reductions for GMRES dot/norms and Picard convergence checks.
  * Added MPI SSA Picard smoke tests (1 and 2 ranks) and enforced convergence in non-trivial Picard test.
  * Added host-only SSA solve toggle for parity testing and CPU/GPU Picard parity smoke test.
  * Added time manager, constant SMB forcing, thickness transport update, and mask update utilities.
  * Extended NetCDF IO to optionally read/write `uvel`, `vvel`, and `usurf`.
  * Wired a minimal timestep loop into `gpism` with interval outputs (file-per-interval).
  * Added thickness transport smoke test and SSA-only timestep loop smoke test.
  * Added NetCDF timestep IO smoke test (conditional on NetCDF builds).
  * Added append-to-time NetCDF outputs (single file, multiple records) and verified `gpism-timestep-io-smoke`.
  * Removed unconditional CUDA syncs and added device-resident thickness/velocity/SSA Picard updates with host staging only for IO.
  * Added CUDA viscosity path, GPU diff norms, and shared field sync helpers; updated GPU-facing tests to sync device/host data.
  * Profiled CUDA build with Nsight Systems; SSA operator + thickness tests show GPU kernels executing (no stray device-wide syncs).
  * Added lightweight profiling hooks (CUDA event timers + halo exchange timers) and validated output via `GPISM_PROFILE=1` on GPU GMRES smoke.
  * Implemented `Field3D` and GPU/CPU vertical diffusion (tridiagonal assembly + batched solve) with a new parity smoke test.
  * Added enthalpy-coupled viscosity scaling with CPU/GPU parity test coverage.
  * Wired thermodynamics into the timestep loop and added an SSA temperature-response smoke test.
  * Added a thermodynamics scale smoke test, fixed `Field3D` move semantics for swap safety, and verified minimal host/device transfers with Nsight Systems.
* 🧱 Blocked:

  * -
* 🎯 Next:

  * Pick the next M10 performance tasks (overlap or bandwidth) and add explicit subtasks before starting.
