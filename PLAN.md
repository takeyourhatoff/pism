# gpism Project Plan (tickable checklist)

## Status legend

* `[ ]` not started
* `[x]` done
* `[-]` blocked / waiting (add a note)
* `[~]` in progress (optional convention)

---

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
* [~] Implement logging + run metadata

  * [x] per-rank logging with rank 0 summary
  * [ ] write build + git hash into output metadata when available
* [x] Implement “dry run” mode

  * [x] `gpism -dry_run` prints resolved config and exits

### M2 Definition of Done

* [x] `gpism -dry_run -config X -config_override Y` produces a deterministic, readable resolved config summary
* [x] Unknown options produce helpful error messages (or warnings if you choose permissive mode)

---

## Milestone M3 — NetCDF I/O: restart-in + output-out (compat v0)

**Goal:** Read a restart-like file and write gpism output in a PISM-friendly layout.

### M3 Tasks

* [~] Implement NetCDF reader

  * [x] read grid metadata (Mx/My from dimensions)
  * [x] read core fields: `thk`, `topg`, `tauc` (at minimum)
  * [ ] read dx/dy or coordinate variables
  * [ ] handle missing optional fields with defaults
* [~] Implement NetCDF writer

  * [x] define dimensions (x/y only; no `time` yet)
  * [x] write core fields
  * [ ] write metadata (units/history/etc.)
  * [ ] enforce dimension order convention consistently (document it)
* [~] Implement restart semantics (minimum viable)

  * [ ] read “state at time t”
  * [ ] write restart file at end of run
* [~] Add I/O tests

  * [x] write then read round-trip test for each field type
  * [ ] multi-rank output correctness (each rank writes its slab)

### M3 Definition of Done

* [x] `gpism -i input.nc -o out.nc` produces a readable NetCDF with expected variables
* [x] round-trip tests pass (single-rank)

---

## Milestone M4 — SSA operator (matrix-free) + RHS assembly

**Goal:** Compute `A(U)` and `b` for SSA on GPU with correct BC handling.

### M4 Tasks

* [ ] Define SSA discretization spec (write it down in `docs/ssa_discretization.md`)

  * [ ] variable locations (cell-centered vs staggered faces)
  * [ ] coefficient interpolation rules
  * [ ] boundary condition treatment (v0: Dirichlet mask/value)
* [ ] Implement geometry diagnostics kernels

  * [ ] compute `usurf = topg + thk` (or flotation-aware later)
  * [ ] surface slopes `∂h/∂x, ∂h/∂y`
* [ ] Implement viscosity update kernel (v0: isothermal)

  * [ ] strain rate invariants on staggered grid
  * [ ] effective viscosity `nu` and `nuH`
* [ ] Implement basal resistance term

  * [ ] v0: simple pseudo-plastic or linear drag using `tauc`
* [ ] Implement RHS assembly kernel `b`
* [ ] Implement matrix-free apply: `y = A(x)`

  * [ ] one kernel per component or fused kernel (start simple)
  * [ ] uses halos correctly
* [ ] Implement Dirichlet BCs

  * [ ] support `vel_bc_mask`, `vel_bc_values` (or gpism equivalents)
  * [ ] enforce in `A.apply` and in residual computation
* [ ] Operator verification tests (non-physics)

  * [ ] “zero solution” sanity test with zero slopes + no forcing
  * [ ] symmetry-ish checks on simple constant coefficient cases (as applicable)
  * [ ] manufactured solution test (optional but recommended)

### M4 Definition of Done

* [ ] `A.apply()` and `b` run on GPU, multi-rank, without NaNs
* [ ] Residual norms behave sensibly for trivial setups

---

## Milestone M5 — Linear solver: GMRES/FGMRES on GPU

**Goal:** Solve `A(U)=b` robustly on GPU without assembling a sparse matrix.

### M5 Tasks

* [ ] Implement vector operations on device

  * [ ] axpy, scal, dot, norm2, norm1
  * [ ] reductions are deterministic option (optional)
* [ ] Implement GMRES (start with restart GMRES(m))

  * [ ] device-side orthogonalization (modified Gram-Schmidt)
  * [ ] host-side small Hessenberg solve (acceptable v0) OR device-side small QR
* [ ] Implement preconditioner interface

  * [ ] Identity preconditioner first
* [ ] Add solver test harness

  * [ ] solve a known linear system from a frozen coefficient operator
  * [ ] measure iterations, residual curves

### M5 Definition of Done

* [ ] SSA linear solve converges for simple cases (flat bed, uniform thk, mild slopes)
* [ ] Iteration counts are stable under MPI decomposition changes (roughly)

---

## Milestone M6 — Geometric Multigrid (GMG) preconditioner (GPU-native)

**Goal:** Add a V-cycle preconditioner that gives grid-independent-ish convergence.

### M6 Tasks

* [ ] Implement multigrid level builder

  * [ ] coarsening rules for grid + halos
  * [ ] allocate per-level fields `U_l, R_l, nuH_l` etc.
* [ ] Implement restriction/prolongation for staggered fields

  * [ ] u-face restriction
  * [ ] v-face restriction
  * [ ] prolongation back to fine
* [ ] Implement smoother

  * [ ] Jacobi smoother first (easy)
  * [ ] Chebyshev-Jacobi smoother next (recommended for GPU)
* [ ] Implement residual computation per level
* [ ] Implement V-cycle

  * [ ] pre-smooth
  * [ ] restrict residual
  * [ ] coarse solve (few GMRES/Jacobi iterations)
  * [ ] prolongate correction
  * [ ] post-smooth
* [ ] Integrate as preconditioner in FGMRES
* [ ] Multigrid tests

  * [ ] constant coefficient cases
  * [ ] variable coefficient cases (nuH varies)
  * [ ] scaling test: iterations vs resolution

### M6 Definition of Done

* [ ] FGMRES+MG converges in dramatically fewer iterations than GMRES alone
* [ ] Iterations grow slowly (or not at all) with grid refinement on representative cases

---

## Milestone M7 — SSA nonlinearity: Picard loop + convergence criteria

**Goal:** Full SSA solve with viscosity updated iteratively until converged.

### M7 Tasks

* [ ] Implement Picard outer loop

  * [ ] initialize U (from file if available; else zeros)
  * [ ] update `nuH(U)` each iteration
  * [ ] solve linear system for U
* [ ] Implement Picard convergence checks

  * [ ] L1 norm change of `nuH` (recommended)
  * [ ] optional U-change norm
  * [ ] max iterations + fallback
* [ ] Add stabilization options

  * [ ] under-relaxation of U or nuH
  * [ ] viscosity floor/ceiling
* [ ] Regression tests: consistent results across GPU/CPU (tolerance-based)

### M7 Definition of Done

* [ ] SSA solve converges robustly on a set of non-trivial geometries
* [ ] Results are stable across MPI decomposition and GPU backend (within tolerance)

---

## Milestone M8 — Time stepping + thickness evolution (minimal dynamics loop)

**Goal:** Run multiple timesteps and evolve thickness (`thk`) using computed velocities + SMB.

### M8 Tasks

* [ ] Implement time manager

  * [ ] `t0`, `dt`, `t_end`, output intervals
* [ ] Implement simple forcing module (v0)

  * [ ] constant surface mass balance (SMB) or read from file
* [ ] Implement thickness update (transport)

  * [ ] compute face fluxes `H_face * U_face`
  * [ ] conservative divergence update
  * [ ] positivity preservation (no negative thickness)
* [ ] Implement mask update (minimal)

  * [ ] define ice-free vs ice-covered
  * [ ] grounding/floating can be deferred (but leave hooks)
* [ ] Output at intervals

  * [ ] thk, usurf, velocities, basic diagnostics
* [ ] Add “end-to-end” test case

  * [ ] 10–50 steps, writes outputs, no blow-up

### M8 Definition of Done

* [ ] `gpism` can run a short simulation end-to-end and write multiple time records
* [ ] thickness remains non-negative and physically plausible

---

## Milestone M9 — Thermodynamics (optional for v0.1; critical for realism)

**Goal:** Add enthalpy/temperature evolution and temperature-dependent viscosity.

### M9 Tasks

* [ ] Decide thermodynamics model scope (v0)

  * [ ] isothermal (skip) vs enthalpy diffusion/advection
* [ ] Implement 3D field storage (`Field3D` column-contiguous)
* [ ] Implement batched per-column vertical diffusion solve

  * [ ] tridiagonal assembly kernel
  * [ ] batched tridiagonal solve kernel
* [ ] Couple temperature/enthalpy to viscosity
* [ ] Add thermodynamics validation cases (column tests)

### M9 Definition of Done

* [ ] Thermodynamics step runs at scale on GPU without excessive host transfers
* [ ] viscosity changes impact SSA in expected ways

---

## Milestone M10 — Performance + scaling + profiling

**Goal:** Ensure gpism actually “makes full use of GPUs” and scales multi-GPU.

### M10 Tasks

* [ ] Add profiling hooks

  * [ ] kernel timing (CUDA events / ROCm / SYCL profiling)
  * [ ] MPI timing for halo exchange
* [ ] Reduce host-device traffic audit

  * [ ] confirm timestep loop is device-resident
  * [ ] no hidden syncs in reductions
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
* 🎯 Next:

  * Finish remaining M3 subtasks (dx/dy metadata, time dim, restart at time t, multi-rank output).
  * Start M4 SSA discretization spec in `docs/ssa_discretization.md`.
