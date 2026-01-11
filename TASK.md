Below is a concrete architecture + execution plan for **gpism**: a PISM‑compatible ice‑sheet model “engine” that is designed *around* GPU strengths (structured stencils + per‑column physics + GPU‑friendly elliptic solvers), while presenting a nearly drop‑in PISM user experience.

---

## 1) Define the compatibility contract (what users see)

**Reference:** Use the original PISM codebase as a primary reference when implementing
compatibility behavior, option semantics, and I/O conventions.

### 1.1 Command-line + configuration compatibility

Implement gpism so that:

* It supports `-config` and `-config_override` the same way PISM does, including the rule that `-config` points at a full config file and `-config_override` may be a subset, and that gpism writes the effective parameters into the output as `pism_config` attributes for provenance. ([PISM][1])
* It accepts “PETSc-style” options on the command line (even if internally you don’t use PETSc everywhere), because a lot of PISM users rely on that habit.
* It supports the executable naming reality:

  * recent PISM changes include renaming `pismr` to `pism` and removing `pismv` in favor of `pism -test X`. ([GitHub][2])
    So gpism should provide `gpism` as the main binary, plus symlinks/aliases `gpismr` and perhaps a `gpism -test X` mode to keep user muscle memory intact.

### 1.2 I/O compatibility (NetCDF layout and metadata)

To behave like PISM in practice:

* Use PISM’s fast storage order for outputs (PISM writes variables as `time,y,x,z` in memory/storage order for performance). ([PISM][3])
* Support the same output format choices (serialized NetCDF3, NetCDF4 parallel, PnetCDF). PISM explicitly supports parallel NetCDF and PnetCDF for high‑resolution runs. ([PISM][3])
* Preserve PISM conventions for:

  * variable names
  * units
  * `history` attribute
  * restart semantics

### 1.3 Numerical “equivalence”

Be explicit in docs: gpism aims for *physics + discretization equivalence* to within tolerances, not bitwise identical results, because GPU floating-point ordering and reductions will differ.

---

## 2) Core architectural decision: split “PISM-compat frontend” from “GPU engine backend”

Think of gpism as two layers:

```
+--------------------------------------------------------------+
|  gpism-compat (frontend)                                     |
|  - CLI options parser (PISM-like)                             |
|  - reads pism_config.nc, applies -config/-config_override     |
|  - NetCDF I/O, restart, diagnostics, history                 |
|  - maps options -> engine settings                            |
+--------------------------|-----------------------------------+
                           v
+--------------------------------------------------------------+
|  gpism-core (GPU-native engine)                              |
|  - distributed grid patches + halos                          |
|  - device-resident fields                                    |
|  - GPU kernels for stencils + per-column physics             |
|  - GPU elliptic solver(s) for SSA/hybrid                     |
|  - time integrator + task graph scheduler                    |
+--------------------------------------------------------------+
```

This separation is what lets you be PISM-compatible *at the surface* while being non‑PISM internally where it matters for GPUs.

---

## 3) Parallelism model: keep PISM’s domain decomposition, but make it GPU-first

PISM’s grid design is already close to what you want for GPUs:

* Horizontal (x/y) domain decomposition across MPI ranks; each rank owns a rectangular patch; vertical columns are not split across ranks. ([PISM][4])
* Ghost cells (“ghost points”) provide neighbor access for stencils. ([PISM][4])

That’s ideal for gpism, because:

* GPU kernels like regular rectangular patches.
* Vertical column physics stays local to a rank (and therefore to a GPU).

### 3.1 Execution model: 1 MPI rank ↔ 1 GPU (by default)

* Each MPI rank owns one patch and binds to one GPU.
* All primary state fields live in device memory permanently during the run.
* Halo exchange uses CUDA/HIP/SYCL-aware MPI where possible; otherwise pack/unpack through pinned host buffers.

### 3.2 Overlap compute and communication

Structure each stencil-based operation as:

1. compute interior region on GPU,
2. start halo exchange,
3. compute boundary region after halos arrive.

This is a big deal for strong scaling.

---

## 4) Data model: GPU-friendly field storage, minimal temporaries, predictable access

### 4.1 Field layout

Use **structure-of-arrays** and contiguous memory in the fastest-varying horizontal index:

* 2D fields: `H(i,j)`, `b(i,j)`, `s(i,j)`, `mask(i,j)`, `u(i,j)`, `v(i,j)`, etc.
* 3D fields for thermodynamics: `E(k,j,i)` or `T(k,j,i)` with `i` contiguous.

Prefer `k` as the slowest index if you do many horizontal stencils over 3D quantities; prefer `k` as fastest if you do mostly per-column vertical operations. In ice sheet models you do *a lot* of per-column work, so a good compromise is:

* store 3D as `(j,i,k)` so each column `(k)` is contiguous, which makes tridiagonal vertical solves and vertical integrals efficient, while 2D stencils still use contiguous `i`.

### 4.2 Keep everything on device

The rule: **the CPU exists for orchestration and I/O only**.
Anything inside the timestep loop must run on GPU:

* stencils
* column solves
* viscosity updates
* norms/reductions (with deterministic options if needed)

### 4.3 Avoid “object-per-gridpoint” patterns

PISM’s style of iterators over points is great for CPU clarity but not for GPUs; replace with bulk kernels over tiles.

---

## 5) GPU-first numerical design: treat the model as two dominant kernel families

Ice sheet modeling in the PISM family naturally splits into:

1. **Local / stencil-heavy operations** (great on GPUs):

   * surface gradient, driving stress
   * SIA diffusivity & fluxes
   * thickness advection/transport
   * mask updates, flotation checks, grounding line logic (some branching, but manageable)

2. **Column-local operations** (fantastic on GPUs):

   * vertical diffusion/advection in energy/enthalpy
   * strain heating, vertical integrals, age tracers, etc.
   * constitutive-law evaluation per grid cell/column

The only “hard GPU problem” is the **SSA/hybrid stress balance solve** (global elliptic solve), so design around that from day 1.

---

## 6) The SSA/hybrid solver: the centerpiece of gpism performance

This is where a GPU-native architecture matters most.

### 6.1 Don’t assemble a giant sparse matrix if you can avoid it

For structured grids, assembled CSR matrices are often memory-bandwidth bound and wasteful on GPUs.

Instead, implement SSA as a **matrix-free operator**:

* `y = A(x)` computed via stencil-like kernels using viscosity/drag coefficients.
* This drastically reduces memory traffic and improves cache/reuse.

### 6.2 Use geometric multigrid as the default preconditioner

On uniform structured grids, **geometric multigrid** is the best match for GPUs:

* Coarsen by factor 2 in x/y.
* Use simple restriction/prolongation stencils.
* Use GPU-friendly smoothers:

  * Chebyshev-Jacobi (very GPU-friendly)
  * or red-black Gauss-Seidel / multi-color (effective but more complex)

Then wrap with Krylov (FGMRES/GMRES) if needed.

### 6.3 Nonlinearity handling

SSA viscosity depends on strain rates → nonlinear.

Use:

* Picard iterations (robust) with multigrid-preconditioned linear solves, or
* Newton-Krylov later (faster but more fragile).

Make Picard the default “PISM-equivalent” path first.

### 6.4 PETSc integration: optional compatibility bridge

You have two viable strategies:

**Strategy A (recommended for “full GPU use”)**
Write your own matrix-free multigrid + Krylov inside gpism-core.

**Decision (2026-01-11):** Select **Strategy A** for gpism. The SSA/hybrid solver will be
implemented as a native matrix-free multigrid + Krylov stack inside `gpism-core`.

**Strategy B (pragmatic early path)**
Use PETSc but force GPU types and keep vectors/mats on device:

* PETSc has explicit GPU matrix types like `MATAIJCUSPARSE` for NVIDIA GPUs. ([PETSc][5])
* PETSc is actively expanding GPU support so more solvers can run entirely on GPU, though some areas are still “work in progress.” ([PETSc][6])

In practice, many teams start with B for speed-to-science, then migrate critical paths to A for peak performance and portability.

If you choose B, strongly prefer matrix-free `MatShell` operators (GPU kernels) over assembled AIJ whenever possible.

---

## 7) Thermodynamics/enthalpy: design around batched column solves

PISM’s vertical grid can be “quite arbitrary.” ([PISM][4])
That’s still GPU-friendly if you treat each column as an independent 1D problem.

### 7.1 Use an implicit vertical diffusion solve per column

* Discretize vertical diffusion implicitly (stable, larger dt).
* Solve tridiagonal systems **batched across all columns** on GPU.

  * NVIDIA: cuSPARSE / custom Thomas kernel
  * AMD: rocSPARSE / custom
  * Or implement a portable batched tridiagonal solver (common in geophysical codes).

### 7.2 Fuse column kernels

For each column, in one kernel (or a small pipeline):

* compute strain heating,
* assemble tridiagonal coefficients,
* solve,
* update enthalpy/temperature and phase change diagnostics.

Minimize global memory reads/writes.

---

## 8) Thickness evolution + transport: stencil kernels + positivity preservation

This part is usually bandwidth-limited and should fly on GPUs if you:

* compute fluxes with a compact stencil,
* use a limiter/positivity-preserving scheme,
* keep halo width minimal (typically 1–2 cells for most schemes).

If you want to match PISM closely, implement its flux-limiter approach for strict non-negativity.

---

## 9) A GPU-native timestep “task graph” (how a run proceeds)

A good timestep loop for gpism looks like a DAG of tasks, not a monolithic routine. Example:

1. **Halo fill** for geometry fields needed in stress balance.
2. **Update masks / grounding** (GPU kernel)
3. **Stress balance solve** (SSA/hybrid):

   * update viscosity (GPU)
   * nonlinear iteration loop:

     * multigrid-preconditioned solve (GPU)
     * convergence checks (GPU reductions)
4. **Compute vertically-averaged/advection velocities** (GPU kernels)
5. **Thickness transport step** (GPU)
6. **Energy/enthalpy step** (batched columns on GPU)
7. **Basal melt / hydrology coupling hooks** (GPU + small reductions)
8. **Diagnostics accumulation** (GPU reductions)
9. **I/O checkpoints / snapshots** (async copy to host + parallel NetCDF write)

Key: make halo exchanges and reductions explicit tasks so you can overlap them.

---

## 10) I/O strategy: match PISM conventions but keep GPUs busy

### 10.1 Match PISM’s performance-oriented dimension order

PISM writes `time,y,x,z` for output efficiency. ([PISM][3])
gpism should do the same to avoid surprising file performance regressions and to stay compatible.

### 10.2 Use parallel NetCDF / PnetCDF when scaling

PISM supports both parallel NetCDF and PnetCDF for better performance at scale. ([PISM][3])
gpism should implement the same `-o_format` behavior and ideally auto-select based on build/runtime.

### 10.3 Asynchronous output (important for GPU utilization)

When it’s time to write:

* stage only the fields you need from device → pinned host buffers asynchronously
* write in parallel while GPUs move on to the next timestep if your model/outputs allow (double-buffer checkpoints)

---

## 11) Extensibility: “PISM-like modules” but GPU-safe

Define a module API like:

```cpp
struct Module {
  virtual void init(Context&) = 0;
  virtual void prepare_timestep(State&, double t, double dt) = 0;
  virtual void run(State&, double t, double dt) = 0;   // GPU work
  virtual void diagnostics(State&, Diagnostics&) = 0;  // GPU reductions
};
```

Rules:

* Module code must be able to run entirely on device.
* Any CPU logic must be outside the hot loop, or behind small control flags.

This gives you PISM’s conceptual modularity without the “CPU-object iteration” pattern.

---

## 12) Development roadmap: build the GPU engine in phases with continuous PISM cross-checks

### Phase 0: “Compat shell”

* Parse PISM-like options.
* Read `-config` and `-config_override` and write `pism_config` provenance. ([PISM][1])
* Read/write NetCDF with PISM-ish dimension order. ([PISM][3])
* Implement distributed grid patches + ghost halos (MPI). Match PISM’s horizontal-only decomposition conceptually. ([PISM][4])

**Deliverable:** `gpism -test smoke` runs, writes a valid restart/output file.

### Phase 1: “Fast SIA + thickness transport”

* Implement SIA velocities (local) + conservative thickness evolution.
* Validate on simplified geometry tests (EISMINT-like) and compare with PISM outputs.

**Deliverable:** gpism reproduces PISM SIA-only runs within tolerances and shows GPU speedup.

### Phase 2: “Thermodynamics/enthalpy GPU column solve”

* Implement cold/enthalpy energy model with batched tridiagonal solves on GPU.
* Validate temperature profiles + melt rates.

**Deliverable:** full 3D thermodynamics at scale, strong GPU utilization.

### Phase 3: “SSA/hybrid with GPU multigrid”

* Implement SSA operator, Picard iterations, multigrid preconditioner.
* Validate velocity fields against PISM SSAFD/SSA variants on standard cases.

**Deliverable:** gpism becomes *meaningfully faster* than PISM on the workloads that matter.

### Phase 4: “PISM ecosystem features”

* calving/front retreat modules
* subglacial hydrology couplers
* paleo / dEBM-simple style forcing options (as needed)
* inversion workflows (if you want them)

At each phase, keep a **compatibility matrix**: which PISM options are supported, which are ignored, which have different defaults.

---

## 13) Practical “gpism run” method (user workflow)

A PISM-style run should look like:

```bash
mpirun -np 8 gpism \
  -i input.nc \
  -config my_config.nc \
  -config_override overrides.nc \
  -y 200 \
  -o out.nc \
  -o_format netcdf4_parallel
```

Add GPU controls that don’t break PISM scripts, e.g.:

* `-gpism_device 0` (or auto)
* `-gpism_backend cuda|hip|sycl`
* `-gpism_streams N`
* `-gpism_overlap_halo 1`
* `-gpism_ssa_solver multigrid|petsc`
* `-gpism_precision mixed|double`

If you prefix them with `gpism_` they won’t collide with existing PISM flags and you can safely ignore unknown gpism flags in “pure compat mode.”

---

## 14) The two biggest technical risks and how to de-risk them

### Risk A: SSA solver performance/robustness

Mitigation:

* Start with Picard + multigrid-preconditioned Krylov.
* Build a solver test harness separate from the full model.
* Implement careful stopping criteria and diagnostics (residual norms, velocity change norms).

### Risk B: “PISM equivalence” across a wide option space

Mitigation:

* Treat compatibility as a product:

  * implement a `gpism -test` suite mirroring PISM’s verification tests (PISM itself encourages `make test` workflows)
  * maintain a CI baseline comparison against PISM for a small set of canonical runs

---

## 15) Recommended implementation stack (balanced for performance + portability)

If you want to make “full use of GPUs” while avoiding a single-vendor trap:

* **C++ + Kokkos** (or equivalent portability layer) for kernels and memory management.
* **MPI** for distributed memory scaling.
* **Parallel NetCDF / PnetCDF** for I/O (match PISM options). ([PISM][3])
* **Custom geometric multigrid** (preferred) for SSA, or PETSc GPU types as an early bridge:

  * PETSc provides GPU matrix types like `MATAIJCUSPARSE`. ([PETSc][5])
  * PETSc’s GPU support is actively evolving. ([PETSc][6])

---

If you want, I can take the next step and produce a more “engineering-ready” spec that includes:

* a proposed gpism state vector (exact fields, halo widths, storage layout),
* a concrete SSA discretization choice (staggered vs collocated, boundary handling),
* a multigrid V-cycle design (levels, smoothers, restriction/prolongation),
* and a minimal subset of PISM options to support first (so you can get to a working prototype fast).

[1]: https://www.pism.io/docs/manual/practical-usage/parameters.html "PISM’s configuration parameters and how to change them — PISM, a Parallel Ice Sheet Model 2.2.2-d6b3a29ca committed by Constantine Khrulev on 2025-03-28 documentation"
[2]: https://github.com/pism/pism/releases "Releases · pism/pism · GitHub"
[3]: https://www.pism.io/docs/manual/practical-usage/input-output.html "Input and output — PISM, a Parallel Ice Sheet Model 2.2.2-d6b3a29ca committed by Constantine Khrulev on 2025-03-28 documentation"
[4]: https://www.pism.io/doxygen/classpism_1_1Grid.html "PISM, A Parallel Ice Sheet Model: pism::Grid Class Reference"
[5]: https://petsc.org/release/manualpages/Mat/MATAIJCUSPARSE/?utm_source=chatgpt.com "MATAIJCUSPARSE — PETSc 3.24.2 documentation"
[6]: https://petsc.org/main/overview/gpu_roadmap/?utm_source=chatgpt.com "GPU Support Roadmap"

Below is an engineering-ready “gpism v0.1” architecture and plan that keeps **user-facing behavior PISM-like** (CLI + NetCDF conventions) while making the **core numerics GPU-native** (matrix-free, stencil-heavy, multigrid-preconditioned Krylov, device-resident state).

I’ll structure this as (1) **core abstractions**, (2) **state vector**, (3) **SSA discretization choice**, (4) **nonlinear + linear solver design**, (5) **GPU multigrid V-cycle**, and (6) a **minimal PISM option subset** that’s actually useful on day 1.

---

## 1) gpism core architecture

### 1.1 Design constraints to satisfy

1. **PISM-compatibility at the interface level**

* Support `-i`, `-o`, `-o_size`, etc. (PISM’s basic I/O options are documented in its manual). ([PISM][1])
* Read/write the same *field names* (`thk`, `topg`, `usurf`, `tauc`, velocity names…) and store run config metadata in the output (`pism_config` attributes). ([PISM][2])
* Preserve PISM’s “storage order” choice when writing NetCDF variables (`time,y,x,z`) for performance/compatibility. ([PISM][1])

2. **GPU-first numerics**

* Avoid assembling large sparse matrices on the GPU every Picard iteration.
* Prefer **matrix-free stencils**, **Krylov methods**, and **geometric multigrid** (GMG) built from regular-grid kernels.

3. **Multi-GPU scaling**

* MPI domain decomposition in x/y only (each rank owns a patch; entire vertical column stays on one rank), mirroring PISM’s grid distribution model.
* Halo/ghost exchange for stencil ops, overlapping comm + compute.

---

## 2) State vector: what gpism must store

A good rule: store **prognostic** (time-evolved) variables + a small set of **persistent solver state**; compute everything else as **diagnostics**.

### 2.1 Grid and decomposition (PISM-like)

PISM’s `Grid` concept:

* rectangular, equally spaced in X and Y
* distributed across processors **in horizontal dimensions only**
* each processor owns a patch `(xs..xs+xm-1, ys..ys+ym-1)`
* surrounded by ghost points for neighbor stencil access

gpism should mirror this because it maps cleanly onto:

* MPI rank ↔ GPU
* “owned region” ↔ interior tiles
* ghost width `gw` ↔ halo layers for stencils

### 2.2 Field locations and memory layout

You’ll want *two* canonical horizontal layouts:

**A) Cell-centered scalars (regular grid)**
Stored at cell centers `(i,j)`:

* `thk` land ice thickness (PISM uses `thk`).
* `topg` bedrock altitude (`topg`).
* `usurf` surface altitude (`usurf`) computed.
* `tauc` basal yield stress (`tauc`) (if you’re doing SSA+sliding).
* `cell_type` / mask fields (byte/int) to tag grounded/floating/ocean/ice-free regions (PISM has a `CellType` concept and multiple masks).

**B) Staggered face fields (MAC-like / SSA grid)**
Velocity unknowns for SSA live on cell faces. PISM has an internal staggered 2D field type that stores two components (dof=2) and defines a “star” accessor corresponding to east/west and north/south cell interfaces.

Concretely:

* `stag(i,j,0)` = quantity at the **east face** of cell `(i,j)` (x-offset)
* `stag(i,j,1)` = quantity at the **north face** of cell `(i,j)` (y-offset)
  and west/south are accessed via index shifts (see `star()` in PISM’s staggered helper).

gpism should store:

* `U_ssa` (staggered) = primary SSA unknown (u on x-faces, v on y-faces)
* `nuH` (staggered) = viscosity×thickness coefficient on faces (SSA operator coefficient)
* `flux_staggered` (staggered) = face fluxes `H_face * velocity_face` (PISM reports `flux_staggered[0]`, `[1]`).

### 2.3 Minimal v0.1 prognostic state (recommended)

**Prognostic:**

* `thk(i,j)` (2D)
* optional: `enthalpy(i,j,k)` (3D) if you want temperature-dependent viscosity; otherwise isothermal

**Quasi-static / from file:**

* `topg(i,j)` (2D)
* `tauc(i,j)` (2D) (constant or read/regridded)

**Persistent solver state:**

* `U_ssa(i,j,2)` (staggered) = last converged SSA solution used as initial guess next step
  PISM can read SSA initial guess velocities (`ubar_ssa` and `vbar_ssa`) unless disabled. ([PISM][1])

**Diagnostics (computed each step, optionally cached):**

* `usurf(i,j)`
* `uvel(i,j)`, `vvel(i,j)` (regular-grid velocities)
* `velbar` (`ubar`,`vbar`) and derived mags
  PISM lists these velocity diagnostics explicitly.

---

## 3) SSA discretization choice: match PISM’s “SSAFD spirit” but GPU-friendly

### 3.1 Why the MAC staggered grid is the right choice

* It’s consistent with PISM’s internal staggered-field conventions (east/west faces = component 0; north/south faces = component 1).
* It avoids checkerboarding and gives clean, symmetric-ish stencils for the stress divergence.
* It maps to GPU kernels naturally: face-centered stencils are regular and memory-friendly.

### 3.2 Continuous form (as in PISM technical notes)

PISM’s calving-front SSA note uses membrane stresses (N_{ij}) and writes SSA in terms of them, then discusses centered finite differences for discretization.

Use the same form:

* (N_{ij} = 2 \bar\nu H D_{ij})

SSA balance (schematically):
[
\nabla\cdot \mathbf{N} = \rho g H \nabla h - \boldsymbol{\tau_b}
]

where (\tau_b) is basal resistance (sliding law).

### 3.3 Discrete operators (second-order, centered + linear interpolation)

Adopt the same “centered FD + linear interpolation to half indices” approach PISM documents in its SSAFD flow-line discretization notes.

Key discrete choices:

* Scalars (`H`, `h=usurf`, `tauc`) at centers.
* Velocities at faces (MAC).
* Compute needed derivatives via centered differences on their natural locations.
* Interpolate coefficients to face locations via simple averaging:
  ((\nu H)*{i+1/2,j} = 0.5[(\nu H)*{i,j}+(\nu H)_{i+1,j}]) etc.
  This matches the half-index averaging style shown in PISM’s SSAFD notes.

### 3.4 Boundary conditions (v0.1 vs later)

PISM has explicit machinery for SSA boundary conditions including:

* masks/values for Dirichlet SSA BCs (`vel_bc_mask`, `vel_bc_values` with `u_bc`, `v_bc`).
* a technical note on calving-front traction boundary conditions and how they alter the discrete SSA equation at a boundary face.

**v0.1 practical plan:**

* Support **Dirichlet** velocity BCs via those mask/value fields.
* Treat calving-front traction BC as a follow-on (but architect the operator to be BC-aware so it’s a localized addition).

---

## 4) Nonlinearity + solver stack

### 4.1 Nonlinearity: Picard loop (PISM-like)

SSA is nonlinear because viscosity depends on strain rate (and temperature).

Use a Picard iteration outer loop:

1. Given `U`, compute strain rates and effective viscosity
2. Build/update face coefficients `nuH`
3. Solve linearized system (A(\nuH) U = b)
4. Check convergence based on `nuH` change

PISM’s SSAFD explicitly discusses tracking norms of `nu H` and recommends an (L^1) criterion for convergence robustness (especially for margin-localized variability).

So, make this your “compatibility knob”:

* Picard convergence test:
  [
  \frac{| \nuH^{k}-\nuH^{k-1}|*{1}}{|\nuH^{k}|*{1} + \epsilon} < \text{tol}_{picard}
  ]
  This is both physically meaningful and aligned with PISM’s own reasoning.

### 4.2 Linear solve: matrix-free + GPU Krylov

To “make full use of GPUs”, **do not assemble a global sparse matrix** each iteration.

Instead:

* Implement `SSAOperator.apply(U_in → R_out)` as a stencil kernel.
* Use GMRES/FGMRES (SSA system is generally not SPD because of coupling + BC handling).

---

## 5) Multigrid V-cycle design (GPU-friendly, concrete)

You asked specifically for a multigrid V-cycle design; here is one that is realistic on GPUs and robust enough for variable-coefficient SSA.

### 5.1 Overall solver structure

Use **MG-preconditioned FGMRES**:

* Outer: FGMRES for robustness with coupled, variable-coefficient systems
* Preconditioner: one V-cycle of geometric multigrid

This gives you both:

* MG speed (grid-independent-ish convergence)
* Krylov robustness (handles the “SSA is not perfectly elliptic SPD” reality)

### 5.2 Hierarchy definition

Levels (l = 0..L) where (l=0) is finest.

* Coarsen by 2× in x and y each level:

  * (N_x^{l+1}=\lceil N_x^l/2\rceil), (N_y^{l+1}=\lceil N_y^l/2\rceil)
  * (dx^{l+1}=2dx^l), (dy^{l+1}=2dy^l)
* Stop when local (per-rank) grid hits e.g. 16×16–32×32.

Store per level:

* `U_l` (staggered solution)
* `R_l` residual
* `nuH_l` coefficients (restricted)
* BC/masks (restricted)

### 5.3 Restriction/prolongation for staggered face grids

Because `u` and `v` live on different face grids, define transfer operators separately, preserving offsets.

**Restriction (full-weighting, face-wise):**

* For u-face grid: average 2×2 fine u-faces into one coarse u-face
* Same for v-face grid.

**Prolongation (bilinear, face-wise):**

* Bilinear interpolate coarse u-faces → fine u-faces (respecting grid offset)
* Same for v-faces.

These are simple stencil kernels → very GPU-friendly.

### 5.4 Smoother choice: Chebyshev-Jacobi (GPU-appropriate)

Avoid Gauss–Seidel (serial dependencies) on GPUs.

Use **Chebyshev polynomial smoothing with Jacobi preconditioning**:

* Needs repeated `A.apply()` plus diagonal scaling
* Entirely parallel

Implementation details:

* Precompute diagonal estimate `D_u`, `D_v` (face-wise) from local coefficients and drag terms.
* Estimate λ_max (needed for Chebyshev) via a few power iterations of the matrix-free operator once per Picard iteration (device-side).
* Use degree 3 Chebyshev, 1–2 pre and post smooths.

### 5.5 Coarse grid solve

At coarsest level:

* run a small fixed number of GMRES iterations with Jacobi
* or do several V-cycles until residual meets a loose tolerance

Avoid CPU direct solves to prevent device↔host transfers.

---

## 6) PISM-compatible interface: what to implement first

### 6.1 Configuration files and provenance

PISM:

* reads defaults from `pism_config.nc`
* supports `-config` (full replacement) and `-config_override` (partial overrides, variable `pism_overrides`)
* saves the actual used configuration into the output file via `pism_config` variable attributes ([PISM][2])

gpism should do the same:

* Bundle a `gpism_config.nc` that matches PISM parameter names (at least the subset you support).
* Implement `-config` and `-config_override` semantics exactly.
* Always write `pism_config` metadata into outputs.

### 6.2 NetCDF variable order

PISM explicitly states it writes `time,y,x,z` storage order for I/O performance. ([PISM][1])

gpism should:

* write in the same order by default (to be “drop-in-ish” for post-processing pipelines)

### 6.3 Variable names

Make gpism output (and accept in input) the same core names PISM documents:

* `thk`, `topg`, `usurf`
* `tauc`
* velocity diagnostics: `uvel`, `vvel`, `velbar` (`ubar`,`vbar`), `uvelsurf`, `vvelsurf`, etc.

---

## 7) Minimal PISM option subset for gpism v0.1

This is the smallest set that still enables real science runs and “feels like PISM”.

### 7.1 I/O and run control (must-have)

From PISM’s manual I/O summary: ([PISM][1])

* `-i <file.nc>` restart
* `-o <file.nc>` output
* `-o_size {small,medium,big_2d,big}` (you can support `small`+`medium` first)
* `-o_format` (start with serial netcdf3/4; add parallel later)
* `-bootstrap` (optional in v0.1; easier to support restarts first)
* `-ssa_read_initial_guess {true/false}` behavior (PISM reads `ubar_ssa`, `vbar_ssa` by default) ([PISM][1])

### 7.2 Grid options (for bootstrap / regrid workflows)

PISM says that with `-bootstrap` you must specify `-Mx`, `-My`, `-Mz` and `-Lz`.

gpism v0.1 recommendation:

* For **restart mode**: read grid from input file (simplest)
* For **bootstrap mode (optional)**: require `-Mx -My` and (if you include 3D thermodynamics) `-Mz -Lz`

### 7.3 Stress balance options (core dynamics)

* `-stress_balance ssa`
* `-stress_balance ssa+sia` (can be v0.2 if you want to ship SSA-only first)

Basal resistance:
PISM’s basal strength section describes the pseudo-plastic sliding law and controlling options.
Implement first:

* `-yield_stress constant` or “read `tauc` from file”
* `-pseudo_plastic`, `-pseudo_plastic_q`, `-pseudo_plastic_uthreshold`

### 7.4 Simple forcing (keep it minimal)

Start with:

* `-surface constant` (uniform SMB)
* optionally: simple constant basal melt

File-based forcing can come later; keep the “forcing interface” modular.

---

## 8) Why this will actually be fast on GPUs

This design is “GPU-native” because the hot path is almost entirely:

* stencil kernels (operator apply, viscosity, flux divergence)
* regular-grid restriction/prolongation kernels
* vector updates + dot products (Krylov)

…and you can keep the entire state on-device except when writing output.

Also: it scales naturally across multiple GPUs using halo exchanges (which are already required by stencil ops). PISM’s own patch+ghost model maps directly.

---

## 9) The recommended MVP sequencing

If you want the shortest path to something usable:

1. **Restart-only, SSA-only, isothermal**

* read `thk`, `topg`, `tauc` (or constant)
* solve SSA, update thickness with simple SMB
* output `thk`, velocities, `usurf`

2. Add **SSA+SIA hybrid** and dt control

3. Add **thermodynamics** (enthalpy/temperature), because it affects viscosity strongly

4. Add **calving-front traction BC** (use PISM’s technical note)

5. Add **more forcing modules** (given/climate coupling), bed deformation, hydrology, etc.

Your project plan is defined in PLAN.md. Keep this file up to date. Use it to identify the next step to take and to track the work you do.
Use git. The "main" branch for this task is gpism-main. Develop features on branch name with the prefix gpism-main-feat-, once these are complete and tested merge them back into gpism-main.
