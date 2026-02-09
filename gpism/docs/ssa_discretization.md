# SSA Discretization (PISM SSAFD-Compatible)

This document defines gpism's matrix-free SSA discretization.
The goal is PISM-compatible behavior (matching PISM's SSAFD method) on a structured 2-D
grid while preserving GPU-friendly stencils.

## Grid and variable locations

- Cell centers:
  - Scalar fields: `H` (thk), `b` (topg), `h` (usurf), `tauc`, masks.
  - SSA unknowns: velocities `u` and `v` are **cell-centered** (as in PISM SSAFD).
- Staggered faces (coefficients):
  - `nuH` is stored on faces:
    - component `0`: x-faces (between `(i,j)` and `(i+1,j)`)
    - component `1`: y-faces (between `(i,j)` and `(i,j+1)`)
- Thickness transport:
  - The transport solver uses a derived face-staggered velocity field computed
    from the cell-centered SSA velocity by averaging to faces.

## Continuous form (SSA)

We solve the depth‑averaged SSA momentum balance:

```
∇·N = ρ g H ∇h − τ_b
```

with membrane stress tensor `N = 2 ν̄ H D`, where `D` is the strain‑rate tensor and `ν̄`
is effective viscosity (depth‑averaged).

## Discrete operators (v0)

### Surface slope and driving stress

- `h = b + H` (or flotation‑aware later).
- `∂h/∂x` and `∂h/∂y` are computed with centered differences at cell centers.
- Driving stress RHS is cell-centered:
  - `b_u = -rho * g * H * (∂h/∂x)`
  - `b_v = -rho * g * H * (∂h/∂y)`

### Viscosity and coefficients

- Compute strain-rate invariants using cell-centered velocities and PISM-style
  staggered formulas.
- Effective viscosity `ν̄` is computed per face, and we store `nuH = ν̄ * H_face`
  on faces by linearly interpolating `H` to faces:
  - `H_{i+1/2,j} = 0.5 (H_{i,j} + H_{i+1,j})`
  - `H_{i,j+1/2} = 0.5 (H_{i,j} + H_{i,j+1})`

### Matrix‑free operator

Given `U = (u, v)` at cell centers, compute `A(U)` using the PISM SSAFD stencil:
- `nuH` provides variable-coefficient coupling on faces.
- Basal drag `beta_u/beta_v` multiplies `u/v` at the same cell center.

### Basal resistance (v0)

Use a simple drag law with `tauc`:

```
τ_b = C * |U|^{q-1} U
```

with `C` derived from `tauc` and default `q = 1` (linear drag) in v0. This matches the
initial PISM‑style behavior and can be extended to pseudo‑plastic later.

## Boundary conditions (v0)

- Dirichlet velocity BCs via `vel_bc_mask` and `vel_bc_values`:
  - If a face is masked, set the operator row to enforce `U = U_bc`.
  - This is applied in both residual computation and in the operator application.

## Notes / follow‑ups

- Calving‑front traction BCs are deferred (see PISM technical note).
- Time‑dependent masks and grounding line coupling deferred.
- This discretization is designed to be directly translatable to GPU stencils
  for matrix‑free Krylov + multigrid.
