# SSA Discretization (v0)

This document defines the initial matrix‑free SSA discretization used in gpism (Strategy A).
The goal is PISM‑compatible behavior on a structured 2‑D grid while preserving GPU‑friendly
stencils.

## Grid and variable locations

- Cell centers: scalar fields `H` (thk), `b` (topg), `h` (usurf), `tauc`, masks.
- Staggered (MAC‑like) faces:
  - `u` on **x‑faces** (east/west faces of a cell).
  - `v` on **y‑faces** (north/south faces of a cell).
  - Staggered fields use component index `0` for x‑faces and `1` for y‑faces.

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
- Driving stress is then interpolated to faces as needed.

### Viscosity and coefficients

- Compute strain‑rate invariants on the staggered grid.
- Effective viscosity `ν̄` is computed per face.
- Define `νH = ν̄ * H_face` at faces by linearly interpolating `H` to faces:
  - `H_{i+1/2,j} = 0.5 (H_{i,j} + H_{i+1,j})`
  - `H_{i,j+1/2} = 0.5 (H_{i,j} + H_{i,j+1})`

### Matrix‑free operator

Given `U = (u, v)` on faces, compute `A(U)` by applying the discrete divergence of
membrane stresses using centered differences and interpolated `νH` coefficients.

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
