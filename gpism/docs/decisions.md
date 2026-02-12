# Architecture Decisions

This file records the active gpism architecture decisions.

## 2026-02-12: GPU-only product contract

- Build contract:
  - CUDA toolkit is required at configure time.
  - NetCDF is required at configure time.
  - MPI remains optional at build time.
- Runtime contract:
  - Production timestep execution is device-resident.
  - No runtime `device.enabled` switch exists.
  - No host fallback halo path exists for MPI exchange.
- MPI contract:
  - Multi-rank execution uses device buffers only.
  - If CUDA-aware MPI is unavailable and `size > 1`, startup fails fast.
- Solver contract:
  - SSA discretization follows PISM SSAFD semantics (cell-centered `u/v`,
    staggered `nuH`).
  - Deterministic host-reduction mode was removed from public linear algebra
    APIs.

Rationale: gpism targets GPU throughput first, with strict parity against the
original PISM SSAFD behavior as the primary correctness gate.
