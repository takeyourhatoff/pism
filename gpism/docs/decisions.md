# Architecture Decisions

This file records gpism technical choices. Update entries as decisions change.

## 2026-01-11: Initial stack (provisional)

- Language: C++
- GPU portability: Kokkos (preferred), with CUDA/HIP backends
- Parallelism: MPI with CUDA-aware support when available
- I/O: NetCDF (netcdf-c + HDF5); add PnetCDF later if needed

Rationale: aligns with gpism's GPU-first plan while keeping portability and PISM-like I/O.
