# gpism benchmarks (repeatable)

## Inputs

- **std-greenland input:** `examples/std-greenland/pism_Greenland_5km_v1.1.nc`
  - Run `examples/std-greenland/preprocess.sh` to generate it, or set
    `PISM_STD_GREENLAND_INPUT` to a custom path.

## Scripts

- `run_std_greenland.sh`
  - Runs a short std-greenland step with a fixed override config.
  - Env vars:
    - `GPISM_BUILD_DIR` (default `gpism/build-cuda-mpi`)
    - `PISM_STD_GREENLAND_INPUT` (default `examples/std-greenland/pism_Greenland_5km_v1.1.nc`)
    - `GPISM_STD_GREENLAND_OUT` (default `/tmp/gpism_std_greenland_out.nc`)
    - `GPISM_STD_GREENLAND_YEARS` (default `0.05`)
    - `GPISM_STD_GREENLAND_OVERRIDE` (default `gpism/scripts/benchmarks/std_greenland_override.cfg`)
    - `GPISM_NP` (if set, runs under `mpirun -np`)

- `run_gpu_smoke.sh`
  - Runs the `gpism-timestep-gpu-smoke` binary to validate the device-resident
    hot loop and provide a baseline runtime.

## Notes

- These scripts are intended for **repeatable benchmarks**; if you change solver
  defaults or tuning, update `std_greenland_override.cfg` and re-record results
  in `gpism/docs/performance.md`.
