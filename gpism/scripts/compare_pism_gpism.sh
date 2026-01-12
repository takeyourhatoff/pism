#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPISM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PISM_ROOT="$(cd "${GPISM_ROOT}/.." && pwd)"

GPISM_BIN="${GPISM_BIN:-${GPISM_ROOT}/build-cuda-mpi/gpism}"
PISM_BIN="${PISM_BIN:-${PISM_ROOT}/build-pism/pism}"
PISM_CONFIG="${PISM_CONFIG:-${PISM_ROOT}/build-pism/pism_config.nc}"
YEARS="${YEARS:-4}"
OUTDIR="${OUTDIR:-/tmp/gpism_pism_compare_$(date +%Y%m%d_%H%M%S)}"

default_input="${PISM_ROOT}/examples/std-greenland/pism_Greenland_5km_v1.1.nc"
alt_input="${HOME}/pism/examples/std-greenland/pism_Greenland_5km_v1.1.nc"
if [[ -f "${default_input}" ]]; then
  INPUT="${INPUT:-${default_input}}"
elif [[ -f "${alt_input}" ]]; then
  INPUT="${INPUT:-${alt_input}}"
else
  INPUT="${INPUT:-}"
fi

if [[ -z "${INPUT}" || ! -f "${INPUT}" ]]; then
  echo "Missing input NetCDF. Set INPUT=... to pism_Greenland_5km_v1.1.nc" >&2
  exit 1
fi

if [[ ! -x "${GPISM_BIN}" ]]; then
  echo "gpism binary not found: ${GPISM_BIN}" >&2
  exit 1
fi

if [[ ! -x "${PISM_BIN}" ]]; then
  echo "pism binary not found: ${PISM_BIN}" >&2
  exit 1
fi

if [[ ! -f "${PISM_CONFIG}" ]]; then
  echo "pism_config.nc not found: ${PISM_CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

CAL_INPUT="${OUTDIR}/pism_Greenland_5km_v1.1_calendar.nc"
python3 - <<PY
import shutil
import netCDF4 as nc
src = "${INPUT}"
dst = "${CAL_INPUT}"
shutil.copyfile(src, dst)
with nc.Dataset(dst, "r+") as ds:
    if "time" in ds.variables:
        ds.variables["time"].calendar = "365_day"
    else:
        ds.createDimension("time", 1)
        t = ds.createVariable("time", "f8", ("time",))
        t[:] = 0.0
        t.units = "days since 0001-01-01"
        t.calendar = "365_day"
print(dst)
PY

GPISM_CFG="${OUTDIR}/gpism_compare.cfg"
cat > "${GPISM_CFG}" <<'EOF'
thermo.enabled=0
thickness.evolve=0
forcing.smb_constant=0
ssa.tauc_default=2e5
EOF

echo "Running gpism..."
GPISM_LOG="${OUTDIR}/gpism_run.log"
{ /usr/bin/time -f "gpism wall %e" "${GPISM_BIN}" \
    -i "${INPUT}" -o "${OUTDIR}/gpism_compare.nc" -y "${YEARS}" \
    -config_override "${GPISM_CFG}"; } 2>&1 | tee "${GPISM_LOG}"

echo "Running pism..."
PISM_LOG="${OUTDIR}/pism_run.log"
{ /usr/bin/time -f "pism wall %e" "${PISM_BIN}" \
    -bootstrap -i "${CAL_INPUT}" -o "${OUTDIR}/pism_compare.nc" -y "${YEARS}" \
    -config "${PISM_CONFIG}" -calendar 365_day \
    -surface given -stress_balance ssa -energy none -no_mass \
    -yield_stress constant -tauc 2e5 -ssa_method fd -o_size small \
    -grid.recompute_longitude_and_latitude false \
    -extra_file "${OUTDIR}/pism_compare_extra.nc" \
    -extra_vars usurf,uvel,vvel -extra_times yearly; } 2>&1 | tee "${PISM_LOG}"

echo "Comparing outputs..."
python3 - <<'PY'
import netCDF4 as nc
import numpy as np

gp = "${OUTDIR}/gpism_compare.nc"
pstate = "${OUTDIR}/pism_compare.nc"
pextra = "${OUTDIR}/pism_compare_extra.nc"

def last2d(var):
    data = var[:]
    if data.ndim == 3:
        return data[-1]
    if data.ndim == 2:
        return data
    return np.squeeze(data)

def depth_mean(var):
    data = var[:]
    if data.ndim == 4:
        return np.nanmean(data[-1], axis=2)
    if data.ndim == 3:
        return np.nanmean(data, axis=2)
    return np.squeeze(data)

with nc.Dataset(gp) as dg, nc.Dataset(pstate) as dp:
    for name in ("thk", "topg", "tauc"):
        if name in dg.variables and name in dp.variables:
            g = last2d(dg.variables[name])
            p = last2d(dp.variables[name])
            if g.shape != p.shape:
                print(f"{name}: shape mismatch {g.shape} vs {p.shape}")
                continue
            diff = g - p
            print(f"{name}: max|diff|={np.nanmax(np.abs(diff)):.6g}, rms={np.sqrt(np.nanmean(diff*diff)):.6g}")

with nc.Dataset(gp) as dg, nc.Dataset(pextra) as dp:
    if "usurf" in dg.variables and "usurf" in dp.variables:
        g = last2d(dg.variables["usurf"])
        p = last2d(dp.variables["usurf"])
        if g.shape == p.shape:
            diff = g - p
            print(f"usurf: max|diff|={np.nanmax(np.abs(diff)):.6g}, rms={np.sqrt(np.nanmean(diff*diff)):.6g}")
        else:
            print(f"usurf: shape mismatch {g.shape} vs {p.shape}")

    for name in ("uvel", "vvel"):
        if name in dg.variables and name in dp.variables:
            g = last2d(dg.variables[name])
            p = depth_mean(dp.variables[name])
            if g.shape != p.shape:
                print(f"{name}: shape mismatch {g.shape} vs {p.shape}")
                continue
            diff = g - p
            print(f"{name} (gpism - pism depth-mean): max|diff|={np.nanmax(np.abs(diff)):.6g}, rms={np.sqrt(np.nanmean(diff*diff)):.6g}")
PY

echo "Done. Outputs in ${OUTDIR}"
