#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPISM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PISM_ROOT_DEFAULT="$(cd "${GPISM_ROOT}/.." && pwd)"
if [[ -d "${PISM_ROOT_DEFAULT}/pism" ]]; then
  PISM_ROOT="${PISM_ROOT:-${PISM_ROOT_DEFAULT}/pism}"
elif [[ -d "${PISM_ROOT_DEFAULT}/../pism" ]]; then
  PISM_ROOT="${PISM_ROOT:-${PISM_ROOT_DEFAULT}/../pism}"
else
  PISM_ROOT="${PISM_ROOT:-${PISM_ROOT_DEFAULT}}"
fi

GPISM_BIN="${GPISM_BIN:-${GPISM_ROOT}/build-cuda-mpi/gpism}"
PISM_BIN="${PISM_BIN:-${PISM_ROOT}/build-pism/pism}"
PISM_CONFIG="${PISM_CONFIG:-${PISM_ROOT}/build-pism/pism_config.nc}"
YEARS="${YEARS:-4}"
MIN_WALL="${MIN_WALL:-10}"
SCALE_FACTOR="${SCALE_FACTOR:-2}"
GPISM_DT="${GPISM_DT:-60}"
OUTDIR="${OUTDIR:-/tmp/gpism_pism_compare_$(date +%Y%m%d_%H%M%S)}"
COMPARE_STRICT="${COMPARE_STRICT:-0}"
COMPARE_TOL_RMS="${COMPARE_TOL_RMS:-0.02}"
COMPARE_TOL_MAX="${COMPARE_TOL_MAX:-0.05}"
COMPARE_REQUIRE_NONZERO="${COMPARE_REQUIRE_NONZERO:-1}"
COMPARE_ALLOW_MISSING="${COMPARE_ALLOW_MISSING:-0}"

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
  if [[ "${COMPARE_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping compare (COMPARE_ALLOW_MISSING=1)."
    exit 0
  fi
  exit 1
fi

if [[ ! -x "${GPISM_BIN}" ]]; then
  echo "gpism binary not found: ${GPISM_BIN}" >&2
  if [[ "${COMPARE_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping compare (COMPARE_ALLOW_MISSING=1)."
    exit 0
  fi
  exit 1
fi

if [[ ! -x "${PISM_BIN}" ]]; then
  echo "pism binary not found: ${PISM_BIN}" >&2
  if [[ "${COMPARE_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping compare (COMPARE_ALLOW_MISSING=1)."
    exit 0
  fi
  exit 1
fi

if [[ ! -f "${PISM_CONFIG}" ]]; then
  echo "pism_config.nc not found: ${PISM_CONFIG}" >&2
  if [[ "${COMPARE_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping compare (COMPARE_ALLOW_MISSING=1)."
    exit 0
  fi
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
GPISM_PISM_CFG="${OUTDIR}/gpism_pism_params.cfg"
PISM_PARAM_LOG="${OUTDIR}/pism_ssa_params.txt"

python3 - <<PY
import netCDF4 as nc

path = "${PISM_CONFIG}"
ds = nc.Dataset(path)
var = ds.variables["pism_config"]

def get(key, default=None):
    return getattr(var, key) if hasattr(var, key) else default

params = {
    "stress_balance.ssa.flow_law": get("stress_balance.ssa.flow_law", "isothermal_glen"),
    "stress_balance.ssa.Glen_exponent": get("stress_balance.ssa.Glen_exponent", 3.0),
    "stress_balance.ssa.enhancement_factor": get("stress_balance.ssa.enhancement_factor", 1.0),
    "stress_balance.ssa.epsilon": get("stress_balance.ssa.epsilon", 1.0e13),
    "stress_balance.ssa.compute_surface_gradient_inward": get(
        "stress_balance.ssa.compute_surface_gradient_inward", "no"
    ),
    "stress_balance.ssa.fd.upstream_surface_slope_approximation": get(
        "stress_balance.ssa.fd.upstream_surface_slope_approximation", "yes"
    ),
    "basal_resistance.pseudo_plastic.enabled": get(
        "basal_resistance.pseudo_plastic.enabled", "no"
    ),
    "basal_resistance.pseudo_plastic.q": get("basal_resistance.pseudo_plastic.q", 0.25),
    "basal_resistance.pseudo_plastic.u_threshold": get(
        "basal_resistance.pseudo_plastic.u_threshold", 100.0
    ),
    "basal_resistance.pseudo_plastic.sliding_scale_factor": get(
        "basal_resistance.pseudo_plastic.sliding_scale_factor", -1.0
    ),
    "basal_resistance.plastic.regularization": get(
        "basal_resistance.plastic.regularization", 0.01
    ),
    "basal_resistance.beta_ice_free_bedrock": get(
        "basal_resistance.beta_ice_free_bedrock", 1.8e9
    ),
    "flow_law.isothermal_Glen.ice_softness": get(
        "flow_law.isothermal_Glen.ice_softness", 3.1689e-24
    ),
    "flow_law.Schoof_regularizing_velocity": get(
        "flow_law.Schoof_regularizing_velocity", 1.0
    ),
    "flow_law.Schoof_regularizing_length": get(
        "flow_law.Schoof_regularizing_length", 1000.0
    ),
    "constants.ice.density": get("constants.ice.density", 910.0),
    "constants.sea_water.density": get("constants.sea_water.density", 1028.0),
    "constants.standard_gravity": get("constants.standard_gravity", 9.81),
    "constants.sea_level": get("constants.sea_level", 0.0),
}

with open("${GPISM_PISM_CFG}", "w") as f:
    for k, v in params.items():
        if isinstance(v, str):
            if v.lower() in ("yes", "no"):
                f.write(f"{k}={'1' if v.lower() == 'yes' else '0'}\n")
            else:
                f.write(f"{k}={v}\n")
        else:
            f.write(f"{k}={v}\n")

with open("${PISM_PARAM_LOG}", "w") as f:
    for k, v in params.items():
        f.write(f"{k} = {v}\n")

print("PISM SSA params:")
for k, v in params.items():
    print(f"  {k} = {v}")
PY

write_gpism_cfg() {
  local years="$1"
  cat > "${GPISM_CFG}" <<EOF
$(cat "${GPISM_PISM_CFG}")
thermo.enabled=0
thickness.evolve=0
forcing.smb_constant=0
ssa.tauc_default=2e5
time.output_interval=${years}
time.dt=${GPISM_DT}
EOF
}

echo "Stage 1: gpism run (target wall >= ${MIN_WALL}s)..."
GPISM_LOG="${OUTDIR}/gpism_run.log"
GPISM_WALL="${OUTDIR}/gpism_wall.txt"
YEARS_ACTUAL="${YEARS}"
while true; do
  write_gpism_cfg "${YEARS_ACTUAL}"
  { /usr/bin/time -f "%e" -o "${GPISM_WALL}" "${GPISM_BIN}" \
      -i "${INPUT}" -o "${OUTDIR}/gpism_60.nc" -y "${YEARS_ACTUAL}" \
      -config_override "${GPISM_CFG}"; } 2>&1 | tee "${GPISM_LOG}"
  wall=$(cat "${GPISM_WALL}")
  echo "gpism wall ${wall}s (years=${YEARS_ACTUAL})"
  if python3 - <<PY
import sys
wall=float("${wall}")
min_wall=float("${MIN_WALL}")
sys.exit(0 if wall >= min_wall else 1)
PY
  then
    break
  fi
  YEARS_ACTUAL=$(python3 - <<PY
years=float("${YEARS_ACTUAL}")
factor=float("${SCALE_FACTOR}")
print(f"{years*factor:g}")
PY
  )
done

GPISM_YEARS="${YEARS_ACTUAL}"
GPISM_WALL_SECS=$(cat "${GPISM_WALL}")
GPISM_YEARS_PER_SEC=$(python3 - <<PY
years=float("${GPISM_YEARS}")
wall=float("${GPISM_WALL_SECS}")
print("{:.6g}".format(years / wall))
PY
)
echo "gpism years/sec: ${GPISM_YEARS_PER_SEC} (years=${GPISM_YEARS}, wall=${GPISM_WALL_SECS}s)"

echo "Stage 2: pism run (target wall >= ${MIN_WALL}s)..."
PISM_LOG="${OUTDIR}/pism_run.log"
PISM_WALL="${OUTDIR}/pism_wall.txt"
YEARS_ACTUAL="${YEARS}"
while true; do
  { /usr/bin/time -f "%e" -o "${PISM_WALL}" "${PISM_BIN}" \
      -bootstrap -i "${CAL_INPUT}" -o "${OUTDIR}/pism_60.nc" -y "${YEARS_ACTUAL}" \
      -config "${PISM_CONFIG}" -calendar 365_day \
      -surface given -stress_balance ssa -energy none -no_mass \
      -yield_stress constant -tauc 2e5 -ssa_method fd -o_size small \
      -grid.recompute_longitude_and_latitude false \
      -extra_file "${OUTDIR}/pism_60_extra.nc" \
      -extra_vars usurf -extra_times "${YEARS_ACTUAL}"; } 2>&1 | tee "${PISM_LOG}"
  wall=$(cat "${PISM_WALL}")
  echo "pism wall ${wall}s (years=${YEARS_ACTUAL})"
  if python3 - <<PY
import sys
wall=float("${wall}")
min_wall=float("${MIN_WALL}")
sys.exit(0 if wall >= min_wall else 1)
PY
  then
    break
  fi
  YEARS_ACTUAL=$(python3 - <<PY
years=float("${YEARS_ACTUAL}")
factor=float("${SCALE_FACTOR}")
print(f"{years*factor:g}")
PY
  )
done

PISM_YEARS="${YEARS_ACTUAL}"
PISM_WALL_SECS=$(cat "${PISM_WALL}")
PISM_YEARS_PER_SEC=$(python3 - <<PY
years=float("${PISM_YEARS}")
wall=float("${PISM_WALL_SECS}")
print("{:.6g}".format(years / wall))
PY
)
echo "pism years/sec: ${PISM_YEARS_PER_SEC} (years=${PISM_YEARS}, wall=${PISM_WALL_SECS}s)"
SPEEDUP_YEARS_PER_SEC=$(python3 - <<PY
gp=float("${GPISM_YEARS_PER_SEC}")
p=float("${PISM_YEARS_PER_SEC}")
print("{:.6g}".format(gp / p))
PY
)
echo "years/sec speedup (gpism/pism): ${SPEEDUP_YEARS_PER_SEC}"

echo "Stage 3: gpism run matching pism years=${PISM_YEARS}..."
write_gpism_cfg "${PISM_YEARS}"
GPISM_MATCH_LOG="${OUTDIR}/gpism_match.log"
GPISM_MATCH_WALL="${OUTDIR}/gpism_match_wall.txt"
{ /usr/bin/time -f "%e" -o "${GPISM_MATCH_WALL}" "${GPISM_BIN}" \
    -i "${INPUT}" -o "${OUTDIR}/gpism_match.nc" -y "${PISM_YEARS}" \
    -config_override "${GPISM_CFG}"; } 2>&1 | tee "${GPISM_MATCH_LOG}"
GPISM_MATCH_WALL_SECS=$(cat "${GPISM_MATCH_WALL}")
echo "gpism match wall ${GPISM_MATCH_WALL_SECS}s (years=${PISM_YEARS})"

echo "Comparing outputs (pism vs gpism match)..."
python3 - <<PY
import netCDF4 as nc
import numpy as np

gp = "${OUTDIR}/gpism_match.nc"
pstate = "${OUTDIR}/pism_60.nc"
pextra = "${OUTDIR}/pism_60_extra.nc"
tol_rms = float("${COMPARE_TOL_RMS}")
tol_max = float("${COMPARE_TOL_MAX}")
strict = "${COMPARE_STRICT}" == "1"
require_nonzero = "${COMPARE_REQUIRE_NONZERO}" == "1"
fail = False

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

seconds_per_year = 31556926.0

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
            rms = np.sqrt(np.nanmean(diff * diff))
            rms_ref = np.sqrt(np.nanmean(p * p))
            max_abs = np.nanmax(np.abs(diff))
            max_ref = np.nanmax(np.abs(p))
            rel_rms = rms / rms_ref if rms_ref > 0 else 0.0
            rel_max = max_abs / max_ref if max_ref > 0 else 0.0
            print(f"usurf: max|diff|={max_abs:.6g}, rms={rms:.6g}, rel_rms={rel_rms:.3g}, rel_max={rel_max:.3g}")
            if strict and (rel_rms > tol_rms or rel_max > tol_max):
                fail = True
        else:
            print(f"usurf: shape mismatch {g.shape} vs {p.shape}")

with nc.Dataset(gp) as dg, nc.Dataset(pstate) as dp:
    for name in ("u_ssa", "v_ssa"):
        if name in dg.variables and name in dp.variables:
            g = last2d(dg.variables[name])  # gpism: m/year
            p = last2d(dp.variables[name]) * seconds_per_year  # pism: m/s -> m/year
            if g.shape != p.shape:
                print(f"{name}: shape mismatch {g.shape} vs {p.shape}")
                continue
            diff = g - p
            rms = np.sqrt(np.nanmean(diff * diff))
            rms_ref = np.sqrt(np.nanmean(p * p))
            max_abs = np.nanmax(np.abs(diff))
            max_ref = np.nanmax(np.abs(p))
            rel_rms = rms / rms_ref if rms_ref > 0 else 0.0
            rel_max = max_abs / max_ref if max_ref > 0 else 0.0
            print(f"{name}: max|diff|={max_abs:.6g}, rms={rms:.6g}, rel_rms={rel_rms:.3g}, rel_max={rel_max:.3g}")
            if require_nonzero and max_ref <= 0:
                print(f"{name}: reference max is zero; expected non-zero velocities")
                fail = True
            if strict and (rel_rms > tol_rms or rel_max > tol_max):
                fail = True

if strict and fail:
    raise SystemExit(2)
PY

echo "Done. Outputs in ${OUTDIR}"
