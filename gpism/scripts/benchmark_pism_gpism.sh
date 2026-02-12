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
YEARS="${YEARS:-0.05}"
MIN_WALL="${MIN_WALL:-120}"
SCALE_FACTOR="${SCALE_FACTOR:-2}"
GPISM_DT="${GPISM_DT:-0.05}"
PISM_MAX_DT="${PISM_MAX_DT:-${GPISM_DT}}"
OUTDIR="${OUTDIR:-/tmp/gpism_pism_bench_$(date +%Y%m%d_%H%M%S)}"
BENCH_ALLOW_MISSING="${BENCH_ALLOW_MISSING:-0}"

GPISM_MG_ENABLED="${GPISM_MG_ENABLED:-1}"
GPISM_ENFORCE_ICE_FREE_BC="${GPISM_ENFORCE_ICE_FREE_BC:-0}"
GPISM_SSA_DIAGNOSTIC="${GPISM_SSA_DIAGNOSTIC:-0}"
GPISM_GMRES_VERBOSE="${GPISM_GMRES_VERBOSE:-0}"
GPISM_SSA_MAX_PICARD="${GPISM_SSA_MAX_PICARD:-300}"
GPISM_SSA_TOL_NUH="${GPISM_SSA_TOL_NUH:-1e-4}"
GPISM_SSA_TOL_VEL="${GPISM_SSA_TOL_VEL:-0}"
GPISM_GMRES_MAX_ITER="${GPISM_GMRES_MAX_ITER:-400}"
GPISM_GMRES_TOL="${GPISM_GMRES_TOL:-1e-5}"
GPISM_GMRES_TOL_RELATIVE_TO_RHS="${GPISM_GMRES_TOL_RELATIVE_TO_RHS:-0}"
GPISM_SSA_FAIL_FAST="${GPISM_SSA_FAIL_FAST:-1}"
GPISM_SSA_FAIL_FAST_REQUIRE_CONVERGED="${GPISM_SSA_FAIL_FAST_REQUIRE_CONVERGED:-1}"
GPISM_SSA_FAIL_FAST_RESIDUAL_MAX="${GPISM_SSA_FAIL_FAST_RESIDUAL_MAX:-0}"
GPISM_SSA_VEL_RELAX="${GPISM_SSA_VEL_RELAX:-1.0}"
GPISM_SSA_NUH_RELAX="${GPISM_SSA_NUH_RELAX:-1.0}"
GPISM_SSA_INITIAL_GUESS_SPEED="${GPISM_SSA_INITIAL_GUESS_SPEED:-0}"

default_input="${PISM_ROOT}/examples/std-greenland/pism_Greenland_5km_v1.1.nc"
alt_input="${HOME}/pism/examples/std-greenland/pism_Greenland_5km_v1.1.nc"
if [[ -f "${default_input}" ]]; then
  INPUT="${INPUT:-${default_input}}"
elif [[ -f "${alt_input}" ]]; then
  INPUT="${INPUT:-${alt_input}}"
else
  INPUT="${INPUT:-}"
fi

require_path() {
  local path="$1"
  local description="$2"
  local mode="${3:-file}"
  if [[ "${mode}" == "exec" ]]; then
    if [[ -x "${path}" ]]; then
      return 0
    fi
  elif [[ -f "${path}" ]]; then
    return 0
  fi
  echo "${description} not found: ${path}" >&2
  if [[ "${BENCH_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping benchmark (BENCH_ALLOW_MISSING=1)."
    exit 0
  fi
  exit 1
}

if [[ -z "${INPUT}" ]]; then
  echo "Missing input NetCDF. Set INPUT=... to pism_Greenland_5km_v1.1.nc" >&2
  if [[ "${BENCH_ALLOW_MISSING}" == "1" ]]; then
    echo "Skipping benchmark (BENCH_ALLOW_MISSING=1)."
    exit 0
  fi
  exit 1
fi
require_path "${INPUT}" "Input NetCDF"
require_path "${GPISM_BIN}" "gpism binary" exec
require_path "${PISM_BIN}" "pism binary" exec
require_path "${PISM_CONFIG}" "pism_config.nc"

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

GPISM_CFG="${OUTDIR}/gpism_bench.cfg"
GPISM_PISM_CFG="${OUTDIR}/gpism_pism_params.cfg"

python3 - <<PY
import netCDF4 as nc

path = "${PISM_CONFIG}"
ds = nc.Dataset(path)
var = ds.variables["pism_config"]

def get(key, default=None):
    return getattr(var, key) if hasattr(var, key) else default

params = {
    "stress_balance.ssa.flow_law": "isothermal_glen",
    "stress_balance.ssa.Glen_exponent": get("stress_balance.ssa.Glen_exponent", 3.0),
    "stress_balance.ssa.enhancement_factor": get("stress_balance.ssa.enhancement_factor", 1.0),
    "stress_balance.ssa.epsilon": get("stress_balance.ssa.epsilon", 1.0e13),
    "stress_balance.ice_free_thickness_standard": get("stress_balance.ice_free_thickness_standard", 10.0),
    "stress_balance.ssa.compute_surface_gradient_inward": get("stress_balance.ssa.compute_surface_gradient_inward", "no"),
    "stress_balance.ssa.fd.upstream_surface_slope_approximation": get("stress_balance.ssa.fd.upstream_surface_slope_approximation", "yes"),
    "stress_balance.ssa.fd.extrapolate_at_margins": get("stress_balance.ssa.fd.extrapolate_at_margins", "true"),
    "stress_balance.ssa.strength_extension.constant_nu": get("stress_balance.ssa.strength_extension.constant_nu", 9.48680701906572e14),
    "stress_balance.ssa.strength_extension.min_thickness": get("stress_balance.ssa.strength_extension.min_thickness", 50.0),
    "basal_resistance.pseudo_plastic.enabled": get("basal_resistance.pseudo_plastic.enabled", "no"),
    "basal_resistance.pseudo_plastic.q": get("basal_resistance.pseudo_plastic.q", 0.25),
    "basal_resistance.pseudo_plastic.u_threshold": get("basal_resistance.pseudo_plastic.u_threshold", 100.0),
    "basal_resistance.pseudo_plastic.sliding_scale_factor": get("basal_resistance.pseudo_plastic.sliding_scale_factor", -1.0),
    "basal_resistance.plastic.regularization": get("basal_resistance.plastic.regularization", 0.01),
    "basal_resistance.beta_ice_free_bedrock": get("basal_resistance.beta_ice_free_bedrock", 1.8e9),
    "basal_resistance.beta_lateral_margin": get("basal_resistance.beta_lateral_margin", 1.0e19),
    "flow_law.isothermal_Glen.ice_softness": get("flow_law.isothermal_Glen.ice_softness", 3.1689e-24),
    "flow_law.Schoof_regularizing_velocity": get("flow_law.Schoof_regularizing_velocity", 1.0),
    "flow_law.Schoof_regularizing_length": get("flow_law.Schoof_regularizing_length", 1000.0),
    "constants.ice.density": get("constants.ice.density", 910.0),
    "constants.sea_water.density": get("constants.sea_water.density", 1028.0),
    "constants.standard_gravity": get("constants.standard_gravity", 9.81),
    "constants.sea_level": get("constants.sea_level", 0.0),
}

with open("${GPISM_PISM_CFG}", "w") as f:
    for k, v in params.items():
        if isinstance(v, str):
            if v.lower() in ("yes", "no"):
                f.write(f"{k}={'1' if v.lower() == 'yes' else '0'}\\n")
            else:
                f.write(f"{k}={v}\\n")
        else:
            f.write(f"{k}={v}\\n")
PY

write_gpism_cfg() {
  local years="$1"
  cat > "${GPISM_CFG}" <<EOF
$(cat "${GPISM_PISM_CFG}")
ssa.mg.enabled=${GPISM_MG_ENABLED}
ssa.enforce_ice_free_bc=${GPISM_ENFORCE_ICE_FREE_BC}
ssa.diagnostic=${GPISM_SSA_DIAGNOSTIC}
ssa.gmres_verbose=${GPISM_GMRES_VERBOSE}
ssa.max_picard=${GPISM_SSA_MAX_PICARD}
ssa.tol_nuH=${GPISM_SSA_TOL_NUH}
ssa.tol_vel=${GPISM_SSA_TOL_VEL}
ssa.gmres_max_iter=${GPISM_GMRES_MAX_ITER}
ssa.gmres_tol=${GPISM_GMRES_TOL}
ssa.gmres_tol_relative_to_rhs=${GPISM_GMRES_TOL_RELATIVE_TO_RHS}
ssa.fail_fast=${GPISM_SSA_FAIL_FAST}
ssa.fail_fast_require_converged=${GPISM_SSA_FAIL_FAST_REQUIRE_CONVERGED}
ssa.fail_fast_residual_max=${GPISM_SSA_FAIL_FAST_RESIDUAL_MAX}
ssa.fail_fast_dump_prefix=${OUTDIR}
ssa.vel_relax=${GPISM_SSA_VEL_RELAX}
ssa.nuH_relax=${GPISM_SSA_NUH_RELAX}
ssa.initial_guess_speed=${GPISM_SSA_INITIAL_GUESS_SPEED}
thermo.enabled=0
thickness.evolve=0
forcing.smb_constant=0
ssa.tauc_default=2e5
time.output_interval=${years}
time.dt=${GPISM_DT}
EOF
}

echo "Stage 1: gpism benchmark (target wall >= ${MIN_WALL}s)..."
GPISM_LOG="${OUTDIR}/gpism_run.log"
GPISM_WALL="${OUTDIR}/gpism_wall.txt"
YEARS_ACTUAL="${YEARS}"
while true; do
  write_gpism_cfg "${YEARS_ACTUAL}"
  { /usr/bin/time -f "%e" -o "${GPISM_WALL}" "${GPISM_BIN}" \
      -i "${INPUT}" -o "${OUTDIR}/gpism.nc" -y "${YEARS_ACTUAL}" \
      -config_override "${GPISM_CFG}"; } 2>&1 | tee "${GPISM_LOG}"
  wall=$(cat "${GPISM_WALL}")
  echo "gpism wall ${wall}s (years=${YEARS_ACTUAL})"
  if python3 - <<PY
import sys
sys.exit(0 if float("${wall}") >= float("${MIN_WALL}") else 1)
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
GPISM_WALL_SECS="$(cat "${GPISM_WALL}")"
GPISM_YEARS_PER_SEC="$(python3 - <<PY
years=float("${GPISM_YEARS}")
wall=float("${GPISM_WALL_SECS}")
print("{:.6g}".format(years / wall))
PY
)"
echo "gpism years/sec: ${GPISM_YEARS_PER_SEC} (years=${GPISM_YEARS}, wall=${GPISM_WALL_SECS}s)"

echo "Stage 2: original PISM benchmark (target wall >= ${MIN_WALL}s)..."
PISM_LOG="${OUTDIR}/pism_run.log"
PISM_WALL="${OUTDIR}/pism_wall.txt"
YEARS_ACTUAL="${YEARS}"
while true; do
  { /usr/bin/time -f "%e" -o "${PISM_WALL}" "${PISM_BIN}" \
      -bootstrap -i "${CAL_INPUT}" -o "${OUTDIR}/pism.nc" -y "${YEARS_ACTUAL}" \
      -config "${PISM_CONFIG}" -calendar 365_day \
      -surface given -stress_balance ssa -energy none -no_mass \
      -stress_balance.ssa.flow_law isothermal_glen \
      -yield_stress constant -tauc 2e5 -ssa_method fd -o_size small \
      -max_dt "${PISM_MAX_DT}" \
      -grid.recompute_longitude_and_latitude false \
      -extra_file "${OUTDIR}/pism_extra.nc" \
      -extra_vars usurf -extra_times "${YEARS_ACTUAL}"; } 2>&1 | tee "${PISM_LOG}"
  wall=$(cat "${PISM_WALL}")
  echo "pism wall ${wall}s (years=${YEARS_ACTUAL})"
  if python3 - <<PY
import sys
sys.exit(0 if float("${wall}") >= float("${MIN_WALL}") else 1)
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
PISM_WALL_SECS="$(cat "${PISM_WALL}")"
PISM_YEARS_PER_SEC="$(python3 - <<PY
years=float("${PISM_YEARS}")
wall=float("${PISM_WALL_SECS}")
print("{:.6g}".format(years / wall))
PY
)"
echo "pism years/sec: ${PISM_YEARS_PER_SEC} (years=${PISM_YEARS}, wall=${PISM_WALL_SECS}s)"

SPEEDUP_YEARS_PER_SEC="$(python3 - <<PY
gp=float("${GPISM_YEARS_PER_SEC}")
p=float("${PISM_YEARS_PER_SEC}")
print("{:.6g}".format(gp / p))
PY
)"
echo "years/sec speedup (gpism/pism): ${SPEEDUP_YEARS_PER_SEC}"

cat > "${OUTDIR}/benchmark_summary.txt" <<EOF
gpism_years=${GPISM_YEARS}
gpism_wall_seconds=${GPISM_WALL_SECS}
gpism_years_per_sec=${GPISM_YEARS_PER_SEC}
pism_years=${PISM_YEARS}
pism_wall_seconds=${PISM_WALL_SECS}
pism_years_per_sec=${PISM_YEARS_PER_SEC}
speedup_gpism_over_pism=${SPEEDUP_YEARS_PER_SEC}
EOF

echo "Done. Outputs in ${OUTDIR}"
