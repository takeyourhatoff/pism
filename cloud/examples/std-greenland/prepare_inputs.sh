#!/usr/bin/env bash
set -euo pipefail

STACK_NAME=${STACK_NAME:-pism-batch}
PREFIX=${PREFIX:-std-greenland}
WORKDIR=${WORKDIR:-""}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PREPROCESS_SRC="${REPO_ROOT}/examples/std-greenland/preprocess.sh"

if [[ ! -f "${PREPROCESS_SRC}" ]]; then
  echo "preprocess.sh not found at ${PREPROCESS_SRC}" >&2
  exit 1
fi

if [[ -z "${WORKDIR}" ]]; then
  WORKDIR=$(mktemp -d /tmp/pism-std-greenland-XXXXXX)
  CLEANUP=1
else
  mkdir -p "${WORKDIR}"
  CLEANUP=0
fi

cp "${PREPROCESS_SRC}" "${WORKDIR}/preprocess.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to run preprocess.sh" >&2
  exit 1
fi

docker run --rm \
  -v "${WORKDIR}:/work" \
  -w /work \
  ubuntu:22.04 \
  bash -lc "DEBIAN_FRONTEND=noninteractive apt-get update && \
           apt-get install -y --no-install-recommends wget nco ca-certificates && \
           chmod +x preprocess.sh && ./preprocess.sh"

pism-cloud upload --stack-name "${STACK_NAME}" --source "${WORKDIR}/pism_Greenland_5km_v1.1.nc" --prefix "${PREFIX}"
pism-cloud upload --stack-name "${STACK_NAME}" --source "${WORKDIR}/pism_dT.nc" --prefix "${PREFIX}"
pism-cloud upload --stack-name "${STACK_NAME}" --source "${WORKDIR}/pism_dSL.nc" --prefix "${PREFIX}"

echo "Prepared std-greenland inputs under prefix '${PREFIX}' in the stack input bucket."

if [[ ${CLEANUP} -eq 1 ]]; then
  rm -rf "${WORKDIR}"
fi
