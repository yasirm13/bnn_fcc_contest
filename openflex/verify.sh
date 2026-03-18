#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure ECE apps (Questa) are on PATH for the OpenFlex Questa flow.
# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null 2>&1 || true

cd "${SCRIPT_DIR}"

openflex bnn_fcc_verify.yml > run.log 2>&1

if grep -qiE 'error:|fatal:|failure:' run.log; then
    echo "Verification FAILED (see ${SCRIPT_DIR}/run.log)"
elif ! grep -q 'SUCCESS:' run.log; then
    echo "Verification FAILED (see ${SCRIPT_DIR}/run.log)"
else
    echo "Verification PASSED"
fi
