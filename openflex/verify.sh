#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_openflex"
LOG_PATH="${SCRIPT_DIR}/run.log"

# Enable Vivado/Questa tools (ECE apps).
# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null

# Ensure OpenFlex is available (use local venv if needed).
if ! command -v openflex >/dev/null 2>&1; then
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "[verify] Creating venv at ${VENV_DIR}"
    python3.11 -m venv "${VENV_DIR}"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"

  if ! command -v openflex >/dev/null 2>&1; then
    echo "[verify] Installing openflex into venv (this may take a minute)..."
    python -m pip install -U pip >/dev/null
    python -m pip install openflex >/dev/null
  fi
fi

pushd "${SCRIPT_DIR}" >/dev/null
openflex bnn_fcc_verify.yml > "${LOG_PATH}" 2>&1
popd >/dev/null

if grep -qiE 'error:|fatal:|failure:' "${LOG_PATH}"; then
  echo "Verification FAILED (see ${LOG_PATH})"
elif ! grep -q 'SUCCESS:' "${LOG_PATH}"; then
  echo "Verification FAILED (see ${LOG_PATH})"
else
  echo "Verification PASSED"
fi
