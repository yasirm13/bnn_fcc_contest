#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./syn/run_syn.sh [--clean] [--pass-name <name>] [--csv <path>] [--yml <path>]

Runs OpenFlex timing/area flow for this repo using the settings in the YAML.
This will invoke Vivado (synthesis + implementation) for the target device.

Defaults:
  --yml  openflex/bnn_fcc_timing.yml
  --csv  syn/bnn_fcc_timing_pass_<pass-name>.csv

Notes:
  - This script sources ./activate.sh to put Vivado on PATH.
  - If the `openflex` CLI is not found, it bootstraps a local Python 3.11 venv
    at ./.venv_openflex and installs `openflex` into it.
  - If `--csv` is not provided, the script generates a unique CSV name using
    the `_pass_<name>` suffix style. The default pass name is timestamp-based.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYN_DIR="${ROOT_DIR}/syn"
VENV_DIR="${ROOT_DIR}/.venv_openflex"

yml_rel="openflex/bnn_fcc_timing.yml"
csv_rel=""
pass_name="run"
clean=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean=1
      shift
      ;;
    --yml)
      yml_rel="${2:?--yml requires a path}"
      shift 2
      ;;
    --pass-name)
      pass_name="${2:?--pass-name requires a name}"
      shift 2
      ;;
    --csv)
      csv_rel="${2:?--csv requires a path}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${SYN_DIR}"

sanitize_pass_name() {
  local raw="${1:-run}"
  local sanitized

  sanitized="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')"
  sanitized="${sanitized##_}"
  sanitized="${sanitized%%_}"

  if [[ -z "${sanitized}" ]]; then
    sanitized="run"
  fi

  printf '%s\n' "${sanitized}"
}

pass_name="$(sanitize_pass_name "${pass_name}")"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${csv_rel}" ]]; then
  csv_rel="syn/timing_${pass_name}_${timestamp}.csv"
fi

if (( clean )); then
  rm -rf "${SYN_DIR}/run.log" "${ROOT_DIR}/${csv_rel}" "${SYN_DIR}/openflex_out" 2>/dev/null || true
  rm -rf "${ROOT_DIR}/openflex/run.log" "${ROOT_DIR}/openflex/bnn_fcc.csv" 2>/dev/null || true
fi

# Vivado path (ECE apps)
# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null

# Ensure OpenFlex is available (use local venv if needed).
if ! command -v openflex >/dev/null 2>&1; then
  if [[ ! -x "${VENV_DIR}/bin/activate" ]]; then
    echo "[run_syn] Creating venv at ${VENV_DIR}"
    python3.11 -m venv "${VENV_DIR}"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"

  if ! command -v openflex >/dev/null 2>&1; then
    echo "[run_syn] Installing openflex into venv (this may take a minute)..."
    python -m pip install -U pip >/dev/null
    python -m pip install openflex >/dev/null
  fi
fi

yml_abs="${ROOT_DIR}/${yml_rel}"
csv_abs="${ROOT_DIR}/${csv_rel}"

if [[ ! -f "${yml_abs}" ]]; then
  echo "[run_syn] YAML not found: ${yml_abs}" >&2
  exit 1
fi

# OpenFlex examples assume running from within the openflex/ folder so relative
# paths inside the YAML resolve as expected.
pushd "${ROOT_DIR}/openflex" >/dev/null

echo "[run_syn] Running: openflex $(basename "${yml_abs}") -c ${csv_abs}"
openflex "$(basename "${yml_abs}")" -c "${csv_abs}" > "${SYN_DIR}/run.log" 2>&1

popd >/dev/null

if [[ -f "${csv_abs}" ]]; then
  echo "[run_syn] Synthesis completed. Results: ${csv_abs}"
  echo "[run_syn] Log: ${SYN_DIR}/run.log"
else
  echo "[run_syn] Synthesis did not produce CSV output. See: ${SYN_DIR}/run.log" >&2
  exit 1
fi
