#!/usr/bin/env bash
set -euo pipefail

VERIF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${VERIF_DIR}/.." && pwd)"
BUILD_DIR="${VERIF_DIR}/.coverage_sim"
TOP_MODULE="bnn_fcc_coverage_tb"
OPT_MODULE="${TOP_MODULE}_opt"
UCDB_PATH="${BUILD_DIR}/coverage.ucdb"

report_arg="${1:-coverage_report.txt}"
if [[ "${report_arg}" = /* ]]; then
    REPORT_PATH="${report_arg}"
else
    REPORT_PATH="${BUILD_DIR}/${report_arg}"
fi

source "${ROOT_DIR}/activate.sh" >/dev/null

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

vlib work >/dev/null
vmap work work >/dev/null 2>&1 || true

sources=(
    "${ROOT_DIR}/rtl/bnn_util_pkg.sv"
    "${ROOT_DIR}/rtl/bnn_types_pkg.sv"
    "${ROOT_DIR}/verification/bnn_fcc_tb_pkg.sv"
    "${ROOT_DIR}/verification/axi4_stream_if.sv"
    "${ROOT_DIR}/rtl/layer_memory.sv"
    "${ROOT_DIR}/rtl/config_parser.sv"
    "${ROOT_DIR}/rtl/compute_layer.sv"
    "${ROOT_DIR}/rtl/neural_processor.sv"
    "${ROOT_DIR}/rtl/bnn_fcc.sv"
    "${ROOT_DIR}/verification/bnn_fcc_coverage_tb.sv"
)

vlog_args=(
    -sv
    -mfcu
    -lint
    +acc=pr
    +cover
    -suppress 2275
    -timescale "1ns/1ps"
    +incdir+"${ROOT_DIR}/rtl"
    +incdir+"${ROOT_DIR}/verification"
    -work work
)

echo "[cov_check] Compiling coverage regression sources..."
vlog "${vlog_args[@]}" "${sources[@]}" > compile.log 2>&1

echo "[cov_check] Optimizing ${TOP_MODULE}..."
vopt "work.${TOP_MODULE}" +acc +cover -o "${OPT_MODULE}" > optimize.log 2>&1

mkdir -p "$(dirname "${REPORT_PATH}")"

echo "[cov_check] Running simulation..."
vsim -c -debugDB -coverage -voptargs=+acc "work.${OPT_MODULE}" \
    -do "run -all; coverage save -onexit ${UCDB_PATH}; coverage report -output ${REPORT_PATH} -srcfile=* -details; quit -f" \
    > sim.log 2>&1

if [[ ! -f "${REPORT_PATH}" ]]; then
    echo "Coverage report not found: ${REPORT_PATH}"
    exit 2
fi

covergroup_pct="$(
    awk '/TOTAL COVERGROUP COVERAGE:/ {print $4; exit}' "${REPORT_PATH}"
)"

if [[ -z "${covergroup_pct}" ]]; then
    echo "Could not parse coverage percentages from ${REPORT_PATH}"
    exit 2
fi

echo "Covergroup coverage: ${covergroup_pct}"
echo "Covergroups:"
awk '
    /^ TYPE / {
        name = $2
        pct = $3
        sub("^.*/", "", name)
        printf "  %s %s\n", name, pct
    }
' "${REPORT_PATH}"

if [[ "${covergroup_pct}" == "100.00%" ]]; then
    echo "All functional covergroup coverage is 100%."
    exit 0
fi

echo "Functional covergroup coverage is NOT 100%."
exit 1
