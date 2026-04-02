#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./coverage/run_coverage.sh [options]

Compile the RTL, verification support files, and coverage testbench, then run
one or more coverage scenarios and emit UCDB/text reports under coverage/results.

Options:
  --clean              Remove coverage/work and coverage/results before running.
  --list               Print the available scenarios and exit.
  --scenario <value>   Run a single scenario by numeric id or scenario name.
                       Repeat to run multiple specific scenarios.
  --help               Show this help text.

Examples:
  ./coverage/run_coverage.sh --clean
  ./coverage/run_coverage.sh --scenario 0
  ./coverage/run_coverage.sh --scenario full_reordered --scenario reset_mid_config
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null

SCENARIO_NAMES=(
  "full_reordered"
  "weights_only_reconfig"
  "thresholds_only_reconfig"
  "partial_subset_reconfig"
  "reset_mid_config"
  "reset_mid_image_and_output"
)

clean=0
declare -a requested_scenarios=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean=1
      shift
      ;;
    --list)
      for idx in "${!SCENARIO_NAMES[@]}"; do
        printf "%s %s\n" "$idx" "${SCENARIO_NAMES[$idx]}"
      done
      exit 0
      ;;
    --scenario)
      requested_scenarios+=("${2:?--scenario requires an id or name}")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_scenario_id() {
  local token="$1"
  local idx

  if [[ "$token" =~ ^[0-9]+$ ]]; then
    if (( token < ${#SCENARIO_NAMES[@]} )); then
      printf "%s\n" "$token"
      return 0
    fi
    return 1
  fi

  for idx in "${!SCENARIO_NAMES[@]}"; do
    if [[ "${SCENARIO_NAMES[$idx]}" == "$token" ]]; then
      printf "%s\n" "$idx"
      return 0
    fi
  done
  return 1
}

declare -a scenario_ids=()
if (( ${#requested_scenarios[@]} == 0 )); then
  for idx in "${!SCENARIO_NAMES[@]}"; do
    scenario_ids+=("$idx")
  done
else
  for token in "${requested_scenarios[@]}"; do
    if ! resolved="$(resolve_scenario_id "$token")"; then
      echo "Unknown scenario: $token" >&2
      exit 2
    fi
    scenario_ids+=("$resolved")
  done
fi

cd "${SCRIPT_DIR}"

if (( clean )); then
  rm -rf work results transcript vsim.wlf modelsim.ini
fi

mkdir -p results
rm -rf work
vlib work >/dev/null
vmap work work >/dev/null 2>&1

pkg_sources=(
  ../rtl/bnn_types_pkg.sv
  ../verification/bnn_fcc_tb_pkg.sv
  ../coverage/bnn_fcc_cov_pkg.sv
)

other_sources=(
  ../verification/axi4_stream_if.sv
  ../rtl/config_parser.sv
  ../rtl/layer_memory.sv
  ../rtl/neural_processor.sv
  ../rtl/compute_layer.sv
  ../rtl/bnn_fcc.sv
  ../coverage/bnn_fcc_coverage_tb.sv
)

vlog_args=(
  -work work
  -sv
  -mfcu
  +cover=bcesf
  +incdir+../rtl
  +incdir+../verification
  +incdir+../coverage
)

echo "[coverage] Compiling coverage harness..."
vlog "${vlog_args[@]}" "${pkg_sources[@]}" "${other_sources[@]}"

declare -a ucdbs=()

for scenario_id in "${scenario_ids[@]}"; do
  scenario_name="${SCENARIO_NAMES[$scenario_id]}"
  ucdb_path="results/${scenario_id}_${scenario_name}.ucdb"
  log_path="results/${scenario_id}_${scenario_name}.log"

  echo "[coverage] Running scenario ${scenario_id} (${scenario_name})..."
  vsim -c -coverage work.bnn_fcc_coverage_tb \
    -GSCENARIO_ID="${scenario_id}" \
    -do "coverage save -onexit ${ucdb_path}; run -all; quit -f" \
    | tee "${log_path}"

  ucdbs+=("${ucdb_path}")
done

merged_ucdb="results/merged.ucdb"
if (( ${#ucdbs[@]} == 1 )); then
  cp "${ucdbs[0]}" "${merged_ucdb}"
else
  echo "[coverage] Merging ${#ucdbs[@]} UCDB files..."
  vcover merge "${merged_ucdb}" "${ucdbs[@]}"
fi

echo "[coverage] Writing reports..."
vcover report -summary -cvg -code bcesf -output results/coverage_summary.txt "${merged_ucdb}"
vcover report -details -cvg -code bcesf -output results/coverage_details.txt "${merged_ucdb}"

echo "[coverage] Done."
echo "[coverage] Merged UCDB: ${SCRIPT_DIR}/results/merged.ucdb"
echo "[coverage] Summary report: ${SCRIPT_DIR}/results/coverage_summary.txt"
echo "[coverage] Detailed report: ${SCRIPT_DIR}/results/coverage_details.txt"
