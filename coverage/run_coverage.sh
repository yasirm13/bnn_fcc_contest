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
  --random-sweeps <n>  Run n extra randomized custom-topology stress sweeps.
  --seed <value>       Base random seed for scenario 6 and topology sweeps.
  --help               Show this help text.

Examples:
  ./coverage/run_coverage.sh --clean
  ./coverage/run_coverage.sh --scenario 0
  ./coverage/run_coverage.sh --scenario full_reordered --scenario reset_mid_config
  ./coverage/run_coverage.sh --scenario randomized_stress --seed 12345
  ./coverage/run_coverage.sh --random-sweeps 3
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
  "randomized_stress"
)

clean=0
random_sweeps=0
random_seed=464384013
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
    --random-sweeps)
      random_sweeps="${2:?--random-sweeps requires a count}"
      shift 2
      ;;
    --seed)
      random_seed="${2:?--seed requires an integer value}"
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
  for idx in 0 1 2 3 4 5; do
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

if ! [[ "$random_sweeps" =~ ^[0-9]+$ ]]; then
  echo "--random-sweeps must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "$random_seed" =~ ^[0-9]+$ ]]; then
  echo "--seed must be a non-negative integer" >&2
  exit 2
fi

pick_random_value() {
  local low="$1"
  local high="$2"
  printf "%s\n" "$(( low + RANDOM % (high - low + 1) ))"
}

pick_partial_friendly_value() {
  local low="$1"
  local high="$2"
  local value
  value="$(pick_random_value "$low" "$high")"
  if (( value % 8 == 0 )); then
    if (( value < high )); then
      value=$(( value + 1 ))
    else
      value=$(( value - 1 ))
    fi
  fi
  printf "%s\n" "$value"
}

build_random_topology() {
  local layers
  local topology=()

  layers="$(pick_random_value 4 5)"
  topology+=("$(pick_partial_friendly_value 9 79)")
  for (( idx = 1; idx < layers - 1; idx++ )); do
    if (( idx == layers - 1 )); then
      :
    else
      topology+=("$(pick_random_value 5 32)")
    fi
  done
  topology+=("$(pick_random_value 2 15)")

  printf "%s\n" "${layers}:${topology[*]}"
}

emit_random_wrapper() {
  local wrapper_path="$1"
  local top_name="$2"
  local layers="$3"
  local seed="$4"
  shift 4
  local values=("$@")
  local topo_init="'{"

  for idx in "${!values[@]}"; do
    if (( idx > 0 )); then
      topo_init+=", "
    fi
    topo_init+="${values[$idx]}"
  done
  topo_init+="}"

  cat > "${wrapper_path}" <<EOF
module ${top_name};
    localparam int CUSTOM_LAYERS = ${layers};
    localparam int CUSTOM_TOPOLOGY[CUSTOM_LAYERS] = ${topo_init};

    bnn_fcc_coverage_tb #(
        .SCENARIO_ID(6),
        .USE_CUSTOM_TOPOLOGY(1'b1),
        .CUSTOM_LAYERS(CUSTOM_LAYERS),
        .CUSTOM_TOPOLOGY(CUSTOM_TOPOLOGY),
        .RANDOM_SEED(${seed})
    ) tb ();
endmodule
EOF
}

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
    -GRANDOM_SEED="$(( random_seed + scenario_id ))" \
    -do "coverage save -onexit ${ucdb_path}; run -all; quit -f" \
    | tee "${log_path}"

  ucdbs+=("${ucdb_path}")
done

if (( random_sweeps > 0 )); then
  for (( sweep_idx = 0; sweep_idx < random_sweeps; sweep_idx++ )); do
    sweep_seed=$(( random_seed + 1000 + sweep_idx ))
    topo_spec="$(build_random_topology)"
    sweep_layers="${topo_spec%%:*}"
    sweep_values_string="${topo_spec#*:}"
    read -r -a sweep_values <<< "${sweep_values_string}"

    top_name="bnn_fcc_random_cov_top_${sweep_idx}"
    wrapper_path="results/${top_name}.sv"
    ucdb_path="results/${top_name}.ucdb"
    log_path="results/${top_name}.log"

    emit_random_wrapper "${wrapper_path}" "${top_name}" "${sweep_layers}" "${sweep_seed}" \
      "${sweep_values[@]}"
    vlog "${vlog_args[@]}" "${wrapper_path}"

    echo "[coverage] Running random topology sweep ${sweep_idx} seed=${sweep_seed} topology=${sweep_values[*]}..."
    vsim -c -coverage "work.${top_name}" \
      -do "coverage save -onexit ${ucdb_path}; run -all; quit -f" \
      | tee "${log_path}"

    ucdbs+=("${ucdb_path}")
  done
fi

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
