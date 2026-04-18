#!/usr/bin/env bash
# Runs the module-level unit testbenches under Questa/ModelSim.
# - Builds each TB in an isolated directory under verification/.unit_sim/<tb_name>
# - Requires each sim log to contain a "SUCCESS:" banner to pass
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/verification/.unit_sim"

source "$ROOT_DIR/activate.sh" >/dev/null

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

run_tb() {
    local tb_name="$1"
    shift

    local work_dir="$BUILD_DIR/$tb_name"
    mkdir -p "$work_dir"

    pushd "$work_dir" >/dev/null
    vlib work >/dev/null
    vmap work work >/dev/null
    vlog -sv -work work "$@" > compile.log 2>&1
    vsim -c "work.${tb_name}" -do "run -all; quit -f" > sim.log 2>&1
    grep -q "SUCCESS:" sim.log
    popd >/dev/null

    echo "[PASS] $tb_name"
}

run_tb \
    config_parser_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/bnn_types_pkg.sv" \
    "$ROOT_DIR/rtl/config_parser.sv" \
    "$ROOT_DIR/verification/config_parser_unit_tb.sv"

run_tb \
    config_parser_unit_tb_32 \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/bnn_types_pkg.sv" \
    "$ROOT_DIR/rtl/config_parser.sv" \
    "$ROOT_DIR/verification/config_parser_unit_tb_32.sv"

run_tb \
    layer_memory_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/layer_memory.sv" \
    "$ROOT_DIR/verification/layer_memory_unit_tb.sv"

run_tb \
    layer_memory_edge_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/layer_memory.sv" \
    "$ROOT_DIR/verification/layer_memory_edge_unit_tb.sv"

run_tb \
    neural_processor_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/neural_processor.sv" \
    "$ROOT_DIR/verification/neural_processor_unit_tb.sv"

run_tb \
    neural_processor_edge_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/neural_processor.sv" \
    "$ROOT_DIR/verification/neural_processor_edge_unit_tb.sv"

run_tb \
    compute_layer_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/neural_processor.sv" \
    "$ROOT_DIR/rtl/compute_layer.sv" \
    "$ROOT_DIR/verification/compute_layer_unit_tb.sv"

run_tb \
    compute_layer_multichunk_unit_tb \
    "$ROOT_DIR/rtl/bnn_util_pkg.sv" \
    "$ROOT_DIR/rtl/neural_processor.sv" \
    "$ROOT_DIR/rtl/compute_layer.sv" \
    "$ROOT_DIR/verification/compute_layer_multichunk_unit_tb.sv"

echo "SUCCESS: all unit testbenches completed successfully."
