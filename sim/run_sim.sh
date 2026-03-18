#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./sim/run_sim.sh [options] [-- <extra vsim args...>]

Compiles all SystemVerilog/Verilog sources under ../rtl and ../verification,
then launches Questa/ModelSim (vsim) on the testbench top (default: bnn_fcc_tb).

Options:
  --gui                 Launch the GUI (default is command-line simulation).
  --clean               Remove ./work and common sim outputs before compiling.
  --top <name>          Top-level module to simulate (default: bnn_fcc_tb).
  -G<Name>=<Value>      Override a Verilog parameter at elaboration (repeatable).
  --vlog-arg <arg>      Extra argument to pass to vlog (repeatable).
  --help                Show this help.

Environment:
  VLOG_ARGS="..."       Extra args appended to vlog (whitespace-split).
  VSIM_ARGS="..."       Extra args appended to vsim (whitespace-split).

Examples:
  ./sim/run_sim.sh
  ./sim/run_sim.sh --gui
  ./sim/run_sim.sh --clean -GNUM_TEST_IMAGES=1 -GVERIFY_MODEL=0
  ./sim/run_sim.sh -- -do "log -r /*; run -all"
EOF
}

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SIM_DIR}/.." && pwd)"

# Make Questa tools available in this script process.
# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null

cd "${SIM_DIR}"

gui=0
clean=0
top="bnn_fcc_tb"
vlog_extra=()
vsim_params=()  # -G<Name>=<Value>...
vsim_extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gui)
      gui=1
      shift
      ;;
    --clean)
      clean=1
      shift
      ;;
    --top)
      top="${2:?--top requires a module name}"
      shift 2
      ;;
    -G*)
      vsim_params+=("$1")
      shift
      ;;
    --vlog-arg)
      vlog_extra+=("${2:?--vlog-arg requires a value}")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      vsim_extra+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( clean )); then
  rm -rf work transcript vsim.wlf modelsim.ini *.log *.wlf 2>/dev/null || true
fi

vlib work >/dev/null
vmap work work >/dev/null 2>&1 || true

shopt -s nullglob
sources=(
  ../rtl/*.sv
  ../rtl/*.v
  ../verification/*.sv
  ../verification/*.v
)
shopt -u nullglob

if (( ${#sources[@]} == 0 )); then
  echo "No HDL sources found under ../rtl or ../verification." >&2
  exit 1
fi

vlog_args=(
  -work work
  -sv
  -mfcu
  +incdir+../rtl
  +incdir+../verification
)

if [[ -n "${VLOG_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  vlog_args+=(${VLOG_ARGS})
fi
vlog_args+=("${vlog_extra[@]}")

echo "[run_sim] Compiling ${#sources[@]} files..."

# Compile packages first to avoid order-dependency issues (e.g. tb imports tb_pkg).
pkg_sources=()
other_sources=()
for f in "${sources[@]}"; do
  base="$(basename "$f")"
  if [[ "$base" == *pkg.sv || "$base" == *pkg.v ]]; then
    pkg_sources+=("$f")
  else
    other_sources+=("$f")
  fi
done

if (( ${#pkg_sources[@]} )); then
  vlog "${vlog_args[@]}" "${pkg_sources[@]}"
fi
vlog "${vlog_args[@]}" "${other_sources[@]}"

vsim_args=()
if [[ -n "${VSIM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  vsim_args+=(${VSIM_ARGS})
fi

echo "[run_sim] Launching vsim top=work.${top}"
if (( gui )); then
  vsim "${vsim_params[@]}" "${vsim_args[@]}" -voptargs=+acc "work.${top}" "${vsim_extra[@]}"
else
  vsim -c "${vsim_params[@]}" "${vsim_args[@]}" -voptargs=+acc "work.${top}" "${vsim_extra[@]}" -do "run -all; quit -f"
fi
