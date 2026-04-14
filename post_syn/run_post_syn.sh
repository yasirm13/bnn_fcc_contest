#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./post_syn/run_post_syn.sh [--clean] [--dcp <path>] [--xdc <path>|--no-xdc] [--out-dir <path>] [--jobs <n>]

Runs Vivado implementation (opt/place/route) starting from a post-synthesis
checkpoint (.dcp) and writes checkpoints + reports to the output directory.

Defaults:
  --dcp     openflex/build_vivado/outputs/post_synth.dcp
  --xdc     openflex/build_vivado/vivado.xdc
  --out-dir post_syn/outputs
  --jobs    8

Notes:
  - This script sources ./activate.sh to put Vivado on PATH.
  - If you generated the post-synth DCP via ./syn/run_syn.sh (OpenFlex),
    the default --dcp path should already exist.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POST_SYN_DIR="${ROOT_DIR}/post_syn"

dcp_rel="openflex/build_vivado/outputs/post_synth.dcp"
xdc_rel="openflex/build_vivado/vivado.xdc"
out_rel="post_syn/outputs"
jobs="8"
clean=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean=1
      shift
      ;;
    --dcp)
      dcp_rel="${2:?--dcp requires a path}"
      shift 2
      ;;
    --xdc)
      xdc_rel="${2:?--xdc requires a path}"
      shift 2
      ;;
    --no-xdc)
      xdc_rel=""
      shift
      ;;
    --out-dir)
      out_rel="${2:?--out-dir requires a path}"
      shift 2
      ;;
    --jobs)
      jobs="${2:?--jobs requires an integer}"
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

mkdir -p "${POST_SYN_DIR}"

out_abs="${ROOT_DIR}/${out_rel}"
dcp_abs="${ROOT_DIR}/${dcp_rel}"

if [[ -z "${xdc_rel}" ]]; then
  xdc_abs=""
else
  xdc_abs="${ROOT_DIR}/${xdc_rel}"
fi

if (( clean )); then
  rm -rf "${out_abs}" "${POST_SYN_DIR}/run.log" "${POST_SYN_DIR}/run.jou" "${POST_SYN_DIR}/run_post_syn_impl.tcl" 2>/dev/null || true
fi

if [[ ! -f "${dcp_abs}" ]]; then
  echo "[run_post_syn] DCP not found: ${dcp_abs}" >&2
  exit 1
fi

if [[ -n "${xdc_abs}" && ! -f "${xdc_abs}" ]]; then
  echo "[run_post_syn] XDC not found: ${xdc_abs}" >&2
  exit 1
fi

if ! [[ "${jobs}" =~ ^[0-9]+$ ]] || [[ "${jobs}" -lt 1 ]]; then
  echo "[run_post_syn] --jobs must be a positive integer (got: ${jobs})" >&2
  exit 2
fi

# Vivado path (ECE apps)
# shellcheck source=/dev/null
source "${ROOT_DIR}/activate.sh" >/dev/null

if ! command -v vivado >/dev/null 2>&1; then
  echo "[run_post_syn] vivado not found on PATH after sourcing activate.sh" >&2
  exit 1
fi

mkdir -p "${out_abs}"

tcl_path="${POST_SYN_DIR}/run_post_syn_impl.tcl"
cat > "${tcl_path}" <<'TCL'
set dcp_path [lindex $argv 0]
set xdc_path [lindex $argv 1]
set out_dir  [lindex $argv 2]
set jobs     [lindex $argv 3]

file mkdir $out_dir

puts "----------------------------------------"
puts " Post-Synth Implementation (opt/place/route)"
puts "----------------------------------------"
puts "DCP:  $dcp_path"
puts "XDC:  $xdc_path"
puts "OUT:  $out_dir"
puts "JOBS: $jobs"

set_param general.maxThreads $jobs

open_checkpoint $dcp_path
if { $xdc_path ne "" } {
  read_xdc $xdc_path
}

opt_design
write_checkpoint -force "$out_dir/post_opt.dcp"

place_design
write_checkpoint -force "$out_dir/post_place.dcp"
report_utilization -file "$out_dir/post_place_util.rpt"
report_timing_summary -file "$out_dir/post_place_timing_summary.rpt"

route_design
write_checkpoint -force "$out_dir/post_route.dcp"
report_route_status -file "$out_dir/post_route_status.rpt"
report_timing_summary -file "$out_dir/post_route_timing_summary.rpt"
report_power -file "$out_dir/post_route_power.rpt"
report_drc -file "$out_dir/post_imp_drc.rpt"

puts "----------------------------------------"
puts " Done"
puts "----------------------------------------"

quit
TCL

pushd "${POST_SYN_DIR}" >/dev/null

echo "[run_post_syn] Running Vivado batch implementation..."
echo "[run_post_syn] DCP: ${dcp_abs}"
if [[ -n "${xdc_abs}" ]]; then
  echo "[run_post_syn] XDC: ${xdc_abs}"
else
  echo "[run_post_syn] XDC: (disabled)"
fi
echo "[run_post_syn] OUT: ${out_abs}"
echo "[run_post_syn] JOBS: ${jobs}"

vivado -mode batch -source "${tcl_path}" -tclargs "${dcp_abs}" "${xdc_abs}" "${out_abs}" "${jobs}" -log "run.log" -journal "run.jou"

popd >/dev/null

if [[ -f "${out_abs}/post_route_timing_summary.rpt" ]]; then
  echo "[run_post_syn] Implementation completed."
  echo "[run_post_syn] Outputs: ${out_abs}"
  echo "[run_post_syn] Log: ${POST_SYN_DIR}/run.log"
else
  echo "[run_post_syn] Implementation did not produce expected reports. See: ${POST_SYN_DIR}/run.log" >&2
  exit 1
fi

