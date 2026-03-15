#!/bin/bash

# Change to the script's directory so relative paths work reliably
cd "$(dirname "$0")"

echo "========================================"
echo "Compiling RTL and Testbench with vlog..."
echo "========================================"

# Clean stale compiled units (avoids picking up removed/renamed RTL files).
rm -rf work
vlib work

# Provide a simulation model for the Agilex/Stratix10 "Reset Release IP".
# Quartus synthesis supplies this via the IP catalog, but simulation needs a model.
IPROOT=""
if [ -n "${QUARTUS_ROOTDIR:-}" ] && [ -d "${QUARTUS_ROOTDIR}/../ip" ]; then
    IPROOT="${QUARTUS_ROOTDIR}/../ip"
elif [ -d "/home/yasir/altera_pro/25.3.1/ip" ]; then
    # Fallback for this environment.
    IPROOT="/home/yasir/altera_pro/25.3.1/ip"
fi

RR_SIM=""
if [ -n "${IPROOT}" ]; then
    RR_SIM="${IPROOT}/altera/pgm/altera_s10_user_rst_clkgate/altera_s10_user_rst_clkgate_sim.sv"
fi

if [ -n "${RR_SIM}" ] && [ -f "${RR_SIM}" ]; then
    vlog -sv "${RR_SIM}" || exit 1
else
    echo "ERROR: Reset Release IP sim model not found. Set QUARTUS_ROOTDIR or install Quartus IP libraries."
    echo "Expected: \${QUARTUS_ROOTDIR}/../ip/altera/pgm/altera_s10_user_rst_clkgate/altera_s10_user_rst_clkgate_sim.sv"
    exit 1
fi

# Compile DUT + testbench (package first).
vlog -sv ../rtl/bnn_types_pkg.sv \
         ../rtl/neural_processor.sv \
         ../rtl/config_parser.sv \
         ../rtl/layer_memory.sv \
         ../rtl/compute_layer.sv \
         ../rtl/bnn_fcc.sv \
         ../verification/bnn_fcc_tb_pkg.sv \
         ../verification/axi4_stream_if.sv \
         ../verification/bnn_fcc_tb.sv

if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "Running Simulation with vsim..."
echo "========================================"
LD_PRELOAD=/home/yasir/BNN/libfakemac.so vsim -c -gBASE_DIR="../python" -gNUM_TEST_IMAGES=10 -gDATA_IN_VALID_PROBABILITY=0.8 work.bnn_fcc_tb -do "run -all; quit"
