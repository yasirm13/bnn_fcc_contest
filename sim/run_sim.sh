#!/bin/bash

# Change to the script's directory so relative paths work reliably
cd "$(dirname "$0")"

echo "========================================"
echo "Compiling RTL and Testbench with vlog..."
echo "========================================"
vlib work
vlog -sv ../rtl/*.sv \
         ../verification/bnn_fcc_tb_pkg.sv \
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
