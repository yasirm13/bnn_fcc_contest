#!/bin/bash

echo "========================================"
echo "Compiling RTL and Testbench with vlog..."
echo "========================================"
vlog -sv ../rtl/*.sv ../verification/bnn_fcc_tb_pkg.sv ../verification/bnn_fcc_tb.sv

if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "Running Simulation with vsim..."
echo "========================================"
vsim -c -gBASE_DIR="../python" -gNUM_TEST_IMAGES=100 -gDATA_IN_VALID_PROBABILITY=0.8 work.bnn_fcc_tb -do "run -all; quit"
