# Makefile for Questa SystemVerilog simulation

# Check if vsim exists in PATH
ifeq (,$(shell which vsim))
$(error "vsim not found in PATH. Please ensure Questa is properly installed and added to PATH")
endif

# Tool and library configuration
VLOG = vlog
VSIM = vsim
VOPT = vopt

# Project configuration
WORK_DIR = work
TOP_MODULE = bnn_fcc_cat1_tb
OPTIMIZED_TOP = $(TOP_MODULE)_opt

# Compilation flags
VLOG_FLAGS = -sv \
	-mfcu \
	-lint \
	+acc=pr \
	-suppress 2275 \
	-timescale "1ns/1ps" \
	+incdir+../rtl \
	+incdir+../verification \
	-work $(WORK_DIR)

# Optimization flags (preserve full visibility with +acc)
VOPT_FLAGS = +acc \
	-o $(OPTIMIZED_TOP)

# Simulation flags
VSIM_FLAGS = -c \
	-debugDB \
	-voptargs="+acc" \
	-do "run -all"

# GUI simulation flags
VSIM_GUI_FLAGS = -gui \
	-debugDB \
	-voptargs="+acc"

# Default target
all: compile optimize

# Create work library
$(WORK_DIR):
	vlib $(WORK_DIR)
	vmap work $(WORK_DIR)

# Read sources from file and compile
compile: $(WORK_DIR)
	$(VLOG) $(VLOG_FLAGS) -f sources.txt

# Optimize design while maintaining full visibility
optimize: compile	
	$(VOPT) $(TOP_MODULE) $(VOPT_FLAGS)

# Run simulation in command-line mode
sim: optimize	
	$(VSIM) $(VSIM_FLAGS) $(OPTIMIZED_TOP)

# Open GUI for interactive simulation
gui: optimize	
	$(VSIM) $(VSIM_GUI_FLAGS) $(OPTIMIZED_TOP) &

# Clean up generated files
clean:
	rm -rf $(WORK_DIR)
	rm -rf transcript
	rm -rf vsim.wlf
	rm -rf *.db
	rm -rf *.dbg
	rm -rf *.vstf
	rm -rf modelsim.ini

.PHONY: all compile optimize sim gui clean
