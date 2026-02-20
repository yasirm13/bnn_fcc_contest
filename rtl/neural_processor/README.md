# Neural Processor for BNN

A SystemVerilog implementation of a neural processing unit designed for Binary Neural Networks (BNN). This module performs efficient XNOR-popcount operations on input vectors and weights, accumulating results against a threshold to produce a binary output.

## Features
- **Parallel Processing**: Processes `N`-bit vectors per cycle (XNOR + popcount).
- **Synthesizable Design**: optimized parallel adder tree for popcount, compatible with FPGA/ASIC synthesis flows.
- **Flow Control**: FSM-based operation with `valid_in` (segment valid) and `last` (end-of-packet).
- **Verification**: Includes a randomized self-checking testbench (Questa/ModelSim `vsim`).
- **Yosys Compatible**: Written in SystemVerilog 2005 compatible syntax (supported by Yosys 0.9).

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `N`       | 8       | Width of input vector `x` and weight `w`. |
| `ACC_WIDTH`| 16     | Width of the internal accumulator. |

## Usage
### Simulation (Questa/ModelSim)
Prerequisites: `vlib`, `vlog`, `vsim`.

```bash
# Build and run the simulation
make run

# Clean build artifacts
make clean
```
The testbench performs:
1. Corner case tests (High match, Zero match).
2. Configurable number of randomized regression tests (default 10000) with self-checking.

### Synthesis (Yosys)
Prerequisites: `yosys`.

```bash
# Run synthesis script
make synth
```
This produces a netlist `synth_neural_processor.v` mapped to a generic gate library.

## Interface
| Signal | Dir | Width | Description |
|--------|-----|-------|-------------|
| `clk` | In | 1 | System clock. |
| `rst` | In | 1 | Active-high asynchronous reset. |
| `valid_in` | In | 1 | Input segment valid. In IDLE, starts a new accumulation. |
| `last` | In | 1 | Marks the final segment of the current accumulation. |
| `x` | In | N | Input activation vector. |
| `w` | In | N | Input weight vector. |
| `threshold` | In | ACC_WIDTH | Threshold for binary activation. |
| `y` | Out | 1 | Binary output result (1 if acc >= threshold). |
| `valid_out` | Out | 1 | Pulses high for 1 cycle when `y`/`popcount_out` are updated with the final result. |
| `popcount_out` | Out | ACC_WIDTH | Running sum (and final sum after `valid_out`). |

## Directory Structure
- `neural_processor.sv`: Main RTL source.
- `tb_neural_processor.sv`: Verification testbench.
- `Makefile`: Build automation.
- `synth.ys`: Yosys synthesis script.
- `work/`: Questa/ModelSim compiled library directory.
