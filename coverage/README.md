# Coverage Flow

This folder adds a dedicated coverage-oriented testbench on top of the existing
`rtl/` DUT and the reusable verification support classes in
`verification/bnn_fcc_tb_pkg.sv`.

## Files

* `bnn_fcc_cov_pkg.sv`
  Functional coverage collector for:
  * AXI config/data gap patterns
  * message ordering and configuration scope
  * weight-density / threshold-range diversity
  * output class, repeated-vs-changing classes, and backpressure timing
  * reset phase, reset boundary, and post-reset reconfiguration behavior
* `bnn_fcc_coverage_tb.sv`
  Scenario-driven coverage testbench that reuses the existing DUT interface and
  model/stimulus package, loads the trained `784->256->256->10` MNIST model,
  and executes a compact suite of directed tests.
* `run_coverage.sh`
  Questa/ModelSim runner that compiles the harness, runs one or more scenarios,
  saves per-test UCDB files, merges them, and emits text coverage reports.

## Scenario Map

The shell runner and the `SCENARIO_ID` parameter use the following scenario ids:

* `0 full_reordered`
  Full configuration with reordered message delivery, all 10 output classes,
  repeated and changing classes, and all output-ready style/relation
  combinations.
* `1 weights_only_reconfig`
  Full configuration followed by a reset and a weights-only reconfiguration.
* `2 thresholds_only_reconfig`
  Full configuration followed by a reset and a thresholds-only reconfiguration.
* `3 partial_subset_reconfig`
  Full configuration followed by a reset and a partial hidden-layer
  weights+thresholds update.
* `4 reset_mid_config`
  Resets asserted both mid-message and on a config `TLAST` boundary.
* `5 reset_mid_image_and_output`
  Resets asserted mid-image, on an image `TLAST` boundary, and while an output
  is pending.

## Running

From the repository root:

```bash
./coverage/run_coverage.sh --clean
```

Run a single scenario:

```bash
./coverage/run_coverage.sh --scenario 0
./coverage/run_coverage.sh --scenario reset_mid_config
```

List scenarios:

```bash
./coverage/run_coverage.sh --list
```

Generated artifacts land in `coverage/results/`:

* per-scenario logs and UCDBs
* `merged.ucdb`
* `coverage_summary.txt`
* `coverage_details.txt`

## Notes

* The harness intentionally stays additive; it does not replace the original
  `verification/bnn_fcc_tb.sv` flow.
* Output checking is strict: every observed output handshake must match a queued
  expectation, so unexpected DUT outputs now fail the scenario instead of being
  ignored.
