// Supplemental coverage-oriented regression for the BNN FCC design.
//
// This testbench is intended for Apple-style functional coverage evaluation via
// `verification/cov_check.sh` and is separate from the standard contest
// end-to-end regression (`verification/bnn_fcc_tb.sv`).
//
// Key ideas:
// - Uses a small custom topology (13->10->10->5) so runs are fast, while
//   intentionally forcing partial final beats on the AXI4-Stream image input
//   (exercising `TKEEP`/`TLAST` handling).
// - Drives randomized valid/ready patterns and output backpressure to stress
//   AXI4-Stream handshakes and internal control.
// - Generates a "diverse" configuration (weight density + threshold ranges) and
//   directed stimulus so all output classes are hit for classification coverage.
// - Injects reconfiguration and reset scenarios at different phases of
//   operation to cover edge cases beyond steady-state streaming.
package bnn_fcc_coverage_pkg;
    import bnn_fcc_tb_pkg::*;

    // Extends the reference model with helpers for:
    // - building deterministic "diverse" weights/thresholds for coverage bins
    // - encoding config streams while randomizing header "don't-care" fields
    //   and `keep` patterns (partial last beats)
    class Extended_BNN_FCC_Model #(
        int BUS_WIDTH = 32
    ) extends BNN_FCC_Model #(BUS_WIDTH);
        // Weight "density" classes used by the config-diversity covergroup.
        localparam int DENSITY_ALL_ZERO = 0;
        localparam int DENSITY_SPARSE   = 1;
        localparam int DENSITY_BALANCED = 2;
        localparam int DENSITY_DENSE    = 3;
        localparam int DENSITY_ALL_ONE  = 4;

        function new();
            super.new();
        endfunction

        protected function automatic bit get_density_bit(input int density_mode, input int bit_idx, input int seed);
            case (density_mode)
                DENSITY_ALL_ZERO: return 1'b0;
                DENSITY_SPARSE:   return (((bit_idx * 7) + seed) % 8) < 2;
                DENSITY_BALANCED: return (((bit_idx + seed) % 2) == 0);
                DENSITY_DENSE:    return (((bit_idx * 5) + seed) % 8) != 0 && (((bit_idx * 5) + seed) % 8) != 1;
                DENSITY_ALL_ONE:  return 1'b1;
                default:          return 1'b0;
            endcase
        endfunction

        protected function automatic int choose_density_mode(input int layer_idx, input int neuron_idx);
            return (layer_idx + neuron_idx) % 5;
        endfunction

        protected function automatic int choose_ones_count(input int fan_in, input int density_mode);
            int sparse_count;
            int balanced_count;
            int dense_count;

            sparse_count   = ((fan_in + 3) / 4);
            if (sparse_count <= 0) sparse_count = 1;
            if ((sparse_count * 4) > fan_in) sparse_count = fan_in / 4;
            if (sparse_count <= 0) sparse_count = 1;

            balanced_count = fan_in / 2;
            dense_count    = ((fan_in * 3) + 3) / 4;
            if ((dense_count * 4) < (fan_in * 3)) dense_count++;
            if (dense_count >= fan_in) dense_count = fan_in - 1;

            case (density_mode)
                DENSITY_ALL_ZERO: return 0;
                DENSITY_SPARSE:   return sparse_count;
                DENSITY_BALANCED: return balanced_count;
                DENSITY_DENSE:    return dense_count;
                DENSITY_ALL_ONE:  return fan_in;
                default:          return balanced_count;
            endcase
        endfunction

        protected function automatic int choose_threshold_value(input int fan_in, input int layer_idx, input int neuron_idx);
            int threshold_class;
            int small_val;
            int medium_val;
            int large_val;

            threshold_class = (layer_idx + (2 * neuron_idx)) % 3;
            small_val  = (neuron_idx % 2 == 0) ? 0 : ((fan_in > 0) ? 1 : 0);
            medium_val = fan_in / 2;
            large_val  = (fan_in > 0) ? (fan_in - (neuron_idx % 2)) : 0;

            case (threshold_class)
                0: return small_val;
                1: return medium_val;
                default: return large_val;
            endcase
        endfunction

        // Construct a deterministic-but-diverse model:
        // - weights span multiple density regimes (all-0, sparse, balanced, dense, all-1)
        // - hidden-layer thresholds span small/medium/large ranges
        // - output-layer thresholds are unused (the DUT outputs raw popcounts), so set to 0
        function void create_diverse_config(int user_topology[]);
            this.topology   = user_topology;
            this.num_layers = user_topology.size() - 1;
            this.weight     = new[num_layers];
            this.threshold  = new[num_layers];

            for (int l = 0; l < num_layers; l++) begin
                int n_inputs;
                int n_neurons;

                n_inputs  = topology[l];
                n_neurons = topology[l+1];

                $display("Creating Diverse Layer %0d: %0d inputs -> %0d neurons", l, n_inputs, n_neurons);
                this.weight[l]    = new[n_neurons];
                this.threshold[l] = new[n_neurons];

                for (int n = 0; n < n_neurons; n++) begin
                    int density_mode;
                    int ones_target;
                    weight_row_t temp_w;

                    density_mode = choose_density_mode(l, n);
                    ones_target = choose_ones_count(n_inputs, density_mode);
                    temp_w = new[n_inputs];

                    for (int i = 0; i < n_inputs; i++) begin
                        temp_w[i] = 1'b0;
                    end

                    for (int i = 0; i < ones_target; i++) begin
                        int bit_pos;
                        bit_pos = ((i * 5) + (l * 11) + (n * 3)) % n_inputs;
                        temp_w[bit_pos] = 1'b1;
                    end

                    // Fill any collisions deterministically until the exact count is reached.
                    for (int i = 0; i < n_inputs; i++) begin
                        int current_ones;
                        current_ones = 0;
                        for (int j = 0; j < n_inputs; j++) begin
                            if (temp_w[j]) current_ones++;
                        end
                        if (current_ones >= ones_target) break;
                        if (!temp_w[i]) temp_w[i] = get_density_bit(density_mode, i, (l * 11) + (n * 3)) || 1'b1;
                    end

                    this.weight[l][n] = temp_w;

                    if (l == num_layers - 1) this.threshold[l][n] = 0;
                    else this.threshold[l][n] = choose_threshold_value(n_inputs, l, n);
                end
            end

            is_loaded     = 1;
            outputs_valid = 0;
        endfunction

        function void get_layer_config_randomized_fields(
            input int layer_idx,
            input bit is_threshold,
            output bus_stream_t stream,
            output keep_stream_t keep
        );
            // Build a single layer's configuration message as an AXI4-Stream word/keep stream.
            // For threshold messages, some header fields are intentionally randomized to
            // ensure the DUT ignores "don't-care" bits as expected.
            bit [7:0] byte_q[$];
            bus_word_t word_q[$];
            bus_keep_t keep_q[$];

            int fan_in;
            int n_neurons;
            int bytes_per_neuron;
            int layer_inputs;
            longint total_payload_bytes;
            bit [127:0] header_val;

            bus_word_t current_word;
            bus_keep_t current_keep;
            int bytes_per_beat;
            int byte_count;

            int w_idx;
            bit [7:0] byte_val;
            bit [31:0] t_val;

            fan_in = this.topology[layer_idx];
            n_neurons = this.topology[layer_idx+1];

            if (is_threshold) begin
                bytes_per_neuron = 4;
                // Some threshold header fields are not used by the DUT; toggle them to
                // create header diversity without changing functional thresholds.
                layer_inputs     = (layer_idx % 2 == 0) ? 16'd32 : $urandom_range(0, 16'hffff);
            end else begin
                bytes_per_neuron = (fan_in + 7) / 8;
                layer_inputs     = fan_in;
            end

            total_payload_bytes = bytes_per_neuron * n_neurons;
            header_val          = '0;

            header_val[07:00]   = (is_threshold) ? 8'd1 : 8'd0;
            header_val[15:08]   = layer_idx[7:0];
            header_val[31:16]   = layer_inputs[15:0];
            header_val[47:32]   = n_neurons[15:0];
            header_val[63:48]   = bytes_per_neuron[15:0];
            header_val[95:64]   = total_payload_bytes[31:0];
            header_val[127:96]  = (is_threshold && (layer_idx % 2 == 1)) ? $urandom() : 32'd0;

            for (int i = 0; i < 16; i++) byte_q.push_back(header_val[i*8+:8]);

            for (int n = 0; n < n_neurons; n++) begin
                if (is_threshold) begin
                    t_val = this.threshold[layer_idx][n];
                    for (int i = 0; i < 4; i++) byte_q.push_back(t_val[i*8+:8]);
                end else begin
                    // Pack weight bits into bytes; padded bits beyond fan_in are filled
                    // to avoid X-propagation and to ensure padding does not impact results.
                    w_idx = 0;
                    for (int b = 0; b < bytes_per_neuron; b++) begin
                        for (int k = 0; k < 8; k++) begin
                            if (w_idx < fan_in) byte_val[k] = this.weight[layer_idx][n][w_idx];
                            else byte_val[k] = 1'b1;
                            w_idx++;
                        end
                        byte_q.push_back(byte_val);
                    end
                end
            end

            bytes_per_beat = BUS_WIDTH / 8;
            byte_count = 0;
            current_word = '0;

            foreach (byte_q[i]) begin
                current_word[byte_count*8+:8] = byte_q[i];
                byte_count++;

                if (byte_count == bytes_per_beat) begin
                    word_q.push_back(current_word);
                    keep_q.push_back({(BUS_WIDTH / 8){1'b1}});
                    current_word = '0;
                    byte_count   = 0;
                end
            end

            if (byte_count > 0) begin
                word_q.push_back(current_word);
                current_keep = '0;
                // Partial last beat: assert keep only for the bytes that were produced.
                for (int k = 0; k < byte_count; k++) current_keep[k] = 1'b1;
                keep_q.push_back(current_keep);
            end

            stream = new[word_q.size()];
            keep   = new[keep_q.size()];
            foreach (word_q[i]) begin
                stream[i] = word_q[i];
                keep[i]   = keep_q[i];
            end
        endfunction

        function void encode_configuration_randomized(output bus_stream_t full_stream, output keep_stream_t full_keep);
            bus_stream_t layer_stream;
            keep_stream_t layer_keep;

            // Build a full configuration by concatenating per-layer messages.
            // Hidden layers may stream weights/thresholds in either order to stress parser sequencing.
            int layer_indices[];

            layer_indices = new[num_layers];
            for (int i=0; i<num_layers; i++) layer_indices[i] = i;

            full_stream = new[0];
            full_keep   = new[0];

            for (int i = 0; i < num_layers; i++) begin
                int l = layer_indices[i];
                bit do_weights_first;

                if (l == num_layers - 1) begin
                    // Output layer: weights only (thresholds are not used for popcount outputs).
                    get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                    full_stream = {full_stream, layer_stream};
                    full_keep   = {full_keep, layer_keep};
                end else begin
                    do_weights_first = ($urandom_range(0, 1) == 1);

                    if (do_weights_first) begin
                        get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};

                        get_layer_config_randomized_fields(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end else begin
                        get_layer_config_randomized_fields(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};

                        get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end
                end
            end
        endfunction
    endclass
endpackage
`timescale 1ns / 100ps

module bnn_fcc_coverage_tb #(
    // Testbench configuration.
    // Uses a small custom topology to keep runs fast and to force image-input partial beats
    // so `TKEEP` handling is exercised in a controlled way.
    parameter int      USE_CUSTOM_TOPOLOGY                      = 1'b1,
    parameter int      CUSTOM_LAYERS                            = 4,
    parameter int      CUSTOM_TOPOLOGY          [CUSTOM_LAYERS] = '{13, 10, 10, 5}, // 13 inputs forces TKEEP edge cases on data stream
    parameter int      NUM_TEST_IMAGES                          = 50,
    parameter bit      VERIFY_MODEL                             = 0, // No python verification for custom
    parameter string   BASE_DIR                                 = "../python",
    parameter bit      TOGGLE_DATA_OUT_READY                    = 1'b1,
    parameter realtime TIMEOUT                                  = 10ms,
    parameter realtime CLK_PERIOD                               = 10ns,
    parameter bit      DEBUG                                    = 1'b0,

    // Bus configuration
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int INPUT_BUS_WIDTH  = 64,
    parameter int OUTPUT_BUS_WIDTH = 8,

    // App configuration
    parameter  int INPUT_DATA_WIDTH  = 8,
    localparam int INPUTS_PER_CYCLE  = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH,
    localparam int BYTES_PER_INPUT   = INPUT_DATA_WIDTH / 8,
    parameter  int OUTPUT_DATA_WIDTH = 4,

    // DUT configuration
    localparam int NON_INPUT_LAYERS = CUSTOM_LAYERS - 1,
    localparam int OUTPUT_CLASSES   = CUSTOM_TOPOLOGY[CUSTOM_LAYERS-1],
    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[NON_INPUT_LAYERS] = '{8, 8, 10}
);
    import bnn_fcc_tb_pkg::*;
    import bnn_fcc_coverage_pkg::*;

    localparam int ACTUAL_TOTAL_LAYERS = CUSTOM_LAYERS;
    localparam int ACTUAL_TOPOLOGY[ACTUAL_TOTAL_LAYERS] = CUSTOM_TOPOLOGY;
    localparam realtime HALF_CLK_PERIOD = CLK_PERIOD / 2.0;

    initial begin
        assert (INPUT_DATA_WIDTH == 8)
        else $fatal(1, "TB ERROR: INPUT_DATA_WIDTH must be 8");
    end

    // Probability helper for randomized valid insertion/gaps.
    function automatic bit chance(real p);
        if (p > 1.0 || p < 0.0) $fatal(1, "Invalid probability in chance()");
        return ($urandom < (p * (2.0 ** 32)));
    endfunction

    // Pattern parameters that change over time to simulate continuous, intermittent, and bursty traffic.
    real current_config_prob = 1.0;
    real current_data_prob = 1.0;

    task update_probabilities();
        // Rotate config/image duty cycles to create a mixture of:
        // - continuous streaming
        // - intermittent gaps (valid deasserted while ready is high)
        // - bursty behavior (short runs of high probability)
        // 0=continuous, 1=intermittent, 2=burst
        int mode;
        mode = $urandom_range(0, 2);
        if (mode == 0) begin
            current_config_prob = 1.0;
            current_data_prob = 1.0;
        end else if (mode == 1) begin
            current_config_prob = 0.5;
            current_data_prob = 0.5;
        end else begin
            // Burst: high probability for a bit
            current_config_prob = 0.9;
            current_data_prob = 0.9;
        end
    endtask

    Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) model;
    BNN_FCC_Stimulus #(INPUT_DATA_WIDTH) stim;
    LatencyTracker latency;
    ThroughputTracker throughput;

    typedef bit [CONFIG_BUS_WIDTH-1:0] config_bus_word_t;
    typedef config_bus_word_t config_bus_data_stream_t[];

    localparam CONFIG_KEEP_WIDTH = CONFIG_BUS_WIDTH / 8;
    typedef bit [CONFIG_KEEP_WIDTH-1:0] config_bus_keep_t;
    typedef config_bus_keep_t config_keep_stream_t[];

    bit [CONFIG_BUS_WIDTH-1:0] config_bus_data_stream[];
    bit [CONFIG_BUS_WIDTH/8-1:0] config_bus_keep_stream[];

    int num_tests;
    int passed;
    int failed;

    logic clk = 1'b0;
    logic rst;

    axi4_stream_if #(.DATA_WIDTH(CONFIG_BUS_WIDTH)) config_in (.aclk(clk), .aresetn(!rst));
    axi4_stream_if #(.DATA_WIDTH(INPUT_BUS_WIDTH)) data_in (.aclk(clk), .aresetn(!rst));
    axi4_stream_if #(.DATA_WIDTH(OUTPUT_BUS_WIDTH)) data_out (.aclk(clk), .aresetn(!rst));

    // -------------------------------------------------------------------------
    // Covergroups (see verification/coverage_plan.txt for intended bin mapping)
    // -------------------------------------------------------------------------

    // Category 1: AXI4-Stream protocol patterns on config stream.
    covergroup cg_axi_config @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint config_in.tvalid;
        cp_ready: coverpoint config_in.tready;
        cp_last:  coverpoint config_in.tlast;
        cr_valid_ready: cross cp_valid, cp_ready {
            ignore_bins no_backpressure = binsof(cp_valid) intersect {1} && binsof(cp_ready) intersect {0};
        }
    endgroup

    // Category 1: AXI4-Stream protocol patterns on image input stream.
    covergroup cg_axi_image @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint data_in.tvalid;
        cp_ready: coverpoint data_in.tready;
        cp_last:  coverpoint data_in.tlast;
        cr_valid_ready: cross cp_valid, cp_ready;
    endgroup

    // Category 1: AXI4-Stream protocol patterns on output stream (with backpressure).
    covergroup cg_axi_output @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint data_out.tvalid;
        cp_ready: coverpoint data_out.tready;
        cr_valid_ready: cross cp_valid, cp_ready;
    endgroup

    // Category 2: configuration diversity (density/threshold classes and header DC randomization).
    covergroup cg_config_diversity with function sample(
        int layer_idx,
        int weight_density_class,
        int threshold_class,
        bit threshold_fields_randomized
    );
        option.per_instance = 1;

        cp_layer: coverpoint layer_idx {
            bins layers[] = {[0:NON_INPUT_LAYERS-1]};
        }

        cp_hidden_layer: coverpoint layer_idx iff (layer_idx < NON_INPUT_LAYERS - 1) {
            bins hidden_layers[] = {[0:NON_INPUT_LAYERS-2]};
        }

        cp_weight_density: coverpoint weight_density_class {
            bins all_zero = {0};
            bins sparse   = {1};
            bins balanced = {2};
            bins dense    = {3};
            bins all_one  = {4};
        }

        cp_threshold_range: coverpoint threshold_class {
            bins thr_small  = {0};
            bins thr_medium = {1};
            bins thr_large  = {2};
        }

        cp_threshold_dc: coverpoint threshold_fields_randomized {
            bins default_fields = {0};
            bins randomized_fields = {1};
        }

        cr_layer_density: cross cp_hidden_layer, cp_weight_density;
        cr_layer_threshold: cross cp_hidden_layer, cp_threshold_range;
    endgroup

    // Category 3: ensure every output class is produced, and exercise repeated/varying class sequences.
    covergroup cg_classification with function sample(
        int output_class,
        bit has_prev,
        bit repeated_class
    );
        option.per_instance = 1;

        cp_output_class: coverpoint output_class {
            bins classes[] = {[0:OUTPUT_CLASSES-1]};
        }

        cp_class_sequence: coverpoint repeated_class iff (has_prev) {
            bins repeated = {1'b1};
            bins varying  = {1'b0};
        }
    endgroup

    // Category 4: reconfiguration coverage (full vs partial updates, weights vs thresholds, layer index).
    covergroup cg_reconfiguration with function sample(
        bit full_config,
        int update_kind,
        int layer_idx
    );
        option.per_instance = 1;

        cp_scope: coverpoint full_config {
            bins full    = {1'b1};
            bins partial = {1'b0};
        }

        cp_update_kind: coverpoint update_kind {
            bins weights_only    = {0};
            bins thresholds_only = {1};
            bins weights_and_thresholds = {2};
        }

        cp_partial_layer: coverpoint layer_idx iff (!full_config) {
            bins layer0 = {0};
            bins layer1 = {1};
            bins layer2 = {2};
        }

        cr_scope_kind: cross cp_scope, cp_update_kind {
            ignore_bins impossible_full_weights = binsof(cp_scope.full) && binsof(cp_update_kind.weights_only);
            ignore_bins impossible_full_thresholds = binsof(cp_scope.full) && binsof(cp_update_kind.thresholds_only);
        }
    endgroup

    // Category 5: reset coverage (when reset occurs, whether config changes, and workload intensity).
    covergroup cg_reset with function sample(
        int reset_phase,
        bit same_config_after_reset,
        bit on_tlast_boundary,
        int workload_bucket
    );
        option.per_instance = 1;

        cp_phase: coverpoint reset_phase {
            bins before_config = {0};
            bins during_config = {1};
            bins during_image  = {2};
            bins during_output = {3};
            bins after_output  = {4};
        }

        cp_post_reset_config: coverpoint same_config_after_reset {
            bins same_config      = {1'b1};
            bins different_config = {1'b0};
        }

        cp_tlast_boundary: coverpoint on_tlast_boundary {
            bins not_on_tlast = {1'b0};
            bins on_tlast     = {1'b1};
        }

        cp_workload: coverpoint workload_bucket {
            bins cold = {0};
            bins warm = {1};
            bins hot  = {2};
        }
    endgroup

    cg_axi_config cg_config_inst;
    cg_axi_image  cg_image_inst;
    cg_axi_output cg_output_inst;
    cg_config_diversity cg_config_diversity_inst;
    cg_classification cg_classification_inst;
    cg_reconfiguration cg_reconfiguration_inst;
    cg_reset cg_reset_inst;

    Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) reconfig_model_a;
    Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) reconfig_model_b;

    typedef bit [INPUT_DATA_WIDTH-1:0] input_vector_t[];
    input_vector_t test_vectors[$];
    int total_test_vectors;

    // Helper: classify weight rows into coarse density buckets for cg_config_diversity bins.
    function automatic int classify_weight_density(input int layer_idx, input int neuron_idx);
        int ones;
        int fan_in;

        ones = 0;
        fan_in = model.weight[layer_idx][neuron_idx].size();
        for (int i = 0; i < fan_in; i++) begin
            if (model.weight[layer_idx][neuron_idx][i]) ones++;
        end

        if (ones == 0) return 0;
        if (ones == fan_in) return 4;
        if ((ones * 4) <= fan_in) return 1;
        if ((ones * 4) >= (fan_in * 3)) return 3;
        return 2;
    endfunction

    // Helper: classify thresholds into small/medium/large ranges relative to fan-in.
    function automatic int classify_threshold_range(input int layer_idx, input int neuron_idx);
        int fan_in;
        int threshold_val;

        fan_in = model.weight[layer_idx][neuron_idx].size();
        threshold_val = model.threshold[layer_idx][neuron_idx];

        if (threshold_val <= (fan_in / 3)) return 0;
        if (threshold_val >= ((fan_in * 2) / 3)) return 2;
        return 1;
    endfunction

    // Sample the per-layer/per-neuron diversity covergroup after a new config is built.
    task automatic sample_cat2_coverage();
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            for (int n = 0; n < model.weight[l].size(); n++) begin
                cg_config_diversity_inst.sample(
                    l,
                    classify_weight_density(l, n),
                    classify_threshold_range(l, n),
                    l[0]
                );
            end
        end
    endtask

    // Convert a bitmask into an 8-bit-per-input activation vector (0x00/0xFF per input).
    function automatic input_vector_t make_vector_from_mask(input int mask);
        input_vector_t vec;

        vec = new[ACTUAL_TOPOLOGY[0]];
        for (int i = 0; i < ACTUAL_TOPOLOGY[0]; i++) begin
            vec[i] = mask[i] ? 8'hff : 8'h00;
        end
        return vec;
    endfunction

    // Compare hidden-layer codes (arrays) for uniqueness checks in Category 3 retuning.
    function automatic bit hidden_code_equal(input int a[], input int b[]);
        if (a.size() != b.size()) return 1'b0;
        for (int i = 0; i < a.size(); i++) begin
            if (a[i] != b[i]) return 1'b0;
        end
        return 1'b1;
    endfunction

    task automatic retune_model_for_cat3(
        output input_vector_t class_examples[OUTPUT_CLASSES]
    );
        // Category 3 (computational stimulus):
        // Find a set of reachable hidden-layer activation codes, then rewrite the output
        // layer so each class maps to a unique reachable code. This yields directed
        // example vectors that deterministically hit each output class.
        int reachable_codes   [OUTPUT_CLASSES][];
        input_vector_t examples[OUTPUT_CLASSES];
        int num_codes;
        int output_layer_idx;

        num_codes = 0;
        output_layer_idx = model.num_layers - 1;

        for (int mask = 0; mask < (1 << ACTUAL_TOPOLOGY[0]); mask++) begin
            input_vector_t candidate_vec;
            int current_code[];
            bit is_new;

            candidate_vec = make_vector_from_mask(mask);
            void'(model.compute_reference(candidate_vec));

            current_code = new[model.layer_outputs[output_layer_idx-1].size()];
            for (int i = 0; i < current_code.size(); i++) begin
                current_code[i] = model.layer_outputs[output_layer_idx-1][i];
            end

            is_new = 1'b1;
            for (int c = 0; c < num_codes; c++) begin
                if (hidden_code_equal(current_code, reachable_codes[c])) begin
                    is_new = 1'b0;
                    break;
                end
            end

            if (is_new) begin
                reachable_codes[num_codes] = new[current_code.size()];
                for (int i = 0; i < current_code.size(); i++) reachable_codes[num_codes][i] = current_code[i];
                examples[num_codes] = candidate_vec;
                num_codes++;
                if (num_codes == OUTPUT_CLASSES) break;
            end
        end

        assert (num_codes == OUTPUT_CLASSES)
        else $fatal(1, "TB ERROR: Unable to find %0d unique hidden codes for Category 3.", OUTPUT_CLASSES);

        for (int cls = 0; cls < OUTPUT_CLASSES; cls++) begin
            model.weight[output_layer_idx][cls] = new[reachable_codes[cls].size()];
            for (int i = 0; i < reachable_codes[cls].size(); i++) begin
                model.weight[output_layer_idx][cls][i] = reachable_codes[cls][i][0];
            end
            class_examples[cls] = examples[cls];
        end

        for (int cls = 0; cls < OUTPUT_CLASSES; cls++) begin
            int observed_class;

            observed_class = model.compute_reference(class_examples[cls]);
            assert (observed_class == cls)
            else $fatal(1, "TB ERROR: Category 3 class example for class %0d resolves to class %0d.", cls, observed_class);
        end
    endtask

    task automatic build_test_vector_database();
        // Build the complete image stimulus database:
        // - first, a directed class sequence (including repeats and changes) for cg_classification
        // - then, random vectors to broaden Cat1 protocol activity
        input_vector_t class_examples[OUTPUT_CLASSES];
        int class_sequence[$];

        test_vectors.delete();

        retune_model_for_cat3(class_examples);

        class_sequence = {0, 0, 1, 2, 2, 3, 4, 4, 1, 3};
        foreach (class_sequence[idx]) begin
            test_vectors.push_back(class_examples[class_sequence[idx]]);
        end

        $display("--- Generating Random Test Vectors (Category 1) ---");
        stim.generate_random_vectors(NUM_TEST_IMAGES);
        for (int i = 0; i < stim.get_num_vectors(); i++) begin
            input_vector_t random_vec;
            stim.get_vector(i, random_vec);
            test_vectors.push_back(random_vec);
        end
    endtask

    task automatic copy_layer_fields(
        input Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) src_model,
        input int layer_idx,
        input bit copy_weights,
        input bit copy_thresholds
    );
        if (copy_weights) begin
            model.weight[layer_idx] = new[src_model.weight[layer_idx].size()];
            for (int n = 0; n < src_model.weight[layer_idx].size(); n++) begin
                model.weight[layer_idx][n] = new[src_model.weight[layer_idx][n].size()];
                for (int i = 0; i < src_model.weight[layer_idx][n].size(); i++) begin
                    model.weight[layer_idx][n][i] = src_model.weight[layer_idx][n][i];
                end
            end
        end

        if (copy_thresholds) begin
            model.threshold[layer_idx] = new[src_model.threshold[layer_idx].size()];
            for (int n = 0; n < src_model.threshold[layer_idx].size(); n++) begin
                model.threshold[layer_idx][n] = src_model.threshold[layer_idx][n];
            end
        end
    endtask

    task automatic stream_config_words(
        input config_bus_word_t data_stream[],
        input config_bus_keep_t keep_stream[],
        input string phase_label
    );
        // Drive a config stream (data+keep) with randomized gaps to hit AXI protocol bins.
        // `tlast` asserts on the final beat; `tkeep` matches the pre-encoded keep stream.
        $display("[%0t] %s", $realtime, phase_label);

        for (int i = 0; i < data_stream.size(); i++) begin
            if (i % 8 == 0) update_probabilities();

            while (!chance(current_config_prob)) begin
                config_in.tvalid <= 1'b0;
                @(posedge clk iff config_in.tready);
            end

            config_in.tvalid <= 1'b1;
            config_in.tdata  <= data_stream[i];
            config_in.tlast  <= (i == data_stream.size() - 1);
            config_in.tkeep  <= keep_stream[i];
            @(posedge clk iff config_in.tready);
        end

        config_in.tvalid <= 1'b0;
        config_in.tlast  <= 1'b0;
        config_in.tkeep  <= '0;
    endtask

    task automatic apply_reconfiguration(
        input bit full_config,
        input int layer_idx,
        input bit include_weights,
        input bit include_thresholds,
        input Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) src_model,
        input string phase_label
    );
        // Category 4: apply either a full reconfiguration (all layers) or a partial
        // reconfiguration (one layer, weights and/or thresholds). The test waits for the
        // output queue to drain to avoid mixing old/new expectations.
        config_bus_word_t local_stream[];
        config_bus_keep_t local_keep[];
        config_bus_word_t temp_stream[];
        config_bus_keep_t temp_keep[];

        wait (expected_outputs.size() == 0);
        wait (data_in.tready);
        repeat (3) @(posedge clk);

        if (full_config) begin
            for (int l = 0; l < model.num_layers; l++) begin
                copy_layer_fields(src_model, l, 1'b1, (l < model.num_layers - 1));
            end
            model.encode_configuration_randomized(local_stream, local_keep);
            cg_reconfiguration_inst.sample(1'b1, 2, -1);
        end else begin
            copy_layer_fields(src_model, layer_idx, include_weights, include_thresholds);
            local_stream = new[0];
            local_keep   = new[0];

            if (include_weights) begin
                model.get_layer_config_randomized_fields(layer_idx, 1'b0, temp_stream, temp_keep);
                local_stream = {local_stream, temp_stream};
                local_keep   = {local_keep, temp_keep};
            end

            if (include_thresholds) begin
                model.get_layer_config_randomized_fields(layer_idx, 1'b1, temp_stream, temp_keep);
                local_stream = {local_stream, temp_stream};
                local_keep   = {local_keep, temp_keep};
            end

            cg_reconfiguration_inst.sample(
                1'b0,
                include_weights && include_thresholds ? 2 : (include_thresholds ? 1 : 0),
                layer_idx
            );
        end

        stream_config_words(local_stream, local_keep, phase_label);
        repeat (3) @(posedge clk);
    endtask

    task automatic discard_pending_expectations();
        // When a reset/reconfig is injected mid-stream, drop any expected outputs that
        // are no longer meaningful (since the model/DUT state has been reset).
        if (expected_outputs.size() > 0) begin
            num_tests -= expected_outputs.size();
            expected_outputs.delete();
        end
    endtask

    task automatic apply_reset_and_full_reconfig(
        input int reset_phase,
        input bit same_config_after_reset,
        input bit on_tlast_boundary,
        input int workload_bucket,
        input Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) src_model,
        input string phase_label
    );
        // Category 5: inject a reset and then stream a full configuration.
        // The covergroup captures when the reset happened and whether the config is
        // kept the same or changed after the reset.
        config_bus_word_t local_stream[];
        config_bus_keep_t local_keep[];

        cg_reset_inst.sample(reset_phase, same_config_after_reset, on_tlast_boundary, workload_bucket);

        discard_pending_expectations();

        config_in.tvalid <= 1'b0;
        config_in.tdata  <= '0;
        config_in.tkeep  <= '0;
        config_in.tlast  <= 1'b0;
        data_in.tvalid   <= 1'b0;
        data_in.tdata    <= '0;
        data_in.tkeep    <= '0;
        data_in.tlast    <= 1'b0;

        @(negedge clk);
        rst <= 1'b1;
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst <= 1'b0;
        repeat (3) @(posedge clk);

        if (!same_config_after_reset) begin
            for (int l = 0; l < model.num_layers; l++) begin
                copy_layer_fields(src_model, l, 1'b1, (l < model.num_layers - 1));
            end
        end

        model.encode_configuration_randomized(local_stream, local_keep);
        stream_config_words(local_stream, local_keep, phase_label);
        wait (data_in.tready);
        repeat (3) @(posedge clk);
    endtask

    initial begin
        cg_config_inst = new();
        cg_image_inst  = new();
        cg_output_inst = new();
        cg_config_diversity_inst = new();
        cg_classification_inst = new();
        cg_reconfiguration_inst = new();
        cg_reset_inst = new();
    end

    bnn_fcc #(
        .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
        .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
        .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
        .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
        .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
        .TOTAL_LAYERS     (ACTUAL_TOTAL_LAYERS),
        .TOPOLOGY         (ACTUAL_TOPOLOGY),
        .PARALLEL_INPUTS  (PARALLEL_INPUTS),
        .PARALLEL_NEURONS (PARALLEL_NEURONS)
    ) DUT (
        .clk(clk),
        .rst(rst),
        .config_valid(config_in.tvalid),
        .config_ready(config_in.tready),
        .config_data (config_in.tdata),
        .config_keep (config_in.tkeep),
        .config_last (config_in.tlast),
        .data_in_valid(data_in.tvalid),
        .data_in_ready(data_in.tready),
        .data_in_data (data_in.tdata),
        .data_in_keep (data_in.tkeep),
        .data_in_last (data_in.tlast),
        .data_out_valid(data_out.tvalid),
        .data_out_ready(data_out.tready),
        .data_out_data (data_out.tdata),
        .data_out_keep (data_out.tkeep),
        .data_out_last (data_out.tlast)
    );

    initial begin : generate_clock
        forever #HALF_CLK_PERIOD clk <= ~clk;
    end

    initial begin : l_init_model
        // Build the initial model, configs, and stimulus database before driving the DUT.
        model = new();
        reconfig_model_a = new();
        reconfig_model_b = new();
        stim = new(ACTUAL_TOPOLOGY[0]);
        latency = new(CLK_PERIOD);
        throughput = new(CLK_PERIOD);
        if (cg_config_diversity_inst == null) cg_config_diversity_inst = new();
        if (cg_classification_inst == null) cg_classification_inst = new();
        if (cg_reconfiguration_inst == null) cg_reconfiguration_inst = new();
        if (cg_reset_inst == null) cg_reset_inst = new();

        $display("--- Loading Diverse Model (Category 1 + Category 2) ---");
        model.create_diverse_config(ACTUAL_TOPOLOGY);
        reconfig_model_a.create_random(ACTUAL_TOPOLOGY);
        reconfig_model_b.create_random(ACTUAL_TOPOLOGY);
        sample_cat2_coverage();

        $display("--- Building Directed + Random Test Vectors (Category 3 + Category 1) ---");
        build_test_vector_database();

        // Randomized stream ordering/timing still drives Cat1 protocol coverage.
        model.encode_configuration_randomized(config_bus_data_stream, config_bus_keep_stream);
        $display("--- Configuration created: %0d words (%0d-bit wide) ---", config_bus_data_stream.size(), CONFIG_BUS_WIDTH);

        total_test_vectors = test_vectors.size();
        num_tests = 0;
        model.print_summary();

        if (DEBUG) model.print_model();
    end

    logic [OUTPUT_DATA_WIDTH-1:0] expected_outputs[$];

    assign config_in.tstrb = config_in.tkeep;
    assign data_in.tstrb   = data_in.tkeep;

    initial begin : l_sequencer_and_driver
        // Main sequencer:
        // - stream an initial configuration (including a partial pre-reset config)
        // - optionally inject resets and partial/full reconfigurations at chosen image indices
        // - stream images with randomized gaps and proper `keep`/`last` signaling
        // - push expected outputs into a queue that the output monitor drains
        $timeformat(-9, 0, " ns", 0);

        rst              <= 1'b1;
        config_in.tvalid <= 1'b0;
        config_in.tdata  <= '0;
        config_in.tkeep  <= '0;
        config_in.tlast  <= 1'b0;
        data_in.tvalid   <= 1'b0;
        data_in.tdata    <= '0;
        data_in.tkeep    <= '0;
        data_in.tlast    <= 1'b0;

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst <= 1'b0;
        repeat (5) @(posedge clk);

        cg_reset_inst.sample(0, 1'b1, 1'b0, 0);
        cg_reconfiguration_inst.sample(1'b1, 2, -1);
        stream_config_words(config_bus_data_stream[0:7], config_bus_keep_stream[0:7], "Streaming initial partial configuration before Category 5 reset.");
        apply_reset_and_full_reconfig(
            1,
            1'b1,
            1'b0,
            0,
            model,
            "Re-applying full configuration after Category 5 reset during config."
        );

        wait (data_in.tready);
        repeat (5) @(posedge clk);

        for (int i = 0; i < total_test_vectors; i++) begin
            int expected_pred;
            input_vector_t current_img;
            int image_delay;

            if (i == 15) begin
                apply_reconfiguration(
                    1'b0,
                    2,
                    1'b1,
                    1'b0,
                    reconfig_model_a,
                    "Applying Category 4 partial reconfiguration: output-layer weights only."
                );
            end
            if (i == 30) begin
                apply_reconfiguration(
                    1'b0,
                    0,
                    1'b0,
                    1'b1,
                    reconfig_model_b,
                    "Applying Category 4 partial reconfiguration: hidden-layer thresholds only."
                );
            end
            if (i == 45) begin
                apply_reconfiguration(
                    1'b0,
                    1,
                    1'b1,
                    1'b1,
                    reconfig_model_a,
                    "Applying Category 4 partial reconfiguration: hidden-layer weights and thresholds."
                );
            end
            if (i == 55) begin
                apply_reconfiguration(
                    1'b1,
                    -1,
                    1'b1,
                    1'b1,
                    reconfig_model_b,
                    "Applying Category 4 full reconfiguration of all layers."
                );
            end
            if (i == 20) begin
                wait (expected_outputs.size() == 0);
            end
            if (i == 35) begin
                wait (expected_outputs.size() > 0);
                apply_reset_and_full_reconfig(
                    3,
                    1'b1,
                    1'b0,
                    2,
                    model,
                    "Re-applying same configuration after Category 5 reset during output drain."
                );
            end
            if (i == 55) begin
                wait (expected_outputs.size() == 0);
                apply_reset_and_full_reconfig(
                    4,
                    1'b0,
                    1'b0,
                    2,
                    reconfig_model_b,
                    "Applying different full configuration after Category 5 reset after outputs drained."
                );
            end

            current_img = test_vectors[i];
            expected_pred = model.compute_reference(current_img);
            expected_outputs.push_back(expected_pred);
            num_tests++;

            $display("[%0t] Streaming image %0d (Category 1 + Category 3 + Category 4 + Category 5).", $realtime, i);

            for (int j = 0; j < current_img.size(); j += INPUTS_PER_CYCLE) begin
                if (j % 16 == 0) update_probabilities(); // Change duty cycle during image

                for (int k = 0; k < INPUTS_PER_CYCLE; k++) begin
                    if (j + k < current_img.size()) begin
                        data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= current_img[j+k];
                        data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '1;
                    end else begin
                        data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= '0;
                        data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '0;
                    end
                end

                while (!chance(current_data_prob)) begin
                    data_in.tvalid <= 1'b0;
                    @(posedge clk iff data_in.tready);
                end
                data_in.tvalid <= 1'b1;
                data_in.tlast  <= (j + INPUTS_PER_CYCLE >= current_img.size());
                @(posedge clk iff data_in.tready);

                if (i == 20 && data_in.tlast) begin
                    apply_reset_and_full_reconfig(
                        2,
                        1'b1,
                        1'b1,
                        1,
                        model,
                        "Re-applying same configuration after Category 5 reset on image TLAST boundary."
                    );
                    break;
                end

                if (i == 0 && j == 0) throughput.start_test();
                if (j == 0) latency.start_event(i);
            end

            data_in.tvalid <= 1'b0;
            data_in.tlast  <= 1'b0;
            data_in.tkeep  <= '0;

            // Random delay between consecutive images
            image_delay = $urandom_range(0, 10);
            repeat (image_delay) @(posedge clk);
        end

        $display("[%0t] All images loaded, waiting for outputs.", $realtime);
        wait (expected_outputs.size() == 0);
        repeat (5) @(posedge clk);

        disable generate_clock;
        disable l_timeout;
        if (passed == num_tests) $display("[%0t] SUCCESS: all %0d tests completed successfully.", $realtime, num_tests);
        else $error("FAILED: %0d out of %0d tests failed.", failed, num_tests);
    end

    initial begin : l_toggle_ready
        // Randomize downstream backpressure to exercise output AXI handshakes and ensure
        // the DUT handles stalls correctly.
        int throttle_len;
        data_out.tready <= 1'b1;
        @(posedge clk iff !rst);
        if (TOGGLE_DATA_OUT_READY) begin
            forever begin
                // 0=continuous readiness, 1=intermittent readiness, 2=backpressure long burst
                int mode;
                mode = $urandom_range(0, 2);
                if (mode == 0) begin
                    throttle_len = $urandom_range(1, 10);
                    repeat(throttle_len) begin
                        data_out.tready <= 1'b1;
                        @(posedge clk);
                    end
                end else if (mode == 1) begin
                    throttle_len = $urandom_range(5, 20);
                    repeat(throttle_len) begin
                        data_out.tready <= chance(0.5); // intermittent
                        @(posedge clk);
                    end
                end else begin
                    // apply backpressure
                    data_out.tready <= 1'b0;
                    throttle_len = $urandom_range(5, 50); // long backpressure
                    repeat(throttle_len) @(posedge clk);
                end
            end
        end else data_out.tready <= 1'b1;
    end

    initial begin : l_output_monitor
        // Scoreboard/monitor:
        // - compares DUT outputs against the reference model's expected class
        // - samples classification coverpoints (hit all classes + repeats/varying)
        // - records latency/throughput statistics
        automatic int output_count = 0;
        automatic logic [OUTPUT_DATA_WIDTH-1:0] expected_output;
        automatic logic [OUTPUT_DATA_WIDTH-1:0] actual_output;
        automatic bit have_prev_class = 1'b0;
        automatic logic [OUTPUT_DATA_WIDTH-1:0] prev_output_class = '0;
        forever begin
            @(posedge clk iff data_out.tvalid && data_out.tready);
            assert (expected_outputs.size() > 0)
            else $fatal(1, "No expected output for actual output");

            expected_output = expected_outputs[0];
            actual_output   = data_out.tdata;

            $display("[MONITOR @%0t] Output %0d -> RTL=%0d, Expected=%0d", $realtime, output_count, actual_output, expected_output);

            if (actual_output == expected_output) begin
                passed++;
            end else begin
                $error("Output incorrect for image %0d: actual = %0d vs expected = %0d", output_count, actual_output, expected_output);
                model.print_inference_trace();
                failed++;
            end

            cg_classification_inst.sample(actual_output, have_prev_class, have_prev_class && (actual_output == prev_output_class));
            prev_output_class = actual_output;
            have_prev_class = 1'b1;

            void'(expected_outputs.pop_front());
            latency.end_event(output_count);
            if (output_count == num_tests - 1) throughput.sample_end();
            output_count++;
        end
    end

    initial begin : l_timeout
        // Guardrail: if something deadlocks, fail rather than hanging indefinitely.
        #TIMEOUT;
        $fatal(1, $sformatf("Simulation failed due to timeout of %0t.", TIMEOUT));
    end

endmodule
