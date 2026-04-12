`timescale 1ns / 100ps

module bnn_fcc_cat1_tb #(
    // Testbench configuration (Using custom topology to test TKEEP edge cases)
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
    import bnn_fcc_cat1_pkg::*;

    localparam int ACTUAL_TOTAL_LAYERS = CUSTOM_LAYERS;
    localparam int ACTUAL_TOPOLOGY[ACTUAL_TOTAL_LAYERS] = CUSTOM_TOPOLOGY;
    localparam realtime HALF_CLK_PERIOD = CLK_PERIOD / 2.0;

    initial begin
        assert (INPUT_DATA_WIDTH == 8)
        else $fatal(1, "TB ERROR: INPUT_DATA_WIDTH must be 8");
    end

    function automatic bit chance(real p);
        if (p > 1.0 || p < 0.0) $fatal(1, "Invalid probability in chance()");
        return ($urandom < (p * (2.0 ** 32)));
    endfunction
    
    // Pattern parameters that change over time to simulate bursts, intermittent, continuous
    real current_config_prob = 1.0;
    real current_data_prob = 1.0;
    
    task update_probabilities();
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

    covergroup cg_axi_config @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint config_in.tvalid;
        cp_ready: coverpoint config_in.tready;
        cp_last:  coverpoint config_in.tlast;
        cr_valid_ready: cross cp_valid, cp_ready {
            ignore_bins no_backpressure = binsof(cp_valid) intersect {1} && binsof(cp_ready) intersect {0};
        }
    endgroup

    covergroup cg_axi_image @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint data_in.tvalid;
        cp_ready: coverpoint data_in.tready;
        cp_last:  coverpoint data_in.tlast;
        cr_valid_ready: cross cp_valid, cp_ready;
    endgroup

    covergroup cg_axi_output @(posedge clk);
        option.per_instance = 1;
        cp_valid: coverpoint data_out.tvalid;
        cp_ready: coverpoint data_out.tready;
        cr_valid_ready: cross cp_valid, cp_ready;
    endgroup

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

    function automatic int classify_threshold_range(input int layer_idx, input int neuron_idx);
        int fan_in;
        int threshold_val;

        fan_in = model.weight[layer_idx][neuron_idx].size();
        threshold_val = model.threshold[layer_idx][neuron_idx];

        if (threshold_val <= (fan_in / 3)) return 0;
        if (threshold_val >= ((fan_in * 2) / 3)) return 2;
        return 1;
    endfunction

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

    function automatic input_vector_t make_vector_from_mask(input int mask);
        input_vector_t vec;

        vec = new[ACTUAL_TOPOLOGY[0]];
        for (int i = 0; i < ACTUAL_TOPOLOGY[0]; i++) begin
            vec[i] = mask[i] ? 8'hff : 8'h00;
        end
        return vec;
    endfunction

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
        #TIMEOUT;
        $fatal(1, $sformatf("Simulation failed due to timeout of %0t.", TIMEOUT));
    end

endmodule
