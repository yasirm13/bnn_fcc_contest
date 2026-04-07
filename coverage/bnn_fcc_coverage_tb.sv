`timescale 1ns / 100ps

module bnn_fcc_coverage_tb #(
    parameter int    SCENARIO_ID = 0,
    parameter string BASE_DIR    = "../python",
    parameter bit    USE_CUSTOM_TOPOLOGY = 1'b0,
    parameter int    CUSTOM_LAYERS = 4,
    parameter int    CUSTOM_TOPOLOGY[CUSTOM_LAYERS] = '{13, 17, 11, 7},
    parameter int    RANDOM_SEED = 32'h1BAD_F00D,
    parameter int    RANDOM_STRESS_ITERATIONS = 24,
    parameter int    RANDOM_VECTOR_POOL = 64,
    parameter int    RANDOM_MAX_IDLE_GAP = 5,
    parameter int    RANDOM_MAX_OUTPUT_STALL = 5,
    parameter int    RANDOM_RESET_PROBABILITY_PCT = 12,

    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int INPUT_BUS_WIDTH  = 64,
    parameter int OUTPUT_BUS_WIDTH = 8,
    parameter int INPUT_DATA_WIDTH = 8,
    parameter int OUTPUT_DATA_WIDTH = 4,

    parameter realtime CLK_PERIOD = 10ns,
    parameter realtime TIMEOUT    = 20ms
);
    import bnn_fcc_tb_pkg::*;
    import bnn_fcc_cov_pkg::*;

    localparam int TRAINED_LAYERS = 4;
    localparam int TRAINED_TOPOLOGY[TRAINED_LAYERS] = '{784, 256, 256, 10};
    localparam int TOTAL_LAYERS = USE_CUSTOM_TOPOLOGY ? CUSTOM_LAYERS : TRAINED_LAYERS;
    localparam int TOPOLOGY[TOTAL_LAYERS] = USE_CUSTOM_TOPOLOGY ? CUSTOM_TOPOLOGY :
        TRAINED_TOPOLOGY;
    localparam int NON_INPUT_LAYERS = TOTAL_LAYERS - 1;
    localparam int PARALLEL_INPUTS = 8;
    localparam int PARALLEL_NEURONS[NON_INPUT_LAYERS] = '{default: 8};
    localparam int INPUTS_PER_CYCLE = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH;
    localparam int BYTES_PER_INPUT = INPUT_DATA_WIDTH / 8;
    localparam int CONFIG_KEEP_WIDTH = CONFIG_BUS_WIDTH / 8;
    localparam int INPUT_KEEP_WIDTH = INPUT_BUS_WIDTH / 8;
    localparam int OUTPUT_CLASSES = TOPOLOGY[TOTAL_LAYERS-1];
    localparam int MNIST_SCAN_LIMIT = 2048;
    localparam string MNIST_MODEL_DATA_PATH = "model_data";
    localparam string MNIST_TEST_VECTOR_INPUT_PATH = "test_vectors/inputs.hex";
    localparam realtime HALF_CLK_PERIOD = CLK_PERIOD / 2.0;

    typedef bit [INPUT_DATA_WIDTH-1:0] pixel_t;
    typedef pixel_t image_t[];
    typedef bit [CONFIG_BUS_WIDTH-1:0] config_word_t;
    typedef config_word_t config_stream_t[];
    typedef bit [CONFIG_KEEP_WIDTH-1:0] config_keep_t;
    typedef config_keep_t config_keep_stream_t[];

    typedef struct packed {
        ready_relation_e relation;
        ready_style_e    style;
        int              stall_cycles;
    } output_plan_t;

    typedef enum int {
        OUTDRV_IDLE = 0,
        OUTDRV_WAIT_VALID = 1,
        OUTDRV_STALL = 2,
        OUTDRV_ACTIVE = 3
    } output_driver_state_e;

    bnn_fcc_cov_collector collector;
    BNN_FCC_Stimulus #(INPUT_DATA_WIDTH) stim_helper;

    BNN_FCC_Model #(CONFIG_BUS_WIDTH) active_model;
    BNN_FCC_Model #(CONFIG_BUS_WIDTH) baseline_model;
    BNN_FCC_Model #(CONFIG_BUS_WIDTH) variant_model;

    image_t class_images[OUTPUT_CLASSES];

    int expected_outputs[$];
    ready_relation_e expected_ready_relations[$];
    ready_style_e expected_ready_styles[$];
    int expected_stall_cycles[$];
    output_plan_t driver_output_plans[$];

    int passed;
    int failed;
    int reset_count;
    int config_message_count;
    int prev_config_layer;
    bit prev_config_is_threshold;
    int last_output_class;
    bit last_output_class_valid;
    int output_handshake_count;
    int driver_handshakes_seen;
    int dataset_cursor;

    output_driver_state_e output_driver_state;
    output_plan_t active_output_plan;
    int active_ready_cycle;
    int remaining_stall_cycles;

    logic clk = 1'b0;
    logic rst;

    axi4_stream_if #(
        .DATA_WIDTH(CONFIG_BUS_WIDTH)
    ) config_in (
        .aclk   (clk),
        .aresetn(!rst)
    );

    axi4_stream_if #(
        .DATA_WIDTH(INPUT_BUS_WIDTH)
    ) data_in (
        .aclk   (clk),
        .aresetn(!rst)
    );

    axi4_stream_if #(
        .DATA_WIDTH(OUTPUT_BUS_WIDTH)
    ) data_out (
        .aclk   (clk),
        .aresetn(!rst)
    );

    bnn_fcc #(
        .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
        .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
        .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
        .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
        .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
        .TOTAL_LAYERS     (TOTAL_LAYERS),
        .TOPOLOGY         (TOPOLOGY),
        .PARALLEL_INPUTS  (PARALLEL_INPUTS),
        .PARALLEL_NEURONS (PARALLEL_NEURONS)
    ) dut (
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

    assign config_in.tstrb = config_in.tkeep;
    assign data_in.tstrb = data_in.tkeep;

    initial begin : generate_clock
        forever #HALF_CLK_PERIOD clk <= ~clk;
    end

    initial begin : parameter_sanity
        if (TOTAL_LAYERS < 2) begin
            $fatal(1, "Coverage TB requires at least an input layer and one output layer.");
        end
        if (OUTPUT_CLASSES > (1 << OUTPUT_DATA_WIDTH)) begin
            $fatal(1, "Coverage TB requires OUTPUT_DATA_WIDTH=%0d to encode %0d classes.",
                OUTPUT_DATA_WIDTH, OUTPUT_CLASSES);
        end
        if (RANDOM_STRESS_ITERATIONS <= 0) begin
            $fatal(1, "Coverage TB requires RANDOM_STRESS_ITERATIONS > 0.");
        end
    end

    function automatic string scenario_name(int id);
        case (id)
            0: return "full_reordered";
            1: return "weights_only_reconfig";
            2: return "thresholds_only_reconfig";
            3: return "partial_subset_reconfig";
            4: return "reset_mid_config";
            5: return "reset_mid_image_and_output";
            6: return "randomized_stress";
            default: return "unknown";
        endcase
    endfunction

    function automatic bit chance_pct(input int pct);
        if (pct <= 0) begin
            return 1'b0;
        end
        if (pct >= 100) begin
            return 1'b1;
        end
        return ($urandom_range(99, 0) < pct);
    endfunction

    function automatic int random_range_inclusive(input int low, input int high);
        if (high <= low) begin
            return low;
        end
        return $urandom_range(high, low);
    endfunction

    function automatic gap_style_e random_gap_style();
        return gap_style_e'(random_range_inclusive(GAP_CONTINUOUS, GAP_BURSTY));
    endfunction

    function automatic ready_relation_e random_ready_relation();
        return ready_relation_e'(random_range_inclusive(READY_BEFORE_VALID, READY_AFTER_VALID));
    endfunction

    function automatic ready_style_e random_ready_style();
        return ready_style_e'(random_range_inclusive(READY_STYLE_CONTINUOUS, READY_STYLE_BURSTY));
    endfunction

    function automatic int random_output_stall(input ready_relation_e relation);
        if (relation == READY_AFTER_VALID) begin
            return random_range_inclusive(0, RANDOM_MAX_OUTPUT_STALL);
        end
        return 0;
    endfunction

    function automatic int count_ones_keep(input logic [255:0] keep_bus, input int keep_width);
        int count;
        begin
            count = 0;
            for (int i = 0; i < keep_width; i++) begin
                count += keep_bus[i];
            end
            return count;
        end
    endfunction

    function automatic int keep_class(input int keep_ones, input int keep_width);
        if (keep_ones == 0) begin
            return 0;
        end
        if (keep_ones == keep_width) begin
            return 2;
        end
        return 1;
    endfunction

    function automatic int next_gap_cycles(input gap_style_e style, input int beat_idx);
        case (style)
            GAP_CONTINUOUS: return 0;
            GAP_INTERMITTENT: return (beat_idx % 2) ? 1 : 0;
            GAP_BURSTY: begin
                if ((beat_idx % 5) < 3) return 0;
                return 3;
            end
            default: return 0;
        endcase
    endfunction

    function automatic bit next_ready_value(input ready_style_e style, input int cycle_idx);
        case (style)
            READY_STYLE_CONTINUOUS: return 1'b1;
            READY_STYLE_INTERMITTENT: return (cycle_idx % 2) == 0;
            READY_STYLE_BURSTY: return ((cycle_idx % 5) < 3);
            default: return 1'b1;
        endcase
    endfunction

    function automatic int classify_order(input int layer_idx, input bit is_threshold);
        if (config_message_count == 0) begin
            return ORDER_FIRST;
        end
        if (layer_idx > prev_config_layer) begin
            return ORDER_ASCENDING_LAYER;
        end
        if (layer_idx < prev_config_layer) begin
            return ORDER_DESCENDING_LAYER;
        end
        if (is_threshold != prev_config_is_threshold) begin
            return ORDER_SAME_LAYER_TYPE_SWAP;
        end
        return ORDER_SAME_LAYER_REPEAT;
    endfunction

    function automatic int clamp_threshold(input int value, input int fan_in);
        if (value < 0) begin
            return 0;
        end
        if (value > fan_in) begin
            return fan_in;
        end
        return value;
    endfunction

    function automatic int threshold_from_mode(input int fan_in, input int neuron_idx, input int mode);
        int base;
        begin
            case (mode)
                0: base = neuron_idx % ((fan_in / 8) + 1);
                1: base = (fan_in / 2) + ((neuron_idx % 7) - 3);
                2: base = fan_in - 1 - (neuron_idx % ((fan_in / 8) + 1));
                default: base = fan_in / 2;
            endcase
            return clamp_threshold(base, fan_in);
        end
    endfunction

    function automatic bit config_partial_keep_possible_f();
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            int weight_msg_bytes;
            weight_msg_bytes = 16 + (TOPOLOGY[l+1] * ((TOPOLOGY[l] + 7) / 8));
            if ((weight_msg_bytes % CONFIG_KEEP_WIDTH) != 0) begin
                return 1'b1;
            end

            if (l < NON_INPUT_LAYERS - 1) begin
                int threshold_msg_bytes;
                threshold_msg_bytes = 16 + (TOPOLOGY[l+1] * 4);
                if ((threshold_msg_bytes % CONFIG_KEEP_WIDTH) != 0) begin
                    return 1'b1;
                end
            end
        end
        return 1'b0;
    endfunction

    function automatic bit image_partial_keep_possible_f();
        return ((TOPOLOGY[0] * BYTES_PER_INPUT) % INPUT_KEEP_WIDTH) != 0;
    endfunction

    task automatic clear_input_drives();
        config_in.tvalid = 1'b0;
        config_in.tdata  = '0;
        config_in.tkeep  = '0;
        config_in.tlast  = 1'b0;
        config_in.tuser  = '0;
        config_in.tid    = '0;
        config_in.tdest  = '0;

        data_in.tvalid = 1'b0;
        data_in.tdata  = '0;
        data_in.tkeep  = '0;
        data_in.tlast  = 1'b0;
        data_in.tuser  = '0;
        data_in.tid    = '0;
        data_in.tdest  = '0;
    endtask

    task automatic clear_all_drives();
        clear_input_drives();
        data_out.tready = 1'b0;
    endtask

    task automatic copy_image(input image_t src, output image_t dst);
        dst = new[src.size()];
        foreach (src[i]) begin
            dst[i] = src[i];
        end
    endtask

    task automatic sample_model_profile(input BNN_FCC_Model #(CONFIG_BUS_WIDTH) model);
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            int fan_in;
            fan_in = TOPOLOGY[l];
            for (int n = 0; n < TOPOLOGY[l+1]; n++) begin
                int ones;
                int threshold_pct;
                ones = 0;
                for (int i = 0; i < fan_in; i++) begin
                    ones += model.weight[l][n][i];
                end

                if (l == NON_INPUT_LAYERS - 1) begin
                    threshold_pct = 0;
                end else begin
                    threshold_pct = (100 * model.threshold[l][n]) / fan_in;
                end

                collector.sample_model(l, (100 * ones) / fan_in, threshold_pct);
            end
        end
    endtask

    task automatic load_baseline_assets();
        string model_path;
        string input_path;
        bit class_found[OUTPUT_CLASSES];
        int classes_seen;

        model_path = $sformatf("%s/%s", BASE_DIR, MNIST_MODEL_DATA_PATH);
        input_path = $sformatf("%s/%s", BASE_DIR, MNIST_TEST_VECTOR_INPUT_PATH);

        baseline_model = new();
        baseline_model.load_from_file(model_path, TOPOLOGY);
        sample_model_profile(baseline_model);

        stim_helper = new(TOPOLOGY[0]);
        stim_helper.load_from_file(input_path);

        for (int i = 0; i < OUTPUT_CLASSES; i++) begin
            class_found[i] = 1'b0;
        end

        classes_seen = 0;
        for (int idx = 0; idx < stim_helper.get_num_vectors(); idx++) begin
            image_t img;
            int predicted_class;

            stim_helper.get_vector(idx, img);
            predicted_class = baseline_model.compute_reference(img);

            if (!class_found[predicted_class]) begin
                copy_image(img, class_images[predicted_class]);
                class_found[predicted_class] = 1'b1;
                classes_seen++;
                if (classes_seen == OUTPUT_CLASSES) begin
                    break;
                end
            end
        end

        if (classes_seen != OUTPUT_CLASSES) begin
            $fatal(1, "Coverage TB could only find %0d/%0d output classes in the scanned MNIST set.",
                classes_seen, OUTPUT_CLASSES);
        end

        active_model = baseline_model;
        dataset_cursor = 0;
    endtask

    task automatic load_random_assets();
        baseline_model = new();
        baseline_model.create_random(TOPOLOGY);
        sample_model_profile(baseline_model);

        stim_helper = new(TOPOLOGY[0]);
        stim_helper.generate_random_vectors(RANDOM_VECTOR_POOL);

        active_model = baseline_model;
        dataset_cursor = 0;
    endtask

    task automatic get_dataset_image(output image_t img);
        int vector_idx;
        vector_idx = dataset_cursor % stim_helper.get_num_vectors();
        stim_helper.get_vector(vector_idx, img);
        dataset_cursor++;
    endtask

    task automatic get_random_image(output image_t img);
        if (stim_helper.get_num_vectors() != 0 && chance_pct(50)) begin
            stim_helper.get_vector(random_range_inclusive(0, stim_helper.get_num_vectors() - 1), img);
        end else begin
            stim_helper.get_random_vector(img);
        end
    endtask

    task automatic clone_model(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) src,
        output BNN_FCC_Model #(CONFIG_BUS_WIDTH) dst
    );
        dst = new();
        dst.topology = src.topology;
        dst.num_layers = src.num_layers;
        dst.weight = new[src.weight.size()];
        dst.threshold = new[src.threshold.size()];

        for (int l = 0; l < src.weight.size(); l++) begin
            dst.weight[l] = new[src.weight[l].size()];
            dst.threshold[l] = new[src.threshold[l].size()];

            for (int n = 0; n < src.weight[l].size(); n++) begin
                dst.weight[l][n] = new[src.weight[l][n].size()];
                for (int i = 0; i < src.weight[l][n].size(); i++) begin
                    dst.weight[l][n][i] = src.weight[l][n][i];
                end
                dst.threshold[l][n] = src.threshold[l][n];
            end
        end

        dst.is_loaded = src.is_loaded;
        dst.outputs_valid = 1'b0;
    endtask

    task automatic randomize_layer_weights(
        ref BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input int layer_idx,
        input int density_pct,
        input int seed
    );
        int fan_in;
        int neuron_count;
        int unsigned seed_state;

        fan_in = TOPOLOGY[layer_idx];
        neuron_count = TOPOLOGY[layer_idx+1];
        seed_state = seed ^ (32'h9E37_79B9 * (layer_idx + 1));

        for (int n = 0; n < neuron_count; n++) begin
            for (int i = 0; i < fan_in; i++) begin
                model.weight[layer_idx][n][i] = (($urandom(seed_state) % 100) < density_pct);
            end
        end
    endtask

    task automatic apply_weight_profile(
        ref BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input int density_pct[NON_INPUT_LAYERS],
        input int seed
    );
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            randomize_layer_weights(model, l, density_pct[l], seed);
        end
    endtask

    task automatic set_layer_thresholds(
        ref BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input int layer_idx,
        input int mode
    );
        int fan_in;
        int neuron_count;

        if (layer_idx >= NON_INPUT_LAYERS - 1) begin
            return;
        end

        fan_in = TOPOLOGY[layer_idx];
        neuron_count = TOPOLOGY[layer_idx+1];

        for (int n = 0; n < neuron_count; n++) begin
            model.threshold[layer_idx][n] = threshold_from_mode(fan_in, n, mode);
        end
    endtask

    task automatic apply_threshold_profile(
        ref BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input int mode[NON_INPUT_LAYERS-1]
    );
        for (int l = 0; l < NON_INPUT_LAYERS - 1; l++) begin
            set_layer_thresholds(model, l, mode[l]);
        end
    endtask

    task automatic queue_output_expectation(
        input int expected_class,
        input ready_relation_e relation,
        input ready_style_e style,
        input int stall_cycles
    );
        output_plan_t plan;

        expected_outputs.push_back(expected_class);
        expected_ready_relations.push_back(relation);
        expected_ready_styles.push_back(style);
        expected_stall_cycles.push_back(stall_cycles);

        plan.relation = relation;
        plan.style = style;
        plan.stall_cycles = stall_cycles;
        driver_output_plans.push_back(plan);
    endtask

    task automatic perform_reset(
        input reset_phase_e phase,
        input bit on_boundary,
        input reset_workload_e workload,
        input post_reset_cfg_e post_reset_cfg
    );
        reset_count++;
        collector.sample_reset(phase, on_boundary, workload, post_reset_cfg, reset_count);

        clear_all_drives();
        expected_outputs.delete();
        expected_ready_relations.delete();
        expected_ready_styles.delete();
        expected_stall_cycles.delete();
        driver_output_plans.delete();
        last_output_class_valid = 1'b0;
        config_message_count = 0;
        prev_config_layer = 0;
        prev_config_is_threshold = 1'b0;

        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        repeat (3) @(posedge clk);
    endtask

    task automatic mutate_threshold_header(ref config_stream_t stream);
        if (stream.size() >= 2) begin
            stream[0][31:16] = 16'hA5C3;
            stream[1][63:32] = 32'hDEAD_BEEF;
        end
    endtask

    task automatic send_layer_message(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input  int layer_idx,
        input  bit is_threshold,
        input  cfg_scope_e scope,
        input  gap_style_e gap_style,
        input  bit mutate_threshold_fields,
        output bit completed,
        input  int reset_after_beat = -1,
        input  bit reset_on_boundary = 1'b0,
        input  bit reset_on_tlast = 1'b0,
        input  reset_phase_e reset_phase = RESET_PHASE_DURING_CONFIG,
        input  reset_workload_e reset_workload = WORKLOAD_CONFIG_ONLY,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        config_stream_t stream;
        config_keep_stream_t keep_stream;
        cfg_order_e order_kind;

        completed = 1'b0;
        model.get_layer_config(layer_idx, is_threshold, stream, keep_stream);
        if (is_threshold && mutate_threshold_fields) begin
            mutate_threshold_header(stream);
        end

        order_kind = cfg_order_e'(classify_order(layer_idx, is_threshold));
        collector.sample_config_message(is_threshold, layer_idx, scope, order_kind,
            is_threshold && mutate_threshold_fields);
        prev_config_layer = layer_idx;
        prev_config_is_threshold = is_threshold;
        config_message_count++;

        for (int beat = 0; beat < stream.size(); beat++) begin
            int gap_cycles;
            int keep_ones;
            bit last_beat;

            gap_cycles = next_gap_cycles(gap_style, beat);
            last_beat = (beat == stream.size() - 1);

            repeat (gap_cycles) begin
                config_in.tvalid = 1'b0;
                @(posedge clk);
            end

            config_in.tvalid = 1'b1;
            config_in.tdata  = stream[beat];
            config_in.tkeep  = keep_stream[beat];
            config_in.tlast  = last_beat;

            do begin
                @(posedge clk);
            end while (!config_in.tready);

            keep_ones = count_ones_keep({248'd0, keep_stream[beat]}, CONFIG_KEEP_WIDTH);
            collector.sample_config_beat(gap_cycles, keep_class(keep_ones, CONFIG_KEEP_WIDTH),
                last_beat);

            if ((reset_after_beat == beat) || (reset_on_tlast && last_beat)) begin
                clear_input_drives();
                perform_reset(reset_phase, reset_on_boundary || last_beat, reset_workload,
                    post_reset_cfg);
                return;
            end
        end

        clear_input_drives();
        completed = 1'b1;
        @(posedge clk);
    endtask

    task automatic send_image(
        input  image_t img,
        input  gap_style_e gap_style,
        input  int inter_image_gap,
        input  int image_count_bucket,
        input  ready_relation_e ready_relation,
        input  ready_style_e ready_style,
        input  int stall_cycles,
        input  bit queue_expected,
        output bit completed,
        input  int reset_after_beat = -1,
        input  bit reset_on_boundary = 1'b0,
        input  bit reset_on_tlast = 1'b0,
        input  reset_phase_e reset_phase = RESET_PHASE_DURING_IMAGE,
        input  reset_workload_e reset_workload = WORKLOAD_PARTIAL_IMAGE,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        if (queue_expected) begin
            int expected_pred;
            expected_pred = active_model.compute_reference(img);
            queue_output_expectation(expected_pred, ready_relation, ready_style, stall_cycles);
        end

        completed = 1'b0;
        repeat (inter_image_gap) @(posedge clk);

        for (int base = 0; base < img.size(); base += INPUTS_PER_CYCLE) begin
            int gap_cycles;
            logic [INPUT_BUS_WIDTH-1:0] beat_data;
            logic [INPUT_KEEP_WIDTH-1:0] beat_keep;
            bit last_beat;

            beat_data = '0;
            beat_keep = '0;
            last_beat = (base + INPUTS_PER_CYCLE >= img.size());

            for (int lane = 0; lane < INPUTS_PER_CYCLE; lane++) begin
                if (base + lane < img.size()) begin
                    beat_data[lane*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH] = img[base + lane];
                    beat_keep[lane*BYTES_PER_INPUT +: BYTES_PER_INPUT] = '1;
                end
            end

            gap_cycles = next_gap_cycles(gap_style, base / INPUTS_PER_CYCLE);
            repeat (gap_cycles) begin
                data_in.tvalid = 1'b0;
                @(posedge clk);
            end

            data_in.tvalid = 1'b1;
            data_in.tdata  = beat_data;
            data_in.tkeep  = beat_keep;
            data_in.tlast  = last_beat;

            do begin
                @(posedge clk);
            end while (!data_in.tready);

            collector.sample_image(gap_cycles,
                keep_class(count_ones_keep({248'd0, beat_keep}, INPUT_KEEP_WIDTH), INPUT_KEEP_WIDTH),
                last_beat, inter_image_gap, image_count_bucket);

            if ((reset_after_beat == (base / INPUTS_PER_CYCLE)) || (reset_on_tlast && last_beat)) begin
                clear_input_drives();
                perform_reset(reset_phase, reset_on_boundary || last_beat, reset_workload,
                    post_reset_cfg);
                return;
            end
        end

        clear_input_drives();
        completed = 1'b1;
    endtask

    task automatic send_layer_message_randomized(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input  int layer_idx,
        input  bit is_threshold,
        input  cfg_scope_e scope,
        input  bit mutate_threshold_fields,
        output bit completed,
        input  int reset_probability_pct = 0,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        config_stream_t stream;
        config_keep_stream_t keep_stream;
        cfg_order_e order_kind;
        int forced_reset_beat;

        completed = 1'b0;
        model.get_layer_config(layer_idx, is_threshold, stream, keep_stream);
        if (is_threshold && mutate_threshold_fields) begin
            mutate_threshold_header(stream);
        end

        order_kind = cfg_order_e'(classify_order(layer_idx, is_threshold));
        collector.sample_config_message(is_threshold, layer_idx, scope, order_kind,
            is_threshold && mutate_threshold_fields);
        prev_config_layer = layer_idx;
        prev_config_is_threshold = is_threshold;
        config_message_count++;
        forced_reset_beat = chance_pct(reset_probability_pct) ?
            random_range_inclusive(0, stream.size() - 1) : -1;

        for (int beat = 0; beat < stream.size(); beat++) begin
            int gap_cycles;
            int keep_ones;
            bit last_beat;

            gap_cycles = random_range_inclusive(0, RANDOM_MAX_IDLE_GAP);
            last_beat = (beat == stream.size() - 1);

            repeat (gap_cycles) begin
                config_in.tvalid = 1'b0;
                @(posedge clk);
            end

            config_in.tvalid = 1'b1;
            config_in.tdata  = stream[beat];
            config_in.tkeep  = keep_stream[beat];
            config_in.tlast  = last_beat;

            do begin
                @(posedge clk);
            end while (!config_in.tready);

            keep_ones = count_ones_keep({248'd0, keep_stream[beat]}, CONFIG_KEEP_WIDTH);
            collector.sample_config_beat(gap_cycles, keep_class(keep_ones, CONFIG_KEEP_WIDTH),
                last_beat);

            if (beat == forced_reset_beat) begin
                clear_input_drives();
                perform_reset(RESET_PHASE_DURING_CONFIG, last_beat, WORKLOAD_CONFIG_ONLY,
                    post_reset_cfg);
                return;
            end
        end

        clear_input_drives();
        completed = 1'b1;
        @(posedge clk);
    endtask

    task automatic send_image_randomized(
        input  image_t img,
        input  int inter_image_gap,
        input  int image_count_bucket,
        input  ready_relation_e ready_relation,
        input  ready_style_e ready_style,
        input  int stall_cycles,
        input  bit queue_expected,
        output bit completed,
        input  int reset_probability_pct = 0,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        int forced_reset_beat;

        if (queue_expected) begin
            int expected_pred;
            expected_pred = active_model.compute_reference(img);
            queue_output_expectation(expected_pred, ready_relation, ready_style, stall_cycles);
        end

        completed = 1'b0;
        forced_reset_beat = chance_pct(reset_probability_pct) ?
            random_range_inclusive(0, ((img.size() + INPUTS_PER_CYCLE - 1) / INPUTS_PER_CYCLE) - 1) :
            -1;
        repeat (inter_image_gap) @(posedge clk);

        for (int base = 0; base < img.size(); base += INPUTS_PER_CYCLE) begin
            int gap_cycles;
            logic [INPUT_BUS_WIDTH-1:0] beat_data;
            logic [INPUT_KEEP_WIDTH-1:0] beat_keep;
            bit last_beat;

            beat_data = '0;
            beat_keep = '0;
            last_beat = (base + INPUTS_PER_CYCLE >= img.size());

            for (int lane = 0; lane < INPUTS_PER_CYCLE; lane++) begin
                if (base + lane < img.size()) begin
                    beat_data[lane*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH] = img[base + lane];
                    beat_keep[lane*BYTES_PER_INPUT +: BYTES_PER_INPUT] = '1;
                end
            end

            gap_cycles = random_range_inclusive(0, RANDOM_MAX_IDLE_GAP);
            repeat (gap_cycles) begin
                data_in.tvalid = 1'b0;
                @(posedge clk);
            end

            data_in.tvalid = 1'b1;
            data_in.tdata  = beat_data;
            data_in.tkeep  = beat_keep;
            data_in.tlast  = last_beat;

            do begin
                @(posedge clk);
            end while (!data_in.tready);

            collector.sample_image(gap_cycles,
                keep_class(count_ones_keep({248'd0, beat_keep}, INPUT_KEEP_WIDTH), INPUT_KEEP_WIDTH),
                last_beat, inter_image_gap, image_count_bucket);

            if ((base / INPUTS_PER_CYCLE) == forced_reset_beat) begin
                clear_input_drives();
                perform_reset(RESET_PHASE_DURING_IMAGE, last_beat, WORKLOAD_PARTIAL_IMAGE,
                    post_reset_cfg);
                return;
            end
        end

        clear_input_drives();
        completed = 1'b1;
    endtask

    task automatic reset_while_output_pending(
        input post_reset_cfg_e post_reset_cfg
    );
        wait (data_out.tvalid === 1'b1);
        perform_reset(RESET_PHASE_DURING_OUTPUT, 1'b0, WORKLOAD_IMAGES_IN_FLIGHT, post_reset_cfg);
    endtask

    task automatic send_full_config_default(
        input BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input cfg_scope_e scope,
        input gap_style_e weight_gap_style,
        input gap_style_e threshold_gap_style,
        input bit mutate_threshold_fields
    );
        bit done_ok;
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            send_layer_message(model, l, 1'b0, scope, weight_gap_style, 1'b0, done_ok);
            if (l < NON_INPUT_LAYERS - 1) begin
                send_layer_message(model, l, 1'b1, scope, threshold_gap_style,
                    mutate_threshold_fields, done_ok);
            end
        end
    endtask

    task automatic send_full_config_reordered(input BNN_FCC_Model #(CONFIG_BUS_WIDTH) model);
        bit done_ok;
        for (int l = NON_INPUT_LAYERS - 2; l >= 0; l--) begin
            send_layer_message(model, l, 1'b1, CFG_SCOPE_FULL,
                (l % 2) ? GAP_INTERMITTENT : GAP_BURSTY, (l == NON_INPUT_LAYERS - 2), done_ok);
        end
        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            send_layer_message(model, l, 1'b0, CFG_SCOPE_FULL,
                (l == 0) ? GAP_CONTINUOUS : ((l % 2) ? GAP_INTERMITTENT : GAP_BURSTY),
                1'b0, done_ok);
        end
    endtask

    task automatic queue_config_descriptor(
        ref int layer_ids[$],
        ref bit is_threshold_flags[$],
        ref bit mutate_flags[$],
        input int layer_idx,
        input bit is_threshold,
        input bit mutate_threshold_fields
    );
        layer_ids.push_back(layer_idx);
        is_threshold_flags.push_back(is_threshold);
        mutate_flags.push_back(mutate_threshold_fields);
    endtask

    task automatic send_config_queue_randomized(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        ref    int layer_ids[$],
        ref    bit is_threshold_flags[$],
        ref    bit mutate_flags[$],
        input  cfg_scope_e scope,
        output bit completed,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        completed = 1'b1;
        while (layer_ids.size() != 0) begin
            int pick_idx;
            int layer_idx;
            bit is_threshold;
            bit mutate_threshold_fields;
            bit message_done;

            pick_idx = random_range_inclusive(0, layer_ids.size() - 1);
            layer_idx = layer_ids[pick_idx];
            is_threshold = is_threshold_flags[pick_idx];
            mutate_threshold_fields = mutate_flags[pick_idx];

            layer_ids.delete(pick_idx);
            is_threshold_flags.delete(pick_idx);
            mutate_flags.delete(pick_idx);

            send_layer_message_randomized(model, layer_idx, is_threshold, scope,
                mutate_threshold_fields, message_done, RANDOM_RESET_PROBABILITY_PCT, post_reset_cfg);
            if (!message_done) begin
                completed = 1'b0;
                return;
            end
        end
    endtask

    task automatic send_full_config_randomized(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        output bit completed,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        int layer_ids[$];
        bit is_threshold_flags[$];
        bit mutate_flags[$];

        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, l, 1'b0, 1'b0);
            if (l < NON_INPUT_LAYERS - 1) begin
                queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, l, 1'b1,
                    chance_pct(50));
            end
        end

        if (layer_ids.size() != 0 && chance_pct(35)) begin
            int dup_idx;
            dup_idx = random_range_inclusive(0, layer_ids.size() - 1);
            queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, layer_ids[dup_idx],
                is_threshold_flags[dup_idx], mutate_flags[dup_idx]);
        end

        send_config_queue_randomized(model, layer_ids, is_threshold_flags, mutate_flags,
            CFG_SCOPE_FULL, completed, post_reset_cfg);
    endtask

    task automatic build_random_variant(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) src_model,
        output BNN_FCC_Model #(CONFIG_BUS_WIDTH) dst_model,
        output cfg_scope_e update_scope,
        output int target_layer
    );
        int density_pct[NON_INPUT_LAYERS];
        int threshold_mode[NON_INPUT_LAYERS-1];
        int random_seed;

        clone_model(src_model, dst_model);
        random_seed = $urandom();

        if (NON_INPUT_LAYERS <= 1) begin
            update_scope = cfg_scope_e'(random_range_inclusive(CFG_SCOPE_FULL,
                CFG_SCOPE_WEIGHTS_ONLY));
        end else begin
            update_scope = cfg_scope_e'(random_range_inclusive(CFG_SCOPE_FULL,
                CFG_SCOPE_PARTIAL_BOTH));
        end

        target_layer = random_range_inclusive(0, NON_INPUT_LAYERS - 1);
        if ((update_scope == CFG_SCOPE_THRESHOLDS_ONLY || update_scope == CFG_SCOPE_PARTIAL_BOTH) &&
            target_layer >= NON_INPUT_LAYERS - 1) begin
            target_layer = NON_INPUT_LAYERS - 2;
        end

        for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
            density_pct[l] = random_range_inclusive(0, 100);
        end
        for (int l = 0; l < NON_INPUT_LAYERS - 1; l++) begin
            threshold_mode[l] = random_range_inclusive(0, 2);
        end

        case (update_scope)
            CFG_SCOPE_FULL: begin
                apply_weight_profile(dst_model, density_pct, random_seed);
                if (NON_INPUT_LAYERS > 1) begin
                    apply_threshold_profile(dst_model, threshold_mode);
                end
            end
            CFG_SCOPE_WEIGHTS_ONLY: begin
                apply_weight_profile(dst_model, density_pct, random_seed);
            end
            CFG_SCOPE_THRESHOLDS_ONLY: begin
                apply_threshold_profile(dst_model, threshold_mode);
            end
            CFG_SCOPE_PARTIAL_SUBSET: begin
                randomize_layer_weights(dst_model, target_layer, random_range_inclusive(0, 100),
                    random_seed ^ target_layer);
            end
            CFG_SCOPE_PARTIAL_BOTH: begin
                randomize_layer_weights(dst_model, target_layer, random_range_inclusive(0, 100),
                    random_seed ^ target_layer);
                set_layer_thresholds(dst_model, target_layer, random_range_inclusive(0, 2));
            end
        endcase
    endtask

    task automatic send_random_scope_update(
        input  BNN_FCC_Model #(CONFIG_BUS_WIDTH) model,
        input  cfg_scope_e update_scope,
        input  int target_layer,
        output bit completed,
        input  post_reset_cfg_e post_reset_cfg = POST_RESET_DIFFERENT_CONFIG
    );
        int layer_ids[$];
        bit is_threshold_flags[$];
        bit mutate_flags[$];

        case (update_scope)
            CFG_SCOPE_FULL: begin
                send_full_config_randomized(model, completed, post_reset_cfg);
                return;
            end
            CFG_SCOPE_WEIGHTS_ONLY: begin
                for (int l = 0; l < NON_INPUT_LAYERS; l++) begin
                    queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, l, 1'b0,
                        1'b0);
                end
            end
            CFG_SCOPE_THRESHOLDS_ONLY: begin
                for (int l = 0; l < NON_INPUT_LAYERS - 1; l++) begin
                    queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, l, 1'b1,
                        chance_pct(50));
                end
            end
            CFG_SCOPE_PARTIAL_SUBSET: begin
                queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, target_layer,
                    1'b0, 1'b0);
                if (chance_pct(35)) begin
                    queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, target_layer,
                        1'b0, 1'b0);
                end
            end
            CFG_SCOPE_PARTIAL_BOTH: begin
                queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, target_layer,
                    1'b1, chance_pct(50));
                queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, target_layer,
                    1'b0, 1'b0);
                if (chance_pct(35)) begin
                    queue_config_descriptor(layer_ids, is_threshold_flags, mutate_flags, target_layer,
                        chance_pct(50), chance_pct(50));
                end
            end
        endcase

        send_config_queue_randomized(model, layer_ids, is_threshold_flags, mutate_flags,
            update_scope, completed, post_reset_cfg);
    endtask

    task automatic run_full_reordered();
        bit completed;

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        send_full_config_reordered(active_model);

        send_image(class_images[0], GAP_CONTINUOUS, 0, 12, READY_BEFORE_VALID,
            READY_STYLE_CONTINUOUS, 0, 1'b1, completed);
        send_image(class_images[0], GAP_INTERMITTENT, 1, 12, READY_BEFORE_VALID,
            READY_STYLE_INTERMITTENT, 0, 1'b1, completed);
        send_image(class_images[1], GAP_BURSTY, 2, 12, READY_BEFORE_VALID,
            READY_STYLE_BURSTY, 0, 1'b1, completed);
        send_image(class_images[2], GAP_CONTINUOUS, 0, 12, READY_SAME_CYCLE,
            READY_STYLE_CONTINUOUS, 0, 1'b1, completed);
        send_image(class_images[3], GAP_INTERMITTENT, 1, 12, READY_SAME_CYCLE,
            READY_STYLE_INTERMITTENT, 0, 1'b1, completed);
        send_image(class_images[3], GAP_BURSTY, 4, 12, READY_SAME_CYCLE,
            READY_STYLE_BURSTY, 0, 1'b1, completed);
        send_image(class_images[4], GAP_CONTINUOUS, 0, 12, READY_AFTER_VALID,
            READY_STYLE_CONTINUOUS, 1, 1'b1, completed);
        send_image(class_images[5], GAP_INTERMITTENT, 2, 12, READY_AFTER_VALID,
            READY_STYLE_INTERMITTENT, 2, 1'b1, completed);
        send_image(class_images[6], GAP_BURSTY, 1, 12, READY_AFTER_VALID,
            READY_STYLE_BURSTY, 4, 1'b1, completed);
        send_image(class_images[7], GAP_CONTINUOUS, 0, 12, READY_BEFORE_VALID,
            READY_STYLE_CONTINUOUS, 0, 1'b1, completed);
        send_image(class_images[8], GAP_INTERMITTENT, 1, 12, READY_SAME_CYCLE,
            READY_STYLE_CONTINUOUS, 0, 1'b1, completed);
        send_image(class_images[9], GAP_BURSTY, 3, 12, READY_AFTER_VALID,
            READY_STYLE_CONTINUOUS, 1, 1'b1, completed);
    endtask

    task automatic run_weights_only_reconfig();
        image_t img;
        bit completed;
        int weight_density[NON_INPUT_LAYERS];

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_INTERMITTENT, GAP_BURSTY, 1'b1);

        get_dataset_image(img);
        send_image(img, GAP_CONTINUOUS, 0, 2, READY_BEFORE_VALID, READY_STYLE_CONTINUOUS, 0,
            1'b1, completed);
        get_dataset_image(img);
        send_image(img, GAP_INTERMITTENT, 1, 2, READY_AFTER_VALID, READY_STYLE_INTERMITTENT, 2,
            1'b1, completed);

        clone_model(baseline_model, variant_model);
        weight_density = '{10, 55, 90};
        apply_weight_profile(variant_model, weight_density, 32'h2201);
        sample_model_profile(variant_model);
        active_model = variant_model;

        perform_reset(RESET_PHASE_AFTER_CONFIG, 1'b1, WORKLOAD_CONFIG_ONLY,
            POST_RESET_DIFFERENT_CONFIG);
        send_layer_message(active_model, 2, 1'b0, CFG_SCOPE_WEIGHTS_ONLY, GAP_INTERMITTENT, 1'b0,
            completed);
        send_layer_message(active_model, 1, 1'b0, CFG_SCOPE_WEIGHTS_ONLY, GAP_BURSTY, 1'b0,
            completed);
        send_layer_message(active_model, 0, 1'b0, CFG_SCOPE_WEIGHTS_ONLY, GAP_CONTINUOUS, 1'b0,
            completed);
        send_layer_message(active_model, 0, 1'b0, CFG_SCOPE_WEIGHTS_ONLY, GAP_INTERMITTENT, 1'b0,
            completed);

        repeat (3) begin
            get_dataset_image(img);
            send_image(img, GAP_BURSTY, 1, 3, READY_SAME_CYCLE, READY_STYLE_BURSTY, 0, 1'b1,
                completed);
        end
    endtask

    task automatic run_thresholds_only_reconfig();
        image_t img;
        bit completed;
        int threshold_mode[NON_INPUT_LAYERS-1];

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_CONTINUOUS, GAP_INTERMITTENT,
            1'b0);

        get_dataset_image(img);
        send_image(img, GAP_INTERMITTENT, 0, 1, READY_BEFORE_VALID, READY_STYLE_INTERMITTENT, 0,
            1'b1, completed);

        clone_model(baseline_model, variant_model);
        threshold_mode = '{0, 2};
        apply_threshold_profile(variant_model, threshold_mode);
        sample_model_profile(variant_model);
        active_model = variant_model;

        perform_reset(RESET_PHASE_AFTER_CONFIG, 1'b1, WORKLOAD_CONFIG_ONLY,
            POST_RESET_DIFFERENT_CONFIG);
        send_layer_message(active_model, 1, 1'b1, CFG_SCOPE_THRESHOLDS_ONLY, GAP_BURSTY, 1'b1,
            completed);
        send_layer_message(active_model, 0, 1'b1, CFG_SCOPE_THRESHOLDS_ONLY, GAP_INTERMITTENT,
            1'b0, completed);

        repeat (3) begin
            get_dataset_image(img);
            send_image(img, GAP_CONTINUOUS, 2, 3, READY_AFTER_VALID, READY_STYLE_CONTINUOUS, 3,
                1'b1, completed);
        end
    endtask

    task automatic run_partial_subset_reconfig();
        image_t img;
        bit completed;

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_CONTINUOUS, GAP_INTERMITTENT,
            1'b1);

        get_dataset_image(img);
        send_image(img, GAP_CONTINUOUS, 0, 1, READY_BEFORE_VALID, READY_STYLE_CONTINUOUS, 0,
            1'b1, completed);

        clone_model(baseline_model, variant_model);
        randomize_layer_weights(variant_model, 1, 85, 32'h4401);
        sample_model_profile(variant_model);
        active_model = variant_model;

        perform_reset(RESET_PHASE_AFTER_CONFIG, 1'b1, WORKLOAD_CONFIG_ONLY,
            POST_RESET_DIFFERENT_CONFIG);
        send_layer_message(active_model, 1, 1'b0, CFG_SCOPE_PARTIAL_SUBSET, GAP_BURSTY, 1'b0,
            completed);

        repeat (2) begin
            get_dataset_image(img);
            send_image(img, GAP_INTERMITTENT, 1, 2, READY_SAME_CYCLE, READY_STYLE_CONTINUOUS, 0,
                1'b1, completed);
        end

        clone_model(active_model, variant_model);
        set_layer_thresholds(variant_model, 1, 0);
        randomize_layer_weights(variant_model, 1, 25, 32'h4402);
        sample_model_profile(variant_model);
        active_model = variant_model;

        perform_reset(RESET_PHASE_AFTER_CONFIG, 1'b1, WORKLOAD_CONFIG_ONLY,
            POST_RESET_DIFFERENT_CONFIG);
        send_layer_message(active_model, 1, 1'b1, CFG_SCOPE_PARTIAL_BOTH, GAP_INTERMITTENT, 1'b1,
            completed);
        send_layer_message(active_model, 1, 1'b0, CFG_SCOPE_PARTIAL_BOTH, GAP_BURSTY, 1'b0,
            completed);

        repeat (3) begin
            get_dataset_image(img);
            send_image(img, GAP_BURSTY, 1, 3, READY_AFTER_VALID, READY_STYLE_BURSTY, 4, 1'b1,
                completed);
        end
    endtask

    task automatic run_reset_mid_config();
        image_t img;
        bit completed;

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;

        send_layer_message(active_model, 0, 1'b0, CFG_SCOPE_FULL, GAP_BURSTY, 1'b0, completed,
            8, 1'b0, 1'b0, RESET_PHASE_DURING_CONFIG, WORKLOAD_CONFIG_ONLY,
            POST_RESET_SAME_CONFIG);

        send_layer_message(active_model, 0, 1'b0, CFG_SCOPE_FULL, GAP_CONTINUOUS, 1'b0, completed);
        send_layer_message(active_model, 0, 1'b1, CFG_SCOPE_FULL, GAP_INTERMITTENT, 1'b1,
            completed, -1, 1'b1, 1'b1, RESET_PHASE_DURING_CONFIG, WORKLOAD_CONFIG_ONLY,
            POST_RESET_SAME_CONFIG);

        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_INTERMITTENT, GAP_BURSTY, 1'b1);

        repeat (3) begin
            get_dataset_image(img);
            send_image(img, GAP_CONTINUOUS, 1, 3, READY_AFTER_VALID, READY_STYLE_INTERMITTENT, 1,
                1'b1, completed);
        end
    endtask

    task automatic run_reset_mid_image_and_output();
        image_t img;
        bit completed;

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_INTERMITTENT, GAP_BURSTY, 1'b1);

        get_dataset_image(img);
        send_image(img, GAP_INTERMITTENT, 0, 5, READY_AFTER_VALID, READY_STYLE_CONTINUOUS, 1,
            1'b0, completed, 3, 1'b0, 1'b0, RESET_PHASE_DURING_IMAGE, WORKLOAD_PARTIAL_IMAGE,
            POST_RESET_SAME_CONFIG);

        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_CONTINUOUS, GAP_INTERMITTENT,
            1'b0);

        get_dataset_image(img);
        send_image(img, GAP_CONTINUOUS, 1, 5, READY_BEFORE_VALID, READY_STYLE_CONTINUOUS, 0,
            1'b0, completed, -1, 1'b1, 1'b1, RESET_PHASE_DURING_IMAGE, WORKLOAD_PARTIAL_IMAGE,
            POST_RESET_SAME_CONFIG);

        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_BURSTY, GAP_INTERMITTENT, 1'b1);

        get_dataset_image(img);
        send_image(img, GAP_BURSTY, 2, 5, READY_BEFORE_VALID, READY_STYLE_CONTINUOUS, 0, 1'b0,
            completed);
        reset_while_output_pending(POST_RESET_SAME_CONFIG);

        send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_CONTINUOUS, GAP_BURSTY, 1'b0);

        repeat (3) begin
            get_dataset_image(img);
            send_image(img, GAP_INTERMITTENT, 2, 3, READY_AFTER_VALID, READY_STYLE_BURSTY, 4,
                1'b1, completed);
        end
    endtask

    task automatic run_randomized_stress();
        image_t img;
        bit completed;
        bit needs_full_config;
        bit force_variant_before_config;
        cfg_scope_e update_scope;
        int target_layer;

        perform_reset(RESET_PHASE_BEFORE_CONFIG, 1'b0, WORKLOAD_NONE, POST_RESET_SAME_CONFIG);
        active_model = baseline_model;
        needs_full_config = 1'b1;
        force_variant_before_config = 1'b0;

        for (int iter = 0; iter < RANDOM_STRESS_ITERATIONS; iter++) begin
            post_reset_cfg_e pre_cfg_reset_policy;

            if (chance_pct(RANDOM_RESET_PROBABILITY_PCT)) begin
                pre_cfg_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;
                perform_reset(RESET_PHASE_BEFORE_CONFIG, chance_pct(50), WORKLOAD_NONE,
                    pre_cfg_reset_policy);
                needs_full_config = 1'b1;
                force_variant_before_config = (pre_cfg_reset_policy == POST_RESET_DIFFERENT_CONFIG);
            end

            if (force_variant_before_config) begin
                build_random_variant(active_model, variant_model, update_scope, target_layer);
                active_model = variant_model;
                sample_model_profile(active_model);
                force_variant_before_config = 1'b0;
            end

            if (needs_full_config) begin
                post_reset_cfg_e cfg_reset_policy;

                cfg_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;
                send_full_config_randomized(active_model, completed, cfg_reset_policy);
                if (!completed) begin
                    needs_full_config = 1'b1;
                    if (cfg_reset_policy == POST_RESET_DIFFERENT_CONFIG) begin
                        build_random_variant(active_model, variant_model, update_scope, target_layer);
                        active_model = variant_model;
                        sample_model_profile(active_model);
                    end
                    continue;
                end
                needs_full_config = 1'b0;
            end

            if (chance_pct(65)) begin
                post_reset_cfg_e update_reset_policy;

                build_random_variant(active_model, variant_model, update_scope, target_layer);
                active_model = variant_model;
                sample_model_profile(active_model);

                update_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;
                send_random_scope_update(active_model, update_scope, target_layer, completed,
                    update_reset_policy);
                if (!completed) begin
                    needs_full_config = 1'b1;
                    if (update_reset_policy == POST_RESET_DIFFERENT_CONFIG) begin
                        build_random_variant(active_model, variant_model, update_scope, target_layer);
                        active_model = variant_model;
                        sample_model_profile(active_model);
                    end
                    continue;
                end
            end

            if (chance_pct(RANDOM_RESET_PROBABILITY_PCT)) begin
                post_reset_cfg_e post_cfg_reset_policy;

                post_cfg_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;
                perform_reset(RESET_PHASE_AFTER_CONFIG, chance_pct(50), WORKLOAD_CONFIG_ONLY,
                    post_cfg_reset_policy);
                needs_full_config = 1'b1;
                if (post_cfg_reset_policy == POST_RESET_DIFFERENT_CONFIG) begin
                    build_random_variant(active_model, variant_model, update_scope, target_layer);
                    active_model = variant_model;
                    sample_model_profile(active_model);
                end
                continue;
            end

            get_random_image(img);
            begin
                ready_relation_e relation;
                ready_style_e style;
                int stall_cycles;
                int inter_image_gap;
                int image_count_bucket;
                post_reset_cfg_e image_reset_policy;

                relation = random_ready_relation();
                style = random_ready_style();
                stall_cycles = random_output_stall(relation);

                // Force the last remaining output cross once per seed sweep.
                if ((iter % 6) == 0) begin
                    relation = READY_AFTER_VALID;
                    stall_cycles = 0;
                end

                inter_image_gap = random_range_inclusive(0, RANDOM_MAX_IDLE_GAP);
                image_count_bucket = random_range_inclusive(1, 6);
                image_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;

                send_image_randomized(img, inter_image_gap, image_count_bucket, relation, style,
                    stall_cycles, 1'b1, completed, RANDOM_RESET_PROBABILITY_PCT / 2,
                    image_reset_policy);
                if (!completed) begin
                    needs_full_config = 1'b1;
                    if (image_reset_policy == POST_RESET_DIFFERENT_CONFIG) begin
                        build_random_variant(active_model, variant_model, update_scope, target_layer);
                        active_model = variant_model;
                        sample_model_profile(active_model);
                    end
                    continue;
                end
            end

            if (chance_pct(RANDOM_RESET_PROBABILITY_PCT)) begin
                post_reset_cfg_e output_reset_policy;

                output_reset_policy = chance_pct(50) ? POST_RESET_SAME_CONFIG :
                    POST_RESET_DIFFERENT_CONFIG;
                reset_while_output_pending(output_reset_policy);
                needs_full_config = 1'b1;
                if (output_reset_policy == POST_RESET_DIFFERENT_CONFIG) begin
                    build_random_variant(active_model, variant_model, update_scope, target_layer);
                    active_model = variant_model;
                    sample_model_profile(active_model);
                end
            end
        end

        if (output_handshake_count == 0) begin
            if (needs_full_config) begin
                send_full_config_default(active_model, CFG_SCOPE_FULL, GAP_CONTINUOUS,
                    GAP_INTERMITTENT, 1'b0);
            end

            get_random_image(img);
            send_image(img, GAP_CONTINUOUS, 0, 1, READY_AFTER_VALID, READY_STYLE_CONTINUOUS, 0,
                1'b1, completed);
        end
    endtask

    initial begin : output_monitor
        forever begin
            int actual_class;
            int expected_class;
            ready_relation_e relation;
            ready_style_e style;
            int stall_cycles;
            bit repeated_class;

            @(posedge clk iff (data_out.tvalid && data_out.tready && !rst));
            output_handshake_count++;

            if (expected_outputs.size() == 0) begin
                failed++;
                $error("Unexpected output %0d observed with no queued expectation.",
                    int'(data_out.tdata[OUTPUT_DATA_WIDTH-1:0]));
                continue;
            end
            if (expected_ready_relations.size() == 0 || expected_ready_styles.size() == 0 ||
                expected_stall_cycles.size() == 0) begin
                failed++;
                $error("Output %0d observed before ready metadata was queued.",
                    int'(data_out.tdata[OUTPUT_DATA_WIDTH-1:0]));
                continue;
            end

            expected_class = expected_outputs.pop_front();
            relation = expected_ready_relations.pop_front();
            style = expected_ready_styles.pop_front();
            stall_cycles = expected_stall_cycles.pop_front();
            actual_class = int'(data_out.tdata[OUTPUT_DATA_WIDTH-1:0]);
            repeated_class = last_output_class_valid && (actual_class == last_output_class);

            if (!data_out.tlast) begin
                failed++;
                $error("Output handshake occurred without TLAST asserted.");
            end

            if (actual_class == expected_class) begin
                passed++;
            end else begin
                failed++;
                $error("Output mismatch: actual=%0d expected=%0d", actual_class, expected_class);
            end

            collector.sample_output(actual_class, repeated_class, relation, style, stall_cycles,
                &data_out.tkeep);
            last_output_class = actual_class;
            last_output_class_valid = 1'b1;
        end
    end

    initial begin : output_ready_driver
        clear_all_drives();
        output_driver_state = OUTDRV_IDLE;
        active_ready_cycle = 0;
        remaining_stall_cycles = 0;
        driver_handshakes_seen = 0;

        forever begin
            @(negedge clk);

            if (driver_handshakes_seen != output_handshake_count) begin
                int handshakes_to_pop;
                handshakes_to_pop = output_handshake_count - driver_handshakes_seen;
                repeat (handshakes_to_pop) begin
                    if (driver_output_plans.size() != 0) begin
                        void'(driver_output_plans.pop_front());
                    end
                end
                driver_handshakes_seen = output_handshake_count;
                output_driver_state = OUTDRV_IDLE;
                active_ready_cycle = 0;
                remaining_stall_cycles = 0;
                data_out.tready = 1'b0;
            end

            if (rst) begin
                output_driver_state = OUTDRV_IDLE;
                active_ready_cycle = 0;
                remaining_stall_cycles = 0;
                driver_handshakes_seen = output_handshake_count;
                data_out.tready = 1'b0;
            end else if (driver_output_plans.size() == 0) begin
                output_driver_state = OUTDRV_IDLE;
                active_ready_cycle = 0;
                remaining_stall_cycles = 0;
                data_out.tready = 1'b0;
            end else begin
                active_output_plan = driver_output_plans[0];

                case (output_driver_state)
                    OUTDRV_IDLE: begin
                        active_ready_cycle = 0;
                        remaining_stall_cycles = active_output_plan.stall_cycles;
                        case (active_output_plan.relation)
                            READY_BEFORE_VALID: begin
                                output_driver_state = OUTDRV_ACTIVE;
                                data_out.tready = next_ready_value(active_output_plan.style,
                                    active_ready_cycle);
                                active_ready_cycle++;
                            end
                            READY_SAME_CYCLE: begin
                                output_driver_state = OUTDRV_WAIT_VALID;
                                data_out.tready = 1'b0;
                            end
                            READY_AFTER_VALID: begin
                                output_driver_state = OUTDRV_WAIT_VALID;
                                data_out.tready = 1'b0;
                            end
                        endcase
                    end

                    OUTDRV_WAIT_VALID: begin
                        data_out.tready = 1'b0;
                        if (data_out.tvalid) begin
                            if (active_output_plan.relation == READY_SAME_CYCLE) begin
                                output_driver_state = OUTDRV_ACTIVE;
                                active_ready_cycle = 0;
                                data_out.tready = next_ready_value(active_output_plan.style,
                                    active_ready_cycle);
                                active_ready_cycle++;
                            end else if (remaining_stall_cycles == 0) begin
                                output_driver_state = OUTDRV_ACTIVE;
                                active_ready_cycle = 0;
                                data_out.tready = next_ready_value(active_output_plan.style,
                                    active_ready_cycle);
                                active_ready_cycle++;
                            end else begin
                                output_driver_state = OUTDRV_STALL;
                                data_out.tready = 1'b0;
                                remaining_stall_cycles--;
                            end
                        end
                    end

                    OUTDRV_STALL: begin
                        data_out.tready = 1'b0;
                        if (remaining_stall_cycles == 0) begin
                            output_driver_state = OUTDRV_ACTIVE;
                            active_ready_cycle = 0;
                            data_out.tready = next_ready_value(active_output_plan.style,
                                active_ready_cycle);
                            active_ready_cycle++;
                        end else begin
                            remaining_stall_cycles--;
                        end
                    end

                    OUTDRV_ACTIVE: begin
                        data_out.tready = next_ready_value(active_output_plan.style,
                            active_ready_cycle);
                        active_ready_cycle++;
                    end
                endcase
            end
        end
    end

    initial begin : main
        collector = new(NON_INPUT_LAYERS, OUTPUT_CLASSES, config_partial_keep_possible_f(),
            image_partial_keep_possible_f());
        clear_all_drives();
        rst = 1'b0;
        void'($urandom(RANDOM_SEED));
        passed = 0;
        failed = 0;
        reset_count = 0;
        config_message_count = 0;
        prev_config_layer = 0;
        prev_config_is_threshold = 1'b0;
        last_output_class_valid = 1'b0;
        output_handshake_count = 0;
        driver_handshakes_seen = 0;
        dataset_cursor = 0;

        $timeformat(-9, 0, " ns", 0);
        collector.sample_scenario(SCENARIO_ID);
        $display("[%0t] Starting coverage scenario %0d (%s)", $realtime, SCENARIO_ID,
            scenario_name(SCENARIO_ID));

        if (USE_CUSTOM_TOPOLOGY && (SCENARIO_ID != 6)) begin
            $fatal(1, "Custom topology coverage runs are only supported for SCENARIO_ID=6.");
        end

        if (SCENARIO_ID == 6) begin
            load_random_assets();
        end else begin
            load_baseline_assets();
        end

        case (SCENARIO_ID)
            0: run_full_reordered();
            1: run_weights_only_reconfig();
            2: run_thresholds_only_reconfig();
            3: run_partial_subset_reconfig();
            4: run_reset_mid_config();
            5: run_reset_mid_image_and_output();
            6: run_randomized_stress();
            default: $fatal(1, "Unknown SCENARIO_ID=%0d", SCENARIO_ID);
        endcase

        wait (expected_outputs.size() == 0);
        wait (driver_output_plans.size() == 0);
        repeat (10) @(posedge clk);

        if (failed != 0) begin
            $fatal(1, "Coverage scenario %0d FAILED: %0d issues detected.", SCENARIO_ID, failed);
        end

        $display("[%0t] Coverage scenario %0d PASSED with %0d checked outputs.",
            $realtime, SCENARIO_ID, passed);

        disable timeout_watchdog;
        $finish;
    end

    initial begin : timeout_watchdog
        #TIMEOUT;
        $fatal(1, "Coverage simulation timed out after %0t.", TIMEOUT);
    end

endmodule
