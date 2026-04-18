// -----------------------------------------------------------------------------
// Top-level streaming BNN fully-connected classifier (FCC).
//
// Responsibilities:
// - AXI4-Stream config input:
//     Parse FINN-style messages and write weights/thresholds into per-layer RAMs.
// - AXI4-Stream image input:
//     Binarize pixels (>= 128 -> 1 else 0) and buffer one image.
// - Inference:
//     Run each layer sequentially (input -> hidden(s) -> output) using a
//     `layer_memory` + `compute_layer` pair per layer.
// - AXI4-Stream output:
//     Argmax across the output-layer popcounts and emit one beat containing the
//     predicted class index.
//
// Note: This implementation buffers a full image before starting inference and
// processes one image at a time. This is simple and contest-friendly, but it
// constrains throughput compared to a fully streaming pipeline.
// -----------------------------------------------------------------------------
module bnn_fcc #(
    parameter int INPUT_DATA_WIDTH  = 8,
    parameter int INPUT_BUS_WIDTH   = 64,
    parameter int CONFIG_BUS_WIDTH  = 64,
    parameter int OUTPUT_DATA_WIDTH = 4,
    parameter int OUTPUT_BUS_WIDTH  = 8,

    parameter int TOTAL_LAYERS = 4,
    parameter int TOPOLOGY[TOTAL_LAYERS] = '{
        0: 784,
        1: 256,
        2: 256,
        3: 10,
        default: 0
    },

    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[TOTAL_LAYERS-1] = '{default: 8}
) (
    input logic clk,
    input logic rst,

    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,

    input  logic                         data_in_valid,
    output logic                         data_in_ready,
    input  logic [  INPUT_BUS_WIDTH-1:0] data_in_data,
    input  logic [INPUT_BUS_WIDTH/8-1:0] data_in_keep,
    input  logic                         data_in_last,

    output logic                          data_out_valid,
    input  logic                          data_out_ready,
    output logic [  OUTPUT_BUS_WIDTH-1:0] data_out_data,
    output logic [OUTPUT_BUS_WIDTH/8-1:0] data_out_keep,
    output logic                          data_out_last
);
    import bnn_util_pkg::*;

    // -------------------------------------------------------------------------
    // Topology-derived constants
    // -------------------------------------------------------------------------
    localparam int LAYERS = TOTAL_LAYERS - 1;
    localparam int INPUT_BUS_ELEMENTS = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH;
    localparam int INPUT_BINARIZATION_THRESHOLD = 1 << (INPUT_DATA_WIDTH - 1);
    localparam int OUTPUT_LAYER = TOTAL_LAYERS - 1;
    localparam int OUTPUT_NEURONS = TOPOLOGY[OUTPUT_LAYER];
    localparam int OUTPUT_INDEX_WIDTH = clog2_safe(OUTPUT_NEURONS);
    localparam int OUTPUT_SCORE_WIDTH = clog2_safe(TOPOLOGY[OUTPUT_LAYER-1] + 1);
    localparam int INPUT_BUFFER_WORDS = div_ceil(TOPOLOGY[0], INPUT_BUS_ELEMENTS);
    localparam int INPUT_WORD_IDX_WIDTH = clog2_safe(INPUT_BUFFER_WORDS);

    // Choose a single "activation storage" width large enough to hold:
    // - Layer 0: TOPOLOGY[0] input bits
    // - Hidden layers: TOPOLOGY[l] activation bits
    // - Output layer: TOPOLOGY[OUTPUT_LAYER] 32-bit popcounts
    function automatic int calc_activation_storage_bits();
        int max_bits;
        int candidate_bits;
        begin
            max_bits = TOPOLOGY[0];
            for (int idx = 1; idx < TOTAL_LAYERS; idx++) begin
                candidate_bits = (idx == OUTPUT_LAYER) ? (TOPOLOGY[idx] * 32) : TOPOLOGY[idx];
                if (candidate_bits > max_bits)
                    max_bits = candidate_bits;
            end
            return max_bits;
        end
    endfunction

    localparam int ACTIVATION_STORAGE_BITS = calc_activation_storage_bits();

    logic [INPUT_DATA_WIDTH-1:0] pixels[INPUT_BUS_ELEMENTS];
    logic [INPUT_BUS_ELEMENTS-1:0] binarized_pixels;
    logic rst_int;
    assign rst_int = rst;

    initial begin
        // Sanity checks on parameters used to compute array sizes/slices.
        if (INPUT_BUS_WIDTH % INPUT_DATA_WIDTH)
            $fatal(1, "bnn_fcc requires INPUT_BUS_WIDTH to be a multiple of INPUT_DATA_WIDTH");
        if (CONFIG_BUS_WIDTH % 8)
            $fatal(1, "bnn_fcc requires CONFIG_BUS_WIDTH to be a multiple of 8");
        if (OUTPUT_BUS_WIDTH % 8)
            $fatal(1, "bnn_fcc requires OUTPUT_BUS_WIDTH to be a multiple of 8");
        for (int i = 0; i < LAYERS; i++) begin
            if (PARALLEL_NEURONS[i] <= 0)
                $fatal(1, "bnn_fcc requires PARALLEL_NEURONS[%0d] > 0", i);
        end
    end

    // -------------------------------------------------------------------------
    // Configuration stream parsing (AXI4-Stream) -> per-layer write ports
    // -------------------------------------------------------------------------
    // The parser converts the byte-streamed header+payload into:
    // - layer_wr_en_weights / layer_wr_en_thresholds : one-hot per layer
    // - layer_wr_addr                               : word address within that layer RAM
    // - layer_wr_data / layer_wr_strb               : bus-width write data + byte strobes
    logic [      TOTAL_LAYERS-1:0] layer_wr_en_weights;
    logic [      TOTAL_LAYERS-1:0] layer_wr_en_thresholds;
    logic [                  31:0] layer_wr_addr;
    logic [  CONFIG_BUS_WIDTH-1:0] layer_wr_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] layer_wr_strb;

    config_parser #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .TOTAL_LAYERS    (TOTAL_LAYERS)
    ) config_parser_inst (
        .clk                   (clk),
        .rst                   (rst_int),
        .config_valid          (config_valid),
        .config_ready          (config_ready),
        .config_data           (config_data),
        .config_keep           (config_keep),
        .config_last           (config_last),
        .layer_wr_en_weights   (layer_wr_en_weights),
        .layer_wr_en_thresholds(layer_wr_en_thresholds),
        .layer_wr_addr         (layer_wr_addr),
        .layer_wr_data         (layer_wr_data),
        .layer_wr_strb         (layer_wr_strb)
    );

    // -------------------------------------------------------------------------
    // Image input binarization
    // -------------------------------------------------------------------------
    // Each byte on the AXI stream is one pixel. Honor TKEEP so unused bytes in
    // the final beat don't leak into the image buffer.
    always_comb begin
        for (int i = 0; i < INPUT_BUS_ELEMENTS; i++) begin
            pixels[i] = data_in_data[i*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH];
            binarized_pixels[i] = data_in_keep[i] && (pixels[i] >= INPUT_BINARIZATION_THRESHOLD);
        end
    end

    // -------------------------------------------------------------------------
    // Activation valid/ready tokens across the pipeline
    // -------------------------------------------------------------------------
    // `layer_valid_reg[l]` asserts when `layer_activations[l]` holds a complete
    // activation vector for that layer.
    // `layer_ready[l]` is asserted by the downstream consumer to pop the token.
    logic layer_valid_reg[0:LAYERS];
    logic layer_ready[0:LAYERS];

    // -------------------------------------------------------------------------
    // Input image buffer (stores TOPOLOGY[0] binarized pixels)
    // -------------------------------------------------------------------------
    logic [INPUT_BUS_ELEMENTS-1:0] input_buffer_words[0:INPUT_BUFFER_WORDS-1];
    logic [INPUT_WORD_IDX_WIDTH-1:0] input_word_idx;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            input_word_idx <= '0;
            layer_valid_reg[0] <= 1'b0;
        end else begin
            if (layer_ready[0] && layer_valid_reg[0]) begin
                layer_valid_reg[0] <= 1'b0;
            end
            if (data_in_valid && data_in_ready) begin
                input_buffer_words[input_word_idx] <= binarized_pixels;

                if (data_in_last || (input_word_idx == INPUT_WORD_IDX_WIDTH'(INPUT_BUFFER_WORDS - 1))) begin
                    layer_valid_reg[0] <= 1'b1;
                    input_word_idx <= '0;
                end else begin
                    input_word_idx <= INPUT_WORD_IDX_WIDTH'(input_word_idx + 1'b1);
                end
            end
        end
    end

    // Backpressure the input stream while an image is buffered + being processed.
    assign data_in_ready = !layer_valid_reg[0];

    // -------------------------------------------------------------------------
    // Per-layer activation storage and compute handshakes
    // -------------------------------------------------------------------------
    logic [ACTIVATION_STORAGE_BITS-1:0] layer_activations[0:LAYERS];
    logic comp_valid_out[1:LAYERS];
    logic comp_ready_in[1:LAYERS];
    logic comp_ready_out[1:LAYERS];

    // Zero pad the MSBs of layer_activations[0] if the shared storage width is
    // wider than the raw input bit-vector.
    if (ACTIVATION_STORAGE_BITS > TOPOLOGY[0]) begin : gen_input_pad_zero
        assign layer_activations[0][ACTIVATION_STORAGE_BITS-1:TOPOLOGY[0]] = '0;
    end

    // Flatten the bus-word input buffer into a single contiguous bit-vector.
    for (genvar input_word = 0; input_word < INPUT_BUFFER_WORDS; input_word++) begin : gen_input_buffer_flatten
        localparam int WORD_LO = input_word * INPUT_BUS_ELEMENTS;
        localparam int WORD_BITS = ((WORD_LO + INPUT_BUS_ELEMENTS) <= TOPOLOGY[0]) ?
            INPUT_BUS_ELEMENTS : (TOPOLOGY[0] - WORD_LO);
        assign layer_activations[0][WORD_LO +: WORD_BITS] = input_buffer_words[input_word][0 +: WORD_BITS];
    end

    // -------------------------------------------------------------------------
    // Output argmax
    // -------------------------------------------------------------------------
    // The output layer produces 32-bit popcounts for each class. This block
    // scans the score vector, tracks the best index/score, and emits the index.
    logic                          argmax_active;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_scan_idx;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_best_idx;
    logic [OUTPUT_SCORE_WIDTH-1:0] argmax_best_score;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_stage_idx;
    logic [OUTPUT_SCORE_WIDTH-1:0] argmax_stage_score;
    logic                          argmax_stage_valid;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_next_best_idx;
    logic [OUTPUT_SCORE_WIDTH-1:0] argmax_next_best_score;
    logic                          out_valid_reg;
    logic [OUTPUT_BUS_WIDTH-1:0]   out_data_reg;

    genvar l;
    generate
        for (l = 1; l <= LAYERS; l++) begin : gen_layers
            localparam int L_PARALLEL_NEURONS = PARALLEL_NEURONS[l-1];
            localparam bit L_IS_OUTPUT_LAYER = (l == LAYERS);
            localparam int L_RESULT_WIDTH = L_IS_OUTPUT_LAYER ? (TOPOLOGY[l] * 32) : TOPOLOGY[l];

            // Per-layer memory scheduler / storage.
            logic mem_layer_start;
            logic mem_read_weight;
            logic mem_read_thresh;
            logic [L_PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] mem_weight_data_raw;
            logic [L_PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] mem_weight_data_q;
            logic [L_PARALLEL_NEURONS-1:0][31:0]                 mem_thresh_data;

            logic [L_RESULT_WIDTH-1:0]   result_vector;

            layer_memory #(
                .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
                .LAYER_INPUTS    (TOPOLOGY[l-1]),
                .NUM_NEURONS     (TOPOLOGY[l]),
                .PARALLEL_NEURONS(L_PARALLEL_NEURONS),
                .PARALLEL_INPUTS (PARALLEL_INPUTS)
            ) mem_inst (
                .clk              (clk),
                .rst              (rst_int),
                .wr_en_weights    (layer_wr_en_weights[l]),
                .wr_en_thresholds (layer_wr_en_thresholds[l]),
                .wr_addr          (layer_wr_addr),
                .wr_data          (layer_wr_data),
                .wr_strb          (layer_wr_strb),
                .layer_start      (mem_layer_start),
                .read_weight_chunk(mem_read_weight),
                .read_threshold   (mem_read_thresh),
                .rd_data_weights  (mem_weight_data_raw),
                .rd_data_threshold(mem_thresh_data)
            );

            // Register the weight read data to cut long URAM→align→compute paths.
            always_ff @(posedge clk) begin
                if (rst_int) begin
                    mem_weight_data_q <= '0;
                end else begin
                    mem_weight_data_q <= mem_weight_data_raw;
                end
            end

            compute_layer #(
                .LAYER_ID        (l),
                .LAYER_INPUTS    (TOPOLOGY[l-1]),
                .NUM_NEURONS     (TOPOLOGY[l]),
                .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
                .PARALLEL_INPUTS (PARALLEL_INPUTS),
                .PARALLEL_NEURONS(L_PARALLEL_NEURONS),
                .IS_OUTPUT_LAYER (L_IS_OUTPUT_LAYER)
            ) comp_inst (
                .clk              (clk),
                .rst              (rst_int),
                .valid_in         (layer_valid_reg[l-1]),
                .ready_out        (comp_ready_out[l]),
                .valid_out        (comp_valid_out[l]),
                .ready_in         (comp_ready_in[l]),
                .result_vector    (result_vector),
                .input_activations(layer_activations[l-1][TOPOLOGY[l-1]-1:0]),
                .mem_layer_start  (mem_layer_start),
                .mem_read_weight  (mem_read_weight),
                .mem_read_thresh  (mem_read_thresh),
                .mem_weight_data  (mem_weight_data_q),
                .mem_thresh_data  (mem_thresh_data)
            );

            if (L_RESULT_WIDTH == ACTIVATION_STORAGE_BITS) begin : gen_store_exact
                // Store result_vector directly.
                always_ff @(posedge clk) begin
                    if (rst_int) begin
                        layer_valid_reg[l] <= 1'b0;
                    end else begin
                        if (layer_ready[l] && layer_valid_reg[l]) begin
                            layer_valid_reg[l] <= 1'b0;
                        end
                        if (comp_valid_out[l] && comp_ready_in[l]) begin
                            layer_activations[l] <= result_vector;
                            layer_valid_reg[l] <= 1'b1;
                        end
                    end
                end
            end else begin : gen_store_padded
                // Store result_vector into the LSBs and clear the padding bits.
                always_ff @(posedge clk) begin
                    if (rst_int) begin
                        layer_valid_reg[l] <= 1'b0;
                    end else begin
                        if (layer_ready[l] && layer_valid_reg[l]) begin
                            layer_valid_reg[l] <= 1'b0;
                        end
                        if (comp_valid_out[l] && comp_ready_in[l]) begin
                            layer_activations[l] <= '0;
                            layer_activations[l][L_RESULT_WIDTH-1:0] <= result_vector;
                            layer_valid_reg[l] <= 1'b1;
                        end
                    end
                end
            end

            // `compute_layer` holds `valid_in` high for the full compute duration,
            // so `ready_out` is used here as a "done/consume" pulse.
            assign layer_ready[l-1] = comp_ready_out[l];
            if (l == 1) begin
                // First hidden layer:
                // - Only allow the layer to complete when the input token is present.
                // - Block completion if this layer's storage is still full.
                assign comp_ready_in[l] = layer_valid_reg[l-1] && (!layer_valid_reg[l] || layer_ready[l]);
            end else if (l == LAYERS) begin
                logic argmax_safe;
                // Output layer:
                // - Do not overwrite the score vector while argmax is scanning it.
                assign argmax_safe = !argmax_active || (argmax_scan_idx == OUTPUT_NEURONS - 1);
                assign comp_ready_in[l] = !layer_valid_reg[l] && argmax_safe;
            end else begin
                // Intermediate hidden layers: simple 1-deep buffering.
                assign comp_ready_in[l] = !layer_valid_reg[l] || layer_ready[l];
            end
        end
    endgenerate

    // Compare pipeline stage against the current best score.
    always_comb begin
        argmax_next_best_idx = argmax_best_idx;
        argmax_next_best_score = argmax_best_score;
        if (argmax_stage_valid && (argmax_stage_score > argmax_best_score)) begin
            argmax_next_best_idx = argmax_stage_idx;
            argmax_next_best_score = argmax_stage_score;
        end
    end

    // Argmax can start when the output layer token is present, the argmax engine is idle,
    // and the output skid register is empty (or will be consumed this cycle).
    assign layer_ready[LAYERS] = (!argmax_active && (!out_valid_reg || data_out_ready));

    always_ff @(posedge clk) begin
        if (rst_int) begin
            argmax_active <= 1'b0;
            argmax_scan_idx <= '0;
            argmax_best_idx <= '0;
            argmax_best_score <= '0;
            argmax_stage_idx <= '0;
            argmax_stage_score <= '0;
            argmax_stage_valid <= 1'b0;
            out_valid_reg <= 1'b0;
            out_data_reg  <= '0;
        end else begin
            if (data_out_ready && out_valid_reg) begin
                out_valid_reg <= 1'b0;
            end

            if (layer_valid_reg[LAYERS] && layer_ready[LAYERS]) begin
                // Begin scan: initialize best with neuron 0 and start reading neuron 1.
                argmax_active <= 1'b1;
                argmax_scan_idx <= OUTPUT_INDEX_WIDTH'(1);
                argmax_best_idx <= '0;
                argmax_best_score <= layer_activations[LAYERS][0 +: OUTPUT_SCORE_WIDTH];
                argmax_stage_idx <= '0;
                argmax_stage_score <= '0;
                argmax_stage_valid <= 1'b0;
            end else if (argmax_active) begin
                if (argmax_stage_valid) begin
                    argmax_best_idx <= argmax_next_best_idx;
                    argmax_best_score <= argmax_next_best_score;
                end

                if (argmax_scan_idx < OUTPUT_NEURONS) begin
                    // Pipeline the next candidate into argmax_stage_*.
                    argmax_stage_idx <= argmax_scan_idx;
                    argmax_stage_score <= layer_activations[LAYERS][argmax_scan_idx*32 +: OUTPUT_SCORE_WIDTH];
                    argmax_stage_valid <= 1'b1;
                    argmax_scan_idx <= argmax_scan_idx + OUTPUT_INDEX_WIDTH'(1);
                end else begin
                    // Finished. If argmax_stage_valid is still high, it contains
                    // the last candidate that hasn't yet been applied to best_idx.
                    argmax_stage_valid <= 1'b0;
                    if (argmax_stage_valid) begin
                        argmax_active <= 1'b0;
                        out_valid_reg <= 1'b1;
                        out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_next_best_idx);
                    end else begin
                        argmax_active <= 1'b0;
                        out_valid_reg <= 1'b1;
                        out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_best_idx);
                    end
                end
            end
        end
    end

    // AXI4-Stream result: one beat per image.
    assign data_out_valid = out_valid_reg;
    assign data_out_data  = out_data_reg;
    assign data_out_keep  = out_valid_reg ? '1 : '0;
    assign data_out_last  = out_valid_reg;

endmodule
