// Top-level streaming BNN classifier.
// Parses AXI4-Stream configuration traffic, buffers a binarized image, runs each
// layer in sequence, and returns the final argmax classification as one output beat.
// Submission parallelism is controlled by PARALLEL_INPUTS and PARALLEL_NEURONS.
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
    localparam int LAYERS = TOTAL_LAYERS - 1;
    localparam int NUM_NEURONS[LAYERS] = TOPOLOGY[1:LAYERS];
    localparam int INPUT_BUS_ELEMENTS = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH;
    localparam int INPUT_BINARIZATION_THRESHOLD = 1 << (INPUT_DATA_WIDTH - 1);
    localparam int OUTPUT_LAYER = TOTAL_LAYERS - 1;
    localparam int OUTPUT_NEURONS = TOPOLOGY[OUTPUT_LAYER];
    localparam int OUTPUT_INDEX_WIDTH = (OUTPUT_NEURONS > 1) ? $clog2(OUTPUT_NEURONS) : 1;
    localparam int INPUT_BUFFER_WORDS = (TOPOLOGY[0] + INPUT_BUS_ELEMENTS - 1) / INPUT_BUS_ELEMENTS;
    localparam int INPUT_WORD_IDX_WIDTH = (INPUT_BUFFER_WORDS > 1) ? $clog2(INPUT_BUFFER_WORDS) : 1;

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
        if (INPUT_BUS_WIDTH % INPUT_DATA_WIDTH)
            $fatal(1, "bnn_fcc requires INPUT_BUS_WIDTH to be a multiple of INPUT_DATA_WIDTH");
        for (int i = 0; i < LAYERS; i++) begin
            if (PARALLEL_NEURONS[i] <= 0)
                $fatal(1, "bnn_fcc requires PARALLEL_NEURONS[%0d] > 0", i);
        end
    end

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

    always_comb begin
        for (int i = 0; i < INPUT_BUS_ELEMENTS; i++) begin
            pixels[i] = data_in_data[i*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH];
            binarized_pixels[i] = data_in_keep[i] && (pixels[i] >= INPUT_BINARIZATION_THRESHOLD);
        end
    end

    logic layer_valid_reg[0:LAYERS];
    logic layer_ready[0:LAYERS];

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

    assign data_in_ready = !layer_valid_reg[0];

    logic [ACTIVATION_STORAGE_BITS-1:0] layer_activations[0:LAYERS];
    logic comp_valid_out[1:LAYERS];
    logic comp_ready_in[1:LAYERS];
    logic comp_ready_out[1:LAYERS];

    if (ACTIVATION_STORAGE_BITS > TOPOLOGY[0]) begin : gen_input_pad_zero
        assign layer_activations[0][ACTIVATION_STORAGE_BITS-1:TOPOLOGY[0]] = '0;
    end

    for (genvar input_word = 0; input_word < INPUT_BUFFER_WORDS; input_word++) begin : gen_input_buffer_flatten
        localparam int WORD_LO = input_word * INPUT_BUS_ELEMENTS;
        localparam int WORD_BITS = ((WORD_LO + INPUT_BUS_ELEMENTS) <= TOPOLOGY[0]) ?
            INPUT_BUS_ELEMENTS : (TOPOLOGY[0] - WORD_LO);
        assign layer_activations[0][WORD_LO +: WORD_BITS] = input_buffer_words[input_word][0 +: WORD_BITS];
    end

    logic                          argmax_active;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_scan_idx;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_best_idx;
    logic signed [31:0]            argmax_best_score;
    logic signed [31:0]            argmax_scan_score;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_next_best_idx;
    logic signed [31:0]            argmax_next_best_score;
    logic                          out_valid_reg;
    logic [OUTPUT_BUS_WIDTH-1:0]   out_data_reg;

    genvar l;
    generate
        for (l = 1; l <= LAYERS; l++) begin : gen_layers
            localparam int L_PARALLEL_NEURONS = PARALLEL_NEURONS[l-1];
            localparam bit L_IS_OUTPUT_LAYER = (l == LAYERS);
            localparam int L_RESULT_WIDTH = L_IS_OUTPUT_LAYER ? (TOPOLOGY[l] * 32) : TOPOLOGY[l];

            logic                        mem_layer_start;
            logic                        mem_read_weight;
            logic                        mem_read_thresh;
            logic [L_PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] mem_weight_data;
            logic [L_PARALLEL_NEURONS-1:0][31:0]                 mem_thresh_data;

            logic                        layer_done;
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
                .rd_data_weights  (mem_weight_data),
                .rd_data_threshold(mem_thresh_data)
            );

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
                .mem_weight_data  (mem_weight_data),
                .mem_thresh_data  (mem_thresh_data)
            );

            if (L_RESULT_WIDTH == ACTIVATION_STORAGE_BITS) begin : gen_store_exact
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
            
            assign layer_ready[l-1] = comp_ready_out[l];
            if (l == 1) begin
                assign comp_ready_in[l] = layer_valid_reg[l-1] && (!layer_valid_reg[l] || layer_ready[l]);
            end else if (l == LAYERS) begin
                logic argmax_safe;
                assign argmax_safe = !argmax_active || (argmax_scan_idx == OUTPUT_NEURONS - 1);
                assign comp_ready_in[l] = !layer_valid_reg[l] && argmax_safe;
            end else begin
                assign comp_ready_in[l] = !layer_valid_reg[l] || layer_ready[l];
            end
        end
    endgenerate

    always_comb begin
        if (argmax_scan_idx < OUTPUT_NEURONS)
            argmax_scan_score = $signed(layer_activations[LAYERS][argmax_scan_idx*32 +: 32]);
        else
            argmax_scan_score = '0;

        argmax_next_best_idx = argmax_best_idx;
        argmax_next_best_score = argmax_best_score;
        if (argmax_scan_score > argmax_best_score) begin
            argmax_next_best_idx = argmax_scan_idx;
            argmax_next_best_score = argmax_scan_score;
        end
    end

    assign layer_ready[LAYERS] = (!argmax_active && (!out_valid_reg || data_out_ready));

    always_ff @(posedge clk) begin
        if (rst_int) begin
            argmax_active <= 1'b0;
            argmax_scan_idx <= '0;
            argmax_best_idx <= '0;
            argmax_best_score <= '0;
            out_valid_reg <= 1'b0;
            out_data_reg  <= '0;
        end else begin
            if (data_out_ready && out_valid_reg) begin
                out_valid_reg <= 1'b0;
            end

            if (layer_valid_reg[LAYERS] && layer_ready[LAYERS]) begin
                argmax_active <= 1'b1;
                argmax_scan_idx <= OUTPUT_INDEX_WIDTH'(1);
                argmax_best_idx <= '0;
                argmax_best_score <= $signed(layer_activations[LAYERS][31:0]);
            end else if (argmax_active) begin
                if (argmax_scan_idx < OUTPUT_NEURONS) begin
                    argmax_best_idx <= argmax_next_best_idx;
                    argmax_best_score <= argmax_next_best_score;

                    if (argmax_scan_idx == OUTPUT_NEURONS - 1) begin
                        argmax_active <= 1'b0;
                        out_valid_reg <= 1'b1;
                        out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_next_best_idx);
                    end else begin
                        argmax_scan_idx <= argmax_scan_idx + OUTPUT_INDEX_WIDTH'(1);
                    end
                end else begin
                    argmax_active <= 1'b0;
                    out_valid_reg <= 1'b1;
                    out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_best_idx);
                end
            end
        end
    end

    assign data_out_valid = out_valid_reg;
    assign data_out_data  = out_data_reg;
    assign data_out_keep  = out_valid_reg ? '1 : '0;
    assign data_out_last  = out_valid_reg;

endmodule
