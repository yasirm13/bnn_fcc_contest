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

    logic [INPUT_BUS_ELEMENTS-1:0] input_buffer_words[0:INPUT_BUFFER_WORDS-1];
    logic [INPUT_WORD_IDX_WIDTH-1:0] input_word_idx;
    logic image_ready;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            input_word_idx <= '0;
            image_ready <= 1'b0;
        end else begin
            image_ready <= 1'b0;
            if (data_in_valid && data_in_ready) begin
                input_buffer_words[input_word_idx] <= binarized_pixels;

                if (data_in_last || (input_word_idx == INPUT_WORD_IDX_WIDTH'(INPUT_BUFFER_WORDS - 1))) begin
                    image_ready <= 1'b1;
                    input_word_idx <= '0;
                end else begin
                    input_word_idx <= INPUT_WORD_IDX_WIDTH'(input_word_idx + 1'b1);
                end
            end
        end
    end

    logic layer_idle[1:LAYERS];
    assign data_in_ready = layer_idle[1] && !image_ready;

    logic [ACTIVATION_STORAGE_BITS-1:0] layer_activations[0:LAYERS];
    logic layer_start_pulse[1:LAYERS+1];

    if (ACTIVATION_STORAGE_BITS > TOPOLOGY[0]) begin : gen_input_pad_zero
        assign layer_activations[0][ACTIVATION_STORAGE_BITS-1:TOPOLOGY[0]] = '0;
    end

    for (genvar input_word = 0; input_word < INPUT_BUFFER_WORDS; input_word++) begin : gen_input_buffer_flatten
        localparam int WORD_LO = input_word * INPUT_BUS_ELEMENTS;
        localparam int WORD_BITS = ((WORD_LO + INPUT_BUS_ELEMENTS) <= TOPOLOGY[0]) ?
            INPUT_BUS_ELEMENTS : (TOPOLOGY[0] - WORD_LO);
        assign layer_activations[0][WORD_LO +: WORD_BITS] = input_buffer_words[input_word][0 +: WORD_BITS];
    end
    assign layer_start_pulse[1] = image_ready;

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
                .start            (layer_start_pulse[l]),
                .done             (layer_done),
                .is_idle          (layer_idle[l]),
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
                        layer_start_pulse[l+1] <= 1'b0;
                    end else begin
                        layer_start_pulse[l+1] <= layer_done;
                        if (layer_done) begin
                            layer_activations[l] <= result_vector;
                        end
                    end
                end
            end else begin : gen_store_padded
                always_ff @(posedge clk) begin
                    if (rst_int) begin
                        layer_start_pulse[l+1] <= 1'b0;
                    end else begin
                        layer_start_pulse[l+1] <= layer_done;
                        if (layer_done) begin
                            layer_activations[l] <= '0;
                            layer_activations[l][L_RESULT_WIDTH-1:0] <= result_vector;
                        end
                    end
                end
            end
        end
    endgenerate

    // Pipelined Argmax Tree for OUTPUT_NEURONS (Assumed up to 16 for simplicity, currently 10)
    logic [OUTPUT_INDEX_WIDTH-1:0]  argmax_l1_idx   [0:7];
    logic signed [31:0]             argmax_l1_score [0:7];
    logic                           argmax_l1_valid;

    logic [OUTPUT_INDEX_WIDTH-1:0]  argmax_l2_idx   [0:3];
    logic signed [31:0]             argmax_l2_score [0:3];
    logic                           argmax_l2_valid;

    logic [OUTPUT_INDEX_WIDTH-1:0]  argmax_l3_idx   [0:1];
    logic signed [31:0]             argmax_l3_score [0:1];
    logic                           argmax_l3_valid;
    
    logic                          out_valid_reg;
    logic [OUTPUT_BUS_WIDTH-1:0]   out_data_reg;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            argmax_l1_valid <= 1'b0;
            argmax_l2_valid <= 1'b0;
            argmax_l3_valid <= 1'b0;
            out_valid_reg   <= 1'b0;
        end else begin
            // Level 1: Pairs (0-1, 2-3, 4-5, 6-7, 8-9)
            if (layer_start_pulse[LAYERS+1]) begin
                for (int i = 0; i < 8; i++) begin
                    int idx0;
                    int idx1;
                    logic signed [31:0] score0;
                    logic signed [31:0] score1;
                    
                    idx0 = i * 2;
                    idx1 = i * 2 + 1;
                    score0 = (idx0 < OUTPUT_NEURONS) ? $signed(layer_activations[LAYERS][idx0*32 +: 32]) : $signed({32{1'b1}}<<31); // -inf
                    score1 = (idx1 < OUTPUT_NEURONS) ? $signed(layer_activations[LAYERS][idx1*32 +: 32]) : $signed({32{1'b1}}<<31); // -inf
                    
                    if (score1 > score0) begin
                        argmax_l1_score[i] <= score1;
                        argmax_l1_idx[i]   <= OUTPUT_INDEX_WIDTH'(idx1);
                    end else begin
                        argmax_l1_score[i] <= score0;
                        argmax_l1_idx[i]   <= OUTPUT_INDEX_WIDTH'(idx0);
                    end
                end
                argmax_l1_valid <= 1'b1;
            end else begin
                argmax_l1_valid <= 1'b0;
            end

            // Level 2: Pairs of L1 (0-1, 2-3, 4-5, 6-7)
            if (argmax_l1_valid) begin
                for (int i = 0; i < 4; i++) begin
                    logic signed [31:0] score0;
                    logic signed [31:0] score1;
                    
                    score0 = argmax_l1_score[i*2];
                    score1 = argmax_l1_score[i*2+1];
                    
                    if (score1 > score0) begin
                        argmax_l2_score[i] <= score1;
                        argmax_l2_idx[i]   <= argmax_l1_idx[i*2+1];
                    end else begin
                        argmax_l2_score[i] <= score0;
                        argmax_l2_idx[i]   <= argmax_l1_idx[i*2];
                    end
                end
                argmax_l2_valid <= 1'b1;
            end else begin
                argmax_l2_valid <= 1'b0;
            end

            // Level 3: Pairs of L2 (0-1, 2-3)
            if (argmax_l2_valid) begin
                for (int i = 0; i < 2; i++) begin
                    logic signed [31:0] score0;
                    logic signed [31:0] score1;
                    
                    score0 = argmax_l2_score[i*2];
                    score1 = argmax_l2_score[i*2+1];
                    
                    if (score1 > score0) begin
                        argmax_l3_score[i] <= score1;
                        argmax_l3_idx[i]   <= argmax_l2_idx[i*2+1];
                    end else begin
                        argmax_l3_score[i] <= score0;
                        argmax_l3_idx[i]   <= argmax_l2_idx[i*2];
                    end
                end
                argmax_l3_valid <= 1'b1;
            end else begin
                argmax_l3_valid <= 1'b0;
            end

            // Level 4: Final winner
            if (argmax_l3_valid) begin
                if (argmax_l3_score[1] > argmax_l3_score[0]) begin
                    out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_l3_idx[1]);
                end else begin
                    out_data_reg <= OUTPUT_BUS_WIDTH'(argmax_l3_idx[0]);
                end
                out_valid_reg <= 1'b1;
            end else if (data_out_ready && out_valid_reg) begin
                out_valid_reg <= 1'b0;
            end
        end
    end

    assign data_out_valid = out_valid_reg;
    assign data_out_data  = out_data_reg;
    assign data_out_keep  = out_valid_reg ? '1 : '0;
    assign data_out_last  = out_valid_reg;

endmodule
