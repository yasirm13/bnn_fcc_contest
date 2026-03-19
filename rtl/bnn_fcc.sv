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
        end
    end

    logic [TOPOLOGY[0]-1:0] input_buffer;
    logic [31:0] input_bit_count;
    logic image_ready;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            input_bit_count <= '0;
            image_ready <= 1'b0;
        end else begin
            image_ready <= 1'b0;
            if (data_in_valid && data_in_ready) begin
                for (int i = 0; i < INPUT_BUS_ELEMENTS; i++) begin
                    if (data_in_keep[i] && (input_bit_count + i < TOPOLOGY[0])) begin
                        input_buffer[input_bit_count+i] <= (pixels[i] >= INPUT_BINARIZATION_THRESHOLD);
                    end
                end

                begin
                    int valid_bytes;
                    valid_bytes = 0;
                    for (int i = 0; i < INPUT_BUS_WIDTH / 8; i++) valid_bytes += data_in_keep[i];

                    input_bit_count <= input_bit_count + valid_bytes;

                    if (data_in_last || (input_bit_count + valid_bytes >= TOPOLOGY[0])) begin
                        image_ready <= 1'b1;
                        input_bit_count <= '0;
                    end
                end
            end
        end
    end

    logic layer_idle[1:LAYERS];
    assign data_in_ready = layer_idle[1] && !image_ready;

    logic [ACTIVATION_STORAGE_BITS-1:0] layer_activations[0:LAYERS];
    logic layer_start_pulse[1:LAYERS+1];

    assign layer_activations[0][TOPOLOGY[0]-1:0] = input_buffer;
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

    logic                         argmax_active;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_scan_idx;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_best_idx;
    logic signed [31:0]            argmax_best_score;
    logic signed [31:0]            argmax_scan_score;
    logic [OUTPUT_INDEX_WIDTH-1:0] argmax_next_best_idx;
    logic signed [31:0]            argmax_next_best_score;
    logic                          out_valid_reg;
    logic [OUTPUT_BUS_WIDTH-1:0]   out_data_reg;

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

    always_ff @(posedge clk) begin
        if (rst_int) begin
            argmax_active <= 1'b0;
            argmax_scan_idx <= '0;
            argmax_best_idx <= '0;
            argmax_best_score <= '0;
            out_valid_reg <= 1'b0;
            out_data_reg  <= '0;
        end else begin
            if (layer_start_pulse[LAYERS+1]) begin
                argmax_active <= 1'b1;
                argmax_scan_idx <= OUTPUT_INDEX_WIDTH'(1);
                argmax_best_idx <= '0;
                argmax_best_score <= $signed(layer_activations[LAYERS][31:0]);
                out_valid_reg <= 1'b0;
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
