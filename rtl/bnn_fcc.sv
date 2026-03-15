// Layer 0 (F[0] inputs per neuron):
//   Neuron 0: W0,0 ... W0,F[0]-1 | Threshold0
//   Neuron 1: W1,0 ... W1,F[0]-1 | Threshold1
//   ...
//   Neuron N[0]-1: ...           | ThresholdN[0]-1

// Layer 1 (F[1] inputs per neuron):
//   Neuron 0: W0,0 ... W0,F[1]-1 | Threshold0
//   Neuron 1: W1,0 ... W1,F[1]-1 | Threshold1
//   ...

module bnn_fcc #(
    parameter int INPUT_DATA_WIDTH  = 8,
    parameter int INPUT_BUS_WIDTH   = 64,
    parameter int CONFIG_BUS_WIDTH  = 32,
    parameter int OUTPUT_DATA_WIDTH = 4,
    parameter int OUTPUT_BUS_WIDTH  = 8,

    parameter int TOTAL_LAYERS = 4,  // Includes input, hidden, and output
    parameter int TOPOLOGY[TOTAL_LAYERS] = '{
        0: 784,
        1: 256,
        2: 256,
        3: 10,
        default: 0
    },  // 0: input, TOTAL_LAYERS-1: output

    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[TOTAL_LAYERS-1] = '{default: 8},

    localparam int THRESHOLD_DATA_WIDTH = 32
) (
    input logic clk,
    input logic rst,

    // AXI streaming configuration interface (consumer)
    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,

    // AXI streaming image input interface (consumer)
    input  logic                         data_in_valid,
    output logic                         data_in_ready,
    input  logic [  INPUT_BUS_WIDTH-1:0] data_in_data,
    input  logic [INPUT_BUS_WIDTH/8-1:0] data_in_keep,
    input  logic                         data_in_last,

    // AXI streaming classification output interface (producer)
    output logic                          data_out_valid,
    input  logic                          data_out_ready,
    output logic [  OUTPUT_BUS_WIDTH-1:0] data_out_data,
    output logic [OUTPUT_BUS_WIDTH/8-1:0] data_out_keep,
    output logic                          data_out_last
);
    localparam int LAYERS = TOTAL_LAYERS - 1;

    function automatic int get_max_parallel_inputs();
        int max_v = PARALLEL_INPUTS;
        for (int i = 0; i < LAYERS - 1; i++) begin
            if (PARALLEL_NEURONS[i] > max_v) max_v = PARALLEL_NEURONS[i];
        end
        return max_v;
    endfunction

    localparam int NUM_NEURONS[LAYERS] = TOPOLOGY[1:LAYERS];
    localparam int INPUT_BUS_ELEMENTS = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH;
    localparam int INPUT_BINARIZATION_THRESHOLD = 1 << (INPUT_DATA_WIDTH - 1);
    localparam int MAX_PARALLEL_INPUTS = get_max_parallel_inputs();

    logic [    INPUT_DATA_WIDTH-1:0] pixels            [        INPUT_BUS_ELEMENTS];

    // (legacy signals removed) - config_parser writes go directly into each
    // layer_memory instance, and compute_layer consumes those memories.

    // Agilex 5 requirement: instantiate exactly one Reset Release IP.
    // Hold internal logic in reset until configuration completes.
    logic ninit_done;
    logic rst_int;
    assign rst_int = rst | ninit_done;

    altera_s10_user_rst_clkgate reset_release_inst (
        .ninit_done(ninit_done)
    );

    initial begin
        if (INPUT_BUS_ELEMENTS != PARALLEL_INPUTS)
            $fatal(1, "bnn_fcc requires PARALLEL_INPUTS to match the pixels/beat");
        for (int i = 0; i < LAYERS - 1; i++) begin
            if (PARALLEL_NEURONS[i] != PARALLEL_INPUTS)
                $fatal(1, "bnn_fcc requires PARALLEL_NEURONS to match PARALLEL_INPUTS in all hidden layers");
        end
        if (TOPOLOGY[0] % PARALLEL_INPUTS)
            $fatal(1, "bnn_fcc requires total inputs to be a multiple of PARALLEL_INPUTS.");
        if (PARALLEL_INPUTS != 8) $fatal(1, "bnn_fcc currently requires PARALLEL_INPUTS=8");
    end

    // ==========================================
    // Config Parser Instantiation
    // ==========================================
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

    // ==========================================
    // Image Input Binarization & Buffering
    // ==========================================
    always_comb begin
        for (int i = 0; i < INPUT_BUS_ELEMENTS; i++) begin
            pixels[i] = data_in_data[i*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH];
        end
    end

    // We assume the compute pipeline can take the data once it's fully buffered.
    // For simplicity, we just assert data_in_ready when we are collecting an image.
    // We need to collect TOPOLOGY[0] bits (784 bits for MNIST) into a buffer.

    logic [TOPOLOGY[0]-1:0] input_buffer;
    logic [31:0] input_bit_count;
    logic image_ready;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            input_bit_count <= '0;
            image_ready <= 1'b0;
        end else begin
            image_ready <= 1'b0;  // Pulse image_ready when a full image is collected
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

                        // DEBUG
                        $display("HW_BIN_PIXELS: %b", input_buffer);
                    end
                end
            end
        end
    end

    // Ready to receive if we are not busy computing and haven't just finished an image.
    // In a real pipeline we could overlap, but here we wait for layer 1 to be IDLE.
    logic layer_idle[1:LAYERS];
    assign data_in_ready = layer_idle[1] && !image_ready;

    // ==========================================
    // Layer Pipeline Generation
    // ==========================================

    // Arrays for interconnection between layers
    logic [32767:0] layer_activations[0:LAYERS];  // Large enough to hold max neurons (e.g. 256)
    logic layer_start_pulse[1:LAYERS+1];

    // Layer 0 is the input buffer
    assign layer_activations[0][TOPOLOGY[0]-1:0] = input_buffer;
    assign layer_start_pulse[1] = image_ready;

    genvar l;
    generate
        for (l = 1; l <= LAYERS; l++) begin : gen_layers

            // Wires for compute <-> memory interface
            logic                        mem_layer_start;
            logic                        mem_read_weight;
            logic                        mem_read_thresh;
            logic [CONFIG_BUS_WIDTH-1:0] mem_weight_data;
            logic [                31:0] mem_thresh_data;

            logic                        layer_done;
            logic [  TOPOLOGY[l]*32-1:0] result_vector;

            layer_memory #(
                .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
                .LAYER_INPUTS    (TOPOLOGY[l-1]),
                .NUM_NEURONS     (TOPOLOGY[l]),
                .PARALLEL_NEURONS(1),
                .PARALLEL_INPUTS (PARALLEL_INPUTS)
	            ) mem_inst (
	                .clk              (clk),
	                .rst              (rst_int),
	                .wr_en_weights    (layer_wr_en_weights[l]),
	                .wr_en_thresholds (layer_wr_en_thresholds[l]),
	                .wr_addr          (layer_wr_addr),
	                .wr_data          (layer_wr_data),
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
                .IS_OUTPUT_LAYER ((l == LAYERS) ? 1'b1 : 1'b0)
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

            // Removed manual assign layer_idle = !mem_layer_start since the module now exports it.

            // Register activations for next layer
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

        end
    endgenerate

    // ==========================================
    // Output Argmax
    // ==========================================
    logic [31:0] max_count;
    logic out_valid_reg;
    logic [OUTPUT_BUS_WIDTH-1:0] out_data_reg;

    always_ff @(posedge clk) begin
        if (rst_int) begin
            out_valid_reg <= 1'b0;
            out_data_reg  <= '0;
        end else begin
            // Pulse data_out_valid when output layer is done
            if (layer_start_pulse[LAYERS+1]) begin
                out_valid_reg <= 1'b1;

                max_count = layer_activations[LAYERS][31:0];
                out_data_reg = 0;
                for (int i = 1; i < TOPOLOGY[LAYERS]; i++) begin
                    if ($signed(layer_activations[LAYERS][i*32+:32]) > $signed(max_count)) begin
                        out_data_reg = i;
                        max_count = layer_activations[LAYERS][i*32+:32];
                    end
                end
            end else if (data_out_ready && out_valid_reg) begin
                out_valid_reg <= 1'b0;  // Deassert after downstream accepts
            end
        end
    end

    assign data_out_valid = out_valid_reg;
    assign data_out_data  = out_data_reg;
    // AXI-Stream note: TKEEP/TLAST are only meaningful when TVALID is asserted.
    // Driving them from `out_valid_reg` avoids "stuck at VCC" warnings and is
    // friendlier to downstream logic that samples sideband signals with TVALID.
    assign data_out_keep  = out_valid_reg ? '1 : '0;  // 1 beat = 1 byte
    assign data_out_last  = out_valid_reg;

endmodule
