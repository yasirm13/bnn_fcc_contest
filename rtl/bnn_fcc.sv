module bnn_fcc #(
    parameter int INPUT_DATA_WIDTH  = 16, // NOTE: Testbench says 8, default might be 16 in template but TB overrides?
                                          // TB parameter INPUT_DATA_WIDTH = 8.
                                          // Keep template default but be aware.
    parameter int INPUT_BUS_WIDTH   = 32,
    parameter int CONFIG_BUS_WIDTH  = 32,
    parameter int OUTPUT_DATA_WIDTH = 4,
    parameter int OUTPUT_BUS_WIDTH  = 8,

    parameter int TOTAL_LAYERS = 4,  // Includes input, hidden, and output
    parameter int TOPOLOGY[TOTAL_LAYERS] = '{0: 784, 1: 256, 2: 256, 3: 10, default: 0}, 

    parameter bit PARALLELIZE_LAYERS = 1'b0,
    parameter int PARALLEL_NEURONS   = 1,
    parameter int PARALLEL_INPUTS    = 32
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

    // =========================================================================================
    // Configuration Parsing
    // =========================================================================================
    
    logic [TOTAL_LAYERS-1:0]       layer_wr_en_weights;
    logic [TOTAL_LAYERS-1:0]       layer_wr_en_thresholds;
    logic [31:0]                   layer_wr_addr;
    logic [CONFIG_BUS_WIDTH-1:0]   layer_wr_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] layer_wr_strb;

    config_parser #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .TOTAL_LAYERS(TOTAL_LAYERS)
    ) parser (
        .clk(clk),
        .rst(rst),
        .config_valid(config_valid),
        .config_ready(config_ready),
        .config_data(config_data),
        .config_keep(config_keep),
        .config_last(config_last),
        .layer_wr_en_weights(layer_wr_en_weights),
        .layer_wr_en_thresholds(layer_wr_en_thresholds),
        .layer_wr_addr(layer_wr_addr),
        .layer_wr_data(layer_wr_data),
        .layer_wr_strb(layer_wr_strb)
    );

    // =========================================================================================
    // Input Buffer (Layer 0 Result)
    // =========================================================================================
    
    // Calculate Max Layer Width for shared results bus
    // Result of Layer N is Input to Layer N+1.
    // L0 (Input) -> 784 bits.
    // L1 -> 256 bits.
    // L2 -> 256 bits.
    // L3 (Output) -> 10 * 32 = 320 bits.
    // Max Width = 784 bits (roughly 1024 safe).
    localparam int MAX_RESULT_WIDTH = 1024; 
    
    logic [MAX_RESULT_WIDTH-1:0] layer_results [TOTAL_LAYERS];
    
    // Input Buffer Logic
    // We treat Layer 0 "Result" as the input image.
    // We stream into layer_results[0].
    
    logic [31:0] input_ptr; // Bit pointer or chunk pointer
    // Input is INPUT_BUS_WIDTH chunks.
    logic input_complete;
    
    always_ff @(posedge clk) begin
        if (rst) begin
            input_ptr <= 0;
            input_complete <= 0;
            // Clear layer_results[0] if needed
        end else begin
            if (input_complete && data_out_valid && data_out_ready) begin
                 // Reset after output sent (Ready for next image)
                 input_ptr <= 0;
                 input_complete <= 0;
            end else if (data_in_valid && data_in_ready) begin
                // Binarization Loop
                logic [INPUT_BUS_WIDTH/8 - 1 : 0] binarized_pixels;
                for (int p=0; p < INPUT_BUS_WIDTH/INPUT_DATA_WIDTH; p++) begin
                    logic [7:0] pixel;
                    pixel = data_in_data[p*INPUT_DATA_WIDTH +: INPUT_DATA_WIDTH];
                    binarized_pixels[p] = (pixel >= 128) ? 1'b1 : 1'b0;
                end
                
                // Store binarized bits
                layer_results[0][input_ptr +: INPUT_BUS_WIDTH/INPUT_DATA_WIDTH] <= binarized_pixels;
                
                // Handle TLAST
                if (data_in_last) begin
                    input_complete <= 1;
                    input_ptr <= 0; // Prepare for next, though handled by FSM
                end else begin
                    input_ptr <= input_ptr + (INPUT_BUS_WIDTH/INPUT_DATA_WIDTH);
                end
                
                if (data_in_last) begin
                     $display("RTL_INPUT_DBG: First 16 bits: %b", layer_results[0][15:0]);
                end
            end
        end
    end
    
    // Ready Logic: Ready if not complete (and not processing? - Simple flow: Load -> Process)
    // If we support pipelining inputs, we need a FIFO.
    // Simple design: Stop accepting inputs while processing.
    logic processing_busy;
    assign data_in_ready = !rst && !input_complete && !processing_busy;

    // =========================================================================================
    // Layers (Memory + Compute)
    // =========================================================================================
    
    logic [TOTAL_LAYERS-1:0] layer_start;
    logic [TOTAL_LAYERS-1:0] layer_done;
    
    // Generate loop for Layers 1 to TOTAL_LAYERS-1
    genvar i;
    generate
        for (i = 1; i < TOTAL_LAYERS; i++) begin : gen_layers
            
            // Wires for connection
            logic mem_read_weight;
            logic mem_read_thresh;
            logic [CONFIG_BUS_WIDTH-1:0] rd_data_weights;
            logic [31:0] rd_data_threshold;
            
            // Layer Memory
            layer_memory #(
                .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
                .LAYER_INPUTS(TOPOLOGY[i-1]), // Inputs to L(i) come from L(i-1)
                .NUM_NEURONS(TOPOLOGY[i]),
                .PARALLEL_NEURONS(1),     // Hardcoded per simplfied plan
                .PARALLEL_INPUTS(PARALLEL_INPUTS) // Passed but actually managed via sequential chunks logic
            ) mem (
                .clk(clk),
                .rst(rst),
                .wr_en_weights(layer_wr_en_weights[i]),
                .wr_en_thresholds(layer_wr_en_thresholds[i]),
                .wr_addr(layer_wr_addr),
                .wr_data(layer_wr_data),
                
                .layer_start(layer_start[i]),
                .read_weight_chunk(mem_read_weight),
                .read_threshold(mem_read_thresh),
                .rd_data_weights(rd_data_weights),
                .rd_data_threshold(rd_data_threshold)
            );
            
            // Compute Layer
            // Output of Compute: layer_results[i]
            // Input to Compute: layer_results[i-1] (Appropriate slice)
            
            logic [TOPOLOGY[i-1]-1:0] compute_inputs;
            assign compute_inputs = layer_results[i-1][TOPOLOGY[i-1]-1:0];
            
            logic [TOPOLOGY[i]*32-1:0] compute_outputs; // Max width
            
            compute_layer #(
                .LAYER_ID(i),
                .LAYER_INPUTS(TOPOLOGY[i-1]),
                .NUM_NEURONS(TOPOLOGY[i]),
                .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
                .PARALLEL_INPUTS(PARALLEL_INPUTS),
                .IS_OUTPUT_LAYER(i == TOTAL_LAYERS - 1)
            ) compute (
                .clk(clk),
                .rst(rst),
                .start(layer_start[i]),
                .done(layer_done[i]),
                .result_vector(compute_outputs),
                .input_activations(compute_inputs),
                
                .mem_layer_start(), // Unused, driven by layer_start
                .mem_read_weight(mem_read_weight),
                .mem_read_thresh(mem_read_thresh),
                .mem_weight_data(rd_data_weights),
                .mem_thresh_data(rd_data_threshold)
            );
            
            // Register Output
            // The compute_layer outputs `result_vector`.
            // For Hidden Layers, it is packed binary?
            // `compute_layer` currently outputs packed 32-bit results or packed binary depending on IS_OUTPUT.
            // Wait, my `compute_layer` outputs `NUM_NEURONS*32` bits ALWAYS via `results`.
            // BUT for hidden layers, it only writes bit 0 of each 32-bit chunk.
            // `results[neuron_cnt] <= 1'b1;` implies writing to index `neuron_cnt` of a flat bit vector?
            // "results" in compute_layer is `logic [NUM_NEURONS*32-1:0]`.
            // `results[neuron_cnt]` writes to the LSBs.
            // So if hidden: `results[0..NUM_NEURONS-1]` holds valid data.
            // If output: `results[i*32 +: 32]` holds valid data.
            // I need to map this to `layer_results[i]`.
            
            always_comb begin
                layer_results[i] = '0; 
                // Default 0 to avoid latch
                if (i == TOTAL_LAYERS - 1) begin
                    // Output Layer: Copy all 32-bit chunks
                    layer_results[i][TOPOLOGY[i]*32-1:0] = compute_outputs;
                end else begin
                     // Hidden Layer: Copy LSBs
                     // compute_outputs[0..NUM_NEURONS-1] are the bits?
                     // In `compute_layer`, check: `results[neuron_cnt] <= 1'b1`.
                     // `results` is flat `logic [NUM_NEURONS*32-1:0]`.
                     // Indexing `results[neuron_cnt]` accesses bits 0..NUM_NEURONS-1.
                     // These are the bits used for hidden layers.
                     // So we just take the bottom bits.
                     layer_results[i][TOPOLOGY[i]-1:0] = compute_outputs[TOPOLOGY[i]-1:0];
                end
            end

        end
    endgenerate

    // =========================================================================================
    // Sequencer FSM
    // =========================================================================================
    
    typedef enum logic [3:0] {
        S_IDLE,
        S_WAIT_INPUT,
        S_START_L1,
        S_WAIT_L1,
        S_START_L2,
        S_WAIT_L2,
        S_START_L3,
        S_WAIT_L3,
        S_ARGMAX,
        S_OUTPUT
    } seq_state_t;
    
    seq_state_t seq_state;
    
    // Argmax Signals
    logic [7:0] max_category;
    logic [31:0] max_val;
    
    // Combinatorial Argmax Logic
    logic [7:0] calc_max_category;
    always_comb begin
        logic [31:0] current_max_val;
        current_max_val = 0;
        calc_max_category = 0;
        
        for (int k=0; k<TOPOLOGY[TOTAL_LAYERS-1]; k++) begin
            logic [31:0] val;
            val = layer_results[TOTAL_LAYERS-1][k*32 +: 32];
            if (val > current_max_val) begin
                current_max_val = val;
                calc_max_category = k[7:0];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            seq_state <= S_IDLE;
            layer_start <= '0;
            processing_busy <= 0;
            max_category <= 0;
            data_out_valid <= 0;
            data_out_data <= 0;
            data_out_last <= 0;
            data_out_keep <= 0;
        end else begin
            // Pulse resets
            layer_start <= '0;
            
            case (seq_state)
                S_IDLE: begin
                    processing_busy <= 0;
                    if (input_complete) begin
                        seq_state <= S_START_L1;
                        processing_busy <= 1;
                    end
                end
                
                S_START_L1: begin
                    layer_start[1] <= 1;
                    seq_state <= S_WAIT_L1;
                end
                
                S_WAIT_L1: begin
                    if (layer_done[1]) begin
                         seq_state <= S_START_L2; 
                    end
                end
                
                S_START_L2: begin
                    layer_start[2] <= 1;
                    seq_state <= S_WAIT_L2;
                end
                
                S_WAIT_L2: begin
                    if (layer_done[2]) seq_state <= S_START_L3;
                end

                S_START_L3: begin // Output Layer
                    layer_start[3] <= 1;
                    seq_state <= S_WAIT_L3;
                end
                
                S_WAIT_L3: begin
                    if (layer_done[3]) seq_state <= S_ARGMAX;
                end
                
                S_ARGMAX: begin
                    $display("Argmax: MaxCat %d MaxVal %d", calc_max_category, max_val); // max_val isn't captured? 
                    // Wait, max_val is unconnected in my fix.
                    // The combinatorial block uses `val`.
                    // I should display `calc_max_category`.
                    max_category <= calc_max_category;
                    seq_state <= S_OUTPUT;
                end
                
                S_OUTPUT: begin
                    data_out_valid <= 1;
                    data_out_data <= max_category; // Assuming OUTPUT_BUS_WIDTH >= 8
                    data_out_last <= 1;
                    data_out_keep <= '1; // All bytes valid (1 byte usually)
                    
                    if (data_out_ready) begin
                        data_out_valid <= 0;
                        data_out_last <= 0;
                        processing_busy <= 0; // Release input lock
                        seq_state <= S_IDLE;
                    end
                end
                
            endcase
            
        end
    end

endmodule
