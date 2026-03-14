module compute_layer #(
    parameter int LAYER_ID = 1,
    parameter int LAYER_INPUTS = 784,
    parameter int NUM_NEURONS = 256,
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int PARALLEL_INPUTS = 64,
    parameter bit IS_OUTPUT_LAYER = 0
) (
    input logic clk,
    input logic rst,

    // Control / Handshake
    input  logic                      start,         // Pulse to start computation
    output logic                      done,          // Single-cycle pulse when layer is complete
    output logic                      is_idle,       // High when the state machine is in IDLE
    output logic [NUM_NEURONS*32-1:0] result_vector, // Packed output datas

    input logic [LAYER_INPUTS-1:0] input_activations,  // From previous layer

    // Memory Interface
    output logic                        mem_layer_start,
    output logic                        mem_read_weight,
    output logic                        mem_read_thresh,
    input  logic [CONFIG_BUS_WIDTH-1:0] mem_weight_data,
    input  logic [                31:0] mem_thresh_data
);

    // Constants
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int BITS_PER_NEURON = BYTES_PER_NEURON * 8;  // Padded
    localparam int CHUNKS_PER_NEURON = (BITS_PER_NEURON + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;

    // State
    typedef enum logic [2:0] {
        IDLE,
        PREFETCH,
        COMPUTE_NEURON,
        FINISH_NEURON,
        DONE_STATE
    } state_t;

    state_t state;

    // Counters
    logic [15:0] neuron_cnt;
    logic [15:0] chunk_cnt;  // Actually req_chunk_idx in previous logic, simplifying to one counter set

    // Result Storage
    logic [NUM_NEURONS*32-1:0] results;
    assign result_vector = results;
    assign done = (state == DONE_STATE);
    assign is_idle = (state == IDLE);

    assign mem_layer_start = (state == IDLE && start);

    // Mux inputs and Generate Valid Mask;
    logic [CONFIG_BUS_WIDTH-1:0] muxed_input;
    logic [CONFIG_BUS_WIDTH-1:0] valid_mask;

    always_comb begin
        for (int i = 0; i < CONFIG_BUS_WIDTH; i++) begin
            if ((chunk_cnt * CONFIG_BUS_WIDTH + i) < LAYER_INPUTS) begin
                muxed_input[i] = input_activations[chunk_cnt*CONFIG_BUS_WIDTH+i];
                valid_mask[i]  = 1'b1;
            end else begin
                muxed_input[i] = 1'b0;
                valid_mask[i]  = 1'b0;
            end
        end
    end

    // Neural Processor Integration
    logic np_valid_in;
    logic np_last;
    logic [CONFIG_BUS_WIDTH-1:0] np_x;
    logic [CONFIG_BUS_WIDTH-1:0] np_w;
    logic [31:0] np_threshold;
    logic np_y;
    logic np_valid_out;
    logic [31:0] np_popcount_out;

    neural_processor #(
        .N(CONFIG_BUS_WIDTH),
        .ACC_WIDTH(32)
    ) np_inst (
        .clk(clk),
        .rst(rst),
        .valid_in(np_valid_in),
        .last(np_last),
        .x(np_x),
        .w(np_w),
        .threshold(np_threshold),
        .y(np_y),
        .valid_out(np_valid_out),
        .popcount_out(np_popcount_out)
    );

    assign np_valid_in = (state == COMPUTE_NEURON);
    assign np_last = (state == COMPUTE_NEURON && chunk_cnt == CHUNKS_PER_NEURON - 1);
    
    // Mask unused padding bits as per neural_processor integration requirements
    assign np_x = muxed_input & valid_mask;
    assign np_w = (mem_weight_data & valid_mask) | (~valid_mask);
    assign np_threshold = mem_thresh_data;

    // Logic
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            neuron_cnt <= '0;
            chunk_cnt <= '0;
            results <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        // Memory is synchronous (BRAM). Give it 1 cycle to prefetch
                        // chunk 0 + threshold 0 after layer_start resets pointers.
                        state <= PREFETCH;
                        neuron_cnt <= '0;
                        chunk_cnt <= '0;
                    end
                end

                PREFETCH: begin
                    // One-cycle bubble so layer_memory's synchronous outputs become valid.
                    state <= COMPUTE_NEURON;
                end

                COMPUTE_NEURON: begin
                    if (chunk_cnt < CHUNKS_PER_NEURON - 1) begin
                        chunk_cnt <= chunk_cnt + 1;
                    end else begin
                        state <= FINISH_NEURON;
                    end
                end

                FINISH_NEURON: begin
                    if (np_valid_out) begin
                        // Debug
                        $display("[Time %0t] Layer %0d Neuron %0d Acc: %0d Thresh: %0d", $time, LAYER_ID,
                                 neuron_cnt, np_popcount_out, $signed(mem_thresh_data));

                        // Apply Threshold / Store Result
                        if (IS_OUTPUT_LAYER) begin
                            results[neuron_cnt*32+:32] <= np_popcount_out;
                        end else begin
                            if ($signed(np_popcount_out) >= $signed(mem_thresh_data)) results[neuron_cnt] <= 1'b1;
                            else results[neuron_cnt] <= 1'b0;
                        end

                        if (neuron_cnt == NUM_NEURONS - 1) begin
                            state <= DONE_STATE;
                        end else begin
                            // After advancing to the next neuron's pointers, insert a
                            // 1-cycle prefetch bubble for synchronous memory.
                            state <= PREFETCH;
                            neuron_cnt <= neuron_cnt + 1;
                            chunk_cnt <= 0;
                        end
                    end
                end

                DONE_STATE: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    // Memory read signals
    // Read Weight: Assert during COMPUTE_NEURON to prepare next chunk's data? 
    // Wait, with 0-latency model, data is valid combinatorially. 
    // We assert `mem_read_weight` to ADVANCE the pointer for the NEXT cycle.
    // So in COMPUTE_NEURON, we assert it.
    assign mem_read_weight = (state == COMPUTE_NEURON);
    assign mem_read_thresh = (state == FINISH_NEURON && np_valid_out);

endmodule
