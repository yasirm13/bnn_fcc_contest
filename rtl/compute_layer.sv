// Compute engine for one hidden or output layer.
// The controller preloads each PARALLEL_NEURONS batch, walks the required weight
// chunks, and emits either binarized activations or 32-bit output-layer popcounts.
module compute_layer #(
    parameter int LAYER_ID = 1,
    parameter int LAYER_INPUTS = 784,
    parameter int NUM_NEURONS = 256,
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int PARALLEL_INPUTS = 64,
    parameter int PARALLEL_NEURONS = 1,
    parameter bit IS_OUTPUT_LAYER = 0
) (
    input logic clk,
    input logic rst,

    input logic valid_in,
    output logic ready_out,
    output logic valid_out,
    input logic ready_in,
    output logic [(IS_OUTPUT_LAYER ? (NUM_NEURONS * 32) : NUM_NEURONS)-1:0] result_vector,

    input logic [LAYER_INPUTS-1:0] input_activations,

    output logic mem_layer_start,
    output logic mem_read_weight,
    output logic mem_read_thresh,
    input var logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] mem_weight_data,
    input var logic [PARALLEL_NEURONS-1:0][31:0]                 mem_thresh_data
);
    import bnn_util_pkg::*;

    localparam int BYTES_PER_NEURON = div_ceil(LAYER_INPUTS, 8);
    localparam int BITS_PER_NEURON = BYTES_PER_NEURON * 8;
    localparam int CHUNKS_PER_NEURON = div_ceil(BITS_PER_NEURON, CONFIG_BUS_WIDTH);
    localparam int RESULT_WIDTH = IS_OUTPUT_LAYER ? (NUM_NEURONS * 32) : NUM_NEURONS;
    localparam int NP_ACC_WIDTH = clog2_safe(LAYER_INPUTS + 1);
    localparam int NEURON_BASE_WIDTH = clog2_safe(NUM_NEURONS + PARALLEL_NEURONS);
    localparam int CHUNK_COUNT_WIDTH = clog2_safe(CHUNKS_PER_NEURON);
    localparam int PADDED_INPUT_BITS = CHUNKS_PER_NEURON * CONFIG_BUS_WIDTH;

    function automatic logic [PARALLEL_NEURONS-1:0] calc_lane_mask(
        input logic [NEURON_BASE_WIDTH-1:0] base_idx
    );
        logic [PARALLEL_NEURONS-1:0] lane_mask;
        begin
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                lane_mask[lane] = ((base_idx + lane) < NUM_NEURONS);
            end
            return lane_mask;
        end
    endfunction

    function automatic logic calc_last_batch(
        input logic [NEURON_BASE_WIDTH-1:0] base_idx
    );
        begin
            return ((base_idx + PARALLEL_NEURONS) >= NUM_NEURONS);
        end
    endfunction

        typedef enum logic [2:0] {
            IDLE,
            PRIME_BATCH,
            PRELOAD_BATCH,
            COMPUTE_BATCH,
            FINISH_BATCH,
            DONE_STATE
        } state_t;

    state_t state;

    logic [NEURON_BASE_WIDTH-1:0] neuron_base_cnt;
    logic [CHUNK_COUNT_WIDTH-1:0] chunk_cnt;

    logic [RESULT_WIDTH-1:0] results;
    assign result_vector = results;
    assign valid_out = (state == DONE_STATE);
    assign ready_out = (state == DONE_STATE) && ready_in;

    assign mem_layer_start = (state == IDLE) && valid_in;

    logic [PADDED_INPUT_BITS-1:0] padded_input_activations;
    logic [CONFIG_BUS_WIDTH-1:0]  input_chunks[0:CHUNKS_PER_NEURON-1];
    logic [CONFIG_BUS_WIDTH-1:0]  chunk_valid_masks[0:CHUNKS_PER_NEURON-1];
    logic [CONFIG_BUS_WIDTH-1:0]  selected_input_chunk;
    logic [CONFIG_BUS_WIDTH-1:0]  selected_valid_mask;

    assign padded_input_activations[LAYER_INPUTS-1:0] = input_activations;
    if (PADDED_INPUT_BITS > LAYER_INPUTS) begin : gen_input_pad
        assign padded_input_activations[PADDED_INPUT_BITS-1:LAYER_INPUTS] = '0;
    end

    for (genvar chunk = 0; chunk < CHUNKS_PER_NEURON; chunk++) begin : gen_input_chunks
        localparam int CHUNK_LO = chunk * CONFIG_BUS_WIDTH;
        localparam int CHUNK_BITS = ((CHUNK_LO + CONFIG_BUS_WIDTH) <= LAYER_INPUTS) ?
            CONFIG_BUS_WIDTH : (LAYER_INPUTS - CHUNK_LO);
        assign input_chunks[chunk] = padded_input_activations[CHUNK_LO +: CONFIG_BUS_WIDTH];
        if (CHUNK_BITS == CONFIG_BUS_WIDTH) begin : gen_full_mask
            assign chunk_valid_masks[chunk] = '1;
        end else begin : gen_partial_mask
            assign chunk_valid_masks[chunk] = {
                {(CONFIG_BUS_WIDTH-CHUNK_BITS){1'b0}},
                {CHUNK_BITS{1'b1}}
            };
        end
    end

    assign selected_input_chunk = input_chunks[chunk_cnt];
    assign selected_valid_mask = chunk_valid_masks[chunk_cnt];

    logic [PARALLEL_NEURONS-1:0] lane_active_q;
    logic                        last_batch_q;
        logic [PARALLEL_NEURONS-1:0] np_valid_in_q;
        logic                        np_last_q;
        logic [CONFIG_BUS_WIDTH-1:0] np_x_q;
        logic [CONFIG_BUS_WIDTH-1:0] valid_mask_q;
        logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] np_w;
        logic [PARALLEL_NEURONS-1:0][31:0] np_thresh_q;
        logic [PARALLEL_NEURONS-1:0] np_y;
        logic [PARALLEL_NEURONS-1:0] np_valid_out;
        logic [PARALLEL_NEURONS-1:0][31:0] np_popcount_out;
        logic [PARALLEL_NEURONS-1:0] np_done_mask;
        logic batch_valid_out;

        always_comb begin
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                // For BNN padding semantics, unused bits must behave like weight=1 so they do not
                // affect the XNOR result. OR with ~valid_mask forces those masked-off bits to 1.
                np_w[lane] = mem_weight_data[lane] | ~valid_mask_q;
            end
        end

        always_comb begin
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                np_done_mask[lane] = np_valid_out[lane] | ~lane_active_q[lane];
            end
        end

    assign batch_valid_out = &np_done_mask;

    for (genvar lane = 0; lane < PARALLEL_NEURONS; lane++) begin : gen_np
        neural_processor #(
            .N(CONFIG_BUS_WIDTH),
            .ACC_WIDTH(NP_ACC_WIDTH)
            ) np_inst (
                .clk(clk),
                .rst(rst),
                .valid_in(np_valid_in_q[lane]),
                .last(np_last_q),
                .x(np_x_q),
                .w(np_w[lane]),
                .threshold(np_thresh_q[lane]),
                .y(np_y[lane]),
                .valid_out(np_valid_out[lane]),
                .popcount_out(np_popcount_out[lane])
            );
        end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            neuron_base_cnt <= '0;
            chunk_cnt <= '0;
            results <= '0;
                lane_active_q <= '0;
                last_batch_q <= 1'b0;
                np_x_q <= '0;
                valid_mask_q <= '0;
                np_last_q <= 1'b0;
                for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                    np_valid_in_q[lane] <= 1'b0;
                    np_thresh_q[lane] <= '0;
                end
            end else begin
                np_last_q <= 1'b0;
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                np_valid_in_q[lane] <= 1'b0;
            end

                if (state == PRELOAD_BATCH || state == COMPUTE_BATCH) begin
                    np_x_q <= selected_input_chunk;
                    valid_mask_q <= selected_valid_mask;
                end

            if (state == PRELOAD_BATCH) begin
                for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                    np_thresh_q[lane] <= mem_thresh_data[lane];
                end
            end

            if (state == COMPUTE_BATCH) begin
                np_last_q <= (chunk_cnt == CHUNKS_PER_NEURON - 1);
                for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                    np_valid_in_q[lane] <= lane_active_q[lane];
                end
            end

                case (state)
                    IDLE: begin
                        if (valid_in) begin
                            state <= PRIME_BATCH;
                            neuron_base_cnt <= '0;
                            chunk_cnt <= '0;
                            lane_active_q <= calc_lane_mask('0);
                            last_batch_q <= calc_last_batch('0);
                        end
                    end

                    PRIME_BATCH: begin
                        state <= PRELOAD_BATCH;
                    end
    
                    PRELOAD_BATCH: begin
                        state <= COMPUTE_BATCH;
                    end

                COMPUTE_BATCH: begin
                    if (chunk_cnt < CHUNKS_PER_NEURON - 1) begin
                        chunk_cnt <= CHUNK_COUNT_WIDTH'(chunk_cnt + 1'b1);
                    end else begin
                        state <= FINISH_BATCH;
                    end
                end
    
                    FINISH_BATCH: begin
                        if (batch_valid_out) begin
                        for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                            if (lane_active_q[lane]) begin
                                if (IS_OUTPUT_LAYER) begin
                                    results[(neuron_base_cnt+lane)*32 +: 32] <= np_popcount_out[lane];
                                end else begin
                                    results[neuron_base_cnt+lane] <= np_y[lane];
                                end
                            end
                        end
    
                            if (last_batch_q) begin
                                state <= DONE_STATE;
                            end else begin
                                state <= PRIME_BATCH;
                                neuron_base_cnt <= NEURON_BASE_WIDTH'(neuron_base_cnt + PARALLEL_NEURONS);
                                chunk_cnt <= '0;
                                lane_active_q <= calc_lane_mask(NEURON_BASE_WIDTH'(neuron_base_cnt + PARALLEL_NEURONS));
                                last_batch_q <= calc_last_batch(NEURON_BASE_WIDTH'(neuron_base_cnt + PARALLEL_NEURONS));
                            end
                        end
                    end

                DONE_STATE: begin
                    if (ready_in) begin
                        state <= IDLE;
                    end
                end
            endcase
        end
        end
    
        assign mem_read_weight = ((state == PRIME_BATCH) && (CHUNKS_PER_NEURON > 1)) ||
                                 ((state == PRELOAD_BATCH) && (CHUNKS_PER_NEURON > 2)) ||
                                 ((state == COMPUTE_BATCH) && (CHUNKS_PER_NEURON > 3) &&
                                  (chunk_cnt < (CHUNKS_PER_NEURON - 3)));
        assign mem_read_thresh = (state == FINISH_BATCH) && batch_valid_out && !last_batch_q;
    
    endmodule
