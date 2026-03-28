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

    input logic start,
    output logic done,
    output logic is_idle,
    output logic [(IS_OUTPUT_LAYER ? (NUM_NEURONS * 32) : NUM_NEURONS)-1:0] result_vector,

    input logic [LAYER_INPUTS-1:0] input_activations,

    output logic mem_layer_start,
    output logic mem_read_weight,
    output logic mem_read_thresh,
    input var logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] mem_weight_data,
    input var logic [PARALLEL_NEURONS-1:0][31:0]                 mem_thresh_data
);

    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int BITS_PER_NEURON = BYTES_PER_NEURON * 8;
    localparam int CHUNKS_PER_NEURON = (BITS_PER_NEURON + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam int RESULT_WIDTH = IS_OUTPUT_LAYER ? (NUM_NEURONS * 32) : NUM_NEURONS;
    localparam int NP_ACC_WIDTH = (LAYER_INPUTS > 1) ? $clog2(LAYER_INPUTS + 1) : 1;
    localparam int NEURON_BASE_WIDTH = (NUM_NEURONS + PARALLEL_NEURONS > 1) ? $clog2(NUM_NEURONS + PARALLEL_NEURONS) : 1;
    localparam int CHUNK_COUNT_WIDTH = (CHUNKS_PER_NEURON > 1) ? $clog2(CHUNKS_PER_NEURON) : 1;

    typedef enum logic [2:0] {
        IDLE,
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
    assign done = (state == DONE_STATE);
    assign is_idle = (state == IDLE);

    assign mem_layer_start = (state == IDLE) && start;

    logic [CONFIG_BUS_WIDTH-1:0] muxed_input;
    logic [CONFIG_BUS_WIDTH-1:0] valid_mask;
    logic [CONFIG_BUS_WIDTH-1:0] masked_input;

    always_comb begin
        for (int i = 0; i < CONFIG_BUS_WIDTH; i++) begin
            if ((chunk_cnt * CONFIG_BUS_WIDTH + i) < LAYER_INPUTS) begin
                muxed_input[i] = input_activations[chunk_cnt*CONFIG_BUS_WIDTH+i];
                valid_mask[i] = 1'b1;
            end else begin
                muxed_input[i] = 1'b0;
                valid_mask[i] = 1'b0;
            end
        end
    end

    logic [PARALLEL_NEURONS-1:0] lane_active;
    logic                        last_batch;
    logic [PARALLEL_NEURONS-1:0] np_valid_in_q;
    logic                        np_last_q;
    logic [CONFIG_BUS_WIDTH-1:0] np_x_q;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] np_w_q;
    logic [PARALLEL_NEURONS-1:0][31:0] np_thresh_q;
    logic [PARALLEL_NEURONS-1:0] np_y;
    logic [PARALLEL_NEURONS-1:0] np_valid_out;
    logic [PARALLEL_NEURONS-1:0][31:0] np_popcount_out;
    logic [PARALLEL_NEURONS-1:0] np_done_mask;
    logic batch_valid_out;

    assign masked_input = muxed_input & valid_mask;

    always_comb begin
        for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
            lane_active[lane] = ((neuron_base_cnt + lane) < NUM_NEURONS);
            np_done_mask[lane] = np_valid_out[lane] | ~lane_active[lane];
        end
    end

    assign batch_valid_out = &np_done_mask;
    assign last_batch = (neuron_base_cnt + PARALLEL_NEURONS >= NUM_NEURONS);

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
            .w(np_w_q[lane]),
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
            np_x_q <= '0;
            np_last_q <= 1'b0;
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                np_valid_in_q[lane] <= 1'b0;
                np_w_q[lane] <= '0;
                np_thresh_q[lane] <= '0;
            end
        end else begin
            np_x_q <= masked_input;
            np_last_q <= (state == COMPUTE_BATCH) && (chunk_cnt == CHUNKS_PER_NEURON - 1);
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                np_valid_in_q[lane] <= 1'b0;
                np_w_q[lane] <= (mem_weight_data[lane] & valid_mask) | (~valid_mask);
                np_thresh_q[lane] <= mem_thresh_data[lane];
            end

            if (state == COMPUTE_BATCH) begin
                for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                    np_valid_in_q[lane] <= lane_active[lane];
                end
            end

            case (state)
                IDLE: begin
                    if (start) begin
                        state <= PRELOAD_BATCH;
                        neuron_base_cnt <= '0;
                        chunk_cnt <= '0;
                    end
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
                            if (lane_active[lane]) begin
                                if (IS_OUTPUT_LAYER) begin
                                    results[(neuron_base_cnt+lane)*32 +: 32] <= np_popcount_out[lane];
                                end else begin
                                    results[neuron_base_cnt+lane] <= np_y[lane];
                                end
                            end
                        end

                        if (last_batch) begin
                            state <= DONE_STATE;
                        end else begin
                            state <= PRELOAD_BATCH;
                            neuron_base_cnt <= NEURON_BASE_WIDTH'(neuron_base_cnt + PARALLEL_NEURONS);
                            chunk_cnt <= '0;
                        end
                    end
                end

                DONE_STATE: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    assign mem_read_weight = ((state == PRELOAD_BATCH) && (CHUNKS_PER_NEURON > 1)) ||
                             ((state == COMPUTE_BATCH) && (CHUNKS_PER_NEURON > 2) &&
                              (chunk_cnt < (CHUNKS_PER_NEURON - 2)));
    assign mem_read_thresh = (state == FINISH_BATCH) && batch_valid_out && !last_batch;

endmodule
