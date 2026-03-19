module layer_memory #(
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int LAYER_INPUTS = 784,
    parameter int NUM_NEURONS = 256,
    parameter int PARALLEL_NEURONS = 1,
    parameter int PARALLEL_INPUTS = 64
) (
    input logic clk,
    input logic rst,

    input logic wr_en_weights,
    input logic wr_en_thresholds,
    input logic [31:0] wr_addr,
    input logic [CONFIG_BUS_WIDTH-1:0] wr_data,
    input logic [CONFIG_BUS_WIDTH/8-1:0] wr_strb,

    input logic layer_start,
    input logic read_weight_chunk,
    input logic read_threshold,

    output logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] rd_data_weights,
    output logic [PARALLEL_NEURONS-1:0][31:0]                 rd_data_threshold
);

    localparam int BUS_BYTES = CONFIG_BUS_WIDTH / 8;
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BYTES_PER_NEURON * 8;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam int THRESH_WORDS_PER_BEAT = CONFIG_BUS_WIDTH / 32;

    logic [31:0] neuron_base_idx;
    logic [31:0] chunk_idx;

    logic [31:0] neuron_base_idx_req;
    logic [31:0] chunk_idx_req;

    logic [PARALLEL_NEURONS-1:0][31:0] word_addr_safe_req;
    logic [PARALLEL_NEURONS-1:0][31:0] word_addr_plus1_safe_req;
    logic [PARALLEL_NEURONS-1:0][31:0] bit_offset_req;
    logic [PARALLEL_NEURONS-1:0][31:0] threshold_idx_req;

    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_lo_q;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_hi_raw_q;
    logic [PARALLEL_NEURONS-1:0]                       weights_hi_valid_q;
    logic [PARALLEL_NEURONS-1:0][31:0]                bit_offset_q;

    logic [31:0] threshold_mem [0:NUM_NEURONS-1];

    always_comb begin
        neuron_base_idx_req = neuron_base_idx;
        chunk_idx_req = chunk_idx;

        if (rst || layer_start) begin
            neuron_base_idx_req = '0;
            chunk_idx_req = '0;
        end else if (read_threshold) begin
            neuron_base_idx_req = neuron_base_idx + PARALLEL_NEURONS;
            chunk_idx_req = '0;
        end else if (read_weight_chunk) begin
            chunk_idx_req = chunk_idx + 1;
        end

        for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
            int neuron_idx_req;
            int current_byte_addr_req;
            int word_addr_req;

            neuron_idx_req = neuron_base_idx_req + lane;
            current_byte_addr_req = neuron_idx_req * BYTES_PER_NEURON + (chunk_idx_req * BUS_BYTES);
            word_addr_req = current_byte_addr_req / BUS_BYTES;
            bit_offset_req[lane] = (current_byte_addr_req % BUS_BYTES) * 8;
            threshold_idx_req[lane] = neuron_idx_req;

            if (word_addr_req < WEIGHT_MEM_DEPTH)
                word_addr_safe_req[lane] = word_addr_req;
            else
                word_addr_safe_req[lane] = (WEIGHT_MEM_DEPTH == 0) ? '0 : (WEIGHT_MEM_DEPTH - 1);

            if (word_addr_safe_req[lane] == (WEIGHT_MEM_DEPTH - 1))
                word_addr_plus1_safe_req[lane] = word_addr_safe_req[lane];
            else
                word_addr_plus1_safe_req[lane] = word_addr_safe_req[lane] + 1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            neuron_base_idx <= '0;
            chunk_idx <= '0;
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                rd_data_threshold[lane] <= '0;
            end
        end else begin
            neuron_base_idx <= neuron_base_idx_req;
            chunk_idx <= chunk_idx_req;

            if (wr_en_thresholds) begin
                for (int word = 0; word < THRESH_WORDS_PER_BEAT; word++) begin
                    int thresh_idx;
                    thresh_idx = wr_addr * THRESH_WORDS_PER_BEAT + word;
                    if ((thresh_idx < NUM_NEURONS) && (&wr_strb[word*4 +: 4])) begin
                        threshold_mem[thresh_idx] <= wr_data[word*32 +: 32];
                    end
                end
            end

            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                if (threshold_idx_req[lane] < NUM_NEURONS)
                    rd_data_threshold[lane] <= threshold_mem[threshold_idx_req[lane]];
                else
                    rd_data_threshold[lane] <= '0;
            end
        end
    end

    for (genvar lane = 0; lane < PARALLEL_NEURONS; lane++) begin : gen_weight_ports
        logic [CONFIG_BUS_WIDTH-1:0] mem_weights_lo [0:WEIGHT_MEM_DEPTH-1];
        logic [CONFIG_BUS_WIDTH-1:0] mem_weights_hi [0:WEIGHT_MEM_DEPTH-1];

        always_ff @(posedge clk) begin
            if (rst) begin
                weights_word_lo_q[lane] <= '0;
                weights_word_hi_raw_q[lane] <= '0;
                weights_hi_valid_q[lane] <= 1'b0;
                bit_offset_q[lane] <= '0;
            end else begin
                if (wr_en_weights && (wr_addr < WEIGHT_MEM_DEPTH)) begin
                    mem_weights_lo[wr_addr] <= wr_data;
                    mem_weights_hi[wr_addr] <= wr_data;
                end

                weights_word_lo_q[lane] <= mem_weights_lo[word_addr_safe_req[lane]];
                weights_word_hi_raw_q[lane] <= mem_weights_hi[word_addr_plus1_safe_req[lane]];
                weights_hi_valid_q[lane] <= (word_addr_safe_req[lane] != (WEIGHT_MEM_DEPTH - 1));
                bit_offset_q[lane] <= bit_offset_req[lane];
            end
        end

        always_comb begin
            logic [CONFIG_BUS_WIDTH-1:0] weights_word_hi_q;
            logic [2*CONFIG_BUS_WIDTH-1:0] double_word_q;
            logic [2*CONFIG_BUS_WIDTH-1:0] shifted_double_word_q;

            weights_word_hi_q = weights_hi_valid_q[lane] ? weights_word_hi_raw_q[lane] : '0;
            double_word_q = {weights_word_hi_q, weights_word_lo_q[lane]};
            shifted_double_word_q = (double_word_q >> bit_offset_q[lane]);
            rd_data_weights[lane] = shifted_double_word_q[CONFIG_BUS_WIDTH-1:0];
        end
    end

endmodule
