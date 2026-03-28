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
    localparam int BUS_BYTE_SHIFT = $clog2(BUS_BYTES);
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int CHUNKS_PER_NEURON = (BYTES_PER_NEURON + BUS_BYTES - 1) / BUS_BYTES;
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BYTES_PER_NEURON * 8;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam int THRESH_WORDS_PER_BEAT = CONFIG_BUS_WIDTH / 32;
    localparam int THRESH_MEM_DEPTH = (NUM_NEURONS + THRESH_WORDS_PER_BEAT - 1) / THRESH_WORDS_PER_BEAT;
    localparam int NEURON_STRIDE_BYTES = PARALLEL_NEURONS * BYTES_PER_NEURON;
    localparam int MAX_NEURON_INDEX = (NUM_NEURONS > 0) ? (NUM_NEURONS + PARALLEL_NEURONS - 2) : 0;
    localparam int MAX_CHUNK_OFFSET = (CHUNKS_PER_NEURON > 0) ? ((CHUNKS_PER_NEURON - 1) * BUS_BYTES) : 0;
    localparam int MAX_BYTE_ADDR = (MAX_NEURON_INDEX * BYTES_PER_NEURON) + MAX_CHUNK_OFFSET;
    localparam int BYTE_ADDR_WIDTH = (MAX_BYTE_ADDR > 0) ? $clog2(MAX_BYTE_ADDR + 1) : 1;
    localparam int THRESH_IDX_WIDTH = (NUM_NEURONS + PARALLEL_NEURONS > 1) ? $clog2(NUM_NEURONS + PARALLEL_NEURONS) : 1;
    localparam int THRESH_ADDR_WIDTH = (THRESH_MEM_DEPTH > 1) ? $clog2(THRESH_MEM_DEPTH) : 1;
    localparam int THRESH_SUBWORD_WIDTH = (THRESH_WORDS_PER_BEAT > 1) ? $clog2(THRESH_WORDS_PER_BEAT) : 1;

    logic [31:0] neuron_base_idx;
    logic [BYTE_ADDR_WIDTH-1:0] neuron_base_byte_addr;
    logic [BYTE_ADDR_WIDTH-1:0] chunk_byte_offset;

    logic [31:0] neuron_base_idx_req;
    logic [BYTE_ADDR_WIDTH-1:0] neuron_base_byte_addr_req;
    logic [BYTE_ADDR_WIDTH-1:0] chunk_byte_offset_req;

    logic [PARALLEL_NEURONS-1:0][31:0] word_addr_safe_req;
    logic [PARALLEL_NEURONS-1:0][31:0] word_addr_plus1_safe_req;
    logic [PARALLEL_NEURONS-1:0][31:0] bit_offset_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_IDX_WIDTH-1:0] threshold_idx_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_ADDR_WIDTH-1:0] threshold_word_addr_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_SUBWORD_WIDTH-1:0] threshold_subword_req;

    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_lo_q;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_hi_raw_q;
    logic [PARALLEL_NEURONS-1:0]                       weights_hi_valid_q;
    logic [PARALLEL_NEURONS-1:0][31:0]                bit_offset_q;

    // Pack thresholds into bus-width words so the config path writes one aligned
    // memory word at a time instead of updating many individual 32-bit registers.
    (* ram_style = "distributed" *) logic [CONFIG_BUS_WIDTH-1:0] threshold_mem [0:THRESH_MEM_DEPTH-1];

    function automatic logic [CONFIG_BUS_WIDTH-1:0] apply_byte_wstrb(
        input logic [CONFIG_BUS_WIDTH-1:0] curr_word,
        input logic [CONFIG_BUS_WIDTH-1:0] new_word,
        input logic [BUS_BYTES-1:0]        byte_strobes
    );
        logic [CONFIG_BUS_WIDTH-1:0] merged_word;
        begin
            merged_word = curr_word;
            for (int byte_idx = 0; byte_idx < BUS_BYTES; byte_idx++) begin
                if (byte_strobes[byte_idx])
                    merged_word[byte_idx*8 +: 8] = new_word[byte_idx*8 +: 8];
            end
            return merged_word;
        end
    endfunction

    function automatic logic [31:0] select_threshold_word(
        input logic [CONFIG_BUS_WIDTH-1:0] threshold_word,
        input logic [THRESH_SUBWORD_WIDTH-1:0] threshold_slot
    );
        begin
            return threshold_word[threshold_slot*32 +: 32];
        end
    endfunction

    always_comb begin
        neuron_base_idx_req = neuron_base_idx;
        neuron_base_byte_addr_req = neuron_base_byte_addr;
        chunk_byte_offset_req = chunk_byte_offset;

        if (rst || layer_start) begin
            neuron_base_idx_req = '0;
            neuron_base_byte_addr_req = '0;
            chunk_byte_offset_req = '0;
        end else if (read_threshold) begin
            neuron_base_idx_req = neuron_base_idx + PARALLEL_NEURONS;
            neuron_base_byte_addr_req = neuron_base_byte_addr + BYTE_ADDR_WIDTH'(NEURON_STRIDE_BYTES);
            chunk_byte_offset_req = '0;
        end else if (read_weight_chunk) begin
            chunk_byte_offset_req = chunk_byte_offset + BYTE_ADDR_WIDTH'(BUS_BYTES);
        end

        for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
            logic [THRESH_IDX_WIDTH-1:0] neuron_idx_req;
            logic [BYTE_ADDR_WIDTH-1:0] lane_byte_addr_req;
            logic [31:0] word_addr_req;
            logic [31:0] thresh_word_addr_full;

            neuron_idx_req = neuron_base_idx_req + lane;
            lane_byte_addr_req = neuron_base_byte_addr_req + BYTE_ADDR_WIDTH'(lane * BYTES_PER_NEURON) + chunk_byte_offset_req;
            word_addr_req = (lane_byte_addr_req >> BUS_BYTE_SHIFT);
            thresh_word_addr_full = neuron_idx_req / THRESH_WORDS_PER_BEAT;
            bit_offset_req[lane] = {26'd0, lane_byte_addr_req[BUS_BYTE_SHIFT-1:0], 3'b000};
            threshold_idx_req[lane] = neuron_idx_req;
            threshold_subword_req[lane] = THRESH_SUBWORD_WIDTH'(neuron_idx_req % THRESH_WORDS_PER_BEAT);

            if (word_addr_req < WEIGHT_MEM_DEPTH)
                word_addr_safe_req[lane] = word_addr_req;
            else
                word_addr_safe_req[lane] = (WEIGHT_MEM_DEPTH == 0) ? '0 : (WEIGHT_MEM_DEPTH - 1);

            if (word_addr_safe_req[lane] == (WEIGHT_MEM_DEPTH - 1))
                word_addr_plus1_safe_req[lane] = word_addr_safe_req[lane];
            else
                word_addr_plus1_safe_req[lane] = word_addr_safe_req[lane] + 1;

            if (thresh_word_addr_full < THRESH_MEM_DEPTH)
                threshold_word_addr_req[lane] = THRESH_ADDR_WIDTH'(thresh_word_addr_full);
            else
                threshold_word_addr_req[lane] = (THRESH_MEM_DEPTH == 0) ? '0 : THRESH_ADDR_WIDTH'(THRESH_MEM_DEPTH - 1);
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            neuron_base_idx <= '0;
            neuron_base_byte_addr <= '0;
            chunk_byte_offset <= '0;
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                rd_data_threshold[lane] <= '0;
            end
        end else begin
            neuron_base_idx <= neuron_base_idx_req;
            neuron_base_byte_addr <= neuron_base_byte_addr_req;
            chunk_byte_offset <= chunk_byte_offset_req;

            if (wr_en_thresholds && (wr_addr < THRESH_MEM_DEPTH)) begin
                threshold_mem[wr_addr] <= apply_byte_wstrb(threshold_mem[wr_addr], wr_data, wr_strb);
            end

            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                if (threshold_idx_req[lane] < NUM_NEURONS)
                    rd_data_threshold[lane] <= select_threshold_word(
                        threshold_mem[threshold_word_addr_req[lane]],
                        threshold_subword_req[lane]
                    );
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
