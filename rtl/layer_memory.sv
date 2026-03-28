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
    localparam int CHUNKS_PER_NEURON = (BYTES_PER_NEURON + BUS_BYTES - 1) / BUS_BYTES;
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BYTES_PER_NEURON * 8;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_MEM_DEPTH > 1) ? $clog2(WEIGHT_MEM_DEPTH) : 1;
    localparam int THRESH_WORDS_PER_BEAT = CONFIG_BUS_WIDTH / 32;
    localparam int THRESH_MEM_DEPTH = (NUM_NEURONS + THRESH_WORDS_PER_BEAT - 1) / THRESH_WORDS_PER_BEAT;
    localparam int NEURON_STRIDE_BYTES = PARALLEL_NEURONS * BYTES_PER_NEURON;
    localparam int NEURON_STRIDE_WORDS = NEURON_STRIDE_BYTES / BUS_BYTES;
    localparam int NEURON_STRIDE_REM_BYTES = NEURON_STRIDE_BYTES % BUS_BYTES;
    localparam int BYTE_OFFSET_WIDTH = (BUS_BYTES > 1) ? $clog2(BUS_BYTES) : 1;
    localparam int BIT_OFFSET_WIDTH = (CONFIG_BUS_WIDTH > 1) ? $clog2(CONFIG_BUS_WIDTH) : 1;
    localparam int THRESH_IDX_WIDTH = (NUM_NEURONS + PARALLEL_NEURONS > 1) ? $clog2(NUM_NEURONS + PARALLEL_NEURONS) : 1;
    localparam int THRESH_ADDR_WIDTH = (THRESH_MEM_DEPTH > 1) ? $clog2(THRESH_MEM_DEPTH) : 1;
    localparam int THRESH_SUBWORD_WIDTH = (THRESH_WORDS_PER_BEAT > 1) ? $clog2(THRESH_WORDS_PER_BEAT) : 1;

    logic [THRESH_IDX_WIDTH-1:0] neuron_base_idx;
    logic [WEIGHT_ADDR_WIDTH-1:0] neuron_base_word_addr;
    logic [BYTE_OFFSET_WIDTH-1:0] neuron_base_byte_offset;

    logic [THRESH_IDX_WIDTH-1:0] neuron_base_idx_req;
    logic [WEIGHT_ADDR_WIDTH-1:0] neuron_base_word_addr_req;
    logic [BYTE_OFFSET_WIDTH-1:0] neuron_base_byte_offset_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_IDX_WIDTH-1:0] threshold_idx_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_ADDR_WIDTH-1:0] threshold_word_addr_req;
    logic [PARALLEL_NEURONS-1:0][THRESH_SUBWORD_WIDTH-1:0] threshold_subword_req;
    logic [PARALLEL_NEURONS-1:0][WEIGHT_ADDR_WIDTH-1:0] weight_word_addr_lo_issue;
    logic [PARALLEL_NEURONS-1:0][WEIGHT_ADDR_WIDTH-1:0] weight_word_addr_hi_issue;
    logic [PARALLEL_NEURONS-1:0]                        weight_hi_valid_issue;
    logic [PARALLEL_NEURONS-1:0][BIT_OFFSET_WIDTH-1:0] weight_bit_offset_issue;

    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_lo_q;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] weights_word_hi_raw_q;
    logic [PARALLEL_NEURONS-1:0]                       weights_hi_valid_q;
    logic [PARALLEL_NEURONS-1:0][BIT_OFFSET_WIDTH-1:0] bit_offset_q;

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

    function automatic logic [WEIGHT_ADDR_WIDTH-1:0] clamp_weight_addr(
        input logic [WEIGHT_ADDR_WIDTH:0] addr_ext
    );
        begin
            if (addr_ext < WEIGHT_MEM_DEPTH)
                return WEIGHT_ADDR_WIDTH'(addr_ext);
            return WEIGHT_ADDR_WIDTH'(WEIGHT_MEM_DEPTH - 1);
        end
    endfunction

    function automatic logic [WEIGHT_ADDR_WIDTH-1:0] next_weight_addr(
        input logic [WEIGHT_ADDR_WIDTH-1:0] curr_addr
    );
        logic [WEIGHT_ADDR_WIDTH:0] next_addr_ext;
        begin
            next_addr_ext = {1'b0, curr_addr} + 1'b1;
            return clamp_weight_addr(next_addr_ext);
        end
    endfunction

    always_comb begin
        logic [BYTE_OFFSET_WIDTH:0] base_byte_sum;

        neuron_base_idx_req = neuron_base_idx;
        neuron_base_word_addr_req = neuron_base_word_addr;
        neuron_base_byte_offset_req = neuron_base_byte_offset;

        if (rst || layer_start) begin
            neuron_base_idx_req = '0;
            neuron_base_word_addr_req = '0;
            neuron_base_byte_offset_req = '0;
        end else if (read_threshold) begin
            base_byte_sum = neuron_base_byte_offset + BYTE_OFFSET_WIDTH'(NEURON_STRIDE_REM_BYTES);
            neuron_base_idx_req = THRESH_IDX_WIDTH'(neuron_base_idx + PARALLEL_NEURONS);
            neuron_base_word_addr_req = neuron_base_word_addr + WEIGHT_ADDR_WIDTH'(NEURON_STRIDE_WORDS);
            if (base_byte_sum >= BUS_BYTES) begin
                neuron_base_word_addr_req = neuron_base_word_addr_req + WEIGHT_ADDR_WIDTH'(1);
                neuron_base_byte_offset_req = BYTE_OFFSET_WIDTH'(base_byte_sum - BUS_BYTES);
            end else begin
                neuron_base_byte_offset_req = BYTE_OFFSET_WIDTH'(base_byte_sum);
            end
        end
    end

    for (genvar lane = 0; lane < PARALLEL_NEURONS; lane++) begin : gen_batch_addrs
        localparam int LANE_BYTE_OFFSET = lane * BYTES_PER_NEURON;
        localparam int LANE_WORD_OFFSET = LANE_BYTE_OFFSET / BUS_BYTES;
        localparam int LANE_REM_BYTES = LANE_BYTE_OFFSET % BUS_BYTES;

        always_comb begin
            logic [THRESH_ADDR_WIDTH-1:0] threshold_word_addr_tmp;
            logic [THRESH_SUBWORD_WIDTH-1:0] threshold_subword_tmp;

            threshold_word_addr_tmp = '0;
            threshold_subword_tmp = '0;

            threshold_idx_req[lane] = THRESH_IDX_WIDTH'(neuron_base_idx_req + lane);
            if (THRESH_WORDS_PER_BEAT > 1) begin
                threshold_word_addr_tmp = THRESH_ADDR_WIDTH'(threshold_idx_req[lane] / THRESH_WORDS_PER_BEAT);
                threshold_subword_tmp = THRESH_SUBWORD_WIDTH'(threshold_idx_req[lane] % THRESH_WORDS_PER_BEAT);
            end else begin
                threshold_word_addr_tmp = THRESH_ADDR_WIDTH'(threshold_idx_req[lane]);
            end

            threshold_word_addr_req[lane] = threshold_word_addr_tmp;
            threshold_subword_req[lane] = threshold_subword_tmp;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            neuron_base_idx <= '0;
            neuron_base_word_addr <= '0;
            neuron_base_byte_offset <= '0;
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                rd_data_threshold[lane] <= '0;
            end
        end else begin
            neuron_base_idx <= neuron_base_idx_req;
            neuron_base_word_addr <= neuron_base_word_addr_req;
            neuron_base_byte_offset <= neuron_base_byte_offset_req;

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
        localparam int LANE_BYTE_OFFSET = lane * BYTES_PER_NEURON;
        localparam int LANE_WORD_OFFSET = LANE_BYTE_OFFSET / BUS_BYTES;
        localparam int LANE_REM_BYTES = LANE_BYTE_OFFSET % BUS_BYTES;
        logic [CONFIG_BUS_WIDTH-1:0] mem_weights_lo [0:WEIGHT_MEM_DEPTH-1];
        logic [CONFIG_BUS_WIDTH-1:0] mem_weights_hi [0:WEIGHT_MEM_DEPTH-1];

        always_ff @(posedge clk) begin
            logic [BYTE_OFFSET_WIDTH:0] lane_byte_sum;
            logic [BYTE_OFFSET_WIDTH-1:0] lane_byte_offset_mod;
            logic [WEIGHT_ADDR_WIDTH:0] start_word_addr_ext;
            logic [WEIGHT_ADDR_WIDTH-1:0] next_lo_addr;
            logic [WEIGHT_ADDR_WIDTH-1:0] next_hi_addr;

            if (rst) begin
                weight_word_addr_lo_issue[lane] <= '0;
                weight_word_addr_hi_issue[lane] <= '0;
                weight_hi_valid_issue[lane] <= 1'b0;
                weight_bit_offset_issue[lane] <= '0;
                weights_word_lo_q[lane] <= '0;
                weights_word_hi_raw_q[lane] <= '0;
                weights_hi_valid_q[lane] <= 1'b0;
                bit_offset_q[lane] <= '0;
            end else begin
                if (wr_en_weights && (wr_addr < WEIGHT_MEM_DEPTH)) begin
                    mem_weights_lo[wr_addr] <= wr_data;
                    mem_weights_hi[wr_addr] <= wr_data;
                end

                weights_word_lo_q[lane] <= mem_weights_lo[weight_word_addr_lo_issue[lane]];
                weights_word_hi_raw_q[lane] <= mem_weights_hi[weight_word_addr_hi_issue[lane]];
                weights_hi_valid_q[lane] <= weight_hi_valid_issue[lane];
                bit_offset_q[lane] <= weight_bit_offset_issue[lane];

                if (layer_start || read_threshold) begin
                    lane_byte_sum = neuron_base_byte_offset_req + BYTE_OFFSET_WIDTH'(LANE_REM_BYTES);
                    lane_byte_offset_mod = BYTE_OFFSET_WIDTH'(lane_byte_sum);
                    start_word_addr_ext = {1'b0, neuron_base_word_addr_req} + (WEIGHT_ADDR_WIDTH + 1)'(LANE_WORD_OFFSET);

                    if (lane_byte_sum >= BUS_BYTES) begin
                        start_word_addr_ext = start_word_addr_ext + 1'b1;
                        lane_byte_offset_mod = BYTE_OFFSET_WIDTH'(lane_byte_sum - BUS_BYTES);
                    end

                    weight_word_addr_lo_issue[lane] <= clamp_weight_addr(start_word_addr_ext);
                    weight_word_addr_hi_issue[lane] <= clamp_weight_addr(start_word_addr_ext + 1'b1);
                    weight_hi_valid_issue[lane] <= (clamp_weight_addr(start_word_addr_ext) != (WEIGHT_MEM_DEPTH - 1));
                    weight_bit_offset_issue[lane] <= BIT_OFFSET_WIDTH'(lane_byte_offset_mod << 3);
                end else if (read_weight_chunk) begin
                    next_lo_addr = next_weight_addr(weight_word_addr_lo_issue[lane]);
                    next_hi_addr = next_weight_addr(next_lo_addr);

                    weight_word_addr_lo_issue[lane] <= next_lo_addr;
                    weight_word_addr_hi_issue[lane] <= next_hi_addr;
                    weight_hi_valid_issue[lane] <= (next_lo_addr != (WEIGHT_MEM_DEPTH - 1));
                end
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
