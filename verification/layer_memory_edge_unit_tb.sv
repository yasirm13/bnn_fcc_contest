`timescale 1ns / 1ps

module layer_memory_edge_unit_tb;
    localparam int CONFIG_BUS_WIDTH = 64;
    localparam int LAYER_INPUTS = 1;
    localparam int NUM_NEURONS = 3;
    localparam int PARALLEL_NEURONS = 2;
    localparam int BUS_BYTES = CONFIG_BUS_WIDTH / 8;
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BYTES_PER_NEURON * 8;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    logic                          wr_en_weights;
    logic                          wr_en_thresholds;
    logic [31:0]                   wr_addr;
    logic [CONFIG_BUS_WIDTH-1:0]   wr_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] wr_strb;

    logic layer_start;
    logic read_weight_chunk;
    logic read_threshold;

    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] rd_data_weights;
    logic [PARALLEL_NEURONS-1:0][31:0]                 rd_data_threshold;

    logic [CONFIG_BUS_WIDTH-1:0] weight_words [0:WEIGHT_MEM_DEPTH-1];

    layer_memory #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .LAYER_INPUTS    (LAYER_INPUTS),
        .NUM_NEURONS     (NUM_NEURONS),
        .PARALLEL_NEURONS(PARALLEL_NEURONS),
        .PARALLEL_INPUTS (1)
    ) dut (
        .clk              (clk),
        .rst              (rst),
        .wr_en_weights    (wr_en_weights),
        .wr_en_thresholds (wr_en_thresholds),
        .wr_addr          (wr_addr),
        .wr_data          (wr_data),
        .wr_strb          (wr_strb),
        .layer_start      (layer_start),
        .read_weight_chunk(read_weight_chunk),
        .read_threshold   (read_threshold),
        .rd_data_weights  (rd_data_weights),
        .rd_data_threshold(rd_data_threshold)
    );

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic logic [CONFIG_BUS_WIDTH-1:0] expected_weight_slice(
        input int low_word_addr,
        input int byte_offset
    );
        logic [CONFIG_BUS_WIDTH-1:0] lo_word;
        logic [CONFIG_BUS_WIDTH-1:0] hi_word;
        logic [2*CONFIG_BUS_WIDTH-1:0] double_word;
        begin
            lo_word = (low_word_addr < WEIGHT_MEM_DEPTH) ? weight_words[low_word_addr] : '0;
            hi_word = ((low_word_addr + 1) < WEIGHT_MEM_DEPTH) ? weight_words[low_word_addr + 1] : '0;
            double_word = {hi_word, lo_word};
            return double_word >> (byte_offset * 8);
        end
    endfunction

    task automatic reset_dut();
        rst <= 1'b1;
        wr_en_weights <= 1'b0;
        wr_en_thresholds <= 1'b0;
        wr_addr <= '0;
        wr_data <= '0;
        wr_strb <= '0;
        layer_start <= 1'b0;
        read_weight_chunk <= 1'b0;
        read_threshold <= 1'b0;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic write_weight_word(input logic [31:0] addr, input logic [CONFIG_BUS_WIDTH-1:0] data_word);
        @(negedge clk);
        wr_en_weights <= 1'b1;
        wr_en_thresholds <= 1'b0;
        wr_addr <= addr;
        wr_data <= data_word;
        wr_strb <= {BUS_BYTES{1'b1}};
        @(posedge clk);
        @(negedge clk);
        wr_en_weights <= 1'b0;
        wr_addr <= '0;
        wr_data <= '0;
        wr_strb <= '0;
        @(posedge clk);
    endtask

    task automatic write_threshold_word(
        input logic [31:0] addr,
        input logic [CONFIG_BUS_WIDTH-1:0] data_word,
        input logic [CONFIG_BUS_WIDTH/8-1:0] strobe_word
    );
        @(negedge clk);
        wr_en_weights <= 1'b0;
        wr_en_thresholds <= 1'b1;
        wr_addr <= addr;
        wr_data <= data_word;
        wr_strb <= strobe_word;
        @(posedge clk);
        @(negedge clk);
        wr_en_thresholds <= 1'b0;
        wr_addr <= '0;
        wr_data <= '0;
        wr_strb <= '0;
        @(posedge clk);
    endtask

    task automatic issue_layer_start();
        @(negedge clk);
        layer_start <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        layer_start <= 1'b0;
        @(posedge clk);
    endtask

    task automatic issue_next_batch();
        @(negedge clk);
        read_threshold <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        read_threshold <= 1'b0;
        @(posedge clk);
    endtask

    task automatic check_batch(
        input logic [31:0] exp_thr0,
        input logic [31:0] exp_thr1,
        input logic [CONFIG_BUS_WIDTH-1:0] exp_w0,
        input logic [CONFIG_BUS_WIDTH-1:0] exp_w1,
        input string phase
    );
        // layer_memory weight alignment is pipelined; rd_data_weights lags
        // control strobes by multiple cycles (issue -> pipe -> output).
        // Wait long enough for rd_data_* to reflect the newly requested batch.
        repeat (2) @(posedge clk);
        #1;
        if (rd_data_threshold[0] !== exp_thr0)
            $fatal(1, "%s: thr0 mismatch got=%h exp=%h", phase, rd_data_threshold[0], exp_thr0);
        if (rd_data_threshold[1] !== exp_thr1)
            $fatal(1, "%s: thr1 mismatch got=%h exp=%h", phase, rd_data_threshold[1], exp_thr1);
        if (rd_data_weights[0] !== exp_w0)
            $fatal(1, "%s: w0 mismatch got=%h exp=%h", phase, rd_data_weights[0], exp_w0);
        if (rd_data_weights[1] !== exp_w1)
            $fatal(1, "%s: w1 mismatch got=%h exp=%h", phase, rd_data_weights[1], exp_w1);
    endtask

    initial begin
        // WEIGHT_MEM_DEPTH should be 1 for these parameters (small model).
        if (WEIGHT_MEM_DEPTH != 1) begin
            $fatal(1, "Expected WEIGHT_MEM_DEPTH=1, got %0d", WEIGHT_MEM_DEPTH);
        end

        // Pack 3 neurons (1 byte each) into one 64-bit word.
        // Lane0 (neuron0) sees offset 0, lane1 (neuron1) sees offset 1.
        weight_words[0] = 64'h0000_0000_0000_0302; // bytes: [1]=0x02, [0]=0x02? set explicitly below
        weight_words[0][7:0]   = 8'hA1; // neuron0
        weight_words[0][15:8]  = 8'hB2; // neuron1
        weight_words[0][23:16] = 8'hC3; // neuron2

        reset_dut();

        write_weight_word(32'd0, weight_words[0]);

        // Threshold word 0 contains thresholds for neuron0 (LSW) and neuron1 (MSW).
        write_threshold_word(32'd0, 64'h0000_0002_0000_0001, 8'hff);
        // Partial write should update only upper 32 bits (threshold for neuron1) and keep neuron0 intact.
        write_threshold_word(32'd0, 64'h0000_0009_0000_0000, 8'hf0);
        // Word 1 would contain neuron2 threshold (LSW) if written; write just lower 4 bytes.
        write_threshold_word(32'd1, 64'h0000_0000_0000_0003, 8'h0f);

        issue_layer_start();
        check_batch(32'h0000_0001, 32'h0000_0009,
                    expected_weight_slice(0, 0),
                    expected_weight_slice(0, 1),
                    "batch0 depth1");

        issue_next_batch();
        // Batch1 base=2: lane0=neuron2, lane1 out-of-range -> 0.
        check_batch(32'h0000_0003, 32'h0000_0000,
                    expected_weight_slice(0, 2),
                    expected_weight_slice(0, 3),
                    "batch1 depth1");

        $display("SUCCESS: layer_memory_edge_unit_tb completed all checks.");
        $finish;
    end
endmodule
