`timescale 1ns / 1ps

module layer_memory_unit_tb;
    localparam int CONFIG_BUS_WIDTH = 64;
    localparam int LAYER_INPUTS = 13;
    localparam int NUM_NEURONS = 5;
    localparam int PARALLEL_NEURONS = 2;
    localparam int BUS_BYTES = CONFIG_BUS_WIDTH / 8;
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BYTES_PER_NEURON * 8;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    logic                        wr_en_weights;
    logic                        wr_en_thresholds;
    logic [31:0]                 wr_addr;
    logic [CONFIG_BUS_WIDTH-1:0] wr_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] wr_strb;

    logic                        layer_start;
    logic                        read_weight_chunk;
    logic                        read_threshold;

    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] rd_data_weights;
    logic [PARALLEL_NEURONS-1:0][31:0]                 rd_data_threshold;

    logic [CONFIG_BUS_WIDTH-1:0] weight_words [0:WEIGHT_MEM_DEPTH-1];

    layer_memory #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .LAYER_INPUTS    (LAYER_INPUTS),
        .NUM_NEURONS     (NUM_NEURONS),
        .PARALLEL_NEURONS(PARALLEL_NEURONS),
        .PARALLEL_INPUTS (8)
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
        rst              <= 1'b1;
        wr_en_weights    <= 1'b0;
        wr_en_thresholds <= 1'b0;
        wr_addr          <= '0;
        wr_data          <= '0;
        wr_strb          <= '0;
        layer_start      <= 1'b0;
        read_weight_chunk <= 1'b0;
        read_threshold   <= 1'b0;

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic write_weight_word(
        input logic [31:0] addr,
        input logic [CONFIG_BUS_WIDTH-1:0] data_word
    );
        @(negedge clk);
        wr_en_weights    <= 1'b1;
        wr_en_thresholds <= 1'b0;
        wr_addr          <= addr;
        wr_data          <= data_word;
        wr_strb          <= {BUS_BYTES{1'b1}};
        @(posedge clk);

        @(negedge clk);
        wr_en_weights <= 1'b0;
        wr_addr       <= '0;
        wr_data       <= '0;
        wr_strb       <= '0;
        @(posedge clk);
    endtask

    task automatic write_threshold_word(
        input logic [31:0] addr,
        input logic [CONFIG_BUS_WIDTH-1:0] data_word,
        input logic [CONFIG_BUS_WIDTH/8-1:0] strobe_word
    );
        @(negedge clk);
        wr_en_weights    <= 1'b0;
        wr_en_thresholds <= 1'b1;
        wr_addr          <= addr;
        wr_data          <= data_word;
        wr_strb          <= strobe_word;
        @(posedge clk);

        @(negedge clk);
        wr_en_thresholds <= 1'b0;
        wr_addr          <= '0;
        wr_data          <= '0;
        wr_strb          <= '0;
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
        #1;
        if (rd_data_threshold[0] !== exp_thr0) begin
            $fatal(1, "%s: threshold lane 0 mismatch. got=%h exp=%h", phase, rd_data_threshold[0], exp_thr0);
        end
        if (rd_data_threshold[1] !== exp_thr1) begin
            $fatal(1, "%s: threshold lane 1 mismatch. got=%h exp=%h", phase, rd_data_threshold[1], exp_thr1);
        end
        if (rd_data_weights[0] !== exp_w0) begin
            $fatal(1, "%s: weight lane 0 mismatch. got=%h exp=%h", phase, rd_data_weights[0], exp_w0);
        end
        if (rd_data_weights[1] !== exp_w1) begin
            $fatal(1, "%s: weight lane 1 mismatch. got=%h exp=%h", phase, rd_data_weights[1], exp_w1);
        end
    endtask

    initial begin
        weight_words[0] = 64'h7766_5544_3322_1100;
        weight_words[1] = 64'hffee_ddcc_bbaa_9988;

        reset_dut();

        write_weight_word(32'd0, weight_words[0]);
        write_weight_word(32'd1, weight_words[1]);

        write_threshold_word(32'd0, 64'h0000_0022_0000_0011, 8'hff);
        write_threshold_word(32'd1, 64'h0000_0044_0000_0033, 8'hff);
        write_threshold_word(32'd2, 64'h0000_0000_0000_0000, 8'hff);
        write_threshold_word(32'd2, 64'hdead_beef_0000_0055, 8'h0f);

        issue_layer_start();
        check_batch(32'h0000_0011, 32'h0000_0022, expected_weight_slice(0, 0), expected_weight_slice(0, 2), "batch 0");

        issue_next_batch();
        check_batch(32'h0000_0033, 32'h0000_0044, expected_weight_slice(0, 4), expected_weight_slice(0, 6), "batch 1");

        issue_next_batch();
        check_batch(32'h0000_0055, 32'h0000_0000, expected_weight_slice(1, 0), expected_weight_slice(1, 2), "batch 2");

        @(negedge clk);
        rst <= 1'b1;
        @(posedge clk);
        #1;
        if (rd_data_threshold[0] !== 32'd0 || rd_data_threshold[1] !== 32'd0) begin
            $fatal(1, "threshold outputs should clear while reset is asserted");
        end
        @(negedge clk);
        rst <= 1'b0;
        @(posedge clk);

        $display("SUCCESS: layer_memory_unit_tb completed all checks.");
        $finish;
    end
endmodule
