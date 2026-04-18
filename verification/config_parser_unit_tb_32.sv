// Unit testbench for config_parser at CONFIG_BUS_WIDTH=32.
// Focuses on header beat splitting, TKEEP-to-strobe propagation, and invalid layer_id rejection.
`timescale 1ns / 1ps

module config_parser_unit_tb_32;
    import bnn_types_pkg::*;

    localparam int CONFIG_BUS_WIDTH = 32;
    localparam int TOTAL_LAYERS = 4;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    logic                          config_valid;
    logic                          config_ready;
    logic [CONFIG_BUS_WIDTH-1:0]   config_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] config_keep;
    logic                          config_last;

    logic [TOTAL_LAYERS-1:0]       layer_wr_en_weights;
    logic [TOTAL_LAYERS-1:0]       layer_wr_en_thresholds;
    logic [31:0]                   layer_wr_addr;
    logic [CONFIG_BUS_WIDTH-1:0]   layer_wr_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] layer_wr_strb;

    config_parser #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .TOTAL_LAYERS    (TOTAL_LAYERS)
    ) dut (
        .clk                   (clk),
        .rst                   (rst),
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

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic logic [31:0] make_hdr_word0(
        input msg_type_e msg_type,
        input logic [7:0] layer_id
    );
        logic [31:0] word0;
        begin
            word0 = '0;
            word0[7:0]  = msg_type;
            word0[15:8] = layer_id;
            return word0;
        end
    endfunction

    task automatic expect_no_write(input string phase);
        #1;
        if (layer_wr_en_weights !== '0)
            $fatal(1, "%s: unexpected weights enable %b", phase, layer_wr_en_weights);
        if (layer_wr_en_thresholds !== '0)
            $fatal(1, "%s: unexpected thresholds enable %b", phase, layer_wr_en_thresholds);
    endtask

    task automatic expect_write(
        input string phase,
        input logic [TOTAL_LAYERS-1:0] exp_weight_en,
        input logic [TOTAL_LAYERS-1:0] exp_threshold_en,
        input logic [31:0] exp_addr,
        input logic [CONFIG_BUS_WIDTH-1:0] exp_data,
        input logic [CONFIG_BUS_WIDTH/8-1:0] exp_keep
    );
        #1;
        if (layer_wr_en_weights !== exp_weight_en)
            $fatal(1, "%s: weights enable mismatch. got=%b exp=%b", phase, layer_wr_en_weights, exp_weight_en);
        if (layer_wr_en_thresholds !== exp_threshold_en)
            $fatal(1, "%s: thresholds enable mismatch. got=%b exp=%b", phase, layer_wr_en_thresholds, exp_threshold_en);
        if (layer_wr_addr !== exp_addr)
            $fatal(1, "%s: address mismatch. got=%0d exp=%0d", phase, layer_wr_addr, exp_addr);
        if (layer_wr_data !== exp_data)
            $fatal(1, "%s: data mismatch. got=%h exp=%h", phase, layer_wr_data, exp_data);
        if (layer_wr_strb !== exp_keep)
            $fatal(1, "%s: strobe mismatch. got=%h exp=%h", phase, layer_wr_strb, exp_keep);
    endtask

    task automatic send_beat_no_write(
        input logic [CONFIG_BUS_WIDTH-1:0] data_word,
        input logic [CONFIG_BUS_WIDTH/8-1:0] keep_word,
        input logic last_word,
        input string phase
    );
        @(negedge clk);
        config_valid <= 1'b1;
        config_data  <= data_word;
        config_keep  <= keep_word;
        config_last  <= last_word;
        expect_no_write(phase);
        @(posedge clk);
    endtask

    task automatic send_beat_with_write(
        input logic [CONFIG_BUS_WIDTH-1:0] data_word,
        input logic [CONFIG_BUS_WIDTH/8-1:0] keep_word,
        input logic last_word,
        input logic [TOTAL_LAYERS-1:0] exp_weight_en,
        input logic [TOTAL_LAYERS-1:0] exp_threshold_en,
        input logic [31:0] exp_addr,
        input string phase
    );
        @(negedge clk);
        config_valid <= 1'b1;
        config_data  <= data_word;
        config_keep  <= keep_word;
        config_last  <= last_word;
        expect_write(phase, exp_weight_en, exp_threshold_en, exp_addr, data_word, keep_word);
        @(posedge clk);
    endtask

    task automatic drive_idle_cycle();
        @(negedge clk);
        config_valid <= 1'b0;
        config_data  <= '0;
        config_keep  <= '0;
        config_last  <= 1'b0;
        expect_no_write("idle");
        @(posedge clk);
    endtask

    task automatic reset_dut();
        rst          <= 1'b1;
        config_valid <= 1'b0;
        config_data  <= '0;
        config_keep  <= '0;
        config_last  <= 1'b0;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        #1;
        if (config_ready !== 1'b1)
            $fatal(1, "config_ready should assert after reset");
        @(posedge clk);
    endtask

    initial begin
        reset_dut();

        // Weights message (layer_id=0 -> enable bit 1). HEADER_BEATS=4 for 32-bit bus.
        send_beat_no_write(make_hdr_word0(MSG_TYPE_WEIGHTS, 8'd0), 4'hf, 1'b0, "weights hdr beat 0");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "weights hdr beat 1");
        send_beat_no_write(32'd4, 4'hf, 1'b0, "weights hdr beat 2 (total_bytes)");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "weights hdr beat 3");
        send_beat_with_write(32'hdead_beef, 4'hf, 1'b0, 4'b0010, 4'b0000, 32'd0, "weights payload beat 0 (auto-done)");
        drive_idle_cycle();

        // Threshold message (layer_id=1 -> enable bit 2) with total_bytes=8 => 2 payload beats.
        send_beat_no_write(make_hdr_word0(MSG_TYPE_THRESHOLDS, 8'd1), 4'hf, 1'b0, "thresh hdr beat 0");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "thresh hdr beat 1");
        send_beat_no_write(32'd8, 4'hf, 1'b0, "thresh hdr beat 2 (total_bytes)");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "thresh hdr beat 3");
        send_beat_with_write(32'h0000_0001, 4'h3, 1'b0, 4'b0000, 4'b0100, 32'd0, "thresh payload beat 0");
        send_beat_with_write(32'h0000_0002, 4'hf, 1'b0, 4'b0000, 4'b0100, 32'd1, "thresh payload beat 1");
        drive_idle_cycle();

        // Invalid layer_id (TOTAL_LAYERS-1) should not write.
        send_beat_no_write(make_hdr_word0(MSG_TYPE_WEIGHTS, 8'd3), 4'hf, 1'b0, "invalid hdr beat 0");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "invalid hdr beat 1");
        send_beat_no_write(32'd4, 4'hf, 1'b0, "invalid hdr beat 2");
        send_beat_no_write(32'h0000_0000, 4'hf, 1'b0, "invalid hdr beat 3");
        send_beat_no_write(32'hcafe_f00d, 4'hf, 1'b1, "invalid payload beat 0");
        drive_idle_cycle();

        $display("SUCCESS: config_parser_unit_tb_32 completed all checks.");
        $finish;
    end
endmodule
