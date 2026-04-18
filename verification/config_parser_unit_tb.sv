// Unit testbench for config_parser (CONFIG_BUS_WIDTH=64).
// Covers header/payload sequencing, TKEEP-to-strobe propagation, address sequencing,
// idle gaps between beats, early/absent TLAST termination, and rejecting invalid messages.
`timescale 1ns / 1ps

module config_parser_unit_tb;
    import bnn_types_pkg::*;

    localparam int CONFIG_BUS_WIDTH = 64;
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

    function automatic logic [63:0] make_header_word0(
        input msg_type_e msg_type,
        input logic [7:0] layer_id,
        input logic [15:0] layer_inputs,
        input logic [15:0] num_neurons,
        input logic [15:0] bytes_per_neuron
    );
        logic [63:0] word0;
        begin
            word0 = '0;
            word0[7:0]   = msg_type;
            word0[15:8]  = layer_id;
            word0[31:16] = layer_inputs;
            word0[47:32] = num_neurons;
            word0[63:48] = bytes_per_neuron;
            return word0;
        end
    endfunction

    function automatic logic [63:0] make_header_word1(input logic [31:0] total_bytes);
        logic [63:0] word1;
        begin
            word1 = '0;
            word1[31:0] = total_bytes;
            return word1;
        end
    endfunction

    task automatic expect_no_write(input string phase);
        #1;
        if (layer_wr_en_weights !== '0) begin
            $fatal(1, "%s: unexpected weights enable %b", phase, layer_wr_en_weights);
        end
        if (layer_wr_en_thresholds !== '0) begin
            $fatal(1, "%s: unexpected thresholds enable %b", phase, layer_wr_en_thresholds);
        end
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
        if (layer_wr_en_weights !== exp_weight_en) begin
            $fatal(1, "%s: weights enable mismatch. got=%b exp=%b", phase, layer_wr_en_weights, exp_weight_en);
        end
        if (layer_wr_en_thresholds !== exp_threshold_en) begin
            $fatal(1, "%s: thresholds enable mismatch. got=%b exp=%b", phase, layer_wr_en_thresholds, exp_threshold_en);
        end
        if (layer_wr_addr !== exp_addr) begin
            $fatal(1, "%s: address mismatch. got=%0d exp=%0d", phase, layer_wr_addr, exp_addr);
        end
        if (layer_wr_data !== exp_data) begin
            $fatal(1, "%s: data mismatch. got=%h exp=%h", phase, layer_wr_data, exp_data);
        end
        if (layer_wr_strb !== exp_keep) begin
            $fatal(1, "%s: strobe mismatch. got=%h exp=%h", phase, layer_wr_strb, exp_keep);
        end
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

    task automatic drive_idle_cycles(input int count);
        for (int i = 0; i < count; i++) begin
            drive_idle_cycle();
        end
    endtask

    task automatic reset_dut();
        rst          <= 1'b1;
        config_valid <= 1'b0;
        config_data  <= '0;
        config_keep  <= '0;
        config_last  <= 1'b0;

        #1;
        if (config_ready !== 1'b0) begin
            $fatal(1, "config_ready should deassert during reset");
        end

        repeat (2) @(posedge clk);
        rst <= 1'b0;
        #1;
        if (config_ready !== 1'b1) begin
            $fatal(1, "config_ready should assert after reset");
        end
        @(posedge clk);
    endtask

    initial begin
        logic [63:0] weights_header0;
        logic [63:0] weights_header1;
        logic [63:0] weights_payload0;
        logic [63:0] weights_payload1;
        logic [63:0] thresh_header0;
        logic [63:0] thresh_header1;
        logic [63:0] thresh_payload0;
        logic [63:0] invalid_header0;
        logic [63:0] invalid_header1;
        logic [63:0] invalid_payload0;
        logic [63:0] short_header0;
        logic [63:0] short_header1;
        logic [63:0] short_payload0;
        logic [63:0] layer2_header0;
        logic [63:0] layer2_header1;
        logic [63:0] layer2_payload0;
        logic [63:0] early_last_header0;
        logic [63:0] early_last_header1;
        logic [63:0] early_last_payload0;
        logic [63:0] unknown_type_header0;
        logic [63:0] unknown_type_header1;
        logic [63:0] unknown_type_payload0;

        weights_header0 = make_header_word0(MSG_TYPE_WEIGHTS, 8'd0, 16'd13, 16'd4, 16'd4);
        weights_header1 = make_header_word1(32'd16);
        weights_payload0 = 64'h0123_4567_89ab_cdef;
        weights_payload1 = 64'hfedc_ba98_7654_3210;

        thresh_header0 = make_header_word0(MSG_TYPE_THRESHOLDS, 8'd1, 16'd32, 16'd1, 16'd4);
        thresh_header1 = make_header_word1(32'd4);
        thresh_payload0 = 64'h0000_0000_dead_beef;

        invalid_header0 = make_header_word0(MSG_TYPE_WEIGHTS, 8'd3, 16'd8, 16'd1, 16'd8);
        invalid_header1 = make_header_word1(32'd8);
        invalid_payload0 = 64'hcafe_f00d_1234_5678;

        // total_bytes <= BYTES_PER_BEAT triggers payload_last_addr=0 and the parser should
        // return to header state after the first payload beat even if TLAST is not asserted.
        short_header0 = make_header_word0(MSG_TYPE_WEIGHTS, 8'd0, 16'd8, 16'd1, 16'd1);
        short_header1 = make_header_word1(32'd8);
        short_payload0 = 64'h1111_2222_3333_4444;

        // Last valid layer_id is TOTAL_LAYERS-2, which maps to layer_wr_en_* at index +1.
        layer2_header0 = make_header_word0(MSG_TYPE_WEIGHTS, 8'd2, 16'd8, 16'd1, 16'd1);
        layer2_header1 = make_header_word1(32'd8);
        layer2_payload0 = 64'h9999_aaaa_bbbb_cccc;

        // If TLAST is asserted early, the parser should terminate the payload immediately.
        early_last_header0 = make_header_word0(MSG_TYPE_WEIGHTS, 8'd0, 16'd13, 16'd4, 16'd4);
        early_last_header1 = make_header_word1(32'd16);
        early_last_payload0 = 64'h0123_4567_89ab_cdef;

        // Unknown msg_type should not assert any write enables.
        unknown_type_header0 = make_header_word0(msg_type_e'(8'd2), 8'd0, 16'd13, 16'd4, 16'd4);
        unknown_type_header1 = make_header_word1(32'd8);
        unknown_type_payload0 = 64'h1357_9bdf_2468_ace0;

        reset_dut();

        // Idle cycles between beats must not affect header/payload sequencing.
        drive_idle_cycles(2);
        send_beat_no_write(weights_header0, 8'hff, 1'b0, "weights header word 0");
        drive_idle_cycles(1);
        send_beat_no_write(weights_header1, 8'hff, 1'b0, "weights header word 1");
        send_beat_with_write(weights_payload0, 8'hff, 1'b0, 4'b0010, 4'b0000, 32'd0, "weights payload beat 0");
        send_beat_with_write(weights_payload1, 8'hff, 1'b1, 4'b0010, 4'b0000, 32'd1, "weights payload beat 1");
        drive_idle_cycle();

        send_beat_no_write(thresh_header0, 8'hff, 1'b0, "threshold header word 0");
        send_beat_no_write(thresh_header1, 8'hff, 1'b0, "threshold header word 1");
        send_beat_with_write(thresh_payload0, 8'h0f, 1'b0, 4'b0000, 4'b0100, 32'd0, "threshold payload beat 0");
        drive_idle_cycle();

        // Short payload should auto-terminate after one beat even without TLAST.
        send_beat_no_write(short_header0, 8'hff, 1'b0, "short header word 0");
        send_beat_no_write(short_header1, 8'hff, 1'b0, "short header word 1");
        send_beat_with_write(short_payload0, 8'hff, 1'b0, 4'b0010, 4'b0000, 32'd0, "short payload beat 0 (no tlast)");
        drive_idle_cycle();

        // Verify layer_id=2 maps to enable at index 3 (TOTAL_LAYERS=4).
        send_beat_no_write(layer2_header0, 8'hff, 1'b0, "layer2 header word 0");
        send_beat_no_write(layer2_header1, 8'hff, 1'b0, "layer2 header word 1");
        send_beat_with_write(layer2_payload0, 8'hff, 1'b1, 4'b1000, 4'b0000, 32'd0, "layer2 payload beat 0");
        drive_idle_cycle();

        // Early TLAST should terminate message after first payload beat.
        send_beat_no_write(early_last_header0, 8'hff, 1'b0, "early-last header word 0");
        send_beat_no_write(early_last_header1, 8'hff, 1'b0, "early-last header word 1");
        send_beat_with_write(early_last_payload0, 8'hff, 1'b1, 4'b0010, 4'b0000, 32'd0, "early-last payload beat 0 (tlast)");
        drive_idle_cycle();

        // Unknown message type should not write.
        send_beat_no_write(unknown_type_header0, 8'hff, 1'b0, "unknown-type header word 0");
        send_beat_no_write(unknown_type_header1, 8'hff, 1'b0, "unknown-type header word 1");
        send_beat_no_write(unknown_type_payload0, 8'hff, 1'b1, "unknown-type payload beat 0");
        drive_idle_cycle();

        send_beat_no_write(invalid_header0, 8'hff, 1'b0, "invalid header word 0");
        send_beat_no_write(invalid_header1, 8'hff, 1'b0, "invalid header word 1");
        send_beat_no_write(invalid_payload0, 8'hff, 1'b1, "invalid payload beat");
        drive_idle_cycle();

        if (layer_wr_addr !== 32'd0) begin
            $fatal(1, "layer_wr_addr should return to zero between messages");
        end

        $display("SUCCESS: config_parser_unit_tb completed all checks.");
        $finish;
    end
endmodule
