// Edge-case unit testbench for neural_processor.
// Stresses back-to-back valid_in chunks for N=4 and N=8, threshold changes between chunks,
// and clean accumulator reset at transaction boundaries.
`timescale 1ns / 1ps

module neural_processor_edge_unit_tb;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    // Minimal supported N: neural_processor requires N divisible by 4.
    localparam int N1 = 4;
    localparam int ACC1 = 4;

    logic           valid_in_1;
    logic           last_1;
    logic [N1-1:0]  x_1;
    logic [N1-1:0]  w_1;
    logic [31:0]    threshold_1;
    logic           y_1;
    logic           valid_out_1;
    logic [31:0]    popcount_out_1;

    neural_processor #(
        .N        (N1),
        .ACC_WIDTH(ACC1)
    ) dut1 (
        .clk         (clk),
        .rst         (rst),
        .valid_in    (valid_in_1),
        .last        (last_1),
        .x           (x_1),
        .w           (w_1),
        .threshold   (threshold_1),
        .y           (y_1),
        .valid_out   (valid_out_1),
        .popcount_out(popcount_out_1)
    );

    // N=8 stress with back-to-back valids
    localparam int N8 = 8;
    localparam int ACC8 = 8;

    logic           valid_in_8;
    logic           last_8;
    logic [N8-1:0]  x_8;
    logic [N8-1:0]  w_8;
    logic [31:0]    threshold_8;
    logic           y_8;
    logic           valid_out_8;
    logic [31:0]    popcount_out_8;

    neural_processor #(
        .N        (N8),
        .ACC_WIDTH(ACC8)
    ) dut8 (
        .clk         (clk),
        .rst         (rst),
        .valid_in    (valid_in_8),
        .last        (last_8),
        .x           (x_8),
        .w           (w_8),
        .threshold   (threshold_8),
        .y           (y_8),
        .valid_out   (valid_out_8),
        .popcount_out(popcount_out_8)
    );

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic int matches1(input logic [N1-1:0] xw, input logic [N1-1:0] ww);
        return $countones(~(xw ^ ww));
    endfunction

    function automatic int matches8(input logic [N8-1:0] xw, input logic [N8-1:0] ww);
        return $countones(~(xw ^ ww));
    endfunction

    task automatic reset_all();
        rst <= 1'b1;
        valid_in_1 <= 1'b0;
        last_1 <= 1'b0;
        x_1 <= '0;
        w_1 <= '0;
        threshold_1 <= '0;
        valid_in_8 <= 1'b0;
        last_8 <= 1'b0;
        x_8 <= '0;
        w_8 <= '0;
        threshold_8 <= '0;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic drive_bb_1(
        input logic [N1-1:0] x0, input logic [N1-1:0] w0, input logic [31:0] t0, input logic l0,
        input logic [N1-1:0] x1, input logic [N1-1:0] w1, input logic [31:0] t1, input logic l1
    );
        @(negedge clk);
        valid_in_1 <= 1'b1;
        x_1 <= x0;
        w_1 <= w0;
        threshold_1 <= t0;
        last_1 <= l0;
        @(posedge clk);
        @(negedge clk);
        valid_in_1 <= 1'b1;
        x_1 <= x1;
        w_1 <= w1;
        threshold_1 <= t1;
        last_1 <= l1;
        @(posedge clk);
        @(negedge clk);
        valid_in_1 <= 1'b0;
        last_1 <= 1'b0;
    endtask

    task automatic wait_out_1(input logic [31:0] exp_sum, input logic exp_y, input string phase);
        int c;
        for (c = 0; c < 20; c++) begin
            @(posedge clk);
            if (valid_out_1) begin
                if (popcount_out_1 !== exp_sum)
                    $fatal(1, "%s: N=1 sum mismatch got=%0d exp=%0d", phase, popcount_out_1, exp_sum);
                if (y_1 !== exp_y)
                    $fatal(1, "%s: N=1 y mismatch got=%0b exp=%0b", phase, y_1, exp_y);
                @(posedge clk);
                if (valid_out_1)
                    $fatal(1, "%s: N=1 valid_out should pulse", phase);
                return;
            end
        end
        $fatal(1, "%s: N=1 timed out waiting for output", phase);
    endtask

    task automatic drive_chunks_8_back_to_back(
        input logic [N8-1:0] a0, input logic [N8-1:0] b0, input logic [31:0] t0, input logic l0,
        input logic [N8-1:0] a1, input logic [N8-1:0] b1, input logic [31:0] t1, input logic l1,
        input logic [N8-1:0] a2, input logic [N8-1:0] b2, input logic [31:0] t2, input logic l2
    );
        @(negedge clk);
        valid_in_8 <= 1'b1; x_8 <= a0; w_8 <= b0; threshold_8 <= t0; last_8 <= l0;
        @(posedge clk);
        @(negedge clk);
        valid_in_8 <= 1'b1; x_8 <= a1; w_8 <= b1; threshold_8 <= t1; last_8 <= l1;
        @(posedge clk);
        @(negedge clk);
        valid_in_8 <= 1'b1; x_8 <= a2; w_8 <= b2; threshold_8 <= t2; last_8 <= l2;
        @(posedge clk);
        @(negedge clk);
        valid_in_8 <= 1'b0; last_8 <= 1'b0;
    endtask

    task automatic wait_out_8(input logic [31:0] exp_sum, input logic exp_y, input string phase);
        int c;
        for (c = 0; c < 40; c++) begin
            @(posedge clk);
            if (valid_out_8) begin
                if (popcount_out_8 !== exp_sum)
                    $fatal(1, "%s: N=8 sum mismatch got=%0d exp=%0d", phase, popcount_out_8, exp_sum);
                if (y_8 !== exp_y)
                    $fatal(1, "%s: N=8 y mismatch got=%0b exp=%0b", phase, y_8, exp_y);
                @(posedge clk);
                if (valid_out_8)
                    $fatal(1, "%s: N=8 valid_out should pulse", phase);
                return;
            end
        end
        $fatal(1, "%s: N=8 timed out waiting for output", phase);
    endtask

    initial begin
        int exp1;
        int exp8;

        reset_all();

        // N=4: two back-to-back chunks with last asserted on second.
        exp1 = matches1(4'b0101, 4'b1100) + matches1(4'b1111, 4'b0011);
        drive_bb_1(4'b0101, 4'b1100, 32'd5, 1'b0,
                   4'b1111, 4'b0011, 32'd5, 1'b1);
        wait_out_1(exp1, (exp1 >= 5), "N=4 back-to-back");

        // N=8: three back-to-back chunks, last on third, threshold changes (final uses last chunk).
        exp8 = matches8(8'hff, 8'h00) + matches8(8'h0f, 8'h33) + matches8(8'ha5, 8'h5a);
        drive_chunks_8_back_to_back(
            8'hff, 8'h00, 32'd1, 1'b0,
            8'h0f, 8'h33, 32'd9, 1'b0,
            8'ha5, 8'h5a, 32'd10, 1'b1
        );
        wait_out_8(exp8, (exp8 >= 10), "N=8 3-chunk back-to-back");

        // Immediately start another single-chunk transaction to ensure accumulator resets.
        exp8 = matches8(8'h00, 8'hff);
        @(negedge clk);
        valid_in_8 <= 1'b1; x_8 <= 8'h00; w_8 <= 8'hff; threshold_8 <= 32'd1; last_8 <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        valid_in_8 <= 1'b0; last_8 <= 1'b0;
        wait_out_8(exp8, 1'b0, "N=8 post-transaction reset");

        $display("SUCCESS: neural_processor_edge_unit_tb completed all checks.");
        $finish;
    end
endmodule
