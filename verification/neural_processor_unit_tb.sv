// Unit testbench for neural_processor (N=8).
// Checks XNOR-popcount accumulation across 1+ chunks, threshold compare behavior,
// one-cycle valid_out pulses, and accumulator reset between transactions.
`timescale 1ns / 1ps

module neural_processor_unit_tb;
    localparam int N = 8;
    localparam int ACC_WIDTH = 8;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    logic             valid_in;
    logic             last;
    logic [N-1:0]     x;
    logic [N-1:0]     w;
    logic [31:0]      threshold;
    logic             y;
    logic             valid_out;
    logic [31:0]      popcount_out;

    neural_processor #(
        .N        (N),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk         (clk),
        .rst         (rst),
        .valid_in    (valid_in),
        .last        (last),
        .x           (x),
        .w           (w),
        .threshold   (threshold),
        .y           (y),
        .valid_out   (valid_out),
        .popcount_out(popcount_out)
    );

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic int count_matches(input logic [N-1:0] x_word, input logic [N-1:0] w_word);
        begin
            return $countones(~(x_word ^ w_word));
        end
    endfunction

    task automatic reset_dut();
        rst       <= 1'b1;
        valid_in  <= 1'b0;
        last      <= 1'b0;
        x         <= '0;
        w         <= '0;
        threshold <= '0;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic drive_chunk(
        input logic [N-1:0] x_word,
        input logic [N-1:0] w_word,
        input logic [31:0] threshold_word,
        input logic last_word
    );
        @(negedge clk);
        valid_in  <= 1'b1;
        x         <= x_word;
        w         <= w_word;
        threshold <= threshold_word;
        last      <= last_word;
        @(posedge clk);

        @(negedge clk);
        valid_in <= 1'b0;
        last     <= 1'b0;
    endtask

    task automatic wait_for_output(
        input logic [31:0] exp_sum,
        input logic exp_y,
        input string phase
    );
        int cycle_count;
        begin
            for (cycle_count = 0; cycle_count < 20; cycle_count++) begin
                @(posedge clk);
                if (valid_out) begin
                    if (popcount_out !== exp_sum) begin
                        $fatal(1, "%s: popcount mismatch. got=%0d exp=%0d", phase, popcount_out, exp_sum);
                    end
                    if (y !== exp_y) begin
                        $fatal(1, "%s: classification mismatch. got=%0b exp=%0b", phase, y, exp_y);
                    end
                    @(posedge clk);
                    if (valid_out !== 1'b0) begin
                        $fatal(1, "%s: valid_out should pulse for one cycle", phase);
                    end
                    return;
                end
            end
            $fatal(1, "%s: timed out waiting for valid_out", phase);
        end
    endtask

    initial begin
        logic [31:0] expected_sum;

        reset_dut();

        expected_sum = count_matches(8'hf0, 8'hf3);
        drive_chunk(8'hf0, 8'hf3, 32'd6, 1'b1);
        wait_for_output(expected_sum, 1'b1, "single-chunk transaction");

        expected_sum = count_matches(8'haa, 8'hff) + count_matches(8'h0f, 8'h33);
        drive_chunk(8'haa, 8'hff, 32'd7, 1'b0);
        drive_chunk(8'h0f, 8'h33, 32'd7, 1'b1);
        wait_for_output(expected_sum, (expected_sum >= 32'd7), "multi-chunk transaction");

        expected_sum = count_matches(8'h00, 8'hff);
        drive_chunk(8'h00, 8'hff, 32'd1, 1'b1);
        wait_for_output(expected_sum, 1'b0, "post-reset accumulation transaction");

        $display("SUCCESS: neural_processor_unit_tb completed all checks.");
        $finish;
    end
endmodule
