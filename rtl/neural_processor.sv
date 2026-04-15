// Chunk-level neuron primitive.
// Each valid chunk performs XNOR+popcount, accumulates across chunks until `last`,
// then emits both the raw population count and the threshold comparison result.
module neural_processor #(
    parameter N = 8,  // input parallel vector width // Pw
    parameter ACC_WIDTH = 16  // threshold register size // population count register width
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 valid_in,
    input  logic                 last,
    input  logic [        N-1:0] x,
    input  logic [        N-1:0] w,
    input  logic [       31:0]   threshold,
    output logic                 y,
    output logic                 valid_out,
    output logic [       31:0]   popcount_out
);
    import bnn_util_pkg::*;

    localparam int POPCOUNT_W = clog2_safe(N + 1);

    logic [N-1:0] xnor_result;
    logic [POPCOUNT_W-1:0] popcount;
    logic [ACC_WIDTH-1:0] acc;
    logic pop_valid_q;
    logic pop_last_q;
    logic [POPCOUNT_W-1:0] popcount_q;
    logic [31:0] threshold_q;
    logic [ACC_WIDTH-1:0] popcount_extended_q;
    logic [ACC_WIDTH-1:0] acc_next_q;
    logic                 final_valid_q;
    logic [ACC_WIDTH-1:0] final_sum_q;
    logic [31:0]          final_threshold_q;

    always_comb begin
        xnor_result = ~(x ^ w);
        popcount = $countones(xnor_result);
    end

    assign popcount_extended_q = ACC_WIDTH'(popcount_q);
    assign acc_next_q = acc + popcount_extended_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            acc <= '0;
            y <= 1'b0;
            valid_out <= 1'b0;
            popcount_out <= '0;
            pop_valid_q <= 1'b0;
            pop_last_q <= 1'b0;
            popcount_q <= '0;
            threshold_q <= '0;
            final_valid_q <= 1'b0;
            final_sum_q <= '0;
            final_threshold_q <= '0;
        end else begin
            valid_out <= 1'b0;

            if (final_valid_q) begin
                y <= (32'(final_sum_q) >= final_threshold_q);
                popcount_out <= 32'(final_sum_q);
                valid_out <= 1'b1;
                final_valid_q <= 1'b0;
            end

            if (pop_valid_q) begin
                if (pop_last_q) begin
                    acc <= '0;
                    final_sum_q <= acc_next_q;
                    final_threshold_q <= threshold_q;
                    final_valid_q <= 1'b1;
                end else begin
                    acc <= acc_next_q;
                end
            end

            pop_valid_q <= valid_in;
            if (valid_in) begin
                pop_last_q <= last;
                popcount_q <= popcount;
                threshold_q <= threshold;
            end
        end
    end

endmodule
