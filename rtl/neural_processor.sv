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

    localparam POPCOUNT_W = (N > 1) ? $clog2(N + 1) : 1;

    logic [N-1:0] xnor_result;
    logic [POPCOUNT_W-1:0] popcount;
    logic [ACC_WIDTH-1:0] acc;
    logic accum_valid;
    logic pending_output;
    logic pop_valid_q;
    logic pop_last_q;
    logic [POPCOUNT_W-1:0] popcount_q;
    logic [31:0] threshold_pipe_q;
    logic [31:0] threshold_q;
    logic [ACC_WIDTH-1:0] popcount_extended_q;
    logic [ACC_WIDTH-1:0] next_acc_q;

    always_comb begin
        xnor_result = ~(x ^ w);
        popcount = $countones(xnor_result);
    end

    assign popcount_extended_q = ACC_WIDTH'(popcount_q);
    assign next_acc_q = (accum_valid ? acc : '0) + popcount_extended_q;
    assign popcount_out = 32'(acc);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            acc       <= '0;
            y         <= 1'b0;
            valid_out <= 1'b0;
            accum_valid <= 1'b0;
            pending_output <= 1'b0;
            pop_valid_q <= 1'b0;
            pop_last_q <= 1'b0;
            popcount_q <= '0;
            threshold_pipe_q <= '0;
            threshold_q <= '0;
        end else begin
            valid_out <= 1'b0;

            if (pending_output) begin
                y <= (32'(acc) >= threshold_q);
                valid_out <= 1'b1;
                pending_output <= 1'b0;
            end

            if (pop_valid_q) begin
                acc <= next_acc_q;
                if (pop_last_q) begin
                    threshold_q <= threshold_pipe_q;
                    accum_valid <= 1'b0;
                    pending_output <= 1'b1;
                end else begin
                    accum_valid <= 1'b1;
                end
            end

            pop_valid_q <= valid_in;
            if (valid_in) begin
                pop_last_q <= last;
                popcount_q <= popcount;
                threshold_pipe_q <= threshold;
            end
        end
    end

endmodule
