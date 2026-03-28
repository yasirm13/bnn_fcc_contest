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
    input  logic [ACC_WIDTH-1:0] threshold,
    output logic                 y,
    output logic                 valid_out,
    output logic [ACC_WIDTH-1:0] popcount_out
);

    localparam POPCOUNT_W = $clog2(N) + 2;

    logic [N-1:0] xnor_result;
    logic [POPCOUNT_W-1:0] popcount;
    logic [ACC_WIDTH-1:0] acc;
    logic accum_valid;
    logic pending_output;
    logic [ACC_WIDTH-1:0] threshold_q;
    wire [ACC_WIDTH-1:0] popcount_extended;
    logic [ACC_WIDTH-1:0] next_acc;

    always_comb begin
        xnor_result = ~(x ^ w);
        popcount = $countones(xnor_result);
    end

    assign popcount_extended = {{(ACC_WIDTH - POPCOUNT_W) {1'b0}}, popcount};
    assign next_acc = (accum_valid ? acc : '0) + popcount_extended;
    assign popcount_out = acc;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            acc       <= '0;
            y         <= 1'b0;
            valid_out <= 1'b0;
            accum_valid <= 1'b0;
            pending_output <= 1'b0;
            threshold_q <= '0;
        end else begin
            valid_out <= 1'b0;

            if (pending_output) begin
                y <= (acc >= threshold_q);
                valid_out <= 1'b1;
                pending_output <= 1'b0;
            end

            if (valid_in) begin
                acc <= next_acc;
                if (last) begin
                    threshold_q <= threshold;
                    accum_valid <= 1'b0;
                    pending_output <= 1'b1;
                end else begin
                    accum_valid <= 1'b1;
                end
            end
        end
    end

endmodule
