// -----------------------------------------------------------------------------
// Chunk-level neuron primitive.
//
// Inputs are provided as a stream of fixed-width chunks:
// - Each `valid_in` beat provides one chunk of activation bits `x` and weights `w`.
// - `last` marks the final chunk for the neuron.
//
// For each chunk:
//   XNOR(x, w) -> popcount(chunk) -> accumulate across chunks
//
// On the cycle after receiving the final chunk (`last=1`):
// - `popcount_out` emits the full-neuron population count
// - `y` emits (popcount >= threshold)
// - `valid_out` pulses for 1 cycle
//
// Notes:
// - Popcount is split into 4 segments to shorten the combinational adder chain.
// - ACC_WIDTH should be wide enough for the maximum sum across all chunks.
// -----------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Popcount sizing
    // -------------------------------------------------------------------------
    localparam int POPCOUNT_W = clog2_safe(N + 1);
    localparam int POPCOUNT_SEGMENTS = 4;
    localparam int SEG_WIDTH = N / POPCOUNT_SEGMENTS;
    localparam int SEG_COUNT_W = clog2_safe(SEG_WIDTH + 1);

    initial begin
        if (N % POPCOUNT_SEGMENTS)
            $fatal(1, "neural_processor requires N divisible by %0d (got %0d)", POPCOUNT_SEGMENTS, N);
    end

    // -------------------------------------------------------------------------
    // Stage 0: XNOR + per-segment popcount (combinational)
    // Stage 1: register segment counts + last/threshold
    // Stage 2: sum segments -> popcount_q, accumulate across chunks
    // Stage 3: register final sum and emit output pulse
    // -------------------------------------------------------------------------
    logic [N-1:0] xnor_result;
    logic [POPCOUNT_SEGMENTS-1:0][SEG_COUNT_W-1:0] seg_popcounts;
    logic [POPCOUNT_SEGMENTS-1:0][SEG_COUNT_W-1:0] seg_popcounts_q;
    logic seg_valid_q;
    logic seg_last_q;
    logic [31:0] seg_threshold_q;
    logic [POPCOUNT_W-1:0] popcount_pipe;
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
        for (int seg = 0; seg < POPCOUNT_SEGMENTS; seg++) begin
            seg_popcounts[seg] = SEG_COUNT_W'($countones(xnor_result[seg*SEG_WIDTH +: SEG_WIDTH]));
        end
    end

    always_comb begin
        popcount_pipe = '0;
        for (int seg = 0; seg < POPCOUNT_SEGMENTS; seg++) begin
            popcount_pipe = popcount_pipe + POPCOUNT_W'(seg_popcounts_q[seg]);
        end
    end

    assign popcount_extended_q = ACC_WIDTH'(popcount_q);
    assign acc_next_q = acc + popcount_extended_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            acc <= '0;
            y <= 1'b0;
            valid_out <= 1'b0;
            popcount_out <= '0;
            seg_valid_q <= 1'b0;
            seg_last_q <= 1'b0;
            seg_threshold_q <= '0;
            for (int seg = 0; seg < POPCOUNT_SEGMENTS; seg++) begin
                seg_popcounts_q[seg] <= '0;
            end
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

            pop_valid_q <= seg_valid_q;
            if (seg_valid_q) begin
                pop_last_q <= seg_last_q;
                popcount_q <= popcount_pipe;
                threshold_q <= seg_threshold_q;
            end

            seg_valid_q <= valid_in;
            if (valid_in) begin
                seg_last_q <= last;
                seg_threshold_q <= threshold;
                for (int seg = 0; seg < POPCOUNT_SEGMENTS; seg++) begin
                    seg_popcounts_q[seg] <= seg_popcounts[seg];
                end
            end
        end
    end

endmodule
