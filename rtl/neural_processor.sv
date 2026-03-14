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

    // State widths
    localparam POPCOUNT_W = $clog2(N) + 2;

    // Internal signals
    logic [N-1:0] xnor_result;
    logic [POPCOUNT_W-1:0] popcount;
    logic [ACC_WIDTH-1:0] acc;

    // State encoding (Yosys 0.9 friendly)
    localparam IDLE = 1'b0;
    localparam CALC = 1'b1;

    logic state, next_state;
    logic [ACC_WIDTH-1:0] final_acc;  // Moved here for Yosys 0.9

    // popcount_out assigned below (uses popcount_extended/final_acc)

    // Stage 1 & 2: Bitwise XNOR and Popcount
    always_comb begin
        xnor_result = ~(x ^ w);
        popcount = $countones(xnor_result);
    end

    // FSM State Register
    always_ff @(posedge clk or posedge rst) begin
        if (rst) state <= IDLE;
        else state <= next_state;
    end

    // FSM Next State Logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (valid_in && !last) next_state = CALC;
            end
            CALC: begin
                if (valid_in && last) next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Combinatorial calculation of potential final accumulator value
    // Extends popcount to ACC_WIDTH before adding.
    // Manual extension for Yosys 0.9 compatibility: {ZeroPadding, popcount}
    // popcount width is $clog2(N)+2. For N=8, width is 5. ACC_WIDTH=16. Padding=11.
    // General way:
    wire [ACC_WIDTH-1:0] popcount_extended;
    assign popcount_extended = {{(ACC_WIDTH - POPCOUNT_W) {1'b0}}, popcount};
    assign final_acc = acc + popcount_extended;
    assign popcount_out = acc;

    // Data Path & Control
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            acc       <= '0;
            y         <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            case (state)
                IDLE: begin
                    if (valid_in) begin
                        acc <= popcount_extended;
                        if (last) begin
                            y <= (popcount_extended >= threshold);
                            valid_out <= 1'b1;
                        end
                    end
                end

                CALC: begin
                    if (valid_in) begin
                        acc <= final_acc;
                        if (last) begin
                            y <= (final_acc >= threshold);
                            valid_out <= 1'b1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
