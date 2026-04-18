// Unit testbench for compute_layer single-chunk operation.
// Runs both hidden-layer and output-layer modes; checks mem_layer_start,
// output formatting, and that no extra memory reads are requested in the single-chunk case.
`timescale 1ns / 1ps

module compute_layer_unit_tb;
    localparam int LAYER_INPUTS = 4;
    localparam int NUM_NEURONS = 2;
    localparam int CONFIG_BUS_WIDTH = 8;
    localparam int PARALLEL_NEURONS = 2;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    logic clk = 1'b0;
    logic rst;

    logic                           hidden_valid_in;
    logic                           hidden_ready_out;
    logic                           hidden_valid_out;
    logic                           hidden_ready_in;
    logic [NUM_NEURONS-1:0]         hidden_result_vector;
    logic [LAYER_INPUTS-1:0]        hidden_input_activations;
    logic                           hidden_mem_layer_start;
    logic                           hidden_mem_read_weight;
    logic                           hidden_mem_read_thresh;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] hidden_mem_weight_data;
    logic [PARALLEL_NEURONS-1:0][31:0]                 hidden_mem_thresh_data;

    logic                           output_valid_in;
    logic                           output_ready_out;
    logic                           output_valid_out;
    logic                           output_ready_in;
    logic [NUM_NEURONS*32-1:0]      output_result_vector;
    logic [LAYER_INPUTS-1:0]        output_input_activations;
    logic                           output_mem_layer_start;
    logic                           output_mem_read_weight;
    logic                           output_mem_read_thresh;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] output_mem_weight_data;
    logic [PARALLEL_NEURONS-1:0][31:0]                 output_mem_thresh_data;

    compute_layer #(
        .LAYER_INPUTS    (LAYER_INPUTS),
        .NUM_NEURONS     (NUM_NEURONS),
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PARALLEL_NEURONS(PARALLEL_NEURONS),
        .IS_OUTPUT_LAYER (1'b0)
    ) hidden_dut (
        .clk              (clk),
        .rst              (rst),
        .valid_in         (hidden_valid_in),
        .ready_out        (hidden_ready_out),
        .valid_out        (hidden_valid_out),
        .ready_in         (hidden_ready_in),
        .result_vector    (hidden_result_vector),
        .input_activations(hidden_input_activations),
        .mem_layer_start  (hidden_mem_layer_start),
        .mem_read_weight  (hidden_mem_read_weight),
        .mem_read_thresh  (hidden_mem_read_thresh),
        .mem_weight_data  (hidden_mem_weight_data),
        .mem_thresh_data  (hidden_mem_thresh_data)
    );

    compute_layer #(
        .LAYER_INPUTS    (LAYER_INPUTS),
        .NUM_NEURONS     (NUM_NEURONS),
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PARALLEL_NEURONS(PARALLEL_NEURONS),
        .IS_OUTPUT_LAYER (1'b1)
    ) output_dut (
        .clk              (clk),
        .rst              (rst),
        .valid_in         (output_valid_in),
        .ready_out        (output_ready_out),
        .valid_out        (output_valid_out),
        .ready_in         (output_ready_in),
        .result_vector    (output_result_vector),
        .input_activations(output_input_activations),
        .mem_layer_start  (output_mem_layer_start),
        .mem_read_weight  (output_mem_read_weight),
        .mem_read_thresh  (output_mem_read_thresh),
        .mem_weight_data  (output_mem_weight_data),
        .mem_thresh_data  (output_mem_thresh_data)
    );

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic int count_matches4(
        input logic [LAYER_INPUTS-1:0] act,
        input logic [LAYER_INPUTS-1:0] weight_bits
    );
        begin
            return $countones(~(act ^ weight_bits));
        end
    endfunction

    task automatic reset_dut();
        rst <= 1'b1;
        hidden_valid_in <= 1'b0;
        output_valid_in <= 1'b0;
        hidden_ready_in <= 1'b1;
        output_ready_in <= 1'b1;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic pulse_hidden_start();
        @(negedge clk);
        hidden_valid_in <= 1'b1;
        #1;
        if (hidden_mem_layer_start !== 1'b1) begin
            $fatal(1, "hidden compute_layer should assert mem_layer_start when valid_in arrives");
        end
        @(posedge clk);
        @(negedge clk);
        hidden_valid_in <= 1'b0;
    endtask

    task automatic pulse_output_start();
        @(negedge clk);
        output_valid_in <= 1'b1;
        #1;
        if (output_mem_layer_start !== 1'b1) begin
            $fatal(1, "output compute_layer should assert mem_layer_start when valid_in arrives");
        end
        @(posedge clk);
        @(negedge clk);
        output_valid_in <= 1'b0;
    endtask

    task automatic wait_hidden_result(input logic [NUM_NEURONS-1:0] expected_bits);
        int cycle_count;
        begin
            for (cycle_count = 0; cycle_count < 20; cycle_count++) begin
                @(posedge clk);
                if (hidden_valid_out) begin
                    if (hidden_result_vector !== expected_bits) begin
                        $fatal(1, "hidden result mismatch. got=%b exp=%b", hidden_result_vector, expected_bits);
                    end
                    if (hidden_mem_read_weight !== 1'b0 || hidden_mem_read_thresh !== 1'b0) begin
                        $fatal(1, "single-chunk hidden test should not request extra memory reads");
                    end
                    return;
                end
            end
            $fatal(1, "timed out waiting for hidden_valid_out");
        end
    endtask

    task automatic wait_output_result(
        input logic [31:0] expected_count0,
        input logic [31:0] expected_count1
    );
        int cycle_count;
        begin
            for (cycle_count = 0; cycle_count < 20; cycle_count++) begin
                @(posedge clk);
                if (output_valid_out) begin
                    if (output_result_vector[31:0] !== expected_count0) begin
                        $fatal(1, "output lane 0 popcount mismatch. got=%0d exp=%0d", output_result_vector[31:0], expected_count0);
                    end
                    if (output_result_vector[63:32] !== expected_count1) begin
                        $fatal(1, "output lane 1 popcount mismatch. got=%0d exp=%0d", output_result_vector[63:32], expected_count1);
                    end
                    if (output_mem_read_weight !== 1'b0 || output_mem_read_thresh !== 1'b0) begin
                        $fatal(1, "single-chunk output test should not request extra memory reads");
                    end
                    return;
                end
            end
            $fatal(1, "timed out waiting for output_valid_out");
        end
    endtask

    initial begin
        logic [31:0] hidden_popcount0;
        logic [31:0] hidden_popcount1;

        hidden_input_activations = 4'b1010;
        output_input_activations = 4'b1010;

        hidden_mem_weight_data[0] = 8'b0000_1010;
        hidden_mem_weight_data[1] = 8'b0000_0101;
        hidden_mem_thresh_data[0] = 32'd4;
        hidden_mem_thresh_data[1] = 32'd1;

        output_mem_weight_data[0] = 8'b0000_1010;
        output_mem_weight_data[1] = 8'b0000_0101;
        output_mem_thresh_data[0] = 32'd0;
        output_mem_thresh_data[1] = 32'd0;

        hidden_popcount0 = count_matches4(hidden_input_activations, hidden_mem_weight_data[0][LAYER_INPUTS-1:0]);
        hidden_popcount1 = count_matches4(hidden_input_activations, hidden_mem_weight_data[1][LAYER_INPUTS-1:0]);

        reset_dut();

        pulse_hidden_start();
        wait_hidden_result({(hidden_popcount1 >= hidden_mem_thresh_data[1]), (hidden_popcount0 >= hidden_mem_thresh_data[0])});

        pulse_output_start();
        wait_output_result(hidden_popcount0, hidden_popcount1);

        $display("SUCCESS: compute_layer_unit_tb completed all checks.");
        $finish;
    end
endmodule
