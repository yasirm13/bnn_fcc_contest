// Unit testbench for compute_layer multi-chunk operation (CHUNKS_PER_NEURON=5).
// Uses a simple pipelined memory model; checks mem_read_weight activity, padding-bit don't-cares,
// X-free outputs, and valid_out/result stability under backpressure.
`timescale 1ns / 1ps

module compute_layer_multichunk_unit_tb;
    // Choose a size that forces multi-chunk operation (CHUNKS_PER_NEURON > 4),
    // matching the controller path used in the full design.
    localparam int LAYER_INPUTS = 37;
    localparam int NUM_NEURONS = 3;
    localparam int CONFIG_BUS_WIDTH = 8;
    localparam int PARALLEL_NEURONS = 2;
    localparam realtime HALF_CLK_PERIOD = 5ns;

    // Derived constants for this testbench.
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int BITS_PER_NEURON = BYTES_PER_NEURON * 8;
    localparam int CHUNKS_PER_NEURON = (BITS_PER_NEURON + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH; // should be 2

    logic clk = 1'b0;
    logic rst;

    // Hidden-layer instance (binarized outputs)
    logic                      h_valid_in;
    logic                      h_ready_out;
    logic                      h_valid_out;
    logic                      h_ready_in;
    logic [NUM_NEURONS-1:0]    h_result_vector;
    logic [LAYER_INPUTS-1:0]   h_input_activations;
    logic                      h_mem_layer_start;
    logic                      h_mem_read_weight;
    logic                      h_mem_read_thresh;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] h_mem_weight_data;
    logic [PARALLEL_NEURONS-1:0][31:0]                 h_mem_thresh_data;

    // Output-layer instance (raw popcounts)
    logic                      o_valid_in;
    logic                      o_ready_out;
    logic                      o_valid_out;
    logic                      o_ready_in;
    logic [NUM_NEURONS*32-1:0] o_result_vector;
    logic [LAYER_INPUTS-1:0]   o_input_activations;
    logic                      o_mem_layer_start;
    logic                      o_mem_read_weight;
    logic                      o_mem_read_thresh;
    logic [PARALLEL_NEURONS-1:0][CONFIG_BUS_WIDTH-1:0] o_mem_weight_data;
    logic [PARALLEL_NEURONS-1:0][31:0]                 o_mem_thresh_data;

    compute_layer #(
        .LAYER_INPUTS    (LAYER_INPUTS),
        .NUM_NEURONS     (NUM_NEURONS),
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PARALLEL_NEURONS(PARALLEL_NEURONS),
        .IS_OUTPUT_LAYER (1'b0)
    ) hidden_dut (
        .clk              (clk),
        .rst              (rst),
        .valid_in         (h_valid_in),
        .ready_out        (h_ready_out),
        .valid_out        (h_valid_out),
        .ready_in         (h_ready_in),
        .result_vector    (h_result_vector),
        .input_activations(h_input_activations),
        .mem_layer_start  (h_mem_layer_start),
        .mem_read_weight  (h_mem_read_weight),
        .mem_read_thresh  (h_mem_read_thresh),
        .mem_weight_data  (h_mem_weight_data),
        .mem_thresh_data  (h_mem_thresh_data)
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
        .valid_in         (o_valid_in),
        .ready_out        (o_ready_out),
        .valid_out        (o_valid_out),
        .ready_in         (o_ready_in),
        .result_vector    (o_result_vector),
        .input_activations(o_input_activations),
        .mem_layer_start  (o_mem_layer_start),
        .mem_read_weight  (o_mem_read_weight),
        .mem_read_thresh  (o_mem_read_thresh),
        .mem_weight_data  (o_mem_weight_data),
        .mem_thresh_data  (o_mem_thresh_data)
    );

    always #HALF_CLK_PERIOD clk = ~clk;

    function automatic int count_matches13(
        input logic [LAYER_INPUTS-1:0] act,
        input logic [LAYER_INPUTS-1:0] weight_bits
    );
        return $countones(~(act ^ weight_bits));
    endfunction

    function automatic logic [CONFIG_BUS_WIDTH-1:0] pack_chunk(
        input logic [LAYER_INPUTS-1:0] bits,
        input int chunk_idx
    );
        logic [CONFIG_BUS_WIDTH-1:0] out_word;
        int lo;
        begin
            lo = chunk_idx * CONFIG_BUS_WIDTH;
            out_word = '0;
            for (int k = 0; k < CONFIG_BUS_WIDTH; k++) begin
                if ((lo + k) < LAYER_INPUTS)
                    out_word[k] = bits[lo + k];
                else
                    out_word[k] = 1'b0;
            end
            return out_word;
        end
    endfunction

    // Simple memory model with registered/pipelined output latency (roughly
    // matches layer_memory timing so chunk/mask alignment matches compute_layer).
    typedef struct packed {
        logic [31:0] thr;
        logic [CHUNKS_PER_NEURON-1:0][CONFIG_BUS_WIDTH-1:0] w_chunks;
    } neuron_model_t;

    neuron_model_t neuron_table[NUM_NEURONS];

    localparam int MEM_LATENCY_CYCLES = 4;

    logic [$clog2(NUM_NEURONS+1)-1:0] h_base_q;
    logic [$clog2(CHUNKS_PER_NEURON+1)-1:0] h_chunk_issue_q;
    logic [$clog2(NUM_NEURONS+1)-1:0] h_base_pipe0_q, h_base_pipe1_q, h_base_pipe2_q, h_base_pipe3_q;
    logic [$clog2(CHUNKS_PER_NEURON+1)-1:0] h_chunk_pipe0_q, h_chunk_pipe1_q, h_chunk_pipe2_q, h_chunk_pipe3_q;

    logic [$clog2(NUM_NEURONS+1)-1:0] o_base_q;
    logic [$clog2(CHUNKS_PER_NEURON+1)-1:0] o_chunk_issue_q;
    logic [$clog2(NUM_NEURONS+1)-1:0] o_base_pipe0_q, o_base_pipe1_q, o_base_pipe2_q, o_base_pipe3_q;
    logic [$clog2(CHUNKS_PER_NEURON+1)-1:0] o_chunk_pipe0_q, o_chunk_pipe1_q, o_chunk_pipe2_q, o_chunk_pipe3_q;
    int unsigned h_read_weight_pulses;
    int unsigned o_read_weight_pulses;

    task automatic mem_model_reset();
        h_base_q = '0;
        h_chunk_issue_q = '0;
        o_base_q = '0;
        o_chunk_issue_q = '0;
        h_base_pipe0_q = '0;
        h_base_pipe1_q = '0;
        h_base_pipe2_q = '0;
        h_base_pipe3_q = '0;
        h_chunk_pipe0_q = '0;
        h_chunk_pipe1_q = '0;
        h_chunk_pipe2_q = '0;
        h_chunk_pipe3_q = '0;
        o_base_pipe0_q = '0;
        o_base_pipe1_q = '0;
        o_base_pipe2_q = '0;
        o_base_pipe3_q = '0;
        o_chunk_pipe0_q = '0;
        o_chunk_pipe1_q = '0;
        o_chunk_pipe2_q = '0;
        o_chunk_pipe3_q = '0;
        h_read_weight_pulses = 0;
        o_read_weight_pulses = 0;
    endtask

    always_ff @(posedge clk) begin
        if (rst) begin
            h_base_q <= '0;
            h_chunk_issue_q <= '0;
            o_base_q <= '0;
            o_chunk_issue_q <= '0;
            h_base_pipe0_q <= '0;
            h_base_pipe1_q <= '0;
            h_base_pipe2_q <= '0;
            h_base_pipe3_q <= '0;
            h_chunk_pipe0_q <= '0;
            h_chunk_pipe1_q <= '0;
            h_chunk_pipe2_q <= '0;
            h_chunk_pipe3_q <= '0;
            o_base_pipe0_q <= '0;
            o_base_pipe1_q <= '0;
            o_base_pipe2_q <= '0;
            o_base_pipe3_q <= '0;
            o_chunk_pipe0_q <= '0;
            o_chunk_pipe1_q <= '0;
            o_chunk_pipe2_q <= '0;
            o_chunk_pipe3_q <= '0;
            h_mem_weight_data <= '0;
            h_mem_thresh_data <= '0;
            o_mem_weight_data <= '0;
            o_mem_thresh_data <= '0;
            h_read_weight_pulses <= 0;
            o_read_weight_pulses <= 0;
        end else begin
            // Pipeline the issued indices to approximate memory latency.
            h_base_pipe3_q <= h_base_pipe2_q;
            h_base_pipe2_q <= h_base_pipe1_q;
            h_base_pipe1_q <= h_base_pipe0_q;
            h_base_pipe0_q <= h_base_q;
            h_chunk_pipe3_q <= h_chunk_pipe2_q;
            h_chunk_pipe2_q <= h_chunk_pipe1_q;
            h_chunk_pipe1_q <= h_chunk_pipe0_q;
            h_chunk_pipe0_q <= h_chunk_issue_q;

            o_base_pipe3_q <= o_base_pipe2_q;
            o_base_pipe2_q <= o_base_pipe1_q;
            o_base_pipe1_q <= o_base_pipe0_q;
            o_base_pipe0_q <= o_base_q;
            o_chunk_pipe3_q <= o_chunk_pipe2_q;
            o_chunk_pipe2_q <= o_chunk_pipe1_q;
            o_chunk_pipe1_q <= o_chunk_pipe0_q;
            o_chunk_pipe0_q <= o_chunk_issue_q;

            // Update issue indices for next cycle based on requests.
            if (h_mem_layer_start) begin
                h_base_q <= '0;
                h_chunk_issue_q <= '0;
            end else if (h_mem_read_thresh) begin
                h_base_q <= h_base_q + $unsigned(PARALLEL_NEURONS);
                h_chunk_issue_q <= '0;
            end else if (h_mem_read_weight) begin
                h_chunk_issue_q <= h_chunk_issue_q + 1'b1;
                h_read_weight_pulses <= h_read_weight_pulses + 1;
            end

            if (o_mem_layer_start) begin
                o_base_q <= '0;
                o_chunk_issue_q <= '0;
            end else if (o_mem_read_thresh) begin
                o_base_q <= o_base_q + $unsigned(PARALLEL_NEURONS);
                o_chunk_issue_q <= '0;
            end else if (o_mem_read_weight) begin
                o_chunk_issue_q <= o_chunk_issue_q + 1'b1;
                o_read_weight_pulses <= o_read_weight_pulses + 1;
            end

            // Drive registered memory outputs.
            for (int lane = 0; lane < PARALLEL_NEURONS; lane++) begin
                int n;
                int c;

                n = int'(h_base_pipe3_q) + lane;
                c = int'(h_chunk_pipe3_q);
                if (c >= CHUNKS_PER_NEURON)
                    c = CHUNKS_PER_NEURON - 1;
                if (n < NUM_NEURONS) begin
                    h_mem_weight_data[lane] <= neuron_table[n].w_chunks[c];
                    h_mem_thresh_data[lane] <= neuron_table[n].thr;
                end else begin
                    h_mem_weight_data[lane] <= '0;
                    h_mem_thresh_data[lane] <= '0;
                end

                n = int'(o_base_pipe3_q) + lane;
                c = int'(o_chunk_pipe3_q);
                if (c >= CHUNKS_PER_NEURON)
                    c = CHUNKS_PER_NEURON - 1;
                if (n < NUM_NEURONS) begin
                    o_mem_weight_data[lane] <= neuron_table[n].w_chunks[c];
                    o_mem_thresh_data[lane] <= neuron_table[n].thr;
                end else begin
                    o_mem_weight_data[lane] <= '0;
                    o_mem_thresh_data[lane] <= '0;
                end
            end
        end
    end

    task automatic reset_dut();
        rst <= 1'b1;
        h_valid_in <= 1'b0;
        o_valid_in <= 1'b0;
        h_ready_in <= 1'b1;
        o_ready_in <= 1'b1;
        repeat (2) @(posedge clk);
        rst <= 1'b0;
        @(posedge clk);
    endtask

    task automatic pulse_start_hidden();
        @(negedge clk);
        h_valid_in <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        h_valid_in <= 1'b0;
    endtask

    task automatic pulse_start_output();
        @(negedge clk);
        o_valid_in <= 1'b1;
        @(posedge clk);
        @(negedge clk);
        o_valid_in <= 1'b0;
    endtask

    task automatic wait_hidden_result(output logic [NUM_NEURONS-1:0] observed_bits);
        int cycles;
        for (cycles = 0; cycles < 200; cycles++) begin
            @(posedge clk);
            if (h_valid_out) begin
                observed_bits = h_result_vector;
                if (^observed_bits === 1'bx)
                    $fatal(1, "hidden result contains X: %b", observed_bits);
                return;
            end
        end
        $fatal(1, "timed out waiting for hidden_valid_out");
    endtask

    task automatic wait_output_result(
        output logic [31:0] obs0,
        output logic [31:0] obs1,
        output logic [31:0] obs2
    );
        int cycles;
        for (cycles = 0; cycles < 200; cycles++) begin
            @(posedge clk);
            if (o_valid_out) begin
                obs0 = o_result_vector[31:0];
                obs1 = o_result_vector[63:32];
                obs2 = o_result_vector[95:64];
                if (^obs0 === 1'bx || ^obs1 === 1'bx || ^obs2 === 1'bx)
                    $fatal(1, "output popcount contains X: %0d %0d %0d", obs0, obs1, obs2);
                return;
            end
        end
        $fatal(1, "timed out waiting for output_valid_out");
    endtask

    initial begin
        logic [LAYER_INPUTS-1:0] act;
        logic [LAYER_INPUTS-1:0] w0;
        logic [LAYER_INPUTS-1:0] w1;
        logic [LAYER_INPUTS-1:0] w2;
        int pc0, pc1, pc2;
        logic [NUM_NEURONS-1:0] exp_hidden_bits;
        logic [NUM_NEURONS-1:0] obs_hidden_bits;
        logic [31:0] obs_o0, obs_o1, obs_o2;

        if (CHUNKS_PER_NEURON != 5) begin
            $fatal(1, "This test expects CHUNKS_PER_NEURON=5, got %0d", CHUNKS_PER_NEURON);
        end

        // Activations: deterministic pattern across all bits.
        act = '0;
        for (int i = 0; i < LAYER_INPUTS; i++) begin
            act[i] = ((i % 3) == 0) ^ ((i % 5) == 0);
        end

        // Define per-neuron weights. Padded bits in the final chunk are outside
        // LAYER_INPUTS and must not affect results.
        w0 = act; // perfect match -> popcount LAYER_INPUTS
        w1 = '0;  // match count = #zeros in act
        w2 = '1;  // match count = #ones in act

        neuron_table[0].thr = 32'(LAYER_INPUTS);
        neuron_table[1].thr = 32'(LAYER_INPUTS / 2);
        neuron_table[2].thr = 32'(LAYER_INPUTS / 2 + 1);

        for (int c = 0; c < CHUNKS_PER_NEURON; c++) begin
            neuron_table[0].w_chunks[c] = pack_chunk(w0, c);
            neuron_table[1].w_chunks[c] = pack_chunk(w1, c);
            neuron_table[2].w_chunks[c] = pack_chunk(w2, c);
        end

        // Force padded bits in the last chunk to 0 for neuron2 to ensure padding logic still makes them don't-care.
        neuron_table[2].w_chunks[CHUNKS_PER_NEURON-1][7:5] = 3'b000;

        h_input_activations = act;
        o_input_activations = act;

        pc0 = count_matches13(act, w0);
        pc1 = count_matches13(act, w1);
        pc2 = count_matches13(act, w2);

        exp_hidden_bits[0] = (pc0 >= neuron_table[0].thr);
        exp_hidden_bits[1] = (pc1 >= neuron_table[1].thr);
        exp_hidden_bits[2] = (pc2 >= neuron_table[2].thr);

        reset_dut();
        mem_model_reset();

        pulse_start_output();
        wait_output_result(obs_o0, obs_o1, obs_o2);
        if (o_read_weight_pulses == 0) begin
            $fatal(1, "expected o_mem_read_weight to pulse for multi-chunk operation");
        end

        @(negedge clk);
        h_ready_in <= 1'b0;
        pulse_start_hidden();
        wait_hidden_result(obs_hidden_bits);
        if (h_read_weight_pulses == 0) begin
            $fatal(1, "expected h_mem_read_weight to pulse for multi-chunk operation");
        end

        // Backpressure behavior: valid_out should remain asserted until ready_in is high.
        repeat (3) @(posedge clk);
        if (!h_valid_out || (h_result_vector !== obs_hidden_bits))
            $fatal(1, "hidden output must remain stable under backpressure");
        @(negedge clk);
        h_ready_in <= 1'b1;
        @(posedge clk);
        #1;
        if (h_valid_out)
            $fatal(1, "hidden_valid_out should clear after ready_in");

        $display("SUCCESS: compute_layer_multichunk_unit_tb completed all checks.");
        $finish;
    end
endmodule
