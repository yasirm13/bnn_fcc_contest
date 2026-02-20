`timescale 1ns / 1ps

module tb_neural_processor;

    // Compile-time maxima (runtime sizes are randomized per test).
    localparam MAX_WORD_W = 256;
    localparam N_MAX = 32;
    localparam ACC_WIDTH = 16;
    localparam CLK_PERIOD = 10;

    // Signals
    logic                 clk;
    logic                 rst;
    logic                 valid_in;
    logic                 last;
    logic [    N_MAX-1:0] x;
    logic [    N_MAX-1:0] w;
    logic [ACC_WIDTH-1:0] threshold;
    logic                 y;
    logic                 valid_out;
    logic [ACC_WIDTH-1:0] popcount_out;  // added

    // DUT
    neural_processor #(
        .N        (N_MAX),
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
        .popcount_out(popcount_out)  // changed
    );

    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // Verification Functions
    function int count_ones_var(input logic [MAX_WORD_W-1:0] val, input int width);
        int count;
        count = 0;
        for (int i = 0; i < width; i++) begin
            if (val[i]) count++;
        end
        return count;
    endfunction

    function int calc_xnor_popcount_var(input logic [MAX_WORD_W-1:0] a, input logic [MAX_WORD_W-1:0] b,
                                        input int width);
        logic [MAX_WORD_W-1:0] nxor;
        nxor = ~(a ^ b);
        return count_ones_var(nxor, width);
    endfunction

    task automatic fill_random_bits(ref logic [MAX_WORD_W-1:0] val);
        if ((MAX_WORD_W % 32) != 0) begin
            $error("MAX_WORD_W (%0d) must be a multiple of 32 for fill_random_bits.", MAX_WORD_W);
            $finish;
        end
        for (int bit_idx = 0; bit_idx < MAX_WORD_W; bit_idx += 32) begin
            val[bit_idx+:32] = $urandom();
        end
    endtask

    // Test Sequence
    initial begin
        int num_tests;
        int unsigned seed;

        // Init
        rst = 1;
        valid_in = 0;
        last = 0;
        x = 0;
        w = 0;
        threshold = 0;


        // Reset
        #(CLK_PERIOD * 2);
        rst = 0;
        #(CLK_PERIOD * 2);

        if (!$value$plusargs("tests=%d", num_tests)) num_tests = 10;
        if (!$value$plusargs("seed=%d", seed)) seed = 32'h1;
        void'($urandom(seed));

        $display("Starting Test (MAX_WORD_W=%0d, N_MAX=%0d, tests=%0d, seed=%0d)...", MAX_WORD_W, N_MAX,
                 num_tests, seed);

        for (int t = 0; t < num_tests; t++) begin
            $display("\nTest %0d/%0d", t, num_tests - 1);
            test_random_word_chunked();
        end

        $display("\nAll Tests PASSED");
        $finish;
    end

    task automatic test_random_word_chunked;
        logic [MAX_WORD_W-1:0] x_word;
        logic [MAX_WORD_W-1:0] w_word;
        int word_w;
        int chunk_w;
        int num_chunks;
        logic [ACC_WIDTH-1:0] threshold_test;
        int expected_total;
        int observed_total;
        int expected_acc;
        bit expected_y;
        bit expected_y_hold;
        bit saw_valid_on_last;
        int max_gap_cycles;
        int tail_garbage_cycles;

        // Randomize chunk width and word width (word_w divisible by chunk_w).
        chunk_w = $urandom_range(1, N_MAX);
        num_chunks = $urandom_range(1, MAX_WORD_W / chunk_w);
        word_w = num_chunks * chunk_w;

        fill_random_bits(x_word);
        fill_random_bits(w_word);
        expected_total = calc_xnor_popcount_var(x_word, w_word, word_w);

        threshold_test = ACC_WIDTH'($urandom_range(0, word_w + 16));
        expected_y = (expected_total >= int'(threshold_test));

        observed_total = -1;
        saw_valid_on_last = 0;
        // Model the DUT's accumulator across gaps; it only changes on valid_in=1 cycles.
        expected_acc = int'(popcount_out);
        expected_y_hold = y;
        max_gap_cycles = 3;
        tail_garbage_cycles = $urandom_range(0, 3);

        $display("Config: word_w=%0d chunk_w=%0d num_chunks=%0d", word_w, chunk_w, num_chunks);
        $display("Threshold=%0d expected_y=%0d", int'(threshold_test), expected_y);
        $display("Word X=%h, W=%h (note: only lowest word_w bits are used), expected_xnor_popcount=%0d",
                 x_word, w_word, expected_total);

        // Drive num_chunks chunks back-to-back, asserting last on final chunk.
        for (int chunk = 0; chunk < num_chunks; chunk++) begin
            int expected_chunk;
            int gap_cycles;
            logic [N_MAX-1:0] x_packed;
            logic [N_MAX-1:0] w_active;
            logic [N_MAX-1:0] w_packed;
            logic in_last;

            // Randomly insert gaps where valid_in=0 and x/w/last are garbage.
            gap_cycles = $urandom_range(0, max_gap_cycles);
            for (int g = 0; g < gap_cycles; g++) begin
                @(negedge clk);
                valid_in = 1'b0;
                x = $urandom();
                w = $urandom();
                last = ($urandom_range(0, 1) != 0);
                threshold = $urandom();

                @(posedge clk);
                #1ps;
                $display(
                    "  gap=%0d/%0d | valid_in=%b last=%b | x_garbage=%h w_garbage=%h thr=%0d | popcount_out=%0d valid_out=%b y=%b (expect hold pop=%0d y=%0d)",
                    g, gap_cycles - 1, valid_in, last, x, w, int'(threshold), popcount_out, valid_out, y,
                    expected_acc, expected_y_hold);

                if (valid_out) $error("valid_out asserted during valid_in=0 gap.");
                if (int'(popcount_out) !== expected_acc) begin
                    $error("Accumulator changed during gap: expected_hold=%0d observed=%0d", expected_acc,
                           int'(popcount_out));
                end
                if (y !== expected_y_hold) begin
                    $error("y changed during gap: expected_hold=%0d observed=%0d", expected_y_hold, y);
                end
            end

            expected_chunk = 0;
            for (int b = 0; b < chunk_w; b++) begin
                bit match;
                match = (x_word[chunk*chunk_w+b] == w_word[chunk*chunk_w+b]);
                expected_chunk += int'(match);
            end

            if (chunk == 0) expected_acc = expected_chunk;
            else expected_acc += expected_chunk;
            in_last  = (chunk == (num_chunks - 1));

            // Pack the active chunk into the LSBs; force unused bits to contribute 0 via XNOR.
            // X unused bits = 0, W unused bits = 1 => XNOR=0.
            x_packed = '0;
            w_active = '0;
            w_packed = '1;
            for (int b = 0; b < chunk_w; b++) begin
                x_packed[b] = x_word[chunk*chunk_w+b];
                w_active[b] = w_word[chunk*chunk_w+b];
                w_packed[b] = w_active[b];
            end

            @(negedge clk);
            valid_in = 1'b1;
            x = x_packed;
            w = w_packed;
            last = in_last;
            threshold = threshold_test;

            @(posedge clk);
            #1ps;

            $display(
                "  data=%0d/%0d valid_in=%b last=%b chunk_w=%0d | x_chunk=%h w_chunk=%h | thr=%0d exp_y=%0d | popcount_out=%0d valid_out=%b y=%b | expected_acc=%0d",
                chunk, num_chunks - 1, valid_in, in_last, chunk_w, x_packed, w_active, int'(threshold_test),
                expected_y, popcount_out, valid_out, y, expected_acc);

            if (valid_out) begin
                if (!in_last) $error("valid_out asserted early (chunk=%0d).", chunk);
                saw_valid_on_last = 1;
                observed_total = int'(popcount_out);
                if (y !== expected_y) begin
                    $error("y mismatch on valid_out: expected=%0d observed=%0d (thr=%0d pop=%0d)",
                           expected_y, y, int'(threshold_test), int'(popcount_out));
                end
                expected_y_hold = expected_y;
            end else begin
                if (in_last) $error("valid_out not asserted on last data chunk.");
                if (y !== expected_y_hold) begin
                    $error("y changed before valid_out: expected_hold=%0d observed=%0d", expected_y_hold, y);
                end
            end

            if (int'(popcount_out) !== expected_acc) begin
                $error("Running popcount mismatch at chunk=%0d: expected_acc=%0d observed=%0d", chunk,
                       expected_acc, int'(popcount_out));
            end
        end

        // After completion, ensure we can spam garbage with valid_in=0 without changing outputs.
        for (int tg = 0; tg < tail_garbage_cycles; tg++) begin
            @(negedge clk);
            valid_in = 1'b0;
            x = $urandom();
            w = $urandom();
            last = ($urandom_range(0, 1) != 0);
            threshold = $urandom();

            @(posedge clk);
            #1ps;
            $display(
                "  tail_garbage=%0d/%0d | valid_in=%b last=%b | x_garbage=%h w_garbage=%h thr=%0d | popcount_out=%0d valid_out=%b y=%b (expect hold pop=%0d y=%0d)",
                tg, tail_garbage_cycles - 1, valid_in, last, x, w, int'(threshold), popcount_out, valid_out,
                y, expected_acc, expected_y_hold);

            if (valid_out) $error("valid_out asserted during tail valid_in=0 garbage.");
            if (int'(popcount_out) !== expected_acc) begin
                $error("Accumulator changed during tail garbage: expected_hold=%0d observed=%0d",
                       expected_acc, int'(popcount_out));
            end
            if (y !== expected_y_hold) begin
                $error("y changed during tail garbage: expected_hold=%0d observed=%0d", expected_y_hold, y);
            end
        end

        @(negedge clk);
        valid_in = 1'b0;
        last = 1'b0;
        x = '0;
        w = '0;

        if (!saw_valid_on_last) begin
            $error("valid_out never asserted on final chunk.");
        end

        if (observed_total !== expected_total) begin
            $error("Popcount mismatch! expected=%0d observed=%0d", expected_total, observed_total);
        end else begin
            $display("  Popcount match: expected=%0d observed=%0d", expected_total, observed_total);
        end
    endtask

endmodule
