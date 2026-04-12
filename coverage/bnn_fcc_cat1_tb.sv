`timescale 1ns / 100ps

module bnn_fcc_cat1_tb #(
    // Testbench configuration (Using custom topology to test TKEEP edge cases)
    parameter int      USE_CUSTOM_TOPOLOGY                      = 1'b1,
    parameter int      CUSTOM_LAYERS                            = 4,
    parameter int      CUSTOM_TOPOLOGY          [CUSTOM_LAYERS] = '{13, 10, 10, 5}, // 13 inputs forces TKEEP edge cases on data stream
    parameter int      NUM_TEST_IMAGES                          = 50,
    parameter bit      VERIFY_MODEL                             = 0, // No python verification for custom
    parameter string   BASE_DIR                                 = "../python",
    parameter bit      TOGGLE_DATA_OUT_READY                    = 1'b1,
    parameter realtime TIMEOUT                                  = 10ms,
    parameter realtime CLK_PERIOD                               = 10ns,
    parameter bit      DEBUG                                    = 1'b0,

    // Bus configuration
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int INPUT_BUS_WIDTH  = 64,
    parameter int OUTPUT_BUS_WIDTH = 8,

    // App configuration
    parameter  int INPUT_DATA_WIDTH  = 8,
    localparam int INPUTS_PER_CYCLE  = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH,
    localparam int BYTES_PER_INPUT   = INPUT_DATA_WIDTH / 8,
    parameter  int OUTPUT_DATA_WIDTH = 4,

    // DUT configuration        
    localparam int NON_INPUT_LAYERS = CUSTOM_LAYERS - 1,
    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[NON_INPUT_LAYERS] = '{8, 8, 10}
);
    import bnn_fcc_tb_pkg::*;
    import bnn_fcc_cat1_pkg::*;

    localparam int ACTUAL_TOTAL_LAYERS = CUSTOM_LAYERS;
    localparam int ACTUAL_TOPOLOGY[ACTUAL_TOTAL_LAYERS] = CUSTOM_TOPOLOGY;
    localparam realtime HALF_CLK_PERIOD = CLK_PERIOD / 2.0;

    initial begin
        assert (INPUT_DATA_WIDTH == 8)
        else $fatal(1, "TB ERROR: INPUT_DATA_WIDTH must be 8");
    end

    function automatic bit chance(real p);
        if (p > 1.0 || p < 0.0) $fatal(1, "Invalid probability in chance()");
        return ($urandom < (p * (2.0 ** 32)));
    endfunction
    
    // Pattern parameters that change over time to simulate bursts, intermittent, continuous
    real current_config_prob = 1.0;
    real current_data_prob = 1.0;
    
    task update_probabilities();
        // 0=continuous, 1=intermittent, 2=burst
        int mode;
        mode = $urandom_range(0, 2);
        if (mode == 0) begin
            current_config_prob = 1.0;
            current_data_prob = 1.0;
        end else if (mode == 1) begin
            current_config_prob = 0.5;
            current_data_prob = 0.5;
        end else begin
            // Burst: high probability for a bit
            current_config_prob = 0.9;
            current_data_prob = 0.9;
        end
    endtask

    Extended_BNN_FCC_Model #(CONFIG_BUS_WIDTH) model;
    BNN_FCC_Stimulus #(INPUT_DATA_WIDTH) stim;
    LatencyTracker latency;
    ThroughputTracker throughput;

    typedef bit [CONFIG_BUS_WIDTH-1:0] config_bus_word_t;
    typedef config_bus_word_t config_bus_data_stream_t[];

    localparam CONFIG_KEEP_WIDTH = CONFIG_BUS_WIDTH / 8;
    typedef bit [CONFIG_KEEP_WIDTH-1:0] config_bus_keep_t;
    typedef config_bus_keep_t config_keep_stream_t[];

    bit [CONFIG_BUS_WIDTH-1:0] config_bus_data_stream[];
    bit [CONFIG_BUS_WIDTH/8-1:0] config_bus_keep_stream[];

    int num_tests;
    int passed;
    int failed;

    logic clk = 1'b0;
    logic rst;

    axi4_stream_if #(.DATA_WIDTH(CONFIG_BUS_WIDTH)) config_in (.aclk(clk), .aresetn(!rst));
    axi4_stream_if #(.DATA_WIDTH(INPUT_BUS_WIDTH)) data_in (.aclk(clk), .aresetn(!rst));
    axi4_stream_if #(.DATA_WIDTH(OUTPUT_BUS_WIDTH)) data_out (.aclk(clk), .aresetn(!rst));

    bnn_fcc #(
        .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
        .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
        .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
        .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
        .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
        .TOTAL_LAYERS     (ACTUAL_TOTAL_LAYERS),
        .TOPOLOGY         (ACTUAL_TOPOLOGY),
        .PARALLEL_INPUTS  (PARALLEL_INPUTS),
        .PARALLEL_NEURONS (PARALLEL_NEURONS)
    ) DUT (
        .clk(clk),
        .rst(rst),
        .config_valid(config_in.tvalid),
        .config_ready(config_in.tready),
        .config_data (config_in.tdata),
        .config_keep (config_in.tkeep),
        .config_last (config_in.tlast),
        .data_in_valid(data_in.tvalid),
        .data_in_ready(data_in.tready),
        .data_in_data (data_in.tdata),
        .data_in_keep (data_in.tkeep),
        .data_in_last (data_in.tlast),
        .data_out_valid(data_out.tvalid),
        .data_out_ready(data_out.tready),
        .data_out_data (data_out.tdata),
        .data_out_keep (data_out.tkeep),
        .data_out_last (data_out.tlast)
    );

    initial begin : generate_clock
        forever #HALF_CLK_PERIOD clk <= ~clk;
    end

    initial begin : l_init_model
        model = new();
        stim = new(ACTUAL_TOPOLOGY[0]);
        latency = new(CLK_PERIOD);
        throughput = new(CLK_PERIOD);

        $display("--- Loading Randomized Model (Category 1) ---");
        model.create_random(ACTUAL_TOPOLOGY);
        // USE ROMDOMIZED CONFIGURATION STREAM
        model.encode_configuration_randomized(config_bus_data_stream, config_bus_keep_stream);
        $display("--- Configuration created: %0d words (%0d-bit wide) ---", config_bus_data_stream.size(), CONFIG_BUS_WIDTH);

        $display("--- Generating Random Test Vectors (Category 1) ---");
        stim.generate_random_vectors(NUM_TEST_IMAGES);

        num_tests = stim.get_num_vectors();
        model.print_summary();

        if (DEBUG) model.print_model();
    end

    logic [OUTPUT_DATA_WIDTH-1:0] expected_outputs[$];

    assign config_in.tstrb = config_in.tkeep;
    assign data_in.tstrb   = data_in.tkeep;

    initial begin : l_sequencer_and_driver
        $timeformat(-9, 0, " ns", 0);

        rst              <= 1'b1;
        config_in.tvalid <= 1'b0;
        config_in.tdata  <= '0;
        config_in.tkeep  <= '0;
        config_in.tlast  <= 1'b0;
        data_in.tvalid   <= 1'b0;
        data_in.tdata    <= '0;
        data_in.tkeep    <= '0;
        data_in.tlast    <= 1'b0;

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst <= 1'b0;
        repeat (5) @(posedge clk);

        $display("[%0t] Streaming weights and thresholds (Category 1).", $realtime);
        
        for (int i = 0; i < config_bus_data_stream.size(); i++) begin
            if (i % 8 == 0) update_probabilities(); // Update patterns regularly
            
            while (!chance(current_config_prob)) begin
                config_in.tvalid <= 1'b0;
                @(posedge clk iff config_in.tready);
            end

            config_in.tvalid <= 1'b1;
            config_in.tdata  <= config_bus_data_stream[i];
            config_in.tlast  <= i == config_bus_data_stream.size() - 1;
            config_in.tkeep  <= config_bus_keep_stream[i];
            @(posedge clk iff config_in.tready);
        end
        config_in.tvalid <= 1'b0;
        config_in.tlast  <= 1'b0;

        wait (data_in.tready);
        repeat (5) @(posedge clk);

        for (int i = 0; i < num_tests; i++) begin
            int expected_pred;
            bit [INPUT_DATA_WIDTH-1:0] current_img[];
            int image_delay;

            stim.get_vector(i, current_img);
            expected_pred = model.compute_reference(current_img);
            expected_outputs.push_back(expected_pred);

            $display("[%0t] Streaming image %0d (Category 1).", $realtime, i);

            for (int j = 0; j < current_img.size(); j += INPUTS_PER_CYCLE) begin
                if (j % 16 == 0) update_probabilities(); // Change duty cycle during image
                
                for (int k = 0; k < INPUTS_PER_CYCLE; k++) begin
                    if (j + k < current_img.size()) begin
                        data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= current_img[j+k];
                        data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '1;
                    end else begin
                        data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= '0;
                        data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '0;
                    end
                end

                while (!chance(current_data_prob)) begin
                    data_in.tvalid <= 1'b0;
                    @(posedge clk iff data_in.tready);
                end
                data_in.tvalid <= 1'b1;
                data_in.tlast  <= (j + INPUTS_PER_CYCLE >= current_img.size());
                @(posedge clk iff data_in.tready);

                if (i == 0 && j == 0) throughput.start_test();
                if (j == 0) latency.start_event(i);
            end

            data_in.tvalid <= 1'b0;
            data_in.tlast  <= 1'b0;
            data_in.tkeep  <= '0;
            
            // Random delay between consecutive images
            image_delay = $urandom_range(0, 10);
            repeat (image_delay) @(posedge clk);
        end

        $display("[%0t] All images loaded, waiting for outputs.", $realtime);
        wait (expected_outputs.size() == 0);
        repeat (5) @(posedge clk);

        disable generate_clock;
        disable l_timeout;
        if (passed == num_tests) $display("[%0t] SUCCESS: all %0d tests completed successfully.", $realtime, num_tests);
        else $error("FAILED: %0d out of %0d tests failed.", failed, num_tests);
    end

    initial begin : l_toggle_ready
        int throttle_len;
        data_out.tready <= 1'b1;
        @(posedge clk iff !rst);
        if (TOGGLE_DATA_OUT_READY) begin
            forever begin
                // 0=continuous readiness, 1=intermittent readiness, 2=backpressure long burst
                int mode;
                mode = $urandom_range(0, 2);
                if (mode == 0) begin
                    throttle_len = $urandom_range(1, 10);
                    repeat(throttle_len) begin
                        data_out.tready <= 1'b1;
                        @(posedge clk);
                    end
                end else if (mode == 1) begin
                    throttle_len = $urandom_range(5, 20);
                    repeat(throttle_len) begin
                        data_out.tready <= chance(0.5); // intermittent
                        @(posedge clk);
                    end
                end else begin
                    // apply backpressure
                    data_out.tready <= 1'b0;
                    throttle_len = $urandom_range(5, 50); // long backpressure
                    repeat(throttle_len) @(posedge clk);
                end
            end
        end else data_out.tready <= 1'b1;
    end

    initial begin : l_output_monitor
        automatic int output_count = 0;
        forever begin
            @(posedge clk iff data_out.tvalid && data_out.tready);
            assert (expected_outputs.size() > 0)
            else $fatal(1, "No expected output for actual output");
            assert (data_out.tdata == expected_outputs[0]) begin
                passed++;
            end else begin
                $error("Output incorrect for image %0d: actual = %0d vs expected = %0d", output_count, data_out.tdata, expected_outputs[0]);
                failed++;
            end
            void'(expected_outputs.pop_front());
            latency.end_event(output_count);
            if (output_count == NUM_TEST_IMAGES - 1) throughput.sample_end();
            output_count++;
        end
    end

    initial begin : l_timeout
        #TIMEOUT;
        $fatal(1, $sformatf("Simulation failed due to timeout of %0t.", TIMEOUT));
    end

endmodule
