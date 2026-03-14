module layer_memory #(
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int LAYER_INPUTS = 784,
    parameter int NUM_NEURONS = 256,
    parameter int PARALLEL_NEURONS = 1, // Only 1 supported for now in this simple implementation
    parameter int PARALLEL_INPUTS = 64
)(
    input logic clk,
    input logic rst,
    
    // Configuration Write Interface
    input logic wr_en_weights,
    input logic wr_en_thresholds,
    input logic [31:0] wr_addr,  // From config parser (0-based index)
    input logic [CONFIG_BUS_WIDTH-1:0] wr_data,
    
    // Compute Read Interface
    input logic        layer_start,      // Reset pointers for a new inference pass
    input logic        read_weight_chunk,// Advance weight pointer
    input logic        read_threshold,   // Advance threshold pointer
    
    output logic [CONFIG_BUS_WIDTH-1:0] rd_data_weights,   // Output full width, consumer handles slicing
    output logic [31:0]                 rd_data_threshold
);

    // Calculate depths
    // Weights: Total bits = NUM_NEURONS * (aligned LAYER_INPUTS)
    // Actually, config parser writes in CONFIG_BUS_WIDTH chunks.
    // Total chunks approx = (NUM_NEURONS * BITS_PER_NEURON) / CONFIG_BUS_WIDTH
    // Just use a sufficiently large memory or calculated depth.
    // Since we don't have dynamic sizing easily without ceils, let's use a safe upper bound or simple math.
    
    // BITS_PER_NEURON: Inputs padded to byte boundary.
    localparam int BITS_PER_NEURON_UNALIGNED = LAYER_INPUTS;
    localparam int BYTES_PER_NEURON = (LAYER_INPUTS + 7) / 8;
    localparam int BITS_PER_NEURON = BYTES_PER_NEURON * 8;
    
    localparam int TOTAL_WEIGHT_BITS = NUM_NEURONS * BITS_PER_NEURON;
    localparam int WEIGHT_MEM_DEPTH = (TOTAL_WEIGHT_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;
    
    // Thresholds: 1 per neuron (32 bits).
    // Config interface writes CONFIG_BUS_WIDTH bits.
    // If CONFIG > 32, we pack multiple thresholds? 
    // README: "For thresholds... payload provides a single 32-bit threshold for each neuron."
    // Does it pack them? "payload provides... single... for each".
    // Usually implies packing if bandwidth allows.
    // Use the generic: "Following the payload is a series of total_bytes bytes".
    // So yes, packed.
    localparam int TOTAL_THRESH_BITS = NUM_NEURONS * 32;
    localparam int THRESH_MEM_DEPTH = (TOTAL_THRESH_BITS + CONFIG_BUS_WIDTH - 1) / CONFIG_BUS_WIDTH;

    // Memory Arrays
    //
    // IMPORTANT: Weights are read in an unaligned fashion (can span two consecutive
    // CONFIG_BUS_WIDTH words). FPGA BRAMs require synchronous reads; combinational
    // reads will infer LUT/FF RAM.
    //
    // To keep reads BRAM-inferrable while supporting "two-word" fetches each cycle,
    // we duplicate the weight memory. This guarantees we can read `word_addr` and
    // `word_addr+1` in the same cycle using two independent 1R ports (at the cost
    // of 2x BRAM for weights).
    (* ram_style = "block" *) logic [CONFIG_BUS_WIDTH-1:0] mem_weights_lo [WEIGHT_MEM_DEPTH];
    (* ram_style = "block" *) logic [CONFIG_BUS_WIDTH-1:0] mem_weights_hi [WEIGHT_MEM_DEPTH];
    // For thresholds, reading 32-bits from arbitrary 64-bit alignment is annoying.
    // Store as 32-bit words internally?
    // Config writes 64-bit. We can unpack on write.
    // Or just Keep 32-bit memory and write 2 words per clock if bus is 64?
    // Simplest: Store as written (CONFIG_BUS_WIDTH) and mux on read.
    (* ram_style = "block" *) logic [CONFIG_BUS_WIDTH-1:0] mem_thresholds [THRESH_MEM_DEPTH];

    // Read Pointers and Logic
    logic [31:0] current_neuron_idx;
    logic [31:0] chunk_idx;
    logic [31:0] ptr_thresholds; // Word index
    logic [31:0] sub_ptr_thresholds; // Sub-word index (for 32-bit chunks)

    // Write Logic
    always_ff @(posedge clk) begin
        if (wr_en_weights) begin
            if (wr_addr < WEIGHT_MEM_DEPTH) begin
                mem_weights_lo[wr_addr] <= wr_data;
                mem_weights_hi[wr_addr] <= wr_data;
            end
        end
        if (wr_en_thresholds) begin
             if (wr_addr < THRESH_MEM_DEPTH)
                mem_thresholds[wr_addr] <= wr_data;
        end
    end

    // Read Logic
    always_ff @(posedge clk) begin
        if (rst || layer_start) begin
            current_neuron_idx <= '0;
            chunk_idx <= '0;
            ptr_thresholds <= '0;
            sub_ptr_thresholds <= '0;
        end else begin
            if (read_weight_chunk) begin
                chunk_idx <= chunk_idx + 1;
            end
            
            if (read_threshold) begin
                // Neuron Done. Advance to next.
                current_neuron_idx <= current_neuron_idx + 1;
                chunk_idx <= 0; // Reset chunk counter for new neuron logic
                
                // Logic to advance to next 32-bit threshold
                // Assuming CONFIG_BUS_WIDTH is multiple of 32
                if (sub_ptr_thresholds == (CONFIG_BUS_WIDTH/32 - 1)) begin
                    sub_ptr_thresholds <= '0;
                    ptr_thresholds <= ptr_thresholds + 1;
                end else begin
                    sub_ptr_thresholds <= sub_ptr_thresholds + 1;
                end
            end
        end
    end

    // ==========================
    // Synchronous Read (BRAM)
    // ==========================
    //
    // The compute pipeline uses `read_weight_chunk` as an "advance pointer" strobe:
    // when asserted in cycle N, the next chunk should be available in cycle N+1.
    // With synchronous RAM, we therefore PREFETCH the *next* chunk on the clock
    // edge, and present it on outputs in the following cycle.
    //
    // Similarly, `read_threshold` advances to the next neuron's threshold; we
    // prefetch that threshold on the same edge.

    localparam int unsigned BUS_BYTES = CONFIG_BUS_WIDTH / 8;

    // Next-state address calculations (combinational)
    logic [31:0] neuron_idx_req;
    logic [31:0] chunk_idx_req;
    logic [31:0] ptr_thresholds_req;
    logic [31:0] sub_ptr_thresholds_req;

    logic [31:0] base_byte_addr_req;
    logic [31:0] current_byte_addr_req;
    logic [31:0] word_addr_req;
    logic [31:0] bit_offset_req;

    always_comb begin
        // Defaults: hold current pointers
        neuron_idx_req         = current_neuron_idx;
        chunk_idx_req          = chunk_idx;
        ptr_thresholds_req     = ptr_thresholds;
        sub_ptr_thresholds_req = sub_ptr_thresholds;

        // Apply pointer updates that happen at the upcoming edge, so the RAM read
        // prefetches the value that will be needed in the next cycle.
        if (rst || layer_start) begin
            neuron_idx_req         = '0;
            chunk_idx_req          = '0;
            ptr_thresholds_req     = '0;
            sub_ptr_thresholds_req = '0;
        end else begin
            if (read_threshold) begin
                // Next neuron starts at chunk 0
                neuron_idx_req = current_neuron_idx + 1;
                chunk_idx_req  = '0;

                // Advance to next 32-bit threshold
                if (sub_ptr_thresholds == (CONFIG_BUS_WIDTH/32 - 1)) begin
                    sub_ptr_thresholds_req = '0;
                    ptr_thresholds_req     = ptr_thresholds + 1;
                end else begin
                    sub_ptr_thresholds_req = sub_ptr_thresholds + 1;
                    ptr_thresholds_req     = ptr_thresholds;
                end
            end else if (read_weight_chunk) begin
                chunk_idx_req = chunk_idx + 1;
            end
        end

        base_byte_addr_req    = neuron_idx_req * BYTES_PER_NEURON;
        current_byte_addr_req = base_byte_addr_req + (chunk_idx_req * BUS_BYTES);

        // Word addressing into the packed weight stream (byte aligned per neuron)
        word_addr_req   = current_byte_addr_req / BUS_BYTES;
        bit_offset_req  = (current_byte_addr_req % BUS_BYTES) * 8;
    end

    logic [CONFIG_BUS_WIDTH-1:0] weights_word_lo_q;
    logic [CONFIG_BUS_WIDTH-1:0] weights_word_hi_q;
    logic [31:0]                 bit_offset_q;

    logic [CONFIG_BUS_WIDTH-1:0] thresh_word_q;
    logic [31:0]                 sub_ptr_thresholds_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            weights_word_lo_q      <= '0;
            weights_word_hi_q      <= '0;
            bit_offset_q           <= '0;
            thresh_word_q          <= '0;
            sub_ptr_thresholds_q   <= '0;
        end else begin
            // Prefetch weights (two consecutive words) for unaligned extraction
            if (word_addr_req < WEIGHT_MEM_DEPTH)
                weights_word_lo_q <= mem_weights_lo[word_addr_req];
            else
                weights_word_lo_q <= '0;

            if (word_addr_req + 1 < WEIGHT_MEM_DEPTH)
                weights_word_hi_q <= mem_weights_hi[word_addr_req + 1];
            else
                weights_word_hi_q <= '0;

            bit_offset_q <= bit_offset_req;

            // Prefetch the threshold word and the sub-index used to slice out 32 bits
            if (ptr_thresholds_req < THRESH_MEM_DEPTH)
                thresh_word_q <= mem_thresholds[ptr_thresholds_req];
            else
                thresh_word_q <= '0;

            sub_ptr_thresholds_q <= sub_ptr_thresholds_req;
        end
    end

    logic [2*CONFIG_BUS_WIDTH-1:0] double_word_q;
    always_comb begin
        double_word_q = {weights_word_hi_q, weights_word_lo_q};
    end

    // Extract chunk with dynamic bit offset (registered data path)
    logic [2*CONFIG_BUS_WIDTH-1:0] shifted_double_word_q;
    always_comb begin
        shifted_double_word_q = (double_word_q >> bit_offset_q);
    end
    assign rd_data_weights = shifted_double_word_q[CONFIG_BUS_WIDTH-1:0];

    // Extract the correct 32-bit slice from the registered threshold word
    assign rd_data_threshold = thresh_word_q[sub_ptr_thresholds_q*32 +: 32];

endmodule
