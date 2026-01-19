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
    logic [CONFIG_BUS_WIDTH-1:0] mem_weights [WEIGHT_MEM_DEPTH];
    // For thresholds, reading 32-bits from arbitrary 64-bit alignment is annoying.
    // Store as 32-bit words internally?
    // Config writes 64-bit. We can unpack on write.
    // Or just Keep 32-bit memory and write 2 words per clock if bus is 64?
    // Simplest: Store as written (CONFIG_BUS_WIDTH) and mux on read.
    logic [CONFIG_BUS_WIDTH-1:0] mem_thresholds [THRESH_MEM_DEPTH];

    // Read Pointers and Logic
    logic [31:0] current_neuron_idx;
    logic [31:0] chunk_idx;
    logic [31:0] ptr_thresholds; // Word index
    logic [31:0] sub_ptr_thresholds; // Sub-word index (for 32-bit chunks)

    // Write Logic
    always_ff @(posedge clk) begin
        if (wr_en_weights) begin
            if (wr_addr < WEIGHT_MEM_DEPTH)
                mem_weights[wr_addr] <= wr_data;
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

    // Data Output
    
    // Unaligned Read Logic for Weights
    logic [31:0] base_byte_addr;
    logic [31:0] current_byte_addr;
    logic [31:0] word_addr;
    logic [31:0] bit_offset;
    logic [127:0] double_word_read;
    
    always_comb begin
        // Calculate addressing
        base_byte_addr = current_neuron_idx * BYTES_PER_NEURON;
        current_byte_addr = base_byte_addr + (chunk_idx * (CONFIG_BUS_WIDTH/8));
        
        word_addr = current_byte_addr / (CONFIG_BUS_WIDTH/8);
        bit_offset = (current_byte_addr % (CONFIG_BUS_WIDTH/8)) * 8;
        
        // Read 2 words (safe bounds check handled by memory logic or simply ignored if valid stream)
        // If word_addr is valid
        if (word_addr < WEIGHT_MEM_DEPTH) 
            double_word_read[CONFIG_BUS_WIDTH-1:0] = mem_weights[word_addr];
        else
            double_word_read[CONFIG_BUS_WIDTH-1:0] = '0;
            
        // If word_addr+1 is valid (needed for crossing boundary)
        if (word_addr + 1 < WEIGHT_MEM_DEPTH)
            double_word_read[2*CONFIG_BUS_WIDTH-1:CONFIG_BUS_WIDTH] = mem_weights[word_addr + 1];
        else 
            double_word_read[2*CONFIG_BUS_WIDTH-1:CONFIG_BUS_WIDTH] = '0;
            
    end
    
    // Extract aligned chunk
    assign rd_data_weights = double_word_read[bit_offset +: CONFIG_BUS_WIDTH];
    
    // Mux for threshold output
    // Extract the correct 32-bit slice
    assign rd_data_threshold = mem_thresholds[ptr_thresholds][sub_ptr_thresholds*32 +: 32];

endmodule
