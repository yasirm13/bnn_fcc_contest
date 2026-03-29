module config_parser #(
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int TOTAL_LAYERS = 4
)(
    input logic clk,
    input logic rst,

    // AXI Stream Input
    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,

    // Memory Write Interface
    // Broadcast data/addr, individual write enables
    output logic [TOTAL_LAYERS-1:0]       layer_wr_en_weights,
    output logic [TOTAL_LAYERS-1:0]       layer_wr_en_thresholds,
    output logic [31:0]                   layer_wr_addr, 
    output logic [CONFIG_BUS_WIDTH-1:0]   layer_wr_data,
    output logic [CONFIG_BUS_WIDTH/8-1:0] layer_wr_strb
);

    import bnn_types_pkg::*;

    // State Machine
    typedef enum logic [1:0] {
        ST_RESET,
        ST_HEADER,
        ST_PAYLOAD
    } state_t;

    state_t state;
    state_t next_state;

    // Header Processing
    logic [127:0] header_shift_reg;
    // Can reach 128 (bits), and we avoid tool "carry-out" truncation warnings by
    // giving this counter an extra bit.
    logic [8:0]   header_bits_count;
    config_header_t current_header;

    // Internal Signals
    logic header_complete;
    logic [31:0] payload_cnt; // To count bytes for message delineation
    logic [31:0] remaining_bytes; // To count down remaining bytes
    logic [4:0] bytes_this_beat; // Changed to 5-bit to fit 8
    
    // Assign outputs
    assign layer_wr_data = config_data;
    assign layer_wr_strb = config_keep;
    assign current_header = config_header_t'(header_shift_reg);

    // Write Enable generation
    always_comb begin
        layer_wr_en_weights = '0;
        layer_wr_en_thresholds = '0;

        if (state == ST_PAYLOAD && config_valid) begin
            if (current_header.msg_type == MSG_TYPE_WEIGHTS) begin
                // Config Layer ID 0 corresponds to RTL Layer 1 (First Compute Layer)
                if (current_header.layer_id + 1 < TOTAL_LAYERS)
                    layer_wr_en_weights[current_header.layer_id + 1] = 1'b1;
            end else if (current_header.msg_type == MSG_TYPE_THRESHOLDS) begin
                if (current_header.layer_id + 1 < TOTAL_LAYERS)
                    layer_wr_en_thresholds[current_header.layer_id + 1] = 1'b1;
            end
        end
    end

    // Sequential Logic
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_HEADER;
            header_bits_count <= '0;
            header_shift_reg <= '0;
            layer_wr_addr <= '0;
        end else begin
            case (state)
                ST_HEADER: begin
                    if (config_valid) begin
                        // Shift in new data
                        // Assuming Little Endian (first beat is LSB of header) for bits 0-63 etc?
                        // README visual: 
                        // Word 0 (LSB): ...
                        // So first beat (if 64 bit) is Word 0.
                        // We shift in such that we fill 0..63 then 64..127.
                        
                        // We need to handle bus widths.
                        // Generic shift logic:
                        // header_shift_reg[header_bits_count +: CONFIG_BUS_WIDTH] <= config_data; 
                        // But need to handle overflow if CONFIG_BUS_WIDTH > remaining bits?
                        // Assuming standard widths for now and simple logic.
                        
                        // For simplicity, shift right into MSB or fill LSB upwards.
                        // Let's fill from header_bits_count upwards.
                        
                        // BUT: SystemVerilog variable slice width must be constant.
                        // So we cannot do [count +: WIDTH].
                        // Solution: Use a huge shift register or case statement, or just assume 32/64 bit widths.
                        
                        // Logic for arbitrary width less than or equal to 128:
                        // (Not strictly arbitrary, but logic for simple accumulation)
                         int bits_to_take;
                         bits_to_take = (128 - header_bits_count) > CONFIG_BUS_WIDTH ? CONFIG_BUS_WIDTH : (128 - header_bits_count);
                         
                         // We can't use dynamic slicing.
                         // Instead, shift the register content?
                         // No, we want to place data into specific position.
                         
                         // Better approach for synthesis:
                         // Just act as a deserializer.
                         
                         // IF CONFIG_BUS_WIDTH == 64
	                         if (CONFIG_BUS_WIDTH == 64) begin
	                             if (header_bits_count == 0) header_shift_reg[63:0] <= config_data;
	                             if (header_bits_count == 64) header_shift_reg[127:64] <= config_data;
	                             header_bits_count <= 9'(header_bits_count + 9'd64);
	                         end
	                         else if (CONFIG_BUS_WIDTH == 32) begin
	                             if (header_bits_count == 0) header_shift_reg[31:0] <= config_data;
	                             if (header_bits_count == 32) header_shift_reg[63:32] <= config_data;
	                             if (header_bits_count == 64) header_shift_reg[95:64] <= config_data;
	                             if (header_bits_count == 96) header_shift_reg[127:96] <= config_data;
	                             header_bits_count <= 9'(header_bits_count + 9'd32);
	                         end
                         // Add more if needed, or make generic using loop (synthesizable in SV usually if constant loop)
                         
                         if ((header_bits_count + CONFIG_BUS_WIDTH) >= 128) begin
                             state <= ST_PAYLOAD;
                             layer_wr_addr <= '0; // Reset address for payload
                             payload_cnt <= '0;   // Reset byte count
                             if (CONFIG_BUS_WIDTH == 64) begin
                                 remaining_bytes <= config_data[31:0]; // [95:64] comes from config_data[31:0]
                             end else begin
                                 remaining_bytes <= config_data; // fallback for 32-bit if it matches the beat
                             end
                             
                             // Debug Header
                             // Note: header_shift_reg might not be fully updated yet if we assume sequential?
                             // wait, we update header_shift_reg inside this FF block.
                             // The displayed value will be the OLD value?
                             // Or we can display the inputs.
                             // Let's print in ST_PAYLOAD entry.
                             // Or next cycle.
                         end
                    end
                end

                ST_PAYLOAD: begin
                    if (layer_wr_addr == 0 && config_valid) begin
                         $display("ConfigParser: MsgType %d LayerID %d TotalBytes %d", current_header.msg_type, current_header.layer_id, current_header.total_bytes);
                    end
                    
                    if (config_valid) begin
                        // Address increments on every accepted beat
                        layer_wr_addr <= layer_wr_addr + 1;
                        
                        // Count bytes
                        // Use TLAST as an override, but primarily use byte counting from header
                        // $countones is generic.
                        bytes_this_beat = 0;
                        for(int i=0; i<CONFIG_BUS_WIDTH/8; i++) bytes_this_beat += 5'(config_keep[i]);
                        
                        payload_cnt <= payload_cnt + 32'(bytes_this_beat);
                        remaining_bytes <= remaining_bytes - 32'(bytes_this_beat);
                        
                        // Debug every 100 cycles/beats
                        if ((payload_cnt & 127) == 0) begin
                             $display("PAYLOAD DBG: Cnt %d Beat %d Total %d Keep %x", payload_cnt, bytes_this_beat, current_header.total_bytes, config_keep);
                        end

                        if (config_last || (32'(bytes_this_beat) >= remaining_bytes)) begin
                            $display("PAYLOAD DONE: Cnt %d Total %d", payload_cnt + 32'(bytes_this_beat), current_header.total_bytes);
                            state <= ST_HEADER;
                            header_bits_count <= '0;
                            // Ensure we don't start shifting header this cycle?
                            // config_data this cycle is PAYLOAD. Next cycle is HEADER.
                        end
                    end
                end
            endcase
        end
    end

    // Ready signal logic
    // We are always ready unless reset is active (handled by modules usually)
    // Actually, we could be backpressured if the memory wasn't ready?
    // But we are designing the memory, and it should be fast (BRAM).
    // So config_ready can be 1.
    assign config_ready = !rst;

endmodule
