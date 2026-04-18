// -----------------------------------------------------------------------------
// AXI4-Stream configuration parser.
//
// Consumes:
// - A fixed 128-bit header (spread across CONFIG_BUS_WIDTH beats)
// - Followed by a payload of `total_bytes` bytes (TKEEP-valid), with TLAST on the
//   final beat.
//
// Produces a simple, word-oriented write interface that is convenient for on-chip
// RAMs:
// - `layer_wr_addr` increments once per accepted beat (word address)
// - `layer_wr_data` is the raw bus beat
// - `layer_wr_strb` mirrors TKEEP so unused bytes are ignored
// - `layer_wr_en_*` is a per-layer one-hot that targets either weights or
//   thresholds for the message currently being parsed.
//
// Layer indexing:
// - The message header's `layer_id` is 0-based *excluding* the input layer.
//   i.e. layer_id=0 targets TOPOLOGY[1] (the first hidden layer), so this parser
//   asserts write-enables at [layer_id + 1].
// -----------------------------------------------------------------------------
module config_parser #(
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int TOTAL_LAYERS = 4
)(
    input logic clk,
    input logic rst,

    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,

    output logic [TOTAL_LAYERS-1:0]       layer_wr_en_weights,
    output logic [TOTAL_LAYERS-1:0]       layer_wr_en_thresholds,
    output logic [31:0]                   layer_wr_addr,
    output logic [CONFIG_BUS_WIDTH-1:0]   layer_wr_data,
    output logic [CONFIG_BUS_WIDTH/8-1:0] layer_wr_strb
);

    import bnn_types_pkg::*;
    import bnn_util_pkg::*;

    // -------------------------------------------------------------------------
    // Header/payload bookkeeping
    // -------------------------------------------------------------------------
    localparam int BYTES_PER_BEAT = CONFIG_BUS_WIDTH / 8;
    localparam int HEADER_BEATS = 128 / CONFIG_BUS_WIDTH;
    localparam int HEADER_IDX_WIDTH = clog2_safe(HEADER_BEATS);
    localparam int BEAT_ADDR_SHIFT = clog2_safe(BYTES_PER_BEAT);

    typedef enum logic {
        ST_HEADER,
        ST_PAYLOAD
    } state_t;

    state_t state;
    logic [HEADER_IDX_WIDTH-1:0] header_word_idx;
    msg_type_e                   active_msg_type;
    logic [7:0]                  active_layer_id;
    logic [31:0]                 payload_last_addr;
    logic                        payload_done;

    function automatic logic [31:0] calc_payload_last_addr(
        input logic [31:0] total_bytes
    );
        // Convert payload size in bytes into a 0-based "last word address" where
        // a word is one CONFIG_BUS_WIDTH beat.
        begin
            if (total_bytes <= BYTES_PER_BEAT) begin
                return '0;
            end
            return (total_bytes - 1'b1) >> BEAT_ADDR_SHIFT;
        end
    endfunction

    // Pass-through payload data with byte enables derived from AXI TKEEP.
    assign layer_wr_data = config_data;
    assign layer_wr_strb = config_keep;

    // End payload either by explicit TLAST, or by reaching the computed size.
    assign payload_done = config_last || (layer_wr_addr == payload_last_addr);

    // Decode only the fields used during payload handling so the parser avoids
    // carrying a wide 128-bit header datapath through the critical control logic.
    always_comb begin
        layer_wr_en_weights = '0;
        layer_wr_en_thresholds = '0;

        if (state == ST_PAYLOAD && config_valid) begin
            if (active_layer_id < TOTAL_LAYERS - 1) begin
                if (active_msg_type == MSG_TYPE_WEIGHTS) begin
                    layer_wr_en_weights[active_layer_id + 1'b1] = 1'b1;
                end else if (active_msg_type == MSG_TYPE_THRESHOLDS) begin
                    layer_wr_en_thresholds[active_layer_id + 1'b1] = 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_HEADER;
            header_word_idx <= '0;
            active_msg_type <= MSG_TYPE_WEIGHTS;
            active_layer_id <= '0;
            payload_last_addr <= '0;
            layer_wr_addr <= '0;
        end else begin
            case (state)
                ST_HEADER: begin
                    if (config_valid) begin
                        // Parse only what we need:
                        // - Beat 0: msg_type + layer_id
                        // - Beat containing total_bytes
                        if (CONFIG_BUS_WIDTH == 64) begin
                            if (header_word_idx == 0) begin
                                active_msg_type <= msg_type_e'(config_data[7:0]);
                                active_layer_id <= config_data[15:8];
                                header_word_idx <= 1'b1;
                            end else begin
                                payload_last_addr <= calc_payload_last_addr(config_data[31:0]);
                                layer_wr_addr <= '0;
                                header_word_idx <= '0;
                                state <= ST_PAYLOAD;
                            end
                        end else begin
                            case (header_word_idx)
                                0: begin
                                    active_msg_type <= msg_type_e'(config_data[7:0]);
                                    active_layer_id <= config_data[15:8];
                                    header_word_idx <= header_word_idx + 1'b1;
                                end
                                1: begin
                                    header_word_idx <= header_word_idx + 1'b1;
                                end
                                2: begin
                                    payload_last_addr <= calc_payload_last_addr(config_data);
                                    header_word_idx <= header_word_idx + 1'b1;
                                end
                                default: begin
                                    layer_wr_addr <= '0;
                                    header_word_idx <= '0;
                                    state <= ST_PAYLOAD;
                                end
                            endcase
                        end
                    end
                end

                ST_PAYLOAD: begin
                    if (config_valid) begin
                        if (payload_done) begin
                            state <= ST_HEADER;
                            header_word_idx <= '0;
                            layer_wr_addr <= '0;
                        end else begin
                            layer_wr_addr <= layer_wr_addr + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

    // This parser never backpressures configuration traffic (outside reset).
    assign config_ready = !rst;

    initial begin
        assert (CONFIG_BUS_WIDTH == 32 || CONFIG_BUS_WIDTH == 64)
        else $fatal(1, "config_parser only supports CONFIG_BUS_WIDTH of 32 or 64.");
    end

endmodule
