// AXI4-Stream configuration parser.
// Consumes the 128-bit message header over CONFIG_BUS_WIDTH beats, tracks payload
// addresses, and emits per-layer write enables plus aligned payload data/strb.
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

    localparam int BYTES_PER_BEAT = CONFIG_BUS_WIDTH / 8;
    localparam int HEADER_BEATS = 128 / CONFIG_BUS_WIDTH;
    localparam int HEADER_IDX_WIDTH = (HEADER_BEATS > 1) ? $clog2(HEADER_BEATS) : 1;
    localparam int BEAT_ADDR_SHIFT = $clog2(BYTES_PER_BEAT);

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
        begin
            if (total_bytes <= BYTES_PER_BEAT) begin
                return '0;
            end
            return (total_bytes - 1'b1) >> BEAT_ADDR_SHIFT;
        end
    endfunction

    assign layer_wr_data = config_data;
    assign layer_wr_strb = config_keep;
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

    assign config_ready = !rst;

    initial begin
        assert (CONFIG_BUS_WIDTH == 32 || CONFIG_BUS_WIDTH == 64)
        else $fatal(1, "config_parser only supports CONFIG_BUS_WIDTH of 32 or 64.");
    end

endmodule
