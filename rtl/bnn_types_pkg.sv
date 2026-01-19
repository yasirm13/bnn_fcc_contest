package bnn_types_pkg;
    
    // Message type from configuration header
    typedef enum logic [7:0] {
        MSG_TYPE_WEIGHTS    = 8'd0,
        MSG_TYPE_THRESHOLDS = 8'd1
    } msg_type_e;

    // Configuration Header Structure (128-bit)
    // Defined as packed to match the bit ordering specified in the README.
    // Word 0 (LSB): bytes_per_neuron (63:48), num_neurons (47:32), layer_inputs (31:16), layer_id (15:8), msg_type (7:0)
    // Word 1 (MSB): reserved (127:96), total_bytes (95:64)
    typedef struct packed {
        logic [31:0] reserved;
        logic [31:0] total_bytes;
        logic [15:0] bytes_per_neuron;
        logic [15:0] num_neurons;
        logic [15:0] layer_inputs;
        logic [7:0]  layer_id;
        msg_type_e   msg_type;
    } config_header_t;

endpackage
