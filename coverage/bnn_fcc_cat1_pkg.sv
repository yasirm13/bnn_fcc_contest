package bnn_fcc_cat1_pkg;
    import bnn_fcc_tb_pkg::*;

    class Extended_BNN_FCC_Model #(
        int BUS_WIDTH = 32
    ) extends BNN_FCC_Model #(BUS_WIDTH);
        localparam int DENSITY_ALL_ZERO = 0;
        localparam int DENSITY_SPARSE   = 1;
        localparam int DENSITY_BALANCED = 2;
        localparam int DENSITY_DENSE    = 3;
        localparam int DENSITY_ALL_ONE  = 4;

        function new();
            super.new();
        endfunction

        protected function automatic bit get_density_bit(input int density_mode, input int bit_idx, input int seed);
            case (density_mode)
                DENSITY_ALL_ZERO: return 1'b0;
                DENSITY_SPARSE:   return (((bit_idx * 7) + seed) % 8) < 2;
                DENSITY_BALANCED: return (((bit_idx + seed) % 2) == 0);
                DENSITY_DENSE:    return (((bit_idx * 5) + seed) % 8) != 0 && (((bit_idx * 5) + seed) % 8) != 1;
                DENSITY_ALL_ONE:  return 1'b1;
                default:          return 1'b0;
            endcase
        endfunction

        protected function automatic int choose_density_mode(input int layer_idx, input int neuron_idx);
            return (layer_idx + neuron_idx) % 5;
        endfunction

        protected function automatic int choose_ones_count(input int fan_in, input int density_mode);
            int sparse_count;
            int balanced_count;
            int dense_count;

            sparse_count   = ((fan_in + 3) / 4);
            if (sparse_count <= 0) sparse_count = 1;
            if ((sparse_count * 4) > fan_in) sparse_count = fan_in / 4;
            if (sparse_count <= 0) sparse_count = 1;

            balanced_count = fan_in / 2;
            dense_count    = ((fan_in * 3) + 3) / 4;
            if ((dense_count * 4) < (fan_in * 3)) dense_count++;
            if (dense_count >= fan_in) dense_count = fan_in - 1;

            case (density_mode)
                DENSITY_ALL_ZERO: return 0;
                DENSITY_SPARSE:   return sparse_count;
                DENSITY_BALANCED: return balanced_count;
                DENSITY_DENSE:    return dense_count;
                DENSITY_ALL_ONE:  return fan_in;
                default:          return balanced_count;
            endcase
        endfunction

        protected function automatic int choose_threshold_value(input int fan_in, input int layer_idx, input int neuron_idx);
            int threshold_class;
            int small_val;
            int medium_val;
            int large_val;

            threshold_class = (layer_idx + (2 * neuron_idx)) % 3;
            small_val  = (neuron_idx % 2 == 0) ? 0 : ((fan_in > 0) ? 1 : 0);
            medium_val = fan_in / 2;
            large_val  = (fan_in > 0) ? (fan_in - (neuron_idx % 2)) : 0;

            case (threshold_class)
                0: return small_val;
                1: return medium_val;
                default: return large_val;
            endcase
        endfunction

        function void create_diverse_config(int user_topology[]);
            this.topology   = user_topology;
            this.num_layers = user_topology.size() - 1;
            this.weight     = new[num_layers];
            this.threshold  = new[num_layers];

            for (int l = 0; l < num_layers; l++) begin
                int n_inputs;
                int n_neurons;

                n_inputs  = topology[l];
                n_neurons = topology[l+1];

                $display("Creating Diverse Layer %0d: %0d inputs -> %0d neurons", l, n_inputs, n_neurons);
                this.weight[l]    = new[n_neurons];
                this.threshold[l] = new[n_neurons];

                for (int n = 0; n < n_neurons; n++) begin
                    int density_mode;
                    int ones_target;
                    weight_row_t temp_w;

                    density_mode = choose_density_mode(l, n);
                    ones_target = choose_ones_count(n_inputs, density_mode);
                    temp_w = new[n_inputs];

                    for (int i = 0; i < n_inputs; i++) begin
                        temp_w[i] = 1'b0;
                    end

                    for (int i = 0; i < ones_target; i++) begin
                        int bit_pos;
                        bit_pos = ((i * 5) + (l * 11) + (n * 3)) % n_inputs;
                        temp_w[bit_pos] = 1'b1;
                    end

                    // Fill any collisions deterministically until the exact count is reached.
                    for (int i = 0; i < n_inputs; i++) begin
                        int current_ones;
                        current_ones = 0;
                        for (int j = 0; j < n_inputs; j++) begin
                            if (temp_w[j]) current_ones++;
                        end
                        if (current_ones >= ones_target) break;
                        if (!temp_w[i]) temp_w[i] = get_density_bit(density_mode, i, (l * 11) + (n * 3)) || 1'b1;
                    end

                    this.weight[l][n] = temp_w;

                    if (l == num_layers - 1) this.threshold[l][n] = 0;
                    else this.threshold[l][n] = choose_threshold_value(n_inputs, l, n);
                end
            end

            is_loaded     = 1;
            outputs_valid = 0;
        endfunction

        function void get_layer_config_randomized_fields(
            input int layer_idx,
            input bit is_threshold,
            output bus_stream_t stream,
            output keep_stream_t keep
        );
            bit [7:0] byte_q[$];
            bus_word_t word_q[$];
            bus_keep_t keep_q[$];

            int fan_in;
            int n_neurons;
            int bytes_per_neuron;
            int layer_inputs;
            longint total_payload_bytes;
            bit [127:0] header_val;

            bus_word_t current_word;
            bus_keep_t current_keep;
            int bytes_per_beat;
            int byte_count;

            int w_idx;
            bit [7:0] byte_val;
            bit [31:0] t_val;

            fan_in = this.topology[layer_idx];
            n_neurons = this.topology[layer_idx+1];

            if (is_threshold) begin
                bytes_per_neuron = 4;
                layer_inputs     = (layer_idx % 2 == 0) ? 16'd32 : $urandom_range(0, 16'hffff);
            end else begin
                bytes_per_neuron = (fan_in + 7) / 8;
                layer_inputs     = fan_in;
            end

            total_payload_bytes = bytes_per_neuron * n_neurons;
            header_val          = '0;

            header_val[07:00]   = (is_threshold) ? 8'd1 : 8'd0;
            header_val[15:08]   = layer_idx[7:0];
            header_val[31:16]   = layer_inputs[15:0];
            header_val[47:32]   = n_neurons[15:0];
            header_val[63:48]   = bytes_per_neuron[15:0];
            header_val[95:64]   = total_payload_bytes[31:0];
            header_val[127:96]  = (is_threshold && (layer_idx % 2 == 1)) ? $urandom() : 32'd0;

            for (int i = 0; i < 16; i++) byte_q.push_back(header_val[i*8+:8]);

            for (int n = 0; n < n_neurons; n++) begin
                if (is_threshold) begin
                    t_val = this.threshold[layer_idx][n];
                    for (int i = 0; i < 4; i++) byte_q.push_back(t_val[i*8+:8]);
                end else begin
                    w_idx = 0;
                    for (int b = 0; b < bytes_per_neuron; b++) begin
                        for (int k = 0; k < 8; k++) begin
                            if (w_idx < fan_in) byte_val[k] = this.weight[layer_idx][n][w_idx];
                            else byte_val[k] = 1'b1;
                            w_idx++;
                        end
                        byte_q.push_back(byte_val);
                    end
                end
            end

            bytes_per_beat = BUS_WIDTH / 8;
            byte_count = 0;
            current_word = '0;

            foreach (byte_q[i]) begin
                current_word[byte_count*8+:8] = byte_q[i];
                byte_count++;

                if (byte_count == bytes_per_beat) begin
                    word_q.push_back(current_word);
                    keep_q.push_back({(BUS_WIDTH / 8){1'b1}});
                    current_word = '0;
                    byte_count   = 0;
                end
            end

            if (byte_count > 0) begin
                word_q.push_back(current_word);
                current_keep = '0;
                for (int k = 0; k < byte_count; k++) current_keep[k] = 1'b1;
                keep_q.push_back(current_keep);
            end

            stream = new[word_q.size()];
            keep   = new[keep_q.size()];
            foreach (word_q[i]) begin
                stream[i] = word_q[i];
                keep[i]   = keep_q[i];
            end
        endfunction

        function void encode_configuration_randomized(output bus_stream_t full_stream, output keep_stream_t full_keep);
            bus_stream_t layer_stream;
            keep_stream_t layer_keep;
            
            // Randomize layer order
            int layer_indices[];
            
            layer_indices = new[num_layers];
            for (int i=0; i<num_layers; i++) layer_indices[i] = i;
            
            
            
            full_stream = new[0];
            full_keep   = new[0];

            for (int i = 0; i < num_layers; i++) begin
                int l = layer_indices[i];
                bit do_weights_first;
                
                if (l == num_layers - 1) begin
                    get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                    full_stream = {full_stream, layer_stream};
                    full_keep   = {full_keep, layer_keep};
                end else begin
                    do_weights_first = ($urandom_range(0, 1) == 1);
                    
                    if (do_weights_first) begin
                        get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};

                        get_layer_config_randomized_fields(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end else begin
                        get_layer_config_randomized_fields(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                        
                        get_layer_config_randomized_fields(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end
                end
            end
        endfunction
    endclass
endpackage
