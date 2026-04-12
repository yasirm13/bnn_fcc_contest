package bnn_fcc_cat1_pkg;
    import bnn_fcc_tb_pkg::*;

    class Extended_BNN_FCC_Model #(
        int BUS_WIDTH = 32
    ) extends BNN_FCC_Model #(BUS_WIDTH);

        function new();
            super.new();
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
                    get_layer_config(l, 0, layer_stream, layer_keep);
                    full_stream = {full_stream, layer_stream};
                    full_keep   = {full_keep, layer_keep};
                end else begin
                    do_weights_first = ($urandom_range(0, 1) == 1);
                    
                    if (do_weights_first) begin
                        get_layer_config(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};

                        get_layer_config(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end else begin
                        get_layer_config(l, 1, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                        
                        get_layer_config(l, 0, layer_stream, layer_keep);
                        full_stream = {full_stream, layer_stream};
                        full_keep   = {full_keep, layer_keep};
                    end
                end
            end
        endfunction
    endclass
endpackage
