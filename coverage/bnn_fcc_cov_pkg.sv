package bnn_fcc_cov_pkg;

    typedef enum int {
        GAP_CONTINUOUS = 0,
        GAP_INTERMITTENT = 1,
        GAP_BURSTY = 2
    } gap_style_e;

    typedef enum int {
        CFG_SCOPE_FULL = 0,
        CFG_SCOPE_WEIGHTS_ONLY = 1,
        CFG_SCOPE_THRESHOLDS_ONLY = 2,
        CFG_SCOPE_PARTIAL_SUBSET = 3,
        CFG_SCOPE_PARTIAL_BOTH = 4
    } cfg_scope_e;

    typedef enum int {
        READY_BEFORE_VALID = 0,
        READY_SAME_CYCLE = 1,
        READY_AFTER_VALID = 2
    } ready_relation_e;

    typedef enum int {
        READY_STYLE_CONTINUOUS = 0,
        READY_STYLE_INTERMITTENT = 1,
        READY_STYLE_BURSTY = 2
    } ready_style_e;

    typedef enum int {
        RESET_PHASE_BEFORE_CONFIG = 0,
        RESET_PHASE_DURING_CONFIG = 1,
        RESET_PHASE_AFTER_CONFIG = 2,
        RESET_PHASE_DURING_IMAGE = 3,
        RESET_PHASE_DURING_OUTPUT = 4
    } reset_phase_e;

    typedef enum int {
        POST_RESET_SAME_CONFIG = 0,
        POST_RESET_DIFFERENT_CONFIG = 1
    } post_reset_cfg_e;

    typedef enum int {
        ORDER_FIRST = 0,
        ORDER_ASCENDING_LAYER = 1,
        ORDER_DESCENDING_LAYER = 2,
        ORDER_SAME_LAYER_TYPE_SWAP = 3,
        ORDER_SAME_LAYER_REPEAT = 4
    } cfg_order_e;

    typedef enum int {
        WORKLOAD_NONE = 0,
        WORKLOAD_CONFIG_ONLY = 1,
        WORKLOAD_PARTIAL_IMAGE = 2,
        WORKLOAD_IMAGES_IN_FLIGHT = 3
    } reset_workload_e;

    class bnn_fcc_cov_collector;
        local int layer_count;
        local int output_class_count;
        local bit config_partial_keep_possible;
        local bit image_partial_keep_possible;

        covergroup cg_scenario with function sample(int scenario_id);
            option.per_instance = 1;

            cp_scenario: coverpoint scenario_id {
                bins scenarios[] = {[0:6]};
            }
        endgroup

        covergroup cg_config_beat with function sample(
            int gap_cycles,
            int keep_kind,
            bit last_beat
        );
            option.per_instance = 1;

            cp_gap: coverpoint gap_cycles {
                bins continuous = {0};
                bins intermittent = {[1:2]};
                bins bursty = {[3:$]};
            }

            cp_keep: coverpoint keep_kind {
                bins partial = {1} iff (config_partial_keep_possible);
                bins full = {2};
            }

            cp_last: coverpoint last_beat {
                bins middle = {0};
                bins last = {1};
            }
        endgroup

        covergroup cg_config_message with function sample(
            int msg_type,
            int layer_id,
            int scope,
            int order_kind,
            bit mutated_threshold_fields
        );
            option.per_instance = 1;

            cp_type: coverpoint msg_type {
                bins weights = {0};
                bins thresholds = {1};
            }

            cp_layer: coverpoint layer_id {
                bins layers[] = {[0:layer_count-1]};
            }

            cp_scope: coverpoint scope {
                bins full = {CFG_SCOPE_FULL};
                bins weights_only = {CFG_SCOPE_WEIGHTS_ONLY};
                bins thresholds_only = {CFG_SCOPE_THRESHOLDS_ONLY};
                bins partial_subset = {CFG_SCOPE_PARTIAL_SUBSET};
                bins partial_both = {CFG_SCOPE_PARTIAL_BOTH};
            }

            cp_order: coverpoint order_kind {
                bins first = {ORDER_FIRST};
                bins ascending = {ORDER_ASCENDING_LAYER};
                bins descending = {ORDER_DESCENDING_LAYER};
                bins swap_type = {ORDER_SAME_LAYER_TYPE_SWAP};
                bins repeat_layer = {ORDER_SAME_LAYER_REPEAT};
            }

            cp_mutation: coverpoint mutated_threshold_fields {
                bins unmodified = {0};
                bins mutated = {1};
            }

            cp_type_x_layer: cross cp_type, cp_layer {
                ignore_bins output_threshold = binsof(cp_type.thresholds) &&
                    binsof(cp_layer) intersect {layer_count - 1};
            }
        endgroup

        covergroup cg_model with function sample(
            int layer_id,
            int weight_density_pct,
            int threshold_pct
        );
            option.per_instance = 1;

            cp_layer: coverpoint layer_id {
                bins layers[] = {[0:layer_count-1]};
            }

            cp_density: coverpoint weight_density_pct {
                bins b_very_low = {[0:10]};
                bins b_low = {[11:35]};
                bins b_mid = {[36:65]};
                bins b_high = {[66:90]};
                bins b_very_high = {[91:100]};
            }

            cp_threshold: coverpoint threshold_pct {
                bins b_small = {[0:20]};
                bins b_mid = {[21:60]};
                bins b_large = {[61:100]};
            }

            cp_layer_x_density: cross cp_layer, cp_density;
            cp_layer_x_threshold: cross cp_layer, cp_threshold {
                ignore_bins output_threshold = binsof(cp_layer) intersect {layer_count - 1} &&
                    (binsof(cp_threshold.b_mid) || binsof(cp_threshold.b_large));
            }
        endgroup

        covergroup cg_image with function sample(
            int gap_cycles,
            int keep_kind,
            bit last_beat,
            int inter_image_gap,
            int image_count
        );
            option.per_instance = 1;

            cp_gap: coverpoint gap_cycles {
                bins continuous = {0};
                bins intermittent = {[1:2]};
                bins bursty = {[3:$]};
            }

            cp_keep: coverpoint keep_kind {
                bins partial = {1} iff (image_partial_keep_possible);
                bins full = {2};
            }

            cp_last: coverpoint last_beat {
                bins middle = {0};
                bins last = {1};
            }

            cp_inter_image_gap: coverpoint inter_image_gap {
                bins b_back_to_back = {0};
                bins b_short_gap = {[1:3]};
                bins b_long_gap = {[4:$]};
            }

            cp_image_count: coverpoint image_count {
                bins b_few = {[1:2]};
                bins b_mid = {[3:5]};
                bins b_many = {[6:$]};
            }
        endgroup

        covergroup cg_output with function sample(
            int class_idx,
            bit repeated_class,
            int ready_relation,
            int ready_style,
            int stall_cycles,
            bit keep_all_ones
        );
            option.per_instance = 1;

            cp_class: coverpoint class_idx {
                bins classes[] = {[0:output_class_count-1]};
            }

            cp_repeat: coverpoint repeated_class {
                bins b_same_as_previous = {1};
                bins b_changed = {0};
            }

            cp_relation: coverpoint ready_relation {
                bins b_before_valid = {READY_BEFORE_VALID};
                bins b_same_cycle = {READY_SAME_CYCLE};
                bins b_after_valid = {READY_AFTER_VALID};
            }

            cp_style: coverpoint ready_style {
                bins b_continuous = {READY_STYLE_CONTINUOUS};
                bins b_intermittent = {READY_STYLE_INTERMITTENT};
                bins b_bursty = {READY_STYLE_BURSTY};
            }

            cp_stall: coverpoint stall_cycles {
                bins b_no_stall = {0};
                bins b_short_stall = {[1:2]};
                bins b_long_stall = {[3:$]};
            }

            cp_keep: coverpoint keep_all_ones {
                bins good_keep = {1};
            }

            cp_relation_x_style: cross cp_relation, cp_style;
            cp_relation_x_stall: cross cp_relation, cp_stall {
                ignore_bins non_after_stalls =
                    (binsof(cp_relation.b_before_valid) || binsof(cp_relation.b_same_cycle)) &&
                    (binsof(cp_stall.b_short_stall) || binsof(cp_stall.b_long_stall));
            }
        endgroup

        covergroup cg_reset with function sample(
            int phase,
            bit on_boundary,
            int workload,
            int post_reset_cfg,
            int reset_count
        );
            option.per_instance = 1;

            cp_phase: coverpoint phase {
                bins before_config = {RESET_PHASE_BEFORE_CONFIG};
                bins during_config = {RESET_PHASE_DURING_CONFIG};
                bins after_config = {RESET_PHASE_AFTER_CONFIG};
                bins during_image = {RESET_PHASE_DURING_IMAGE};
                bins during_output = {RESET_PHASE_DURING_OUTPUT};
            }

            cp_boundary: coverpoint on_boundary {
                bins off_boundary = {0};
                bins on_boundary = {1};
            }

            cp_workload: coverpoint workload {
                bins none = {WORKLOAD_NONE};
                bins config_only = {WORKLOAD_CONFIG_ONLY};
                bins partial_image = {WORKLOAD_PARTIAL_IMAGE};
                bins images_in_flight = {WORKLOAD_IMAGES_IN_FLIGHT};
            }

            cp_post_cfg: coverpoint post_reset_cfg {
                bins same_cfg = {POST_RESET_SAME_CONFIG};
                bins different_cfg = {POST_RESET_DIFFERENT_CONFIG};
            }

            cp_count: coverpoint reset_count {
                bins single = {1};
                bins multiple = {[2:$]};
            }

            cp_phase_x_boundary: cross cp_phase, cp_boundary;
            cp_phase_x_post_cfg: cross cp_phase, cp_post_cfg;
        endgroup

        function new(
            int layer_count_i,
            int output_class_count_i,
            bit config_partial_keep_possible_i,
            bit image_partial_keep_possible_i
        );
            this.layer_count = layer_count_i;
            this.output_class_count = output_class_count_i;
            this.config_partial_keep_possible = config_partial_keep_possible_i;
            this.image_partial_keep_possible = image_partial_keep_possible_i;
            cg_scenario = new();
            cg_config_beat = new();
            cg_config_message = new();
            cg_model = new();
            cg_image = new();
            cg_output = new();
            cg_reset = new();
        endfunction

        function void sample_scenario(int scenario_id);
            cg_scenario.sample(scenario_id);
        endfunction

        function void sample_config_beat(int gap_cycles, int keep_kind, bit last_beat);
            cg_config_beat.sample(gap_cycles, keep_kind, last_beat);
        endfunction

        function void sample_config_message(
            int msg_type,
            int layer_id,
            cfg_scope_e scope,
            cfg_order_e order_kind,
            bit mutated_threshold_fields
        );
            cg_config_message.sample(msg_type, layer_id, scope, order_kind, mutated_threshold_fields);
        endfunction

        function void sample_model(int layer_id, int weight_density_pct, int threshold_pct);
            cg_model.sample(layer_id, weight_density_pct, threshold_pct);
        endfunction

        function void sample_image(
            int gap_cycles,
            int keep_kind,
            bit last_beat,
            int inter_image_gap,
            int image_count
        );
            cg_image.sample(gap_cycles, keep_kind, last_beat, inter_image_gap, image_count);
        endfunction

        function void sample_output(
            int class_idx,
            bit repeated_class,
            ready_relation_e ready_relation,
            ready_style_e ready_style,
            int stall_cycles,
            bit keep_all_ones
        );
            cg_output.sample(class_idx, repeated_class, ready_relation, ready_style, stall_cycles,
                keep_all_ones);
        endfunction

        function void sample_reset(
            reset_phase_e phase,
            bit on_boundary,
            reset_workload_e workload,
            post_reset_cfg_e post_reset_cfg,
            int reset_count
        );
            cg_reset.sample(phase, on_boundary, workload, post_reset_cfg, reset_count);
        endfunction
    endclass

endpackage
