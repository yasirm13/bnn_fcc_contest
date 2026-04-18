package bnn_util_pkg;

    // Integer ceil division for compile-time sizing.
    function automatic int div_ceil(
        input int n,
        input int d
    );
        begin
            if (d <= 0) begin
                $fatal(1, "div_ceil requires d > 0");
            end
            if (n < 0) begin
                $fatal(1, "div_ceil requires n >= 0");
            end
            return (n + d - 1) / d;
        end
    endfunction

    // Safe $clog2 wrapper:
    // - Returns 1 for v <= 1 so zero-width vectors are avoided.
    // - Fatal on negative values to catch parameter bugs early.
    function automatic int clog2_safe(
        input int v
    );
        begin
            if (v < 0) begin
                $fatal(1, "clog2_safe requires v >= 0");
            end
            if (v <= 1) begin
                return 1;
            end
            return $clog2(v);
        end
    endfunction

    // Unsigned-style max for ints used in parameter math.
    function automatic int max_u(
        input int a,
        input int b
    );
        begin
            return (a > b) ? a : b;
        end
    endfunction

    // Unsigned-style min for ints used in parameter math.
    function automatic int min_u(
        input int a,
        input int b
    );
        begin
            return (a < b) ? a : b;
        end
    endfunction

endpackage
