set dcp_path [lindex $argv 0]
set xdc_path [lindex $argv 1]
set out_dir  [lindex $argv 2]
set jobs     [lindex $argv 3]

file mkdir $out_dir

puts "----------------------------------------"
puts " Post-Synth Implementation (opt/place/route)"
puts "----------------------------------------"
puts "DCP:  $dcp_path"
puts "XDC:  $xdc_path"
puts "OUT:  $out_dir"
puts "JOBS: $jobs"

set_param general.maxThreads $jobs

open_checkpoint $dcp_path
if { $xdc_path ne "" } {
  read_xdc $xdc_path
}

opt_design
write_checkpoint -force "$out_dir/post_opt.dcp"

place_design
write_checkpoint -force "$out_dir/post_place.dcp"
report_utilization -file "$out_dir/post_place_util.rpt"
report_timing_summary -file "$out_dir/post_place_timing_summary.rpt"

route_design
write_checkpoint -force "$out_dir/post_route.dcp"
report_route_status -file "$out_dir/post_route_status.rpt"
report_timing_summary -file "$out_dir/post_route_timing_summary.rpt"
report_power -file "$out_dir/post_route_power.rpt"
report_drc -file "$out_dir/post_imp_drc.rpt"

puts "----------------------------------------"
puts " Done"
puts "----------------------------------------"

quit
