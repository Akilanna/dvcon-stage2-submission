# ============================================================
# CDAC VEGA SoC RankingMLP HLS project — Vitis HLS 2024.2
# Target FPGA: Kintex-7 XC7K325T-2FFG (fallback to XC7K70T), 50 MHz
# ============================================================

# Step 1: create the project (reset if it already exists)
open_project -reset ranking_mlp_hls

# Step 2: set top-level function name
set_top ranking_mlp_top

# Step 3: add source and testbench files
add_files top.cpp
add_files -tb tb.cpp

# Step 4: create solution, set FPGA part and clock
open_solution -reset solution1 -flow_target vivado

# Try targeting the specified part first, fallback if not installed
if {[catch {set_part xc7k325tffg900-2}]} {
    puts "WARNING: xc7k325tffg900-2 is not installed. Trying fallback xc7k70tfbg676-2..."
    if {[catch {set_part xc7k70tfbg676-2}]} {
        puts "WARNING: xc7k70tfbg676-2 is also not installed. Trying fallback xck26-sfvc784-2LV-c..."
        set_part xck26-sfvc784-2LV-c
    }
}

create_clock -period 20 -name default

# Configure AXI interface properties matching CDAC VEGA SoC requirements
# -m_axi_addr64   : Force 64-bit AXI master addresses
config_interface -m_axi_addr64

# Step 5: C simulation
puts "===== RUNNING CSIM ====="
csim_design -clean
puts "===== CSIM DONE ====="

# Step 6: C synthesis
puts "===== RUNNING SYNTHESIS ====="
csynth_design
puts "===== SYNTHESIS DONE ====="

# Step 7: export RTL as Vivado IP
puts "===== EXPORTING IP ====="
export_design -format ip_catalog -vendor cdac -library rankmlp -version 1.0
puts "===== EXPORT DONE ====="

puts "===== ALL STEPS COMPLETE ====="
exit
