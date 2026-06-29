# Tcl script to compile and run simulation in Questa Sim
# Run this from the root of the submission directory using: vsim -c -do sim/run_sim_questa.tcl

vlib work
vmap work work

echo "Copying ROM initialization files (.dat) to current directory"
foreach f [glob -nocomplain rtl/*.dat] {
    file copy -force $f .
}

echo "Compiling HLS-generated Verilog RTL files"
vlog -work work rtl/*.v

echo "Compiling SystemVerilog Memory Model and Testbench"
vlog -sv -work work sim/axi4_mem_model.sv sim/rankmlp_accel_tb.sv

echo "Launching simulation in batch mode"
vsim -c work.rankmlp_accel_tb -do "run -all; quit"

echo "Cleaning up ROM initialization files (.dat)"
foreach f [glob -nocomplain *.dat] {
    file delete -force $f
}
