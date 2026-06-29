# Unified Makefile for CDAC VEGA SoC RankingMLP Accelerator (Stage 2B Submission)
# Supports both Vivado Simulator (default) and Questa Sim.

.PHONY: all sim sim_vivado sim_questa hls sim_ref report clean

all:
	@echo "========================================================================="
	@echo "CDAC VEGA SoC RankingMLP Accelerator — Stage 2B Submission Makefile"
	@echo "Available targets:"
	@echo "  make sim_ref     - Compile and run fixed-point C++ reference simulation"
	@echo "  make sim_vivado  - Compile and run RTL simulation in Vivado Simulator (xsim)"
	@echo "  make sim_questa  - Compile and run RTL simulation in Mentor/Siemens Questa Sim"
	@echo "  make hls         - Launch Vitis HLS 2024.2 synthesis and IP packaging"
	@echo "  make report      - Display HLS synthesis utilization and timing report"
	@echo "  make clean       - Clean temporary simulation and synthesis outputs"
	@echo "========================================================================="

# 1. Run C++ reference simulation
sim_ref:
	g++ -O2 sim/ref_kernel.cpp sim/tb_ref.cpp -o sim/ref_sim
	./sim/ref_sim

# 2. Run RTL simulation using Vivado Simulator (xsim)
sim: sim_vivado

sim_vivado:
	powershell -ExecutionPolicy Bypass -File sim/run_sim.ps1

# 3. Run RTL simulation using Questa Sim
sim_questa:
	vsim -c -do sim/run_sim_questa.tcl

# 4. Run High-Level Synthesis (HLS)
hls:
	cd hls && vitis_hls -f run_hls.tcl

# 5. Display HLS synthesis report
report:
	@if exist hls\ranking_mlp_hls\solution1\syn\report\ranking_mlp_top_csynth.rpt ( \
		type hls\ranking_mlp_hls\solution1\syn\report\ranking_mlp_top_csynth.rpt \
	) else ( \
		echo "HLS synthesis report not found. Please run 'make hls' first." \
	)

# 6. Clean temporary files
clean:
	@if exist sim\ref_sim (del /Q sim\ref_sim)
	@if exist sim\ref_sim.exe (del /Q sim\ref_sim.exe)
	@if exist xsim.dir (rmdir /S /Q xsim.dir)
	@if exist xsim.log (del /Q xsim.log)
	@if exist xsim.jou (del /Q xsim.jou)
	@if exist xelab.log (del /Q xelab.log)
	@if exist xelab.pb (del /Q xelab.pb)
	@if exist xvlog.log (del /Q xvlog.log)
	@if exist xvlog.pb (del /Q xvlog.pb)
	@if exist sim\hls_files.f (del /Q sim\hls_files.f)
	@if exist work (rmdir /S /Q work)
	@if exist transcript (del /Q transcript)
	@if exist vsim.wlf (del /Q vsim.wlf)
	@if exist wlft* (del /Q wlft*)
	@if exist modelsim.ini (del /Q modelsim.ini)
