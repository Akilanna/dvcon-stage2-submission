$ErrorActionPreference = "Stop"

Write-Host "Running xvlog for glbl.v"
& D:/Xilinx/Vivado/2024.2/bin/xvlog.bat -work work D:/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v

Write-Host "Generating file list for HLS verilog"
Get-ChildItem -Path "rtl/*.v" | Select-Object -ExpandProperty FullName | Out-File -FilePath "sim/hls_files.f" -Encoding ascii

Write-Host "Copying .dat weight/bias initialization files to current directory"
Copy-Item -Path "rtl/*.dat" -Destination "."

Write-Host "Running xvlog for HLS verilog using file list"
& D:/Xilinx/Vivado/2024.2/bin/xvlog.bat -work work -f sim/hls_files.f

Write-Host "Running xvlog for SV files"
& D:/Xilinx/Vivado/2024.2/bin/xvlog.bat -sv -work work sim/axi4_mem_model.sv sim/rankmlp_accel_tb.sv

Write-Host "Running xelab"
& D:/Xilinx/Vivado/2024.2/bin/xelab.bat -L unisims_ver -L unimacro_ver -L secureip -L work -s sim_snapshot work.rankmlp_accel_tb work.glbl

Write-Host "Running xsim"
try {
    & D:/Xilinx/Vivado/2024.2/bin/xsim.bat sim_snapshot -tclbatch sim/rankmlp_accel_tb.tcl -log xsim.log
} finally {
    Write-Host "Cleaning up copied .dat files"
    Get-ChildItem -Path "*.dat" | Remove-Item -Force
}
