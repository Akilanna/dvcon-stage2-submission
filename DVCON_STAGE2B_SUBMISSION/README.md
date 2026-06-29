# CDAC VEGA SoC RankingMLP FPGA Hardware Accelerator (Stage 2B)

This directory contains the resource-optimized, AXI4-compliant hardware accelerator core and verification testbenches for the 3-layer RankingMLP head (`352 -> 128 -> 64 -> 1`) on the CDAC VEGA SoC.

---

## 1. Directory Structure

- **`hls/`**   : High-Level Synthesis sources (`top.cpp`, `top.h`, `weights.h`, `golden_vectors.h`, `tb.cpp`, and `run_hls.tcl`).
- **`rtl/`**   : Standalone synthesized Verilog RTL modules and ROM block memory weight tables (`.dat`).
- **`sim/`**   : SystemVerilog verification suite (`rankmlp_accel_tb.sv`, `axi4_mem_model.sv`), reference C++ code (`tb_ref.cpp`, `ref_kernel.cpp`), and tool simulation scripts.
- **`sw/`**    : Bare-metal software driver skeleton and AXI interface definitions.
- **`Makefile`**: Unified Makefile supporting C++ reference run, HLS synthesis, Vivado simulation, and Questa simulation.

---

## 2. AXI4-Lite Control Register Map
Base Address: `0x2006_0000` (Control bundle name: `control`)

| Address Offset | Register Name | Width | Access | Description |
| :--- | :--- | :---: | :---: | :--- |
| **`0x00`** | `AP_CTRL` | 32b | R/W | bit[0]=ap_start, bit[1]=ap_done, bit[2]=ap_idle, bit[3]=ap_ready |
| **`0x04`** | `GIE` | 32b | R/W | Global Interrupt Enable (bit[0]=1 to enable interrupts) |
| **`0x08`** | `IER` | 32b | R/W | IP Interrupt Enable (bit[0]=ap_done, bit[1]=ap_ready) |
| **`0x0C`** | `ISR` | 32b | R/W | IP Interrupt Status (TOW to clear) |
| **`0x10`** | `DDR_INPUT_LSB` | 32b | R/W | DDR3 input buffer base address [31:0] (LSB) |
| **`0x14`** | `DDR_INPUT_MSB` | 32b | R/W | DDR3 input buffer base address [63:32] (MSB) |
| **`0x1C`** | `DDR_OUTPUT_LSB`| 32b | R/W | DDR3 output buffer base address [31:0] (LSB) |
| **`0x20`** | `DDR_OUTPUT_MSB`| 32b | R/W | DDR3 output buffer base address [63:32] (MSB) |
| **`0x28`** | `N_OBJECTS` | 16b | R/W | Number of objects to process (batch size) |
| **`0x30`** | `TASK_ID` | 32b | R/W | Task/context identifier (used for telemetry) |
| **`0x38`** | `PERF_TOTAL_CYCLES`| 32b | RO | Telemetry: Total cycles elapsed (HLS clock) |
| **`0x48`** | `PERF_OBJECT_COUNT`| 16b | RO | Telemetry: Total objects processed |

---

## 3. Step-by-Step Reproduction Guide

### Prerequisites
- **Xilinx Vivado 2024.2** & **Vitis HLS 2024.2** installed and added to the system `PATH`.
- **Questa Sim** (Mentor/Siemens) installed and added to the `PATH` (only required for Questa simulation).
- **GCC / G++ compiler** (for compiling C++ reference simulation).

### Step 3.1: Run C++ Reference Simulation
To verify the fixed-point quantization behavior against the reference design:
```bash
# Compiles sim/tb_ref.cpp and runs verification
make sim_ref
```

### Step 3.2: Run High-Level Synthesis (HLS)
To rerun the synthesis, optimize the arrays, pipeline the layers, and repackage the Vivado IP core:
```bash
# Triggers vitis_hls run_hls.tcl inside the hls/ folder
make hls
```

### Step 3.3: Run RTL Simulation in Vivado Simulator
To compile the generated Verilog modules, mount the testbench and memory model, and run in Vivado `xsim`:
```bash
# Runs the automated powershell compilation and simulation script
make sim_vivado
```
Upon completion, the simulator terminal output displays:
```
[261610000 ns] --- Verification Results in RTL Simulation ---
  Object 0 score: -0.063477 (Expected: -0.063477)
  Object 1 score: 0.026367 (Expected:  0.026367)
  Object 2 score: 0.096680 (Expected:  0.096680)
  ...
[261610000 ns] RTL SIMULATION PASSED!
Performance Telemetry Registers:
  perf_total_cycles: 12700
  perf_object_count: 20
```

### Step 3.4: Run RTL Simulation in Questa Sim
To compile the design and execute verify in Mentor/Siemens Questa Sim:
```bash
# Compiles RTL, SV files and runs vsim in batch console mode
make sim_questa
```

---

## 4. Host Integration Guide
To run the accelerator in hardware:
1. Write inputs to the DDR3 memory layout. Inputs are pre-fused 352-dimensional float-to-fixed vectors. Each 64-bit word carries 4 packed 16-bit elements of type `ap_fixed<16,6>`.
2. Configure base pointers and batch parameters using offsets `0x10`, `0x1C`, `0x28`.
3. Kick off execution by writing `1` to `0x00`. Poll for completion (bit[1] = `1` of `0x00` or handle interrupt line).
4. Extract the raw MLP scores from output buffer and execute CPU post-processing (affordance and compatibility sums) to compute the final zero-shot rankings. (Reference driver logic is located at `sw/driver_skeleton.c`).
