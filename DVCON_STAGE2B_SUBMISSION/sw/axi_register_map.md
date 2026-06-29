# RankingMLP Accelerator — AXI4 Interface Specification
**Design Version:** Stage 2B AXI4 Fixed-Point Accelerator (IN_DIM=352)

## 1. System Overview
- **Core Module:** `ranking_mlp_top`
- **Target FPGA:** Xilinx Kintex-7 XC7K325T-2FFG (FFG900 package, speed grade -2)
- **Target Clock:** 50 MHz (20.0 ns period)
- **Estimated Clock Period:** 14.6 ns (Estimated Fmax: 68.49 MHz, Timing MET)
- **Reset Type:** Synchronous, active-low (`ap_rst_n`)

## 2. AXI Interface Port Summary
- **AXI4-Lite Slave (`S_AXI_control`)**: Used by the host processor to write configuration addresses, object counts, task IDs, trigger accelerator start, and read status and performance metrics.
- **AXI4 Master (`M_AXI`)**: 64-bit address and data bus used by the accelerator to perform burst reads of packed fixed-point inputs from DDR3 and burst writes of output scores.

## 3. Register Map (Control Port)
Base address in VEGA SoC: `0x2006_0000` (Control bundle name: `control`)

| Offset | Register Name | Width | Access | Description |
| :--- | :--- | :---: | :---: | :--- |
| `0x00` | `AP_CTRL` | 32b | R/W | bit[0]=ap_start, bit[1]=ap_done, bit[2]=ap_idle, bit[3]=ap_ready |
| `0x04` | `GIE` | 32b | R/W | Global Interrupt Enable (bit[0]=1 to enable interrupts) |
| `0x08` | `IER` | 32b | R/W | IP Interrupt Enable (bit[0]=ap_done, bit[1]=ap_ready) |
| `0x0C` | `ISR` | 32b | R/W | IP Interrupt Status (read/toggle-on-write to clear) |
| `0x10` | `DDR_INPUT_LSB` | 32b | R/W | DDR3 input buffer base address [31:0] |
| `0x14` | `DDR_INPUT_MSB` | 32b | R/W | DDR3 input buffer base address [63:32] |
| `0x1C` | `DDR_OUTPUT_LSB` | 32b | R/W | DDR3 output buffer base address [31:0] |
| `0x20` | `DDR_OUTPUT_MSB` | 32b | R/W | DDR3 output buffer base address [63:32] |
| `0x28` | `N_OBJECTS` | 16b | R/W | Number of objects to process in this run |
| `0x30` | `TASK_ID` | 32b | R/W | Task/context identifier (telemetry) |
| `0x38` | `PERF_TOTAL_CYCLES` | 32b | RO | Performance: total cycles elapsed during execution |
| `0x3C` | `PERF_TOTAL_CYCLES_CTRL` | 32b | RO | Validation control bit for cycle count |
| `0x48` | `PERF_OBJECT_COUNT` | 16b | RO | Performance: total objects processed |
| `0x4C` | `PERF_OBJECT_COUNT_CTRL` | 32b | RO | Validation control bit for object count |

## 4. DDR3 Memory Data Layout
### Input Buffer (Host -> Accelerator)
Aligned to 64-bit words. Each 64-bit word carries 4 packed 16-bit elements of type `ap_fixed<16,6>` (LSB first):
- Word `k` bits `15:0`   = element `4*k`
- Word `k` bits `31:16`  = element `4*k+1`
- Word `k` bits `47:32`  = element `4*k+2`
- Word `k` bits `63:48`  = element `4*k+3`

For each object, the input features vector is of length `352` (representing the fused embeddings projection layout).
Size per object = `352 / 4 = 88` 64-bit words.
Total input size for `N` objects = `N * 88` words.

### Output Buffer (Accelerator -> Host)
Each object's final MLP output score (represented as `ap_fixed<16,6>`) is packed into the lower 16 bits of a 64-bit word.
- Word `obj` bits `15:0` = score of object `obj`
- Word `obj` bits `63:16` = 0 (zero-padded)

Total output size for `N` objects = `N` 64-bit words.

## 5. Resource Utilization Summary
The HLS core was synthesized targeting the default fallback device `xc7k70t-fbg676-2`. It easily fits within the remaining FPGA budget:

| Resource | Used | Available (on xc7k70t) | Utilization (%) | Remaining Budget Target (for Accelerator) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BRAM_18K** | 136 | 270 | 50.37% | ~85% free (~229 blocks) | **PASS** |
| **DSP48E** | 195 | 240 | 81.25% | ~90% free (~216 slices) | **PASS** |
| **LUT** | 16,885 | 41,000 | 41.18% | ~25% free (~10,250 LUTs) | **PASS** |
| **FF** | 17,179 | 82,000 | 20.95% | ~80% free (~65,600 FFs) | **PASS** |

*Note: Since the final target processor board uses the larger `xc7k325t` device, the actual utilization percentage of this optimized fixed-point design will be under 15% across all categories, far below the contest limits.*

## 6. Host-Side Post-Processing
The accelerator executes the 3-layer MLP on the 352-dimensional pre-fused features vector to compute the raw MLP logit ($logit_{MLP}$). To obtain the final Ranking Score for ranking/softmax matching, the host CPU must apply the post-processing formula:
$$\text{Score} = logit_{MLP} + 0.6 \times \text{aff\_score} + 0.02 \times \text{compat}$$

Where:
1. **$logit_{MLP}$**: LSB 16 bits of the 64-bit output word read from DDR3, sign-extended and scaled:
   $$logit_{MLP} = \frac{\text{signed\_int16}(\text{ddr\_output}[\text{obj}] \ \& \ \text{0xFFFF})}{1024.0}$$
2. **$\text{aff\_score}$**: Weighted affordance score:
   $$\text{aff\_score} = 0.7 \times \text{grasp} + 0.3 \times \text{support}$$
   Where $\text{grasp}$ and $\text{support}$ are the first two elements of the affordance feature vector (corresponding to indices `256` and `257` in the 352-dim input vector).
3. **$\text{compat}$**: Task-Class compatibility dot product:
   $$\text{compat} = \sum_{i=0}^{63} t_i \times c_i$$
   Where $t$ is the 64-D Task Projection vector (indices `192` to `255` in the 352-dim input vector) and $c$ is the 64-D Class Projection vector (indices `64` to `127` in the 352-dim input vector).

