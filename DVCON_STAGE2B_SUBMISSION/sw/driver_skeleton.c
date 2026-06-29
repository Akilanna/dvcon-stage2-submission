/* Driver skeleton (bare-metal style) for CDAC VEGA SoC
 * Updated with exact register offsets auto-assigned by Vitis HLS 2024.2
 */

#include <stdint.h>
#include <stdio.h>

#define RANKER_BASE 0x20060000UL
#define REG_CONTROL      (RANKER_BASE + 0x00) // control signals (ap_start, ap_done, ap_idle, ap_ready)
#define REG_IN_LSB       (RANKER_BASE + 0x10) // ddr_input base address [31:0]
#define REG_IN_MSB       (RANKER_BASE + 0x14) // ddr_input base address [63:32]
#define REG_OUT_LSB      (RANKER_BASE + 0x1C) // ddr_output base address [31:0]
#define REG_OUT_MSB      (RANKER_BASE + 0x20) // ddr_output base address [63:32]
#define REG_NOBJECTS     (RANKER_BASE + 0x28) // number of objects to process (16-bit)
#define REG_TASKID       (RANKER_BASE + 0x30) // task identifier (32-bit)
#define REG_PERF_CYCLES  (RANKER_BASE + 0x38) // performance counter: clock cycles (read-only)
#define REG_PERF_OBJS    (RANKER_BASE + 0x48) // performance counter: processed objects (read-only)

static inline void write_reg(uint32_t addr, uint32_t val) {
    *((volatile uint32_t*)addr) = val;
}
static inline uint32_t read_reg(uint32_t addr) {
    return *((volatile uint32_t*)addr);
}

int main() {
    uint64_t in_addr = 0x80000000ULL; // DDR3 base address for inputs
    uint64_t out_addr = 0x80010000ULL; // DDR3 base address for outputs
    
    // 1. Configure the accelerator via AXI4-Lite
    write_reg(REG_IN_LSB, (uint32_t)(in_addr & 0xFFFFFFFF));
    write_reg(REG_IN_MSB, (uint32_t)(in_addr >> 32));
    write_reg(REG_OUT_LSB, (uint32_t)(out_addr & 0xFFFFFFFF));
    write_reg(REG_OUT_MSB, (uint32_t)(out_addr >> 32));
    write_reg(REG_NOBJECTS, 14); // Process 14 objects
    write_reg(REG_TASKID, 42);   // Task ID 42
    
    // 2. Start execution (write ap_start = 1)
    write_reg(REG_CONTROL, 1);

    // 3. Poll for completion (ap_done = bit 1)
    printf("Accelerator started. Polling for completion...\n");
    while (!(read_reg(REG_CONTROL) & 0x2)) {}
    
    // 4. Read performance counters
    uint32_t cycles = read_reg(REG_PERF_CYCLES);
    uint32_t objects = read_reg(REG_PERF_OBJS);
    
    printf("Done!\n");
    printf("Performance Telemetry:\n");
    printf("  Total Cycles:   %u\n", cycles);
    printf("  Object Count:   %u\n", objects);
    
    // 5. Host-Side Post-Processing
    // The accelerator outputs the raw MLP head logit. The host must add the 
    // affordance and task-class compatibility terms to compute the final score:
    //   score = logit + 0.6 * aff_score + 0.02 * compat
    //
    // Example host calculation for object 'obj':
    //
    // float logit = (float)((int16_t)(read_reg(out_addr + obj * 8) & 0xFFFF)) / 1024.0f;
    //
    // // Extract features directly from the host-prefused 352-dim input vector
    // // (assuming float input_vec[352] is available on host)
    // float support = input_vec[256];
    // float grasp   = input_vec[257];
    // float aff_score = 0.7f * grasp + 0.3f * support;
    //
    // float compat = 0.0f;
    // for (int i = 0; i < 64; i++) {
    //     float c_i = input_vec[64 + i];  // Class projection (64-D)
    //     float t_i = input_vec[192 + i]; // Task projection (64-D)
    //     compat += t_i * c_i;
    // }
    //
    // float final_score = logit + 0.6f * aff_score + 0.02f * compat;
    // printf("Object %d final ranking score: %f\n", obj, final_score);
    
    return 0;
}
