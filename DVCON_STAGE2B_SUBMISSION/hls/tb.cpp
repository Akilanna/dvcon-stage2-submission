#include "top.h"
#include "golden_vectors.h"
#include <iostream>
#include <cmath>

int main() {
    std::cout << "Starting C-Simulation..." << std::endl;

    const int n_objects = GOLDEN_N_OBJS;
    const ap_uint<32> task_id = 123;

    // Allocate simulated DDR memory buffers
    // 88 words/object * n_objects
    const int in_words = (IN_DIM / 4) * n_objects;
    ddr_word_t ddr_input[1760];
    ddr_word_t ddr_output[20];

    // Initialize inputs in simulated DDR (packed 64-bit format)
    for (int obj = 0; obj < n_objects; obj++) {
        for (int word_idx = 0; word_idx < IN_DIM / 4; word_idx++) {
            ddr_word_t word_val = 0;
            for (int e = 0; e < 4; e++) {
                int feat_idx = word_idx * 4 + e;
                float float_val = golden_inputs[obj][feat_idx];
                
                // Truncate to input_t (ap_fixed<16,6>)
                input_t q_val = (input_t)float_val;
                
                // Pack raw 16-bit integer bits
                ap_uint<16> raw_bits = q_val.range(15, 0);
                word_val.range((e + 1) * 16 - 1, e * 16) = raw_bits;
            }
            ddr_input[obj * (IN_DIM / 4) + word_idx] = word_val;
        }
    }

    // Initialize output buffer with sentinel values
    for (int i = 0; i < 20; i++) {
        ddr_output[i] = 0xDEADBEEFDEADBEEFULL;
    }

    ap_uint<32> perf_total_cycles = 0;
    ap_uint<16> perf_object_count = 0;

    // Run the HLS top-level function
    ranking_mlp_top(
        ddr_input,
        ddr_output,
        n_objects,
        task_id,
        perf_total_cycles,
        perf_object_count
    );

    // Verify and print the outputs
    int mismatches = 0;
    std::cout << std::endl << "--- Verification Results ---" << std::endl;
    for (int obj = 0; obj < n_objects; obj++) {
        ddr_word_t out_word = ddr_output[obj];
        
        // Unpack score from lower 16 bits
        act_t score;
        score.range(15, 0) = out_word.range(15, 0);
        float score_float = score.to_float();
        
        float float_ref = golden_outputs_float[obj];
        float fixed_ref = golden_outputs_fixed[obj];
        
        float diff = std::abs(score_float - fixed_ref);
        bool match = (diff < 1e-5);
        
        std::cout << "Object " << obj << ":" << std::endl;
        std::cout << "  Float Golden Ref: " << float_ref << std::endl;
        std::cout << "  Fixed Golden Ref: " << fixed_ref << std::endl;
        std::cout << "  Accelerator Out:  " << score_float << " (raw=" << std::hex << out_word << std::dec << ")" << std::endl;
        std::cout << "  Match Status:     " << (match ? "PASS" : "FAIL") << std::endl;
        
        if (!match) {
            mismatches++;
        }
    }

    std::cout << std::endl << "--- Telemetry & Performance ---" << std::endl;
    std::cout << "  perf_total_cycles: " << perf_total_cycles << std::endl;
    std::cout << "  perf_object_count: " << perf_object_count << std::endl;

    if (mismatches == 0) {
        std::cout << std::endl << "C-SIMULATION PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "C-SIMULATION FAILED! " << mismatches << " score mismatch(es)." << std::endl;
        return 1;
    }
}
