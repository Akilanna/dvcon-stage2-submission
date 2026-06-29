#include "top.h"
#include "weights.h"

void ranking_mlp_top(
    ddr_word_t *ddr_input,
    ddr_word_t *ddr_output,
    ap_uint<16> n_objects,
    ap_uint<32> task_id,
    ap_uint<32> &perf_total_cycles,
    ap_uint<16> &perf_object_count
) {
    // Configure AXI Master port (M_AXI) matching CDAC VEGA SoC constraints
#pragma HLS INTERFACE m_axi port=ddr_input bundle=M_AXI depth=1760 offset=slave \
    max_read_burst_length=128 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=ddr_output bundle=M_AXI depth=20 offset=slave \
    max_write_burst_length=16 num_write_outstanding=4

    // Partition the weights for completely unrolled parallel access
#pragma HLS ARRAY_PARTITION variable=mlp_0_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=mlp_0_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=mlp_2_w complete dim=1
#pragma HLS ARRAY_PARTITION variable=mlp_2_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=mlp_4_w complete dim=2
#pragma HLS ARRAY_PARTITION variable=mlp_4_b complete dim=1

    // Configure AXI4-Lite Control port (control bundle)
    // Note: Vitis HLS 2024.2 auto-assigns register offsets (do not use offset=)
#pragma HLS INTERFACE s_axilite port=ddr_input bundle=control
#pragma HLS INTERFACE s_axilite port=ddr_output bundle=control
#pragma HLS INTERFACE s_axilite port=n_objects bundle=control
#pragma HLS INTERFACE s_axilite port=task_id bundle=control
#pragma HLS INTERFACE s_axilite port=perf_total_cycles bundle=control
#pragma HLS INTERFACE s_axilite port=perf_object_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Process each object sequentially to minimize hardware resource usage
    for (int obj = 0; obj < n_objects; obj++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=20 avg=14

        input_t input_vec[IN_DIM];
#pragma HLS ARRAY_PARTITION variable=input_vec cyclic factor=8 dim=1

        // Stage 1: Read 64-bit packed words from DDR3 and unpack to fixed-point
    read_input_loop:
        for (int word_idx = 0; word_idx < IN_DIM / 4; word_idx++) {
#pragma HLS PIPELINE II=1
            ddr_word_t val = ddr_input[obj * (IN_DIM / 4) + word_idx];
            for (int e = 0; e < 4; e++) {
#pragma HLS UNROLL
                ap_int<16> raw_val = val.range((e + 1) * 16 - 1, e * 16);
                input_t temp;
                temp.range(15, 0) = raw_val;
                input_vec[word_idx * 4 + e] = temp;
            }
        }

        // Stage 2: FC1 Layer (352 -> 128) + ReLU
        act_t h1[HID_DIM1];
#pragma HLS ARRAY_PARTITION variable=h1 complete dim=1

        ap_fixed<24,10> h1_acc[HID_DIM1];
#pragma HLS ARRAY_PARTITION variable=h1_acc complete dim=1

    fc1_init:
        for (int i = 0; i < HID_DIM1; i++) {
#pragma HLS UNROLL
            h1_acc[i] = (ap_fixed<24,10>)mlp_0_b[i];
        }

    fc1_loop:
        for (int j = 0; j < IN_DIM; j++) {
#pragma HLS PIPELINE II=1
            input_t xj = input_vec[j];
            for (int i = 0; i < HID_DIM1; i++) {
#pragma HLS UNROLL
                h1_acc[i] += (ap_fixed<24,10>)mlp_0_w[i][j] * (ap_fixed<24,10>)xj;
            }
        }

    fc1_relu:
        for (int i = 0; i < HID_DIM1; i++) {
#pragma HLS UNROLL
            h1[i] = (h1_acc[i] > (ap_fixed<24,10>)0) ? (act_t)h1_acc[i] : (act_t)0;
        }

        // Stage 3: FC2 Layer (128 -> 64) + ReLU
        act_t h2[HID_DIM2];
#pragma HLS ARRAY_PARTITION variable=h2 complete dim=1

        ap_fixed<24,10> h2_acc[HID_DIM2];
#pragma HLS ARRAY_PARTITION variable=h2_acc complete dim=1

    fc2_init:
        for (int i = 0; i < HID_DIM2; i++) {
#pragma HLS UNROLL
            h2_acc[i] = (ap_fixed<24,10>)mlp_2_b[i];
        }

    fc2_loop:
        for (int j = 0; j < HID_DIM1; j++) {
#pragma HLS PIPELINE II=1
            act_t hj = h1[j];
            for (int i = 0; i < HID_DIM2; i++) {
#pragma HLS UNROLL
                h2_acc[i] += (ap_fixed<24,10>)mlp_2_w[i][j] * (ap_fixed<24,10>)hj;
            }
        }

    fc2_relu:
        for (int i = 0; i < HID_DIM2; i++) {
#pragma HLS UNROLL
            h2[i] = (h2_acc[i] > (ap_fixed<24,10>)0) ? (act_t)h2_acc[i] : (act_t)0;
        }

        // Stage 4: FC3 Layer (64 -> 1) (Linear output)
        ap_fixed<24,10> out_acc = (ap_fixed<24,10>)mlp_4_b[0];

    fc3_loop:
        for (int j = 0; j < HID_DIM2; j++) {
#pragma HLS PIPELINE II=1
            out_acc += (ap_fixed<24,10>)mlp_4_w[0][j] * (ap_fixed<24,10>)h2[j];
        }

        act_t score = (act_t)out_acc;

        // Stage 5: Pack score into the lower 16 bits of a 64-bit word and write to DDR3
        ddr_word_t out_val = 0;
        out_val.range(15, 0) = score.range(15, 0);
        ddr_output[obj] = out_val;
    }

    // Set performance metrics
    perf_total_cycles = n_objects * 635; // Latency approximation (632 active pipeline steps + overhead)
    perf_object_count = n_objects;
}
