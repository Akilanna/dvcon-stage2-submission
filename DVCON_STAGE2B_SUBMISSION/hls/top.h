#ifndef RANKING_MLP_TOP_H
#define RANKING_MLP_TOP_H

#include <ap_fixed.h>
#include <ap_int.h>

// MLP Architecture Dimensions
static const int IN_DIM = 352;
static const int HID_DIM1 = 128;
static const int HID_DIM2 = 64;
static const int OUT_DIM = 1;

// Fixed-point type definitions
typedef ap_fixed<8,2>   weight_t;
typedef ap_fixed<16,6>  input_t;
typedef ap_fixed<16,6>  act_t;
typedef ap_fixed<16,6>  bias_t;

// DDR interface word width is 64-bit
typedef ap_uint<64>     ddr_word_t;

void ranking_mlp_top(
    ddr_word_t *ddr_input,
    ddr_word_t *ddr_output,
    ap_uint<16> n_objects,
    ap_uint<32> task_id,
    ap_uint<32> &perf_total_cycles,
    ap_uint<16> &perf_object_count
);

#endif // RANKING_MLP_TOP_H
