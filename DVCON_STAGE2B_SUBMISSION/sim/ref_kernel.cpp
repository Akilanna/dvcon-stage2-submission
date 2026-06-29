#include <vector>
#include <cmath>
#include <algorithm>

// Pure-C++ reference implementation of the 3-layer MLP (352 -> 128 -> 64 -> 1)
void ref_ranking_mlp(const std::vector<float> &input_vec, float &out_score) {
    const int IN_DIM = 352;
    const int HID_DIM1 = 128;
    const int HID_DIM2 = 64;

    // Weight matrices and biases (initialized deterministically)
    static std::vector<float> fc1_w(HID_DIM1 * IN_DIM, 0.0f);
    static std::vector<float> fc1_b(HID_DIM1, 0.0f);
    static std::vector<float> fc2_w(HID_DIM2 * HID_DIM1, 0.0f);
    static std::vector<float> fc2_b(HID_DIM2, 0.0f);
    static std::vector<float> fc3_w(HID_DIM2, 0.0f);
    static float fc3_b = 0.0f;

    static bool weights_initialized = false;
    if (!weights_initialized) {
        // Initialize FC1 weights & biases
        for (int i = 0; i < HID_DIM1; ++i) {
            for (int j = 0; j < IN_DIM; ++j) {
                fc1_w[i * IN_DIM + j] = ((i + j) % 17 - 8) * 1e-3f;
            }
            fc1_b[i] = ((i % 5) - 2) * 1e-3f;
        }

        // Initialize FC2 weights & biases
        for (int i = 0; i < HID_DIM2; ++i) {
            for (int j = 0; j < HID_DIM1; ++j) {
                fc2_w[i * HID_DIM1 + j] = ((i + j) % 13 - 6) * 1e-3f;
            }
            fc2_b[i] = ((i % 7) - 3) * 1e-3f;
        }

        // Initialize FC3 weights & biases
        for (int i = 0; i < HID_DIM2; ++i) {
            fc3_w[i] = ((i % 9) - 4) * 1e-3f;
        }
        fc3_b = 0.005f;

        weights_initialized = true;
    }

    // Layer 1: FC1 + ReLU
    std::vector<float> h1(HID_DIM1, 0.0f);
    for (int i = 0; i < HID_DIM1; ++i) {
        float acc = fc1_b[i];
        for (int j = 0; j < IN_DIM; ++j) {
            acc += fc1_w[i * IN_DIM + j] * input_vec[j];
        }
        h1[i] = std::max(0.0f, acc);
    }

    // Layer 2: FC2 + ReLU
    std::vector<float> h2(HID_DIM2, 0.0f);
    for (int i = 0; i < HID_DIM2; ++i) {
        float acc = fc2_b[i];
        for (int j = 0; j < HID_DIM1; ++j) {
            acc += fc2_w[i * HID_DIM1 + j] * h1[j];
        }
        h2[i] = std::max(0.0f, acc);
    }

    // Layer 3: FC3 (linear)
    float acc = fc3_b;
    for (int i = 0; i < HID_DIM2; ++i) {
        acc += fc3_w[i] * h2[i];
    }
    out_score = acc;
}
