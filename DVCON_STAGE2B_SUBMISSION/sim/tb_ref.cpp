#include <iostream>
#include <vector>
#include "ref_kernel.cpp"

int main() {
    const int IN_DIM = 352;
    const int N = 3;
    for (int obj=0; obj<N; ++obj) {
        std::vector<float> in(IN_DIM);
        for (int i=0;i<IN_DIM;i++) in[i] = ((i+obj)%17 - 8) * 0.01f;
        float out;
        ref_ranking_mlp(in, out);
        std::cout << "obj="<<obj<<" score="<<out<<"\n";
    }
    return 0;
}
