#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HEAD_DIM 64
#define BLOCK_N 32

__global__ void flash_attention_kernel(const float* Q,
                                       const float* K,
                                       const float* V,
                                       float* O,
                                       int seq_len,
                                       int head_dim,
                                       float scale) {
}

void attention_cpu(const float* Q,
                   const float* K,
                   const float* V,
                   float* O,
                   int seq_len,
                   int head_dim) {
}

int main() {
    return 0;
}
