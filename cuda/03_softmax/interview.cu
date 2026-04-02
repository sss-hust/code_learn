#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
}

void softmax_cpu(const float* input, float* output, int rows, int cols) {
}

int main() {
    return 0;
}
