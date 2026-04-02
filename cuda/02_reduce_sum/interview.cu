#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
}

float reduce_sum_cpu(const float* input, int n) {
    return 0.0f;
}

int main() {

    return 0;
}
