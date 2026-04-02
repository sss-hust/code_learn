#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 32

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
}

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
}

int main() {
    return 0;
}
