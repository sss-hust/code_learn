#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void rms_norm_kernel(const float* input,
                                const float* weight,
                                float* output,
                                int rows,
                                int cols,
                                float eps) {
}

void rms_norm_cpu(const float* input,
                  const float* weight,
                  float* output,
                  int rows,
                  int cols,
                  float eps) {
}

int main() {
    return 0;
}
