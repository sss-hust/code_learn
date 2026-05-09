/*
 * 01d_warp_reduce - 参考答案
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

__global__ void warp_reduce_kernel(const float* input, float* output) {
    int tid = threadIdx.x;
    float v = input[tid];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (tid == 0) *output = v;
}

int main() {
    const int N = WARP_SIZE;
    float h_input[WARP_SIZE];
    float cpu_sum = 0.0f;
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        cpu_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    warp_reduce_kernel<<<1, WARP_SIZE>>>(d_input, d_output);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU sum: %f, GPU sum: %f\n", cpu_sum, gpu_sum);
    printf("误差: %e\n", fabsf(gpu_sum - cpu_sum));
    printf("测试%s\n", fabsf(gpu_sum - cpu_sum) < 1e-4f ? "通过 ✓" : "失败 ✗");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
