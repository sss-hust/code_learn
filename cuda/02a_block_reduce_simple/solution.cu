/*
 * 02a_block_reduce_simple - 参考答案
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void block_reduce_kernel(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    sdata[tid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[0] = sdata[0];
}

int main() {
    const int N = BLOCK_SIZE;
    const int size = N * sizeof(float);

    float h_input[BLOCK_SIZE];
    double cpu_sum = 0.0;
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 0.01f;
        cpu_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    block_reduce_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output, N);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU sum: %f, GPU sum: %f\n", (float)cpu_sum, gpu_sum);
    printf("误差: %e\n", fabsf(gpu_sum - (float)cpu_sum));
    printf("测试%s\n", fabsf(gpu_sum - (float)cpu_sum) < 1e-4f ? "通过 ✓" : "失败 ✗");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
