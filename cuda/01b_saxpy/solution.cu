/*
 * 01b_saxpy - 参考答案
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy_kernel(const float* A, float a, float b, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = a * A[i] + b;
}

void saxpy_cpu(const float* A, float a, float b, float* C, int n) {
    for (int i = 0; i < n; i++) C[i] = a * A[i] + b;
}

int main() {
    const int N = 100000;
    const int size = N * sizeof(float);
    const float a = 2.5f, b = -0.5f;

    float *h_A = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    srand(42);
    for (int i = 0; i < N; i++) h_A[i] = (float)rand() / RAND_MAX;
    saxpy_cpu(h_A, a, b, h_ref, N);

    float *d_A, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    saxpy_kernel<<<gridSize, blockSize>>>(d_A, a, b, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabs(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-5 ? "通过 ✓" : "失败 ✗");

    cudaFree(d_A); cudaFree(d_C);
    free(h_A); free(h_C); free(h_ref);
    return 0;
}
