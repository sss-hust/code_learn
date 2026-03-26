/*
 * 01_vector_add - 参考答案
 *
 * 【关键点】
 * 1. 全局线程索引 = blockIdx.x * blockDim.x + threadIdx.x
 * 2. 边界检查防止越界（N 不一定是 blockDim 整数倍）
 * 3. 这是最基础的 CUDA kernel 模式：一个线程处理一个元素
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    // 计算全局线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 100000;
    const int size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    
    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    vector_add_cpu(h_A, h_B, h_ref, N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabs(h_C[i] - h_ref[i]);
        if (err > max_error) max_error = err;
    }
    
    printf("最大误差: %e\n", max_error);
    printf("测试%s\n", max_error < 1e-5 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
