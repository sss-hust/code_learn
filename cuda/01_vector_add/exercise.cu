/*
 * 01_vector_add - CUDA 向量加法
 *
 * 【核心概念】
 * - __global__: 标记 GPU kernel 函数
 * - threadIdx.x, blockIdx.x, blockDim.x: 线程索引体系
 * - cudaMalloc / cudaMemcpy / cudaFree: GPU 内存管理
 * - <<<grid, block>>>: kernel 启动配置
 *
 * 【任务】
 * 实现一个 CUDA kernel，计算 C[i] = A[i] + B[i]
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// 练习：实现向量加法 kernel
// ============================================================

__global__ void vector_add_kernel(
    const float* A,     // 输入向量 A
    const float* B,     // 输入向量 B
    float* C,           // 输出向量 C
    int n               // 向量长度
) {
    /*
     * 提示：
     * 1. 计算全局线程索引: int i = blockIdx.x * blockDim.x + threadIdx.x;
     * 2. 边界检查: if (i < n)
     * 3. C[i] = A[i] + B[i]
     */
    // TODO: 在此实现你的代码
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
    return;
}

// ============================================================
// 主函数（已完成，无需修改）
// ============================================================

void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 100000;
    const int size = N * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    
    // 初始化
    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU 参考结果
    vector_add_cpu(h_A, h_B, h_ref, N);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 拷贝到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 启动 kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // 拷贝回 CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 验证
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabs(h_C[i] - h_ref[i]);
        if (err > max_error) max_error = err;
    }
    
    printf("最大误差: %e\n", max_error);
    printf("测试%s\n", max_error < 1e-5 ? "通过 ✓" : "失败 ✗");
    
    // 释放
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
