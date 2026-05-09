/*
 * 01b_saxpy - CUDA SAXPY (y = a * x + b)
 *
 * 【核心概念】
 * - vector_add 的微升级：把"两个输入张量相加"换成"标量参数 + 一个输入张量"
 * - 标量 a, b 直接作为 kernel 的传值参数（不需要 cudaMemcpy）
 * - kernel 调用 ABI：标量按值拷贝，指针拷贝的是地址
 *
 * 【任务】
 * 实现 saxpy_kernel: C[i] = a * A[i] + b
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================
// 练习：实现 saxpy kernel
// ============================================================

__global__ void saxpy_kernel(
    const float* A,
    float a,           // 标量乘子
    float b,           // 标量偏置
    float* C,
    int n
) {
    /*
     * 提示：
     * 1. int i = blockIdx.x * blockDim.x + threadIdx.x;
     * 2. if (i < n) C[i] = a * A[i] + b;
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

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
