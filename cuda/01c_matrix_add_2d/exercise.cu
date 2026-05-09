/*
 * 01c_matrix_add_2d - CUDA 2D 矩阵逐元素相加
 *
 * 【核心概念】
 * - 第一次用 2D block / 2D grid：dim3 blockDim(16, 16) 表示一个 block 16x16 共 256 线程
 * - 双轴线程索引：row = blockIdx.y * blockDim.y + threadIdx.y, col = ...
 * - 行优先线性化：C[row * width + col]
 * - 注意 dim3 第一个参数对应 .x（列方向，width），第二个对应 .y（行方向，height）
 *
 * 【任务】
 * 实现 matrix_add_2d_kernel: C[row, col] = A[row, col] + B[row, col]
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================
// 练习：实现 2D 矩阵加法 kernel
// ============================================================

__global__ void matrix_add_2d_kernel(
    const float* A,
    const float* B,
    float* C,
    int height,
    int width
) {
    /*
     * 提示：
     * 1. int col = blockIdx.x * blockDim.x + threadIdx.x;
     * 2. int row = blockIdx.y * blockDim.y + threadIdx.y;
     * 3. if (row < height && col < width) C[row*width + col] = A[row*width + col] + B[row*width + col];
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

void matrix_add_cpu(const float* A, const float* B, float* C, int height, int width) {
    for (int i = 0; i < height * width; i++) C[i] = A[i] + B[i];
}

int main() {
    const int H = 257, W = 513;   // 故意选非整除尺寸，逼出边界检查
    const int N = H * W;
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
    matrix_add_cpu(h_A, h_B, h_ref, H, W);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    matrix_add_2d_kernel<<<grid, block>>>(d_A, d_B, d_C, H, W);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabs(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-5 ? "通过 ✓" : "失败 ✗");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
