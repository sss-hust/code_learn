/*
 * 09_fused_add_rmsnorm - CUDA 融合 Add + RMSNorm
 *
 * 【核心概念】
 * - 融合 residual add 和 RMSNorm：一次读写完成两个操作
 * - x_new = x + residual
 * - output = (x_new / rms(x_new)) * weight
 * - 减少 global memory 访问次数
 *
 * 【任务】
 * 实现融合 Add + RMSNorm 的 CUDA kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void fused_add_rmsnorm_kernel(
    float* x,             // 输入/输出（原地更新为 x + residual）
    const float* residual,
    float* output,        // RMSNorm 输出
    const float* weight,
    int rows, int cols,
    float eps
) {
    /*
     * 提示：
     * 1. 加载 x 和 residual，计算 x_new = x + residual
     * 2. 原地更新 x（写回 x_new）
     * 3. 用 shared memory 归约求 sum(x_new^2)
     * 4. 计算 rms 并归一化: output = (x_new / rms) * weight
     */
    // TODO: 在此实现你的代码
}

void fused_add_rmsnorm_cpu(float* x, const float* res, float* out,
                           const float* w, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum_sq = 0;
        for (int c = 0; c < cols; c++) {
            x[r*cols+c] += res[r*cols+c];
            sum_sq += x[r*cols+c] * x[r*cols+c];
        }
        float rms = sqrtf(sum_sq / cols + eps);
        for (int c = 0; c < cols; c++)
            out[r*cols+c] = (x[r*cols+c] / rms) * w[c];
    }
}

int main() {
    const int rows = 128, cols = 512;
    const float eps = 1e-6f;
    const int mat_size = rows * cols * sizeof(float);
    const int vec_size = cols * sizeof(float);
    
    float *h_x = (float*)malloc(mat_size), *h_x_ref = (float*)malloc(mat_size);
    float *h_res = (float*)malloc(mat_size), *h_w = (float*)malloc(vec_size);
    float *h_out = (float*)malloc(mat_size), *h_ref = (float*)malloc(mat_size);
    
    srand(42);
    for (int i = 0; i < rows*cols; i++) {
        h_x[i] = h_x_ref[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
        h_res[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    }
    for (int i = 0; i < cols; i++) h_w[i] = 1.0f;
    
    fused_add_rmsnorm_cpu(h_x_ref, h_res, h_ref, h_w, rows, cols, eps);
    
    float *d_x, *d_res, *d_out, *d_w;
    cudaMalloc(&d_x, mat_size); cudaMalloc(&d_res, mat_size);
    cudaMalloc(&d_out, mat_size); cudaMalloc(&d_w, vec_size);
    cudaMemcpy(d_x, h_x, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, vec_size, cudaMemcpyHostToDevice);
    
    fused_add_rmsnorm_kernel<<<rows, BLOCK_SIZE>>>(d_x, d_res, d_out, d_w, rows, cols, eps);
    cudaMemcpy(h_out, d_out, mat_size, cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < rows*cols; i++) {
        float e = fabs(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-4 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_out); cudaFree(d_w);
    free(h_x); free(h_x_ref); free(h_res); free(h_w); free(h_out); free(h_ref);
    return 0;
}
