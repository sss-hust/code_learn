/*
 * 05_rms_norm - CUDA RMS Normalization
 *
 * 【核心概念】
 * - RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
 * - 比 LayerNorm 简单：不需要减均值，没有 bias
 * - LLaMA 等模型的主流归一化
 *
 * 【任务】
 * 实现 CUDA RMSNorm kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int rows, int cols,
    float eps
) {
    /*
     * 提示：
     * 1. 每个 block 处理一行
     * 2. 求 sum(x^2)（shared memory 归约）
     * 3. rms = sqrt(sum_sq / cols + eps)
     * 4. output = (x / rms) * weight
     */
    // TODO: 在此实现你的代码
}

void rms_norm_cpu(const float* input, const float* weight, float* output,
                  int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum_sq = 0;
        for (int c = 0; c < cols; c++) sum_sq += input[r*cols+c] * input[r*cols+c];
        float rms = sqrtf(sum_sq / cols + eps);
        for (int c = 0; c < cols; c++)
            output[r*cols+c] = (input[r*cols+c] / rms) * weight[c];
    }
}

int main() {
    const int rows = 128, cols = 512;
    const float eps = 1e-6f;
    const int mat_size = rows * cols * sizeof(float);
    const int vec_size = cols * sizeof(float);
    
    float *h_in = (float*)malloc(mat_size), *h_w = (float*)malloc(vec_size);
    float *h_out = (float*)malloc(mat_size), *h_ref = (float*)malloc(mat_size);
    
    srand(42);
    for (int i = 0; i < rows*cols; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < cols; i++) h_w[i] = 1.0f;
    
    rms_norm_cpu(h_in, h_w, h_ref, rows, cols, eps);
    
    float *d_in, *d_w, *d_out;
    cudaMalloc(&d_in, mat_size); cudaMalloc(&d_w, vec_size); cudaMalloc(&d_out, mat_size);
    cudaMemcpy(d_in, h_in, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, vec_size, cudaMemcpyHostToDevice);
    
    rms_norm_kernel<<<rows, BLOCK_SIZE>>>(d_in, d_w, d_out, rows, cols, eps);
    cudaMemcpy(h_out, d_out, mat_size, cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < rows*cols; i++) {
        float e = fabs(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-4 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    free(h_in); free(h_w); free(h_out); free(h_ref);
    return 0;
}
