/*
 * 09_fused_add_rmsnorm - 参考答案
 *
 * 【关键点】
 * 1. 先做 add 并写回 x（一次遍历同时完成 add + 计算 sum_sq）
 * 2. 归约 sum_sq 后再遍历一次做归一化
 * 3. 相比分开两个 kernel：节省了 x+res 的一次全局内存写和一次全局读
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void fused_add_rmsnorm_kernel(
    float* x,
    const float* residual,
    float* output,
    const float* weight,
    int rows, int cols,
    float eps
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* row_x = x + row * cols;
    const float* row_res = residual + row * cols;
    float* row_out = output + row * cols;
    
    // Step 1: Add + 计算 sum(x_new^2)
    float local_sum_sq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float x_new = row_x[c] + row_res[c];
        row_x[c] = x_new;  // 原地更新
        local_sum_sq += x_new * x_new;
    }
    sdata[tid] = local_sum_sq;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float rms_inv = rsqrtf(sdata[0] / cols + eps);
    __syncthreads();
    
    // Step 2: 归一化 + weight
    for (int c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_x[c] * rms_inv * weight[c];
    }
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
