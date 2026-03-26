/*
 * 04_layer_norm - 参考答案
 *
 * 【关键点】
 * 1. 三遍扫描：先求 mean，再求 var，最后归一化
 * 2. stride loop 处理 cols > blockDim.x 的情况
 * 3. 每次归约后要 __syncthreads 和广播结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void layer_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows, int cols,
    float eps
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // Step 1: 求均值
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += row_in[c];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;
    __syncthreads();
    
    // Step 2: 求方差
    float local_var = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float d = row_in[c] - mean;
        local_var += d * d;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / cols;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();
    
    // Step 3: 归一化 + weight/bias
    for (int c = tid; c < cols; c += blockDim.x) {
        float x_norm = (row_in[c] - mean) * inv_std;
        row_out[c] = x_norm * weight[c] + bias[c];
    }
}

void layer_norm_cpu(const float* input, const float* weight, const float* bias,
                    float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float mean = 0, var = 0;
        for (int c = 0; c < cols; c++) mean += input[r * cols + c];
        mean /= cols;
        for (int c = 0; c < cols; c++) {
            float d = input[r * cols + c] - mean;
            var += d * d;
        }
        var /= cols;
        for (int c = 0; c < cols; c++) {
            float x_norm = (input[r * cols + c] - mean) / sqrtf(var + eps);
            output[r * cols + c] = x_norm * weight[c] + bias[c];
        }
    }
}

int main() {
    const int rows = 128, cols = 512;
    const float eps = 1e-5f;
    const int mat_size = rows * cols * sizeof(float);
    const int vec_size = cols * sizeof(float);
    
    float *h_input = (float*)malloc(mat_size);
    float *h_weight = (float*)malloc(vec_size);
    float *h_bias = (float*)malloc(vec_size);
    float *h_output = (float*)malloc(mat_size);
    float *h_ref = (float*)malloc(mat_size);
    
    srand(42);
    for (int i = 0; i < rows * cols; i++) h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < cols; i++) { h_weight[i] = 1.0f; h_bias[i] = 0.0f; }
    
    layer_norm_cpu(h_input, h_weight, h_bias, h_ref, rows, cols, eps);
    
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, mat_size); cudaMalloc(&d_weight, vec_size);
    cudaMalloc(&d_bias, vec_size); cudaMalloc(&d_output, mat_size);
    cudaMemcpy(d_input, h_input, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, vec_size, cudaMemcpyHostToDevice);
    
    layer_norm_kernel<<<rows, BLOCK_SIZE>>>(d_input, d_weight, d_bias, d_output, rows, cols, eps);
    cudaMemcpy(h_output, d_output, mat_size, cudaMemcpyDeviceToHost);
    
    float max_error = 0;
    for (int i = 0; i < rows * cols; i++) {
        float err = fabs(h_output[i] - h_ref[i]);
        if (err > max_error) max_error = err;
    }
    printf("最大误差: %e\n", max_error);
    printf("测试%s\n", max_error < 1e-4 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_bias); cudaFree(d_output);
    free(h_input); free(h_weight); free(h_bias); free(h_output); free(h_ref);
    return 0;
}
