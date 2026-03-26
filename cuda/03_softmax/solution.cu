/*
 * 03_softmax - 参考答案
 *
 * 【关键点】
 * 1. 每个 block 处理一行，blockIdx.x = 行索引
 * 2. 一个线程可能需要处理多个列（stride loop：tid, tid+blockDim, tid+2*blockDim, ...）
 * 3. 行内归约分为三步：max归约、sum归约、归一化
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // Step 1: 求行最大值
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, row_in[c]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    // shared memory 归约求 max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();
    
    // Step 2: 求 exp 之和
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += expf(row_in[c] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    // shared memory 归约求 sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();
    
    // Step 3: 归一化
    for (int c = tid; c < cols; c += blockDim.x) {
        row_out[c] = expf(row_in[c] - row_max) / row_sum;
    }
}

void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        
        float max_val = row_in[0];
        for (int c = 1; c < cols; c++)
            if (row_in[c] > max_val) max_val = row_in[c];
        
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum += row_out[c];
        }
        for (int c = 0; c < cols; c++)
            row_out[c] /= sum;
    }
}

int main() {
    const int rows = 128, cols = 512;
    const int size = rows * cols * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    
    srand(42);
    for (int i = 0; i < rows * cols; i++)
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    
    softmax_cpu(h_input, h_ref, rows, cols);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    softmax_kernel<<<rows, BLOCK_SIZE>>>(d_input, d_output, rows, cols);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float err = fabs(h_output[i] - h_ref[i]);
        if (err > max_error) max_error = err;
    }
    
    printf("最大误差: %e\n", max_error);
    printf("测试%s\n", max_error < 1e-5 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output); free(h_ref);
    return 0;
}
