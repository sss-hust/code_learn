/*
 * 03_softmax - CUDA Softmax
 *
 * 【核心概念】
 * - 行级并行：每个 block 处理一行
 * - Shared memory 用于行内归约（max、sum）
 * - 数值稳定性：先减 max 再 exp
 *
 * 【任务】
 * 实现 CUDA softmax kernel，对 (rows, cols) 矩阵每一行做 softmax
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ============================================================
// 练习：实现 Softmax kernel
// ============================================================

__global__ void softmax_kernel(
    const float* input,    // (rows, cols)
    float* output,         // (rows, cols)
    int rows,
    int cols
) {
    /*
     * 每个 block 处理一行（blockIdx.x = 行索引）
     * 每个线程可负责多个列（当 cols > blockDim.x 时）
     *
     * 提示：
     * Step 1: 找行最大值（shared memory 归约）
     *   - 每个线程遍历自己负责的列，求局部 max
     *   - shared memory 归约得到全局 max
     *
     * Step 2: 求 exp 之和
     *   - 每个线程计算 sum(exp(x - max))
     *   - shared memory 归约得到总 sum
     *
     * Step 3: 归一化
     *   - output[i] = exp(input[i] - max) / sum
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

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
    
    // 每个 block 处理一行
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
