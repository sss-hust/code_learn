/*
 * 10_flash_attention - CUDA Flash Attention
 *
 * 【核心概念】
 * - Flash Attention: 分块 Attention，不存储完整 N×N 矩阵
 * - 每个 block 处理 Q 的一行（一个 query token）
 * - 内循环沿 K/V 的 token 维度分块遍历
 * - 在线 softmax: 维护 running max 和 running sum
 * - 大量使用 shared memory 缓存 K/V 块
 *
 * 【任务】
 * 实现简化版 Flash Attention 的 CUDA kernel
 * 输入: Q(seq_len, head_dim), K(seq_len, head_dim), V(seq_len, head_dim)
 * 输出: O(seq_len, head_dim)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define HEAD_DIM 64
#define BLOCK_N 32  // K/V 每次处理的 token 数

// ============================================================
// 练习：实现 Flash Attention kernel（简化版）
// ============================================================

__global__ void flash_attention_kernel(
    const float* Q,    // (seq_len, head_dim)
    const float* K,    // (seq_len, head_dim)
    const float* V,    // (seq_len, head_dim)
    float* O,          // (seq_len, head_dim)
    int seq_len,
    int head_dim,
    float scale        // 1/sqrt(head_dim)
) {
    /*
     * grid: (seq_len,) — 每个 block 处理一个 query token
     * blockDim: (HEAD_DIM,) — 每个线程负责一个 head_dim 维度
     * 
     * 提示：
     * 1. 加载当前 query 行到寄存器: q[tid]
     * 2. 初始化: m = -inf, l = 0, acc = 0（每个线程一个 acc 值）
     * 3. 遍历 K/V 的每个块 (BLOCK_N 个 token):
     *    a. 将 K 块加载到 shared memory
     *    b. 计算 scores: 每个线程计算一个 q·k 的部分积，然后 warp/shared归约
     *    c. 更新在线 softmax 状态 (m, l)
     *    d. 将 V 块加载到 shared memory
     *    e. 累加 acc += p * V
     * 4. 归一化: O = acc / l
     *
     * 简化版思路（每线程处理一列）：
     * - 因为 head_dim 较小，每个线程可以在寄存器中保存 q 的一个元素
     * - scores 的计算需要跨线程归约（dot product）
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int i = 0; i < seq_len; i++) {
        // 计算 scores 和 softmax
        float max_score = -INFINITY;
        float* scores = (float*)malloc(seq_len * sizeof(float));
        for (int j = 0; j < seq_len; j++) {
            float s = 0;
            for (int d = 0; d < head_dim; d++)
                s += Q[i*head_dim+d] * K[j*head_dim+d];
            scores[j] = s * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        float sum = 0;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum += scores[j];
        }
        // 加权求和
        for (int d = 0; d < head_dim; d++) {
            float out = 0;
            for (int j = 0; j < seq_len; j++)
                out += (scores[j] / sum) * V[j*head_dim+d];
            O[i*head_dim+d] = out;
        }
        free(scores);
    }
}

int main() {
    const int seq_len = 128, head_dim = HEAD_DIM;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int size = seq_len * head_dim * sizeof(float);
    
    float *h_Q = (float*)malloc(size), *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size), *h_O = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    
    srand(42);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
    }
    
    attention_cpu(h_Q, h_K, h_V, h_ref, seq_len, head_dim);
    
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size); cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size); cudaMalloc(&d_O, size);
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
    
    flash_attention_kernel<<<seq_len, HEAD_DIM>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < seq_len * head_dim; i++) {
        float e = fabs(h_O[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-2 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    return 0;
}
