/*
 * 10_flash_attention - 参考答案（简化版）
 *
 * 【关键点】
 * 1. 每个 block 处理一个 query token（grid = seq_len）
 * 2. 每个线程负责 output 的一个维度（blockDim = head_dim）
 * 3. K/V 分块加载到 shared memory
 * 4. scores 的 dot product 通过 shared memory 归约完成
 * 5. 在线 softmax 避免了存储 N×N 矩阵
 *
 * 注意：这是教学简化版。生产级 Flash Attention 的优化策略更复杂:
 * - 更高效的分块策略
 * - warp-level primitives
 * - 寄存器复用
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define HEAD_DIM 64
#define BLOCK_N 32

__global__ void flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim,
    float scale
) {
    int query_idx = blockIdx.x;   // 当前 query token
    int tid = threadIdx.x;        // 维度索引
    
    // 加载当前 query 到寄存器
    float q_val = (tid < head_dim) ? Q[query_idx * head_dim + tid] : 0.0f;
    
    // Shared memory
    __shared__ float s_k[BLOCK_N * HEAD_DIM];   // K 块
    __shared__ float s_scores[BLOCK_N];          // attention scores
    __shared__ float s_v[BLOCK_N * HEAD_DIM];   // V 块
    __shared__ float s_dot[HEAD_DIM];            // 临时 dot product
    
    // 在线 softmax 状态
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;  // 每个线程维护一个维度的累加
    
    // 遍历 K/V 的所有块
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
        int block_size = min(BLOCK_N, seq_len - kv_start);
        
        // 协作加载 K 块到 shared memory
        for (int i = tid; i < block_size * head_dim; i += blockDim.x) {
            int token = i / head_dim;
            int dim = i % head_dim;
            s_k[token * head_dim + dim] = K[(kv_start + token) * head_dim + dim];
        }
        __syncthreads();
        
        // 计算 scores: q · k_j for each k_j in block
        for (int j = 0; j < block_size; j++) {
            // 每个线程计算 dot product 的一个分量
            s_dot[tid] = (tid < head_dim) ? q_val * s_k[j * head_dim + tid] : 0.0f;
            __syncthreads();
            
            // 归约求 dot product
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && tid + s < head_dim) {
                    s_dot[tid] += s_dot[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                s_scores[j] = s_dot[0] * scale;
            }
            __syncthreads();
        }
        
        // 在线 softmax 更新
        // 找当前块的 max
        float block_max = -INFINITY;
        for (int j = 0; j < block_size; j++) {
            if (s_scores[j] > block_max) block_max = s_scores[j];
        }
        
        float new_max = fmaxf(running_max, block_max);
        
        // 修正历史值
        float alpha = expf(running_max - new_max);
        acc *= alpha;
        running_sum *= alpha;
        
        // 协作加载 V 块
        for (int i = tid; i < block_size * head_dim; i += blockDim.x) {
            int token = i / head_dim;
            int dim = i % head_dim;
            s_v[token * head_dim + dim] = V[(kv_start + token) * head_dim + dim];
        }
        __syncthreads();
        
        // 累加 p * V
        if (tid < head_dim) {
            for (int j = 0; j < block_size; j++) {
                float p = expf(s_scores[j] - new_max);
                acc += p * s_v[j * head_dim + tid];
                if (tid == 0) {
                    running_sum += p;
                }
            }
        }
        
        // 广播 running_sum（只有 tid==0 更新了）
        __syncthreads();
        if (tid == 0) {
            s_dot[0] = running_sum;
        }
        __syncthreads();
        running_sum = s_dot[0];
        running_max = new_max;
        __syncthreads();
    }
    
    // 归一化并写回
    if (tid < head_dim) {
        O[query_idx * head_dim + tid] = acc / running_sum;
    }
}

void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int i = 0; i < seq_len; i++) {
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
