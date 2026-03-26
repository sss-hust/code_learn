/*
 * 07_rope - CUDA 旋转位置编码 (RoPE)
 *
 * 【核心概念】
 * - 对 (x0, x1) 对应用旋转矩阵:
 *     x0' = x0 * cos(θ*pos) - x1 * sin(θ*pos)
 *     x1' = x0 * sin(θ*pos) + x1 * cos(θ*pos)
 * - θ_i = 1 / base^(2i/d), 其中 base=10000
 * - 2D 线程索引: (token, dim_pair)
 *
 * 【任务】
 * 实现 CUDA RoPE kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BASE 10000.0f

__global__ void rope_kernel(
    const float* input,   // (seq_len, head_dim)
    float* output,        // (seq_len, head_dim)
    int seq_len,
    int head_dim
) {
    /*
     * 提示：
     * 1. 2D 索引: pos = blockIdx.x * blockDim.x + threadIdx.x (token位置)
     *            i = blockIdx.y * blockDim.y + threadIdx.y (维度对索引)
     * 2. 计算角度: theta_i = 1 / base^(2i/head_dim)
     *            angle = pos * theta_i
     * 3. 加载 x0 = input[pos, 2*i], x1 = input[pos, 2*i+1]
     * 4. 旋转并写回
     */
    // TODO: 在此实现你的代码
}

void rope_cpu(const float* input, float* output, int seq_len, int head_dim) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float theta = 1.0f / powf(BASE, 2.0f * i / head_dim);
            float angle = pos * theta;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            int idx0 = pos * head_dim + 2 * i;
            int idx1 = pos * head_dim + 2 * i + 1;
            output[idx0] = input[idx0] * cos_a - input[idx1] * sin_a;
            output[idx1] = input[idx0] * sin_a + input[idx1] * cos_a;
        }
    }
}

int main() {
    const int seq_len = 128, head_dim = 128;
    const int size = seq_len * head_dim * sizeof(float);
    
    float *h_in = (float*)malloc(size), *h_out = (float*)malloc(size), *h_ref = (float*)malloc(size);
    srand(42);
    for (int i = 0; i < seq_len * head_dim; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    
    rope_cpu(h_in, h_ref, seq_len, head_dim);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size); cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((seq_len + 15) / 16, (head_dim / 2 + 15) / 16);
    rope_kernel<<<grid, block>>>(d_in, d_out, seq_len, head_dim);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < seq_len * head_dim; i++) {
        float e = fabs(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-4 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
