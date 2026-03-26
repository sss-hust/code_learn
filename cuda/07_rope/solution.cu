/*
 * 07_rope - 参考答案
 *
 * 【关键点】
 * 1. 2D grid: x维度对应 token 位置, y维度对应 head_dim 的维度对
 * 2. 每个线程处理一对 (x0, x1)
 * 3. powf 和 sincosf 在 GPU 上有硬件加速
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BASE 10000.0f

__global__ void rope_kernel(
    const float* input,
    float* output,
    int seq_len,
    int head_dim
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;  // token 位置
    int i = blockIdx.y * blockDim.y + threadIdx.y;     // 维度对索引
    
    if (pos >= seq_len || i >= head_dim / 2) return;
    
    // 计算旋转角度
    float theta = 1.0f / powf(BASE, 2.0f * i / head_dim);
    float angle = pos * theta;
    float cos_a, sin_a;
    sincosf(angle, &sin_a, &cos_a);
    
    // 加载一对元素
    int idx0 = pos * head_dim + 2 * i;
    int idx1 = pos * head_dim + 2 * i + 1;
    float x0 = input[idx0];
    float x1 = input[idx1];
    
    // 旋转
    output[idx0] = x0 * cos_a - x1 * sin_a;
    output[idx1] = x0 * sin_a + x1 * cos_a;
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
