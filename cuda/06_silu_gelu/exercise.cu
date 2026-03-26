/*
 * 06_silu_gelu - CUDA 激活函数
 *
 * 【核心概念】
 * - SiLU: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * - GELU (tanh近似): f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * - 逐元素操作，最简单的 kernel 模式
 *
 * 【任务】
 * 实现 SiLU 和 GELU 两个 CUDA kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int n) {
    /*
     * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     */
    // TODO: 在此实现你的代码
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    /*
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     */
    // TODO: 在此实现你的代码
}

void silu_cpu(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = in[i] / (1.0f + expf(-in[i]));
}

void gelu_cpu(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

int main() {
    const int N = 100000;
    const int size = N * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    float *h_silu = (float*)malloc(size), *h_gelu = (float*)malloc(size);
    float *h_silu_ref = (float*)malloc(size), *h_gelu_ref = (float*)malloc(size);
    
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 6;
    
    silu_cpu(h_in, h_silu_ref, N);
    gelu_cpu(h_in, h_gelu_ref, N);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size); cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    int block = 256, grid = (N + block - 1) / block;
    
    silu_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_silu, d_out, size, cudaMemcpyDeviceToHost);
    
    gelu_kernel<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_gelu, d_out, size, cudaMemcpyDeviceToHost);
    
    float silu_err = 0, gelu_err = 0;
    for (int i = 0; i < N; i++) {
        float e1 = fabs(h_silu[i] - h_silu_ref[i]); if (e1 > silu_err) silu_err = e1;
        float e2 = fabs(h_gelu[i] - h_gelu_ref[i]); if (e2 > gelu_err) gelu_err = e2;
    }
    
    printf("SiLU 最大误差: %e\n", silu_err);
    printf("GELU 最大误差: %e\n", gelu_err);
    printf("测试%s\n", (silu_err < 1e-5 && gelu_err < 1e-5) ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_silu); free(h_gelu); free(h_silu_ref); free(h_gelu_ref);
    return 0;
}
