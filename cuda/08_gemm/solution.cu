/*
 * 08_gemm - 参考答案
 *
 * 【关键点】
 * 1. Shared memory tiling 将 global memory 访问量降低到 1/TILE
 * 2. 两次 __syncthreads：加载后同步 + 计算后同步
 * 3. 边界检查：M, N, K 不一定是 TILE 整数倍
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int tx = threadIdx.x;  // 列方向
    int ty = threadIdx.y;  // 行方向
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    
    float sum = 0.0f;
    
    // 沿 K 维度分块
    for (int k = 0; k < K; k += TILE) {
        // 协作加载 A 的 tile (TILE × TILE)
        if (row < M && (k + tx) < K)
            As[ty][tx] = A[row * K + k + tx];
        else
            As[ty][tx] = 0.0f;
        
        // 协作加载 B 的 tile
        if ((k + ty) < K && col < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // 计算当前 tile 的贡献
        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    // 写回
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

int main() {
    const int M = 256, N = 256, K = 256;
    
    float *h_A = (float*)malloc(M*K*sizeof(float));
    float *h_B = (float*)malloc(K*N*sizeof(float));
    float *h_C = (float*)malloc(M*N*sizeof(float));
    float *h_ref = (float*)malloc(M*N*sizeof(float));
    
    srand(42);
    for (int i = 0; i < M*K; i++) h_A[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < K*N; i++) h_B[i] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    
    gemm_cpu(h_A, h_B, h_ref, M, N, K);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_err = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabs(h_C[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-2 ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
