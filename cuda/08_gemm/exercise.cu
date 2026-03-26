/*
 * 08_gemm - CUDA 矩阵乘法 (GEMM)
 *
 * 【核心概念】
 * - C = A @ B, A(M,K), B(K,N), C(M,N)
 * - Tiling: 将矩阵分块，利用 shared memory 缓存子矩阵
 * - 每个 block 负责 C 的一个 (TILE_M, TILE_N) 块
 * - 内循环沿 K 维度: 每次加载 (TILE_M, TILE_K) 和 (TILE_K, TILE_N) 到 shared memory
 *
 * 【任务】
 * 实现一个 tiled GEMM kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 32

// ============================================================
// 练习：实现 Tiled GEMM kernel
// ============================================================

__global__ void gemm_kernel(
    const float* A,   // (M, K)
    const float* B,   // (K, N)
    float* C,         // (M, N)
    int M, int N, int K
) {
    /*
     * 提示：
     * 1. 声明 __shared__ float As[TILE][TILE], Bs[TILE][TILE]
     * 2. 计算当前线程负责 C 的 (row, col) 位置
     * 3. 沿 K 维度分块循环 (k = 0; k < K; k += TILE)：
     *    a. 加载 A 的 tile 到 As  (协作加载，每线程加载一个元素)
     *    b. 加载 B 的 tile 到 Bs
     *    c. __syncthreads()
     *    d. 内层循环累加: sum += As[ty][i] * Bs[i][tx]
     *    e. __syncthreads()
     * 4. 写回 C[row][col] = sum
     */
    // TODO: 在此实现你的代码
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
