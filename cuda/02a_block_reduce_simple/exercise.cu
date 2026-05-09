/*
 * 02a_block_reduce_simple - 单 block + shared memory 树形归约（简化版）
 *
 * 【核心概念】
 * - 这是 02_reduce_sum 的"无 grid 累加版"：固定 N == BLOCK_SIZE，只发一个 block
 * - shared memory: __shared__ float sdata[BLOCK_SIZE]，block 内的快速共享内存
 * - 树形归约：每轮把活跃线程数减半，log2(BLOCK_SIZE) 步出结果
 * - __syncthreads()：写完 shared memory 后必须先同步再读，否则数据竞争
 *
 * 【任务】
 * 实现 block_reduce_kernel：N = BLOCK_SIZE，单 block，求和到 output[0]
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ============================================================
// 练习：实现单 block 树形归约 kernel
// ============================================================

__global__ void block_reduce_kernel(const float* input, float* output, int n) {
    /*
     * 假设 n == BLOCK_SIZE，只发起一个 block。
     *
     * 提示：
     * 1. __shared__ float sdata[BLOCK_SIZE];
     * 2. int tid = threadIdx.x;
     * 3. sdata[tid] = (tid < n) ? input[tid] : 0.0f;
     * 4. __syncthreads();
     * 5. for (int s = blockDim.x / 2; s > 0; s >>= 1) {
     *        if (tid < s) sdata[tid] += sdata[tid + s];
     *        __syncthreads();
     *    }
     * 6. if (tid == 0) output[0] = sdata[0];
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

int main() {
    const int N = BLOCK_SIZE;
    const int size = N * sizeof(float);

    float h_input[BLOCK_SIZE];
    double cpu_sum = 0.0;
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 0.01f;
        cpu_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    block_reduce_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output, N);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU sum: %f, GPU sum: %f\n", (float)cpu_sum, gpu_sum);
    printf("误差: %e\n", fabsf(gpu_sum - (float)cpu_sum));
    printf("测试%s\n", fabsf(gpu_sum - (float)cpu_sum) < 1e-4f ? "通过 ✓" : "失败 ✗");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
