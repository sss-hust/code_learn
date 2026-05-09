/*
 * 01d_warp_reduce - CUDA Warp 内归约（不用 shared memory）
 *
 * 【核心概念】
 * - 一个 warp = 32 个连续线程，硬件层面同步执行
 * - __shfl_down_sync(mask, val, offset)：让线程 t 拿到线程 (t+offset) 的 val
 *   不需要 __syncthreads，也不需要 shared memory
 * - 蝶式归约：offset 从 16 → 8 → 4 → 2 → 1，每一步把数据折半
 * - 这是后面"block-level reduce 最后那一档"通常会切到的写法，因此先单独练
 *
 * 【任务】
 * 实现 warp_reduce_kernel：固定 N=32，单 block 单 warp，把 32 个数加成 1 个
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

// ============================================================
// 练习：实现 warp 内归约 kernel
// ============================================================

__global__ void warp_reduce_kernel(const float* input, float* output) {
    /*
     * 提示：
     * 1. int tid = threadIdx.x;  // 0..31
     * 2. float v = input[tid];
     * 3. for (int offset = 16; offset > 0; offset /= 2)
     *        v += __shfl_down_sync(0xffffffff, v, offset);
     * 4. if (tid == 0) *output = v;  // 最终结果在 lane 0
     */
    // TODO: 在此实现你的代码
}

// ============================================================
// 主函数
// ============================================================

int main() {
    const int N = WARP_SIZE;
    const int size = N * sizeof(float);

    float h_input[WARP_SIZE];
    float cpu_sum = 0.0f;
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        cpu_sum += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    warp_reduce_kernel<<<1, WARP_SIZE>>>(d_input, d_output);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CPU sum: %f\n", cpu_sum);
    printf("GPU sum: %f\n", gpu_sum);
    printf("误差: %e\n", fabsf(gpu_sum - cpu_sum));
    printf("测试%s\n", fabsf(gpu_sum - cpu_sum) < 1e-4f ? "通过 ✓" : "失败 ✗");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
