/*
 * 02a_block_reduce_simple - 面试模式骨架。
 *
 * 自己补：
 * 1. block_reduce_kernel：固定 N=BLOCK_SIZE=256
 *    - __shared__ float sdata[BLOCK_SIZE]
 *    - tid 把自己的元素装进 sdata
 *    - __syncthreads
 *    - for s = blockDim.x/2; s > 0; s >>= 1 → sdata[tid] += sdata[tid+s]
 *    - tid == 0 写到 output[0]
 * 2. main()：单 block，N=256，比较 cpu_sum 和 gpu_sum
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void block_reduce_kernel(const float* input, float* output, int n) {
    // TODO
}

int main() {
    // TODO: 准备 256 个小数（避免浮点累加误差），cpu_sum 累加，
    //       cudaMalloc / Memcpy，<<<1, 256>>> 发起，结果对比
    return 0;
}
