/*
 * 01d_warp_reduce - 面试模式骨架。
 *
 * 自己补：
 * 1. warp_reduce_kernel：用 __shfl_down_sync 做蝶式 reduce，不要用 shared memory
 * 2. main()：N 固定 32，单 block 单 warp
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

__global__ void warp_reduce_kernel(const float* input, float* output) {
    // TODO: __shfl_down_sync 蝶式 reduce, lane 0 写到 *output
}

int main() {
    // TODO: 准备 32 个随机数，cpu 求 sum，
    //       发起 <<<1, 32>>> kernel，比较 gpu_sum 和 cpu_sum
    return 0;
}
