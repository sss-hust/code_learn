/*
 * 01c_matrix_add_2d - 面试模式骨架。
 *
 * 自己补：
 * 1. matrix_add_2d_kernel：用 2D 索引（blockIdx.x/y, threadIdx.x/y）
 * 2. CPU reference
 * 3. main()：dim3 block(16,16); grid((W+15)/16, (H+15)/16); 注意 H/W 不一定是 16 的倍数
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_add_2d_kernel(const float* A, const float* B, float* C,
                                     int height, int width) {
    // TODO
}

void matrix_add_cpu(const float* A, const float* B, float* C, int height, int width) {
    // TODO
}

int main() {
    // TODO: 推荐用 H=257, W=513 让边界检查必须发挥作用
    return 0;
}
