/*
 * 01b_saxpy - 面试模式骨架。
 *
 * 自己补：
 * 1. saxpy_kernel 完整逻辑
 * 2. CPU reference saxpy_cpu
 * 3. main() 完整链路：分配 / 拷贝 / launch / 校验 / 释放
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy_kernel(const float* A, float a, float b, float* C, int n) {
    // TODO: 计算 C[i] = a * A[i] + b
}

void saxpy_cpu(const float* A, float a, float b, float* C, int n) {
    // TODO: CPU 版本，给 GPU 校验用
}

int main() {
    // TODO:
    // 1. const int N = 100000; size = N * sizeof(float)
    // 2. malloc host buffers, 初始化 h_A
    // 3. saxpy_cpu 算 reference
    // 4. cudaMalloc / cudaMemcpy
    // 5. blockSize=256, gridSize=(N+blockSize-1)/blockSize, launch kernel
    // 6. cudaMemcpy back, max_err 校验, printf 通过/失败
    // 7. cudaFree / free
    return 0;
}
