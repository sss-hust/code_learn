/*
 * 01c_matrix_add_2d - 参考答案
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_add_2d_kernel(const float* A, const float* B, float* C,
                                     int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

void matrix_add_cpu(const float* A, const float* B, float* C, int height, int width) {
    for (int i = 0; i < height * width; i++) C[i] = A[i] + B[i];
}

int main() {
    const int H = 257, W = 513;
    const int N = H * W;
    const int size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    matrix_add_cpu(h_A, h_B, h_ref, H, W);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    matrix_add_2d_kernel<<<grid, block>>>(d_A, d_B, d_C, H, W);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabs(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("最大误差: %e\n", max_err);
    printf("测试%s\n", max_err < 1e-5 ? "通过 ✓" : "失败 ✗");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
