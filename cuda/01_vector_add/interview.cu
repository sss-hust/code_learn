#include <__clang_cuda_builtin_vars.h>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add_kernel(const float *A, const float *B, float *C,
                                  int n) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid < n) {
    C[pid] = A[pid] + B[pid];
  }
}

void vector_add_cpu(const float *A, const float *B, float *C, int n) {

  for (int i = 0; i < n; i++) {
    C[i] = A[i] + B[i];
  }
}

int main() {

  int N = 100000;
  int size = N * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  float *h_ref = (float *)malloc(size);

  srand(42);
  for (int i = 0; i < N; i++) {
    h_A[i] = (float)rand() / RAND_MAX;
    h_B[i] = (float)rand() / RAND_MAX;
  }

  vector_add_cpu(h_A, h_B, h_ref, N);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;
  vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  return 0;
}
