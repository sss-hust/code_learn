/*
 * 02_reduce_sum - CUDA 归约求和
 *
 * 【核心概念】
 * - Shared Memory: __shared__ 声明的 block 内共享快速内存
 * - 归约树: 每一步将数据量减半，log2(N) 步完成
 * - __syncthreads(): 同步 block 内所有线程
 * - Warp Shuffle: __shfl_down_sync() 无需 shared memory 的 warp 内归约
 *
 * 【任务】
 * 实现一个 CUDA kernel，对长度为 N 的数组求和
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ============================================================
// 练习：实现归约求和 kernel
// ============================================================

__global__ void
reduce_sum_kernel(const float *input, // 输入数组
                  float *output, // 输出（每个 block 输出一个部分和）
                  int n) {
  /*
   * 提示（Shared Memory 版本）：
   * 1. 声明 __shared__ float sdata[BLOCK_SIZE]
   * 2. 每个线程加载一个元素到 shared memory
   * 3. 归约循环: for (int s = blockDim.x/2; s > 0; s >>= 1)
   *    - if (tid < s) sdata[tid] += sdata[tid + s]
   *    - __syncthreads()
   * 4. 最后 tid==0 写入 output[blockIdx.x]
   *
   * 进阶提示（Warp Shuffle）：
   * - 最后 32 个元素可用 __shfl_down_sync 代替 shared memory
   */
  // TODO: 在此实现你的代码
  __shared__ float sdata[BLOCK_SIZE];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int index = i % blockDim.x;
  // gridDim.x 就是代码里的发起参数 gridSize
  // blockDim.x 就是 BLOCK_SIZE
  // 因此 gridDim.x * blockDim.x 就是【网格中所有发起的线程总数】
  // 有了这一步就不需要单独区分是第几次规约了，因为第一次在这里会拿到自己的数据
  // 第二次会拿到和别人数据的和
  float local_sum = 0.0f;
  for (int idx = i; idx < n; idx += gridDim.x * blockDim.x) {
    local_sum += input[idx];
  }
  sdata[index] = local_sum;

  // 极其重要：写完共享内存后必须立刻进行一次同步，然后再进入 for 循环折半！
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (index < s) {
      sdata[index] += sdata[index + s];
    }
    // 同步是等所有线程，如果放在if里面会导致死锁
    __syncthreads();
  }
  if (index == 0) {
    output[blockIdx.x] = sdata[index];
  }
}

// ============================================================
// 主函数
// ============================================================

int main() {
  const int N = 1000000;
  const int size = N * sizeof(float);

  float *h_input = (float *)malloc(size);
  srand(42);

  // 初始化为小值以避免浮点误差
  double cpu_sum = 0.0;
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)rand() / RAND_MAX * 0.01f;
    cpu_sum += h_input[i];
  }

  float *d_input, *d_output;
  cudaMalloc(&d_input, size);

  int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaMalloc(&d_output, gridSize * sizeof(float));

  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

  // 第一次归约
  reduce_sum_kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);
  // 第二次归约（将 gridSize 个部分和继续累加）
  float *d_final;
  cudaMalloc(&d_final, sizeof(float));
  reduce_sum_kernel<<<1, BLOCK_SIZE>>>(d_output, d_final, gridSize);

  float gpu_sum;
  cudaMemcpy(&gpu_sum, d_final, sizeof(float), cudaMemcpyDeviceToHost);

  printf("CPU sum: %f\n", (float)cpu_sum);
  printf("GPU sum: %f\n", gpu_sum);
  printf("误差: %e\n", fabs(gpu_sum - (float)cpu_sum));
  printf("测试%s\n",
         fabs(gpu_sum - (float)cpu_sum) < 0.1f ? "通过 ✓" : "失败 ✗");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_final);
  free(h_input);
  return 0;
}
