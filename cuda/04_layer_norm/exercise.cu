/*
 * 04_layer_norm - CUDA Layer Normalization
 *
 * 【核心概念】
 * - LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 * - 每个 block 处理一行，需要两次归约（mean, variance）
 * - 和 Softmax 类似的模式，但多了 weight/bias 参数
 *
 * 【任务】
 * 实现 CUDA LayerNorm kernel
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void layer_norm_kernel(const float *input,  // (rows, cols)
                                  const float *weight, // (cols,)
                                  const float *bias,   // (cols,)
                                  float *output,       // (rows, cols)
                                  int rows, int cols, float eps) {
  /*
   * 提示：
   * Step 1: 求均值 (shared memory 归约)
   * Step 2: 求方差 (shared memory 归约)
   * Step 3: 归一化并应用 weight/bias
   */
  // TODO: 在此实现你的代码
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float *row_in = input + row * cols;
  float *row_out = output + row * cols;
  __shared__ float sdata[BLOCK_SIZE];
  float local_sum = 0;
  // 求部分和
  for (int c = tid; c < cols; c += BLOCK_SIZE) {
    local_sum += row_in[c];
  }
  sdata[tid] = local_sum;
  __syncthreads();
  // 规约求全部和
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float row_mean = sdata[0] / cols;
  local_sum = 0;
  // 求部分和-var
  for (int c = tid; c < cols; c += BLOCK_SIZE) {
    float d = row_in[c] - row_mean;
    local_sum += d * d;
  }
  sdata[tid] = local_sum;
  __syncthreads();
  // 规约求全部和
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float row_var = sdata[0] / cols;
  //   __syncthreads();
  for (int c = tid; c < cols; c += BLOCK_SIZE) {
    row_out[c] =
        (row_in[c] - row_mean) * rsqrtf(row_var + eps) * weight[c] + bias[c];
  }
}

void layer_norm_cpu(const float *input, const float *weight, const float *bias,
                    float *output, int rows, int cols, float eps) {
  for (int r = 0; r < rows; r++) {
    float mean = 0, var = 0;
    for (int c = 0; c < cols; c++)
      mean += input[r * cols + c];
    mean /= cols;
    for (int c = 0; c < cols; c++) {
      float d = input[r * cols + c] - mean;
      var += d * d;
    }
    var /= cols;
    for (int c = 0; c < cols; c++) {
      float x_norm = (input[r * cols + c] - mean) / sqrtf(var + eps);
      output[r * cols + c] = x_norm * weight[c] + bias[c];
    }
  }
}

int main() {
  const int rows = 128, cols = 512;
  const float eps = 1e-5f;
  const int mat_size = rows * cols * sizeof(float);
  const int vec_size = cols * sizeof(float);

  float *h_input = (float *)malloc(mat_size);
  float *h_weight = (float *)malloc(vec_size);
  float *h_bias = (float *)malloc(vec_size);
  float *h_output = (float *)malloc(mat_size);
  float *h_ref = (float *)malloc(mat_size);

  srand(42);
  for (int i = 0; i < rows * cols; i++)
    h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  for (int i = 0; i < cols; i++) {
    h_weight[i] = 1.0f;
    h_bias[i] = 0.0f;
  }

  layer_norm_cpu(h_input, h_weight, h_bias, h_ref, rows, cols, eps);

  float *d_input, *d_weight, *d_bias, *d_output;
  cudaMalloc(&d_input, mat_size);
  cudaMalloc(&d_weight, vec_size);
  cudaMalloc(&d_bias, vec_size);
  cudaMalloc(&d_output, mat_size);
  cudaMemcpy(d_input, h_input, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight, vec_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias, vec_size, cudaMemcpyHostToDevice);

  layer_norm_kernel<<<rows, BLOCK_SIZE>>>(d_input, d_weight, d_bias, d_output,
                                          rows, cols, eps);
  cudaMemcpy(h_output, d_output, mat_size, cudaMemcpyDeviceToHost);

  float max_error = 0;
  for (int i = 0; i < rows * cols; i++) {
    float err = fabs(h_output[i] - h_ref[i]);
    if (err > max_error)
      max_error = err;
  }
  printf("最大误差: %e\n", max_error);
  printf("测试%s\n", max_error < 1e-4 ? "通过 ✓" : "失败 ✗");

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_bias);
  cudaFree(d_output);
  free(h_input);
  free(h_weight);
  free(h_bias);
  free(h_output);
  free(h_ref);
  return 0;
}
