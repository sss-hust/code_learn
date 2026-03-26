/*
 * 02_reduce_sum - 参考答案
 *
 * 【关键点】
 * 1. 用 shared memory 进行 block 内归约（归约树）
 * 2. 每一步 stride 减半: N→N/2→N/4→...→1
 * 3. __syncthreads() 确保所有线程完成当前步骤才进入下一步
 * 4. 最后 warp（32 threads）内可用 __shfl_down_sync 省掉 __syncthreads
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum_kernel(
    const float* input,
    float* output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载到 shared memory（越界的加载 0）
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    // 归约树
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后一个 warp 用 warp shuffle（不需要 __syncthreads）
    if (tid < 32) {
        // 先从 shared memory 合并到 warp
        volatile float* vmem = sdata;
        vmem[tid] += vmem[tid + 32];
        
        float val = vmem[tid];
        // Warp shuffle 归约
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

int main() {
    const int N = 1000000;
    const int size = N * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    srand(42);
    
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
    
    reduce_sum_kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);
    
    float *d_final;
    cudaMalloc(&d_final, sizeof(float));
    reduce_sum_kernel<<<1, BLOCK_SIZE>>>(d_output, d_final, gridSize);
    
    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("CPU sum: %f\n", (float)cpu_sum);
    printf("GPU sum: %f\n", gpu_sum);
    printf("误差: %e\n", fabs(gpu_sum - (float)cpu_sum));
    printf("测试%s\n", fabs(gpu_sum - (float)cpu_sum) < 0.1f ? "通过 ✓" : "失败 ✗");
    
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_final);
    free(h_input);
    return 0;
}
