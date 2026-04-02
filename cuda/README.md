# CUDA 推理加速练习

本目录现在同时提供两种训练模式：

- `exercise.cu`：补函数模式。主流程已经写好，适合先熟悉 kernel 核心逻辑。
- `interview.cu`：面试模式。你需要自己补完整链路，只保留题目骨架和函数签名。
- `solution.cu`：参考实现。
- `test.py`：针对 `exercise.cu` / `solution.cu` 的自动化测试。

## 环境要求

```bash
# 直接编译运行某道题的 interview 模式
cd 01_vector_add
nvcc -O2 -o interview interview.cu && ./interview

# 运行补函数模式
nvcc -O2 -o solution solution.cu && ./solution

# Python 测试（仅 exercise / solution 使用）
pip install cupy-cuda12x pytest
pytest test.py -v
```

## 题目列表

| # | 题目 | 核心知识点 | 难度 |
|---|---|---|---|
| 01 | `vector_add` | 线程索引、基本 kernel、global memory | ⭐ |
| 02 | `reduce_sum` | shared memory 归约、两阶段 reduction | ⭐⭐ |
| 03 | `softmax` | 行级并行、数值稳定性、shared memory | ⭐⭐ |
| 04 | `layer_norm` | mean/var 归约、归一化、weight/bias | ⭐⭐ |
| 05 | `rms_norm` | sum(x^2) 归约、RMS 归一化 | ⭐⭐ |
| 06 | `silu_gelu` | 逐元素激活函数 | ⭐ |
| 07 | `rope` | 2D launch、三角函数、成对旋转 | ⭐⭐⭐ |
| 08 | `gemm` | tiling、shared memory、矩阵乘 | ⭐⭐⭐ |
| 09 | `fused_add_rmsnorm` | 算子融合、原地更新 | ⭐⭐⭐ |
| 10 | `flash_attention` | online softmax、分块 attention | ⭐⭐⭐⭐ |

## 面试模式怎么用

每个 `interview.cu` 都只保留以下骨架：

1. 题目对应的 kernel 函数签名。
2. 一个空的 CPU reference 函数签名。
3. 一个空的 `main()` 流程清单。

你需要自己补：

1. 问题规模和数据构造。
2. CPU reference。
3. `cudaMalloc` / `cudaMemcpy`。
4. block / grid 配置。
5. kernel 实现与 launch。
6. 结果回传、误差校验和资源释放。

## 推荐训练顺序

1. 先做 `01_vector_add`、`06_silu_gelu`，把 CUDA 基本链路写顺。
2. 再做 `02_reduce_sum`、`03_softmax`、`04_layer_norm`、`05_rms_norm`。
3. 然后做 `07_rope`、`08_gemm`、`09_fused_add_rmsnorm`。
4. 最后做 `10_flash_attention`。

## 使用建议

- 面试模式不会开箱即过，它的目的就是逼你从空骨架写到可运行。
- 如果你想先确认公式和接口，再切回 `exercise.cu` / `solution.cu` 对照。
- 建议每题先只做 correctness，再补性能优化。
