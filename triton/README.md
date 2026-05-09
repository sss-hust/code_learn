# Triton 推理加速练习

本目录现在同时提供两种训练模式：

- `exercise.py`：补函数模式。wrapper 和测试已经写好，适合先熟悉 Triton kernel 核心逻辑。
- `interview.py`：面试模式。你需要自己补 kernel、wrapper、grid 配置和 `main()` 数据流。
- `solution.py`：参考实现。
- `test.py`：针对 `exercise.py` / `solution.py` 的自动化测试。

## 环境要求

```bash
pip install torch triton pytest

# 运行某道题的 interview 模式
cd 01_vector_add
python interview.py

# 运行补函数模式的测试
pytest test.py -v
```

## 题目列表

| # | 题目 | 核心知识点 | 难度 |
|---|---|---|---|
| 01 | `vector_add` | `program_id`、`load/store`、mask | ⭐ |
| 01b | `axpy` | 标量参数 (`a*x + b`)、复用 1D launch 模式 | ⭐ |
| 01c | `vector_sum` | 1D reduction、`tl.sum` + `tl.atomic_add` | ⭐⭐ |
| 01d | `row_max` | 2D 行级 launch、单 reduction、`other=-inf` | ⭐⭐ |
| 02 | `softmax` | 行级并行、数值稳定性、归约 | ⭐⭐ |
| 02b | `row_normalize` | 不带数值稳定的"reduction + 广播写回"（softmax 前一步） | ⭐⭐ |
| 03 | `layer_norm` | 均值/方差、逐行归一化 | ⭐⭐ |
| 04 | `rms_norm` | RMS 归一化 | ⭐⭐ |
| 05 | `silu_gelu` | 逐元素激活函数 | ⭐ |
| 06 | `rope` | cos/sin 频率表、成对旋转 | ⭐⭐⭐ |
| 07 | `online_softmax` | 单遍 softmax、running max/sum | ⭐⭐⭐ |
| 08 | `matrix_mul` | tile、`tl.dot`、矩阵乘 | ⭐⭐⭐ |
| 09 | `fused_add_rmsnorm` | residual add 与 RMSNorm 融合 | ⭐⭐⭐ |
| 10 | `flash_attention` | 分块 attention、online softmax | ⭐⭐⭐⭐ |

## 面试模式怎么用

每个 `interview.py` 都只保留以下骨架：

1. Triton kernel 函数签名。
2. Python wrapper 函数签名。
3. 一个空的 `main()` 流程清单。

你需要自己补：

1. 输入张量构造。
2. PyTorch reference。
3. `BLOCK_SIZE` / `BLOCK_M` / `BLOCK_N` 等元参数。
4. grid 配置。
5. Triton kernel 逻辑。
6. wrapper launch 与最终结果校验。

## 推荐训练顺序

1. 先做 `01_vector_add`、`01b_axpy`、`05_silu_gelu`，把"1D elementwise + load/store/mask"练成肌肉记忆。
2. 接着做 `01c_vector_sum`，第一次面对 reduction + atomic 写回。
3. 再做 `01d_row_max`、`02b_row_normalize`，把"每行一个 program + reduction + 写回"这套模板吃透。
4. 然后做 `02_softmax`、`03_layer_norm`、`04_rms_norm`，加上数值稳定与多次 reduction。
5. 接着 `06_rope`、`07_online_softmax`、`08_matrix_mul`、`09_fused_add_rmsnorm`。
6. 最后 `10_flash_attention`。

## 使用建议

- 面试模式故意不会直接跑通，它的目的是训练你从空骨架写到可验证结果。
- 如果你卡在公式或签名上，再回头参考 `exercise.py` / `solution.py`。
- 先把 correctness 写通，再考虑 autotune、向量化和性能参数。
