# Triton 推理加速函数练习

本目录包含 10 个 Triton kernel 练习，覆盖推理加速中最常用的算子，按难度由浅入深排列。

## 环境要求

```bash
pip install triton torch pytest
```

## 练习列表

| # | 练习 | 核心知识点 | 难度 |
|---|------|----------|------|
| 01 | vector_add 向量加法 | grid、program_id、load/store、mask | ⭐ |
| 02 | softmax | 行级并行、在线max-exp-sum、数值稳定性 | ⭐⭐ |
| 03 | layer_norm 层归一化 | 均值/方差计算、归一化、可学习参数 | ⭐⭐ |
| 04 | rms_norm RMS归一化 | 简化版LayerNorm、推理主流归一化 | ⭐⭐ |
| 05 | silu_gelu 激活函数 | 逐元素运算、SiLU/GELU公式 | ⭐ |
| 06 | rope 旋转位置编码 | 复数旋转、频率计算、成对元素 | ⭐⭐⭐ |
| 07 | online_softmax 在线Softmax | 单遍softmax、FlashAttention前置 | ⭐⭐⭐ |
| 08 | matrix_mul 矩阵乘法 | 分块计算、tl.dot、tiling | ⭐⭐⭐ |
| 09 | fused_add_rmsnorm | 算子融合、residual连接 | ⭐⭐⭐ |
| 10 | flash_attention | 分块Attention、在线Softmax | ⭐⭐⭐⭐ |

## 使用方法

每个练习包含三个文件：
- `exercise.py` — 填写 `# TODO` 标记处的代码
- `solution.py` — 参考答案
- `test.py` — 自动测试

```bash
# 测试你的实现
cd 01_vector_add
pytest test.py -v

# 测试参考答案
pytest test.py -v --check-solution

# 运行全部测试
cd d:\学习\code_learn\triton
pytest -v
```
