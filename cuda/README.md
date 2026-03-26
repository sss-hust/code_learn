# CUDA 推理加速函数练习

本目录包含 10 个 CUDA kernel 练习，覆盖推理加速中最常用的算子，按难度由浅入深排列。

## 环境要求

```bash
# 编译运行（需要 NVIDIA GPU 和 nvcc）
nvcc -o solution solution.cu && ./solution

# 或通过 Python 测试（需要 cupy）
pip install cupy-cuda12x pytest  # 根据 CUDA 版本选择 cupy 版本
```

## 练习列表

| # | 练习 | 核心知识点 | 难度 |
|---|------|----------|------|
| 01 | vector_add 向量加法 | kernel启动、线程索引、内存操作 | ⭐ |
| 02 | reduce_sum 归约求和 | shared memory、warp shuffle、归约树 | ⭐⭐ |
| 03 | softmax | 行级并行、shared memory归约 | ⭐⭐ |
| 04 | layer_norm 层归一化 | 均值/方差归约、可学习参数 | ⭐⭐ |
| 05 | rms_norm RMS归一化 | warp级归约、推理常用算子 | ⭐⭐ |
| 06 | silu_gelu 激活函数 | 逐元素kernel、内存带宽分析 | ⭐ |
| 07 | rope 旋转位置编码 | 2D线程索引、三角函数 | ⭐⭐⭐ |
| 08 | gemm 矩阵乘法 | 分块计算、shared memory tiling | ⭐⭐⭐ |
| 09 | fused_add_rmsnorm | 算子融合、减少global memory访问 | ⭐⭐⭐ |
| 10 | flash_attention | 分块Attention、shared memory管理 | ⭐⭐⭐⭐ |

## 使用方法

每个练习包含：
- `exercise.cu` — 填写 `// TODO` 标记处的代码
- `solution.cu` — 参考答案
- `test.py` — Python 自动测试（编译并运行 CUDA 代码对比 PyTorch 结果）

```bash
# 直接编译运行
cd 01_vector_add
nvcc -O2 -o solution solution.cu && ./solution

# pytest 自动测试
pytest test.py -v

# 测试参考答案
pytest test.py -v --check-solution
```

## CUDA 编程入门要点

```
Host (CPU)  ──调用──>  Device (GPU)
                        │
                    Grid (网格)
                    ├── Block 0
                    │   ├── Thread 0
                    │   ├── Thread 1
                    │   └── ...
                    ├── Block 1
                    └── ...

内存层级（速度递减）：
  Register > Shared Memory > L2 Cache > Global Memory
```
