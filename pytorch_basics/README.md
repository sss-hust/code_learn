# PyTorch 基础练习

这个目录是 `model_layers/` 之前的 PyTorch 入门梯子。目标不是覆盖整个 PyTorch，
而是把"读懂 / 写出 model_layers 那种 `nn.Module` 子类"所需要的基础打通：

1. 张量怎么造、怎么算、怎么 reshape。
2. 广播、reduction、matmul、einsum 这些 attention/MLP 里反复用到的操作。
3. autograd 怎么跑通、`nn.Module` 怎么注册参数和 buffer。
4. 一个最小训练循环长什么样。

## 文件说明

每道题都有：

- `interview.py`：骨架。函数签名、shape 说明和 `main()` 清单留空，自己补完。
- `solution.py`：参考答案，可直接 `python solution.py` 跑。
- `test.py`：pytest 自动测试，默认测 `interview.py`，加 `--check-solution` 测参考答案。

## 题目列表

| # | 题目 | 核心知识点 | 难度 |
|---|---|---|---|
| 01 | `tensor_basics` | `zeros/ones/arange/randn`、`dtype`、`device`、`.to/.cpu/.cuda`、`.item` | ⭐ |
| 02 | `indexing_slicing` | 切片、bool mask、advanced indexing、`gather`、`masked_fill`、`where` | ⭐ |
| 03 | `broadcasting` | 广播规则、`expand` vs `repeat`、`unsqueeze`/`squeeze` | ⭐ |
| 04 | `reduction_ops` | `sum/mean/max/argmax` 沿 `dim`、`keepdim`、`softmax`、`logsumexp` | ⭐⭐ |
| 05 | `reshape_view_permute` | `view` vs `reshape`、`contiguous`、`transpose`、`permute` | ⭐⭐ |
| 06 | `matmul_einsum` | `matmul`、`bmm`、`einsum`（含 `q @ k^T` 注意力得分写法） | ⭐⭐ |
| 07 | `autograd_basics` | `requires_grad`、`backward`、`.grad`、`no_grad`、`detach` | ⭐⭐ |
| 08 | `nn_module_basics` | 子类化 `nn.Module`、`Parameter`、`register_buffer`、`train/eval` | ⭐⭐ |
| 09 | `linear_by_hand` | 手写 `Linear`，与 `nn.Linear` 数值对齐 | ⭐⭐ |
| 10 | `minimal_training_loop` | 最小 MLP + MSE + SGD 完整训练循环 | ⭐⭐⭐ |

## 推荐学习顺序

1. **01 → 03**：先把张量怎么造、怎么取、怎么广播这一层打通。
2. **04 → 06**：reduction、reshape、matmul，这一组是 attention/FFN 反复用到的"动词"。
3. **07 → 08**：autograd 跟 `nn.Module` 是写 `model_layers/` 的前置技能。
4. **09 → 10**：手写一个 `Linear` 和最小训练循环，把"前面学的东西"串成一个能跑的东西。

完成 10 题后再做 `model_layers/01_embedding`，会无缝对接。

## 运行方式

```bash
cd pytorch_basics/01_tensor_basics

# 跑你自己写的代码
python interview.py

# 跑 pytest 校验 interview.py
pytest test.py -v

# 跑 pytest 校验参考答案
pytest test.py -v --check-solution
```

## 设计说明

- 全部用 CPU 也能跑通，不强依赖 GPU；GPU 题目会显式标注。
- 每题 `main()` 都至少打印一次 shape 和一次数值对照，养成"先打 shape 再校验"的习惯。
- 测试只校验"功能等价"，不限制你怎么写（比如 `softmax` 你可以手写也可以调 `F.softmax`），重点是搞清楚每个 API 的语义。
