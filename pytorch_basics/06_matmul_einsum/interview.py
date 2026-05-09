"""06_matmul_einsum - 矩阵乘法 / einsum

【目标】熟悉三种矩阵乘的写法，以及 einsum 的索引语法。

【三种写法】
- torch.matmul(a, b) 或 a @ b：通用，处理 2D / 3D / 高维 batch matmul
- torch.bmm(a, b)：固定批量矩阵乘 [B, M, K] @ [B, K, N]，明示 batch
- torch.einsum('btd,bsd->bts', q, k)：用爱因斯坦求和约定，索引最直观

attention 里 q @ k.T 用 einsum 写出来一目了然。

【任务】补全 5 个函数。
"""
from __future__ import annotations

import torch


def mm_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D 矩阵乘：a [M, K] @ b [K, N] -> [M, N]。

    提示：a @ b 或 torch.matmul(a, b)。
    """
    raise NotImplementedError("请补全 mm_2d")


def bmm_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """批量矩阵乘：a [B, M, K] @ b [B, K, N] -> [B, M, N]。

    提示：torch.bmm(a, b) 或 a @ b（matmul 也支持 batch dim）。
    """
    raise NotImplementedError("请补全 bmm_3d")


def attention_scores_matmul(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """用 matmul 计算 attention scores：q [B, T, D], k [B, S, D] -> [B, T, S]。

    形状对齐：先把 k 的最后两维交换，得到 [B, D, S]，再 q @ k_T。

    提示：q @ k.transpose(-2, -1)。
    """
    raise NotImplementedError("请补全 attention_scores_matmul")


def attention_scores_einsum(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """同上，但用 einsum：'btd,bsd->bts'。

    一旦看懂这个 einsum，attention 算 score 就再也不用反复 transpose。
    """
    raise NotImplementedError("请补全 attention_scores_einsum")


def weighted_combine(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """用 einsum 计算 attention 输出：
    weights [B, T, S], values [B, S, D] -> [B, T, D]，对应 'bts,bsd->btd'。

    这正是 attention 里 softmax(score) @ V 的写法。
    """
    raise NotImplementedError("请补全 weighted_combine")


def main() -> None:
    """最小可运行示例：
    1. mm_2d 测一下普通矩阵乘
    2. bmm_3d 测一下批量矩阵乘
    3. 用相同 q, k 跑 matmul 版本和 einsum 版本，验证两边数值一致
    4. weighted_combine 在 fake attention 上跑通
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
