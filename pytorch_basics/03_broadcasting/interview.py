"""03_broadcasting - 广播

【目标】掌握 PyTorch 广播规则。两个张量做运算时，从最右边开始对齐维度，
每一维要么相等，要么其中一个是 1，要么其中一个不存在。

广播是 attention / FFN / loss 里"少写一层 for 循环"的核心机制。

【任务】补全 4 个函数。每个函数都至少能写出一种用广播实现的版本。
"""
from __future__ import annotations

import torch


def add_row_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """matrix [M, N] + bias [N] -> [M, N]，每一行加同一个 bias。

    提示：直接 matrix + bias 即可，PyTorch 会自动把 [N] 当成 [1, N] 广播。
    这正好对应 nn.Linear 加 bias 的行为。
    """
    raise NotImplementedError("请补全 add_row_bias")


def add_col_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """matrix [M, N] + bias [M] -> [M, N]，每一列加同一个 bias（按行不同）。

    提示：要先把 bias 变成 [M, 1] 才能广播到 [M, N]。
    用 bias.unsqueeze(-1) 或 bias[:, None]。
    """
    raise NotImplementedError("请补全 add_col_bias")


def outer_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [M], b [N] -> [M, N]，结果 out[i, j] = a[i] + b[j]。

    提示：a[:, None] + b[None, :]，把它们造成 [M, 1] 和 [1, N] 然后广播。
    这是位置编码和 RoPE 的基本建构。
    """
    raise NotImplementedError("请补全 outer_add")


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x [M, D], y [N, D] -> [M, N]，结果 out[i, j] = ||x[i] - y[j]||^2。

    提示：
    - x[:, None, :] 形状是 [M, 1, D]
    - y[None, :, :] 形状是 [1, N, D]
    - 相减广播得 [M, N, D]
    - 平方后沿最后一维 sum 得 [M, N]
    """
    raise NotImplementedError("请补全 pairwise_squared_distance")


def main() -> None:
    """最小可运行示例：
    1. add_row_bias 在 [3, 4] 矩阵上加 [4] 偏置
    2. add_col_bias 在 [3, 4] 矩阵上加 [3] 偏置
    3. outer_add 演示 [3] + [4] -> [3, 4]
    4. pairwise_squared_distance 在两组小点上计算距离矩阵
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
