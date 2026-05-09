"""02_indexing_slicing - 索引与切片

【目标】熟悉 PyTorch 的几种取值方式：
- 切片 / bool mask / advanced indexing
- gather（按索引在指定维度上取）
- masked_fill（按 mask 替换）
- where（按条件二选一）

这些是后面写 attention、router、loss 函数的基本动作。

【任务】补全 5 个函数。
"""
from __future__ import annotations

import torch


def select_rows(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """从 2D 张量 x [N, D] 中按 indices [K] (long) 取出 K 行，返回 [K, D]。

    提示：x[indices] 就够了。也可以用 torch.index_select(x, dim=0, index=indices)。
    """
    raise NotImplementedError("请补全 select_rows")


def top_k_values_per_row(x: torch.Tensor, k: int) -> torch.Tensor:
    """对 2D 张量 x [B, N] 的每一行，返回前 k 大的值，形状 [B, k]，按降序排列。

    提示：torch.topk(x, k, dim=-1) 返回 (values, indices)，这里只要 values。
    """
    raise NotImplementedError("请补全 top_k_values_per_row")


def mask_below(x: torch.Tensor, threshold: float, fill_value: float) -> torch.Tensor:
    """把 x 中所有 < threshold 的位置替换成 fill_value，返回新张量。

    提示：x.masked_fill(x < threshold, fill_value)。注意 masked_fill 的语义是
    "mask 为 True 的位置替换"。
    """
    raise NotImplementedError("请补全 mask_below")


def gather_per_row(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """x [B, N] 是数据，indices [B, K] (long) 是每行要取的列号，返回 [B, K]。

    例如 x[i] = [10, 20, 30, 40], indices[i] = [3, 0] -> 返回 [40, 10]。

    提示：torch.gather(x, dim=1, index=indices)。
    """
    raise NotImplementedError("请补全 gather_per_row")


def where_positive_else(x: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """逐元素：x[i] > 0 时取 x[i]，否则取 fallback[i]。x 和 fallback 形状相同。

    提示：torch.where(x > 0, x, fallback)。
    """
    raise NotImplementedError("请补全 where_positive_else")


def main() -> None:
    """最小可运行示例：
    1. 造一个 [5, 3] 张量，select_rows 取 [0, 2, 4] 行
    2. top_k_values_per_row 在 [3, 6] 张量上取前 2 大
    3. mask_below 把负数替换为 0
    4. gather_per_row 演示按列号取值
    5. where_positive_else 演示条件选择
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
