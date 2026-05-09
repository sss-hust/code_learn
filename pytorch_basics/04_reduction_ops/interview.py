"""04_reduction_ops - 归约操作

【目标】熟悉沿着指定 dim 做 sum / mean / max / argmax，以及最常用的
组合操作 softmax / logsumexp。这些是 attention、loss、layer norm 的常驻动作。

【数值稳定要点】
- softmax: 先 x - x.max(dim, keepdim=True).values，再 exp / sum(exp)
- logsumexp: log(sum(exp(x))) = m + log(sum(exp(x - m)))，其中 m = max(x)

【任务】补全 4 个函数。本题不允许直接 return torch.softmax / torch.logsumexp，
要自己拼出来；后面有一个例外测试是允许调用 PyTorch 内建的，仅做对照。
"""
from __future__ import annotations

import torch


def row_mean(x: torch.Tensor) -> torch.Tensor:
    """对 2D 张量 x [B, N] 沿最后一维求均值，返回 [B]。

    提示：x.mean(dim=-1)。
    """
    raise NotImplementedError("请补全 row_mean")


def argmax_per_row(x: torch.Tensor) -> torch.Tensor:
    """对 2D 张量 x [B, N] 每行求 argmax，返回 [B] (long)。

    提示：x.argmax(dim=-1)。
    """
    raise NotImplementedError("请补全 argmax_per_row")


def softmax_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 计算数值稳定的 softmax。结果与 torch.softmax(x, dim) 一致到 1e-6。

    步骤：
    1. m = x.max(dim, keepdim=True).values
    2. e = (x - m).exp()
    3. return e / e.sum(dim, keepdim=True)
    """
    raise NotImplementedError("请补全 softmax_along")


def logsumexp_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 计算数值稳定的 logsumexp。结果与 torch.logsumexp(x, dim) 一致到 1e-6。

    步骤：
    1. m = x.max(dim, keepdim=True).values
    2. shifted = x - m
    3. return m.squeeze(dim) + shifted.exp().sum(dim).log()
    """
    raise NotImplementedError("请补全 logsumexp_stable")


def main() -> None:
    """最小可运行示例：
    1. 造 [3, 5] 张量，row_mean 打印 [3]
    2. argmax_per_row 打印每行最大值的列号
    3. softmax_along 与 torch.softmax 对比误差
    4. logsumexp_stable 与 torch.logsumexp 对比误差
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
