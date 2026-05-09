"""07_autograd_basics - 自动微分基础

【目标】把 autograd 的核心几个动作过一遍：
- requires_grad / backward / .grad
- with torch.no_grad() 和 .detach()
- 手动一步 SGD 更新

【任务】补全 4 个函数。其中 numerical_grad 是数值梯度对照，autograd_grad
是真用 autograd。后续测试会用它们做"正确性"对比。
"""
from __future__ import annotations

from typing import Callable

import torch


def numerical_grad(fn: Callable[[torch.Tensor], torch.Tensor], x_value: float, h: float = 1e-4) -> float:
    """用中心差分估计 f 在 x_value 处的导数。fn 接受一个 0-D 张量并返回一个 0-D 张量。

    公式：(f(x + h) - f(x - h)) / (2 * h)。
    返回 Python float（可以用 .item()）。
    """
    raise NotImplementedError("请补全 numerical_grad")


def autograd_grad(fn: Callable[[torch.Tensor], torch.Tensor], x_value: float) -> float:
    """用 autograd 计算 f 在 x_value 处的导数，返回 float。

    步骤：
    1. x = torch.tensor(x_value, requires_grad=True)
    2. y = fn(x)
    3. y.backward()
    4. return x.grad.item()
    """
    raise NotImplementedError("请补全 autograd_grad")


def manual_sgd_step(
    param: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    lr: float,
) -> None:
    """对 param 做一步 SGD：loss = loss_fn(param)，param.data -= lr * param.grad。

    要求：
    - param 必须 requires_grad=True
    - 在 with torch.no_grad() 里更新 param.data，避免污染计算图
    - 更新完后把 param.grad 清零（否则下一步会累加）
    - 这个函数是原地修改 param，没有返回值

    提示：
        loss = loss_fn(param)
        if param.grad is not None:
            param.grad.zero_()
        loss.backward()
        with torch.no_grad():
            param -= lr * param.grad
        param.grad.zero_()
    """
    raise NotImplementedError("请补全 manual_sgd_step")


def detached_does_not_track_grad(x: torch.Tensor) -> bool:
    """判断"对 x.detach() 的操作是否会污染 x 的梯度图"。
    返回 True 表示 detach 之后的运算不会反向传到原图（这是 detach 的正确语义）。

    实现思路（直接照抄即可，本题重点是理解结论）：
    1. 假设 x 是 requires_grad=True 的张量
    2. y = x.detach() * 3
    3. 如果对 y.sum().backward() 调用，x.grad 应该仍然是 None
    4. 函数不要真的去 backward，避免把外部状态搞乱
    5. 直接根据 x.detach() 的 requires_grad 字段判断：detach 出来的张量 requires_grad 默认为 False
    """
    raise NotImplementedError("请补全 detached_does_not_track_grad")


def main() -> None:
    """最小可运行示例：
    1. 对 f(x) = x**3 + 2*x，比较 autograd 和 numerical 在 x=1.5 处的导数
    2. 给一个 param=torch.tensor(0.0, requires_grad=True)，损失 (param - 3.0)**2，
       连续做 50 步 manual_sgd_step，看 param 收敛到 3.0 附近
    3. 调一次 detached_does_not_track_grad 验证返回 True
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
