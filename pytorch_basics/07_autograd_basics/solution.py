"""07_autograd_basics - 参考答案"""
from __future__ import annotations

from typing import Callable

import torch


def numerical_grad(fn: Callable[[torch.Tensor], torch.Tensor], x_value: float, h: float = 1e-4) -> float:
    plus = fn(torch.tensor(x_value + h, dtype=torch.float64))
    minus = fn(torch.tensor(x_value - h, dtype=torch.float64))
    return ((plus - minus) / (2.0 * h)).item()


def autograd_grad(fn: Callable[[torch.Tensor], torch.Tensor], x_value: float) -> float:
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    y = fn(x)
    y.backward()
    return x.grad.item()


def manual_sgd_step(
    param: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    lr: float,
) -> None:
    if param.grad is not None:
        param.grad.zero_()
    loss = loss_fn(param)
    loss.backward()
    with torch.no_grad():
        param.sub_(lr * param.grad)
    param.grad.zero_()


def detached_does_not_track_grad(x: torch.Tensor) -> bool:
    return not x.detach().requires_grad


def main() -> None:
    fn = lambda x: x ** 3 + 2 * x
    print("autograd  =", autograd_grad(fn, 1.5))
    print("numerical =", numerical_grad(fn, 1.5))

    param = torch.tensor(0.0, requires_grad=True)
    target = torch.tensor(3.0)
    for _ in range(50):
        manual_sgd_step(param, lambda p: (p - target) ** 2, lr=0.1)
    print("converged param =", param.item())

    x = torch.randn(3, requires_grad=True)
    print("detach safe =", detached_does_not_track_grad(x))


if __name__ == "__main__":
    main()
