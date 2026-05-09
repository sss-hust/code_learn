"""10_minimal_training_loop - 最小训练循环

【目标】把"模型 + 损失 + 优化器 + 训练循环"这条主链路串起来。
完成后再去看 model_layers 的题，"forward 写完之后怎么训"就不再陌生。

【任务】
1. make_simple_mlp：返回一个 Linear -> ReLU -> Linear 的小 MLP
2. train_step：一步前向 + 反向 + 优化器更新，返回 loss 标量
3. fit_regression：跑完整训练循环，返回每个 epoch 的 loss 列表

测试会用一个简单的线性回归任务（X @ W_true + b_true + 噪声），
验证 fit_regression 跑完后 loss 显著下降。
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn


def make_simple_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """返回一个 Linear -> ReLU -> Linear 的 nn.Sequential。

    提示：nn.Sequential(nn.Linear(...), nn.ReLU(), nn.Linear(...))。
    """
    raise NotImplementedError("请补全 make_simple_mlp")


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """一步训练。返回 loss 的 Python float。

    步骤：
    1. optimizer.zero_grad()
    2. pred = model(x)
    3. loss = loss_fn(pred, y)
    4. loss.backward()
    5. optimizer.step()
    6. return loss.item()
    """
    raise NotImplementedError("请补全 train_step")


def fit_regression(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[float]:
    """跑 epochs 个 epoch 的全 batch 训练，loss 用 MSE，优化器用 SGD。
    每个 epoch 把 train_step 的 loss 收集到列表里返回。

    提示：
    - loss_fn = nn.MSELoss()
    - optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    - 每个 epoch 调一次 train_step(model, X, y, loss_fn, optimizer)
    """
    raise NotImplementedError("请补全 fit_regression")


def main() -> None:
    """最小可运行示例：
    1. 造一个 X [200, 4], 真实关系 y = X @ W_true + b_true + 0.1 * noise
    2. make_simple_mlp(4, 16, 1)
    3. fit_regression 100 个 epoch，lr=0.05
    4. 打印 losses[0] vs losses[-1]，应该看到显著下降
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
