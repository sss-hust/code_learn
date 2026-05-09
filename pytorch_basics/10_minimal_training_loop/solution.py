"""10_minimal_training_loop - 参考答案"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def make_simple_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def fit_regression(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[float]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        losses.append(train_step(model, X, y, loss_fn, optimizer))
    return losses


def main() -> None:
    torch.manual_seed(0)

    X = torch.randn(200, 4)
    W_true = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    b_true = torch.tensor([0.5])
    y = X @ W_true + b_true + 0.1 * torch.randn(200, 1)

    model = make_simple_mlp(4, 16, 1)
    losses = fit_regression(model, X, y, epochs=100, lr=0.05)
    print(f"loss[0] = {losses[0]:.4f}, loss[-1] = {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
