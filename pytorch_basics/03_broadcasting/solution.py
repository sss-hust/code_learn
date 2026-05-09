"""03_broadcasting - 参考答案"""
from __future__ import annotations

import torch


def add_row_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return matrix + bias


def add_col_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return matrix + bias.unsqueeze(-1)


def outer_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, None] + b[None, :]


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = x[:, None, :] - y[None, :, :]
    return (diff * diff).sum(dim=-1)


def main() -> None:
    torch.manual_seed(0)

    matrix = torch.zeros(3, 4)
    print("row_bias =\n", add_row_bias(matrix, torch.tensor([1.0, 2.0, 3.0, 4.0])))
    print("col_bias =\n", add_col_bias(matrix, torch.tensor([10.0, 20.0, 30.0])))

    print("outer_add =\n", outer_add(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0, 40.0])))

    x = torch.randn(3, 2)
    y = torch.randn(4, 2)
    d = pairwise_squared_distance(x, y)
    print("dist.shape =", tuple(d.shape))
    print("dist =\n", d)


if __name__ == "__main__":
    main()
