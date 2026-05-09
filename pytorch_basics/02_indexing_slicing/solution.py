"""02_indexing_slicing - 参考答案"""
from __future__ import annotations

import torch


def select_rows(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def top_k_values_per_row(x: torch.Tensor, k: int) -> torch.Tensor:
    values, _ = torch.topk(x, k, dim=-1)
    return values


def mask_below(x: torch.Tensor, threshold: float, fill_value: float) -> torch.Tensor:
    return x.masked_fill(x < threshold, fill_value)


def gather_per_row(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, dim=1, index=indices)


def where_positive_else(x: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, fallback)


def main() -> None:
    torch.manual_seed(0)

    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    print("rows[0,2,4] =\n", select_rows(x, torch.tensor([0, 2, 4])))

    y = torch.randn(3, 6)
    print("topk =", top_k_values_per_row(y, 2))

    z = torch.tensor([-1.0, 0.5, -2.0, 3.0])
    print("mask_below =", mask_below(z, 0.0, 0.0))

    data = torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=torch.float32)
    idx = torch.tensor([[3, 0], [1, 2]])
    print("gather =", gather_per_row(data, idx))

    fallback = torch.zeros_like(z)
    print("where =", where_positive_else(z, fallback))


if __name__ == "__main__":
    main()
