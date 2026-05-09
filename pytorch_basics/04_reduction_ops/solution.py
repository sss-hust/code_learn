"""04_reduction_ops - 参考答案"""
from __future__ import annotations

import torch


def row_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=-1)


def argmax_per_row(x: torch.Tensor) -> torch.Tensor:
    return x.argmax(dim=-1)


def softmax_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)


def logsumexp_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    shifted = x - m
    return m.squeeze(dim) + shifted.exp().sum(dim=dim).log()


def main() -> None:
    torch.manual_seed(0)

    x = torch.randn(3, 5)
    print("row_mean =", row_mean(x))
    print("argmax_per_row =", argmax_per_row(x))

    sm = softmax_along(x, dim=-1)
    ref = torch.softmax(x, dim=-1)
    print("softmax max_err =", (sm - ref).abs().max().item())

    lse = logsumexp_stable(x, dim=-1)
    ref_lse = torch.logsumexp(x, dim=-1)
    print("logsumexp max_err =", (lse - ref_lse).abs().max().item())


if __name__ == "__main__":
    main()
