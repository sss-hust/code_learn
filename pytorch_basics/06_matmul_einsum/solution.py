"""06_matmul_einsum - 参考答案"""
from __future__ import annotations

import torch


def mm_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def bmm_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b)


def attention_scores_matmul(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return q @ k.transpose(-2, -1)


def attention_scores_einsum(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("btd,bsd->bts", q, k)


def weighted_combine(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bts,bsd->btd", weights, values)


def main() -> None:
    torch.manual_seed(0)

    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    print("mm_2d.shape =", tuple(mm_2d(a, b).shape))

    a3 = torch.randn(2, 3, 4)
    b3 = torch.randn(2, 4, 5)
    print("bmm_3d.shape =", tuple(bmm_3d(a3, b3).shape))

    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 5, 8)
    s_mm = attention_scores_matmul(q, k)
    s_ein = attention_scores_einsum(q, k)
    print("score max_err =", (s_mm - s_ein).abs().max().item())

    weights = torch.softmax(s_mm, dim=-1)
    v = torch.randn(2, 5, 8)
    out = weighted_combine(weights, v)
    print("attention out.shape =", tuple(out.shape))


if __name__ == "__main__":
    main()
