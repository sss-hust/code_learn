"""05_reshape_view_permute - 参考答案"""
from __future__ import annotations

import torch


def flatten_btc_to_nc(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1))


def unflatten_nc_to_btc(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return x.reshape(batch_size, -1, x.size(-1))


def swap_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1)


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def main() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    flat = flatten_btc_to_nc(x)
    print("flat.shape =", tuple(flat.shape))

    back = unflatten_nc_to_btc(flat, batch_size=2)
    print("back.shape =", tuple(back.shape))
    print("round-trip equal =", torch.equal(back, x))

    swapped = swap_last_two(x)
    print("swap.shape =", tuple(swapped.shape), "swap.stride =", swapped.stride())

    nhwc = torch.randn(1, 2, 2, 3)
    nchw = nhwc_to_nchw(nhwc)
    print("nchw.shape =", tuple(nchw.shape), "is_contiguous =", nchw.is_contiguous())


if __name__ == "__main__":
    main()
