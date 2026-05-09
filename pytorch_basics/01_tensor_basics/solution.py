"""01_tensor_basics - 参考答案"""
from __future__ import annotations

from typing import Any

import torch


def make_arange_2d(rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)


def as_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype=dtype)


def tensor_info(x: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "numel": x.numel(),
        "ndim": x.ndim,
    }


def safe_to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return x.to(device=device)


def main() -> None:
    grid = make_arange_2d(3, 4)
    print("grid.shape =", tuple(grid.shape))
    print("grid =\n", grid)

    grid64 = as_dtype(grid, torch.float64)
    print("grid64.dtype =", grid64.dtype)

    info = tensor_info(torch.zeros(2, 5))
    print("info =", info)

    moved = safe_to_device(grid, "cuda")
    print("moved.device =", moved.device)


if __name__ == "__main__":
    main()
