"""01_tensor_basics - 张量基础

【目标】掌握张量的创建、属性查询、dtype 转换、设备迁移。
这些操作会在后面所有题里反复出现，先把肌肉记忆打通。

【任务】
补全下面 4 个函数，并在 main() 里写最小可运行示例。
"""
from __future__ import annotations

from typing import Any

import torch


def make_arange_2d(rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """造一个 [rows, cols] 的张量，元素值依次为 0, 1, 2, ..., rows*cols-1。

    提示：
    - torch.arange(N, dtype=...) 给你一个 1D 等差张量
    - .reshape(rows, cols) 改成 2D
    """
    raise NotImplementedError("请补全 make_arange_2d")


def as_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """把 x 转成目标 dtype，返回一个新张量（不修改原 x）。

    提示：x.to(dtype=dtype) 即可，注意它返回新张量。
    """
    raise NotImplementedError("请补全 as_dtype")


def tensor_info(x: torch.Tensor) -> dict[str, Any]:
    """返回张量元信息字典，至少包含 shape / dtype / numel / ndim 四个字段。

    - shape: tuple(x.shape)，转成 int 元组好打印好对比
    - dtype: str(x.dtype)，转成字符串避免序列化问题
    - numel / ndim: 直接调 x.numel() / x.ndim
    """
    raise NotImplementedError("请补全 tensor_info")


def safe_to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    """把 x 搬到 device。如果指定 'cuda' 但 GPU 不可用，自动回退到 cpu，不抛异常。

    提示：torch.cuda.is_available() 判断 GPU 是否可用。
    """
    raise NotImplementedError("请补全 safe_to_device")


def main() -> None:
    """最小可运行示例：
    1. make_arange_2d(3, 4) 造一个张量并打印形状和内容
    2. 用 as_dtype 把它转成 float64，验证 dtype 改变
    3. 用 tensor_info 打印 [2, 5] 张量的元信息
    4. safe_to_device 试一次 'cuda'，看搬到了哪
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
