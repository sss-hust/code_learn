"""05_reshape_view_permute - reshape / view / permute / contiguous

【目标】把张量的形状变换吃透。这是 attention 拆头/合头、conv 切换 NHWC/NCHW
反复要做的事。

【关键概念】
- view 要求连续内存（contiguous），如果不连续会报错；reshape 不连续时会自动 copy。
- transpose / permute 只改 stride，不动数据；之后再 view 之前往往要 contiguous()。
- view 适用于"形状能整除 stride" 的情况；不放心时用 reshape 更安全。

【任务】补全 4 个函数。
"""
from __future__ import annotations

import torch


def flatten_btc_to_nc(x: torch.Tensor) -> torch.Tensor:
    """把 [B, T, C] 的张量 flatten 成 [B*T, C]，方便按 token 处理（MoE 路由的标准动作）。

    提示：x.reshape(-1, x.size(-1))。也可以用 x.flatten(0, 1)。
    """
    raise NotImplementedError("请补全 flatten_btc_to_nc")


def unflatten_nc_to_btc(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """把 [N, C] 还原成 [B, T, C]，其中 N = B*T。要求 x.size(0) % batch_size == 0。

    提示：x.reshape(batch_size, -1, x.size(-1))。
    """
    raise NotImplementedError("请补全 unflatten_nc_to_btc")


def swap_last_two(x: torch.Tensor) -> torch.Tensor:
    """把张量的最后两维交换。适用于 attention 里的 k.transpose(-2, -1)。

    提示：x.transpose(-2, -1)。
    """
    raise NotImplementedError("请补全 swap_last_two")


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """把 [N, H, W, C] 转成 [N, C, H, W]，并保证返回的张量是 contiguous 的。

    提示：x.permute(0, 3, 1, 2).contiguous()。
    """
    raise NotImplementedError("请补全 nhwc_to_nchw")


def main() -> None:
    """最小可运行示例：
    1. flatten_btc_to_nc 在 [2, 3, 4] 上演示
    2. unflatten_nc_to_btc 把 [6, 4] 还原回 [2, 3, 4]
    3. swap_last_two 在 [2, 3, 4] 上观察 stride 的变化
    4. nhwc_to_nchw 在 [1, 2, 2, 3] 上演示，并 print is_contiguous
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
