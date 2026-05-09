"""
02b_row_normalize - 面试模式骨架。

需要你自己补：
1. row_normalize_kernel：load 时 mask 之外用 0 填，避免污染 sum
2. row_normalize wrapper：BLOCK_SIZE 用 next_power_of_2
3. main()：构造非负 [B, N] 张量、跑 kernel、和 x / x.sum(dim=-1, keepdim=True) 对比
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_normalize_kernel(
    input_ptr, output_ptr, n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pass


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("请补全 row_normalize wrapper")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全数据构造 + 调用 + 校验")


if __name__ == "__main__":
    main()
