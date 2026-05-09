"""
01d_row_max - 面试模式骨架。

需要你自己补：
1. row_max_kernel：注意 mask 之外要用 -inf 初始化（不然 max 会被污染）
2. row_max wrapper：BLOCK_SIZE 用 triton.next_power_of_2(n_cols)
3. main()：构造 [B, N] 张量、跑 kernel、和 x.max(dim=-1).values 对比
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(input_ptr, output_ptr, n_cols, input_row_stride, BLOCK_SIZE: tl.constexpr):
    pass


def row_max(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("请补全 row_max wrapper")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全数据构造 + 调用 + 校验")


if __name__ == "__main__":
    main()
