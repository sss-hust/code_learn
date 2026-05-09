"""
01c_vector_sum - 面试模式骨架。

需要你自己补：
1. sum_kernel：用 tl.sum 求 program 内局部和，tl.atomic_add 写到输出
2. vector_sum wrapper：output 用 torch.zeros 而不是 empty_like，否则会带初始残值
3. main()：构造数据 + 跑 + 与 x.sum() 对比误差
"""
import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pass


def vector_sum(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("请补全 vector_sum wrapper")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全数据构造 + 调用 + 校验")


if __name__ == "__main__":
    main()
