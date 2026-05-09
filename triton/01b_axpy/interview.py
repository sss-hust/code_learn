"""
01b_axpy - 面试模式骨架。

需要你自己补：
1. axpy_kernel 的完整 kernel 逻辑
2. axpy(x, a, b) wrapper 的 BLOCK_SIZE / grid / 启动
3. main() 数据流：构造 x、跑 axpy、和 PyTorch 参考做数值对比
"""
import torch
import triton
import triton.language as tl


@triton.jit
def axpy_kernel(x_ptr, output_ptr, a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    pass


def axpy(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    raise NotImplementedError("请补全 axpy 的 wrapper")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全数据构造 + 调用 + 校验")


if __name__ == "__main__":
    main()
