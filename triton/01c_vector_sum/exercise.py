"""
01c_vector_sum - Triton 1D 求和

【核心概念】
- 第一个真正的 reduction：全局把所有元素加起来，输出一个标量
- 每个 program 用 tl.sum 算自己这一块的局部和
- 用 tl.atomic_add 把局部和并到唯一的输出标量上
- 体会"为什么需要 atomic"：多个 program 同时要写同一个地址

【任务】
实现 sum_kernel: 把 x [N] 的所有元素加起来，写到 output [1]。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,
    output_ptr,    # 形状 [1] 的张量
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D reduction kernel.

    提示：
    1. pid = tl.program_id(0)
    2. offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    3. mask = offsets < n_elements
    4. x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # 越界位置补 0 避免污染
    5. partial = tl.sum(x, axis=0)
    6. tl.atomic_add(output_ptr, partial)
    """
    pass


def vector_sum(x: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sum_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(100000, device="cuda")
    out = vector_sum(x)
    ref = x.sum()
    print(f"out = {out.item():.6f}, ref = {ref.item():.6f}")
    print(f"err = {(out - ref).abs().item():.6f}")
