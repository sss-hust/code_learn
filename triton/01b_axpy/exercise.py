"""
01b_axpy - Triton AXPY (y = a*x + b)

【核心概念】
- 在 vector_add 的基础上多两个标量参数 a 和 b
- 标量参数直接当作 Python 浮点数传给 kernel，不需要 tl.constexpr
- 体会"标量在 kernel 里就是个常数"

【任务】
实现一个 Triton kernel：output = a * x + b（a, b 是 Python float）。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def axpy_kernel(
    x_ptr,
    output_ptr,
    a,                # 标量乘子（运行时传入）
    b,                # 标量偏置
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    AXPY kernel: output[i] = a * x[i] + b

    提示：
    1. pid = tl.program_id(0)
    2. offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    3. mask = offsets < n_elements
    4. x = tl.load(x_ptr + offsets, mask=mask)
    5. output = a * x + b
    6. tl.store(output_ptr + offsets, output, mask=mask)
    """
    pass


def axpy(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    axpy_kernel[grid](x, output, a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    a, b = 2.5, -0.5
    output = axpy(x, a, b)
    expected = a * x + b
    print(f"max_err = {(output - expected).abs().max().item():.6f}")
    print("通过" if torch.allclose(output, expected, atol=1e-5) else "失败")
