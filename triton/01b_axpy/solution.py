"""01b_axpy - 参考答案"""
import torch
import triton
import triton.language as tl


@triton.jit
def axpy_kernel(x_ptr, output_ptr, a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = a * x + b
    tl.store(output_ptr + offsets, output, mask=mask)


def axpy(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    axpy_kernel[grid](x, output, a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(98432, device="cuda")
    a, b = 2.5, -0.5
    out = axpy(x, a, b)
    ref = a * x + b
    print(f"max_err = {(out - ref).abs().max().item():.6f}")
    print("通过" if torch.allclose(out, ref, atol=1e-5) else "失败")
