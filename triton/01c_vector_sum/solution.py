"""01c_vector_sum - 参考答案"""
import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, partial)


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
    print(f"err = {(out - ref).abs().item():.6f}")
