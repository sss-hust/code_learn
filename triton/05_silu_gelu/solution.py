"""
05_silu_gelu - 参考答案

【关键点】
1. SiLU 非常简单：x * sigmoid(x)
2. GELU 的 tanh 近似公式中，tanh 可以用 sigmoid 表示：tanh(x) = 2*sigmoid(2x) - 1
3. 这类逐元素 kernel 是 memory-bound 的，性能瓶颈在内存带宽而非计算
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SiLU = x * sigmoid(x)
    output = x * tl.sigmoid(x)
    
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU (tanh 近似)
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tanh(a) = 2 * sigmoid(2a) - 1
    SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    output = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(output_ptr + offsets, output, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU 的 Python 封装"""
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    silu_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU 的 Python 封装"""
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    gelu_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(10000, device='cuda')
    
    out_silu = silu(x)
    ref_silu = torch.nn.functional.silu(x)
    print(f"SiLU 最大误差: {(out_silu - ref_silu).abs().max().item():.6f}")
    
    out_gelu = gelu(x)
    ref_gelu = torch.nn.functional.gelu(x, approximate='tanh')
    print(f"GELU 最大误差: {(out_gelu - ref_gelu).abs().max().item():.6f}")
