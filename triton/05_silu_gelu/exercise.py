"""
05_silu_gelu - Triton 激活函数

【核心概念】
- SiLU (Swish): f(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
- GELU: f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
- 逐元素运算，典型的 memory-bound kernel
- LLaMA 用 SiLU，GPT/BERT 用 GELU

【任务】
实现两个 Triton kernel：SiLU 和 GELU 激活函数
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
    """
    SiLU kernel: output = x * sigmoid(x)
    
    提示：
    1. sigmoid(x) = 1 / (1 + exp(-x))
    2. Triton 提供 tl.sigmoid() 内置函数
    """
    # TODO: 在此实现你的代码
    pass


@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU kernel (tanh 近似):
    output = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    提示：
    1. 常量 sqrt(2/pi) ≈ 0.7978845608
    2. Triton 没有 tanh，用公式: tanh(x) = 2 * sigmoid(2x) - 1
    """
    # TODO: 在此实现你的代码
    pass


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
    
    # SiLU
    out_silu = silu(x)
    ref_silu = torch.nn.functional.silu(x)
    print(f"SiLU 最大误差: {(out_silu - ref_silu).abs().max().item():.6f}")
    
    # GELU
    out_gelu = gelu(x)
    ref_gelu = torch.nn.functional.gelu(x, approximate='tanh')
    print(f"GELU 最大误差: {(out_gelu - ref_gelu).abs().max().item():.6f}")
