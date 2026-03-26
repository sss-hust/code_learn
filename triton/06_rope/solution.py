"""
06_rope - 参考答案

【关键点】
1. 将 head_dim 个元素分为 (head_dim/2) 对：(x0,x1), (x2,x3), ...
2. 每对应用相同角度旋转，但不同对有不同频率 θ_i
3. cos/sin 表是预计算的，kernel 只需要查表
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_kernel(
    x_ptr,
    output_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    head_dim,
    x_batch_stride,
    x_seq_stride,
    cos_seq_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 对应的 batch 和 seq 索引
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # 半维度偏移
    half_offsets = tl.arange(0, BLOCK_SIZE)  # [0, 1, ..., head_dim/2 - 1]
    
    # 加载偶数位和奇数位
    x_base = batch_idx * x_batch_stride + seq_idx * x_seq_stride
    x0 = tl.load(x_ptr + x_base + half_offsets * 2)       # 偶数位: 0, 2, 4, ...
    x1 = tl.load(x_ptr + x_base + half_offsets * 2 + 1)   # 奇数位: 1, 3, 5, ...
    
    # 加载 cos/sin
    freq_base = seq_idx * cos_seq_stride
    cos_val = tl.load(cos_ptr + freq_base + half_offsets)
    sin_val = tl.load(sin_ptr + freq_base + half_offsets)
    
    # 旋转
    out0 = x0 * cos_val - x1 * sin_val
    out1 = x0 * sin_val + x1 * cos_val
    
    # 写回（交错存储）
    out_base = batch_idx * x_batch_stride + seq_idx * x_seq_stride
    tl.store(output_ptr + out_base + half_offsets * 2, out0)
    tl.store(output_ptr + out_base + half_offsets * 2 + 1, out1)


def precompute_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                     device='cuda') -> tuple:
    """预计算 cos/sin 频率表"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE 的 Python 封装"""
    batch, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0
    
    output = torch.empty_like(x)
    half_dim = head_dim // 2
    
    grid = (batch * seq_len,)
    rope_kernel[grid](
        x, output, cos, sin,
        seq_len, head_dim,
        x.stride(0), x.stride(1),
        cos.stride(0),
        BLOCK_SIZE=half_dim,
    )
    return output


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现"""
    x0 = x[..., ::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack([out0, out1], dim=-1).flatten(-2)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, head_dim = 2, 64, 128
    x = torch.randn(batch, seq_len, head_dim, device='cuda')
    cos, sin = precompute_freqs(head_dim, seq_len)
    
    output = rope(x, cos, sin)
    expected = rope_ref(x, cos, sin)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
