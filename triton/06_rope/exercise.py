"""
06_rope - Triton 旋转位置编码 (RoPE)

【核心概念】
- RoPE: 将位置信息编码为旋转矩阵，应用于 Q/K 向量
- 对每对相邻元素 (x0, x1)，旋转角度 θ*pos:
    x0' = x0 * cos(θ*pos) - x1 * sin(θ*pos)
    x1' = x0 * sin(θ*pos) + x1 * cos(θ*pos)
- θ_i = 1 / base^(2i/d), 其中 base=10000, i 是维度索引

【任务】
实现一个 Triton kernel，对 (batch, seq_len, head_dim) 的张量应用 RoPE
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_kernel(
    x_ptr,
    output_ptr,
    cos_ptr,        # 预计算的 cos 表 (seq_len, head_dim // 2)
    sin_ptr,        # 预计算的 sin 表 (seq_len, head_dim // 2)
    seq_len,
    head_dim,
    x_batch_stride,
    x_seq_stride,
    cos_seq_stride,
    BLOCK_SIZE: tl.constexpr,  # = head_dim // 2
):
    """
    RoPE kernel
    
    grid: (batch * seq_len,)
    每个 program 处理一个 token 的所有维度对
    
    提示：
    1. 从 program_id 推算 batch_idx 和 seq_idx
    2. 加载 head_dim 个元素，分为前半和后半（或奇偶交错）
    3. 加载对应位置的 cos/sin 值
    4. 应用旋转: (x0*cos - x1*sin, x0*sin + x1*cos)
    """
    # TODO: 在此实现你的代码
    pass


def precompute_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                     device='cuda') -> tuple:
    """预计算 cos/sin 频率表"""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
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
    x0 = x[..., ::2]   # 偶数位
    x1 = x[..., 1::2]  # 奇数位
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
