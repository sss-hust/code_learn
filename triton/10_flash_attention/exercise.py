"""
10_flash_attention - Triton Flash Attention

【核心概念】
- Flash Attention: 分块计算 Attention，避免存储完整的 N×N attention矩阵
- 结合了在线 softmax（练习07）和分块矩阵乘法（练习08）的思想
- 核心算法：
    对 Q 的每个块 q_block:
        初始化 acc=0, m=-inf, l=0
        对 K/V 的每个块：
            s = q_block @ k_block^T / sqrt(d)
            m_new = max(m, row_max(s))
            p = exp(s - m_new)
            l = l * exp(m - m_new) + row_sum(p)
            acc = acc * exp(m - m_new) + p @ v_block
            m = m_new
        output = acc / l

【任务】
实现 Flash Attention 的 Triton kernel（forward only）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, head_dim,
    scale,  # 1 / sqrt(head_dim)
    BLOCK_M: tl.constexpr,   # Q 块大小
    BLOCK_N: tl.constexpr,   # K/V 块大小
    BLOCK_D: tl.constexpr,   # head_dim
):
    """
    Flash Attention kernel (causal=False)
    
    grid: (cdiv(seq_len, BLOCK_M), batch * num_heads)
    每个 program 计算输出的 (BLOCK_M, head_dim) 块
    
    提示：
    1. 从 program_id 计算 block_m 索引和 (batch, head) 索引
    2. 加载 Q 块: (BLOCK_M, BLOCK_D)
    3. 初始化 acc=0, m=-inf, l=0
    4. 遍历 K/V 的所有块：
       a. 加载 K 块，计算 scores = Q @ K^T * scale
       b. 更新 m_new, 修正 acc 和 l
       c. 计算 p = exp(scores - m_new)
       d. 加载 V 块，acc += p @ V
    5. 归一化: output = acc / l
    """
    # TODO: 在此实现你的代码
    pass


def flash_attention(q, k, v):
    """Flash Attention 的 Python 封装
    
    Args:
        q, k, v: (batch, num_heads, seq_len, head_dim)
    Returns:
        output:  (batch, num_heads, seq_len, head_dim)
    """
    batch, num_heads, seq_len, head_dim = q.shape
    
    output = torch.empty_like(q)
    scale = head_dim ** -0.5
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)
    
    flash_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len, head_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return output


def attention_ref(q, k, v):
    """PyTorch 参考实现"""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 2, 4, 256, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    output = flash_attention(q, k, v)
    expected = attention_ref(q, k, v)
    
    print(f"最大误差: {(output - expected).abs().max().item():.4f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-1) else '失败 ✗'}")
