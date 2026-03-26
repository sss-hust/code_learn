"""
10_flash_attention - 参考答案

【关键点】
1. 外循环遍历 Q 的块（由 grid 调度），内循环遍历 K/V 的所有块
2. 在线 softmax 的三个状态量: m (running max), l (running sum), acc (running output)
3. 每次碰到新 K 块时，需要用 exp(old_m - new_m) 修正历史累加值
4. FP16 输入 + FP32 累加器是标准做法
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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 当前 program 处理 Q 的哪个块
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)  # batch * head 的联合索引
    
    # Q 块的行范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 基础偏移（batch 和 head 维度）
    q_base = pid_bh * stride_qh if stride_qb == 0 else pid_bh * stride_qh
    # 更通用的做法：
    num_heads = stride_qb // stride_qh if stride_qh > 0 else 1
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    
    q_base = batch_idx * stride_qb + head_idx * stride_qh
    k_base = batch_idx * stride_kb + head_idx * stride_kh
    v_base = batch_idx * stride_vb + head_idx * stride_vh
    o_base = batch_idx * stride_ob + head_idx * stride_oh
    
    # 加载 Q 块: (BLOCK_M, BLOCK_D)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptr + q_ptrs, mask=q_mask, other=0.0)
    
    # 初始化在线 softmax 状态
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # running output
    
    # 遍历 K/V 的所有块
    for block_n_start in range(0, seq_len, BLOCK_N):
        offs_n = block_n_start + tl.arange(0, BLOCK_N)
        
        # 加载 K 块: (BLOCK_N, BLOCK_D) → 转置后做点积
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptr + k_ptrs, mask=k_mask, other=0.0)
        
        # scores = Q @ K^T * scale → (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # 对超出 seq_len 的位置 mask 为 -inf
        scores = tl.where(offs_n[None, :] < seq_len, scores, float('-inf'))
        
        # 在线 softmax 更新
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # 修正历史值
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        
        # 更新 l 和 acc
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # 加载 V 块并累加
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptr + v_ptrs, mask=v_mask, other=0.0)
        
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
    
    # 归一化
    acc = acc / l_i[:, None]
    
    # 写回
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(output_ptr + o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=o_mask)


def flash_attention(q, k, v):
    """Flash Attention 的 Python 封装"""
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
