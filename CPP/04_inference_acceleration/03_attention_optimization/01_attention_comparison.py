"""
01_attention_comparison.py - Attention优化对比

【核心问题】
标准Attention的复杂度是 O(n^2)，对于长序列：
- 内存：存储 n×n 的attention矩阵
- 计算：n^2 次乘法

【优化方向】
1. Flash Attention：分块计算，减少内存访问
2. Multi-Query Attention：共享KV，减少内存
3. Grouped-Query Attention：折中方案

运行: python 01_attention_comparison.py
"""

import numpy as np
import time
from typing import Optional

# ============================================================================
# 第一部分：标准Attention
# ============================================================================

def standard_attention(
    Q: np.ndarray,  # (batch, heads, seq_q, head_dim)
    K: np.ndarray,  # (batch, heads, seq_k, head_dim)
    V: np.ndarray,  # (batch, heads, seq_k, head_dim)
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """标准Scaled Dot-Product Attention"""
    
    d_k = Q.shape[-1]
    
    # (batch, heads, seq_q, seq_k)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask  # mask为负无穷处变为0
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 加权求和
    output = np.matmul(attention, V)
    
    return output


def analyze_memory(seq_len: int, heads: int, head_dim: int, batch: int = 1):
    """分析Attention的内存占用"""
    
    # Q, K, V
    qkv_memory = 3 * batch * heads * seq_len * head_dim * 4  # FP32
    
    # Attention矩阵
    attn_memory = batch * heads * seq_len * seq_len * 4
    
    total = qkv_memory + attn_memory
    
    print(f"序列长度: {seq_len}")
    print(f"  QKV内存: {qkv_memory / 1024 / 1024:.2f} MB")
    print(f"  Attention矩阵: {attn_memory / 1024 / 1024:.2f} MB")
    print(f"  总计: {total / 1024 / 1024:.2f} MB")


# ============================================================================
# 第二部分：Flash Attention（分块计算）
# ============================================================================

def flash_attention_simplified(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 64
) -> np.ndarray:
    """
    Flash Attention简化版
    
    核心思想：
    1. 分块计算attention，不需要存储完整的n×n矩阵
    2. 使用在线softmax算法
    
    注意：这是简化的教学版本，实际Flash Attention需要CUDA实现
    """
    batch, heads, seq_len, head_dim = Q.shape
    d_k = head_dim
    
    # 输出和归一化因子
    O = np.zeros_like(Q)
    L = np.zeros((batch, heads, seq_len, 1))  # log-sum-exp
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        
        # 当前块的K, V
        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]
        
        # 计算当前块的attention分数
        scores = np.matmul(Q, K_block.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # 在线softmax更新
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        
        if block_idx == 0:
            L = max_scores + np.log(np.sum(exp_scores, axis=-1, keepdims=True))
            O = np.matmul(exp_scores, V_block)
        else:
            # 更新log-sum-exp
            new_max = np.maximum(L, max_scores)
            exp_old = np.exp(L - new_max)
            exp_new = np.exp(max_scores - new_max) * np.sum(exp_scores, axis=-1, keepdims=True)
            
            # 更新输出
            O = O * exp_old + np.matmul(exp_scores * np.exp(max_scores - new_max), V_block)
            L = new_max + np.log(exp_old + exp_new)
    
    # 归一化
    O = O / np.exp(L)
    
    return O


# ============================================================================
# 第三部分：Multi-Query Attention
# ============================================================================

def multi_query_attention(
    Q: np.ndarray,     # (batch, num_heads, seq_q, head_dim)
    K: np.ndarray,     # (batch, 1, seq_k, head_dim) - 共享
    V: np.ndarray,     # (batch, 1, seq_k, head_dim) - 共享
) -> np.ndarray:
    """
    Multi-Query Attention (MQA)
    
    所有查询头共享一组K, V
    - 内存减少: heads倍
    - KV Cache减少: heads倍
    """
    d_k = Q.shape[-1]
    
    # K, V会自动广播到所有heads
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.matmul(attention, V)
    
    return output


def grouped_query_attention(
    Q: np.ndarray,     # (batch, num_heads, seq_q, head_dim)
    K: np.ndarray,     # (batch, num_kv_heads, seq_k, head_dim)
    V: np.ndarray,     # (batch, num_kv_heads, seq_k, head_dim)
    num_kv_heads: int
) -> np.ndarray:
    """
    Grouped-Query Attention (GQA)
    
    多个查询头共享一组K, V
    例如：32个query heads，8个kv heads -> 每4个query共享1个kv
    """
    batch, num_heads, seq_q, head_dim = Q.shape
    heads_per_kv = num_heads // num_kv_heads
    
    # 扩展K, V
    K_expanded = np.repeat(K, heads_per_kv, axis=1)
    V_expanded = np.repeat(V, heads_per_kv, axis=1)
    
    return standard_attention(Q, K_expanded, V_expanded)


# ============================================================================
# 第四部分：KV Cache对比
# ============================================================================

def compare_kv_cache_memory(
    batch: int,
    layers: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int = 2  # FP16
):
    """比较不同Attention变体的KV Cache大小"""
    
    print(f"\nKV Cache内存对比 (batch={batch}, layers={layers}, seq={seq_len}):")
    
    # Multi-Head Attention
    mha_cache = 2 * batch * layers * num_heads * seq_len * head_dim * dtype_size
    print(f"  MHA ({num_heads} heads): {mha_cache / 1024 / 1024:.2f} MB")
    
    # Multi-Query Attention
    mqa_cache = 2 * batch * layers * 1 * seq_len * head_dim * dtype_size
    print(f"  MQA (1 head): {mqa_cache / 1024 / 1024:.2f} MB")
    
    # Grouped-Query Attention (8 kv heads)
    num_kv = 8
    gqa_cache = 2 * batch * layers * num_kv * seq_len * head_dim * dtype_size
    print(f"  GQA ({num_kv} heads): {gqa_cache / 1024 / 1024:.2f} MB")
    
    print(f"  MQA节省: {(1 - mqa_cache / mha_cache) * 100:.1f}%")
    print(f"  GQA节省: {(1 - gqa_cache / mha_cache) * 100:.1f}%")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Attention优化对比")
    print("=" * 50)
    
    # 1. 内存分析
    print("\n--- 标准Attention内存占用 ---")
    analyze_memory(2048, 32, 128, batch=1)
    analyze_memory(8192, 32, 128, batch=1)  # 长序列
    
    # 2. 标准Attention vs Flash Attention
    print("\n--- 计算对比 ---")
    batch, heads, seq_len, head_dim = 2, 8, 128, 64
    
    Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
    
    # 标准
    start = time.perf_counter()
    out_std = standard_attention(Q, K, V)
    std_time = time.perf_counter() - start
    print(f"标准Attention: {std_time * 1000:.2f} ms")
    
    # Flash（简化版）
    start = time.perf_counter()
    out_flash = flash_attention_simplified(Q, K, V, block_size=32)
    flash_time = time.perf_counter() - start
    print(f"Flash Attention: {flash_time * 1000:.2f} ms")
    
    # 验证
    diff = np.max(np.abs(out_std - out_flash))
    print(f"输出差异: {diff:.6f}")
    
    # 3. MQA vs GQA
    print("\n--- MQA/GQA对比 ---")
    Q = np.random.randn(1, 32, 64, 64).astype(np.float32)
    K_mqa = np.random.randn(1, 1, 64, 64).astype(np.float32)
    V_mqa = np.random.randn(1, 1, 64, 64).astype(np.float32)
    K_gqa = np.random.randn(1, 8, 64, 64).astype(np.float32)
    V_gqa = np.random.randn(1, 8, 64, 64).astype(np.float32)
    
    out_mqa = multi_query_attention(Q, K_mqa, V_mqa)
    out_gqa = grouped_query_attention(Q, K_gqa, V_gqa, num_kv_heads=8)
    
    print(f"MQA输出形状: {out_mqa.shape}")
    print(f"GQA输出形状: {out_gqa.shape}")
    
    # 4. KV Cache对比
    compare_kv_cache_memory(
        batch=1, layers=32, seq_len=2048,
        num_heads=32, head_dim=128
    )
    
    print("\n总结: Flash Attention减少内存访问，MQA/GQA减少KV Cache")
