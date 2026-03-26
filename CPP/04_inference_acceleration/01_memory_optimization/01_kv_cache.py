"""
01_kv_cache.py - KV Cache实现

【核心概念】
在自回归生成中，每个token的生成都需要计算所有之前token的K和V。
KV Cache缓存已计算的K、V，避免重复计算。

【内存占用】
KV Cache大小 = 2 * layers * batch * seq_len * heads * head_dim * dtype_size

例如 LLaMA-7B，seq_len=2048:
= 2 * 32 * 1 * 2048 * 32 * 128 * 2 bytes (FP16)
≈ 1 GB

运行: python 01_kv_cache.py
"""

import numpy as np
from typing import Tuple, Optional, List
import time

# ============================================================================
# 第一部分：基础KV Cache实现
# ============================================================================

class KVCache:
    """
    简单的KV Cache实现
    
    存储格式: (batch, num_heads, seq_len, head_dim)
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype=np.float32
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        
        # 预分配内存
        self.k_cache = [
            np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        
        # 当前序列长度
        self.seq_len = 0
        
        print(f"KV Cache初始化:")
        print(f"  形状: ({max_batch_size}, {num_heads}, {max_seq_len}, {head_dim})")
        print(f"  每层内存: {self.k_cache[0].nbytes / 1024 / 1024:.2f} MB * 2 = {self.k_cache[0].nbytes * 2 / 1024 / 1024:.2f} MB")
        print(f"  总内存: {self.total_memory() / 1024 / 1024:.2f} MB")
    
    def total_memory(self) -> int:
        """计算总内存占用"""
        return sum(k.nbytes + v.nbytes for k, v in zip(self.k_cache, self.v_cache))
    
    def update(
        self, 
        layer_idx: int, 
        new_k: np.ndarray, 
        new_v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新缓存并返回完整的K、V
        
        new_k, new_v: (batch, num_heads, new_len, head_dim)
        """
        new_len = new_k.shape[2]
        end_pos = self.seq_len + new_len
        
        if end_pos > self.max_seq_len:
            raise ValueError(f"序列长度超出最大值: {end_pos} > {self.max_seq_len}")
        
        # 存入缓存
        batch = new_k.shape[0]
        self.k_cache[layer_idx][:batch, :, self.seq_len:end_pos, :] = new_k
        self.v_cache[layer_idx][:batch, :, self.seq_len:end_pos, :] = new_v
        
        # 返回完整的K、V（包括历史）
        full_k = self.k_cache[layer_idx][:batch, :, :end_pos, :]
        full_v = self.v_cache[layer_idx][:batch, :, :end_pos, :]
        
        return full_k, full_v
    
    def advance(self, steps: int = 1):
        """推进序列位置"""
        self.seq_len += steps
    
    def reset(self):
        """重置缓存"""
        self.seq_len = 0
        # 可选：清零（通常不需要，会被覆盖）
        # for k, v in zip(self.k_cache, self.v_cache):
        #     k.fill(0)
        #     v.fill(0)

# ============================================================================
# 第二部分：模拟Attention计算
# ============================================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Q: (batch, heads, q_len, head_dim)
    K: (batch, heads, kv_len, head_dim)
    V: (batch, heads, kv_len, head_dim)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
    
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, V)
    return output


def attention_with_kv_cache(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    kv_cache: KVCache,
    layer_idx: int
) -> np.ndarray:
    """带KV Cache的Attention"""
    
    # 更新缓存
    full_K, full_V = kv_cache.update(layer_idx, K, V)
    
    # 使用完整的K、V计算attention
    output = scaled_dot_product_attention(Q, full_K, full_V)
    
    return output

# ============================================================================
# 第三部分：无缓存 vs 有缓存对比
# ============================================================================

def demo_without_cache():
    """无缓存：每次都重新计算所有K、V"""
    print("\n--- 无KV Cache ---")
    
    batch, heads, head_dim = 1, 32, 64
    total_len = 100
    
    start = time.perf_counter()
    
    all_K = []
    all_V = []
    
    for step in range(total_len):
        # 生成当前token的Q、K、V
        Q = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        K = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        V = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        
        all_K.append(K)
        all_V.append(V)
        
        # 每次都用所有历史K、V（重新拼接）
        full_K = np.concatenate(all_K, axis=2)
        full_V = np.concatenate(all_V, axis=2)
        
        output = scaled_dot_product_attention(Q, full_K, full_V)
    
    elapsed = time.perf_counter() - start
    print(f"生成 {total_len} tokens，耗时: {elapsed*1000:.2f} ms")


def demo_with_cache():
    """有缓存：只计算新token的K、V"""
    print("\n--- 有KV Cache ---")
    
    batch, heads, head_dim = 1, 32, 64
    num_layers = 1
    total_len = 100
    
    kv_cache = KVCache(
        max_batch_size=batch,
        max_seq_len=total_len,
        num_heads=heads,
        head_dim=head_dim,
        num_layers=num_layers
    )
    
    start = time.perf_counter()
    
    for step in range(total_len):
        Q = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        K = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        V = np.random.randn(batch, heads, 1, head_dim).astype(np.float32)
        
        output = attention_with_kv_cache(Q, K, V, kv_cache, layer_idx=0)
        kv_cache.advance(1)
    
    elapsed = time.perf_counter() - start
    print(f"生成 {total_len} tokens，耗时: {elapsed*1000:.2f} ms")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("KV Cache 实现")
    print("=" * 50)
    
    # 1. 创建KV Cache
    print("\n--- 创建KV Cache ---")
    cache = KVCache(
        max_batch_size=4,
        max_seq_len=2048,
        num_heads=32,
        head_dim=128,
        num_layers=32,
        dtype=np.float16
    )
    
    # 2. 模拟使用
    print("\n--- 模拟推理 ---")
    batch, heads, head_dim = 1, 32, 128
    
    # Prefill阶段：处理prompt
    prompt_len = 10
    Q = np.random.randn(batch, heads, prompt_len, head_dim).astype(np.float16)
    K = np.random.randn(batch, heads, prompt_len, head_dim).astype(np.float16)
    V = np.random.randn(batch, heads, prompt_len, head_dim).astype(np.float16)
    
    full_K, full_V = cache.update(0, K, V)
    cache.seq_len = prompt_len  # 设置序列长度
    print(f"Prefill: 处理 {prompt_len} tokens")
    print(f"Cache K shape: {full_K.shape}")
    
    # Decode阶段：逐token生成
    for i in range(5):
        Q = np.random.randn(batch, heads, 1, head_dim).astype(np.float16)
        K = np.random.randn(batch, heads, 1, head_dim).astype(np.float16)
        V = np.random.randn(batch, heads, 1, head_dim).astype(np.float16)
        
        full_K, full_V = cache.update(0, K, V)
        cache.advance(1)
        print(f"Decode step {i+1}: cache长度 = {cache.seq_len}")
    
    # 3. 性能对比
    demo_without_cache()
    demo_with_cache()
    
    print("\n总结: KV Cache是自回归生成的核心优化")
