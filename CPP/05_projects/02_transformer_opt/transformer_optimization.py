"""
transformer_optimization.py - Transformer优化对比

对比优化前后的Transformer实现：
1. 朴素实现
2. 优化实现（KV Cache、融合算子等）
3. 性能对比

运行: python transformer_optimization.py
"""

import numpy as np
import time
from typing import Optional, Tuple

# ============================================================================
# 第一部分：朴素Transformer
# ============================================================================

class NaiveTransformer:
    """朴素Transformer实现（无优化）"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers
        
        # 初始化权重
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'Wq': np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                'Wk': np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                'Wv': np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                'Wo': np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                'W1': np.random.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.02,
                'W2': np.random.randn(hidden_size * 4, hidden_size).astype(np.float32) * 0.02,
            })
    
    def attention(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """标准attention（每次重新计算所有KV）"""
        batch, seq_len, _ = x.shape
        layer = self.layers[layer_idx]
        
        # 投影
        Q = x @ layer['Wq']
        K = x @ layer['Wk']
        V = x @ layer['Wv']
        
        # Reshape
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Output
        out = np.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)
        out = out @ layer['Wo']
        
        return out
    
    def ffn(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """前馈网络"""
        layer = self.layers[layer_idx]
        h = x @ layer['W1']
        h = np.maximum(h, 0)  # ReLU
        return h @ layer['W2']
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """完整前向"""
        for i in range(self.num_layers):
            x = x + self.attention(x, i)
            x = x + self.ffn(x, i)
        return x
    
    def generate(self, prompt: np.ndarray, max_tokens: int) -> np.ndarray:
        """朴素生成：每次重新计算所有"""
        current = prompt.copy()
        
        for _ in range(max_tokens):
            output = self.forward(current)
            next_token = np.random.randn(1, 1, self.hidden_size).astype(np.float32)
            current = np.concatenate([current, next_token], axis=1)
        
        return current


# ============================================================================
# 第二部分：优化Transformer
# ============================================================================

class OptimizedTransformer:
    """优化的Transformer实现"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, max_seq_len: int = 2048):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 权重（与朴素版本相同）
        self.layers = []
        for _ in range(num_layers):
            # 优化1：融合QKV投影
            self.layers.append({
                'Wqkv': np.random.randn(hidden_size, 3 * hidden_size).astype(np.float32) * 0.02,
                'Wo': np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                'W1': np.random.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.02,
                'W2': np.random.randn(hidden_size * 4, hidden_size).astype(np.float32) * 0.02,
            })
        
        # KV Cache
        self.kv_cache = None
    
    def init_kv_cache(self, batch_size: int):
        """初始化KV Cache"""
        self.kv_cache = []
        for _ in range(self.num_layers):
            self.kv_cache.append({
                'k': np.zeros((batch_size, self.num_heads, self.max_seq_len, self.head_dim), dtype=np.float32),
                'v': np.zeros((batch_size, self.num_heads, self.max_seq_len, self.head_dim), dtype=np.float32),
            })
        self.cache_len = 0
    
    def attention_with_cache(
        self, 
        x: np.ndarray, 
        layer_idx: int, 
        use_cache: bool = True
    ) -> np.ndarray:
        """带KV Cache的Attention"""
        batch, seq_len, _ = x.shape
        layer = self.layers[layer_idx]
        
        # 优化2：融合QKV投影
        qkv = x @ layer['Wqkv']
        Q, K, V = np.split(qkv, 3, axis=-1)
        
        # Reshape
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        if use_cache and self.kv_cache is not None:
            # 更新cache
            cache = self.kv_cache[layer_idx]
            start = self.cache_len
            end = start + seq_len
            cache['k'][:batch, :, start:end, :] = K
            cache['v'][:batch, :, start:end, :] = V
            
            # 使用完整的KV
            K = cache['k'][:batch, :, :end, :]
            V = cache['v'][:batch, :, :end, :]
        
        # Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        out = np.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)
        out = out @ layer['Wo']
        
        return out
    
    def ffn_fused(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """融合FFN（理论上可以融合更多操作）"""
        layer = self.layers[layer_idx]
        # 这里模拟融合：减少中间结果的内存分配
        return np.maximum(x @ layer['W1'], 0) @ layer['W2']
    
    def forward(self, x: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """前向传播"""
        for i in range(self.num_layers):
            x = x + self.attention_with_cache(x, i, use_cache)
            x = x + self.ffn_fused(x, i)
        return x
    
    def generate(self, prompt: np.ndarray, max_tokens: int) -> np.ndarray:
        """优化生成：使用KV Cache"""
        batch = prompt.shape[0]
        self.init_kv_cache(batch)
        
        # Prefill
        current = prompt.copy()
        output = self.forward(current, use_cache=True)
        self.cache_len = prompt.shape[1]
        
        # Decode
        for _ in range(max_tokens):
            # 只处理最后一个token
            last_hidden = output[:, -1:, :]
            output = self.forward(last_hidden, use_cache=True)
            self.cache_len += 1
        
        return self.cache_len


# ============================================================================
# 第三部分：性能对比
# ============================================================================

def benchmark_generation(model, prompt, max_tokens, name):
    """测试生成性能"""
    # 预热
    _ = model.generate(prompt.copy(), 2)
    
    # 正式测试
    start = time.perf_counter()
    result = model.generate(prompt.copy(), max_tokens)
    elapsed = time.perf_counter() - start
    
    print(f"{name}:")
    print(f"  生成 {max_tokens} tokens")
    print(f"  总耗时: {elapsed * 1000:.2f} ms")
    print(f"  每token: {elapsed / max_tokens * 1000:.2f} ms")
    
    return elapsed


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Transformer优化对比")
    print("=" * 60)
    
    # 配置
    hidden_size = 256
    num_heads = 8
    num_layers = 4
    prompt_len = 32
    max_tokens = 20
    
    print(f"\n配置:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  max_tokens: {max_tokens}")
    
    # 创建模型
    naive = NaiveTransformer(hidden_size, num_heads, num_layers)
    optimized = OptimizedTransformer(hidden_size, num_heads, num_layers)
    
    # 测试数据
    prompt = np.random.randn(1, prompt_len, hidden_size).astype(np.float32)
    
    # 性能对比
    print("\n" + "-" * 40)
    print("性能对比")
    print("-" * 40)
    
    naive_time = benchmark_generation(naive, prompt, max_tokens, "朴素实现")
    print()
    opt_time = benchmark_generation(optimized, prompt, max_tokens, "优化实现(KV Cache)")
    
    print(f"\n加速比: {naive_time / opt_time:.2f}x")
    
    # 复杂度分析
    print("\n" + "-" * 40)
    print("复杂度分析")
    print("-" * 40)
    
    total_len = prompt_len + max_tokens
    
    # 朴素：每次都重新计算
    naive_flops = 0
    for t in range(prompt_len, total_len):
        naive_flops += t * t  # O(n^2) attention for each step
    
    # 优化：增量计算
    opt_prefill = prompt_len * prompt_len
    opt_decode = max_tokens * total_len  # 每次只计算新增的
    opt_flops = opt_prefill + opt_decode
    
    print(f"朴素实现 Attention计算量: O({naive_flops})")
    print(f"优化实现 Attention计算量: O({opt_flops})")
    print(f"理论加速: {naive_flops / opt_flops:.2f}x")
    
    # 内存分析
    print("\n" + "-" * 40)
    print("内存分析")
    print("-" * 40)
    
    # KV Cache内存
    kv_cache_size = 2 * num_layers * 1 * num_heads * total_len * (hidden_size // num_heads) * 4
    print(f"KV Cache大小: {kv_cache_size / 1024:.2f} KB")
    print(f"用空间换时间，显著减少计算量")
    
    print("\n" + "=" * 60)
    print("总结: KV Cache将生成复杂度从O(n^2)降到O(n)")
    print("=" * 60)
