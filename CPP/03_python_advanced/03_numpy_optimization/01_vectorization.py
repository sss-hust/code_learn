"""
01_vectorization.py - NumPy向量化

【推理加速场景】
- 矩阵运算替代循环
- 批量数据处理
- 激活函数计算
- 张量操作

运行: python 01_vectorization.py
"""

import numpy as np
import time
from typing import Callable

# ============================================================================
# 性能计时工具
# ============================================================================

def benchmark(func: Callable, *args, iterations: int = 100) -> float:
    """测量函数执行时间（毫秒）"""
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    return elapsed, result


# ============================================================================
# 第一部分：循环 vs 向量化
# ============================================================================

def add_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """使用循环的加法（慢）"""
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i]
    return result


def add_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """向量化加法（快）"""
    return a + b


def relu_loop(x: np.ndarray) -> np.ndarray:
    """循环ReLU"""
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = x[i] if x[i] > 0 else 0
    return result


def relu_vectorized(x: np.ndarray) -> np.ndarray:
    """向量化ReLU"""
    return np.maximum(x, 0)


# ============================================================================
# 第二部分：常用激活函数
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活"""
    return 1 / (1 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU激活（Transformer中常用）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish激活（LLaMA中常用）"""
    return x * sigmoid(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 减去最大值防止溢出
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# 第三部分：矩阵运算
# ============================================================================

def matmul_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """循环矩阵乘法"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """向量化矩阵乘法"""
    return A @ B  # 或 np.matmul(A, B)


def linear_layer(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """线性层: y = xW + b"""
    return x @ weight + bias


# ============================================================================
# 第四部分：广播机制
# ============================================================================

def demo_broadcasting():
    """广播机制演示"""
    print("\n--- 广播机制 ---")
    
    # 1. 标量与数组
    a = np.array([1, 2, 3])
    print(f"a + 10 = {a + 10}")
    
    # 2. 不同形状数组
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20, 30])  # (3,)
    
    print(f"A + b =\n{A + b}")  # 自动广播
    
    # 3. LayerNorm中的应用
    x = np.random.randn(2, 3, 4)  # (batch, seq, hidden)
    mean = x.mean(axis=-1, keepdims=True)  # (2, 3, 1)
    std = x.std(axis=-1, keepdims=True)    # (2, 3, 1)
    normalized = (x - mean) / (std + 1e-6)  # 广播到(2, 3, 4)
    
    print(f"LayerNorm输入: {x.shape}")
    print(f"归一化后: {normalized.shape}")


# ============================================================================
# 第五部分：Attention计算
# ============================================================================

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    缩放点积注意力
    
    Q: (batch, heads, seq_q, head_dim)
    K: (batch, heads, seq_k, head_dim)
    V: (batch, heads, seq_k, head_dim)
    
    返回: (batch, heads, seq_q, head_dim)
    """
    d_k = Q.shape[-1]
    
    # Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Softmax
    attention_weights = softmax(scores, axis=-1)
    
    # 加权求和
    output = np.matmul(attention_weights, V)
    
    return output


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("NumPy 向量化")
    print("=" * 50)
    
    # 准备数据
    N = 100000
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    
    # 1. 加法对比
    print("\n--- 加法性能对比 ---")
    loop_time, _ = benchmark(add_loop, a, b, iterations=10)
    vec_time, _ = benchmark(add_vectorized, a, b, iterations=1000)
    print(f"循环: {loop_time:.2f} ms")
    print(f"向量化: {vec_time:.4f} ms")
    print(f"加速比: {loop_time/vec_time:.0f}x")
    
    # 2. ReLU对比
    print("\n--- ReLU性能对比 ---")
    x = np.random.randn(N).astype(np.float32)
    loop_time, _ = benchmark(relu_loop, x, iterations=10)
    vec_time, _ = benchmark(relu_vectorized, x, iterations=1000)
    print(f"循环: {loop_time:.2f} ms")
    print(f"向量化: {vec_time:.4f} ms")
    print(f"加速比: {loop_time/vec_time:.0f}x")
    
    # 3. 激活函数
    print("\n--- 激活函数 ---")
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    print(f"输入: {x}")
    print(f"ReLU: {relu_vectorized(x)}")
    print(f"Sigmoid: {np.round(sigmoid(x), 3)}")
    print(f"GELU: {np.round(gelu(x), 3)}")
    print(f"SiLU: {np.round(silu(x), 3)}")
    
    # 4. 矩阵乘法
    print("\n--- 矩阵乘法 ---")
    M, K, N = 64, 128, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    vec_time, _ = benchmark(matmul_vectorized, A, B, iterations=100)
    print(f"矩阵乘法 ({M}x{K}) @ ({K}x{N}): {vec_time:.2f} ms")
    
    # 5. 广播
    demo_broadcasting()
    
    # 6. Attention
    print("\n--- Attention ---")
    batch, heads, seq, dim = 2, 8, 32, 64
    Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    V = np.random.randn(batch, heads, seq, dim).astype(np.float32)
    
    output = scaled_dot_product_attention(Q, K, V)
    print(f"输入 Q: {Q.shape}")
    print(f"输出: {output.shape}")
    
    attn_time, _ = benchmark(scaled_dot_product_attention, Q, K, V, iterations=100)
    print(f"Attention耗时: {attn_time:.2f} ms")
    
    print("\n总结: 向量化可带来100x以上加速！")
