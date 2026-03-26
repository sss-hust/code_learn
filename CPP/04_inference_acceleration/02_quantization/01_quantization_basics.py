"""
01_quantization_basics.py - 量化基础

【核心概念】
量化将浮点数映射到低精度整数：
- FP32 -> INT8: 减少4倍内存，加速计算
- FP16 -> INT4: 减少4倍内存

量化公式:
q = round(x / scale) + zero_point
x = (q - zero_point) * scale

运行: python 01_quantization_basics.py
"""

import numpy as np
from typing import Tuple

# ============================================================================
# 第一部分：对称量化
# ============================================================================

def symmetric_quantize(
    x: np.ndarray, 
    bits: int = 8
) -> Tuple[np.ndarray, float]:
    """
    对称量化：zero_point = 0
    
    适用于权重（通常以0为中心分布）
    """
    # 计算scale
    max_val = np.max(np.abs(x))
    q_max = (1 << (bits - 1)) - 1  # 127 for INT8
    scale = max_val / q_max
    
    # 量化
    x_q = np.round(x / scale).astype(np.int8)
    x_q = np.clip(x_q, -q_max - 1, q_max)
    
    return x_q, scale


def symmetric_dequantize(x_q: np.ndarray, scale: float) -> np.ndarray:
    """反量化"""
    return x_q.astype(np.float32) * scale


# ============================================================================
# 第二部分：非对称量化
# ============================================================================

def asymmetric_quantize(
    x: np.ndarray, 
    bits: int = 8
) -> Tuple[np.ndarray, float, int]:
    """
    非对称量化：带zero_point
    
    适用于激活值（可能不以0为中心，如ReLU后的值）
    """
    x_min, x_max = x.min(), x.max()
    q_min, q_max = 0, (1 << bits) - 1  # 0-255 for UINT8
    
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = int(round(q_min - x_min / scale))
    zero_point = np.clip(zero_point, q_min, q_max)
    
    x_q = np.round(x / scale + zero_point).astype(np.uint8)
    x_q = np.clip(x_q, q_min, q_max)
    
    return x_q, scale, zero_point


def asymmetric_dequantize(
    x_q: np.ndarray, 
    scale: float, 
    zero_point: int
) -> np.ndarray:
    """反量化"""
    return (x_q.astype(np.float32) - zero_point) * scale


# ============================================================================
# 第三部分：逐通道量化
# ============================================================================

def per_channel_quantize(
    weight: np.ndarray,  # (out_features, in_features)
    bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐通道量化：每个输出通道有独立的scale
    
    精度更高，适用于权重
    """
    out_features = weight.shape[0]
    q_max = (1 << (bits - 1)) - 1
    
    scales = np.zeros(out_features, dtype=np.float32)
    weight_q = np.zeros_like(weight, dtype=np.int8)
    
    for i in range(out_features):
        max_val = np.max(np.abs(weight[i]))
        scales[i] = max_val / q_max if max_val > 0 else 1.0
        weight_q[i] = np.round(weight[i] / scales[i]).astype(np.int8)
    
    return weight_q, scales


def per_channel_dequantize(
    weight_q: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    """逐通道反量化"""
    return weight_q.astype(np.float32) * scales[:, np.newaxis]


# ============================================================================
# 第四部分：量化误差分析
# ============================================================================

def analyze_quantization_error(
    original: np.ndarray,
    quantized: np.ndarray,
    dequantized: np.ndarray
):
    """分析量化误差"""
    error = original - dequantized
    
    print("量化误差分析:")
    print(f"  原始范围: [{original.min():.4f}, {original.max():.4f}]")
    print(f"  量化范围: [{quantized.min()}, {quantized.max()}]")
    print(f"  反量化范围: [{dequantized.min():.4f}, {dequantized.max():.4f}]")
    print(f"  最大绝对误差: {np.max(np.abs(error)):.6f}")
    print(f"  均方误差 (MSE): {np.mean(error**2):.6f}")
    print(f"  相对误差: {np.mean(np.abs(error / (original + 1e-8))) * 100:.2f}%")


# ============================================================================
# 第五部分：INT8矩阵乘法
# ============================================================================

def int8_matmul(
    A_q: np.ndarray,  # INT8
    B_q: np.ndarray,  # INT8
    scale_A: float,
    scale_B: float
) -> np.ndarray:
    """
    INT8矩阵乘法
    
    C = A @ B
    C_fp = (A_q @ B_q) * (scale_A * scale_B)
    """
    # INT8乘法，INT32累加
    C_int32 = A_q.astype(np.int32) @ B_q.astype(np.int32)
    
    # 反量化
    C_fp = C_int32.astype(np.float32) * (scale_A * scale_B)
    
    return C_fp


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("量化基础")
    print("=" * 50)
    
    # 1. 对称量化
    print("\n--- 对称量化 (权重) ---")
    weights = np.random.randn(256, 768).astype(np.float32)
    weights_q, scale = symmetric_quantize(weights)
    weights_deq = symmetric_dequantize(weights_q, scale)
    
    print(f"原始: FP32, {weights.nbytes / 1024:.2f} KB")
    print(f"量化: INT8, {weights_q.nbytes / 1024:.2f} KB")
    print(f"压缩比: {weights.nbytes / weights_q.nbytes}x")
    analyze_quantization_error(weights, weights_q, weights_deq)
    
    # 2. 非对称量化
    print("\n--- 非对称量化 (激活值) ---")
    activations = np.abs(np.random.randn(32, 768).astype(np.float32))  # ReLU后
    act_q, scale, zp = asymmetric_quantize(activations)
    act_deq = asymmetric_dequantize(act_q, scale, zp)
    
    print(f"zero_point: {zp}")
    analyze_quantization_error(activations, act_q, act_deq)
    
    # 3. 逐通道量化
    print("\n--- 逐通道量化 ---")
    weights = np.random.randn(256, 768).astype(np.float32)
    # 人为使不同通道有不同范围
    weights[0] *= 10
    weights[1] *= 0.1
    
    # 逐tensor量化
    weights_q1, scale1 = symmetric_quantize(weights)
    weights_deq1 = symmetric_dequantize(weights_q1, scale1)
    
    # 逐通道量化
    weights_q2, scales2 = per_channel_quantize(weights)
    weights_deq2 = per_channel_dequantize(weights_q2, scales2)
    
    print("逐Tensor量化:")
    print(f"  MSE: {np.mean((weights - weights_deq1)**2):.6f}")
    
    print("逐通道量化:")
    print(f"  MSE: {np.mean((weights - weights_deq2)**2):.6f}")
    
    # 4. INT8矩阵乘法
    print("\n--- INT8矩阵乘法 ---")
    A = np.random.randn(32, 128).astype(np.float32)
    B = np.random.randn(128, 64).astype(np.float32)
    
    C_fp32 = A @ B  # FP32参考
    
    A_q, scale_A = symmetric_quantize(A)
    B_q, scale_B = symmetric_quantize(B)
    C_int8 = int8_matmul(A_q, B_q, scale_A, scale_B)
    
    error = np.max(np.abs(C_fp32 - C_int8))
    print(f"FP32 结果范围: [{C_fp32.min():.4f}, {C_fp32.max():.4f}]")
    print(f"INT8 结果范围: [{C_int8.min():.4f}, {C_int8.max():.4f}]")
    print(f"最大误差: {error:.4f}")
    
    print("\n总结: 量化可显著减少内存和加速计算")
