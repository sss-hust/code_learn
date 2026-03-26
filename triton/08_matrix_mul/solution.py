"""
08_matrix_mul - 参考答案

【关键点】
1. 2D grid → 1D program_id 的映射: pid_m = pid // num_n, pid_n = pid % num_n
2. 沿 K 维度的内循环：每次加载 BLOCK_K 列的 A 和 BLOCK_K 行的 B
3. tl.dot 执行小矩阵乘法（硬件加速）
4. FP16 输入通常用 FP32 累加器避免精度损失
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 从 1D program_id 映射到 2D 块坐标
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n
    
    # 计算当前块在 C 中的行列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_k = tl.arange(0, BLOCK_K)                     # (BLOCK_K,)
    
    # A 和 B 的指针偏移
    # A: (BLOCK_M, BLOCK_K) 块的起始地址
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B: (BLOCK_K, BLOCK_N) 块的起始地址
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加器（用 FP32 防止精度损失）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 沿 K 维度分块累加
    for k_start in range(0, K, BLOCK_K):
        # 边界检查
        k_offs = k_start + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        # 加载 A 和 B 的当前块
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 块矩阵乘法
        acc += tl.dot(a, b)
        
        # 移动到下一个 K 块
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # 将结果写回 C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """矩阵乘法的 Python 封装"""
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    output = matmul(a, b)
    expected = a @ b
    
    print(f"输出形状: {output.shape}")
    print(f"最大误差: {(output - expected).abs().max().item():.4f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-1, rtol=1e-2) else '失败 ✗'}")
