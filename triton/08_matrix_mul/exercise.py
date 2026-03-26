"""
08_matrix_mul - Triton 矩阵乘法 (GEMM)

【核心概念】
- GEMM: C = A @ B, A(M,K), B(K,N), C(M,N)
- 分块计算（Tiling）：将大矩阵分成小 tile，逐块累加
- 每个 program 负责 C 矩阵的一个 (BLOCK_M, BLOCK_N) 块
- 内循环沿 K 维度累加：C_tile += A_tile @ B_tile

【任务】
实现一个 Triton GEMM kernel
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
    """
    矩阵乘法 kernel: C = A @ B
    
    grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
    每个 program 计算 C 的一个 (BLOCK_M, BLOCK_N) 块
    
    提示：
    1. 从 program_id 计算出 (block_m, block_n) 索引
    2. 初始化累加器 acc = zeros((BLOCK_M, BLOCK_N))
    3. 沿 K 维度循环：
       - 加载 A 的 (BLOCK_M, BLOCK_K) 块
       - 加载 B 的 (BLOCK_K, BLOCK_N) 块
       - acc += tl.dot(a_tile, b_tile)
    4. 将 acc 写回 C
    """
    # TODO: 在此实现你的代码
    pass


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
