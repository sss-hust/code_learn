"""
09_fused_add_rmsnorm - Triton 融合 Add + RMSNorm

【核心概念】
- Transformer 中的 residual 连接: x = x + residual
- 紧接着做 RMSNorm: output = rms_norm(x + residual)
- 融合优势：一次读写完成两个操作，减少 global memory 访问
- 实际模型中这两步总是连续出现的

【任务】
实现一个 Triton kernel，融合 residual add 和 RMS Normalization:
    x_new = x + residual
    output = rms_norm(x_new) = (x_new / rms(x_new)) * weight
同时原地更新 x（供下一层 residual 使用）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr,          # 输入/输出（原地更新）
    residual_ptr,   # residual 输入
    output_ptr,     # 归一化后的输出
    weight_ptr,     # RMSNorm 的 weight
    n_cols,
    eps,
    stride_x,
    stride_r,
    stride_o,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Add + RMSNorm kernel
    
    提示：
    1. 加载 x 和 residual
    2. x_new = x + residual（原地更新 x）
    3. 计算 rms = sqrt(mean(x_new^2) + eps)
    4. output = (x_new / rms) * weight
    5. 写回 x_new 到 x_ptr（原地），写 output 到 output_ptr
    """
    # TODO: 在此实现你的代码
    pass


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float = 1e-6):
    """融合 Add + RMSNorm 的 Python 封装
    
    返回: (output, x_updated)
    - output: 归一化后的结果
    - x 被原地更新为 x + residual
    """
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    
    fused_add_rmsnorm_kernel[(n_rows,)](
        x, residual, output, weight, n_cols, eps,
        x.stride(0), residual.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda')
    residual = torch.randn(128, 512, device='cuda')
    weight = torch.ones(512, device='cuda')
    
    x_ref = x.clone()
    x_new = x_ref + residual
    rms = torch.sqrt(torch.mean(x_new ** 2, dim=-1, keepdim=True) + 1e-6)
    expected = (x_new / rms) * weight
    
    output = fused_add_rmsnorm(x, residual, weight)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"x 原地更新: {'正确 ✓' if torch.allclose(x, x_new, atol=1e-5) else '失败 ✗'}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
