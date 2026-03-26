"""
09_fused_add_rmsnorm - 参考答案

【关键点】
1. 融合 = 一个 kernel 完成两个操作，减少 global memory 读写
2. 非融合版本：读 x → 写 x+res → 读 x+res → 写 output（4次全局访存）
3. 融合版本：读 x, res → 写 x+res 和 output（3次全局访存，且过程中数据在寄存器中）
4. x 的原地更新是为了后续层的 residual 连接
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    n_cols,
    eps,
    stride_x,
    stride_r,
    stride_o,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载 x 和 residual
    x_start = row_idx * stride_x
    r_start = row_idx * stride_r
    x = tl.load(x_ptr + x_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + r_start + col_offsets, mask=mask, other=0.0)
    
    # Add: x_new = x + residual
    x_new = x + residual
    
    # 原地更新 x
    tl.store(x_ptr + x_start + col_offsets, x_new, mask=mask)
    
    # RMSNorm
    x_sq = tl.where(mask, x_new * x_new, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)
    x_norm = x_new / rms
    
    # 应用 weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = x_norm * weight
    
    # 写回
    o_start = row_idx * stride_o
    tl.store(output_ptr + o_start + col_offsets, output, mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float = 1e-6):
    """融合 Add + RMSNorm 的 Python 封装"""
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
