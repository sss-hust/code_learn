"""
04_rms_norm - 参考答案

【关键点】
1. RMS = sqrt(mean(x^2) + eps)，不需要减均值
2. 比 LayerNorm 少一次归约（不需要计算均值）
3. 推理时计算更快，是现代 LLM 的主流归一化方式
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    n_cols,
    eps,
    x_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载一行数据
    row_start = row_idx * x_row_stride
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # 计算 mean(x^2)
    x_sq = tl.where(mask, x * x, 0.0)
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    
    # RMS = sqrt(mean(x^2) + eps)
    rms = tl.sqrt(mean_sq + eps)
    
    # 归一化
    x_norm = x / rms
    
    # 应用 weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    output = x_norm * weight
    
    # 写回
    output_start = row_idx * output_row_stride
    tl.store(output_ptr + output_start + col_offsets, output, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    """RMS Normalization 的 Python 封装"""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    
    rms_norm_kernel[(n_rows,)](
        x, output, weight, n_cols, eps,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """PyTorch 参考实现"""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda')
    weight = torch.ones(512, device='cuda')
    
    output = rms_norm(x, weight)
    expected = rms_norm_ref(x, weight)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
