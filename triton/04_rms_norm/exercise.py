"""
04_rms_norm - Triton RMS Normalization

【核心概念】
- RMS Normalization: y = x / sqrt(mean(x^2) + eps) * weight
- 相比 LayerNorm：不需要减去均值，也没有 bias
- LLaMA / Qwen 等模型的主流归一化方法

【任务】
实现一个 Triton kernel，对 2D 张量的每一行做 RMS Normalization
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
    """
    RMS Normalization kernel
    
    提示：
    1. 加载一行数据
    2. 计算 RMS: rms = sqrt(mean(x^2) + eps)
    3. 归一化: output = (x / rms) * weight
    """
    # TODO: 在此实现你的代码
    pass


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
