"""
03_layer_norm - Triton 层归一化

【核心概念】
- Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
- 每个 program 处理一行（一个 token 的所有特征）
- 需要计算行的均值和方差

【任务】
实现一个 Triton kernel，对 2D 张量的每一行做 Layer Normalization
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,       # 可学习的缩放参数 gamma
    bias_ptr,         # 可学习的偏移参数 beta
    n_cols,
    eps,
    x_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization kernel
    
    提示：
    1. 加载一行数据
    2. 计算均值: mean = sum(x) / n_cols
    3. 计算方差: var = sum((x - mean)^2) / n_cols
    4. 归一化: x_norm = (x - mean) / sqrt(var + eps)
    5. 应用 weight 和 bias: output = x_norm * weight + bias
    """
    # TODO: 在此实现你的代码
    pass


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Layer Normalization 的 Python 封装"""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    
    layer_norm_kernel[(n_rows,)](
        x, output, weight, bias, n_cols, eps,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device='cuda')
    weight = torch.ones(512, device='cuda')
    bias = torch.zeros(512, device='cuda')
    
    output = layer_norm(x, weight, bias)
    expected = torch.nn.functional.layer_norm(x, (512,), weight, bias)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
