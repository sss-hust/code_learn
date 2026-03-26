"""
03_layer_norm - 参考答案

【关键点】
1. 均值和方差通过 tl.sum 在行方向上归约
2. eps 防止除零
3. weight/bias 是逐元素操作，与 softmax 的归一化不同
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
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
    
    # 计算均值
    mean = tl.sum(x, axis=0) / n_cols
    
    # 计算方差
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    
    # 归一化
    x_norm = x_centered / tl.sqrt(var + eps)
    
    # 加载 weight 和 bias，应用仿射变换
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    bias = tl.load(bias_ptr + col_offsets, mask=mask)
    output = x_norm * weight + bias
    
    # 写回
    output_start = row_idx * output_row_stride
    tl.store(output_ptr + output_start + col_offsets, output, mask=mask)


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
