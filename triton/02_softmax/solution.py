"""
02_softmax - 参考答案

【关键点】
1. 每个 program 处理一行 → grid = (n_rows,)
2. BLOCK_SIZE 取 n_cols 的上取二次幂，确保每行一次性加载
3. 数值稳定：先减 max 再 exp，避免浮点溢出
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前行索引
    row_idx = tl.program_id(0)
    
    # 当前行的起始地址
    row_start = row_idx * input_row_stride
    
    # 列偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 加载当前行
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # 数值稳定的 softmax
    # Step 1: 减去最大值
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Step 2: exp
    numerator = tl.exp(row)
    
    # Step 3: 求和
    denominator = tl.sum(numerator, axis=0)
    
    # Step 4: 归一化
    softmax_output = numerator / denominator
    
    # 写回
    output_start = row_idx * output_row_stride
    tl.store(output_ptr + output_start + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax 的 Python 封装"""
    assert x.ndim == 2, "输入必须是 2D 张量"
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    
    softmax_kernel[(n_rows,)](
        x, output, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 256, device='cuda')
    output = softmax(x)
    expected = torch.softmax(x, dim=-1)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
