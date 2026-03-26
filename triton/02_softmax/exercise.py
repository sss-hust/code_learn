"""
02_softmax - Triton Softmax

【核心概念】
- 数值稳定性：先减去最大值再 exp，防止溢出
- 行级并行：每个 program 处理一行
- tl.max / tl.sum: Triton 内置归约操作

【任务】
实现一个 Triton kernel，对 2D 张量的每一行计算 softmax:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,             # 每行的列数
    input_row_stride,   # 输入每行的 stride
    output_row_stride,  # 输出每行的 stride
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel：对每一行做 softmax
    
    提示：
    1. 用 tl.program_id(0) 获取行索引
    2. 加载整行数据（注意 mask）
    3. 计算 row_max = tl.max(row, axis=0)
    4. 计算 exp(row - row_max)
    5. 计算 sum，然后归一化
    """
    # TODO: 在此实现你的代码
    pass


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
