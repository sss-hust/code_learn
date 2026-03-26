"""
07_online_softmax - Triton 在线 Softmax

【核心概念】
- 标准 softmax 需要两遍遍历：第一遍求 max，第二遍求 exp 和 sum
- 在线 softmax（Online Softmax）只需一遍遍历
- 核心思想：维护一个 running max 和 running sum，动态更新
- 这是 Flash Attention 的核心前置知识

【在线 Softmax 算法】
初始化: m = -inf, d = 0
对于每个元素 x_i:
    m_new = max(m, x_i)
    d = d * exp(m - m_new) + exp(x_i - m_new)
    m = m_new
最终: softmax(x_i) = exp(x_i - m) / d

【任务】
实现在线 Softmax 的 Triton kernel，分块处理每行数据
"""

import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    在线 Softmax kernel（分块版本）
    
    与标准 softmax 不同，这里假设 n_cols 可能大于 BLOCK_SIZE，
    需要分多个块处理，每块更新 running max 和 running sum。
    
    提示：
    Pass 1 - 在线计算 max 和 sum-of-exp：
      1. 遍历每个 block
      2. 更新 running_max 和 running_sum
      3. running_sum *= exp(old_max - new_max) 修正历史累加值
    
    Pass 2 - 归一化：
      1. 再遍历一次，计算 exp(x - final_max) / final_sum
    """
    # TODO: 在此实现你的代码
    pass


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """在线 Softmax 的 Python 封装"""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = 256  # 固定较小的 block size，模拟分块处理
    output = torch.empty_like(x)
    
    online_softmax_kernel[(n_rows,)](
        x, output, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 1024, device='cuda')
    
    output = online_softmax(x)
    expected = torch.softmax(x, dim=-1)
    
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected, atol=1e-5) else '失败 ✗'}")
