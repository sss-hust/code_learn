"""
07_online_softmax - 参考答案

【关键点】
1. Pass 1（在线累加）: 每个 block 只需要更新 running_max 和 running_sum
   - 新 max = max(旧 max, 当前 block 的 max)  
   - 旧 sum 需要修正: sum *= exp(旧 max - 新 max)
   - 新 sum += sum(exp(当前 block - 新 max))
2. Pass 2（归一化）: exp(x - final_max) / final_sum
3. 这个两遍算法在 n_cols >> BLOCK_SIZE 时仍然正确
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
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    
    # ===== Pass 1: 在线计算 max 和 sum-of-exp =====
    running_max = float('-inf')
    running_sum = 0.0
    
    # 分块遍历
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # 加载当前块
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        
        # 当前块的最大值
        block_max = tl.max(x, axis=0)
        
        # 更新 running max
        new_max = tl.maximum(running_max, block_max)
        
        # 修正历史累加值：旧的 sum 是基于旧 max 的，需要缩放
        running_sum = running_sum * tl.exp(running_max - new_max)
        
        # 累加当前块
        running_sum += tl.sum(tl.exp(x - new_max), axis=0)
        
        # 更新 max
        running_max = new_max
    
    # ===== Pass 2: 归一化 =====
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        
        # softmax = exp(x - max) / sum
        softmax_out = tl.exp(x - running_max) / running_sum
        
        out_start = row_idx * output_row_stride
        tl.store(output_ptr + out_start + col_offsets, softmax_out, mask=mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """在线 Softmax 的 Python 封装"""
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    
    BLOCK_SIZE = 256
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
