"""
02b_row_normalize - Triton 行归一化 (no max-subtract softmax)

【核心概念】
- 介于 row_max 和 softmax 之间：要做 reduction，又要把 reduction 结果广播回去
- 不做 exp，也不做数值稳定，仅 x[i] / sum(row)，让你专注于"reduction + 广播写回"
- 完成本题再做 02_softmax，相当于把"减 max + exp"两步加上去

【任务】
对 2D 张量 x [n_rows, n_cols] 做行归一化：output[i, j] = x[i, j] / sum(x[i])。
约定输入 x 都是非负的，确保 sum > 0。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_normalize_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一行。

    提示：
    1. row_idx = tl.program_id(0)
    2. col_offsets = tl.arange(0, BLOCK_SIZE)
    3. mask = col_offsets < n_cols
    4. row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets,
                     mask=mask, other=0.0)
    5. row_sum = tl.sum(row, axis=0)
    6. row = row / row_sum
    7. tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
                row, mask=mask)
    """
    pass


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    row_normalize_kernel[(n_rows,)](
        x, output, n_cols, x.stride(0), output.stride(0), BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(64, 200, device="cuda") + 1e-3
    out = row_normalize(x)
    ref = x / x.sum(dim=-1, keepdim=True)
    print(f"max_err = {(out - ref).abs().max().item():.6f}")
    print("row sums =", out.sum(dim=-1)[:4])
