"""
01d_row_max - Triton 行最大值

【核心概念】
- 第一次进入 2D 张量的行级并行：每个 program 处理一行
- 引入 input_row_stride（每行在内存中的步长，通常等于 n_cols）
- 单一 reduction：tl.max 求行最大值
- 是为后面的 softmax 铺路：先把"按行处理 + 单 reduction"练熟

【任务】
对 2D 张量 x [n_rows, n_cols] 求每行最大值，输出 [n_rows]。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(
    input_ptr,
    output_ptr,        # [n_rows]
    n_cols,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一行。

    提示：
    1. row_idx = tl.program_id(0)
    2. row_start = input_ptr + row_idx * input_row_stride
    3. col_offsets = tl.arange(0, BLOCK_SIZE)
    4. mask = col_offsets < n_cols
    5. row = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
    6. row_max = tl.max(row, axis=0)
    7. tl.store(output_ptr + row_idx, row_max)
    """
    pass


def row_max(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    row_max_kernel[(n_rows,)](
        x, output, n_cols, x.stride(0), BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 200, device="cuda")
    out = row_max(x)
    ref = x.max(dim=-1).values
    print(f"max_err = {(out - ref).abs().max().item():.6f}")
    print("通过" if torch.allclose(out, ref, atol=1e-6) else "失败")
