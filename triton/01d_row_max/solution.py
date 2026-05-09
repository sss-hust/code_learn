"""01d_row_max - 参考答案"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(input_ptr, output_ptr, n_cols, input_row_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
    row_max_val = tl.max(row, axis=0)
    tl.store(output_ptr + row_idx, row_max_val)


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
