"""02b_row_normalize - 参考答案"""
import torch
import triton
import triton.language as tl


@triton.jit
def row_normalize_kernel(
    input_ptr, output_ptr, n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets,
                  mask=mask, other=0.0)
    row_sum = tl.sum(row, axis=0)
    row = row / row_sum
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets,
             row, mask=mask)


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
