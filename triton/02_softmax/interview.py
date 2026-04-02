import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr,
                   output_ptr,
                   m_rows,n_cols,
                   input_row_stride,
                   output_row_stride,
                   BLOCK_SIZE: tl.constexpr,
                   BLOCK_SIZE_M:tl.constexpr):
    pid = tl.program_id(0)

    row_start = pid * BLOCK_SIZE_M 
    row_offsets = row_start + tl.arange(0,BLOCK_SIZE_M)
    row_mask = row_offsets[:,None] < m_rows

    col_offsets = tl.arange(0,BLOCK_SIZE)
    col_mask = col_offsets[None,:] < n_cols

    mask = row_mask & col_mask

    row = tl.load(input_ptr + row_offsets[:,None] * input_row_stride + col_offsets[None,:],mask = mask,other=-float('inf'))

    row_max = tl.max(row,axis = 1)
    row = row - row_max[:,None]

    ne = tl.exp(row)

    s = tl.sum(ne,axis = 1)

    s_o = ne / s[:,None]

    tl.store(output_ptr + row_offsets[:,None] * output_row_stride + col_offsets[None,:],s_o,mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement softmax end-to-end.")

def main() -> None:
    M = 1000
    N = 1000
    src = torch.rand(M,N,device = 'cuda')
    dest = torch.empty_like(src)

    BLOCK_SIZE_M = 4
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(M,BLOCK_SIZE_M),)
    softmax_kernel[grid](src,dest,
                        M,N,
                        src.stride(0),dest.stride(0),
                        BLOCK_SIZE = BLOCK_SIZE,BLOCK_SIZE_M = BLOCK_SIZE_M
    )
    expected = torch.softmax(src,dim = 1)
    print(torch.allclose(dest, expected, atol=1e-5))
if __name__ == "__main__":
    main()
