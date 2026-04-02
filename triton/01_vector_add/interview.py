import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets,mask = mask)
    y = tl.load(y_ptr + offsets,mask = mask)

    output = x + y

    tl.store(output_ptr + offsets,output,mask = mask)



def main() -> None:
    N =100000
    A = torch.rand(N,device = 'cuda')
    B = torch.rand(N,device = 'cuda')
    C = torch.empty_like(A)
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](A,B,C,N,BLOCK_SIZE = BLOCK_SIZE)
    print(C)

if __name__ == "__main__":
    main()
