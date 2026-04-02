import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr,
                  b_ptr,
                  c_ptr,
                  M,
                  N,
                  K,
                  stride_am,
                  stride_ak,
                  stride_bk,
                  stride_bn,
                  stride_cm,
                  stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pass

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement matmul end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
