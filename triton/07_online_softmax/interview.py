import torch
import triton
import triton.language as tl

@triton.jit
def online_softmax_kernel(input_ptr,
                          output_ptr,
                          n_cols,
                          input_row_stride,
                          output_row_stride,
                          BLOCK_SIZE: tl.constexpr):
    pass

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement online_softmax end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
