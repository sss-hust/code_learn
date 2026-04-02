import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(x_ptr,
                      output_ptr,
                      weight_ptr,
                      bias_ptr,
                      n_cols,
                      eps,
                      x_row_stride,
                      output_row_stride,
                      BLOCK_SIZE: tl.constexpr):
    pass

def layer_norm(x: torch.Tensor,
               weight: torch.Tensor,
               bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    raise NotImplementedError("Implement layer_norm end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
