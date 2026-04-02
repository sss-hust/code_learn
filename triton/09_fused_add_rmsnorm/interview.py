import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_rmsnorm_kernel(x_ptr,
                             residual_ptr,
                             output_ptr,
                             weight_ptr,
                             n_cols,
                             eps,
                             stride_x,
                             stride_r,
                             stride_o,
                             BLOCK_SIZE: tl.constexpr):
    pass

def fused_add_rmsnorm(x: torch.Tensor,
                      residual: torch.Tensor,
                      weight: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("Implement fused_add_rmsnorm end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
