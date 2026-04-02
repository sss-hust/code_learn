import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(x_ptr,
                    output_ptr,
                    weight_ptr,
                    n_cols,
                    eps,
                    x_row_stride,
                    output_row_stride,
                    BLOCK_SIZE: tl.constexpr):
    pass

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("Implement rms_norm end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
