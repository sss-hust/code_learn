import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(q_ptr,
                           k_ptr,
                           v_ptr,
                           output_ptr,
                           stride_qb,
                           stride_qh,
                           stride_qm,
                           stride_qk,
                           stride_kb,
                           stride_kh,
                           stride_kn,
                           stride_kk,
                           stride_vb,
                           stride_vh,
                           stride_vn,
                           stride_vk,
                           stride_ob,
                           stride_oh,
                           stride_om,
                           stride_ok,
                           seq_len,
                           head_dim,
                           scale,
                           BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    pass

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement flash_attention end-to-end.")

def main() -> None:
    raise NotImplementedError("Fill main() for the interview workflow.")

if __name__ == "__main__":
    main()
