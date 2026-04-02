import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int,
                 dropout_p: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 蹇呴』鑳借 num_heads 鏁撮櫎"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?CausalSelfAttention.forward")


def main() -> None:
    raise NotImplementedError("璇峰湪 main() 涓ˉ鍏ㄦ渶灏忓彲杩愯绀轰緥")


if __name__ == "__main__":
    main()
