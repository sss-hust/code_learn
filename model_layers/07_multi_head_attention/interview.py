import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_p: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 蹇呴』鑳借 num_heads 鏁撮櫎"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout_p


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?MultiHeadAttention.forward")


def main() -> None:


    raise NotImplementedError("璇峰湪 main() 涓ˉ鍏ㄦ渶灏忓彲杩愯绀轰緥")


if __name__ == "__main__":
    main()
