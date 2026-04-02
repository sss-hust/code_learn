import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 hidden_dim: int,
                 max_seq_len: int,
                 dropout_p: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?TransformerBlock.forward")


def main() -> None:

    raise NotImplementedError("璇峰湪 main() 涓ˉ鍏ㄦ渶灏忓彲杩愯绀轰緥")


if __name__ == "__main__":
    main()
