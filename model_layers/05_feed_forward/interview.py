import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?FeedForward.forward")


def main() -> None:

    raise NotImplementedError("璇峰湪 main() 涓ˉ鍏ㄦ渶灏忓彲杩愯绀轰緥")


if __name__ == "__main__":
    main()
