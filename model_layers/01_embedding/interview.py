import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?Embedding.forward")


def main() -> None:

    raise NotImplementedError("璇峰湪 main() 涓ˉ鍏ㄦ渶灏忓彲杩愯绀轰緥")


if __name__ == "__main__":
    main()
