import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 32000
    d_model = 512
    batch = 2
    seq_len = 16

    token_ids = torch.randint(0, vocab_size, (batch, seq_len))
    model = Embedding(vocab_size, d_model)
    output = model(token_ids)
    reference = F.embedding(token_ids, model.weight)

    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - reference).abs().max().item())


if __name__ == "__main__":
    main()
