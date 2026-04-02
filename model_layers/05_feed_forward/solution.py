import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512
    hidden_dim = 2048

    x = torch.randn(batch, seq_len, d_model)
    model = FeedForward(d_model, hidden_dim)
    output = model(x)

    print("output.shape =", tuple(output.shape))


if __name__ == "__main__":
    main()
