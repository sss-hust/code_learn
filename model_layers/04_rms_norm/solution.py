import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = (x * x).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return (x / rms) * self.weight


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512

    x = torch.randn(batch, seq_len, d_model)
    model = RMSNorm(d_model)
    output = model(x)

    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    expected = (x / torch.sqrt(mean_sq + model.eps)) * model.weight

    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
