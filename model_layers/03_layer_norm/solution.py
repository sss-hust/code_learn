import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512

    x = torch.randn(batch, seq_len, d_model)
    model = LayerNorm(d_model)
    output = model(x)

    ref = nn.LayerNorm(d_model, eps=model.eps)
    with torch.no_grad():
        ref.weight.copy_(model.weight)
        ref.bias.copy_(model.bias)
    expected = ref(x)

    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
