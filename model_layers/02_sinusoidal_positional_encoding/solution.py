import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len].to(dtype=x.dtype, device=x.device)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512

    x = torch.randn(batch, seq_len, d_model)
    model = SinusoidalPositionalEncoding(d_model)
    output = model(x)

    print("output.shape =", tuple(output.shape))
    print("position0_head8 =", output[0, 0, :8])


if __name__ == "__main__":
    main()
