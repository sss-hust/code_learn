"""08_nn_module_basics - 参考答案"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearNoBias(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.register_buffer("forward_count", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        return x @ self.weight.T


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def main() -> None:
    torch.manual_seed(0)

    layer = LinearNoBias(8, 4)
    x = torch.randn(2, 8)
    print("out.shape =", tuple(layer(x).shape))

    layer(x)
    layer(x)
    print("forward_count =", int(layer.forward_count.item()))

    print("trainable params =", count_parameters(layer))

    freeze_module(layer)
    print("after freeze, trainable params =", count_parameters(layer))


if __name__ == "__main__":
    main()
