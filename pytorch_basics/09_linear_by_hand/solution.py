"""09_linear_by_hand - 参考答案"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def main() -> None:
    torch.manual_seed(0)

    my = MyLinear(8, 4)
    x = torch.randn(3, 8)
    print("my.shape =", tuple(my(x).shape))

    ref = nn.Linear(8, 4)
    with torch.no_grad():
        my.weight.copy_(ref.weight)
        my.bias.copy_(ref.bias)
    print("max diff vs nn.Linear =", (my(x) - ref(x)).abs().max().item())

    no_bias = MyLinear(8, 4, bias=False)
    print("no_bias.bias is None =", no_bias.bias is None)


if __name__ == "__main__":
    main()
