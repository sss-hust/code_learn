import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("璇疯ˉ鍏?RMSNorm.forward")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全最小可运行示例")

if __name__ == "__main__":
    main()
