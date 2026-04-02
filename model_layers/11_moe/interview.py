import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("请补全 FeedForwardExpert.forward")


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k 必须在合法范围内"
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("请补全 TopKMoE.forward")


def main() -> None:
    raise NotImplementedError("请在 main() 中补全最小可运行示例")


if __name__ == "__main__":
    main()
