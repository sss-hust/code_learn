import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TopKMoE(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k 必须在合法范围内"
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForwardExpert(d_model, hidden_dim) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(batch * seq_len, d_model)

        router_logits = self.router(x_flat)
        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                token_mask = topk_indices[:, slot] == expert_id
                if not token_mask.any():
                    continue
                expert_input = x_flat[token_mask]
                expert_output = expert(expert_input)
                weight = topk_weights[token_mask, slot].unsqueeze(-1)
                output[token_mask] += weight * expert_output

        return output.view(batch, seq_len, d_model)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512
    hidden_dim = 1024
    num_experts = 4
    top_k = 2

    x = torch.randn(batch, seq_len, d_model)
    model = TopKMoE(d_model, hidden_dim, num_experts, top_k=top_k)
    output = model(x)

    with torch.no_grad():
        x_flat = x.view(batch * seq_len, d_model)
        router_logits = model.router(x_flat)
        top1_indices = router_logits.argmax(dim=-1)
        counts = torch.bincount(top1_indices, minlength=num_experts)

    print("output.shape =", tuple(output.shape))
    print("top1_counts =", counts.tolist())


if __name__ == "__main__":
    main()
