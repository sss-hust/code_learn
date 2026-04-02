import math

import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_q_heads == 0, "d_model 必须能被 num_q_heads 整除"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(context)


def gqa_reference(model: GroupedQueryAttention, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    q = model.q_proj(x).view(batch, seq_len, model.num_q_heads, model.head_dim).transpose(1, 2)
    k = model.k_proj(x).view(batch, seq_len, model.num_kv_heads, model.head_dim).transpose(1, 2)
    v = model.v_proj(x).view(batch, seq_len, model.num_kv_heads, model.head_dim).transpose(1, 2)
    k = k.repeat_interleave(model.group_size, dim=1)
    v = v.repeat_interleave(model.group_size, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.head_dim)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, v)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, model.d_model)
    return model.out_proj(context)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512
    num_q_heads = 8
    num_kv_heads = 2

    x = torch.randn(batch, seq_len, d_model)
    model = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
    output = model(x)
    expected = gqa_reference(model, x)

    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
