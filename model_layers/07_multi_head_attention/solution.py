import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout_p

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = self._merge_heads(context)
        return self.out_proj(context)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    tq = 16
    tk = 16
    d_model = 512
    num_heads = 8

    query = torch.randn(batch, tq, d_model)
    key = torch.randn(batch, tk, d_model)
    value = torch.randn(batch, tk, d_model)

    model = MultiHeadAttention(d_model, num_heads)
    output = model(query, key, value)

    ref = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    )
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([model.q_proj.weight, model.k_proj.weight, model.v_proj.weight], dim=0)
        )
        ref.in_proj_bias.copy_(
            torch.cat([model.q_proj.bias, model.k_proj.bias, model.v_proj.bias], dim=0)
        )
        ref.out_proj.weight.copy_(model.out_proj.weight)
        ref.out_proj.bias.copy_(model.out_proj.bias)

    expected, _ = ref(query, key, value, need_weights=False)
    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
