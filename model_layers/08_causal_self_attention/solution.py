import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = self._merge_heads(context)
        return self.out_proj(context)


def main() -> None:
    torch.manual_seed(0)
    batch = 2
    seq_len = 16
    d_model = 512
    num_heads = 8

    x = torch.randn(batch, seq_len, d_model)
    model = CausalSelfAttention(d_model, num_heads, max_seq_len=seq_len)
    output = model(x)

    ref = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    )
    with torch.no_grad():
        q_weight, k_weight, v_weight = model.qkv_proj.weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = model.qkv_proj.bias.chunk(3, dim=0)
        ref.in_proj_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
        ref.in_proj_bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
        ref.out_proj.weight.copy_(model.out_proj.weight)
        ref.out_proj.bias.copy_(model.out_proj.bias)

    causal_mask = torch.full((seq_len, seq_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    expected, _ = ref(x, x, x, attn_mask=causal_mask, need_weights=False)

    print("output.shape =", tuple(output.shape))
    print("max_error =", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
