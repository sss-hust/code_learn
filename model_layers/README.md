# 常见模型层手写练习

这个目录用于练习 `torch.nn.Module` 风格的常见模型层实现，重点是：

1. 能从空白类写出 `__init__` 和 `forward`。
2. 能说清楚每个张量的 shape 变化。
3. 能在 `main()` 里自己构造数据并做最小校验。

## 文件说明

每道题现在包含两种文件：

- `interview.py`：面试骨架，只保留类签名、核心提示和 `main()` 清单。
- `solution.py`：参考答案，可直接运行。

## 目录结构

- `01_embedding`
- `02_sinusoidal_positional_encoding`
- `03_layer_norm`
- `04_rms_norm`
- `05_feed_forward`
- `06_swiglu_feed_forward`
- `07_multi_head_attention`
- `08_causal_self_attention`
- `09_grouped_query_attention`
- `10_transformer_block`
- `11_moe`

## 推荐练习顺序

1. 先写 `Embedding`、`LayerNorm`、`RMSNorm`，把参数注册和广播写顺。
2. 再写 `FeedForward`、`SwiGLUFeedForward`，把线性层链路写顺。
3. 然后重点练 `MultiHeadAttention` 和 `CausalSelfAttention`。
4. 再做 `GroupedQueryAttention` 和 `TransformerBlock`。
5. 最后做 `MoE`，把 router、top-k 和 expert dispatch 理顺。

## 多头注意力手写清单

面试里写 `MultiHeadAttention`，通常按这个顺序最稳：

1. 输入约定：`query/key/value` 都是 `[B, T, C]`。
2. 线性投影：`q_proj / k_proj / v_proj / out_proj`。
3. 拆头：`[B, T, C] -> [B, T, H, D] -> [B, H, T, D]`。
4. 打分：`scores = q @ k.transpose(-2, -1) / sqrt(D)`。
5. 掩码：支持 `attn_mask`，或额外做 `causal mask`。
6. 归一化：`softmax(scores, dim=-1)`。
7. 聚合：`attn @ v`。
8. 合头：`[B, H, T, D] -> [B, T, C]`。
9. 输出投影：`out_proj(context)`。

## MoE 手写清单

面试里写一个简化版 `MoE`，建议先只写 top-k forward，不要一上来做辅助 loss：

1. 输入约定：`x` 是 `[B, T, C]`。
2. flatten 成 `[B*T, C]`，方便按 token 路由。
3. `router(x)` 产生 `[B*T, num_experts]`。
4. `topk` 选出每个 token 的 expert id 和 gate logits。
5. 对 top-k logits 做 softmax，得到 mixing weights。
6. 对每个 expert 收集分配到的 token。
7. 跑 expert MLP。
8. 按 gate weight 加权写回输出。
9. 最后 reshape 回 `[B, T, C]`。

## 建议的自测方式

每次练习都至少做两件事：

1. 打印输入输出 shape，确认维度流是对的。
2. 用 PyTorch 内置实现或手写 reference 做一次数值对比。

## 运行方式

```bash
ssh 187
source /home/yangfu/anaconda3/etc/profile.d/conda.sh
conda activate vllm-env
cd /home/yangfu/workspace/code_learn/model_layers/07_multi_head_attention
python interview.py
python solution.py
```
