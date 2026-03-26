"""
01_parallel_strategies.py - 并行策略

【核心概念】
大模型无法放入单个GPU，需要分布式推理：
- 数据并行 (DP): 不同GPU处理不同batch
- 张量并行 (TP): 单层切分到多个GPU
- 流水线并行 (PP): 不同层放在不同GPU

运行: python 01_parallel_strategies.py
"""

import numpy as np
from typing import List, Tuple
import time

# ============================================================================
# 第一部分：数据并行
# ============================================================================

class DataParallel:
    """
    数据并行模拟
    
    训练时：每个GPU有完整模型，处理不同数据
    推理时：多个请求分配到不同GPU
    """
    
    def __init__(self, model_weights: np.ndarray, num_gpus: int):
        self.num_gpus = num_gpus
        # 每个GPU都有完整的模型副本
        self.weights = [model_weights.copy() for _ in range(num_gpus)]
        
        print(f"数据并行: {num_gpus} GPUs，每个GPU存储完整模型")
        print(f"  权重大小: {model_weights.nbytes / 1024 / 1024:.2f} MB")
        print(f"  总内存: {model_weights.nbytes * num_gpus / 1024 / 1024:.2f} MB")
    
    def forward(self, batch: np.ndarray) -> np.ndarray:
        """分布式前向传播"""
        batch_per_gpu = batch.shape[0] // self.num_gpus
        
        results = []
        for i in range(self.num_gpus):
            start = i * batch_per_gpu
            end = start + batch_per_gpu
            local_batch = batch[start:end]
            
            # 在"GPU i"上计算
            local_result = local_batch @ self.weights[i]
            results.append(local_result)
        
        # 合并结果
        return np.concatenate(results, axis=0)


# ============================================================================
# 第二部分：张量并行
# ============================================================================

class TensorParallel:
    """
    张量并行模拟
    
    将权重矩阵切分到多个GPU：
    - 列并行：W = [W1, W2, ...]
    - 行并行：W = [W1; W2; ...]
    
    适用于大的FFN和Attention层
    """
    
    def __init__(self, weight: np.ndarray, num_gpus: int, mode: str = "column"):
        """
        mode: "column" 按列切分, "row" 按行切分
        """
        self.num_gpus = num_gpus
        self.mode = mode
        
        if mode == "column":
            # 列并行：切分输出维度
            out_features = weight.shape[1]
            assert out_features % num_gpus == 0
            chunk_size = out_features // num_gpus
            self.weights = [
                weight[:, i * chunk_size:(i + 1) * chunk_size]
                for i in range(num_gpus)
            ]
            print(f"张量并行(列): {weight.shape} -> {num_gpus} x {self.weights[0].shape}")
        
        else:  # row
            # 行并行：切分输入维度
            in_features = weight.shape[0]
            assert in_features % num_gpus == 0
            chunk_size = in_features // num_gpus
            self.weights = [
                weight[i * chunk_size:(i + 1) * chunk_size, :]
                for i in range(num_gpus)
            ]
            print(f"张量并行(行): {weight.shape} -> {num_gpus} x {self.weights[0].shape}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        列并行：x @ [W1, W2] = [x @ W1, x @ W2]，结果拼接
        行并行：[x1, x2] @ [W1; W2] = x1 @ W1 + x2 @ W2，结果求和
        """
        if self.mode == "column":
            # 每个GPU计算部分输出
            partial_results = [x @ w for w in self.weights]
            # 拼接（AllGather）
            return np.concatenate(partial_results, axis=-1)
        
        else:  # row
            # 切分输入
            chunk_size = x.shape[-1] // self.num_gpus
            x_chunks = [
                x[..., i * chunk_size:(i + 1) * chunk_size]
                for i in range(self.num_gpus)
            ]
            # 每个GPU计算部分
            partial_results = [xc @ w for xc, w in zip(x_chunks, self.weights)]
            # AllReduce求和
            return sum(partial_results)


class AttentionTensorParallel:
    """
    Attention的张量并行
    
    将attention heads分配到不同GPU
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_gpus: int
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_gpus = num_gpus
        self.heads_per_gpu = num_heads // num_gpus
        self.head_dim = hidden_size // num_heads
        
        # 每个GPU的Q、K、V投影
        local_qkv_size = self.heads_per_gpu * self.head_dim
        
        self.Wq = [np.random.randn(hidden_size, local_qkv_size).astype(np.float32) * 0.02
                   for _ in range(num_gpus)]
        self.Wk = [np.random.randn(hidden_size, local_qkv_size).astype(np.float32) * 0.02
                   for _ in range(num_gpus)]
        self.Wv = [np.random.randn(hidden_size, local_qkv_size).astype(np.float32) * 0.02
                   for _ in range(num_gpus)]
        
        print(f"Attention张量并行:")
        print(f"  总heads: {num_heads}, 每GPU: {self.heads_per_gpu}")
    
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """每个GPU计算部分heads"""
        batch, seq, hidden = x.shape
        
        outputs = []
        for i in range(self.num_gpus):
            Q = x @ self.Wq[i]  # (batch, seq, heads_per_gpu * head_dim)
            K = x @ self.Wk[i]
            V = x @ self.Wv[i]
            
            # Reshape for attention
            Q = Q.reshape(batch, seq, self.heads_per_gpu, self.head_dim)
            K = K.reshape(batch, seq, self.heads_per_gpu, self.head_dim)
            V = V.reshape(batch, seq, self.heads_per_gpu, self.head_dim)
            
            # 简化的attention计算
            Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, dim)
            K = K.transpose(0, 2, 1, 3)
            V = V.transpose(0, 2, 1, 3)
            
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
            attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            out = np.matmul(attn, V)
            
            outputs.append(out)
        
        return outputs


# ============================================================================
# 第三部分：流水线并行
# ============================================================================

class PipelineParallel:
    """
    流水线并行模拟
    
    将模型层分配到不同GPU：
    - GPU 0: Layers 0-7
    - GPU 1: Layers 8-15
    - ...
    
    通过micro-batch实现流水线效率
    """
    
    def __init__(self, layers: List[np.ndarray], num_stages: int):
        self.num_stages = num_stages
        layers_per_stage = len(layers) // num_stages
        
        # 分配层到不同stage
        self.stages = []
        for i in range(num_stages):
            stage_layers = layers[i * layers_per_stage:(i + 1) * layers_per_stage]
            self.stages.append(stage_layers)
        
        print(f"流水线并行: {len(layers)} 层 -> {num_stages} stages")
        print(f"  每stage: {layers_per_stage} 层")
    
    def forward_stage(self, x: np.ndarray, stage_idx: int) -> np.ndarray:
        """在特定stage执行前向"""
        for layer in self.stages[stage_idx]:
            x = np.maximum(x @ layer, 0)  # Linear + ReLU
        return x
    
    def forward_pipeline(
        self, 
        batches: List[np.ndarray],
        num_micro_batches: int
    ) -> List[np.ndarray]:
        """
        流水线执行
        
        时间步：
        t=0: Stage0处理batch0
        t=1: Stage0处理batch1, Stage1处理batch0
        ...
        """
        outputs = []
        
        # 简化版：顺序处理每个batch
        for batch in batches:
            x = batch
            for stage_idx in range(self.num_stages):
                x = self.forward_stage(x, stage_idx)
            outputs.append(x)
        
        return outputs


# ============================================================================
# 第四部分：内存计算
# ============================================================================

def calculate_parallel_memory(
    model_size_gb: float,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    num_gpus: int,
    strategy: str
):
    """计算不同并行策略的内存需求"""
    
    print(f"\n{strategy}并行 ({num_gpus} GPUs):")
    
    if strategy == "data":
        # 每个GPU存完整模型
        model_per_gpu = model_size_gb
        batch_per_gpu = batch_size // num_gpus
        activation = batch_per_gpu * seq_len * hidden_size * 2 / 1e9  # FP16
        
    elif strategy == "tensor":
        # 权重切分到多GPU
        model_per_gpu = model_size_gb / num_gpus
        batch_per_gpu = batch_size
        activation = batch_per_gpu * seq_len * hidden_size * 2 / 1e9
        
    elif strategy == "pipeline":
        # 层切分
        model_per_gpu = model_size_gb / num_gpus
        batch_per_gpu = batch_size
        layers_per_gpu = num_layers // num_gpus
        activation = batch_per_gpu * seq_len * hidden_size * 2 * layers_per_gpu / 1e9
    
    total = model_per_gpu + activation
    print(f"  模型: {model_per_gpu:.2f} GB")
    print(f"  激活: {activation:.2f} GB")
    print(f"  总计/GPU: {total:.2f} GB")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("并行策略")
    print("=" * 50)
    
    # 1. 数据并行
    print("\n--- 数据并行 ---")
    weights = np.random.randn(768, 768).astype(np.float32)
    dp = DataParallel(weights, num_gpus=4)
    
    batch = np.random.randn(32, 768).astype(np.float32)
    output = dp.forward(batch)
    print(f"输入: {batch.shape} -> 输出: {output.shape}")
    
    # 2. 张量并行
    print("\n--- 张量并行 ---")
    big_weight = np.random.randn(768, 3072).astype(np.float32)
    
    tp_col = TensorParallel(big_weight, num_gpus=4, mode="column")
    x = np.random.randn(8, 128, 768).astype(np.float32)
    out_col = tp_col.forward(x)
    print(f"输入: {x.shape} -> 输出: {out_col.shape}")
    
    # 3. Attention并行
    print("\n--- Attention张量并行 ---")
    attn_tp = AttentionTensorParallel(
        hidden_size=768,
        num_heads=12,
        num_gpus=4
    )
    
    x = np.random.randn(2, 64, 768).astype(np.float32)
    outputs = attn_tp.forward(x)
    print(f"输入: {x.shape}")
    print(f"每GPU输出形状: {outputs[0].shape}")
    
    # 4. 流水线并行
    print("\n--- 流水线并行 ---")
    layers = [np.random.randn(768, 768).astype(np.float32) * 0.01 for _ in range(12)]
    pp = PipelineParallel(layers, num_stages=4)
    
    batches = [np.random.randn(8, 768).astype(np.float32) for _ in range(4)]
    outputs = pp.forward_pipeline(batches, num_micro_batches=4)
    print(f"处理 {len(batches)} 个batch，每个输出: {outputs[0].shape}")
    
    # 5. 内存对比
    print("\n" + "=" * 50)
    print("内存需求对比 (7B模型, batch=32, seq=2048)")
    print("=" * 50)
    
    calculate_parallel_memory(14, 32, 2048, 4096, 32, 4, "data")
    calculate_parallel_memory(14, 32, 2048, 4096, 32, 4, "tensor")
    calculate_parallel_memory(14, 32, 2048, 4096, 32, 4, "pipeline")
    
    print("\n总结:")
    print("- 数据并行: 简单，但每GPU需完整模型")
    print("- 张量并行: 单层切分，通信多但延迟低")
    print("- 流水线并行: 层切分，通信少但需要micro-batch")
