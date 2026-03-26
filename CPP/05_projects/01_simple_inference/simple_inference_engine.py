"""
simple_inference_engine.py - 简单推理引擎

本项目实现一个迷你推理框架，包含：
1. Tensor类：数据存储和操作
2. Operator基类：算子接口
3. 常用算子：Linear, ReLU, Softmax等
4. 简单的MLP推理

运行: python simple_inference_engine.py
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import time

# ============================================================================
# 第一部分：Tensor类
# ============================================================================

class Tensor:
    """
    简单的Tensor实现
    
    特点：
    - 封装NumPy数组
    - 记录形状和数据类型
    - 支持基本操作
    """
    
    def __init__(
        self, 
        data: np.ndarray = None,
        shape: Tuple[int, ...] = None,
        dtype: np.dtype = np.float32,
        name: str = ""
    ):
        if data is not None:
            self.data = data.astype(dtype)
            self.shape = data.shape
        else:
            self.data = np.zeros(shape, dtype=dtype)
            self.shape = shape
        
        self.dtype = dtype
        self.name = name
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray, name: str = "") -> 'Tensor':
        return cls(data=arr, name=name)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], dtype=np.float32) -> 'Tensor':
        return cls(shape=shape, dtype=dtype)
    
    @classmethod
    def randn(cls, *shape, dtype=np.float32) -> 'Tensor':
        data = np.random.randn(*shape).astype(dtype)
        return cls(data=data)
    
    def numpy(self) -> np.ndarray:
        return self.data
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def nbytes(self) -> int:
        return self.data.nbytes
    
    def reshape(self, *shape) -> 'Tensor':
        return Tensor(data=self.data.reshape(shape))
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, name='{self.name}')"
    
    # 运算符重载
    def __add__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(data=self.data + other.data)
    
    def __mul__(self, other) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(data=self.data * other.data)
        return Tensor(data=self.data * other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(data=self.data @ other.data)


# ============================================================================
# 第二部分：算子基类
# ============================================================================

class Operator(ABC):
    """算子基类"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.weights: Dict[str, Tensor] = {}
    
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """前向计算"""
        pass
    
    def __call__(self, *inputs: Tensor) -> Tensor:
        return self.forward(*inputs)
    
    def num_parameters(self) -> int:
        """参数量"""
        return sum(w.size for w in self.weights.values())


# ============================================================================
# 第三部分：具体算子实现
# ============================================================================

class Linear(Operator):
    """线性层: y = xW + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__("Linear")
        
        # Xavier初始化
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights['weight'] = Tensor(
            data=np.random.randn(in_features, out_features).astype(np.float32) * scale
        )
        
        if bias:
            self.weights['bias'] = Tensor(
                data=np.zeros(out_features, dtype=np.float32)
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
    
    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weights['weight']
        if self.has_bias:
            output = output + self.weights['bias']
        return output


class ReLU(Operator):
    """ReLU激活函数"""
    
    def __init__(self):
        super().__init__("ReLU")
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(data=np.maximum(x.data, 0))


class GELU(Operator):
    """GELU激活函数（Transformer常用）"""
    
    def __init__(self):
        super().__init__("GELU")
    
    def forward(self, x: Tensor) -> Tensor:
        # 近似GELU
        data = 0.5 * x.data * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data ** 3)
        ))
        return Tensor(data=data)


class Softmax(Operator):
    """Softmax"""
    
    def __init__(self, axis: int = -1):
        super().__init__("Softmax")
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        exp_x = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        return Tensor(data=exp_x / np.sum(exp_x, axis=self.axis, keepdims=True))


class LayerNorm(Operator):
    """层归一化"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__("LayerNorm")
        self.hidden_size = hidden_size
        self.eps = eps
        
        self.weights['gamma'] = Tensor(data=np.ones(hidden_size, dtype=np.float32))
        self.weights['beta'] = Tensor(data=np.zeros(hidden_size, dtype=np.float32))
    
    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        output = normalized * self.weights['gamma'].data + self.weights['beta'].data
        return Tensor(data=output)


class Embedding(Operator):
    """嵌入层"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__("Embedding")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.weights['embedding'] = Tensor(
            data=np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.02
        )
    
    def forward(self, input_ids: np.ndarray) -> Tensor:
        """input_ids: (batch, seq_len)"""
        return Tensor(data=self.weights['embedding'].data[input_ids])


# ============================================================================
# 第四部分：MLP模块
# ============================================================================

class MLP(Operator):
    """MLP块（Transformer FFN）"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__("MLP")
        
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)
        self.activation = GELU()
        
        self.weights.update({
            'fc1.weight': self.fc1.weights['weight'],
            'fc1.bias': self.fc1.weights['bias'],
            'fc2.weight': self.fc2.weights['weight'],
            'fc2.bias': self.fc2.weights['bias'],
        })
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 第五部分：简单模型
# ============================================================================

class SimpleLM:
    """简单语言模型"""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 组件
        self.embedding = Embedding(vocab_size, hidden_size)
        self.layers = [
            {
                'mlp': MLP(hidden_size, hidden_size * 4),
                'ln': LayerNorm(hidden_size)
            }
            for _ in range(num_layers)
        ]
        self.final_ln = LayerNorm(hidden_size)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: np.ndarray) -> Tensor:
        """
        input_ids: (batch, seq_len)
        returns: logits (batch, seq_len, vocab_size)
        """
        # Embedding
        hidden = self.embedding(input_ids)
        
        # Layers
        for layer in self.layers:
            residual = hidden
            hidden = layer['ln'](hidden)
            hidden = layer['mlp'](hidden)
            hidden = Tensor(data=hidden.data + residual.data)
        
        # Final
        hidden = self.final_ln(hidden)
        logits = self.lm_head(hidden)
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 10) -> np.ndarray:
        """自回归生成"""
        current_ids = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(current_ids)
            
            # 取最后一个token的预测
            next_token_logits = logits.data[:, -1, :]
            
            # Greedy解码
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
            # 拼接
            current_ids = np.concatenate([current_ids, next_token], axis=1)
        
        return current_ids
    
    def num_parameters(self) -> int:
        """计算总参数量"""
        total = self.embedding.num_parameters()
        for layer in self.layers:
            total += layer['mlp'].num_parameters()
            total += layer['ln'].num_parameters()
        total += self.final_ln.num_parameters()
        total += self.lm_head.num_parameters()
        return total


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("简单推理引擎")
    print("=" * 60)
    
    # 1. 测试基本Tensor
    print("\n--- Tensor测试 ---")
    t1 = Tensor.randn(2, 3)
    t2 = Tensor.randn(2, 3)
    t3 = t1 + t2
    print(f"t1 + t2 = {t3}")
    
    # 2. 测试算子
    print("\n--- 算子测试 ---")
    linear = Linear(768, 3072)
    relu = ReLU()
    
    x = Tensor.randn(2, 10, 768)
    y = linear(x)
    y = relu(y)
    print(f"Linear+ReLU: {x.shape} -> {y.shape}")
    print(f"Linear参数量: {linear.num_parameters():,}")
    
    # 3. 测试MLP
    print("\n--- MLP测试 ---")
    mlp = MLP(768, 3072)
    x = Tensor.randn(2, 10, 768)
    y = mlp(x)
    print(f"MLP: {x.shape} -> {y.shape}")
    print(f"MLP参数量: {mlp.num_parameters():,}")
    
    # 4. 测试完整模型
    print("\n--- 简单LM测试 ---")
    model = SimpleLM(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4
    )
    
    print(f"模型参数量: {model.num_parameters():,}")
    print(f"约 {model.num_parameters() / 1e6:.2f}M 参数")
    
    # 前向传播
    input_ids = np.random.randint(0, 1000, (1, 5))
    print(f"\n输入: {input_ids}")
    
    start = time.perf_counter()
    logits = model.forward(input_ids)
    fwd_time = time.perf_counter() - start
    
    print(f"输出形状: {logits.shape}")
    print(f"前向耗时: {fwd_time * 1000:.2f} ms")
    
    # 生成
    print("\n--- 生成测试 ---")
    prompt = np.array([[1, 2, 3]])  # 起始token
    
    start = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=10)
    gen_time = time.perf_counter() - start
    
    print(f"生成结果: {generated}")
    print(f"生成耗时: {gen_time * 1000:.2f} ms")
    print(f"每token: {gen_time / 10 * 1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("完成！这是一个教学用的简单推理引擎。")
    print("实际框架(PyTorch, TensorRT)会有更多优化。")
    print("=" * 60)
