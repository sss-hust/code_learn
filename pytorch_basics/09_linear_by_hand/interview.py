"""09_linear_by_hand - 手写 Linear 层

【目标】把 nn.Linear 拆开自己写一遍：weight、bias、kaiming 初始化、
F.linear 调用，整套都和 nn.Linear 在数值上对齐。

之后做 model_layers 里的 FeedForward / MultiHeadAttention 时，相当于
反复用这套模板。

【对齐细节（参考 PyTorch 源码 nn.Linear.reset_parameters）】
- weight: kaiming_uniform_(weight, a=sqrt(5))
- bias  : 用 fan_in 求 bound，bias 在 [-bound, bound] 上 uniform
  fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
  bound = 1 / math.sqrt(fan_in)
  nn.init.uniform_(bias, -bound, bound)

【任务】实现 MyLinear。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    """复刻 nn.Linear。bias=False 时不注册 self.bias。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        raise NotImplementedError(
            "请在 __init__ 中：\n"
            "  1) 用 nn.Parameter 注册 self.weight，形状 [out_features, in_features]\n"
            "  2) 如果 bias=True，注册 self.bias 形状 [out_features]，否则 register_parameter('bias', None)\n"
            "  3) 调用 self.reset_parameters() 完成 kaiming 初始化"
        )

    def reset_parameters(self) -> None:
        """与 nn.Linear.reset_parameters 等价的初始化。"""
        raise NotImplementedError("请补全 reset_parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接调 F.linear(x, self.weight, self.bias)。"""
        raise NotImplementedError("请补全 MyLinear.forward")


def main() -> None:
    """最小可运行示例：
    1. 造 MyLinear(8, 4)，喂 [3, 8] 张量，看 shape
    2. 复制 nn.Linear 的 weight/bias 到 MyLinear，验证两者输出完全相同
    3. 试一下 bias=False 的版本，确认 self.bias 是 None
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
