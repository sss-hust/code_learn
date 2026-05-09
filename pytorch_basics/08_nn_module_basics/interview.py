"""08_nn_module_basics - nn.Module 基础

【目标】掌握 nn.Module 子类化的标准动作：
- nn.Parameter 注册可训练参数
- register_buffer 注册不训练但要随模型保存/搬运的状态
- train() / eval() 切换模式
- self.parameters() / self.buffers() 怎么遍历

【任务】
1. 实现 LinearNoBias，包含一个 Parameter (weight) 和一个 buffer (forward_count)
2. 实现 count_parameters / freeze_module 两个工具函数
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearNoBias(nn.Module):
    """无 bias 的 Linear 层，并且记录被 forward 调用了多少次。

    要求：
    - self.weight: nn.Parameter，形状 [out_features, in_features]
      初始化用 nn.init.normal_(weight, std=0.02)
    - self.forward_count: 用 register_buffer 注册的 long 张量，初始为 0
      每次 forward 时 +1。这个 buffer 不参与训练，但会被 .to('cuda')
      和 state_dict() 一起搬运/保存。
    - forward(x) 返回 x @ self.weight.T
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        raise NotImplementedError("请在 __init__ 中创建 self.weight 和 self.forward_count")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("请补全 LinearNoBias.forward")


def count_parameters(module: nn.Module) -> int:
    """统计 module 里所有 requires_grad=True 的参数总元素数。

    提示：sum(p.numel() for p in module.parameters() if p.requires_grad)。
    """
    raise NotImplementedError("请补全 count_parameters")


def freeze_module(module: nn.Module) -> None:
    """把 module 里所有参数的 requires_grad 设为 False，原地修改。
    用于"只训练 head 层"这类微调场景。

    提示：for p in module.parameters(): p.requires_grad = False。
    """
    raise NotImplementedError("请补全 freeze_module")


def main() -> None:
    """最小可运行示例：
    1. 造一个 LinearNoBias(8, 4)，喂一个 [2, 8] 的张量看输出 shape
    2. 连续 forward 3 次，确认 forward_count 变成 3
    3. count_parameters 应该返回 32 (4*8)
    4. freeze_module 后 count_parameters 返回 0
    """
    raise NotImplementedError("请在 main() 里补全演示代码")


if __name__ == "__main__":
    main()
