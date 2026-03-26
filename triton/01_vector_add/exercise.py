"""
01_vector_add - Triton 向量加法

【核心概念】
- triton.jit: 将 Python 函数编译为 GPU kernel
- tl.program_id(0): 获取当前 block 的索引
- tl.load / tl.store: 从全局内存读写数据
- mask: 处理数组长度不是 BLOCK_SIZE 整数倍的情况

【任务】
实现一个 Triton kernel，计算 output = x + y（逐元素加法）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,        # 输入向量 x 的指针
    y_ptr,        # 输入向量 y 的指针
    output_ptr,   # 输出向量的指针
    n_elements,   # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 block 处理的元素数
):
    """
    向量加法 kernel: output[i] = x[i] + y[i]
    
    提示：
    1. 用 tl.program_id(0) 获取当前 block 索引
    2. 计算当前 block 负责的元素偏移: block_start + tl.arange(0, BLOCK_SIZE)
    3. 创建 mask 防止越界访问
    4. 用 tl.load 读取数据，tl.store 写入结果
    """
    # TODO: 在此实现你的代码
    pass


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法的 Python 封装函数"""
    assert x.shape == y.shape, "输入张量形状必须一致"
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    output = vector_add(x, y)
    expected = x + y
    
    print(f"输出形状: {output.shape}")
    print(f"最大误差: {(output - expected).abs().max().item():.6f}")
    print(f"测试{'通过 ✓' if torch.allclose(output, expected) else '失败 ✗'}")
