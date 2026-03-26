"""
01_vector_add - 参考答案

【关键点】
1. 每个 program（block）处理 BLOCK_SIZE 个连续元素
2. mask 确保不会越界访问（当 n_elements 不是 BLOCK_SIZE 整数倍时）
3. Triton 自动处理 grid 调度，每个 block 独立并行执行
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前 block 的索引（第0维）
    pid = tl.program_id(0)
    
    # 计算当前 block 负责的元素偏移量
    # 例如 pid=2, BLOCK_SIZE=1024 → offsets = [2048, 2049, ..., 3071]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # mask: 防止越界（最后一个 block 可能不满）
    mask = offsets < n_elements
    
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)


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
