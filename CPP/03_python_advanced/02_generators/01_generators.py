"""
01_generators.py - Python生成器详解

【推理加速场景】
- 大数据集流式加载（不需要一次性加载到内存）
- Token流式生成
- 批量数据迭代
- 管道式数据处理

运行: python 01_generators.py
"""

import sys
from typing import Iterator, Generator, List

# ============================================================================
# 第一部分：基本生成器
# ============================================================================

def simple_generator(n: int) -> Generator[int, None, None]:
    """生成0到n-1的数字"""
    for i in range(n):
        yield i  # yield返回值并暂停


def countdown(n: int) -> Generator[int, None, None]:
    """倒计时生成器"""
    while n > 0:
        yield n
        n -= 1


# ============================================================================
# 第二部分：生成器表达式
# ============================================================================

def demo_generator_expression():
    """生成器表达式 vs 列表推导"""
    
    # 列表推导：一次性创建所有元素
    list_comp = [x ** 2 for x in range(1000000)]
    
    # 生成器表达式：按需生成
    gen_expr = (x ** 2 for x in range(1000000))
    
    print(f"列表内存: {sys.getsizeof(list_comp) / 1024:.2f} KB")
    print(f"生成器内存: {sys.getsizeof(gen_expr)} bytes")  # 很小！
    
    # 取前5个
    for i, val in enumerate(gen_expr):
        if i >= 5:
            break
        print(f"  {i}: {val}")


# ============================================================================
# 第三部分：推理场景 - 批量数据迭代
# ============================================================================

def batch_iterator(data: List, batch_size: int) -> Generator[List, None, None]:
    """批量迭代器（DataLoader核心逻辑）"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def infinite_data_stream() -> Generator[dict, None, None]:
    """无限数据流（模拟实时请求）"""
    request_id = 0
    while True:
        yield {
            "id": request_id,
            "prompt": f"Request {request_id}",
            "timestamp": request_id * 100
        }
        request_id += 1


# ============================================================================
# 第四部分：Token流式生成（大模型核心场景）
# ============================================================================

def generate_tokens(prompt: str, max_tokens: int = 10) -> Generator[str, None, None]:
    """模拟Token流式生成"""
    import time
    
    # 模拟tokenization
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", 
              "the", "lazy", "dog", "."]
    
    for i in range(min(max_tokens, len(tokens))):
        time.sleep(0.1)  # 模拟推理延迟
        yield tokens[i]


def streaming_inference(prompt: str) -> Generator[str, None, None]:
    """
    流式推理接口
    
    类似于：
    for token in client.generate(prompt, stream=True):
        print(token, end='', flush=True)
    """
    for token in generate_tokens(prompt):
        yield token


# ============================================================================
# 第五部分：生成器管道
# ============================================================================

def read_lines(filename: str) -> Generator[str, None, None]:
    """逐行读取文件"""
    # 模拟文件读取
    lines = [
        "  Hello World  ",
        "  Python Generator  ",
        "  Deep Learning  ",
    ]
    for line in lines:
        yield line


def strip_lines(lines: Iterator[str]) -> Generator[str, None, None]:
    """去除空白"""
    for line in lines:
        yield line.strip()


def lowercase_lines(lines: Iterator[str]) -> Generator[str, None, None]:
    """转小写"""
    for line in lines:
        yield line.lower()


def pipeline_demo():
    """管道式处理"""
    # 组装管道：read -> strip -> lowercase
    lines = read_lines("dummy.txt")
    lines = strip_lines(lines)
    lines = lowercase_lines(lines)
    
    # 按需执行
    for line in lines:
        print(f"  处理: {line}")


# ============================================================================
# 第六部分：yield from（子生成器委托）
# ============================================================================

def sub_generator(n: int) -> Generator[int, None, None]:
    for i in range(n):
        yield i


def main_generator() -> Generator[int, None, None]:
    """使用yield from委托给子生成器"""
    yield from sub_generator(3)
    yield from sub_generator(2)
    yield -1


# ============================================================================
# 第七部分：内存对比
# ============================================================================

def memory_comparison():
    """展示生成器的内存优势"""
    import sys
    
    n = 1000000
    
    # 列表：占用大量内存
    data_list = list(range(n))
    list_size = sys.getsizeof(data_list) + sum(sys.getsizeof(x) for x in data_list[:100]) * (n // 100)
    
    # 生成器：几乎不占内存
    data_gen = (x for x in range(n))
    gen_size = sys.getsizeof(data_gen)
    
    print(f"100万元素:")
    print(f"  列表约: {list_size / 1024 / 1024:.1f} MB")
    print(f"  生成器: {gen_size} bytes")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Python 生成器详解")
    print("=" * 50)
    
    # 1. 基本生成器
    print("\n--- 基本生成器 ---")
    gen = simple_generator(5)
    print(f"类型: {type(gen)}")
    print(f"值: {list(gen)}")
    
    print("倒计时:", list(countdown(5)))
    
    # 2. 生成器表达式
    print("\n--- 生成器表达式 ---")
    demo_generator_expression()
    
    # 3. 批量迭代
    print("\n--- 批量迭代 ---")
    data = list(range(10))
    for batch in batch_iterator(data, batch_size=3):
        print(f"  batch: {batch}")
    
    # 4. 无限数据流
    print("\n--- 无限数据流 (取前3个) ---")
    stream = infinite_data_stream()
    for _ in range(3):
        print(f"  {next(stream)}")
    
    # 5. Token流式生成
    print("\n--- Token流式生成 ---")
    print("  生成: ", end="")
    for token in streaming_inference("Hello"):
        print(token, end=" ", flush=True)
    print()
    
    # 6. 管道
    print("\n--- 生成器管道 ---")
    pipeline_demo()
    
    # 7. yield from
    print("\n--- yield from ---")
    print(f"  结果: {list(main_generator())}")
    
    # 8. 内存对比
    print("\n--- 内存对比 ---")
    memory_comparison()
    
    print("\n总结: 生成器是处理大数据和流式数据的关键")
