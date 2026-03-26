"""
01_decorators.py - Python装饰器详解

【推理加速场景】
- 性能计时：监控推理时间
- 结果缓存：避免重复计算
- 输入验证：确保数据格式正确
- 重试机制：处理临时错误

运行: python 01_decorators.py
"""

import time
import functools
from typing import Callable, Any

# ============================================================================
# 第一部分：基本装饰器
# ============================================================================

def timer(func: Callable) -> Callable:
    """性能计时装饰器"""
    @functools.wraps(func)  # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[Timer] {func.__name__}: {elapsed*1000:.2f} ms")
        return result
    return wrapper


def logger(func: Callable) -> Callable:
    """日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[Log] 调用 {func.__name__}, args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"[Log] 返回 {result}")
        return result
    return wrapper


# 使用装饰器
@timer
def slow_function(n: int) -> int:
    """模拟耗时计算"""
    total = 0
    for i in range(n):
        total += i
    return total


@logger
def greet(name: str) -> str:
    return f"Hello, {name}!"


# ============================================================================
# 第二部分：带参数的装饰器
# ============================================================================

def repeat(times: int):
    """重复执行装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0):
    """重试装饰器（适用于网络请求等）"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[Retry] 尝试 {attempt+1}/{max_attempts} 失败: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            raise Exception(f"{func.__name__} 失败，已重试 {max_attempts} 次")
        return wrapper
    return decorator


@repeat(times=3)
def say_hello():
    print("Hello!")


# ============================================================================
# 第三部分：缓存装饰器（推理中常用）
# ============================================================================

def memoize(func: Callable) -> Callable:
    """简单缓存装饰器"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
            print(f"[Cache] 计算并缓存 {func.__name__}{args}")
        else:
            print(f"[Cache] 命中缓存 {func.__name__}{args}")
        return cache[args]
    return wrapper


@memoize
def fibonacci(n: int) -> int:
    """斐波那契（演示缓存效果）"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# 使用内置缓存（推荐）
@functools.lru_cache(maxsize=128)
def expensive_embedding_lookup(token_id: int) -> list:
    """模拟embedding查找（实际中会访问大矩阵）"""
    print(f"[Embedding] 计算 token {token_id}")
    return [0.1 * token_id] * 768  # 模拟768维embedding


# ============================================================================
# 第四部分：类装饰器
# ============================================================================

class CountCalls:
    """统计函数调用次数的类装饰器"""
    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func
        self.calls = 0
    
    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)
    
    def get_count(self) -> int:
        return self.calls


@CountCalls
def inference(input_ids: list) -> list:
    """模拟推理"""
    return [x * 2 for x in input_ids]


# ============================================================================
# 第五部分：实际推理场景
# ============================================================================

def validate_tensor(func: Callable) -> Callable:
    """验证张量形状的装饰器"""
    @functools.wraps(func)
    def wrapper(tensor, *args, **kwargs):
        if not isinstance(tensor, list):
            raise TypeError(f"期望list，得到{type(tensor)}")
        if len(tensor) == 0:
            raise ValueError("张量不能为空")
        return func(tensor, *args, **kwargs)
    return wrapper


def batch_decorator(batch_size: int = 32):
    """批处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(inputs: list, *args, **kwargs):
            results = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                print(f"[Batch] 处理 {i} 到 {i + len(batch)}")
                results.extend(func(batch, *args, **kwargs))
            return results
        return wrapper
    return decorator


@timer
@validate_tensor
def compute_attention(query: list) -> list:
    """模拟attention计算"""
    return [x * 0.5 for x in query]


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Python 装饰器详解")
    print("=" * 50)
    
    # 1. 基本装饰器
    print("\n--- 基本装饰器 ---")
    result = slow_function(100000)
    print(f"结果: {result}")
    
    greet("World")
    
    # 2. 带参数装饰器
    print("\n--- 带参数装饰器 ---")
    say_hello()
    
    # 3. 缓存
    print("\n--- 缓存装饰器 ---")
    print(f"fib(10) = {fibonacci(10)}")
    print(f"fib(5) = {fibonacci(5)}")  # 命中缓存
    
    # Embedding缓存
    print("\n--- Embedding缓存 ---")
    _ = expensive_embedding_lookup(100)
    _ = expensive_embedding_lookup(100)  # 命中
    _ = expensive_embedding_lookup(200)
    
    print(f"缓存状态: {expensive_embedding_lookup.cache_info()}")
    
    # 4. 类装饰器
    print("\n--- 类装饰器 ---")
    _ = inference([1, 2, 3])
    _ = inference([4, 5, 6])
    print(f"inference调用次数: {inference.get_count()}")
    
    # 5. 实际场景
    print("\n--- 实际场景 ---")
    query = [1.0] * 768
    output = compute_attention(query)
    print(f"输出长度: {len(output)}")
    
    print("\n总结: 装饰器是Python中实现AOP的利器")
