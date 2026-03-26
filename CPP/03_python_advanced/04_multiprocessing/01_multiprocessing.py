"""
01_multiprocessing.py - Python多进程

【推理加速场景】
- 数据预处理并行化
- 批量tokenization
- CPU密集型计算
- 绕过GIL限制

运行: python 01_multiprocessing.py
"""

import multiprocessing as mp
import time
import numpy as np
from typing import List

# ============================================================================
# 第一部分：基本多进程
# ============================================================================

def cpu_bound_task(n: int) -> float:
    """CPU密集型任务"""
    total = 0.0
    for i in range(n):
        total += i ** 0.5
    return total


def basic_multiprocessing():
    """基本多进程使用"""
    print("\n--- 基本多进程 ---")
    
    # 创建进程
    def worker(name: str):
        print(f"进程 {name} 开始, PID: {mp.current_process().pid}")
        time.sleep(0.5)
        print(f"进程 {name} 结束")
    
    processes = []
    for i in range(3):
        p = mp.Process(target=worker, args=(f"Worker-{i}",))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("所有进程完成")


# ============================================================================
# 第二部分：进程池
# ============================================================================

def process_pool_demo():
    """使用进程池"""
    print("\n--- 进程池 ---")
    
    data = [10**6] * 8  # 8个任务
    
    # 串行执行
    start = time.perf_counter()
    results_serial = [cpu_bound_task(n) for n in data]
    serial_time = time.perf_counter() - start
    print(f"串行: {serial_time:.2f}s")
    
    # 并行执行
    start = time.perf_counter()
    with mp.Pool(processes=4) as pool:
        results_parallel = pool.map(cpu_bound_task, data)
    parallel_time = time.perf_counter() - start
    print(f"并行(4进程): {parallel_time:.2f}s")
    print(f"加速比: {serial_time/parallel_time:.2f}x")


# ============================================================================
# 第三部分：进程间通信
# ============================================================================

def producer(queue: mp.Queue, n: int):
    """生产者：生成数据"""
    for i in range(n):
        data = {"id": i, "value": np.random.randn(100).tolist()}
        queue.put(data)
        time.sleep(0.01)
    queue.put(None)  # 结束信号


def consumer(queue: mp.Queue, result_queue: mp.Queue):
    """消费者：处理数据"""
    while True:
        data = queue.get()
        if data is None:
            break
        # 处理数据
        processed = {
            "id": data["id"],
            "mean": np.mean(data["value"])
        }
        result_queue.put(processed)


def producer_consumer_demo():
    """生产者-消费者模式"""
    print("\n--- 生产者-消费者 ---")
    
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 启动进程
    prod = mp.Process(target=producer, args=(data_queue, 10))
    cons = mp.Process(target=consumer, args=(data_queue, result_queue))
    
    prod.start()
    cons.start()
    
    prod.join()
    cons.join()
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f"处理了 {len(results)} 条数据")


# ============================================================================
# 第四部分：推理场景 - 并行Tokenization
# ============================================================================

def tokenize_text(text: str) -> List[int]:
    """模拟tokenization（实际中会调用tokenizer）"""
    # 简单模拟：将每个字符转为ASCII
    tokens = [ord(c) for c in text]
    return tokens[:128]  # 截断到128


def parallel_tokenize(texts: List[str], num_workers: int = 4) -> List[List[int]]:
    """并行tokenization"""
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(tokenize_text, texts)
    return results


def tokenization_demo():
    """Tokenization性能对比"""
    print("\n--- 并行Tokenization ---")
    
    # 生成测试数据
    texts = ["Hello world! " * 50 for _ in range(100)]
    
    # 串行
    start = time.perf_counter()
    serial_results = [tokenize_text(t) for t in texts]
    serial_time = time.perf_counter() - start
    print(f"串行: {serial_time*1000:.2f} ms")
    
    # 并行
    start = time.perf_counter()
    parallel_results = parallel_tokenize(texts, num_workers=4)
    parallel_time = time.perf_counter() - start
    print(f"并行: {parallel_time*1000:.2f} ms")
    
    print(f"处理 {len(texts)} 条文本，每条 {len(texts[0])} 字符")


# ============================================================================
# 第五部分：共享内存
# ============================================================================

def shared_memory_demo():
    """共享内存（避免数据拷贝）"""
    print("\n--- 共享内存 ---")
    
    # 共享数组
    shared_arr = mp.Array('d', 10)  # 10个double
    
    def worker_shared(arr, index, value):
        arr[index] = value
    
    processes = []
    for i in range(10):
        p = mp.Process(target=worker_shared, args=(shared_arr, i, i * 1.5))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print(f"共享数组: {list(shared_arr)}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Python 多进程")
    print("=" * 50)
    
    print(f"CPU核心数: {mp.cpu_count()}")
    
    # 1. 基本多进程
    basic_multiprocessing()
    
    # 2. 进程池
    process_pool_demo()
    
    # 3. 生产者-消费者
    producer_consumer_demo()
    
    # 4. 并行Tokenization
    tokenization_demo()
    
    # 5. 共享内存
    shared_memory_demo()
    
    print("\n总结: 多进程可绕过GIL，实现真正的CPU并行")
