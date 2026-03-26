/**
 * @file 02_mutex_atomic.cpp
 * @brief 互斥锁与原子操作 - 线程安全
 * 
 * 【推理场景】
 * - 共享资源保护（KV Cache）
 * - 计数器（请求统计）
 * - 队列操作
 * 
 * 编译: g++ -std=c++17 -pthread -o 02_mutex 02_mutex_atomic.cpp
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

// ============================================================================
// 数据竞争问题
// ============================================================================

int unsafe_counter = 0;

void unsafe_increment(int n) {
    for (int i = 0; i < n; ++i) {
        unsafe_counter++;  // 不安全！多线程同时修改
    }
}

// ============================================================================
// 使用mutex保护
// ============================================================================

int safe_counter = 0;
std::mutex counter_mutex;

void safe_increment(int n) {
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lock(counter_mutex);
        safe_counter++;  // 安全：同一时刻只有一个线程能进入
    }
}

// ============================================================================
// 使用atomic（更高效）
// ============================================================================

std::atomic<int> atomic_counter{0};

void atomic_increment(int n) {
    for (int i = 0; i < n; ++i) {
        atomic_counter++;  // 原子操作，无需锁
    }
}

// ============================================================================
// 实际场景：线程安全的推理统计器
// ============================================================================

class InferenceStats {
private:
    std::atomic<int64_t> total_tokens_{0};
    std::atomic<int64_t> total_requests_{0};
    std::atomic<int64_t> total_latency_ms_{0};
    
public:
    void record_request(int tokens, int latency_ms) {
        total_tokens_ += tokens;
        total_requests_++;
        total_latency_ms_ += latency_ms;
    }
    
    void print() const {
        int64_t requests = total_requests_.load();
        if (requests > 0) {
            std::cout << "统计信息:\n";
            std::cout << "  总请求: " << requests << "\n";
            std::cout << "  总token: " << total_tokens_.load() << "\n";
            std::cout << "  平均延迟: " << total_latency_ms_.load() / requests << " ms\n";
        }
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "互斥锁与原子操作\n";
    std::cout << "========================================\n";
    
    const int N = 100000;
    const int THREADS = 4;
    
    // 1. 不安全的并发（演示问题）
    std::cout << "\n--- 不安全计数 ---\n";
    unsafe_counter = 0;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < THREADS; ++i) {
        threads.emplace_back(unsafe_increment, N);
    }
    for (auto& t : threads) t.join();
    
    std::cout << "预期: " << N * THREADS << "\n";
    std::cout << "实际: " << unsafe_counter << " (可能不对)\n";
    
    // 2. 使用mutex
    std::cout << "\n--- mutex保护 ---\n";
    threads.clear();
    safe_counter = 0;
    
    for (int i = 0; i < THREADS; ++i) {
        threads.emplace_back(safe_increment, N);
    }
    for (auto& t : threads) t.join();
    
    std::cout << "结果: " << safe_counter << " (正确)\n";
    
    // 3. 使用atomic
    std::cout << "\n--- atomic ---\n";
    threads.clear();
    atomic_counter = 0;
    
    for (int i = 0; i < THREADS; ++i) {
        threads.emplace_back(atomic_increment, N);
    }
    for (auto& t : threads) t.join();
    
    std::cout << "结果: " << atomic_counter << " (正确且更快)\n";
    
    // 4. 推理统计
    std::cout << "\n--- 推理统计器 ---\n";
    InferenceStats stats;
    threads.clear();
    
    for (int i = 0; i < THREADS; ++i) {
        threads.emplace_back([&stats]() {
            for (int j = 0; j < 1000; ++j) {
                stats.record_request(100, 50);  // 模拟请求
            }
        });
    }
    for (auto& t : threads) t.join();
    
    stats.print();
    
    return 0;
}
