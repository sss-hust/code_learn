/**
 * @file 01_thread_basics.cpp
 * @brief 多线程基础 - 并行计算
 * 
 * 【推理场景】
 * - 并行数据预处理
 * - 多请求并发处理
 * - 异步tokenization
 * 
 * 编译: g++ -std=c++17 -pthread -o 01_thread 01_thread_basics.cpp
 */

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// 简单线程函数
void compute(int id, int iterations) {
    std::cout << "线程 " << id << " 开始\n";
    
    double sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += i * 0.001;
    }
    
    std::cout << "线程 " << id << " 完成, sum=" << sum << "\n";
}

// 带引用参数的函数
void process_data(std::vector<float>& data, int start, int end) {
    for (int i = start; i < end; ++i) {
        data[i] = data[i] * 2.0f + 1.0f;  // 一些计算
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "多线程基础\n";
    std::cout << "========================================\n";
    
    // 硬件并发数
    unsigned int n_threads = std::thread::hardware_concurrency();
    std::cout << "硬件线程数: " << n_threads << "\n";
    
    // 1. 创建和join线程
    std::cout << "\n--- 基本线程 ---\n";
    std::thread t1(compute, 1, 1000000);
    std::thread t2(compute, 2, 1000000);
    
    t1.join();  // 等待线程完成
    t2.join();
    
    // 2. Lambda作为线程函数
    std::cout << "\n--- Lambda线程 ---\n";
    int result = 0;
    std::thread t3([&result]() {
        for (int i = 0; i < 100; ++i) {
            result += i;
        }
    });
    t3.join();
    std::cout << "Lambda结果: " << result << "\n";
    
    // 3. 并行处理数据
    std::cout << "\n--- 并行数据处理 ---\n";
    std::vector<float> data(1000000, 1.0f);
    int chunk_size = data.size() / 4;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? data.size() : (i + 1) * chunk_size;
        threads.emplace_back(process_data, std::ref(data), start_idx, end_idx);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "并行处理耗时: " << duration.count() << " us\n";
    std::cout << "验证 data[0] = " << data[0] << " (应为3.0)\n";
    
    // 4. detach（分离线程）
    std::cout << "\n--- 分离线程 ---\n";
    std::thread daemon([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "(后台线程完成)\n";
    });
    daemon.detach();  // 分离后不需要join
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    std::cout << "\n主线程结束\n";
    return 0;
}
