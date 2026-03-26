/**
 * @file 03_thread_pool.cpp
 * @brief 线程池 - 批量推理调度
 * 
 * 【推理场景】
 * - 批量请求处理
 * - CPU并行前处理
 * - 异步任务执行
 * 
 * 编译: g++ -std=c++17 -pthread -o 03_thread_pool 03_thread_pool.cpp
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <vector>

// ============================================================================
// 简单线程池实现
// ============================================================================

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    
public:
    ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this]() {
                            return stop_ || !tasks_.empty();
                        });
                        
                        if (stop_ && tasks_.empty()) return;
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();  // 执行任务
                }
            });
        }
        std::cout << "线程池启动: " << num_threads << " 个工作线程\n";
    }
    
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        std::cout << "线程池关闭\n";
    }
    
    // 提交任务
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<decltype(f(args...))> {
        
        using ReturnType = decltype(f(args...));
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<ReturnType> result = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        
        return result;
    }
};

// ============================================================================
// 模拟推理任务
// ============================================================================

struct InferenceRequest {
    int request_id;
    std::string prompt;
    int max_tokens;
};

struct InferenceResult {
    int request_id;
    std::string response;
    int latency_ms;
};

InferenceResult process_request(const InferenceRequest& req) {
    // 模拟推理耗时
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    return {
        req.request_id,
        "Response to: " + req.prompt.substr(0, 10) + "...",
        50
    };
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "线程池\n";
    std::cout << "========================================\n";
    
    // 创建线程池
    ThreadPool pool(4);
    
    // 模拟批量请求
    std::cout << "\n--- 批量推理 ---\n";
    std::vector<std::future<InferenceResult>> futures;
    
    for (int i = 0; i < 10; ++i) {
        InferenceRequest req{i, "Hello, how are you?", 100};
        futures.push_back(pool.submit(process_request, req));
    }
    
    // 收集结果
    std::cout << "等待结果...\n";
    for (auto& f : futures) {
        InferenceResult result = f.get();
        std::cout << "请求 " << result.request_id 
                  << ": " << result.response << "\n";
    }
    
    // 计算任务
    std::cout << "\n--- 计算任务 ---\n";
    auto compute_future = pool.submit([](int n) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += i;
        return sum;
    }, 1000000);
    
    std::cout << "计算结果: " << compute_future.get() << "\n";
    
    return 0;
}
