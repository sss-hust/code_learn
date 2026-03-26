/**
 * @file 02_memory_pool.cpp
 * @brief 内存池实现 - 减少分配开销
 * 
 * 【推理场景】
 * 推理过程中频繁分配/释放tensor内存会导致：
 * - 内存碎片
 * - 分配延迟
 * - 内存峰值高
 * 
 * 内存池预分配大块内存，按需分配小块，大幅减少开销。
 * 
 * 编译: g++ -std=c++17 -o 02_memory_pool 02_memory_pool.cpp
 */

#include <iostream>
#include <vector>
#include <list>
#include <chrono>
#include <cstring>

// ============================================================================
// 简单内存池实现
// ============================================================================

class MemoryPool {
private:
    struct Block {
        size_t size;
        size_t offset;
        bool in_use;
    };
    
    char* memory_;           // 内存块
    size_t total_size_;      // 总大小
    size_t alignment_;       // 对齐
    std::list<Block> blocks_; // 块列表
    
    size_t align_up(size_t size) {
        return (size + alignment_ - 1) & ~(alignment_ - 1);
    }
    
public:
    MemoryPool(size_t size, size_t alignment = 64) 
        : total_size_(size), alignment_(alignment) {
        // 分配对齐内存
        memory_ = static_cast<char*>(std::aligned_alloc(alignment_, size));
        if (!memory_) {
            throw std::bad_alloc();
        }
        
        // 初始化为一个大的空闲块
        blocks_.push_back({size, 0, false});
        
        std::cout << "[MemoryPool] 初始化 " << size / 1024.0 / 1024.0 << " MB\n";
    }
    
    ~MemoryPool() {
        std::free(memory_);
        std::cout << "[MemoryPool] 释放\n";
    }
    
    void* allocate(size_t size) {
        size_t aligned_size = align_up(size);
        
        // 首次适配：找到第一个足够大的空闲块
        for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
            if (!it->in_use && it->size >= aligned_size) {
                // 如果块太大，分割
                if (it->size > aligned_size + alignment_) {
                    Block new_block = {
                        it->size - aligned_size,
                        it->offset + aligned_size,
                        false
                    };
                    blocks_.insert(std::next(it), new_block);
                    it->size = aligned_size;
                }
                
                it->in_use = true;
                return memory_ + it->offset;
            }
        }
        
        std::cerr << "[MemoryPool] 分配失败: " << size << " bytes\n";
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        size_t offset = static_cast<char*>(ptr) - memory_;
        
        for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
            if (it->offset == offset && it->in_use) {
                it->in_use = false;
                
                // 合并相邻空闲块
                // 与后一个合并
                auto next = std::next(it);
                if (next != blocks_.end() && !next->in_use) {
                    it->size += next->size;
                    blocks_.erase(next);
                }
                
                // 与前一个合并
                if (it != blocks_.begin()) {
                    auto prev = std::prev(it);
                    if (!prev->in_use) {
                        prev->size += it->size;
                        blocks_.erase(it);
                    }
                }
                
                return;
            }
        }
    }
    
    void print_status() {
        std::cout << "内存池状态:\n";
        size_t used = 0, free = 0;
        for (const auto& block : blocks_) {
            if (block.in_use) used += block.size;
            else free += block.size;
        }
        std::cout << "  已用: " << used / 1024.0 << " KB\n";
        std::cout << "  空闲: " << free / 1024.0 << " KB\n";
        std::cout << "  块数: " << blocks_.size() << "\n";
    }
};

// ============================================================================
// 性能对比
// ============================================================================

void benchmark_standard_alloc(int iterations, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = std::malloc(size);
        std::memset(ptr, 0, size);  // 触发实际分配
        std::free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "标准malloc: " << duration.count() << " us\n";
}

void benchmark_pool_alloc(MemoryPool& pool, int iterations, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = pool.allocate(size);
        std::memset(ptr, 0, size);
        pool.deallocate(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "内存池: " << duration.count() << " us\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "内存池实现\n";
    std::cout << "========================================\n";
    
    // 1. 创建内存池
    MemoryPool pool(64 * 1024 * 1024);  // 64MB
    
    // 2. 基本使用
    std::cout << "\n--- 基本使用 ---\n";
    
    void* tensor1 = pool.allocate(1024 * 1024);  // 1MB
    std::cout << "分配tensor1: 1MB at " << tensor1 << "\n";
    
    void* tensor2 = pool.allocate(2 * 1024 * 1024);  // 2MB
    std::cout << "分配tensor2: 2MB at " << tensor2 << "\n";
    
    pool.print_status();
    
    pool.deallocate(tensor1);
    std::cout << "释放tensor1\n";
    pool.print_status();
    
    // 3. 性能对比
    std::cout << "\n--- 性能对比 (10000次分配/释放 1KB) ---\n";
    benchmark_standard_alloc(10000, 1024);
    
    MemoryPool bench_pool(16 * 1024 * 1024);
    benchmark_pool_alloc(bench_pool, 10000, 1024);
    
    // 4. 模拟推理场景
    std::cout << "\n--- 模拟推理场景 ---\n";
    MemoryPool inference_pool(128 * 1024 * 1024);  // 128MB
    
    for (int batch = 0; batch < 3; ++batch) {
        std::cout << "Batch " << batch << ":\n";
        
        // 分配activation
        void* hidden = inference_pool.allocate(32 * 128 * 768 * sizeof(float));
        void* attention = inference_pool.allocate(32 * 12 * 128 * 128 * sizeof(float));
        
        // 模拟计算...
        
        // 释放
        inference_pool.deallocate(attention);
        inference_pool.deallocate(hidden);
    }
    
    inference_pool.print_status();
    
    return 0;
}
